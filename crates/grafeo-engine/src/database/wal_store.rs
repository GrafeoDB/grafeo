//! WAL-aware graph store wrapper.
//!
//! Wraps an [`LpgStore`] and logs every mutation to the WAL so that
//! query-engine mutations (INSERT, DELETE, SET via GQL/Cypher/etc.)
//! survive a close/reopen cycle.

use std::sync::Arc;

use grafeo_common::grafeo_warn;
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::hash::FxHashMap;
use grafeo_core::graph::lpg::{CompareOp, Edge, LpgStore, Node};
use grafeo_core::graph::{Direction, GraphStore, GraphStoreMut, GraphStoreSearch};
use grafeo_core::statistics::Statistics;
use grafeo_storage::wal::{LpgWal, WalRecord};

use arcstr::ArcStr;

/// A [`GraphStoreMut`] decorator that delegates every call to an inner
/// [`LpgStore`] and additionally logs mutation operations to the WAL.
///
/// Read-only methods are forwarded without any WAL interaction.
///
/// For named graphs, emits a [`WalRecord::SwitchGraph`] before data mutations
/// when the WAL context differs from this store's graph. The shared
/// `wal_graph_context` mutex ensures atomicity of context-switch + mutation
/// pairs across concurrent sessions.
pub(crate) struct WalGraphStore {
    inner: Arc<LpgStore>,
    wal: Arc<LpgWal>,
    /// Which named graph this store represents (`None` = default graph).
    graph_name: Option<String>,
    /// Shared tracker: the last graph context emitted to the WAL.
    /// Held across a (SwitchGraph + mutation) pair to prevent interleaving.
    wal_graph_context: Arc<parking_lot::Mutex<Option<String>>>,
}

impl WalGraphStore {
    /// Creates a new WAL-aware store wrapper for the default graph.
    pub fn new(
        inner: Arc<LpgStore>,
        wal: Arc<LpgWal>,
        wal_graph_context: Arc<parking_lot::Mutex<Option<String>>>,
    ) -> Self {
        Self {
            inner,
            wal,
            graph_name: None,
            wal_graph_context,
        }
    }

    /// Creates a new WAL-aware store wrapper for a named graph.
    pub fn new_for_graph(
        inner: Arc<LpgStore>,
        wal: Arc<LpgWal>,
        graph_name: String,
        wal_graph_context: Arc<parking_lot::Mutex<Option<String>>>,
    ) -> Self {
        Self {
            inner,
            wal,
            graph_name: Some(graph_name),
            wal_graph_context,
        }
    }

    /// Logs a WAL record with graph context tracking.
    ///
    /// Acquires the shared context lock, emits a `SwitchGraph` record if the
    /// WAL context differs from this store's graph, then logs the data record.
    /// Both writes happen under the same lock to prevent concurrent sessions
    /// from interleaving context switches with unrelated mutations.
    fn log_with_context(&self, record: &WalRecord) {
        let mut ctx = self.wal_graph_context.lock();
        if *ctx != self.graph_name {
            let _ = self.wal.log(&WalRecord::SwitchGraph {
                name: self.graph_name.clone(),
            });
            (*ctx).clone_from(&self.graph_name);
        }
        if let Err(e) = self.wal.log(record) {
            grafeo_warn!("WAL log failed: {e}");
        }
    }
}

// ---------------------------------------------------------------------------
// GraphStore (read-only): pure delegation
// ---------------------------------------------------------------------------

impl GraphStore for WalGraphStore {
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
        GraphStore::neighbors(self.inner.as_ref(), node, direction)
    }

    fn edges_from(&self, node: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)> {
        GraphStore::edges_from(self.inner.as_ref(), node, direction)
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

// Pure delegation: the WAL wrapper logs mutations but owns no index state,
// so every text/vector lookup has to fall through to the underlying
// `LpgStore`. A stub impl silently turns into "no index exists" at every
// call site (has_text_index → false, text_search → [], etc.), which
// regressed hybrid queries on persistent DBs until it was caught by the
// `_persistent` spec variants — see issue #308.
impl GraphStoreSearch for WalGraphStore {
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
    fn vector_index_metric(
        &self,
        label: &str,
        property: &str,
    ) -> Option<grafeo_core::index::vector::DistanceMetric> {
        self.inner.vector_index_metric(label, property)
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
// GraphStoreMut: delegate + WAL log
// ---------------------------------------------------------------------------

impl GraphStoreMut for WalGraphStore {
    fn create_node(&self, labels: &[&str]) -> NodeId {
        let id = self.inner.create_node(labels);
        self.log_with_context(&WalRecord::CreateNode {
            id,
            labels: labels.iter().map(|s| (*s).to_string()).collect(),
        });
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
        self.log_with_context(&WalRecord::CreateNode {
            id,
            labels: labels.iter().map(|s| (*s).to_string()).collect(),
        });
        id
    }

    fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId {
        let id = self.inner.create_edge(src, dst, edge_type);
        self.log_with_context(&WalRecord::CreateEdge {
            id,
            src,
            dst,
            edge_type: edge_type.to_string(),
        });
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
        self.log_with_context(&WalRecord::CreateEdge {
            id,
            src,
            dst,
            edge_type: edge_type.to_string(),
        });
        id
    }

    fn batch_create_edges(&self, edges: &[(NodeId, NodeId, &str)]) -> Vec<EdgeId> {
        let ids = self.inner.batch_create_edges(edges);
        for (id, (src, dst, edge_type)) in ids.iter().zip(edges) {
            self.log_with_context(&WalRecord::CreateEdge {
                id: *id,
                src: *src,
                dst: *dst,
                edge_type: (*edge_type).to_string(),
            });
        }
        ids
    }

    fn delete_node(&self, id: NodeId) -> bool {
        let deleted = self.inner.delete_node(id);
        if deleted {
            self.log_with_context(&WalRecord::DeleteNode { id });
        }
        deleted
    }

    fn delete_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let deleted = self.inner.delete_node_versioned(id, epoch, transaction_id);
        if deleted {
            self.log_with_context(&WalRecord::DeleteNode { id });
        }
        deleted
    }

    fn delete_node_edges(&self, node_id: NodeId) {
        // Collect edge IDs before deletion so we can log them
        let outgoing: Vec<EdgeId> = self
            .inner
            .edges_from(node_id, Direction::Outgoing)
            .map(|(_, eid)| eid)
            .collect();
        let incoming: Vec<EdgeId> = self
            .inner
            .edges_from(node_id, Direction::Incoming)
            .map(|(_, eid)| eid)
            .collect();

        self.inner.delete_node_edges(node_id);

        for id in outgoing.into_iter().chain(incoming) {
            self.log_with_context(&WalRecord::DeleteEdge { id });
        }
    }

    fn delete_edge(&self, id: EdgeId) -> bool {
        let deleted = self.inner.delete_edge(id);
        if deleted {
            self.log_with_context(&WalRecord::DeleteEdge { id });
        }
        deleted
    }

    fn delete_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let deleted = self.inner.delete_edge_versioned(id, epoch, transaction_id);
        if deleted {
            self.log_with_context(&WalRecord::DeleteEdge { id });
        }
        deleted
    }

    fn set_node_property(&self, id: NodeId, key: &str, value: Value) {
        // Store first, WAL second: consistent lock ordering with create/delete
        // methods to prevent ABBA deadlock between store locks and WAL locks.
        self.inner.set_node_property(id, key, value.clone());
        self.log_with_context(&WalRecord::SetNodeProperty {
            id,
            key: key.to_string(),
            value,
        });
    }

    fn set_edge_property(&self, id: EdgeId, key: &str, value: Value) {
        self.inner.set_edge_property(id, key, value.clone());
        self.log_with_context(&WalRecord::SetEdgeProperty {
            id,
            key: key.to_string(),
            value,
        });
    }

    fn remove_node_property(&self, id: NodeId, key: &str) -> Option<Value> {
        let removed = self.inner.remove_node_property(id, key);
        if removed.is_some() {
            self.log_with_context(&WalRecord::RemoveNodeProperty {
                id,
                key: key.to_string(),
            });
        }
        removed
    }

    fn remove_edge_property(&self, id: EdgeId, key: &str) -> Option<Value> {
        let removed = self.inner.remove_edge_property(id, key);
        if removed.is_some() {
            self.log_with_context(&WalRecord::RemoveEdgeProperty {
                id,
                key: key.to_string(),
            });
        }
        removed
    }

    fn add_label(&self, node_id: NodeId, label: &str) -> bool {
        let added = self.inner.add_label(node_id, label);
        if added {
            self.log_with_context(&WalRecord::AddNodeLabel {
                id: node_id,
                label: label.to_string(),
            });
        }
        added
    }

    fn remove_label(&self, node_id: NodeId, label: &str) -> bool {
        let removed = self.inner.remove_label(node_id, label);
        if removed {
            self.log_with_context(&WalRecord::RemoveNodeLabel {
                id: node_id,
                label: label.to_string(),
            });
        }
        removed
    }

    // The transactional mutators must reach the inner store's versioned
    // methods, which record undo entries; the trait defaults would call the
    // plain methods above and a rollback could not restore the old values.

    fn set_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        self.inner
            .set_node_property_versioned(id, key, value.clone(), transaction_id);
        self.log_with_context(&WalRecord::SetNodeProperty {
            id,
            key: key.to_string(),
            value,
        });
    }

    fn set_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        self.inner
            .set_edge_property_versioned(id, key, value.clone(), transaction_id);
        self.log_with_context(&WalRecord::SetEdgeProperty {
            id,
            key: key.to_string(),
            value,
        });
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
        if removed.is_some() {
            self.log_with_context(&WalRecord::RemoveNodeProperty {
                id,
                key: key.to_string(),
            });
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
        if removed.is_some() {
            self.log_with_context(&WalRecord::RemoveEdgeProperty {
                id,
                key: key.to_string(),
            });
        }
        removed
    }

    fn add_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        let added = self
            .inner
            .add_label_versioned(node_id, label, transaction_id);
        if added {
            self.log_with_context(&WalRecord::AddNodeLabel {
                id: node_id,
                label: label.to_string(),
            });
        }
        added
    }

    fn remove_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        let removed = self
            .inner
            .remove_label_versioned(node_id, label, transaction_id);
        if removed {
            self.log_with_context(&WalRecord::RemoveNodeLabel {
                id: node_id,
                label: label.to_string(),
            });
        }
        removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_storage::wal::TypedWal;

    fn setup() -> (WalGraphStore, Arc<LpgWal>) {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LpgStore::new().unwrap());
        let wal = Arc::new(TypedWal::open(dir.path()).unwrap());
        let wal_ref = Arc::clone(&wal);
        let ctx = Arc::new(parking_lot::Mutex::new(None));
        (WalGraphStore::new(store, wal, ctx), wal_ref)
    }

    #[test]
    fn create_node_delegates_and_logs() {
        let (ws, wal) = setup();
        let id = ws.create_node(&["Person", "Employee"]);

        assert!(ws.get_node(id).is_some());
        assert_eq!(ws.node_count(), 1);
        assert_eq!(wal.record_count(), 1);
    }

    #[test]
    fn create_edge_delegates_and_logs() {
        let (ws, wal) = setup();
        let a = ws.create_node(&["Node"]);
        let b = ws.create_node(&["Node"]);
        let eid = ws.create_edge(a, b, "KNOWS");

        assert!(ws.get_edge(eid).is_some());
        assert_eq!(ws.edge_count(), 1);
        // 2 CreateNode + 1 CreateEdge
        assert_eq!(wal.record_count(), 3);
    }

    #[test]
    fn set_property_delegates_and_logs() {
        let (ws, wal) = setup();
        let nid = ws.create_node(&["Person"]);
        ws.set_node_property(nid, "name", Value::String("Alix".into()));

        assert_eq!(
            ws.get_node_property(nid, &PropertyKey::from("name")),
            Some(Value::String("Alix".into()))
        );
        // CreateNode + SetNodeProperty
        assert_eq!(wal.record_count(), 2);

        let a = ws.create_node(&["Node"]);
        let b = ws.create_node(&["Node"]);
        let eid = ws.create_edge(a, b, "LINK");
        ws.set_edge_property(eid, "weight", Value::Int64(42));

        assert_eq!(
            ws.get_edge_property(eid, &PropertyKey::from("weight")),
            Some(Value::Int64(42))
        );
        // +2 CreateNode + 1 CreateEdge + 1 SetEdgeProperty = 6 total
        assert_eq!(wal.record_count(), 6);
    }

    #[test]
    fn delete_node_only_logs_on_success() {
        let (ws, wal) = setup();
        let id = ws.create_node(&["Person"]);
        assert_eq!(wal.record_count(), 1);

        // Delete nonexistent: no new record
        assert!(!ws.delete_node(NodeId::new(999)));
        assert_eq!(wal.record_count(), 1);

        // Delete real node: logs
        assert!(ws.delete_node(id));
        assert_eq!(wal.record_count(), 2);
        assert!(ws.get_node(id).is_none());
    }

    #[test]
    fn delete_edge_only_logs_on_success() {
        let (ws, wal) = setup();
        let a = ws.create_node(&["Node"]);
        let b = ws.create_node(&["Node"]);
        let eid = ws.create_edge(a, b, "LINK");
        assert_eq!(wal.record_count(), 3);

        // Delete nonexistent: no new record
        assert!(!ws.delete_edge(EdgeId::new(999)));
        assert_eq!(wal.record_count(), 3);

        // Delete real edge: logs
        assert!(ws.delete_edge(eid));
        assert_eq!(wal.record_count(), 4);
        assert!(ws.get_edge(eid).is_none());
    }

    #[test]
    fn remove_property_only_logs_on_success() {
        let (ws, wal) = setup();
        let id = ws.create_node(&["Person"]);
        ws.set_node_property(id, "age", Value::Int64(30));
        assert_eq!(wal.record_count(), 2);

        // Remove nonexistent: no log
        assert!(ws.remove_node_property(id, "missing").is_none());
        assert_eq!(wal.record_count(), 2);

        // Remove real property: logs
        assert_eq!(ws.remove_node_property(id, "age"), Some(Value::Int64(30)));
        assert_eq!(wal.record_count(), 3);

        // Edge property variant
        let a = ws.create_node(&["Node"]);
        let b = ws.create_node(&["Node"]);
        let eid = ws.create_edge(a, b, "X");
        ws.set_edge_property(eid, "w", Value::Int64(1));
        let before = wal.record_count();

        assert!(ws.remove_edge_property(eid, "missing").is_none());
        assert_eq!(wal.record_count(), before);

        assert_eq!(ws.remove_edge_property(eid, "w"), Some(Value::Int64(1)));
        assert_eq!(wal.record_count(), before + 1);
    }

    #[test]
    fn add_remove_label_conditional_logging() {
        let (ws, wal) = setup();
        let id = ws.create_node(&["Person"]);
        assert_eq!(wal.record_count(), 1);

        // Add duplicate label: no log
        assert!(!ws.add_label(id, "Person"));
        assert_eq!(wal.record_count(), 1);

        // Add new label: logs
        assert!(ws.add_label(id, "Employee"));
        assert_eq!(wal.record_count(), 2);

        // Remove label: logs
        assert!(ws.remove_label(id, "Employee"));
        assert_eq!(wal.record_count(), 3);

        // Remove absent label: no log
        assert!(!ws.remove_label(id, "Employee"));
        assert_eq!(wal.record_count(), 3);
    }

    #[test]
    fn batch_create_edges_logs_each() {
        let (ws, wal) = setup();
        let a = ws.create_node(&["Node"]);
        let b = ws.create_node(&["Node"]);
        let c = ws.create_node(&["Node"]);
        assert_eq!(wal.record_count(), 3);

        let eids = ws.batch_create_edges(&[(a, b, "X"), (b, c, "Y")]);
        assert_eq!(eids.len(), 2);
        assert_eq!(ws.edge_count(), 2);
        // One WAL record per edge
        assert_eq!(wal.record_count(), 5);
    }

    #[test]
    fn delete_node_edges_logs_each_edge() {
        let (ws, wal) = setup();
        let a = ws.create_node(&["Node"]);
        let b = ws.create_node(&["Node"]);
        let c = ws.create_node(&["Node"]);
        ws.create_edge(a, b, "X");
        ws.create_edge(c, a, "Y");
        assert_eq!(wal.record_count(), 5);

        ws.delete_node_edges(a);
        // 2 DeleteEdge records (one outgoing, one incoming)
        assert_eq!(wal.record_count(), 7);
        assert_eq!(ws.edge_count(), 0);
    }

    fn setup_named_graph() -> (WalGraphStore, Arc<LpgWal>) {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LpgStore::new().unwrap());
        let wal = Arc::new(TypedWal::open(dir.path()).unwrap());
        let wal_ref = Arc::clone(&wal);
        let ctx = Arc::new(parking_lot::Mutex::new(None));
        (
            WalGraphStore::new_for_graph(store, wal, "social".to_string(), ctx),
            wal_ref,
        )
    }

    #[test]
    fn named_graph_emits_switch_graph_record() {
        let (ws, wal) = setup_named_graph();
        let _id = ws.create_node(&["Person"]);

        // Should have SwitchGraph + CreateNode = 2 records
        assert_eq!(wal.record_count(), 2);
    }

    #[test]
    fn named_graph_context_not_repeated() {
        let (ws, wal) = setup_named_graph();
        // First mutation: emits SwitchGraph + CreateNode
        ws.create_node(&["Person"]);
        assert_eq!(wal.record_count(), 2);

        // Second mutation: context already set, no extra SwitchGraph
        ws.create_node(&["Person"]);
        assert_eq!(wal.record_count(), 3); // just CreateNode
    }

    #[test]
    fn create_node_versioned_delegates_and_logs() {
        let (ws, wal) = setup();
        let epoch = ws.current_epoch();
        let tx = TransactionId::new(1);
        let id = ws.create_node_versioned(&["Person"], epoch, tx);

        assert!(id.is_valid());
        assert_eq!(wal.record_count(), 1);
    }

    #[test]
    fn create_edge_versioned_delegates_and_logs() {
        let (ws, wal) = setup();
        let epoch = ws.current_epoch();
        let tx = TransactionId::new(1);
        let a = ws.create_node(&["Node"]);
        let b = ws.create_node(&["Node"]);
        let eid = ws.create_edge_versioned(a, b, "KNOWS", epoch, tx);

        assert!(eid.is_valid());
        // 2 CreateNode + 1 CreateEdge
        assert_eq!(wal.record_count(), 3);
    }

    /// The versioned property and label mutators must record undo entries in
    /// the inner store (so a rollback restores them) and still log each change.
    #[test]
    fn versioned_property_and_label_changes_log_and_roll_back() {
        let (ws, wal) = setup();
        let a = ws.create_node(&["Base"]);
        let b = ws.create_node(&["Base"]);
        let e = ws.create_edge(a, b, "KNOWS");
        ws.set_node_property(a, "v", Value::Int64(1));
        ws.set_node_property(a, "x", Value::from("keep"));
        ws.set_edge_property(e, "w", Value::Int64(5));
        ws.set_edge_property(e, "note", Value::from("n"));
        let logged = wal.record_count();

        let tx = TransactionId::new(7);
        ws.set_node_property_versioned(a, "v", Value::Int64(2), tx);
        assert_eq!(
            ws.remove_node_property_versioned(a, "x", tx),
            Some(Value::from("keep"))
        );
        ws.set_edge_property_versioned(e, "w", Value::Int64(9), tx);
        assert_eq!(
            ws.remove_edge_property_versioned(e, "note", tx),
            Some(Value::from("n"))
        );
        assert!(ws.add_label_versioned(a, "Extra", tx));
        assert!(ws.remove_label_versioned(a, "Base", tx));
        assert_eq!(wal.record_count(), logged + 6, "each change is logged");

        // No-op changes are not logged.
        assert_eq!(ws.remove_node_property_versioned(a, "missing", tx), None);
        assert!(!ws.add_label_versioned(a, "Extra", tx));
        assert_eq!(wal.record_count(), logged + 6);

        // Undo-log replay (without `temporal`; temporal rollback pops pending
        // versions and is covered end to end by the
        // persistent_transaction_rollback integration tests).
        #[cfg(not(feature = "temporal"))]
        {
            ws.inner.rollback_transaction_properties(tx);
            assert_eq!(ws.get_node_property(a, &"v".into()), Some(Value::Int64(1)));
            assert_eq!(
                ws.get_node_property(a, &"x".into()),
                Some(Value::from("keep"))
            );
            assert_eq!(ws.get_edge_property(e, &"w".into()), Some(Value::Int64(5)));
            assert_eq!(
                ws.get_edge_property(e, &"note".into()),
                Some(Value::from("n"))
            );
            let labels = ws.get_node(a).unwrap().labels;
            assert!(labels.iter().any(|l| l.as_str() == "Base"));
            assert!(!labels.iter().any(|l| l.as_str() == "Extra"));
        }
    }

    #[test]
    fn delete_node_versioned_only_logs_on_success() {
        let (ws, wal) = setup();
        let epoch = ws.current_epoch();
        let tx = TransactionId::new(1);
        let id = ws.create_node_versioned(&["Person"], epoch, tx);
        assert_eq!(wal.record_count(), 1);

        // Delete nonexistent: no log
        assert!(!ws.delete_node_versioned(NodeId::new(999), epoch, tx));
        assert_eq!(wal.record_count(), 1);

        // Delete real node: logs
        assert!(ws.delete_node_versioned(id, epoch, tx));
        assert_eq!(wal.record_count(), 2);
    }

    #[test]
    fn delete_edge_versioned_only_logs_on_success() {
        let (ws, wal) = setup();
        let epoch = ws.current_epoch();
        let tx = TransactionId::new(1);
        let a = ws.create_node(&["Node"]);
        let b = ws.create_node(&["Node"]);
        let eid = ws.create_edge_versioned(a, b, "LINK", epoch, tx);
        assert_eq!(wal.record_count(), 3);

        // Delete nonexistent: no log
        assert!(!ws.delete_edge_versioned(EdgeId::new(999), epoch, tx));
        assert_eq!(wal.record_count(), 3);

        // Delete real edge: logs
        assert!(ws.delete_edge_versioned(eid, epoch, tx));
        assert_eq!(wal.record_count(), 4);
    }

    #[test]
    fn create_node_with_props_via_trait_default() {
        use grafeo_core::graph::GraphStoreMut;

        let (ws, wal) = setup();
        let store: &dyn GraphStoreMut = &ws;
        let id = store.create_node_with_props(
            &["Person"],
            &[
                (PropertyKey::from("name"), Value::String("Alix".into())),
                (PropertyKey::from("age"), Value::Int64(30)),
            ],
        );

        assert!(ws.get_node(id).is_some());
        // 1 CreateNode + 2 SetNodeProperty
        assert_eq!(wal.record_count(), 3);

        assert_eq!(
            ws.get_node_property(id, &PropertyKey::from("name")),
            Some(Value::String("Alix".into()))
        );
        assert_eq!(
            ws.get_node_property(id, &PropertyKey::from("age")),
            Some(Value::Int64(30))
        );
    }

    #[test]
    fn create_edge_with_props_via_trait_default() {
        use grafeo_core::graph::GraphStoreMut;

        let (ws, wal) = setup();
        let store: &dyn GraphStoreMut = &ws;
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let eid = store.create_edge_with_props(
            a,
            b,
            "KNOWS",
            &[(PropertyKey::from("since"), Value::Int64(2020))],
        );

        assert!(ws.get_edge(eid).is_some());
        // 2 CreateNode + 1 CreateEdge + 1 SetEdgeProperty
        assert_eq!(wal.record_count(), 4);

        assert_eq!(
            ws.get_edge_property(eid, &PropertyKey::from("since")),
            Some(Value::Int64(2020))
        );
    }

    #[test]
    fn read_operations_do_not_log() {
        let (ws, wal) = setup();
        let id = ws.create_node(&["Person"]);
        ws.set_node_property(id, "name", Value::String("Alix".into()));
        assert_eq!(wal.record_count(), 2);

        // Exercise read-only methods
        let _ = ws.get_node(id);
        let _ = ws.node_count();
        let _ = ws.node_ids();
        let _ = ws.nodes_by_label("Person");
        let _ = ws.get_node_property(id, &PropertyKey::from("name"));
        let _ = ws.neighbors(id, Direction::Outgoing);
        let _ = ws.edge_count();
        let _ = ws.out_degree(id);
        let _ = ws.in_degree(id);
        let _ = ws.has_backward_adjacency();
        let _ = ws.statistics();

        // No additional records
        assert_eq!(wal.record_count(), 2);
    }
}
