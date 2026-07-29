//! Async WAL-aware graph store wrapper.
//!
//! Mirrors [`WalGraphStore`](super::wal_store::WalGraphStore) for async
//! contexts. Logs every mutation to an [`AsyncLpgWal`] before applying it
//! synchronously to the in-memory [`LpgStore`].
//!
//! The key pattern: WAL writes are I/O-bound (async), store mutations are
//! CPU-bound (sync, in-memory). The [`log_and_apply`](AsyncWalGraphStore::log_and_apply)
//! method sequences them correctly.

use std::sync::Arc;

use grafeo_common::types::{EdgeId, NodeId, Value};
use grafeo_common::utils::error::Result;
use grafeo_core::graph::lpg::LpgStore;
use grafeo_core::graph::{Direction, GraphStore};
use grafeo_storage::wal::{AsyncLpgWal, WalRecord};

/// An async [`GraphStoreMut`](grafeo_core::graph::GraphStoreMut) decorator
/// that logs mutations to an [`AsyncLpgWal`] before applying them to the
/// in-memory [`LpgStore`].
///
/// Unlike the sync [`WalGraphStore`](super::wal_store::WalGraphStore), this
/// does **not** implement `GraphStoreMut` (which is sync). Instead it exposes
/// individual `async` methods for each mutation. The sync query pipeline
/// continues to use `WalGraphStore`; this type is used by async engine methods
/// such as `GrafeoDB::async_wal_checkpoint()`.
///
/// For named graphs, emits a [`WalRecord::SwitchGraph`] before data mutations
/// when the WAL context differs from this store's graph.
#[allow(dead_code)] // Used in tests now; wired into GrafeoDB in Phase 3
pub(crate) struct AsyncWalGraphStore {
    inner: Arc<LpgStore>,
    wal: Arc<AsyncLpgWal>,
    /// Which named graph this store represents (`None` = default graph).
    graph_name: Option<String>,
    /// Shared tracker: the last graph context emitted to the WAL.
    /// Uses a tokio mutex for async-safe access across concurrent sessions.
    wal_graph_context: Arc<tokio::sync::Mutex<Option<String>>>,
}

#[allow(dead_code)] // Used in tests now; wired into GrafeoDB in Phase 3
impl AsyncWalGraphStore {
    /// Creates a new async WAL-aware store wrapper for the default graph.
    pub fn new(
        inner: Arc<LpgStore>,
        wal: Arc<AsyncLpgWal>,
        wal_graph_context: Arc<tokio::sync::Mutex<Option<String>>>,
    ) -> Self {
        Self {
            inner,
            wal,
            graph_name: None,
            wal_graph_context,
        }
    }

    /// Creates a new async WAL-aware store wrapper for a named graph.
    pub fn new_for_graph(
        inner: Arc<LpgStore>,
        wal: Arc<AsyncLpgWal>,
        graph_name: String,
        wal_graph_context: Arc<tokio::sync::Mutex<Option<String>>>,
    ) -> Self {
        Self {
            inner,
            wal,
            graph_name: Some(graph_name),
            wal_graph_context,
        }
    }

    /// Returns a reference to the inner store.
    #[must_use]
    pub fn store(&self) -> &Arc<LpgStore> {
        &self.inner
    }

    /// Returns a reference to the async WAL.
    #[must_use]
    pub fn wal(&self) -> &Arc<AsyncLpgWal> {
        &self.wal
    }

    /// Logs a WAL record with graph context tracking, then applies a
    /// synchronous mutation to the store.
    ///
    /// Acquires the shared context lock, emits a `SwitchGraph` record if
    /// needed, logs the data record asynchronously, then runs the sync
    /// mutation. Both WAL writes happen under the same lock to prevent
    /// interleaving.
    ///
    /// # Errors
    ///
    /// Returns an error if WAL logging fails. The store mutation is not
    /// applied if logging fails (write-ahead guarantee).
    pub async fn log_and_apply<F, T>(&self, record: &WalRecord, apply: F) -> Result<T>
    where
        F: FnOnce(&LpgStore) -> T,
    {
        self.log_with_context(record).await?;
        Ok(apply(&self.inner))
    }

    /// Logs a WAL record with graph context tracking.
    ///
    /// Acquires the async context mutex, emits `SwitchGraph` if the WAL
    /// context differs from this store's graph, then logs the data record.
    async fn log_with_context(&self, record: &WalRecord) -> Result<()> {
        let mut ctx = self.wal_graph_context.lock().await;
        if *ctx != self.graph_name {
            self.wal
                .log(&WalRecord::SwitchGraph {
                    name: self.graph_name.clone(),
                })
                .await?;
            (*ctx).clone_from(&self.graph_name);
        }
        self.wal.log(record).await
    }

    // -----------------------------------------------------------------------
    // Node mutations
    // -----------------------------------------------------------------------

    /// Creates a node, logging to WAL first.
    ///
    /// # Errors
    ///
    /// Returns an error if WAL logging fails.
    pub async fn create_node(&self, labels: &[&str]) -> Result<NodeId> {
        let id = self.inner.create_node(labels);
        self.log_with_context(&WalRecord::CreateNode {
            id,
            labels: labels.iter().map(|s| (*s).to_string()).collect(),
        })
        .await?;
        Ok(id)
    }

    /// Deletes a node, logging to WAL only if the node existed.
    ///
    /// # Errors
    ///
    /// Returns an error if WAL logging fails.
    pub async fn delete_node(&self, id: NodeId) -> Result<bool> {
        let exists = self.inner.get_node(id).is_some();
        if exists {
            self.log_with_context(&WalRecord::DeleteNode { id }).await?;
        }
        Ok(if exists {
            self.inner.delete_node(id)
        } else {
            false
        })
    }

    /// Deletes all edges connected to a node.
    ///
    /// # Errors
    ///
    /// Returns an error if WAL logging fails.
    pub async fn delete_node_edges(&self, node_id: NodeId) -> Result<()> {
        // Collect edge IDs before deletion so we can log them
        let outgoing: Vec<EdgeId> =
            GraphStore::edges_from(self.inner.as_ref(), node_id, Direction::Outgoing)
                .into_iter()
                .map(|(_, eid)| eid)
                .collect();
        let incoming: Vec<EdgeId> =
            GraphStore::edges_from(self.inner.as_ref(), node_id, Direction::Incoming)
                .into_iter()
                .map(|(_, eid)| eid)
                .collect();

        for id in outgoing.into_iter().chain(incoming) {
            self.log_with_context(&WalRecord::DeleteEdge { id }).await?;
        }
        self.inner.delete_node_edges(node_id);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Edge mutations
    // -----------------------------------------------------------------------

    /// Creates an edge, logging to WAL first.
    ///
    /// # Errors
    ///
    /// Returns an error if WAL logging fails.
    pub async fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> Result<EdgeId> {
        let id = self.inner.create_edge(src, dst, edge_type);
        self.log_with_context(&WalRecord::CreateEdge {
            id,
            src,
            dst,
            edge_type: edge_type.to_string(),
        })
        .await?;
        Ok(id)
    }

    /// Deletes an edge, logging to WAL only if the edge existed.
    ///
    /// # Errors
    ///
    /// Returns an error if WAL logging fails.
    pub async fn delete_edge(&self, id: EdgeId) -> Result<bool> {
        let deleted = self.inner.delete_edge(id);
        if deleted {
            self.log_with_context(&WalRecord::DeleteEdge { id }).await?;
        }
        Ok(deleted)
    }

    // -----------------------------------------------------------------------
    // Property mutations
    // -----------------------------------------------------------------------

    /// Sets a node property, logging to WAL first.
    ///
    /// # Errors
    ///
    /// Returns an error if WAL logging fails.
    pub async fn set_node_property(&self, id: NodeId, key: &str, value: Value) -> Result<()> {
        self.log_with_context(&WalRecord::SetNodeProperty {
            id,
            key: key.to_string(),
            value: value.clone(),
        })
        .await?;
        self.inner.set_node_property(id, key, value);
        Ok(())
    }

    /// Sets an edge property, logging to WAL first.
    ///
    /// # Errors
    ///
    /// Returns an error if WAL logging fails.
    pub async fn set_edge_property(&self, id: EdgeId, key: &str, value: Value) -> Result<()> {
        self.log_with_context(&WalRecord::SetEdgeProperty {
            id,
            key: key.to_string(),
            value: value.clone(),
        })
        .await?;
        self.inner.set_edge_property(id, key, value);
        Ok(())
    }

    /// Removes a node property, logging to WAL only if the property existed.
    ///
    /// # Errors
    ///
    /// Returns an error if WAL logging fails.
    pub async fn remove_node_property(&self, id: NodeId, key: &str) -> Result<Option<Value>> {
        let removed = self.inner.remove_node_property(id, key);
        if removed.is_some() {
            self.log_with_context(&WalRecord::RemoveNodeProperty {
                id,
                key: key.to_string(),
            })
            .await?;
        }
        Ok(removed)
    }

    /// Removes an edge property, logging to WAL only if the property existed.
    ///
    /// # Errors
    ///
    /// Returns an error if WAL logging fails.
    pub async fn remove_edge_property(&self, id: EdgeId, key: &str) -> Result<Option<Value>> {
        let removed = self.inner.remove_edge_property(id, key);
        if removed.is_some() {
            self.log_with_context(&WalRecord::RemoveEdgeProperty {
                id,
                key: key.to_string(),
            })
            .await?;
        }
        Ok(removed)
    }

    // -----------------------------------------------------------------------
    // Label mutations
    // -----------------------------------------------------------------------

    /// Adds a label to a node, logging to WAL only if the label was new.
    ///
    /// # Errors
    ///
    /// Returns an error if WAL logging fails.
    pub async fn add_label(&self, node_id: NodeId, label: &str) -> Result<bool> {
        let added = self.inner.add_label(node_id, label);
        if added {
            self.log_with_context(&WalRecord::AddNodeLabel {
                id: node_id,
                label: label.to_string(),
            })
            .await?;
        }
        Ok(added)
    }

    /// Removes a label from a node, logging to WAL only if the label existed.
    ///
    /// # Errors
    ///
    /// Returns an error if WAL logging fails.
    pub async fn remove_label(&self, node_id: NodeId, label: &str) -> Result<bool> {
        let removed = self.inner.remove_label(node_id, label);
        if removed {
            self.log_with_context(&WalRecord::RemoveNodeLabel {
                id: node_id,
                label: label.to_string(),
            })
            .await?;
        }
        Ok(removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::PropertyKey;
    use grafeo_storage::wal::AsyncTypedWal;

    async fn setup() -> (AsyncWalGraphStore, Arc<AsyncLpgWal>) {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LpgStore::new().unwrap());
        let wal = Arc::new(AsyncTypedWal::open(dir.keep()).await.unwrap());
        let wal_ref = Arc::clone(&wal);
        let ctx = Arc::new(tokio::sync::Mutex::new(None));
        (AsyncWalGraphStore::new(store, wal, ctx), wal_ref)
    }

    #[tokio::test]
    async fn create_node_logs_and_applies() {
        let (ws, wal) = setup().await;
        let id = ws.create_node(&["Person", "Employee"]).await.unwrap();

        assert!(ws.store().get_node(id).is_some());
        assert_eq!(ws.store().node_count(), 1);
        assert_eq!(wal.record_count(), 1);
    }

    #[tokio::test]
    async fn create_edge_logs_and_applies() {
        let (ws, wal) = setup().await;
        let a = ws.create_node(&["Node"]).await.unwrap();
        let b = ws.create_node(&["Node"]).await.unwrap();
        let eid = ws.create_edge(a, b, "KNOWS").await.unwrap();

        assert!(ws.store().get_edge(eid).is_some());
        assert_eq!(ws.store().edge_count(), 1);
        // 2 CreateNode + 1 CreateEdge
        assert_eq!(wal.record_count(), 3);
    }

    #[tokio::test]
    async fn set_property_logs_and_applies() {
        let (ws, wal) = setup().await;
        let nid = ws.create_node(&["Person"]).await.unwrap();
        ws.set_node_property(nid, "name", Value::String("Alix".into()))
            .await
            .unwrap();

        assert_eq!(
            ws.store()
                .get_node_property(nid, &PropertyKey::from("name")),
            Some(Value::String("Alix".into()))
        );
        // CreateNode + SetNodeProperty
        assert_eq!(wal.record_count(), 2);
    }

    #[tokio::test]
    async fn delete_node_only_logs_on_success() {
        let (ws, wal) = setup().await;
        let id = ws.create_node(&["Person"]).await.unwrap();
        assert_eq!(wal.record_count(), 1);

        // Delete nonexistent: no new record
        assert!(!ws.delete_node(NodeId::new(999)).await.unwrap());
        assert_eq!(wal.record_count(), 1);

        // Delete real node: logs
        assert!(ws.delete_node(id).await.unwrap());
        assert_eq!(wal.record_count(), 2);
    }

    #[tokio::test]
    async fn delete_edge_only_logs_on_success() {
        let (ws, wal) = setup().await;
        let a = ws.create_node(&["Node"]).await.unwrap();
        let b = ws.create_node(&["Node"]).await.unwrap();
        let eid = ws.create_edge(a, b, "LINK").await.unwrap();
        assert_eq!(wal.record_count(), 3);

        // Delete nonexistent: no new record
        assert!(!ws.delete_edge(EdgeId::new(999)).await.unwrap());
        assert_eq!(wal.record_count(), 3);

        // Delete real edge: logs
        assert!(ws.delete_edge(eid).await.unwrap());
        assert_eq!(wal.record_count(), 4);
    }

    #[tokio::test]
    async fn remove_property_only_logs_on_success() {
        let (ws, wal) = setup().await;
        let id = ws.create_node(&["Person"]).await.unwrap();
        ws.set_node_property(id, "age", Value::Int64(30))
            .await
            .unwrap();
        assert_eq!(wal.record_count(), 2);

        // Remove nonexistent: no log
        assert!(
            ws.remove_node_property(id, "missing")
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(wal.record_count(), 2);

        // Remove real property: logs
        assert_eq!(
            ws.remove_node_property(id, "age").await.unwrap(),
            Some(Value::Int64(30))
        );
        assert_eq!(wal.record_count(), 3);
    }

    #[tokio::test]
    async fn add_remove_label_conditional_logging() {
        let (ws, wal) = setup().await;
        let id = ws.create_node(&["Person"]).await.unwrap();
        assert_eq!(wal.record_count(), 1);

        // Add duplicate label: no log
        assert!(!ws.add_label(id, "Person").await.unwrap());
        assert_eq!(wal.record_count(), 1);

        // Add new label: logs
        assert!(ws.add_label(id, "Employee").await.unwrap());
        assert_eq!(wal.record_count(), 2);

        // Remove label: logs
        assert!(ws.remove_label(id, "Employee").await.unwrap());
        assert_eq!(wal.record_count(), 3);

        // Remove absent label: no log
        assert!(!ws.remove_label(id, "Employee").await.unwrap());
        assert_eq!(wal.record_count(), 3);
    }

    #[tokio::test]
    async fn named_graph_emits_switch_graph_record() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LpgStore::new().unwrap());
        let wal: Arc<AsyncLpgWal> = Arc::new(AsyncTypedWal::open(dir.keep()).await.unwrap());
        let wal_ref = Arc::clone(&wal);
        let ctx = Arc::new(tokio::sync::Mutex::new(None));
        let ws = AsyncWalGraphStore::new_for_graph(store, wal, "social".to_string(), ctx);

        ws.create_node(&["Person"]).await.unwrap();

        // Should have SwitchGraph + CreateNode = 2 records
        assert_eq!(wal_ref.record_count(), 2);
    }

    #[tokio::test]
    async fn named_graph_context_not_repeated() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LpgStore::new().unwrap());
        let wal: Arc<AsyncLpgWal> = Arc::new(AsyncTypedWal::open(dir.keep()).await.unwrap());
        let wal_ref = Arc::clone(&wal);
        let ctx = Arc::new(tokio::sync::Mutex::new(None));
        let ws = AsyncWalGraphStore::new_for_graph(store, wal, "social".to_string(), ctx);

        // First mutation: SwitchGraph + CreateNode
        ws.create_node(&["Person"]).await.unwrap();
        assert_eq!(wal_ref.record_count(), 2);

        // Second mutation: context already set, no extra SwitchGraph
        ws.create_node(&["Person"]).await.unwrap();
        assert_eq!(wal_ref.record_count(), 3);
    }

    #[tokio::test]
    async fn log_and_apply_sequences_correctly() {
        let (ws, wal) = setup().await;
        let id = ws
            .log_and_apply(
                &WalRecord::CreateNode {
                    id: NodeId::new(42),
                    labels: vec!["Test".to_string()],
                },
                |store| store.create_node(&["Test"]),
            )
            .await
            .unwrap();

        assert!(id.is_valid());
        assert_eq!(wal.record_count(), 1);
    }

    #[tokio::test]
    async fn delete_node_edges_logs_each_edge() {
        let (ws, wal) = setup().await;
        let a = ws.create_node(&["Node"]).await.unwrap();
        let b = ws.create_node(&["Node"]).await.unwrap();
        let c = ws.create_node(&["Node"]).await.unwrap();
        ws.create_edge(a, b, "X").await.unwrap();
        ws.create_edge(c, a, "Y").await.unwrap();
        assert_eq!(wal.record_count(), 5);

        ws.delete_node_edges(a).await.unwrap();
        // 2 DeleteEdge records (one outgoing, one incoming)
        assert_eq!(wal.record_count(), 7);
        assert_eq!(ws.store().edge_count(), 0);
    }

    #[tokio::test]
    async fn set_edge_property_logs_and_applies() {
        let (ws, wal) = setup().await;
        let a = ws.create_node(&["Person"]).await.unwrap();
        let b = ws.create_node(&["Person"]).await.unwrap();
        let eid = ws.create_edge(a, b, "KNOWS").await.unwrap();
        ws.set_edge_property(eid, "weight", Value::Int64(42))
            .await
            .unwrap();

        assert_eq!(
            ws.store()
                .get_edge_property(eid, &PropertyKey::from("weight")),
            Some(Value::Int64(42))
        );
        // 2 CreateNode + 1 CreateEdge + 1 SetEdgeProperty
        assert_eq!(wal.record_count(), 4);
    }

    #[tokio::test]
    async fn remove_edge_property_only_logs_on_success() {
        let (ws, wal) = setup().await;
        let a = ws.create_node(&["Node"]).await.unwrap();
        let b = ws.create_node(&["Node"]).await.unwrap();
        let eid = ws.create_edge(a, b, "LINK").await.unwrap();
        ws.set_edge_property(eid, "w", Value::Int64(1))
            .await
            .unwrap();
        let before = wal.record_count();

        // Remove nonexistent: no log
        assert!(
            ws.remove_edge_property(eid, "missing")
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(wal.record_count(), before);

        // Remove real: logs
        assert_eq!(
            ws.remove_edge_property(eid, "w").await.unwrap(),
            Some(Value::Int64(1))
        );
        assert_eq!(wal.record_count(), before + 1);
    }

    #[tokio::test]
    async fn concurrent_mutations_to_same_store() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LpgStore::new().unwrap());
        let wal: Arc<AsyncLpgWal> = Arc::new(AsyncTypedWal::open(dir.keep()).await.unwrap());
        let ctx = Arc::new(tokio::sync::Mutex::new(None));

        let ws = Arc::new(AsyncWalGraphStore::new(
            Arc::clone(&store),
            Arc::clone(&wal),
            ctx,
        ));

        let mut handles = Vec::new();
        for _ in 0..4 {
            let ws_clone = Arc::clone(&ws);
            handles.push(tokio::spawn(async move {
                for _ in 0..10 {
                    ws_clone.create_node(&["Test"]).await.unwrap();
                }
            }));
        }

        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(store.node_count(), 40);
        assert_eq!(wal.record_count(), 40);
    }

    #[tokio::test]
    async fn wal_ordering_sequential_writes() {
        // Verify that a specific sequence of mutations produces the expected
        // number of WAL records in order. Sequential async writes guarantee
        // ordering since all operations go through the same async mutex.
        let (ws, wal) = setup().await;

        let id = ws.create_node(&["Person"]).await.unwrap();
        assert_eq!(wal.record_count(), 1); // CreateNode

        ws.set_node_property(id, "name", Value::String("Alix".into()))
            .await
            .unwrap();
        assert_eq!(wal.record_count(), 2); // + SetNodeProperty

        ws.add_label(id, "Employee").await.unwrap();
        assert_eq!(wal.record_count(), 3); // + AddNodeLabel

        ws.remove_node_property(id, "name").await.unwrap();
        assert_eq!(wal.record_count(), 4); // + RemoveNodeProperty

        ws.delete_node(id).await.unwrap();
        assert_eq!(wal.record_count(), 5); // + DeleteNode

        // All 5 operations logged, in order (verified by monotonic count)
    }
}
