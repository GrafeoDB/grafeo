//! The direct write API: each call checked through a [`GraphWriter`] and
//! committed at an epoch of its own.
//!
//! While no transaction is open, a call runs without a session or a
//! transaction-manager transaction. It holds the manager's idle gate, so no
//! transaction can begin meanwhile and there is nothing it could conflict
//! with, and it writes as the system, stamped at its new epoch, which it
//! publishes when done. Every check of a single call runs before it writes,
//! so the call cannot half-apply; a batch writes as a private transaction
//! instead, which is undone when a later row fails. A call that fails still
//! uses up its epoch: the stores take it before the write, to stamp what they
//! record themselves, and it is not handed out twice. The gap it leaves holds
//! no data. While a transaction is open, the call runs as an implicit
//! transaction of a session and is checked for conflicts with the open one.
//!
//! This keeps a direct call close to the cost of the store write itself. Once
//! transactions own their change set (#448), every direct call becomes an
//! implicit transaction again at about this cost, and this path goes.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, Value};
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};
use grafeo_core::execution::operators::{GraphWriter, OperatorError};
use grafeo_core::graph::lpg::{Edge, LpgStore, Node};
use grafeo_core::graph::{GraphStoreMut, GraphStoreSearch};

use super::GrafeoDB;
use crate::catalog::CatalogConstraintValidator;
use crate::session::graph_storage_key;

/// The graph a direct call works in.
#[derive(Clone, Copy)]
pub(crate) enum DirectTarget<'a> {
    /// The graph `set_current_graph` and `set_current_schema` select, or the
    /// default graph when they select none. A selected graph that no longer
    /// exists is an error.
    Current,
    /// A named graph of `schema`, which must exist (a graph handle).
    Named {
        schema: Option<&'a str>,
        name: &'a str,
    },
}

/// Buffers the direct calls outside a transaction share. Only the call
/// holding the transaction manager's idle gate uses them.
#[cfg(any(feature = "wal", feature = "cdc"))]
#[derive(Default)]
pub(crate) struct ImplicitWrites {
    /// The WAL records of the running call, written as one group.
    #[cfg(feature = "wal")]
    wal: std::sync::OnceLock<Arc<crate::transaction::wal_buffer::WalBuffer>>,
    /// The CDC events of the running call, recorded at its epoch.
    #[cfg(feature = "cdc")]
    cdc_events: Arc<parking_lot::Mutex<Vec<crate::cdc::ChangeEvent>>>,
}

/// An edge to create with [`GrafeoDB::batch_create_edges`]: its endpoints,
/// type and properties.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchEdge {
    /// The source node.
    pub src: NodeId,
    /// The target node.
    pub dst: NodeId,
    /// The edge type.
    pub edge_type: String,
    /// The edge's properties.
    pub properties: HashMap<PropertyKey, Value>,
}

impl BatchEdge {
    /// An edge without properties.
    #[must_use]
    pub fn new(src: NodeId, dst: NodeId, edge_type: impl Into<String>) -> Self {
        Self {
            src,
            dst,
            edge_type: edge_type.into(),
            properties: HashMap::new(),
        }
    }

    /// The edge with `properties`.
    #[must_use]
    pub fn with_properties(
        mut self,
        properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>,
    ) -> Self {
        self.properties = properties
            .into_iter()
            .map(|(key, value)| (key.into(), value.into()))
            .collect();
        self
    }
}

/// The direct API on one graph, shared by [`GrafeoDB`] (its current graph)
/// and [`GraphHandle`](super::GraphHandle) (the handle's graph).
pub(crate) struct DirectCalls<'a> {
    db: &'a GrafeoDB,
    target: DirectTarget<'a>,
}

impl GrafeoDB {
    /// The direct API on `target`.
    pub(crate) fn direct<'a>(&'a self, target: DirectTarget<'a>) -> DirectCalls<'a> {
        DirectCalls { db: self, target }
    }

    /// Whether direct calls outside a transaction can skip the session: not
    /// on a read-only database, and not on an external store.
    fn writes_without_session(&self) -> bool {
        !self.read_only && self.root_store().is_some() && self.external_read_store.is_none()
    }

    /// The built-in store of the graph `target` names and its storage key
    /// (`None`: the default graph), or `None` without a built-in store.
    ///
    /// # Errors
    ///
    /// Returns an error if `target` names a graph that does not exist.
    fn direct_store(
        &self,
        target: DirectTarget<'_>,
    ) -> Result<Option<(Arc<LpgStore>, Option<String>)>> {
        let Some(root) = self.root_store() else {
            return Ok(None);
        };
        let key = match target {
            DirectTarget::Current => graph_storage_key(
                self.current_schema.read().as_deref(),
                self.current_graph.read().as_deref(),
            ),
            DirectTarget::Named { schema, name } => graph_storage_key(schema, Some(name)),
        };
        let Some(key) = key else {
            return Ok(Some((root, None)));
        };
        if let Some(store) = root.graph(&key) {
            return Ok(Some((store, Some(key))));
        }
        // The graph was dropped, or its schema: never fall back to another.
        let name = match target {
            DirectTarget::Named { name, .. } => name.to_string(),
            DirectTarget::Current => self
                .current_graph
                .read()
                .clone()
                .unwrap_or_else(|| "default".to_string()),
        };
        Err(missing_graph(&name))
    }

    /// The store the direct API reads the graph `target` names from: its own
    /// store for a named graph; for the default graph the store queries read,
    /// which is the external store of a database built with `with_store` or
    /// `with_read_store`.
    ///
    /// # Errors
    ///
    /// Returns an error if `target` names a graph that does not exist.
    pub(crate) fn read_store(&self, target: DirectTarget<'_>) -> Result<Arc<dyn GraphStoreSearch>> {
        Ok(match self.direct_store(target)? {
            Some((store, Some(_))) => store,
            _ => self.graph_store(),
        })
    }

    /// Runs one direct call on `target`: without a session while no
    /// transaction is open, otherwise as an implicit transaction of a session.
    fn write_direct<T>(
        &self,
        target: DirectTarget<'_>,
        batch: bool,
        write: impl FnOnce(&GraphWriter) -> std::result::Result<T, OperatorError>,
    ) -> Result<T> {
        if let Some((store, graph)) = self.direct_store(target)?
            && self.writes_without_session()
            && let Some(_gate) = self.transaction_manager.idle_gate()
        {
            return self.write_outside_transaction(&store, graph.as_deref(), batch, write);
        }
        let session = match target {
            DirectTarget::Current => self.session(),
            DirectTarget::Named { schema, name } => self.graph_in(schema, name)?.session()?,
        };
        session.write(write)
    }

    /// Writes one direct call to `store` (the graph with storage key `graph`)
    /// and commits it at a new epoch. The caller holds the idle gate; this
    /// holds commits off (see
    /// [`TransactionManager::hold_commits`](crate::transaction::TransactionManager))
    /// from its epoch until it is published, so no checkpoint holds part of
    /// it (lock order: the idle gate, then the commit lock, as in `begin`).
    /// Fails, like a commit, once the database is closed.
    fn write_outside_transaction<T>(
        &self,
        store: &Arc<LpgStore>,
        graph: Option<&str>,
        batch: bool,
        write: impl FnOnce(&GraphWriter) -> std::result::Result<T, OperatorError>,
    ) -> Result<T> {
        let commits = self.transaction_manager.hold_commits_for_change()?;
        let root = self.lpg_store();
        let read_epoch = self.transaction_manager.current_epoch();
        let epoch = EpochId::new(read_epoch.as_u64() + 1);
        // A batch versions its writes so it can undo them; a single call
        // writes as the system at the new epoch, which the stores use for
        // what they stamp themselves (property and label history, CDC).
        let transaction = batch.then(|| self.transaction_manager.reserve_transaction_id());
        let view = if transaction.is_some() {
            read_epoch
        } else {
            root.sync_epoch(epoch);
            store.sync_epoch(epoch);
            epoch
        };
        // Tests start a checkpoint or `close()` here (a single call's epoch
        // has moved, nothing is published yet), which must wait.
        #[cfg(feature = "testing-statement-injection")]
        grafeo_common::testing::commit_hook::run_during_held_change();

        let target: Arc<dyn GraphStoreMut> = Arc::clone(store) as Arc<dyn GraphStoreMut>;
        #[cfg(feature = "wal")]
        let wal = self.wal.as_ref().map(|wal| {
            Arc::clone(self.implicit_writes.wal.get_or_init(|| {
                Arc::new(crate::transaction::wal_buffer::WalBuffer::new(Arc::clone(
                    wal,
                )))
            }))
        });
        // The buffers hold only this call's records and events: clear what a
        // call that failed without cleaning up may have left.
        #[cfg(feature = "wal")]
        if let Some(buffer) = &wal {
            buffer.clear();
        }
        #[cfg(feature = "cdc")]
        self.implicit_writes.cdc_events.lock().clear();
        #[cfg(feature = "wal")]
        let target: Arc<dyn GraphStoreMut> = match &wal {
            Some(buffer) => {
                use super::wal_store::WalGraphStore;
                Arc::new(match graph {
                    None => WalGraphStore::new(Arc::clone(store), Arc::clone(buffer)),
                    Some(name) => WalGraphStore::new_for_graph(
                        Arc::clone(store),
                        Arc::clone(buffer),
                        name.to_string(),
                    ),
                })
            }
            None => target,
        };
        #[cfg(not(feature = "wal"))]
        let _ = graph;
        #[cfg(feature = "cdc")]
        let target: Arc<dyn GraphStoreMut> = if self.cdc_active() {
            Arc::new(super::cdc_store::CdcGraphStore::wrap_buffered(
                target,
                Arc::clone(&self.cdc_log),
                Arc::clone(&self.implicit_writes.cdc_events),
            ))
        } else {
            target
        };

        // A graph of a schema has the storage key `schema/graph`: the types of
        // that schema check the call.
        let schema = graph
            .and_then(|key| key.split_once('/'))
            .map(|(schema, _)| schema);
        let mut validator = CatalogConstraintValidator::new(Arc::clone(&self.catalog))
            .with_store(Arc::clone(store) as Arc<dyn GraphStoreSearch>)
            .with_max_property_size(self.config.max_property_size)
            .with_transaction_context(view, transaction)
            .with_schema(schema);
        if let Some(graph) = graph {
            validator = validator.with_graph_name(graph);
        }
        let writer = GraphWriter::new(target)
            .with_transaction_context(view, transaction)
            .with_validator(Arc::new(validator));
        // A panic in the call is handled like an error, then raised again:
        // left alone, its records and events would stay in the buffers for
        // the next call to commit.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| write(&writer)));
        drop(writer);
        let (result, panic) = match outcome {
            Ok(result) => (
                result.map_err(crate::query::executor::convert_operator_error),
                None,
            ),
            Err(panic) => (
                Err(Error::Internal("the direct call panicked".to_string())),
                Some(panic),
            ),
        };

        match (&result, transaction) {
            (Err(_), Some(transaction)) => {
                store.discard_uncommitted_versions(transaction);
                #[cfg(feature = "wal")]
                if let Some(buffer) = &wal {
                    buffer.clear();
                }
                #[cfg(feature = "cdc")]
                self.implicit_writes.cdc_events.lock().clear();
                if let Some(panic) = panic {
                    std::panic::resume_unwind(panic);
                }
                return result;
            }
            (Ok(_), Some(transaction)) => {
                store.finalize_version_epochs(transaction, epoch);
                store.commit_transaction_properties(transaction);
                root.sync_epoch(epoch);
                store.sync_epoch(epoch);
            }
            // A single call fails before it writes; should it ever fail
            // later, what it wrote is committed so the WAL matches memory.
            (_, None) => {}
        }

        #[cfg(feature = "wal")]
        if let Some(buffer) = &wal
            && buffer.len() > 0
        {
            use grafeo_storage::wal::WalRecord;
            if let Err(e) = buffer.flush(&[
                WalRecord::TransactionCommit {
                    transaction_id: transaction
                        .unwrap_or(grafeo_common::types::TransactionId::SYSTEM),
                },
                WalRecord::EpochAdvance { epoch },
            ]) {
                grafeo_common::grafeo_warn!("Failed to write a direct write to the WAL: {}", e);
            }
        }
        self.transaction_manager.sync_epoch(epoch);
        drop(commits);
        #[cfg(feature = "cdc")]
        {
            let events = std::mem::take(&mut *self.implicit_writes.cdc_events.lock());
            if !events.is_empty() {
                self.cdc_log
                    .record_batch(crate::cdc::fold_into_creates(events).into_iter().map(
                        |mut event| {
                            event.epoch = epoch;
                            event
                        },
                    ));
            }
        }

        // Every gc_interval commits, prune versions no reader needs.
        if self.config.gc_interval > 0 {
            let count = self.commit_counter.fetch_add(1, Ordering::Relaxed) + 1;
            if count.is_multiple_of(self.config.gc_interval) {
                let min_epoch = self.transaction_manager.min_active_epoch();
                root.gc_versions(min_epoch);
                if !Arc::ptr_eq(&root, store) {
                    store.gc_versions(min_epoch);
                }
                self.transaction_manager.gc();
            }
        }
        if let Some(panic) = panic {
            std::panic::resume_unwind(panic);
        }
        result
    }
}

impl DirectCalls<'_> {
    fn write<T>(
        &self,
        write: impl FnOnce(&GraphWriter) -> std::result::Result<T, OperatorError>,
    ) -> Result<T> {
        self.db.write_direct(self.target, false, write)
    }

    fn write_batch<T>(
        &self,
        write: impl FnOnce(&GraphWriter) -> std::result::Result<T, OperatorError>,
    ) -> Result<T> {
        self.db.write_direct(self.target, true, write)
    }

    /// The store of the graph, for reads (see [`GrafeoDB::read_store`]).
    fn store(&self) -> Result<Arc<dyn GraphStoreSearch>> {
        self.db.read_store(self.target)
    }

    pub(crate) fn create_node_with_props(
        &self,
        labels: &[&str],
        properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>,
    ) -> Result<NodeId> {
        let labels: Vec<String> = labels.iter().map(|label| (*label).to_string()).collect();
        let properties = direct_properties(properties);
        self.write(|writer| writer.create_node(&labels, properties))
    }

    pub(crate) fn create_edge_with_props(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>,
    ) -> Result<EdgeId> {
        let properties = direct_properties(properties);
        self.write(|writer| create_edge(writer, src, dst, edge_type, properties))
    }

    pub(crate) fn set_node_property(&self, id: NodeId, key: &str, value: Value) -> Result<()> {
        self.write(|writer| set_node_property(writer, id, key, value))
    }

    pub(crate) fn set_edge_property(&self, id: EdgeId, key: &str, value: Value) -> Result<()> {
        self.write(|writer| set_edge_property(writer, id, key, value))
    }

    pub(crate) fn remove_node_property(&self, id: NodeId, key: &str) -> Result<bool> {
        self.write(
            // The direct API reports a missing node as `false`; a query that
            // writes to one fails (see `GraphWriter`).
            |writer| {
                if writer.has_node(id) {
                    writer.remove_node_property(id, key)
                } else {
                    Ok(false)
                }
            },
        )
    }

    pub(crate) fn remove_edge_property(&self, id: EdgeId, key: &str) -> Result<bool> {
        self.write(|writer| {
            if writer.has_edge(id) {
                writer.remove_edge_property(id, key)
            } else {
                Ok(false)
            }
        })
    }

    pub(crate) fn add_node_label(&self, id: NodeId, label: &str) -> Result<bool> {
        self.write(|writer| add_node_label(writer, id, label))
    }

    pub(crate) fn remove_node_label(&self, id: NodeId, label: &str) -> Result<bool> {
        self.write(|writer| remove_node_label(writer, id, label))
    }

    pub(crate) fn delete_node(&self, id: NodeId) -> Result<bool> {
        self.write(|writer| writer.delete_node(id, false))
    }

    pub(crate) fn delete_edge(&self, id: EdgeId) -> Result<bool> {
        self.write(|writer| writer.delete_edge(id))
    }

    pub(crate) fn batch_create_nodes(
        &self,
        label: &str,
        property: &str,
        vectors: Vec<Vec<f32>>,
    ) -> Result<Vec<NodeId>> {
        self.write_batch(|writer| create_vector_nodes(writer, label, property, vectors))
    }

    pub(crate) fn batch_create_nodes_with_labels(
        &self,
        labels: &[&str],
        properties_list: Vec<HashMap<PropertyKey, Value>>,
    ) -> Result<Vec<NodeId>> {
        self.write_batch(|writer| create_nodes(writer, labels, properties_list))
    }

    pub(crate) fn batch_create_edges(&self, edges: Vec<BatchEdge>) -> Result<Vec<EdgeId>> {
        self.write_batch(|writer| create_edges(writer, edges))
    }

    pub(crate) fn get_node(&self, id: NodeId) -> Result<Option<Node>> {
        let epoch = self.db.read_epoch();
        Ok(self.store()?.get_node_at_epoch(id, epoch))
    }

    pub(crate) fn get_edge(&self, id: EdgeId) -> Result<Option<Edge>> {
        let epoch = self.db.read_epoch();
        Ok(self.store()?.get_edge_at_epoch(id, epoch))
    }
}

// === The writes of the direct API, shared with `Session` ===

/// Creates an edge between two nodes the writer sees.
pub(crate) fn create_edge(
    writer: &GraphWriter,
    src: NodeId,
    dst: NodeId,
    edge_type: &str,
    properties: Vec<(String, Value)>,
) -> std::result::Result<EdgeId, OperatorError> {
    for endpoint in [src, dst] {
        if !writer.has_node(endpoint) {
            return Err(missing_node(endpoint));
        }
    }
    writer.create_edge(src, dst, edge_type, properties)
}

/// Sets a property of a node the writer sees.
pub(crate) fn set_node_property(
    writer: &GraphWriter,
    id: NodeId,
    key: &str,
    value: Value,
) -> std::result::Result<(), OperatorError> {
    if !writer.has_node(id) {
        return Err(missing_node(id));
    }
    writer.set_node_properties(id, &[(key.to_string(), value)], false)
}

/// Sets a property of an edge the writer sees.
pub(crate) fn set_edge_property(
    writer: &GraphWriter,
    id: EdgeId,
    key: &str,
    value: Value,
) -> std::result::Result<(), OperatorError> {
    if !writer.has_edge(id) {
        return Err(OperatorError::Execution(format!(
            "edge {} does not exist",
            id.as_u64()
        )));
    }
    writer.set_edge_properties(id, &[(key.to_string(), value)], false)
}

/// Adds a label; whether the node exists and lacked it.
pub(crate) fn add_node_label(
    writer: &GraphWriter,
    id: NodeId,
    label: &str,
) -> std::result::Result<bool, OperatorError> {
    if !writer.has_node(id) {
        return Ok(false);
    }
    Ok(writer.add_labels(id, &[label.to_string()])? == 1)
}

/// Removes a label; whether the node exists and had it.
pub(crate) fn remove_node_label(
    writer: &GraphWriter,
    id: NodeId,
    label: &str,
) -> std::result::Result<bool, OperatorError> {
    if !writer.has_node(id) {
        return Ok(false);
    }
    Ok(writer.remove_labels(id, &[label.to_string()])? == 1)
}

/// Creates one node with `label` per vector, the vector as `property`.
pub(crate) fn create_vector_nodes(
    writer: &GraphWriter,
    label: &str,
    property: &str,
    vectors: Vec<Vec<f32>>,
) -> std::result::Result<Vec<NodeId>, OperatorError> {
    let labels = [label.to_string()];
    vectors
        .into_iter()
        .map(|vector| {
            writer.create_node(
                &labels,
                vec![(property.to_string(), Value::Vector(vector.into()))],
            )
        })
        .collect()
}

/// Creates one node with all of `labels` per property map.
pub(crate) fn create_nodes(
    writer: &GraphWriter,
    labels: &[&str],
    properties_list: Vec<HashMap<PropertyKey, Value>>,
) -> std::result::Result<Vec<NodeId>, OperatorError> {
    let labels: Vec<String> = labels.iter().map(|label| (*label).to_string()).collect();
    properties_list
        .into_iter()
        .map(|properties| writer.create_node(&labels, direct_properties(properties)))
        .collect()
}

/// Creates the edges, each between two nodes the writer sees.
pub(crate) fn create_edges(
    writer: &GraphWriter,
    edges: Vec<BatchEdge>,
) -> std::result::Result<Vec<EdgeId>, OperatorError> {
    edges
        .into_iter()
        .map(|edge| {
            create_edge(
                writer,
                edge.src,
                edge.dst,
                &edge.edge_type,
                direct_properties(edge.properties),
            )
        })
        .collect()
}

/// The properties of a direct write as `(key, value)` pairs.
pub(crate) fn direct_properties(
    properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>,
) -> Vec<(String, Value)> {
    properties
        .into_iter()
        .map(|(key, value)| {
            let key: PropertyKey = key.into();
            (key.as_str().to_string(), value.into())
        })
        .collect()
}

/// The error for a direct write to a node that does not exist.
fn missing_node(id: NodeId) -> OperatorError {
    OperatorError::Execution(format!("node {} does not exist", id.as_u64()))
}

/// The error for a graph handle whose graph does not exist.
pub(crate) fn missing_graph(name: &str) -> Error {
    Error::Query(QueryError::new(
        QueryErrorKind::Semantic,
        format!("Graph '{name}' does not exist"),
    ))
}

#[cfg(all(
    test,
    feature = "wal",
    feature = "grafeo-file",
    feature = "cdc",
    feature = "gql"
))]
mod tests {
    use grafeo_common::types::EpochId;

    use super::*;
    use crate::cdc::EntityId;
    use crate::config::{Config, StorageFormat};

    /// Runs a direct call on `db` that creates a `label` node and panics.
    fn panicking_call(db: &GrafeoDB, batch: bool, label: &str) -> NodeId {
        let created = parking_lot::Mutex::new(None);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            db.write_direct::<()>(DirectTarget::Current, batch, |writer| {
                *created.lock() = Some(writer.create_node(&[label.to_string()], Vec::new())?);
                panic!("the call fails halfway");
            })
        }));
        assert!(outcome.is_err(), "the panic comes through");
        created.into_inner().unwrap()
    }

    /// Tells [`a_panicking_call_leaves_nothing_for_the_next`], run in a child
    /// process, where its database is.
    const CHILD_PATH_VAR: &str = "GRAFEO_DIRECT_PANICKING_CALL_PATH";

    /// A batch that panics leaves nothing: its versions are gone, and the
    /// next call writes none of its WAL records or change events. A single
    /// call writes in place, so what it wrote before the panic is committed,
    /// as after an error: the WAL matches memory. The calls run in a child
    /// process that exits without `close()`, so the reopen replays the WAL.
    #[test]
    fn a_panicking_call_leaves_nothing_for_the_next() {
        let config = |path: &std::path::Path| {
            Config::persistent(path)
                .with_storage_format(StorageFormat::Auto)
                .with_cdc()
        };
        if let Some(path) = std::env::var_os(CHILD_PATH_VAR) {
            let db = GrafeoDB::with_config(config(std::path::Path::new(&path))).unwrap();
            writes_and_panics(&db);
            // Crash: no close(), no checkpoint, no destructors.
            std::process::exit(0);
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        let status = grafeo_common::testing::child_process::run(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "database::direct::tests::a_panicking_call_leaves_nothing_for_the_next",
                    "--nocapture",
                ])
                .env(CHILD_PATH_VAR, &path),
        )
        .unwrap();
        assert!(status.success(), "the child process failed");
        assert!(path.exists(), "the child process created no database");

        let db = GrafeoDB::with_config(config(&path)).unwrap();
        let labels = db
            .execute("MATCH (n) RETURN labels(n)[0] AS label ORDER BY label")
            .unwrap();
        assert_eq!(
            labels.rows(),
            [[Value::from("Person")], [Value::from("Single")]]
        );
        db.close().unwrap();
    }

    /// The calls of [`a_panicking_call_leaves_nothing_for_the_next`], with
    /// what memory holds after them.
    fn writes_and_panics(db: &GrafeoDB) {
        let batch = panicking_call(db, true, "Batch");
        let single = panicking_call(db, false, "Single");
        let alix = db
            .create_node_with_props(&["Person"], [("name", Value::from("Alix"))])
            .unwrap();

        assert!(
            db.get_node(batch).is_none(),
            "the batch's node is discarded"
        );
        assert!(db.get_node(single).is_some());
        let events: Vec<EntityId> = db
            .changes_between(EpochId::new(0), db.current_epoch())
            .unwrap()
            .into_iter()
            .map(|event| event.entity_id)
            .collect();
        assert_eq!(events, [EntityId::Node(single), EntityId::Node(alix)]);
    }
}
