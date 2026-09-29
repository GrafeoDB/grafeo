//! Main entry point for using Grafeo from Node.js.
//!
//! [`JsGrafeoDB`] wraps the Rust database engine and gives you a JavaScript API.

use std::collections::HashMap;
use std::sync::Arc;

use napi::JsString;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use parking_lot::RwLock;

use grafeo_common::types::{EdgeId, NodeId, Value};
use grafeo_engine::config::Config;
use grafeo_engine::database::{GrafeoDB, QueryResult as EngineQueryResult};

use crate::error::NodeGrafeoError;
use crate::graph::{JsEdge, JsNode};
use crate::query::QueryResult;
use crate::transaction::Transaction;
use crate::types;

/// Converts a serde_json filter map to a Grafeo filter map.
fn convert_json_filters(
    filters: Option<HashMap<String, serde_json::Value>>,
) -> Result<Option<HashMap<String, Value>>> {
    let Some(map) = filters else {
        return Ok(None);
    };
    let mut result = HashMap::new();
    for (key, val) in &map {
        let grafeo_val = json_to_value(val)?;
        result.insert(key.clone(), grafeo_val);
    }
    Ok(Some(result))
}

/// Validate a JavaScript number as a safe node ID.
///
/// JavaScript numbers are f64, but entity IDs are u64. This rejects
/// negative values, NaN, Infinity, and values beyond `Number.MAX_SAFE_INTEGER`.
fn validate_node_id(id: f64) -> Result<NodeId> {
    if !(0.0..=9_007_199_254_740_991.0).contains(&id) {
        return Err(NodeGrafeoError::InvalidArgument(format!("Invalid node ID: {id}")).into());
    }
    // reason: Range check above guarantees the value is in [0, 2^53-1], safe for u64
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Ok(NodeId(id as u64))
}

/// Validate a JavaScript number as a safe edge ID.
fn validate_edge_id(id: f64) -> Result<EdgeId> {
    if !(0.0..=9_007_199_254_740_991.0).contains(&id) {
        return Err(NodeGrafeoError::InvalidArgument(format!("Invalid edge ID: {id}")).into());
    }
    // reason: Range check above guarantees the value is in [0, 2^53-1], safe for u64
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Ok(EdgeId(id as u64))
}

/// Validate a JavaScript number as a non-negative epoch ID.
///
/// Rejects negative values, NaN, Infinity, and values beyond
/// `Number.MAX_SAFE_INTEGER`. Epochs are unsigned 64-bit integers internally.
fn validate_epoch(epoch: f64) -> Result<grafeo_common::types::EpochId> {
    if !(0.0..=9_007_199_254_740_991.0).contains(&epoch) {
        return Err(NodeGrafeoError::InvalidArgument(format!("Invalid epoch: {epoch}")).into());
    }
    // reason: Range check above guarantees the value is in [0, 2^53-1], safe for u64
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Ok(grafeo_common::types::EpochId::new(epoch as u64))
}

/// Your connection to a Grafeo database.
#[napi(js_name = "GrafeoDB")]
pub struct JsGrafeoDB {
    inner: Arc<RwLock<GrafeoDB>>,
}

#[napi]
impl JsGrafeoDB {
    /// Create a database. Pass a path for persistence, or omit for in-memory.
    #[napi(factory)]
    pub fn create(path: Option<String>) -> Result<Self> {
        let config = match path {
            Some(p) => Config::persistent(p),
            None => Config::in_memory(),
        };
        let db = GrafeoDB::with_config(config).map_err(NodeGrafeoError::from)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(db)),
        })
    }

    /// Open an existing database at the given path.
    #[napi(factory)]
    pub fn open(path: String) -> Result<Self> {
        let config = Config::persistent(path);
        let db = GrafeoDB::with_config(config).map_err(NodeGrafeoError::from)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(db)),
        })
    }

    /// Open an existing database in read-only mode.
    ///
    /// Uses a shared file lock, so multiple processes can read the same
    /// .grafeo file concurrently. Mutations will throw an error.
    #[napi(factory)]
    pub fn open_read_only(path: String) -> Result<Self> {
        let config = Config::read_only(path);
        let db = GrafeoDB::with_config(config).map_err(NodeGrafeoError::from)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(db)),
        })
    }

    /// Shared implementation for all language-specific execute methods.
    async fn execute_language_impl(
        &self,
        language: &'static str,
        query: String,
        params: Option<serde_json::Value>,
    ) -> Result<QueryResult> {
        let db = self.inner.clone();
        let mut result = tokio::task::spawn_blocking(move || {
            let db = db.read();
            execute_language_query(&db, &query, language, params.as_ref())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))??;

        let db = self.inner.read();
        let (nodes, edges) = extract_entities(&result, &db);
        let columns = std::mem::take(&mut result.columns);
        let exec_time = result.execution_time_ms;
        let scanned = result.rows_scanned;

        Ok(QueryResult::with_metrics(
            columns,
            result.into_rows(),
            nodes,
            edges,
            exec_time,
            scanned,
        ))
    }

    /// Execute a GQL query. Returns a Promise<QueryResult>.
    #[napi]
    pub async fn execute(
        &self,
        query: String,
        params: Option<serde_json::Value>,
    ) -> Result<QueryResult> {
        self.execute_language_impl("gql", query, params).await
    }

    /// Runs a read-only GQL query and returns an async cursor.
    ///
    /// The returned `ResultStream` yields one row at a time via `await
    /// stream.next()`. Memory stays bounded regardless of result-set size.
    ///
    /// Rejects mutations (INSERT / DELETE / SET), EXPLAIN / PROFILE,
    /// SESSION / SCHEMA commands, and queries that require push-based
    /// operators (ORDER BY, aggregate, DISTINCT). Use `execute()` for those.
    ///
    /// **Stability:** experimental. Parameterized streaming queries are not
    /// yet supported.
    #[napi(js_name = "executeStream")]
    pub async fn execute_stream(&self, query: String) -> Result<crate::stream::JsResultStream> {
        let db_arc = Arc::clone(&self.inner);
        let keepalive = Arc::clone(&self.inner);
        let stream = tokio::task::spawn_blocking(move || {
            let db = db_arc.read();
            db.execute_streaming(&query).map_err(NodeGrafeoError::from)
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))??;
        Ok(crate::stream::JsResultStream::new(keepalive, stream))
    }

    /// Create a node with labels and optional properties.
    #[napi(js_name = "createNode")]
    pub fn create_node(
        &self,
        env: Env,
        labels: Vec<String>,
        properties: Option<Object<'_>>,
    ) -> Result<JsNode> {
        let db = self.inner.read();
        let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

        let id = if let Some(props_obj) = properties {
            let mut props = Vec::new();
            let keys = props_obj.get_property_names()?;
            let len = keys.get_array_length()?;
            for i in 0..len {
                let key: JsString = keys.get_element(i)?;
                let key_str = key.into_utf8()?.into_owned()?;
                let value: Unknown<'_> = props_obj.get_named_property(&key_str)?;
                let val = types::js_to_value(&env, value)?;
                props.push((grafeo_common::types::PropertyKey::new(key_str), val));
            }
            db.create_node_with_props(&label_refs, props)
                .map_err(NodeGrafeoError::from)?
        } else {
            db.create_node(&label_refs).map_err(NodeGrafeoError::from)?
        };

        fetch_node(&db, id)
    }

    /// Create an edge between two nodes.
    #[napi(js_name = "createEdge")]
    pub fn create_edge(
        &self,
        env: Env,
        source_id: f64,
        target_id: f64,
        edge_type: String,
        properties: Option<Object<'_>>,
    ) -> Result<JsEdge> {
        let db = self.inner.read();
        let src = validate_node_id(source_id)?;
        let dst = validate_node_id(target_id)?;

        let id = if let Some(props_obj) = properties {
            let mut props = Vec::new();
            let keys = props_obj.get_property_names()?;
            let len = keys.get_array_length()?;
            for i in 0..len {
                let key: JsString = keys.get_element(i)?;
                let key_str = key.into_utf8()?.into_owned()?;
                let value: Unknown<'_> = props_obj.get_named_property(&key_str)?;
                let val = types::js_to_value(&env, value)?;
                props.push((grafeo_common::types::PropertyKey::new(key_str), val));
            }
            db.create_edge_with_props(src, dst, &edge_type, props)
                .map_err(NodeGrafeoError::from)?
        } else {
            db.create_edge(src, dst, &edge_type)
                .map_err(NodeGrafeoError::from)?
        };

        fetch_edge(&db, id)
    }

    /// Get a node by ID.
    #[napi(js_name = "getNode")]
    pub fn get_node(&self, id: f64) -> Result<Option<JsNode>> {
        let node_id = validate_node_id(id)?;
        let db = self.inner.read();
        Ok(db.get_node(node_id).map(|node| {
            let labels: Vec<String> = node.labels.iter().map(|s| s.to_string()).collect();
            let properties = node.properties.into_iter().collect();
            JsNode::new(node_id, labels, properties)
        }))
    }

    /// Get an edge by ID.
    #[napi(js_name = "getEdge")]
    pub fn get_edge(&self, id: f64) -> Result<Option<JsEdge>> {
        let edge_id = validate_edge_id(id)?;
        let db = self.inner.read();
        Ok(db.get_edge(edge_id).map(|edge| {
            let properties = edge.properties.into_iter().collect();
            JsEdge::new(
                edge_id,
                edge.edge_type.to_string(),
                edge.src,
                edge.dst,
                properties,
            )
        }))
    }

    /// Delete a node by ID. Returns true if the node existed.
    #[napi(js_name = "deleteNode")]
    pub fn delete_node(&self, id: f64) -> Result<bool> {
        let node_id = validate_node_id(id)?;
        let db = self.inner.read();
        Ok(db.delete_node(node_id).map_err(NodeGrafeoError::from)?)
    }

    /// Delete an edge by ID. Returns true if the edge existed.
    #[napi(js_name = "deleteEdge")]
    pub fn delete_edge(&self, id: f64) -> Result<bool> {
        let edge_id = validate_edge_id(id)?;
        let db = self.inner.read();
        Ok(db.delete_edge(edge_id).map_err(NodeGrafeoError::from)?)
    }

    /// Set a property on a node.
    #[napi(js_name = "setNodeProperty")]
    pub fn set_node_property(
        &self,
        env: Env,
        id: f64,
        key: String,
        value: Unknown<'_>,
    ) -> Result<()> {
        let node_id = validate_node_id(id)?;
        let db = self.inner.read();
        let val = types::js_to_value(&env, value)?;
        db.set_node_property(node_id, &key, val)
            .map_err(NodeGrafeoError::from)?;
        Ok(())
    }

    /// Set a property on an edge.
    #[napi(js_name = "setEdgeProperty")]
    pub fn set_edge_property(
        &self,
        env: Env,
        id: f64,
        key: String,
        value: Unknown<'_>,
    ) -> Result<()> {
        let edge_id = validate_edge_id(id)?;
        let db = self.inner.read();
        let val = types::js_to_value(&env, value)?;
        db.set_edge_property(edge_id, &key, val)
            .map_err(NodeGrafeoError::from)?;
        Ok(())
    }

    /// Get the number of nodes.
    #[napi(js_name = "nodeCount")]
    pub fn node_count(&self) -> u32 {
        // reason: WASM targets use 32-bit usize; on 64-bit, graphs with >4B nodes are unrealistic
        #[allow(clippy::cast_possible_truncation)]
        let count = self.inner.read().node_count() as u32;
        count
    }

    /// Get the number of edges.
    #[napi(js_name = "edgeCount")]
    pub fn edge_count(&self) -> u32 {
        // reason: WASM targets use 32-bit usize; on 64-bit, graphs with >4B edges are unrealistic
        #[allow(clippy::cast_possible_truncation)]
        let count = self.inner.read().edge_count() as u32;
        count
    }

    /// Begin a transaction with an optional isolation level.
    ///
    /// Isolation levels: "read_committed", "snapshot" (default), "serializable".
    #[napi(js_name = "beginTransaction")]
    pub fn begin_transaction(&self, isolation_level: Option<String>) -> Result<Transaction> {
        Transaction::new(self.inner.clone(), isolation_level.as_deref())
    }

    /// Create a vector similarity index on a node property.
    #[napi(js_name = "createVectorIndex")]
    #[allow(clippy::too_many_arguments)]
    pub async fn create_vector_index(
        &self,
        label: String,
        property: String,
        dimensions: Option<u32>,
        metric: Option<String>,
        m: Option<u32>,
        ef_construction: Option<u32>,
        quantization: Option<String>,
    ) -> Result<()> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            db.create_vector_index(
                &label,
                &property,
                dimensions.map(|d| d as usize),
                metric.as_deref(),
                m.map(|v| v as usize),
                ef_construction.map(|v| v as usize),
                quantization.as_deref(),
            )
            .map_err(NodeGrafeoError::from)
            .map_err(napi::Error::from)
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Search for the k nearest neighbors of a query vector.
    ///
    /// Returns an array of [nodeId, distance] pairs sorted by distance
    /// ascending (lower = more similar). The distance scale depends on
    /// the metric configured at index creation.
    #[napi(js_name = "vectorSearch")]
    // reason: f64->f32 is intentional: HNSW index uses f32 vectors
    #[allow(clippy::cast_possible_truncation)]
    pub async fn vector_search(
        &self,
        label: String,
        property: String,
        query: Vec<f64>,
        k: u32,
        ef: Option<u32>,
        filters: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<Vec<f64>>> {
        let filter_map = convert_json_filters(filters)?;
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let query_f32: Vec<f32> = query.iter().map(|&v| v as f32).collect();
            let results = db
                .vector_search(
                    &label,
                    &property,
                    &query_f32,
                    k as usize,
                    ef.map(|v| v as usize),
                    filter_map.as_ref(),
                )
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            // Return as [[nodeId, distance], ...] since napi doesn't have tuples
            Ok(results
                .into_iter()
                .map(|(id, dist)| vec![id.as_u64() as f64, dist as f64])
                .collect::<Vec<Vec<f64>>>())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Bulk-insert nodes with vector properties.
    #[napi(js_name = "batchCreateNodes")]
    // reason: f64->f32 is intentional: HNSW index uses f32 vectors
    #[allow(clippy::cast_possible_truncation)]
    pub async fn batch_create_nodes(
        &self,
        label: String,
        property: String,
        vectors: Vec<Vec<f64>>,
    ) -> Result<Vec<f64>> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let vecs_f32: Vec<Vec<f32>> = vectors
                .into_iter()
                .map(|v| v.into_iter().map(|x| x as f32).collect())
                .collect();
            let ids = db
                .batch_create_nodes(&label, &property, vecs_f32)
                .map_err(NodeGrafeoError::from)?;
            Ok(ids
                .into_iter()
                .map(|id| id.as_u64() as f64)
                .collect::<Vec<f64>>())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Remove a property from a node. Returns true if the property existed.
    #[napi(js_name = "removeNodeProperty")]
    pub fn remove_node_property(&self, id: f64, key: String) -> Result<bool> {
        let node_id = validate_node_id(id)?;
        let db = self.inner.read();
        Ok(db
            .remove_node_property(node_id, &key)
            .map_err(NodeGrafeoError::from)?)
    }

    /// Remove a property from an edge. Returns true if the property existed.
    #[napi(js_name = "removeEdgeProperty")]
    pub fn remove_edge_property(&self, id: f64, key: String) -> Result<bool> {
        let edge_id = validate_edge_id(id)?;
        let db = self.inner.read();
        Ok(db
            .remove_edge_property(edge_id, &key)
            .map_err(NodeGrafeoError::from)?)
    }

    /// Add a label to an existing node. Returns true if the label was added.
    #[napi(js_name = "addNodeLabel")]
    pub fn add_node_label(&self, id: f64, label: String) -> Result<bool> {
        let node_id = validate_node_id(id)?;
        let db = self.inner.read();
        Ok(db
            .add_node_label(node_id, &label)
            .map_err(NodeGrafeoError::from)?)
    }

    /// Remove a label from a node. Returns true if the label was removed.
    #[napi(js_name = "removeNodeLabel")]
    pub fn remove_node_label(&self, id: f64, label: String) -> Result<bool> {
        let node_id = validate_node_id(id)?;
        let db = self.inner.read();
        Ok(db
            .remove_node_label(node_id, &label)
            .map_err(NodeGrafeoError::from)?)
    }

    /// Get all labels for a node. Returns null if the node doesn't exist.
    #[napi(js_name = "getNodeLabels")]
    pub fn get_node_labels(&self, id: f64) -> Result<Option<Vec<String>>> {
        let node_id = validate_node_id(id)?;
        let db = self.inner.read();
        Ok(db.get_node_labels(node_id))
    }

    /// Returns high-level database information as a JSON object.
    #[napi]
    pub fn info(&self) -> Result<serde_json::Value> {
        let db = self.inner.read();
        let info = db.info();
        serde_json::to_value(&info).map_err(|e| NodeGrafeoError::Database(e.to_string()).into())
    }

    /// Returns schema information (labels, edge types, property keys) as a JSON object.
    #[napi]
    pub fn schema(&self) -> Result<serde_json::Value> {
        let db = self.inner.read();
        let schema = db.schema();
        serde_json::to_value(&schema).map_err(|e| NodeGrafeoError::Database(e.to_string()).into())
    }

    /// Returns the Grafeo engine version string.
    #[napi]
    pub fn version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    /// Clear all cached query plans.
    ///
    /// Forces re-parsing and re-optimization on next execution.
    /// Called automatically after DDL operations, but can be invoked manually.
    #[napi(js_name = "clearPlanCache")]
    pub fn clear_plan_cache(&self) {
        self.inner.read().clear_plan_cache();
    }

    /// Forces a WAL checkpoint.
    ///
    /// Flushes all pending WAL records to the main storage.
    #[napi(js_name = "walCheckpoint")]
    pub fn wal_checkpoint(&self) -> Result<()> {
        let db = self.inner.read();
        db.wal_checkpoint()
            .map_err(NodeGrafeoError::from)
            .map_err(napi::Error::from)
    }

    /// Saves the database to a file path.
    ///
    /// If in-memory, creates a new persistent database at the given path.
    /// If file-backed, creates a copy at the new path.
    /// The original database remains unchanged.
    #[napi]
    pub fn save(&self, path: String) -> Result<()> {
        let db = self.inner.read();
        db.save(path)
            .map_err(NodeGrafeoError::from)
            .map_err(napi::Error::from)
    }

    /// Create a full backup of the database.
    #[napi]
    pub fn backup_full(&self, backup_dir: String) -> Result<()> {
        let db = self.inner.read();
        db.backup_full(std::path::Path::new(&backup_dir))
            .map(|_| ())
            .map_err(NodeGrafeoError::from)
            .map_err(napi::Error::from)
    }

    /// Create an incremental backup (WAL records since last backup).
    #[napi]
    pub fn backup_incremental(&self, backup_dir: String) -> Result<()> {
        let db = self.inner.read();
        db.backup_incremental(std::path::Path::new(&backup_dir))
            .map(|_| ())
            .map_err(NodeGrafeoError::from)
            .map_err(napi::Error::from)
    }

    /// Restore a database to a specific epoch from a backup chain.
    #[napi]
    pub fn restore_to_epoch(backup_dir: String, epoch: f64, output_path: String) -> Result<()> {
        let epoch_id = validate_epoch(epoch)?;
        grafeo_engine::GrafeoDB::restore_to_epoch(
            std::path::Path::new(&backup_dir),
            epoch_id,
            std::path::Path::new(&output_path),
        )
        .map_err(NodeGrafeoError::from)
        .map_err(napi::Error::from)
    }

    /// Close the database.
    #[napi]
    pub fn close(&self) -> Result<()> {
        self.inner
            .read()
            .close()
            .map_err(NodeGrafeoError::from)
            .map_err(napi::Error::from)
    }

    // ── Schema context ───────────────────────────────────────────────────

    /// Sets the current schema for subsequent `execute()` calls.
    ///
    /// Equivalent to running `SESSION SET SCHEMA <name>` but persists across
    /// calls. Use `resetSchema()` to clear it.
    #[napi(js_name = "setSchema")]
    pub fn set_schema(&self, name: String) -> napi::Result<()> {
        self.inner
            .read()
            .set_current_schema(Some(&name))
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Clears the current schema context.
    ///
    /// Subsequent `execute()` calls will use the default (no-schema) namespace.
    #[napi(js_name = "resetSchema")]
    pub fn reset_schema(&self) {
        let _ = self.inner.read().set_current_schema(None);
    }

    /// Returns the current schema name, or `null` if no schema is set.
    #[napi(js_name = "currentSchema")]
    pub fn current_schema(&self) -> Option<String> {
        self.inner.read().current_schema()
    }

    // ── Graph projections ───────────────────────────────────────────────

    /// Creates a named graph projection. Returns `true` if created, `false`
    /// if a projection with that name already exists.
    ///
    /// A projection is a read-only, filtered view of the default graph.
    /// Only nodes with matching labels and edges with matching types are visible.
    #[napi(js_name = "createProjection")]
    pub fn create_projection(
        &self,
        name: String,
        node_labels: Option<Vec<String>>,
        edge_types: Option<Vec<String>>,
    ) -> bool {
        use grafeo_core::graph::ProjectionSpec;

        let mut spec = ProjectionSpec::new();
        if let Some(labels) = node_labels.filter(|l| !l.is_empty()) {
            spec = spec.with_node_labels(labels);
        }
        if let Some(types) = edge_types.filter(|t| !t.is_empty()) {
            spec = spec.with_edge_types(types);
        }
        self.inner.read().create_projection(name, spec)
    }

    /// Drops a named graph projection. Returns `true` if it existed.
    #[napi(js_name = "dropProjection")]
    pub fn drop_projection(&self, name: String) -> bool {
        self.inner.read().drop_projection(&name)
    }

    /// Returns the names of all graph projections.
    #[napi(js_name = "listProjections")]
    pub fn list_projections(&self) -> Vec<String> {
        self.inner.read().list_projections()
    }
}

// Vector-index methods live in a separate impl block so the entire block can
// be conditionally compiled.  napi-rs generates callback registrations for
// every method inside a `#[napi]` impl, so a per-method `#[cfg]` doesn't work.
#[cfg(feature = "vector-index")]
#[napi]
impl JsGrafeoDB {
    /// Drop a vector index for the given label and property.
    /// Returns true if the index existed and was removed.
    #[napi(js_name = "dropVectorIndex")]
    pub async fn drop_vector_index(&self, label: String, property: String) -> Result<bool> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            Ok(db.drop_vector_index(&label, &property))
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Rebuild a vector index by rescanning all matching nodes.
    /// Preserves the original index configuration.
    ///
    /// Note: Vector indexes auto-sync when you call setNodeProperty(),
    /// batchCreateNodes(), or batchCreateNodesWithProps() with vector data.
    /// You only need this after non-standard data imports or to compact
    /// the index after many deletions.
    #[napi(js_name = "rebuildVectorIndex")]
    pub async fn rebuild_vector_index(&self, label: String, property: String) -> Result<()> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            db.rebuild_vector_index(&label, &property)
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Batch search for nearest neighbors of multiple query vectors.
    // reason: f64->f32 is intentional: HNSW index uses f32 vectors
    #[allow(clippy::cast_possible_truncation)]
    #[napi(js_name = "batchVectorSearch")]
    pub async fn batch_vector_search(
        &self,
        label: String,
        property: String,
        queries: Vec<Vec<f64>>,
        k: u32,
        ef: Option<u32>,
        filters: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<Vec<Vec<f64>>>> {
        let filter_map = convert_json_filters(filters)?;
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let queries_f32: Vec<Vec<f32>> = queries
                .into_iter()
                .map(|v| v.into_iter().map(|x| x as f32).collect())
                .collect();
            let results = db
                .batch_vector_search(
                    &label,
                    &property,
                    &queries_f32,
                    k as usize,
                    ef.map(|v| v as usize),
                    filter_map.as_ref(),
                )
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            Ok(results
                .into_iter()
                .map(|inner| {
                    inner
                        .into_iter()
                        .map(|(id, dist)| vec![id.as_u64() as f64, dist as f64])
                        .collect::<Vec<Vec<f64>>>()
                })
                .collect::<Vec<Vec<Vec<f64>>>>())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Search for diverse nearest neighbors using Maximal Marginal Relevance (MMR).
    ///
    /// Returns an array of [nodeId, distance] pairs in MMR selection order.
    /// The distance values match vectorSearch() for the same nodes
    /// (lower = more similar). The ordering reflects relevance-diversity
    /// balance, not distance sorting.
    #[napi(js_name = "mmrSearch")]
    // reason: f64->f32 is intentional: HNSW index uses f32 vectors
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::too_many_arguments)]
    pub async fn mmr_search(
        &self,
        label: String,
        property: String,
        query: Vec<f64>,
        k: u32,
        fetch_k: Option<u32>,
        lambda_mult: Option<f64>,
        ef: Option<u32>,
        filters: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<Vec<f64>>> {
        let filter_map = convert_json_filters(filters)?;
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let query_f32: Vec<f32> = query.iter().map(|&v| v as f32).collect();
            let results = db
                .mmr_search(
                    &label,
                    &property,
                    &query_f32,
                    k as usize,
                    fetch_k.map(|v| v as usize),
                    lambda_mult.map(|v| v as f32),
                    ef.map(|v| v as usize),
                    filter_map.as_ref(),
                )
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            Ok(results
                .into_iter()
                .map(|(id, dist)| vec![id.as_u64() as f64, dist as f64])
                .collect::<Vec<Vec<f64>>>())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }
}

// Text-index methods live in a separate impl block for the same reason.
#[cfg(feature = "text-index")]
#[napi]
impl JsGrafeoDB {
    /// Create a BM25 text index on a node property for full-text search.
    ///
    /// The index is automatically kept in sync as nodes are created,
    /// updated, or deleted. You do not need to call rebuildTextIndex()
    /// after normal write operations.
    #[napi(js_name = "createTextIndex")]
    pub async fn create_text_index(&self, label: String, property: String) -> Result<()> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            db.create_text_index(&label, &property)
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Drop a text index for the given label and property.
    #[napi(js_name = "dropTextIndex")]
    pub async fn drop_text_index(&self, label: String, property: String) -> Result<bool> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            Ok(db.drop_text_index(&label, &property))
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Rebuild a text index by rescanning all matching nodes.
    ///
    /// Note: Text indexes auto-sync on normal writes. You only need this
    /// after importing data through non-standard paths.
    #[napi(js_name = "rebuildTextIndex")]
    pub async fn rebuild_text_index(&self, label: String, property: String) -> Result<()> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            db.rebuild_text_index(&label, &property)
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Search a text index using BM25 scoring.
    ///
    /// Returns an array of [nodeId, score] pairs sorted by descending
    /// relevance (higher score = more relevant). BM25 scores are
    /// unbounded positive floats.
    #[napi(js_name = "textSearch")]
    pub async fn text_search(
        &self,
        label: String,
        property: String,
        query: String,
        k: u32,
    ) -> Result<Vec<Vec<f64>>> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let results = db
                .text_search(&label, &property, &query, k as usize)
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            Ok(results
                .into_iter()
                .map(|(id, score)| vec![id.as_u64() as f64, score])
                .collect::<Vec<Vec<f64>>>())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }
}

// Hybrid-search methods live in a separate impl block for the same reason.
#[cfg(feature = "hybrid-search")]
#[napi]
impl JsGrafeoDB {
    /// Perform hybrid search combining text (BM25) and vector similarity.
    ///
    /// Requires both a text index (createTextIndex) and a vector index
    /// (createVectorIndex). If either is missing, that source is silently
    /// omitted from fusion.
    ///
    /// Returns an array of [nodeId, score] pairs sorted by fused score
    /// descending (higher = more relevant). These are fusion scores,
    /// NOT distances.
    #[napi(js_name = "hybridSearch")]
    #[allow(clippy::too_many_arguments)]
    // reason: f64->f32 is intentional: HNSW index uses f32 vectors
    #[allow(clippy::cast_possible_truncation)]
    pub async fn hybrid_search(
        &self,
        label: String,
        text_property: String,
        vector_property: String,
        query_text: String,
        k: u32,
        query_vector: Option<Vec<f64>>,
        fusion: Option<String>,
        weights: Option<Vec<f64>>,
    ) -> Result<Vec<Vec<f64>>> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let fusion_method = match fusion.as_deref() {
                Some("weighted") => {
                    let w = weights.unwrap_or_else(|| vec![0.5, 0.5]);
                    Some(grafeo_core::index::text::FusionMethod::Weighted { weights: w })
                }
                _ => None,
            };

            let query_vec_f32: Option<Vec<f32>> =
                query_vector.map(|v| v.iter().map(|&x| x as f32).collect());

            let db = db.read();
            let results = db
                .hybrid_search(
                    &label,
                    &text_property,
                    &vector_property,
                    &query_text,
                    query_vec_f32.as_deref(),
                    k as usize,
                    fusion_method,
                )
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            Ok(results
                .into_iter()
                .map(|(id, score)| vec![id.as_u64() as f64, score])
                .collect::<Vec<Vec<f64>>>())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }
}

// Compact-store methods live in a separate impl block for the same reason.
#[cfg(feature = "compact-store")]
#[napi]
impl JsGrafeoDB {
    /// Converts the database to a read-only CompactStore for faster queries.
    ///
    /// Takes a snapshot of all nodes and edges, builds a columnar store with
    /// CSR adjacency, and switches to read-only mode. After this call, write
    /// operations will fail.
    #[napi]
    pub fn compact(&self) -> Result<()> {
        let mut db = self.inner.write();
        db.compact()
            .map_err(NodeGrafeoError::from)
            .map_err(napi::Error::from)
    }
}

// CDC methods live in a separate impl block for the same reason.
#[cfg(feature = "cdc")]
#[napi]
impl JsGrafeoDB {
    /// Enable CDC for all future sessions.
    #[napi(js_name = "enableCdc")]
    pub fn enable_cdc(&self) {
        self.inner.read().set_cdc_enabled(true);
    }

    /// Disable CDC for all future sessions.
    #[napi(js_name = "disableCdc")]
    pub fn disable_cdc(&self) {
        self.inner.read().set_cdc_enabled(false);
    }

    /// Returns whether CDC is currently enabled for new sessions.
    #[napi(js_name = "isCdcEnabled", getter)]
    pub fn is_cdc_enabled(&self) -> bool {
        self.inner.read().is_cdc_enabled()
    }

    /// Returns the full change history for a node.
    #[napi(js_name = "nodeHistory")]
    pub async fn node_history(&self, node_id: f64) -> Result<Vec<serde_json::Value>> {
        let id = validate_node_id(node_id)?;
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let events = db
                .history(id)
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            Ok(events.iter().map(change_event_to_json).collect())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Returns the full change history for an edge.
    #[napi(js_name = "edgeHistory")]
    pub async fn edge_history(&self, edge_id: f64) -> Result<Vec<serde_json::Value>> {
        let id = validate_edge_id(edge_id)?;
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let events = db
                .history(id)
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            Ok(events.iter().map(change_event_to_json).collect())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Returns change events for a node since a given epoch.
    #[napi(js_name = "nodeHistorySince")]
    pub async fn node_history_since(
        &self,
        node_id: f64,
        since_epoch: f64,
    ) -> Result<Vec<serde_json::Value>> {
        let id = validate_node_id(node_id)?;
        let epoch = validate_epoch(since_epoch)?;
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let events = db
                .history_since(id, epoch)
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            Ok(events.iter().map(change_event_to_json).collect())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Returns all change events across entities in an epoch range.
    #[napi(js_name = "changesBetween")]
    pub async fn changes_between(
        &self,
        start_epoch: f64,
        end_epoch: f64,
    ) -> Result<Vec<serde_json::Value>> {
        let start = validate_epoch(start_epoch)?;
        let end = validate_epoch(end_epoch)?;
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let events = db
                .changes_between(start, end)
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            Ok(events.iter().map(change_event_to_json).collect())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }
}

// Embed methods live in a separate impl block so the entire block can be
// conditionally compiled.  napi-rs generates callback registrations for every
// method inside a `#[napi]` impl, so a per-method `#[cfg]` doesn't work.
#[cfg(feature = "embed")]
#[napi]
impl JsGrafeoDB {
    /// Register an ONNX embedding model for text-to-vector conversion.
    ///
    /// Once registered, use embedText() and vectorSearchText() with the model name.
    #[napi(js_name = "registerEmbeddingModel")]
    pub async fn register_embedding_model(
        &self,
        name: String,
        model_path: String,
        tokenizer_path: String,
        batch_size: Option<u32>,
    ) -> Result<()> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut model = grafeo_engine::embedding::OnnxEmbeddingModel::from_files(
                &name,
                &model_path,
                &tokenizer_path,
            )
            .map_err(NodeGrafeoError::from)
            .map_err(napi::Error::from)?;
            if let Some(bs) = batch_size {
                model = model.with_batch_size(bs as usize);
            }
            let db = db.read();
            db.register_embedding_model(&name, std::sync::Arc::new(model));
            Ok(())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Generate embeddings for a list of texts using a registered model.
    ///
    /// Returns an array of float arrays, one per input text.
    #[napi(js_name = "embedText")]
    pub async fn embed_text(
        &self,
        model_name: String,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f64>>> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            let results = db
                .embed_text(&model_name, &text_refs)
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            // Convert f32 → f64 for JavaScript number compatibility
            Ok(results
                .into_iter()
                .map(|v| v.into_iter().map(f64::from).collect())
                .collect())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }
}

// This method requires both embed and vector-index, so it gets its own block.
#[cfg(all(feature = "embed", feature = "vector-index"))]
#[napi]
impl JsGrafeoDB {
    /// Search a vector index using a text query, generating the embedding on-the-fly.
    ///
    /// Returns an array of [nodeId, distance] pairs.
    #[napi(js_name = "vectorSearchText")]
    pub async fn vector_search_text(
        &self,
        label: String,
        property: String,
        model_name: String,
        query_text: String,
        k: u32,
        ef: Option<u32>,
    ) -> Result<Vec<Vec<f64>>> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let results = db
                .vector_search_text(
                    &label,
                    &property,
                    &model_name,
                    &query_text,
                    k as usize,
                    ef.map(|e| e as usize),
                )
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            Ok(results
                .into_iter()
                .map(|(id, dist)| vec![id.as_u64() as f64, f64::from(dist)])
                .collect::<Vec<Vec<f64>>>())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }
}

// Language-specific execute methods live in separate impl blocks so the
// `#[napi]` macro only generates C callback symbols when the feature is active.

#[cfg(feature = "cypher")]
#[napi]
impl JsGrafeoDB {
    /// Execute a Cypher query.
    #[napi(js_name = "executeCypher")]
    pub async fn execute_cypher(
        &self,
        query: String,
        params: Option<serde_json::Value>,
    ) -> Result<QueryResult> {
        self.execute_language_impl("cypher", query, params).await
    }
}

#[cfg(feature = "sql-pgq")]
#[napi]
impl JsGrafeoDB {
    /// Execute a SQL/PGQ query (SQL:2023 GRAPH_TABLE).
    #[napi(js_name = "executeSql")]
    pub async fn execute_sql(
        &self,
        query: String,
        params: Option<serde_json::Value>,
    ) -> Result<QueryResult> {
        self.execute_language_impl("sql", query, params).await
    }
}

#[cfg(feature = "gremlin")]
#[napi]
impl JsGrafeoDB {
    /// Execute a Gremlin query.
    #[napi(js_name = "executeGremlin")]
    pub async fn execute_gremlin(
        &self,
        query: String,
        params: Option<serde_json::Value>,
    ) -> Result<QueryResult> {
        self.execute_language_impl("gremlin", query, params).await
    }
}

#[cfg(feature = "graphql")]
#[napi]
impl JsGrafeoDB {
    /// Execute a GraphQL query.
    #[napi(js_name = "executeGraphql")]
    pub async fn execute_graphql(
        &self,
        query: String,
        params: Option<serde_json::Value>,
    ) -> Result<QueryResult> {
        self.execute_language_impl("graphql", query, params).await
    }
}

#[cfg(feature = "sparql")]
#[napi]
impl JsGrafeoDB {
    /// Execute a SPARQL query against the RDF triple store.
    #[napi(js_name = "executeSparql")]
    pub async fn execute_sparql(
        &self,
        query: String,
        params: Option<serde_json::Value>,
    ) -> Result<QueryResult> {
        self.execute_language_impl("sparql", query, params).await
    }

    /// Execute a query in a named language (e.g. `"graphql-rdf"`).
    #[napi(js_name = "executeLanguage")]
    pub async fn execute_language(
        &self,
        language: String,
        query: String,
        params: Option<serde_json::Value>,
    ) -> Result<QueryResult> {
        let db = self.inner.clone();
        let mut result = tokio::task::spawn_blocking(move || {
            let db = db.read();
            execute_language_query(&db, &query, &language, params.as_ref())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))??;

        let db = self.inner.read();
        let (nodes, edges) = extract_entities(&result, &db);
        let columns = std::mem::take(&mut result.columns);
        let exec_time = result.execution_time_ms;
        let scanned = result.rows_scanned;

        Ok(QueryResult::with_metrics(
            columns,
            result.into_rows(),
            nodes,
            edges,
            exec_time,
            scanned,
        ))
    }
}

// -- File import methods --------------------------------------------------

#[napi]
impl JsGrafeoDB {
    /// Import a CSV file as graph nodes.
    ///
    /// Each row becomes a node with the given label. Column headers are used
    /// as property names. Returns the number of nodes created.
    #[napi(js_name = "importCsv")]
    pub async fn import_csv(&self, path: String, options: Option<CsvImportOptions>) -> Result<i64> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let opts = options.unwrap_or_default();
            let label = sanitize_gql_identifier(&opts.label.unwrap_or_else(|| "Row".to_string()));
            let headers = opts.headers.unwrap_or(true);

            let abs_path = std::path::Path::new(&path)
                .canonicalize()
                .map_err(|e| NodeGrafeoError::Database(format!("{path}: {e}")))?;
            let path_str = escape_gql_string(&abs_path.to_string_lossy().replace('\\', "/"));

            let header_clause = if headers { " WITH HEADERS" } else { "" };

            let insert_clause = if headers {
                let columns = read_csv_headers(&abs_path, ',')
                    .map_err(|e| NodeGrafeoError::Database(e.clone()))?;
                if columns.is_empty() {
                    format!("INSERT (:{label} {{}})")
                } else {
                    let props = columns
                        .iter()
                        .map(|col| {
                            let safe = sanitize_gql_identifier(col);
                            format!("{safe}: row.{safe}")
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("INSERT (:{label} {{{props}}})")
                }
            } else {
                format!("INSERT (:{label} {{}})")
            };

            let query = format!(
                "LOAD DATA FROM '{path_str}' FORMAT CSV{header_clause} AS row {insert_clause}"
            );

            let db = db.read();
            let session = db.session();

            let before_count = count_nodes(&session, &label);

            session.execute(&query).map_err(NodeGrafeoError::from)?;

            let count = count_nodes(&session, &label) - before_count;

            Ok(count)
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Import a JSON Lines file as graph nodes.
    ///
    /// Each line must be a valid JSON object. Object keys become property names.
    /// Returns the number of nodes created.
    #[napi(js_name = "importJsonl")]
    pub async fn import_jsonl(
        &self,
        path: String,
        options: Option<JsonlImportOptions>,
    ) -> Result<i64> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let opts = options.unwrap_or_default();
            let label = sanitize_gql_identifier(&opts.label.unwrap_or_else(|| "Row".to_string()));

            let abs_path = std::path::Path::new(&path)
                .canonicalize()
                .map_err(|e| NodeGrafeoError::Database(format!("{path}: {e}")))?;
            let path_str = escape_gql_string(&abs_path.to_string_lossy().replace('\\', "/"));

            let keys =
                read_jsonl_keys(&abs_path).map_err(|e| NodeGrafeoError::Database(e.clone()))?;

            let insert_clause = if keys.is_empty() {
                format!("INSERT (:{label} {{}})")
            } else {
                let props = keys
                    .iter()
                    .map(|key| {
                        let safe = sanitize_gql_identifier(key);
                        format!("{safe}: row.{safe}")
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("INSERT (:{label} {{{props}}})")
            };

            let query = format!("LOAD DATA FROM '{path_str}' FORMAT JSONL AS row {insert_clause}");

            let db = db.read();
            let session = db.session();

            let before_count = count_nodes(&session, &label);

            session.execute(&query).map_err(NodeGrafeoError::from)?;

            let count = count_nodes(&session, &label) - before_count;

            Ok(count)
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }
}

/// Options for CSV import.
#[derive(Default)]
#[napi(object)]
pub struct CsvImportOptions {
    /// Label to assign to created nodes (default: "Row")
    pub label: Option<String>,
    /// Whether the first row contains headers (default: true)
    pub headers: Option<bool>,
}

/// Options for JSON Lines import.
#[derive(Default)]
#[napi(object)]
pub struct JsonlImportOptions {
    /// Label to assign to created nodes (default: "Row")
    pub label: Option<String>,
}

/// Read CSV headers from the first line of a file.
fn read_csv_headers(
    path: &std::path::Path,
    delimiter: char,
) -> std::result::Result<Vec<String>, String> {
    use std::io::{BufRead, BufReader};
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut reader = BufReader::new(f);
    let mut header_line = String::new();
    reader
        .read_line(&mut header_line)
        .map_err(|e| format!("Failed to read headers: {e}"))?;
    Ok(header_line
        .trim()
        .split(delimiter)
        .map(|h| h.trim().trim_matches('"').to_string())
        .filter(|h| !h.is_empty())
        .collect())
}

/// Escape single quotes in a string for embedding in a GQL string literal.
fn escape_gql_string(s: &str) -> String {
    s.replace('\'', "\\'")
}

/// Sanitize a name for use as a GQL identifier.
fn sanitize_gql_identifier(name: &str) -> String {
    let sanitized: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "_col".to_string()
    } else if sanitized.starts_with(|c: char| c.is_ascii_digit()) {
        format!("_{sanitized}")
    } else {
        sanitized
    }
}

/// Count nodes with a given label.
fn count_nodes(session: &grafeo_engine::Session, label: &str) -> i64 {
    session
        .execute(&format!("MATCH (n:{label}) RETURN count(n) AS c"))
        .ok()
        .and_then(|r| r.rows().first().cloned())
        .and_then(|row| row.first().cloned())
        .and_then(|v| match v {
            Value::Int64(n) => Some(n),
            _ => None,
        })
        .unwrap_or(0)
}

/// Read JSON keys from the first non-empty line of a JSONL file.
fn read_jsonl_keys(path: &std::path::Path) -> std::result::Result<Vec<String>, String> {
    use std::io::{BufRead, BufReader};
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let reader = BufReader::new(f);
    for line in reader.lines() {
        let line = line.map_err(|e| format!("Failed to read JSONL file: {e}"))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(obj) = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(trimmed)
        {
            return Ok(obj.keys().cloned().collect());
        }
        break;
    }
    Ok(Vec::new())
}

/// Execute a query in a given language with optional JSON params.
fn execute_language_query(
    db: &GrafeoDB,
    query: &str,
    language: &str,
    params: Option<&serde_json::Value>,
) -> std::result::Result<EngineQueryResult, napi::Error> {
    let param_map = convert_json_params(params)?;
    db.execute_language(query, language, param_map)
        .map_err(NodeGrafeoError::from)
        .map_err(napi::Error::from)
}

/// Convert JSON params to a HashMap<String, Value>.
fn convert_json_params(
    params: Option<&serde_json::Value>,
) -> std::result::Result<Option<HashMap<String, Value>>, napi::Error> {
    grafeo_bindings_common::json::json_params_to_map(params)
        .map_err(|msg| NodeGrafeoError::InvalidArgument(msg).into())
}

/// Convert a serde_json::Value to a Grafeo Value.
pub(crate) fn json_to_value(v: &serde_json::Value) -> std::result::Result<Value, napi::Error> {
    Ok(grafeo_bindings_common::json::json_to_value(v))
}

/// Fetch a node from the database and wrap it as JsNode.
fn fetch_node(db: &GrafeoDB, id: NodeId) -> Result<JsNode> {
    db.get_node(id)
        .map(|node| {
            let labels: Vec<String> = node.labels.iter().map(|s| s.to_string()).collect();
            let properties = node.properties.into_iter().collect();
            JsNode::new(id, labels, properties)
        })
        .ok_or_else(|| NodeGrafeoError::Database("Failed to fetch created node".into()).into())
}

/// Fetch an edge from the database and wrap it as JsEdge.
fn fetch_edge(db: &GrafeoDB, id: EdgeId) -> Result<JsEdge> {
    db.get_edge(id)
        .map(|edge| {
            let properties = edge.properties.into_iter().collect();
            JsEdge::new(
                id,
                edge.edge_type.to_string(),
                edge.src,
                edge.dst,
                properties,
            )
        })
        .ok_or_else(|| NodeGrafeoError::Database("Failed to fetch created edge".into()).into())
}

/// Extract nodes and edges from query results based on column types.
pub(crate) fn extract_entities(
    result: &EngineQueryResult,
    _db: &GrafeoDB,
) -> (Vec<JsNode>, Vec<JsEdge>) {
    grafeo_bindings_common::entity::extract_and_map(
        result,
        |n| JsNode::new(n.id, n.labels, n.properties),
        |e| JsEdge::new(e.id, e.edge_type, e.source_id, e.target_id, e.properties),
    )
}

/// Convert a Grafeo Value to serde_json::Value.
#[cfg(feature = "cdc")]
fn grafeo_value_to_json(v: &Value) -> serde_json::Value {
    grafeo_bindings_common::json::value_to_json(v)
}

/// Convert a CDC ChangeEvent to a JSON object.
#[cfg(feature = "cdc")]
fn change_event_to_json(event: &grafeo_engine::cdc::ChangeEvent) -> serde_json::Value {
    let entity_type = if event.entity_id.is_node() {
        "node"
    } else {
        "edge"
    };
    let kind = match event.kind {
        grafeo_engine::cdc::ChangeKind::Create => "create",
        grafeo_engine::cdc::ChangeKind::Update => "update",
        grafeo_engine::cdc::ChangeKind::Delete => "delete",
        _ => "unknown",
    };

    let before = match &event.before {
        Some(props) => {
            let obj: serde_json::Map<String, serde_json::Value> = props
                .iter()
                .map(|(k, v)| (k.clone(), grafeo_value_to_json(v)))
                .collect();
            serde_json::Value::Object(obj)
        }
        None => serde_json::Value::Null,
    };

    let after = match &event.after {
        Some(props) => {
            let obj: serde_json::Map<String, serde_json::Value> = props
                .iter()
                .map(|(k, v)| (k.clone(), grafeo_value_to_json(v)))
                .collect();
            serde_json::Value::Object(obj)
        }
        None => serde_json::Value::Null,
    };

    serde_json::json!({
        "entity_id": event.entity_id.as_u64(),
        "entity_type": entity_type,
        "kind": kind,
        "epoch": event.epoch.0,
        "timestamp": event.timestamp,
        "before": before,
        "after": after,
        "labels": event.labels,
        "before_labels": event.before_labels,
        "edge_type": event.edge_type,
        "src_id": event.src_id,
        "dst_id": event.dst_id,
    })
}
