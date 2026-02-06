//! Main entry point for using Grafeo from Node.js.
//!
//! [`JsGrafeoDB`] wraps the Rust database engine and gives you a JavaScript API.

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi::{JsObject, JsString, JsUnknown};
use napi_derive::napi;
use parking_lot::RwLock;

use grafeo_common::types::{EdgeId, LogicalType, NodeId, Value};
use grafeo_engine::config::Config;
use grafeo_engine::database::{GrafeoDB, QueryResult as EngineQueryResult};

use crate::error::NodeGrafeoError;
use crate::graph::{JsEdge, JsNode};
use crate::query::QueryResult;
use crate::transaction::Transaction;
use crate::types;

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

    /// Execute a GQL query. Returns a Promise<QueryResult>.
    #[napi]
    pub async fn execute(
        &self,
        query: String,
        params: Option<serde_json::Value>,
    ) -> Result<QueryResult> {
        let db = self.inner.clone();
        let result = tokio::task::spawn_blocking(move || {
            let db = db.read();
            execute_query(&db, &query, params.as_ref())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))??;

        let db = self.inner.read();
        let (nodes, edges) = extract_entities(&result, &db);

        Ok(QueryResult::with_metrics(
            result.columns,
            result.rows,
            nodes,
            edges,
            result.execution_time_ms,
            result.rows_scanned,
        ))
    }

    /// Execute a Cypher query.
    #[cfg(feature = "cypher")]
    #[napi(js_name = "executeCypher")]
    pub async fn execute_cypher(
        &self,
        query: String,
        params: Option<serde_json::Value>,
    ) -> Result<QueryResult> {
        let db = self.inner.clone();
        let result = tokio::task::spawn_blocking(move || -> std::result::Result<_, napi::Error> {
            let db = db.read();
            let param_map = convert_json_params(params.as_ref())?;
            if let Some(p) = param_map {
                db.execute_cypher_with_params(&query, p)
                    .map_err(NodeGrafeoError::from)
                    .map_err(napi::Error::from)
            } else {
                db.execute_cypher(&query)
                    .map_err(NodeGrafeoError::from)
                    .map_err(napi::Error::from)
            }
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))??;

        let db = self.inner.read();
        let (nodes, edges) = extract_entities(&result, &db);

        Ok(QueryResult::with_metrics(
            result.columns,
            result.rows,
            nodes,
            edges,
            result.execution_time_ms,
            result.rows_scanned,
        ))
    }

    /// Execute a Gremlin query.
    #[cfg(feature = "gremlin")]
    #[napi(js_name = "executeGremlin")]
    pub async fn execute_gremlin(
        &self,
        query: String,
        params: Option<serde_json::Value>,
    ) -> Result<QueryResult> {
        let db = self.inner.clone();
        let result = tokio::task::spawn_blocking(move || -> std::result::Result<_, napi::Error> {
            let db = db.read();
            let param_map = convert_json_params(params.as_ref())?;
            if let Some(p) = param_map {
                db.execute_gremlin_with_params(&query, p)
                    .map_err(NodeGrafeoError::from)
                    .map_err(napi::Error::from)
            } else {
                db.execute_gremlin(&query)
                    .map_err(NodeGrafeoError::from)
                    .map_err(napi::Error::from)
            }
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))??;

        let db = self.inner.read();
        let (nodes, edges) = extract_entities(&result, &db);

        Ok(QueryResult::with_metrics(
            result.columns,
            result.rows,
            nodes,
            edges,
            result.execution_time_ms,
            result.rows_scanned,
        ))
    }

    /// Execute a GraphQL query.
    #[cfg(feature = "graphql")]
    #[napi(js_name = "executeGraphql")]
    pub async fn execute_graphql(
        &self,
        query: String,
        params: Option<serde_json::Value>,
    ) -> Result<QueryResult> {
        let db = self.inner.clone();
        let result = tokio::task::spawn_blocking(move || -> std::result::Result<_, napi::Error> {
            let db = db.read();
            let param_map = convert_json_params(params.as_ref())?;
            if let Some(p) = param_map {
                db.execute_graphql_with_params(&query, p)
                    .map_err(NodeGrafeoError::from)
                    .map_err(napi::Error::from)
            } else {
                db.execute_graphql(&query)
                    .map_err(NodeGrafeoError::from)
                    .map_err(napi::Error::from)
            }
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))??;

        let db = self.inner.read();
        let (nodes, edges) = extract_entities(&result, &db);

        Ok(QueryResult::with_metrics(
            result.columns,
            result.rows,
            nodes,
            edges,
            result.execution_time_ms,
            result.rows_scanned,
        ))
    }

    /// Execute a SPARQL query against the RDF triple store.
    #[cfg(feature = "sparql")]
    #[napi(js_name = "executeSparql")]
    pub async fn execute_sparql(&self, query: String) -> Result<QueryResult> {
        let db = self.inner.clone();
        let result = tokio::task::spawn_blocking(move || {
            let db = db.read();
            db.execute_sparql(&query).map_err(NodeGrafeoError::from)
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))??;

        Ok(QueryResult::with_metrics(
            result.columns,
            result.rows,
            Vec::new(),
            Vec::new(),
            result.execution_time_ms,
            result.rows_scanned,
        ))
    }

    /// Create a node with labels and optional properties.
    #[napi(js_name = "createNode")]
    pub fn create_node(
        &self,
        env: Env,
        labels: Vec<String>,
        properties: Option<JsObject>,
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
                let value: JsUnknown = props_obj.get_named_property(&key_str)?;
                let val = types::js_to_value(&env, value)?;
                props.push((grafeo_common::types::PropertyKey::new(key_str), val));
            }
            db.create_node_with_props(&label_refs, props)
        } else {
            db.create_node(&label_refs)
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
        properties: Option<JsObject>,
    ) -> Result<JsEdge> {
        let db = self.inner.read();
        let src = NodeId(source_id as u64);
        let dst = NodeId(target_id as u64);

        let id = if let Some(props_obj) = properties {
            let mut props = Vec::new();
            let keys = props_obj.get_property_names()?;
            let len = keys.get_array_length()?;
            for i in 0..len {
                let key: JsString = keys.get_element(i)?;
                let key_str = key.into_utf8()?.into_owned()?;
                let value: JsUnknown = props_obj.get_named_property(&key_str)?;
                let val = types::js_to_value(&env, value)?;
                props.push((grafeo_common::types::PropertyKey::new(key_str), val));
            }
            db.create_edge_with_props(src, dst, &edge_type, props)
        } else {
            db.create_edge(src, dst, &edge_type)
        };

        fetch_edge(&db, id)
    }

    /// Get a node by ID.
    #[napi(js_name = "getNode")]
    pub fn get_node(&self, id: f64) -> Option<JsNode> {
        let db = self.inner.read();
        let node_id = NodeId(id as u64);
        db.get_node(node_id).map(|node| {
            let labels: Vec<String> = node.labels.iter().map(|s| s.to_string()).collect();
            let properties = node
                .properties
                .into_iter()
                .map(|(k, v)| (k.as_str().to_string(), v))
                .collect();
            JsNode::new(node_id, labels, properties)
        })
    }

    /// Get an edge by ID.
    #[napi(js_name = "getEdge")]
    pub fn get_edge(&self, id: f64) -> Option<JsEdge> {
        let db = self.inner.read();
        let edge_id = EdgeId(id as u64);
        db.get_edge(edge_id).map(|edge| {
            let properties = edge
                .properties
                .into_iter()
                .map(|(k, v)| (k.as_str().to_string(), v))
                .collect();
            JsEdge::new(
                edge_id,
                edge.edge_type.to_string(),
                edge.src,
                edge.dst,
                properties,
            )
        })
    }

    /// Delete a node by ID. Returns true if the node existed.
    #[napi(js_name = "deleteNode")]
    pub fn delete_node(&self, id: f64) -> bool {
        let db = self.inner.read();
        db.delete_node(NodeId(id as u64))
    }

    /// Delete an edge by ID. Returns true if the edge existed.
    #[napi(js_name = "deleteEdge")]
    pub fn delete_edge(&self, id: f64) -> bool {
        let db = self.inner.read();
        db.delete_edge(EdgeId(id as u64))
    }

    /// Set a property on a node.
    #[napi(js_name = "setNodeProperty")]
    pub fn set_node_property(
        &self,
        env: Env,
        id: f64,
        key: String,
        value: JsUnknown,
    ) -> Result<()> {
        let db = self.inner.read();
        let val = types::js_to_value(&env, value)?;
        db.set_node_property(NodeId(id as u64), &key, val);
        Ok(())
    }

    /// Set a property on an edge.
    #[napi(js_name = "setEdgeProperty")]
    pub fn set_edge_property(
        &self,
        env: Env,
        id: f64,
        key: String,
        value: JsUnknown,
    ) -> Result<()> {
        let db = self.inner.read();
        let val = types::js_to_value(&env, value)?;
        db.set_edge_property(EdgeId(id as u64), &key, val);
        Ok(())
    }

    /// Get the number of nodes.
    #[napi(getter, js_name = "nodeCount")]
    pub fn node_count(&self) -> u32 {
        self.inner.read().node_count() as u32
    }

    /// Get the number of edges.
    #[napi(getter, js_name = "edgeCount")]
    pub fn edge_count(&self) -> u32 {
        self.inner.read().edge_count() as u32
    }

    /// Begin a transaction.
    #[napi(js_name = "beginTransaction")]
    pub fn begin_transaction(&self) -> Result<Transaction> {
        Transaction::new(self.inner.clone())
    }

    /// Create a vector similarity index on a node property.
    #[napi(js_name = "createVectorIndex")]
    pub async fn create_vector_index(
        &self,
        label: String,
        property: String,
        dimensions: Option<u32>,
        metric: Option<String>,
        m: Option<u32>,
        ef_construction: Option<u32>,
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
            )
            .map_err(NodeGrafeoError::from)
            .map_err(napi::Error::from)
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Search for the k nearest neighbors of a query vector.
    #[napi(js_name = "vectorSearch")]
    pub async fn vector_search(
        &self,
        label: String,
        property: String,
        query: Vec<f64>,
        k: u32,
        ef: Option<u32>,
    ) -> Result<Vec<Vec<f64>>> {
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
            let ids = db.batch_create_nodes(&label, &property, vecs_f32);
            Ok(ids
                .into_iter()
                .map(|id| id.as_u64() as f64)
                .collect::<Vec<f64>>())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Batch search for nearest neighbors of multiple query vectors.
    #[cfg(feature = "vector-index")]
    #[napi(js_name = "batchVectorSearch")]
    pub async fn batch_vector_search(
        &self,
        label: String,
        property: String,
        queries: Vec<Vec<f64>>,
        k: u32,
        ef: Option<u32>,
    ) -> Result<Vec<Vec<Vec<f64>>>> {
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

    /// Close the database.
    #[napi]
    pub fn close(&self) -> Result<()> {
        self.inner
            .read()
            .close()
            .map_err(NodeGrafeoError::from)
            .map_err(napi::Error::from)
    }
}

/// Execute a query with optional JSON params.
fn execute_query(
    db: &GrafeoDB,
    query: &str,
    params: Option<&serde_json::Value>,
) -> std::result::Result<EngineQueryResult, napi::Error> {
    let param_map = convert_json_params(params)?;
    if let Some(p) = param_map {
        db.execute_with_params(query, p)
            .map_err(NodeGrafeoError::from)
            .map_err(napi::Error::from)
    } else {
        db.execute(query)
            .map_err(NodeGrafeoError::from)
            .map_err(napi::Error::from)
    }
}

/// Convert JSON params to a HashMap<String, Value>.
fn convert_json_params(
    params: Option<&serde_json::Value>,
) -> std::result::Result<Option<HashMap<String, Value>>, napi::Error> {
    let Some(params) = params else {
        return Ok(None);
    };
    let Some(obj) = params.as_object() else {
        return Err(NodeGrafeoError::InvalidArgument("params must be an object".into()).into());
    };
    let mut map = HashMap::with_capacity(obj.len());
    for (key, value) in obj {
        map.insert(key.clone(), json_to_value(value)?);
    }
    Ok(Some(map))
}

/// Convert a serde_json::Value to a Grafeo Value.
pub(crate) fn json_to_value(v: &serde_json::Value) -> std::result::Result<Value, napi::Error> {
    match v {
        serde_json::Value::Null => Ok(Value::Null),
        serde_json::Value::Bool(b) => Ok(Value::Bool(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(Value::Int64(i))
            } else if let Some(f) = n.as_f64() {
                Ok(Value::Float64(f))
            } else {
                Err(NodeGrafeoError::Type("Unsupported number type".into()).into())
            }
        }
        serde_json::Value::String(s) => Ok(Value::String(s.clone().into())),
        serde_json::Value::Array(arr) => {
            let items: std::result::Result<Vec<Value>, napi::Error> =
                arr.iter().map(json_to_value).collect();
            Ok(Value::List(items?.into()))
        }
        serde_json::Value::Object(obj) => {
            let mut map = std::collections::BTreeMap::new();
            for (k, v) in obj {
                map.insert(
                    grafeo_common::types::PropertyKey::new(k.clone()),
                    json_to_value(v)?,
                );
            }
            Ok(Value::Map(Arc::new(map)))
        }
    }
}

/// Fetch a node from the database and wrap it as JsNode.
fn fetch_node(db: &GrafeoDB, id: NodeId) -> Result<JsNode> {
    db.get_node(id)
        .map(|node| {
            let labels: Vec<String> = node.labels.iter().map(|s| s.to_string()).collect();
            let properties = node
                .properties
                .into_iter()
                .map(|(k, v)| (k.as_str().to_string(), v))
                .collect();
            JsNode::new(id, labels, properties)
        })
        .ok_or_else(|| NodeGrafeoError::Database("Failed to fetch created node".into()).into())
}

/// Fetch an edge from the database and wrap it as JsEdge.
fn fetch_edge(db: &GrafeoDB, id: EdgeId) -> Result<JsEdge> {
    db.get_edge(id)
        .map(|edge| {
            let properties = edge
                .properties
                .into_iter()
                .map(|(k, v)| (k.as_str().to_string(), v))
                .collect();
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
    db: &GrafeoDB,
) -> (Vec<JsNode>, Vec<JsEdge>) {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut seen_node_ids = HashSet::new();
    let mut seen_edge_ids = HashSet::new();

    let node_cols: Vec<usize> = result
        .column_types
        .iter()
        .enumerate()
        .filter_map(|(i, t)| {
            if *t == LogicalType::Node {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    let edge_cols: Vec<usize> = result
        .column_types
        .iter()
        .enumerate()
        .filter_map(|(i, t)| {
            if *t == LogicalType::Edge {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    for row in &result.rows {
        for &col_idx in &node_cols {
            if let Some(Value::Int64(id)) = row.get(col_idx) {
                let node_id = NodeId(*id as u64);
                if seen_node_ids.insert(node_id) {
                    if let Some(node) = db.get_node(node_id) {
                        let labels: Vec<String> =
                            node.labels.iter().map(|s| s.to_string()).collect();
                        let properties: HashMap<String, Value> = node
                            .properties
                            .into_iter()
                            .map(|(k, v)| (k.as_str().to_string(), v))
                            .collect();
                        nodes.push(JsNode::new(node_id, labels, properties));
                    }
                }
            }
        }

        for &col_idx in &edge_cols {
            if let Some(Value::Int64(id)) = row.get(col_idx) {
                let edge_id = EdgeId(*id as u64);
                if seen_edge_ids.insert(edge_id) {
                    if let Some(edge) = db.get_edge(edge_id) {
                        let properties: HashMap<String, Value> = edge
                            .properties
                            .into_iter()
                            .map(|(k, v)| (k.as_str().to_string(), v))
                            .collect();
                        edges.push(JsEdge::new(
                            edge_id,
                            edge.edge_type.to_string(),
                            edge.src,
                            edge.dst,
                            properties,
                        ));
                    }
                }
            }
        }
    }

    (nodes, edges)
}
