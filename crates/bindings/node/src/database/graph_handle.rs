//! `db.graph(name)`: work in one named graph without switching the database's
//! current graph.

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use parking_lot::RwLock;

use grafeo_engine::GraphHandle;
use grafeo_engine::database::{EdgeUpsertOptions, GrafeoDB};

use super::batch::{BatchEdgeInput, batch_edges, label_list};
use super::upsert::{
    UpsertEdgesOptions, UpsertNodesOptions, UpsertSummary, edge_options, node_options,
};
use super::{
    JsGrafeoDB, NodeEdgesOptions, edge_pairs, js_edge, js_node, js_result, node_edges_options,
    object_properties, validate_edge_id, validate_node_id,
};
use crate::error::NodeGrafeoError;
use crate::graph::{JsEdge, JsNode};
use crate::query::QueryResult;
use crate::transaction::Transaction;
use crate::types;

/// A handle on one named graph, from `db.graph(name)`.
///
/// It has the query and direct API of `GrafeoDB`, working in its graph
/// whatever graph the database selects. Each call is its own transaction
/// (use `beginTransaction()` to group several). Every call throws once the
/// graph no longer exists.
#[napi(js_name = "GraphHandle")]
pub struct JsGraphHandle {
    db: Arc<RwLock<GrafeoDB>>,
    schema: Option<String>,
    name: String,
}

/// Runs `call` with the engine's handle on graph `name` of `schema`.
fn in_graph<T>(
    db: &RwLock<GrafeoDB>,
    schema: Option<&str>,
    name: &str,
    call: impl FnOnce(&GraphHandle<'_>) -> Result<T>,
) -> Result<T> {
    let db = db.read();
    let graph = db.graph_in(schema, name).map_err(NodeGrafeoError::from)?;
    call(&graph)
}

/// Maps an engine error to a JavaScript one.
fn js<T>(result: grafeo_common::utils::error::Result<T>) -> Result<T> {
    result.map_err(|e| NodeGrafeoError::from(e).into())
}

impl JsGraphHandle {
    /// Runs `call` with the engine's handle on this graph.
    fn with_graph<T>(&self, call: impl FnOnce(&GraphHandle<'_>) -> Result<T>) -> Result<T> {
        in_graph(&self.db, self.schema.as_deref(), &self.name, call)
    }

    /// Runs `call` on a blocking thread with the engine's handle on this
    /// graph.
    async fn blocking<T: Send + 'static>(
        &self,
        call: impl FnOnce(&GraphHandle<'_>) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let db = Arc::clone(&self.db);
        let schema = self.schema.clone();
        let name = self.name.clone();
        tokio::task::spawn_blocking(move || in_graph(&db, schema.as_deref(), &name, call))
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }
}

#[napi]
impl JsGrafeoDB {
    /// A handle on named graph `name` of the current schema, to work in it
    /// without switching the current graph. Throws when it does not exist.
    #[napi]
    pub fn graph(&self, name: String) -> Result<JsGraphHandle> {
        let schema = {
            let db = self.inner.read();
            let graph = db.graph(&name).map_err(NodeGrafeoError::from)?;
            graph.schema().map(ToString::to_string)
        };
        Ok(JsGraphHandle {
            db: Arc::clone(&self.inner),
            schema,
            name,
        })
    }
}

#[napi]
impl JsGraphHandle {
    /// The graph's name.
    #[napi(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// Runs a GQL query in this graph. Returns a Promise<QueryResult>.
    #[napi]
    pub async fn execute(
        &self,
        query: String,
        #[napi(ts_arg_type = "any | undefined | null")] params: Option<crate::types::Params>,
    ) -> Result<QueryResult> {
        self.execute_language("gql".to_string(), query, params)
            .await
    }

    /// Runs a query in a named language (for example `"cypher"`) in this
    /// graph. Returns a Promise<QueryResult>.
    #[napi(js_name = "executeLanguage")]
    pub async fn execute_language(
        &self,
        language: String,
        query: String,
        #[napi(ts_arg_type = "any | undefined | null")] params: Option<crate::types::Params>,
    ) -> Result<QueryResult> {
        let params = params.map(crate::types::Params::into_map).transpose()?;
        let result = self
            .blocking(move |graph| js(graph.execute_language(&query, &language, params)))
            .await?;
        Ok(js_result(result))
    }

    /// Begins a transaction in this graph.
    #[napi(js_name = "beginTransaction")]
    pub fn begin_transaction(&self, isolation_level: Option<String>) -> Result<Transaction> {
        let session = self.with_graph(|graph| js(graph.session()))?;
        Transaction::begin(Arc::clone(&self.db), session, isolation_level.as_deref())
    }

    /// Creates a node with labels and optional properties.
    #[napi(js_name = "createNode")]
    pub fn create_node(
        &self,
        env: Env,
        labels: Vec<String>,
        properties: Option<Object<'_>>,
    ) -> Result<JsNode> {
        let properties = match properties {
            Some(object) => object_properties(env, &object)?,
            None => Vec::new(),
        };
        self.with_graph(|graph| {
            let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
            let id = js(graph.create_node_with_props(&labels, properties))?;
            fetched(js(graph.get_node(id))?.map(|node| js_node(id, node)))
        })
    }

    /// Creates an edge between two nodes of this graph.
    #[napi(js_name = "createEdge")]
    pub fn create_edge(
        &self,
        env: Env,
        source_id: f64,
        target_id: f64,
        edge_type: String,
        properties: Option<Object<'_>>,
    ) -> Result<JsEdge> {
        let src = validate_node_id(source_id)?;
        let dst = validate_node_id(target_id)?;
        let properties = match properties {
            Some(object) => object_properties(env, &object)?,
            None => Vec::new(),
        };
        self.with_graph(|graph| {
            let id = js(graph.create_edge_with_props(src, dst, &edge_type, properties))?;
            fetched(js(graph.get_edge(id))?.map(|edge| js_edge(id, edge)))
        })
    }

    /// Gets a node by ID, or null.
    #[napi(js_name = "getNode")]
    pub fn get_node(&self, id: f64) -> Result<Option<JsNode>> {
        let id = validate_node_id(id)?;
        self.with_graph(|graph| Ok(js(graph.get_node(id))?.map(|node| js_node(id, node))))
    }

    /// `GrafeoDB.getNodeEdges` in this graph; throws if the graph no longer
    /// exists.
    #[napi(js_name = "getNodeEdges")]
    pub fn get_node_edges(
        &self,
        id: f64,
        options: Option<NodeEdgesOptions>,
    ) -> Result<Vec<Vec<f64>>> {
        let id = validate_node_id(id)?;
        let (direction, edge_types) = node_edges_options(options)?;
        let edge_types: Vec<&str> = edge_types.iter().map(String::as_str).collect();
        self.with_graph(|graph| {
            Ok(edge_pairs(js(graph.get_node_edges(
                id,
                direction,
                &edge_types,
            ))?))
        })
    }

    /// Gets an edge by ID, or null.
    #[napi(js_name = "getEdge")]
    pub fn get_edge(&self, id: f64) -> Result<Option<JsEdge>> {
        let id = validate_edge_id(id)?;
        self.with_graph(|graph| Ok(js(graph.get_edge(id))?.map(|edge| js_edge(id, edge))))
    }

    /// Sets a node property; throws if the node does not exist.
    #[napi(js_name = "setNodeProperty")]
    pub fn set_node_property(
        &self,
        env: Env,
        id: f64,
        key: String,
        value: Unknown<'_>,
    ) -> Result<()> {
        let id = validate_node_id(id)?;
        let value = types::js_to_named_value(env, value, "property", &key)?;
        self.with_graph(|graph| js(graph.set_node_property(id, &key, value)))
    }

    /// Sets an edge property; throws if the edge does not exist.
    #[napi(js_name = "setEdgeProperty")]
    pub fn set_edge_property(
        &self,
        env: Env,
        id: f64,
        key: String,
        value: Unknown<'_>,
    ) -> Result<()> {
        let id = validate_edge_id(id)?;
        let value = types::js_to_named_value(env, value, "property", &key)?;
        self.with_graph(|graph| js(graph.set_edge_property(id, &key, value)))
    }

    /// Removes a node property; returns whether the node had it.
    #[napi(js_name = "removeNodeProperty")]
    pub fn remove_node_property(&self, id: f64, key: String) -> Result<bool> {
        let id = validate_node_id(id)?;
        self.with_graph(|graph| js(graph.remove_node_property(id, &key)))
    }

    /// Removes an edge property; returns whether the edge had it.
    #[napi(js_name = "removeEdgeProperty")]
    pub fn remove_edge_property(&self, id: f64, key: String) -> Result<bool> {
        let id = validate_edge_id(id)?;
        self.with_graph(|graph| js(graph.remove_edge_property(id, &key)))
    }

    /// Adds a label; returns false if the node does not exist or has it.
    #[napi(js_name = "addNodeLabel")]
    pub fn add_node_label(&self, id: f64, label: String) -> Result<bool> {
        let id = validate_node_id(id)?;
        self.with_graph(|graph| js(graph.add_node_label(id, &label)))
    }

    /// Removes a label; returns false if the node does not exist or lacks it.
    #[napi(js_name = "removeNodeLabel")]
    pub fn remove_node_label(&self, id: f64, label: String) -> Result<bool> {
        let id = validate_node_id(id)?;
        self.with_graph(|graph| js(graph.remove_node_label(id, &label)))
    }

    /// Deletes a node; returns false if it does not exist, throws if it
    /// still has edges.
    #[napi(js_name = "deleteNode")]
    pub fn delete_node(&self, id: f64) -> Result<bool> {
        let id = validate_node_id(id)?;
        self.with_graph(|graph| js(graph.delete_node(id)))
    }

    /// Deletes a node with all its edges, as `DETACH DELETE` does; returns
    /// false if it does not exist.
    #[napi(js_name = "detachDeleteNode")]
    pub fn detach_delete_node(&self, id: f64) -> Result<bool> {
        let id = validate_node_id(id)?;
        self.with_graph(|graph| js(graph.detach_delete_node(id)))
    }

    /// Deletes an edge; returns false if it does not exist.
    #[napi(js_name = "deleteEdge")]
    pub fn delete_edge(&self, id: f64) -> Result<bool> {
        let id = validate_edge_id(id)?;
        self.with_graph(|graph| js(graph.delete_edge(id)))
    }

    /// `GrafeoDB.batchCreateNodesWithProps` in this graph.
    #[napi(js_name = "batchCreateNodesWithProps")]
    pub async fn batch_create_nodes_with_props(
        &self,
        labels: Either<String, Vec<String>>,
        #[napi(ts_arg_type = "Array<any>")] properties_list: Vec<crate::types::Properties>,
    ) -> Result<Vec<f64>> {
        let labels = label_list(labels);
        let properties_list = crate::types::Properties::rows(properties_list)?;
        self.blocking(move |graph| {
            let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
            let ids = js(graph.batch_create_nodes_with_labels(&labels, properties_list))?;
            Ok(ids.into_iter().map(|id| id.as_u64() as f64).collect())
        })
        .await
    }

    /// `GrafeoDB.batchCreateEdges` in this graph.
    #[napi(js_name = "batchCreateEdges")]
    pub async fn batch_create_edges(&self, edges: Vec<BatchEdgeInput>) -> Result<Vec<f64>> {
        let edges = batch_edges(edges)?;
        self.blocking(move |graph| {
            let ids = js(graph.batch_create_edges(edges))?;
            Ok(ids.into_iter().map(|id| id.as_u64() as f64).collect())
        })
        .await
    }

    /// `GrafeoDB.upsertNodes` in this graph.
    #[napi(js_name = "upsertNodes")]
    pub async fn upsert_nodes(
        &self,
        labels: Vec<String>,
        #[napi(ts_arg_type = "Array<any>")] rows: Vec<crate::types::Properties>,
        options: Option<UpsertNodesOptions>,
    ) -> Result<UpsertSummary> {
        let rows = crate::types::Properties::rows(rows)?;
        let (key, replace) = node_options(options);
        self.blocking(move |graph| {
            let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
            js(graph.upsert_nodes(&labels, &key, rows, replace)).map(UpsertSummary::from)
        })
        .await
    }

    /// `GrafeoDB.upsertEdges` in this graph.
    #[napi(js_name = "upsertEdges")]
    pub async fn upsert_edges(
        &self,
        edge_type: String,
        #[napi(ts_arg_type = "Array<any>")] rows: Vec<crate::types::Properties>,
        options: Option<UpsertEdgesOptions>,
    ) -> Result<UpsertSummary> {
        let rows = crate::types::Properties::rows(rows)?;
        let options = options.map_or_else(EdgeUpsertOptions::new, edge_options);
        self.blocking(move |graph| {
            js(graph.upsert_edges(&edge_type, rows, &options)).map(UpsertSummary::from)
        })
        .await
    }

    /// String representation.
    #[napi(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!("GraphHandle({:?})", self.name)
    }
}

/// The node or edge a call just created, read back.
fn fetched<T>(entity: Option<T>) -> Result<T> {
    entity.ok_or_else(|| {
        NodeGrafeoError::Database("Failed to fetch the created entity".into()).into()
    })
}
