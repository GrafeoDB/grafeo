//! Batch writes for the Node.js API: many nodes or edges in one transaction.

use napi::bindgen_prelude::*;
use napi_derive::napi;

use grafeo_engine::database::BatchEdge;

use super::{JsGrafeoDB, validate_node_id};
use crate::error::NodeGrafeoError;
use crate::types::Properties;

/// An edge for `batchCreateEdges`.
#[napi(object, object_to_js = false)]
pub struct BatchEdgeInput {
    /// The source node's ID.
    pub src: f64,
    /// The target node's ID.
    pub dst: f64,
    /// The edge type.
    #[napi(js_name = "type")]
    pub edge_type: String,
    /// The edge's properties.
    #[napi(ts_type = "any")]
    pub properties: Option<Properties>,
}

/// A label or a list of labels, as a list.
pub(super) fn label_list(labels: Either<String, Vec<String>>) -> Vec<String> {
    match labels {
        Either::A(label) => vec![label],
        Either::B(labels) => labels,
    }
}

/// The engine edges of `batchCreateEdges`.
pub(super) fn batch_edges(edges: Vec<BatchEdgeInput>) -> Result<Vec<BatchEdge>> {
    edges
        .into_iter()
        .map(|edge| {
            let mut batch_edge = BatchEdge::new(
                validate_node_id(edge.src)?,
                validate_node_id(edge.dst)?,
                edge.edge_type,
            );
            if let Some(properties) = edge.properties {
                batch_edge = batch_edge.with_properties(properties.into_entries()?);
            }
            Ok(batch_edge)
        })
        .collect()
}

#[napi]
impl JsGrafeoDB {
    /// Creates one node per properties object, each with `labels` (a label
    /// or a list of labels), in one transaction: if one breaks a constraint,
    /// none is created. Returns a Promise of the node IDs, in input order.
    #[napi(js_name = "batchCreateNodesWithProps")]
    pub async fn batch_create_nodes_with_props(
        &self,
        labels: Either<String, Vec<String>>,
        #[napi(ts_arg_type = "Array<any>")] properties_list: Vec<Properties>,
    ) -> Result<Vec<f64>> {
        let labels = label_list(labels);
        let properties_list = Properties::rows(properties_list)?;
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
            let ids = db
                .read()
                .batch_create_nodes_with_labels(&labels, properties_list)
                .map_err(NodeGrafeoError::from)?;
            Ok(ids.into_iter().map(|id| id.as_u64() as f64).collect())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Creates the edges, each with its own `type` and `properties`, in one
    /// transaction: if one names a node that does not exist or breaks the
    /// schema, none is created. Returns a Promise of the edge IDs, in input
    /// order.
    #[napi(js_name = "batchCreateEdges")]
    pub async fn batch_create_edges(&self, edges: Vec<BatchEdgeInput>) -> Result<Vec<f64>> {
        let edges = batch_edges(edges)?;
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let ids = db
                .read()
                .batch_create_edges(edges)
                .map_err(NodeGrafeoError::from)?;
            Ok(ids.into_iter().map(|id| id.as_u64() as f64).collect())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }
}
