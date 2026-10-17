//! Batch writes for the Node.js API: many nodes or edges in one transaction.

use napi::bindgen_prelude::*;
use napi_derive::napi;

use grafeo_engine::database::BatchEdge;

use super::{JsGrafeoDB, json_to_value, upsert::rows, validate_node_id};
use crate::error::NodeGrafeoError;

/// An edge for `batchCreateEdges`.
#[napi(object)]
pub struct BatchEdgeInput {
    /// The source node's ID.
    pub src: f64,
    /// The target node's ID.
    pub dst: f64,
    /// The edge type.
    #[napi(js_name = "type")]
    pub edge_type: String,
    /// The edge's properties.
    pub properties: Option<serde_json::Value>,
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
        properties_list: Vec<serde_json::Value>,
    ) -> Result<Vec<f64>> {
        let labels = match labels {
            Either::A(label) => vec![label],
            Either::B(labels) => labels,
        };
        let properties_list = rows(&properties_list)?;
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
        let edges = edges
            .into_iter()
            .map(|edge| {
                let mut batch_edge = BatchEdge::new(
                    validate_node_id(edge.src)?,
                    validate_node_id(edge.dst)?,
                    edge.edge_type,
                );
                if let Some(properties) = edge.properties {
                    let serde_json::Value::Object(fields) = properties else {
                        return Err(NodeGrafeoError::InvalidArgument(
                            "edge properties must be an object".to_string(),
                        )
                        .into());
                    };
                    batch_edge = batch_edge.with_properties(
                        fields
                            .iter()
                            .map(|(name, value)| Ok((name.as_str(), json_to_value(value)?)))
                            .collect::<Result<Vec<_>>>()?,
                    );
                }
                Ok(batch_edge)
            })
            .collect::<Result<Vec<_>>>()?;
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
