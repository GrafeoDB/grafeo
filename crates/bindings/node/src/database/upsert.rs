//! Upserts for the Node.js API: create or update nodes and edges by key.

use std::collections::HashMap;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::database::{EdgeUpsertOptions, UpsertSummary as EngineUpsertSummary};

use super::{JsGrafeoDB, json_to_value};
use crate::error::NodeGrafeoError;

/// Options for `upsertNodes`.
#[napi(object)]
pub struct UpsertNodesOptions {
    /// The property that identifies a node (default `id`).
    pub key: Option<String>,
    /// Whether a node's properties become exactly the row's (default
    /// `false`: the row's properties are merged in and none is removed).
    pub replace: Option<bool>,
}

/// Options for `upsertEdges`.
#[napi(object)]
pub struct UpsertEdgesOptions {
    /// The property that identifies an edge between two nodes (default `id`).
    pub key: Option<String>,
    /// The node property the endpoint fields hold (default `id`). A property
    /// index on it makes the lookups fast.
    pub endpoint_key: Option<String>,
    /// Labels an endpoint must have (default none: the key alone).
    pub endpoint_labels: Option<Vec<String>>,
    /// The row field with the source node's key (default `src`).
    pub src_field: Option<String>,
    /// The row field with the target node's key (default `dst`).
    pub dst_field: Option<String>,
    /// Whether an edge's properties become exactly the row's (default
    /// `false`: the row's properties are merged in).
    pub replace: Option<bool>,
}

/// What an upsert did with its rows.
#[napi(object)]
pub struct UpsertSummary {
    /// Rows that created a node or edge.
    pub created: u32,
    /// Rows that updated an existing node or edge.
    pub updated: u32,
    /// Rows that were not written: without their key, edge rows without a
    /// source or target field, and edge rows whose endpoint key matches no
    /// node or more than one node.
    pub skipped: u32,
    /// The indices of the skipped rows, in order (at most 1,000).
    pub skipped_rows: Vec<u32>,
}

impl From<EngineUpsertSummary> for UpsertSummary {
    fn from(summary: EngineUpsertSummary) -> Self {
        let count = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
        Self {
            created: count(summary.created),
            updated: count(summary.updated),
            skipped: count(summary.skipped),
            skipped_rows: summary.skipped_rows.into_iter().map(count).collect(),
        }
    }
}

/// The engine options for `upsertEdges`: each option given, the default for
/// the rest.
fn edge_options(options: UpsertEdgesOptions) -> EdgeUpsertOptions {
    let mut edge = EdgeUpsertOptions::new();
    if let Some(key) = options.key {
        edge = edge.with_key(key);
    }
    if let Some(endpoint_key) = options.endpoint_key {
        edge = edge.with_endpoint_key(endpoint_key);
    }
    if let Some(labels) = options.endpoint_labels {
        edge = edge.with_endpoint_labels(labels);
    }
    if let Some(src_field) = options.src_field {
        edge = edge.with_src_field(src_field);
    }
    if let Some(dst_field) = options.dst_field {
        edge = edge.with_dst_field(dst_field);
    }
    if let Some(replace) = options.replace {
        edge = edge.with_replace(replace);
    }
    edge
}

/// Rows of properties from JSON objects.
pub(super) fn rows(rows: &[serde_json::Value]) -> Result<Vec<HashMap<PropertyKey, Value>>> {
    rows.iter()
        .map(|row| {
            let serde_json::Value::Object(fields) = row else {
                return Err(NodeGrafeoError::InvalidArgument(
                    "each upsert row must be an object".to_string(),
                )
                .into());
            };
            fields
                .iter()
                .map(|(name, value)| Ok((PropertyKey::new(name.as_str()), json_to_value(value)?)))
                .collect()
        })
        .collect()
}

#[napi]
impl JsGrafeoDB {
    /// Creates or updates one node per row, matched by `key` and all of
    /// `labels`, in one statement. Returns a Promise.
    ///
    /// Each row is an object of properties holding the key; a row without it
    /// is skipped. By default a row's properties are merged into the node's;
    /// with `replace: true` they become exactly the row's. Labels are never
    /// removed. A key repeated within one call creates one node, which the
    /// later rows update. The call is checked like a query and writes all
    /// rows or none.
    #[napi(js_name = "upsertNodes")]
    pub async fn upsert_nodes(
        &self,
        labels: Vec<String>,
        rows: Vec<serde_json::Value>,
        options: Option<UpsertNodesOptions>,
    ) -> Result<UpsertSummary> {
        let rows = self::rows(&rows)?;
        let (key, replace) = options.map_or((None, None), |options| (options.key, options.replace));
        let key = key.unwrap_or_else(|| "id".to_string());
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
            db.read()
                .upsert_nodes(&labels, &key, rows, replace.unwrap_or(false))
                .map(UpsertSummary::from)
                .map_err(|e| NodeGrafeoError::from(e).into())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Creates or updates one edge of `edgeType` per row between existing
    /// nodes, in one statement. Returns a Promise.
    ///
    /// Each row names its endpoints in the source and target fields (`src`
    /// and `dst` by default) and holds the edge key; every other field is an
    /// edge property. A row is skipped when it lacks the edge key, the
    /// source field or the target field, or when no node or more than one
    /// node has its endpoint key; endpoints are never created. The key and
    /// the two endpoint fields must be different fields.
    #[napi(js_name = "upsertEdges")]
    pub async fn upsert_edges(
        &self,
        edge_type: String,
        rows: Vec<serde_json::Value>,
        options: Option<UpsertEdgesOptions>,
    ) -> Result<UpsertSummary> {
        let rows = self::rows(&rows)?;
        let options = options.map_or_else(EdgeUpsertOptions::new, edge_options);
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            db.read()
                .upsert_edges(&edge_type, rows, &options)
                .map(UpsertSummary::from)
                .map_err(|e| NodeGrafeoError::from(e).into())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }
}
