//! Batch writes, upserts and batch vector search, and the conversions the
//! database and its graph handles share.

use std::collections::HashMap;

use js_sys::Array;
use wasm_bindgen::prelude::*;

use crate::{database_error, js_error};

use crate::types;
use grafeo_common::types::{NodeId, PropertyKey, Value};
use grafeo_engine::database::{BatchEdge, EdgeUpsertOptions, UpsertSummary};

use super::{Database, ImportLabel, value_to_node_id};

/// Options of `upsertNodes`.
#[derive(Default, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct NodeUpsertInput {
    key: Option<String>,
    replace: Option<bool>,
}

/// Options of `upsertEdges`.
#[derive(Default, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct EdgeUpsertInput {
    key: Option<String>,
    endpoint_key: Option<String>,
    endpoint_labels: Option<Vec<String>>,
    src_field: Option<String>,
    dst_field: Option<String>,
    replace: Option<bool>,
}

fn invalid(what: &str, error: impl std::fmt::Display) -> JsValue {
    js_error(&format!("{what}: {error}"))
}

/// Options from a JS value; undefined or null means the defaults.
fn options<T: Default + serde::de::DeserializeOwned>(
    value: JsValue,
    what: &str,
) -> Result<T, JsValue> {
    if value.is_undefined() || value.is_null() {
        return Ok(T::default());
    }
    serde_wasm_bindgen::from_value(value).map_err(|e| invalid(what, e))
}

/// A label or a list of labels, as a list.
pub(crate) fn labels(value: JsValue) -> Result<Vec<String>, JsValue> {
    let labels: ImportLabel =
        serde_wasm_bindgen::from_value(value).map_err(|e| invalid("labels", e))?;
    Ok(match labels {
        ImportLabel::Single(label) => vec![label],
        ImportLabel::Multiple(labels) => labels,
    })
}

/// Rows of properties from an array of objects, each converted from the
/// JavaScript value as a property is; an error names the row.
pub(crate) fn rows(value: JsValue) -> Result<Vec<HashMap<PropertyKey, Value>>, JsValue> {
    let rows = types::js_rows(&value).map_err(|error| js_error(&error))?;
    Ok(rows
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|(name, value)| (PropertyKey::new(name), value))
                .collect()
        })
        .collect())
}

/// The engine edges of `batchCreateEdges`: each `{src, dst, type,
/// properties}` read from the JavaScript value, as a property is.
pub(crate) fn edges(value: JsValue) -> Result<Vec<BatchEdge>, JsValue> {
    if !Array::is_array(&value) {
        return Err(js_error(
            "edges must be an array of {src, dst, type, properties}",
        ));
    }
    Array::from(&value)
        .iter()
        .enumerate()
        .map(|(i, edge)| {
            let endpoint = |name: &str| -> Result<NodeId, JsValue> {
                let value = types::js_to_value(&types::js_field(&edge, name))
                    .map_err(|error| js_error(&format!("edges[{i}].{name}: {error}")))?;
                value_to_node_id(&value, name, i)
            };
            let src = endpoint("src")?;
            let dst = endpoint("dst")?;
            let edge_type = types::js_field(&edge, "type")
                .as_string()
                .ok_or_else(|| js_error(&format!("edges[{i}].type must be a string")))?;
            let properties = types::js_properties(&types::js_field(&edge, "properties"))
                .map_err(|error| js_error(&format!("edges[{i}]: {error}")))?;
            Ok(BatchEdge::new(src, dst, edge_type).with_properties(properties))
        })
        .collect()
}

/// The key and replace flag of `upsertNodes`, defaults filled in.
pub(crate) fn node_options(value: JsValue) -> Result<(String, bool), JsValue> {
    let input: NodeUpsertInput = options(value, "upsertNodes options")?;
    Ok((
        input.key.unwrap_or_else(|| "id".to_string()),
        input.replace.unwrap_or(false),
    ))
}

/// The engine options of `upsertEdges`: each option given, the default for
/// the rest.
pub(crate) fn edge_options(value: JsValue) -> Result<EdgeUpsertOptions, JsValue> {
    let input: EdgeUpsertInput = options(value, "upsertEdges options")?;
    let mut options = EdgeUpsertOptions::new().with_replace(input.replace.unwrap_or(false));
    if let Some(key) = input.key {
        options = options.with_key(key);
    }
    if let Some(endpoint_key) = input.endpoint_key {
        options = options.with_endpoint_key(endpoint_key);
    }
    if let Some(endpoint_labels) = input.endpoint_labels {
        options = options.with_endpoint_labels(endpoint_labels);
    }
    if let Some(src_field) = input.src_field {
        options = options.with_src_field(src_field);
    }
    if let Some(dst_field) = input.dst_field {
        options = options.with_dst_field(dst_field);
    }
    Ok(options)
}

/// The largest integer a JS number holds exactly, 2^53 - 1.
const MAX_SAFE_INTEGER: u64 = (1 << 53) - 1;

/// IDs as a JS array of numbers; an error for an ID a number cannot hold.
pub(crate) fn ids_to_js(ids: impl ExactSizeIterator<Item = u64>) -> Result<JsValue, JsValue> {
    let array = Array::new_with_length(ids.len() as u32);
    for (i, id) in ids.enumerate() {
        if id > MAX_SAFE_INTEGER {
            return Err(js_error(&format!(
                "ID {id} is larger than a JS number holds exactly"
            )));
        }
        #[allow(
            clippy::cast_precision_loss,
            reason = "checked above to be at most 2^53 - 1, which an f64 holds exactly"
        )]
        array.set(i as u32, JsValue::from_f64(id as f64));
    }
    Ok(array.into())
}

/// An upsert summary as `{created, updated, skipped, skippedRows}`.
pub(crate) fn summary_to_js(summary: &UpsertSummary) -> JsValue {
    let object = js_sys::Object::new();
    let set = |name: &str, value: JsValue| {
        let _ = js_sys::Reflect::set(&object, &JsValue::from_str(name), &value);
    };
    set("created", JsValue::from(summary.created as u32));
    set("updated", JsValue::from(summary.updated as u32));
    set("skipped", JsValue::from(summary.skipped as u32));
    let skipped_rows = Array::new_with_length(summary.skipped_rows.len() as u32);
    for (i, row) in summary.skipped_rows.iter().enumerate() {
        skipped_rows.set(i as u32, JsValue::from(*row as u32));
    }
    set("skippedRows", skipped_rows.into());
    object.into()
}

/// An error for a call that would write around the open transaction.
pub(crate) fn transaction_open(call: &str) -> JsValue {
    js_error(&format!(
        "{call} commits on its own: commit or roll back the open transaction first"
    ))
}

fn js<T>(result: grafeo_common::utils::error::Result<T>) -> Result<T, JsValue> {
    result.map_err(database_error)
}

#[wasm_bindgen]
impl Database {
    /// Creates one node per row, each with `labels` (a label or a list of
    /// labels) and the row's properties, in one transaction: if one breaks a
    /// constraint, none is created. Returns the node IDs in row order.
    ///
    /// Throws while a transaction is open: the batch commits on its own.
    ///
    /// ```js
    /// const ids = db.batchCreateNodesWithProps(["Person"], [{ name: "Alix" }, { name: "Gus" }]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the arguments are malformed, a transaction is
    /// open, or a node breaks a constraint.
    #[wasm_bindgen(js_name = "batchCreateNodesWithProps")]
    pub fn batch_create_nodes_with_props(
        &self,
        labels: JsValue,
        rows: JsValue,
    ) -> Result<JsValue, JsValue> {
        self.check_open()?;
        let labels = self::labels(labels)?;
        let rows = self::rows(rows)?;
        if self.tx.borrow().is_some() {
            return Err(transaction_open("batchCreateNodesWithProps"));
        }
        let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
        let ids = js(self.db().batch_create_nodes_with_labels(&labels, rows))?;
        ids_to_js(ids.into_iter().map(|id| id.as_u64()))
    }

    /// Creates the edges, each `{src, dst, type, properties}` (`properties`
    /// optional), in one transaction: if one names a node that does not exist
    /// or breaks the schema, none is created. Returns the edge IDs in input
    /// order.
    ///
    /// Throws while a transaction is open: the batch commits on its own.
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the edges are malformed, a transaction is open,
    /// or an edge cannot be created.
    #[wasm_bindgen(js_name = "batchCreateEdges")]
    pub fn batch_create_edges(&self, edges: JsValue) -> Result<JsValue, JsValue> {
        self.check_open()?;
        let edges = self::edges(edges)?;
        if self.tx.borrow().is_some() {
            return Err(transaction_open("batchCreateEdges"));
        }
        let ids = js(self.db().batch_create_edges(edges))?;
        ids_to_js(ids.into_iter().map(|id| id.as_u64()))
    }

    /// Creates or updates one node per row, matched by the key property
    /// (`options.key`, default `id`) and all of `labels`, in one statement;
    /// a row without the key is skipped. With `options.replace` a node's
    /// properties become exactly the row's, without it the row's are merged
    /// in. Runs in the open transaction when there is one.
    ///
    /// Returns `{created, updated, skipped, skippedRows}`.
    ///
    /// ```js
    /// db.upsertNodes(["File"], [{ path: "a.rs", size: 3 }], { key: "path" });
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the arguments are malformed or a row breaks a
    /// constraint or the schema.
    #[wasm_bindgen(js_name = "upsertNodes")]
    pub fn upsert_nodes(
        &self,
        labels: JsValue,
        rows: JsValue,
        options: JsValue,
    ) -> Result<JsValue, JsValue> {
        self.check_open()?;
        let labels = self::labels(labels)?;
        let rows = self::rows(rows)?;
        let (key, replace) = node_options(options)?;
        let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
        let summary = match self.tx.borrow().as_ref() {
            Some(session) => js(session.upsert_nodes(&labels, &key, rows, replace))?,
            None => js(self.db().upsert_nodes(&labels, &key, rows, replace))?,
        };
        Ok(summary_to_js(&summary))
    }

    /// Creates or updates one edge of `edgeType` per row between existing
    /// nodes, in one statement. Each row names its endpoints in the source
    /// and target fields (`options.srcField` and `options.dstField`, default
    /// `src` and `dst`), which hold the `options.endpointKey` property
    /// (default `id`) of nodes with all of `options.endpointLabels`, and holds
    /// the edge key (`options.key`, default `id`). A row whose endpoint
    /// matches no node or several is skipped; endpoints are never created.
    /// Runs in the open transaction when there is one.
    ///
    /// Returns `{created, updated, skipped, skippedRows}`.
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the arguments are malformed or a row breaks the
    /// schema.
    #[wasm_bindgen(js_name = "upsertEdges")]
    pub fn upsert_edges(
        &self,
        edge_type: &str,
        rows: JsValue,
        options: JsValue,
    ) -> Result<JsValue, JsValue> {
        self.check_open()?;
        let rows = self::rows(rows)?;
        let options = edge_options(options)?;
        let summary = match self.tx.borrow().as_ref() {
            Some(session) => js(session.upsert_edges(edge_type, rows, &options))?,
            None => js(self.db().upsert_edges(edge_type, rows, &options))?,
        };
        Ok(summary_to_js(&summary))
    }
}
