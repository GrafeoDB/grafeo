//! Graph elements exposed to JavaScript - nodes and edges with their properties.

use std::collections::HashMap;

use napi::bindgen_prelude::*;
use napi::{JsObject, JsUnknown};
use napi_derive::napi;

use grafeo_common::types::{EdgeId, NodeId, Value};

use crate::types::value_to_js;

/// A node in the graph with labels and properties.
#[napi]
#[derive(Clone)]
pub struct JsNode {
    pub(crate) id: NodeId,
    pub(crate) labels: Vec<String>,
    pub(crate) properties: HashMap<String, Value>,
}

#[napi]
impl JsNode {
    /// Get the node ID.
    #[napi(getter)]
    pub fn id(&self) -> f64 {
        self.id.0 as f64
    }

    /// Get the node labels.
    #[napi(getter)]
    pub fn labels(&self) -> Vec<String> {
        self.labels.clone()
    }

    /// Get a property value by key.
    #[napi]
    pub fn get(&self, env: Env, key: String) -> Result<JsUnknown> {
        match self.properties.get(&key) {
            Some(v) => value_to_js(&env, v),
            None => Ok(env.get_undefined()?.into_unknown()),
        }
    }

    /// Get all properties as a plain object.
    #[napi]
    pub fn properties(&self, env: Env) -> Result<JsObject> {
        let mut obj = env.create_object()?;
        for (k, v) in &self.properties {
            obj.set_named_property(k, value_to_js(&env, v)?)?;
        }
        Ok(obj)
    }

    /// Check if the node has a specific label.
    #[napi(js_name = "hasLabel")]
    pub fn has_label(&self, label: String) -> bool {
        self.labels.iter().any(|l| l == &label)
    }

    /// String representation.
    #[napi(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!("(:{} {{id: {}}})", self.labels.join(":"), self.id.0)
    }
}

impl JsNode {
    pub fn new(id: NodeId, labels: Vec<String>, properties: HashMap<String, Value>) -> Self {
        Self {
            id,
            labels,
            properties,
        }
    }
}

/// An edge (relationship) between two nodes with a type and properties.
#[napi]
#[derive(Clone)]
pub struct JsEdge {
    pub(crate) id: EdgeId,
    pub(crate) edge_type: String,
    pub(crate) source_id: NodeId,
    pub(crate) target_id: NodeId,
    pub(crate) properties: HashMap<String, Value>,
}

#[napi]
impl JsEdge {
    /// Get the edge ID.
    #[napi(getter)]
    pub fn id(&self) -> f64 {
        self.id.0 as f64
    }

    /// Get the edge type (relationship type).
    #[napi(getter, js_name = "edgeType")]
    pub fn edge_type(&self) -> String {
        self.edge_type.clone()
    }

    /// Get the source node ID.
    #[napi(getter, js_name = "sourceId")]
    pub fn source_id(&self) -> f64 {
        self.source_id.0 as f64
    }

    /// Get the target node ID.
    #[napi(getter, js_name = "targetId")]
    pub fn target_id(&self) -> f64 {
        self.target_id.0 as f64
    }

    /// Get a property value by key.
    #[napi]
    pub fn get(&self, env: Env, key: String) -> Result<JsUnknown> {
        match self.properties.get(&key) {
            Some(v) => value_to_js(&env, v),
            None => Ok(env.get_undefined()?.into_unknown()),
        }
    }

    /// Get all properties as a plain object.
    #[napi]
    pub fn properties(&self, env: Env) -> Result<JsObject> {
        let mut obj = env.create_object()?;
        for (k, v) in &self.properties {
            obj.set_named_property(k, value_to_js(&env, v)?)?;
        }
        Ok(obj)
    }

    /// String representation.
    #[napi(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!(
            "()-[:{}]->() (id={})",
            self.edge_type, self.id.0
        )
    }
}

impl JsEdge {
    pub fn new(
        id: EdgeId,
        edge_type: String,
        source_id: NodeId,
        target_id: NodeId,
        properties: HashMap<String, Value>,
    ) -> Self {
        Self {
            id,
            edge_type,
            source_id,
            target_id,
            properties,
        }
    }
}
