//! Handles on one named graph, for working in several graphs at once.

use std::collections::HashMap;

use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};
use grafeo_common::utils::error::Result;
use grafeo_core::graph::lpg::{Edge, Node};

use super::direct::{BatchEdge, DirectCalls, DirectTarget, missing_graph};
use super::{GrafeoDB, QueryResult};
use crate::session::{Session, graph_storage_key};

/// A handle on one named graph of a database.
///
/// A handle neither reads nor changes the database's current graph
/// ([`set_current_graph`](GrafeoDB::set_current_graph)): handles on
/// different graphs can be used side by side, and one handle from several
/// threads at once. Each query through [`execute`](Self::execute) and each
/// direct write ([`create_node`](Self::create_node) and the rest, which work
/// like the database's) commits on its own; begin a transaction on a
/// [`session`](Self::session) to group several.
///
/// A handle keeps the schema that was current when it was made. Every call
/// fails if the graph no longer exists, instead of working in another graph.
///
/// # Example
///
/// ```
/// # use grafeo_engine::GrafeoDB;
/// # use grafeo_common::types::Value;
/// let db = GrafeoDB::new_in_memory();
/// db.execute("CREATE GRAPH model")?;
/// let model = db.graph("model")?;
///
/// model.create_node_with_props(&["Component"], [("name", Value::from("Billing"))])?;
///
/// let count = "MATCH (n) RETURN count(n)";
/// assert_eq!(model.execute(count)?.rows()[0][0], Value::Int64(1));
/// assert_eq!(db.execute(count)?.rows()[0][0], Value::Int64(0));
/// # Ok::<(), grafeo_common::utils::error::Error>(())
/// ```
#[derive(Clone)]
pub struct GraphHandle<'db> {
    db: &'db GrafeoDB,
    schema: Option<String>,
    name: String,
}

impl std::fmt::Debug for GraphHandle<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphHandle")
            .field("schema", &self.schema)
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl GrafeoDB {
    /// Returns a handle on the named graph of the current schema.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph does not exist.
    pub fn graph(&self, name: &str) -> Result<GraphHandle<'_>> {
        let schema = self.current_schema();
        self.graph_in(schema.as_deref(), name)
    }

    /// Returns a handle on the named graph of `schema` (`None`: no schema).
    ///
    /// # Errors
    ///
    /// Returns an error if the graph does not exist.
    pub fn graph_in(&self, schema: Option<&str>, name: &str) -> Result<GraphHandle<'_>> {
        let handle = GraphHandle {
            db: self,
            schema: schema.map(ToString::to_string),
            name: name.to_string(),
        };
        handle.check_exists()?;
        Ok(handle)
    }
}

impl GraphHandle<'_> {
    /// The graph's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The schema the graph belongs to, if any.
    #[must_use]
    pub fn schema(&self) -> Option<&str> {
        self.schema.as_deref()
    }

    /// Opens a session working in this graph.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph no longer exists.
    pub fn session(&self) -> Result<Session> {
        self.check_exists()?;
        let session = self.db.session();
        match &self.schema {
            Some(schema) => session.set_schema(schema),
            None => session.reset_schema(),
        }
        session.use_graph(&self.name);
        Ok(session)
    }

    /// Runs a GQL query on this graph.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph no longer exists or the query fails.
    pub fn execute(&self, query: &str) -> Result<QueryResult> {
        self.session()?.execute(query)
    }

    /// Runs a GQL query with parameters on this graph.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph no longer exists or the query fails.
    pub fn execute_with_params(
        &self,
        query: &str,
        params: HashMap<String, Value>,
    ) -> Result<QueryResult> {
        self.session()?.execute_with_params(query, params)
    }

    /// Runs a query in the named language (`"gql"`, `"cypher"`, ...) on this
    /// graph.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph no longer exists, the language is not
    /// available or the query fails.
    pub fn execute_language(
        &self,
        query: &str,
        language: &str,
        params: Option<HashMap<String, Value>>,
    ) -> Result<QueryResult> {
        self.session()?.execute_language(query, language, params)
    }

    fn check_exists(&self) -> Result<()> {
        let exists = match graph_storage_key(self.schema.as_deref(), Some(&self.name)) {
            // The default graph always exists.
            None => true,
            Some(_) if self.name.eq_ignore_ascii_case("default") => true,
            Some(key) => self
                .db
                .store
                .as_ref()
                .is_some_and(|store| store.graph(&key).is_some()),
        };
        if exists {
            Ok(())
        } else {
            Err(missing_graph(&self.name))
        }
    }

    fn direct(&self) -> DirectCalls<'_> {
        self.db.direct(DirectTarget::Named {
            schema: self.schema.as_deref(),
            name: &self.name,
        })
    }

    // === Direct API: the database's calls, in this graph ===

    /// Creates a node in this graph, like [`GrafeoDB::create_node`].
    ///
    /// # Errors
    ///
    /// Returns an error if the graph no longer exists or the node violates
    /// the schema or a constraint.
    pub fn create_node(&self, labels: &[&str]) -> Result<NodeId> {
        self.create_node_with_props(labels, std::iter::empty::<(PropertyKey, Value)>())
    }

    /// Creates a node with properties in this graph, like
    /// [`GrafeoDB::create_node_with_props`].
    ///
    /// # Errors
    ///
    /// Returns an error if the graph no longer exists or the node violates
    /// the schema or a constraint.
    pub fn create_node_with_props(
        &self,
        labels: &[&str],
        properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>,
    ) -> Result<NodeId> {
        self.direct().create_node_with_props(labels, properties)
    }

    /// Creates an edge in this graph, like [`GrafeoDB::create_edge`].
    ///
    /// # Errors
    ///
    /// Returns an error if the graph no longer exists, an endpoint does not
    /// exist or the edge violates the schema.
    pub fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> Result<EdgeId> {
        self.create_edge_with_props(
            src,
            dst,
            edge_type,
            std::iter::empty::<(PropertyKey, Value)>(),
        )
    }

    /// Creates an edge with properties in this graph, like
    /// [`GrafeoDB::create_edge_with_props`].
    ///
    /// # Errors
    ///
    /// Returns an error if the graph no longer exists, an endpoint does not
    /// exist or the edge violates the schema.
    pub fn create_edge_with_props(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>,
    ) -> Result<EdgeId> {
        self.direct()
            .create_edge_with_props(src, dst, edge_type, properties)
    }

    /// Sets a node property, like [`GrafeoDB::set_node_property`].
    ///
    /// # Errors
    ///
    /// Returns an error if the graph or the node does not exist, or the
    /// value violates a constraint.
    pub fn set_node_property(&self, id: NodeId, key: &str, value: Value) -> Result<()> {
        self.direct().set_node_property(id, key, value)
    }

    /// Sets an edge property, like [`GrafeoDB::set_edge_property`].
    ///
    /// # Errors
    ///
    /// Returns an error if the graph or the edge does not exist, or the
    /// value violates the edge's type.
    pub fn set_edge_property(&self, id: EdgeId, key: &str, value: Value) -> Result<()> {
        self.direct().set_edge_property(id, key, value)
    }

    /// Removes a node property; whether the node had it.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph no longer exists or a constraint
    /// requires the property.
    pub fn remove_node_property(&self, id: NodeId, key: &str) -> Result<bool> {
        self.direct().remove_node_property(id, key)
    }

    /// Removes an edge property; whether the edge had it.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph no longer exists or the edge's type
    /// requires the property.
    pub fn remove_edge_property(&self, id: EdgeId, key: &str) -> Result<bool> {
        self.direct().remove_edge_property(id, key)
    }

    /// Adds a label; whether the node exists and lacked it.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph no longer exists or the node violates a
    /// constraint of the label.
    pub fn add_node_label(&self, id: NodeId, label: &str) -> Result<bool> {
        self.direct().add_node_label(id, label)
    }

    /// Removes a label; whether the node exists and had it.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph no longer exists.
    pub fn remove_node_label(&self, id: NodeId, label: &str) -> Result<bool> {
        self.direct().remove_node_label(id, label)
    }

    /// Deletes a node without edges; whether it existed.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph no longer exists or the node still has
    /// edges.
    pub fn delete_node(&self, id: NodeId) -> Result<bool> {
        self.direct().delete_node(id)
    }

    /// Deletes an edge; whether it existed.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph no longer exists.
    pub fn delete_edge(&self, id: EdgeId) -> Result<bool> {
        self.direct().delete_edge(id)
    }

    /// Creates one node with `label` per vector, all or none, like
    /// [`GrafeoDB::batch_create_nodes`].
    ///
    /// # Errors
    ///
    /// Returns an error if the graph no longer exists, or the first node's
    /// error; nothing of the batch is created then.
    pub fn batch_create_nodes(
        &self,
        label: &str,
        property: &str,
        vectors: Vec<Vec<f32>>,
    ) -> Result<Vec<NodeId>> {
        self.direct().batch_create_nodes(label, property, vectors)
    }

    /// Creates one node with `label` per property map, all or none, like
    /// [`GrafeoDB::batch_create_nodes_with_props`].
    ///
    /// # Errors
    ///
    /// Returns an error if the graph no longer exists, or the first node's
    /// error; nothing of the batch is created then.
    pub fn batch_create_nodes_with_props(
        &self,
        label: &str,
        properties_list: Vec<HashMap<PropertyKey, Value>>,
    ) -> Result<Vec<NodeId>> {
        self.batch_create_nodes_with_labels(&[label], properties_list)
    }

    /// Creates one node with all of `labels` per property map, all or none,
    /// like [`GrafeoDB::batch_create_nodes_with_labels`].
    ///
    /// # Errors
    ///
    /// Returns an error if the graph no longer exists, or the first node's
    /// error; nothing of the batch is created then.
    pub fn batch_create_nodes_with_labels(
        &self,
        labels: &[&str],
        properties_list: Vec<HashMap<PropertyKey, Value>>,
    ) -> Result<Vec<NodeId>> {
        self.direct()
            .batch_create_nodes_with_labels(labels, properties_list)
    }

    /// Creates the edges, all or none, like [`GrafeoDB::batch_create_edges`].
    ///
    /// # Errors
    ///
    /// Returns an error if the graph no longer exists, or the first edge's
    /// error; nothing of the batch is created then.
    pub fn batch_create_edges(&self, edges: Vec<BatchEdge>) -> Result<Vec<EdgeId>> {
        self.direct().batch_create_edges(edges)
    }

    /// Gets a node of this graph.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph no longer exists.
    pub fn get_node(&self, id: NodeId) -> Result<Option<Node>> {
        self.direct().get_node(id)
    }

    /// Gets an edge of this graph.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph no longer exists.
    pub fn get_edge(&self, id: EdgeId) -> Result<Option<Edge>> {
        self.direct().get_edge(id)
    }
}
