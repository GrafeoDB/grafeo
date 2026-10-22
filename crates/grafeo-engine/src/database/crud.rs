//! The direct node and edge API of GrafeoDB.
//!
//! Every write goes to the current graph (the one
//! [`set_current_graph`](super::GrafeoDB::set_current_graph) and
//! [`set_current_schema`](super::GrafeoDB::set_current_schema) select) and
//! commits at an epoch of its own: it is checked against the schema and
//! constraints, logged to the WAL and reported to CDC like the same write in a
//! query, and it either applies completely or fails with an error (see
//! [`direct`](super::direct) for how). Reads see the current graph as queries
//! do (on an external store too), and nothing when the
//! selected graph no longer exists. Like queries, they read at the last epoch
//! whose commit is complete, so they never see part of a commit.

use std::collections::HashMap;
use std::sync::Arc;

use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, Value};
use grafeo_common::utils::error::Result;
use grafeo_core::graph::lpg::{Edge, LpgStore, Node};

use super::direct::{BatchEdge, DirectTarget};

impl super::GrafeoDB {
    /// The store of the current graph: the one `set_current_graph` and
    /// `set_current_schema` select, or the default graph when they select
    /// none (or one that no longer exists).
    pub(crate) fn current_lpg_store(&self) -> Arc<LpgStore> {
        crate::session::graph_storage_key(
            self.current_schema.read().as_deref(),
            self.current_graph.read().as_deref(),
        )
        .and_then(|key| self.lpg_store().graph(&key))
        .unwrap_or_else(|| self.lpg_store())
    }

    /// The epoch the direct reads see: the last one whose commit is
    /// complete, as for queries. A store's own epoch moves when a commit
    /// stamps its versions, before its WAL records are written and the
    /// commit is complete (or when it never completes, see
    /// [`TransactionManager`](crate::transaction::TransactionManager)), so a
    /// read at it could see part of a commit.
    pub(crate) fn read_epoch(&self) -> EpochId {
        self.transaction_manager.current_epoch()
    }

    // === Node Operations ===

    /// Creates a node with the given labels and returns its ID.
    ///
    /// Labels categorize nodes - think of them like tags. A node can have
    /// multiple labels (e.g., `["Person", "Employee"]`).
    ///
    /// # Errors
    ///
    /// Returns an error if the node violates the schema, for example a label
    /// a closed graph type does not allow or a `NOT NULL` property it lacks.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let alix = db.create_node(&["Person"])?;
    /// let company = db.create_node(&["Company", "Startup"])?;
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    pub fn create_node(&self, labels: &[&str]) -> Result<NodeId> {
        self.create_node_with_props(labels, std::iter::empty::<(PropertyKey, Value)>())
    }

    /// Creates a node with labels and properties and returns its ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the node violates the schema or a constraint, for
    /// example a `UNIQUE` value another node already has.
    pub fn create_node_with_props(
        &self,
        labels: &[&str],
        properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>,
    ) -> Result<NodeId> {
        self.direct(DirectTarget::Current)
            .create_node_with_props(labels, properties)
    }

    /// Gets a node by ID, as of the current epoch.
    #[must_use]
    pub fn get_node(&self, id: NodeId) -> Option<Node> {
        let epoch = self.read_epoch();
        self.read_store(DirectTarget::Current)
            .ok()?
            .get_node_at_epoch(id, epoch)
    }

    /// Gets a node as it existed at a specific epoch.
    ///
    /// Uses pure epoch-based visibility (not transaction-aware), so the node
    /// is visible if and only if `created_epoch <= epoch` and it was not
    /// deleted at or before `epoch`. An epoch after the
    /// [current one](Self::current_epoch) reads at the current one.
    #[must_use]
    pub fn get_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> Option<Node> {
        let epoch = epoch.min(self.read_epoch());
        self.read_store(DirectTarget::Current)
            .ok()?
            .get_node_at_epoch(id, epoch)
    }

    /// Gets an edge as it existed at a specific epoch.
    ///
    /// Uses pure epoch-based visibility (not transaction-aware). An epoch
    /// after the [current one](Self::current_epoch) reads at the current one.
    #[must_use]
    pub fn get_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> Option<Edge> {
        let epoch = epoch.min(self.read_epoch());
        self.read_store(DirectTarget::Current)
            .ok()?
            .get_edge_at_epoch(id, epoch)
    }

    /// Returns all versions of a node with their creation/deletion epochs,
    /// up to the current epoch.
    ///
    /// Properties and labels reflect the current state (not versioned per-epoch).
    #[must_use]
    pub fn get_node_history(&self, id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        let epoch = self.read_epoch();
        self.read_store(DirectTarget::Current)
            .map_or_else(|_| Vec::new(), |store| store.get_node_history(id))
            .into_iter()
            .filter_map(|version| history_at(version, epoch))
            .collect()
    }

    /// Returns all versions of an edge with their creation/deletion epochs,
    /// up to the current epoch.
    ///
    /// Properties reflect the current state (not versioned per-epoch).
    #[must_use]
    pub fn get_edge_history(&self, id: EdgeId) -> Vec<(EpochId, Option<EpochId>, Edge)> {
        let epoch = self.read_epoch();
        self.read_store(DirectTarget::Current)
            .map_or_else(|_| Vec::new(), |store| store.get_edge_history(id))
            .into_iter()
            .filter_map(|version| history_at(version, epoch))
            .collect()
    }

    /// Returns a property value as it existed at a specific epoch.
    ///
    /// Uses the internal `VersionLog` to do a point-in-time read. Returns
    /// `None` if the property didn't exist or was deleted at that epoch. An
    /// epoch after the [current one](Self::current_epoch) reads at the
    /// current one.
    #[cfg(feature = "temporal")]
    #[must_use]
    pub fn get_node_property_at_epoch(
        &self,
        id: NodeId,
        key: &str,
        epoch: EpochId,
    ) -> Option<Value> {
        let epoch = epoch.min(self.read_epoch());
        self.current_lpg_store()
            .get_node_property_at_epoch(id, &PropertyKey::new(key), epoch)
    }

    /// Returns the version timeline for a single property of a node, up to
    /// the current epoch.
    ///
    /// Each entry is `(epoch, value)` in ascending epoch order. Tombstones
    /// (deletions) appear as `Value::Null`.
    #[cfg(feature = "temporal")]
    #[must_use]
    pub fn get_node_property_history(&self, id: NodeId, key: &str) -> Vec<(EpochId, Value)> {
        let epoch = self.read_epoch();
        let mut history = self
            .current_lpg_store()
            .node_property_history_for_key(id, key);
        history.retain(|(at, _)| *at <= epoch);
        history
    }

    /// Returns the version history for ALL properties of a node, up to the
    /// current epoch.
    ///
    /// Each entry is `(property_key, Vec<(epoch, value)>)`; a property whose
    /// versions are all later is left out.
    #[cfg(feature = "temporal")]
    #[must_use]
    pub fn get_all_node_property_history(
        &self,
        id: NodeId,
    ) -> Vec<(PropertyKey, Vec<(EpochId, Value)>)> {
        let epoch = self.read_epoch();
        let mut history = self.current_lpg_store().node_property_history(id);
        for (_, versions) in &mut history {
            versions.retain(|(at, _)| *at <= epoch);
        }
        history.retain(|(_, versions)| !versions.is_empty());
        history
    }

    /// Returns the current epoch of the database: the last one whose commit
    /// is complete.
    ///
    /// Every committed write advances it, from a query or from the direct
    /// API, once the commit is complete. A reopen continues where the
    /// database left off: at the epoch of the file's last checkpoint or of
    /// the last commit its WAL logs (after a crash), whichever is higher. With
    /// an external store, the epochs start from the store's own when it is
    /// opened.
    #[must_use]
    pub fn current_epoch(&self) -> EpochId {
        self.read_epoch()
    }

    /// Deletes a node and returns whether it existed.
    ///
    /// A node that still has edges is not deleted: delete them first with
    /// [`delete_edge`](Self::delete_edge), or use `DETACH DELETE` in a query.
    ///
    /// # Errors
    ///
    /// Returns an error if the node still has edges.
    pub fn delete_node(&self, id: NodeId) -> Result<bool> {
        self.direct(DirectTarget::Current).delete_node(id)
    }

    /// Sets a property on a node.
    ///
    /// # Errors
    ///
    /// Returns an error if the node does not exist or the value violates a
    /// constraint of its labels (type, `NOT NULL`, `UNIQUE`, vector index size).
    pub fn set_node_property(&self, id: NodeId, key: &str, value: Value) -> Result<()> {
        self.direct(DirectTarget::Current)
            .set_node_property(id, key, value)
    }

    /// Adds a label to an existing node.
    ///
    /// Returns `true` if the label was added, `false` if the node doesn't exist
    /// or already has the label.
    ///
    /// # Errors
    ///
    /// Returns an error if the node violates a constraint of the new label.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let alix = db.create_node(&["Person"])?;
    ///
    /// // Promote Alix to Employee
    /// assert!(db.add_node_label(alix, "Employee")?);
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    pub fn add_node_label(&self, id: NodeId, label: &str) -> Result<bool> {
        self.direct(DirectTarget::Current).add_node_label(id, label)
    }

    /// Removes a label from a node.
    ///
    /// Returns `true` if the label was removed, `false` if the node doesn't exist
    /// or doesn't have the label.
    ///
    /// # Errors
    ///
    /// Returns an error if another transaction is writing the node.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let alix = db.create_node(&["Person", "Employee"])?;
    ///
    /// // Remove Employee status
    /// assert!(db.remove_node_label(alix, "Employee")?);
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    pub fn remove_node_label(&self, id: NodeId, label: &str) -> Result<bool> {
        self.direct(DirectTarget::Current)
            .remove_node_label(id, label)
    }

    /// Gets all labels for a node.
    ///
    /// Returns `None` if the node doesn't exist.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let alix = db.create_node(&["Person", "Employee"])?;
    ///
    /// let labels = db.get_node_labels(alix).unwrap();
    /// assert!(labels.contains(&"Person".to_string()));
    /// assert!(labels.contains(&"Employee".to_string()));
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    #[must_use]
    pub fn get_node_labels(&self, id: NodeId) -> Option<Vec<String>> {
        self.get_node(id)
            .map(|node| node.labels.iter().map(|s| s.to_string()).collect())
    }

    // === Edge Operations ===

    /// Creates an edge (relationship) between two nodes.
    ///
    /// Edges connect nodes and have a type that describes the relationship.
    /// They're directed - the order of `src` and `dst` matters.
    ///
    /// # Errors
    ///
    /// Returns an error if an endpoint does not exist or the edge violates the
    /// schema (its type or its endpoints' labels).
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let alix = db.create_node(&["Person"])?;
    /// let gus = db.create_node(&["Person"])?;
    ///
    /// // Alix knows Gus (directed: Alix -> Gus)
    /// let edge = db.create_edge(alix, gus, "KNOWS")?;
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    pub fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> Result<EdgeId> {
        self.create_edge_with_props(
            src,
            dst,
            edge_type,
            std::iter::empty::<(PropertyKey, Value)>(),
        )
    }

    /// Creates an edge with properties.
    ///
    /// # Errors
    ///
    /// Returns an error if an endpoint does not exist or the edge violates the
    /// schema.
    pub fn create_edge_with_props(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>,
    ) -> Result<EdgeId> {
        self.direct(DirectTarget::Current)
            .create_edge_with_props(src, dst, edge_type, properties)
    }

    /// Gets an edge by ID, as of the current epoch.
    #[must_use]
    pub fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        let epoch = self.read_epoch();
        self.read_store(DirectTarget::Current)
            .ok()?
            .get_edge_at_epoch(id, epoch)
    }

    /// Deletes an edge and returns whether it existed.
    ///
    /// # Errors
    ///
    /// Returns an error if another transaction is writing the edge.
    pub fn delete_edge(&self, id: EdgeId) -> Result<bool> {
        self.direct(DirectTarget::Current).delete_edge(id)
    }

    /// Sets a property on an edge.
    ///
    /// # Errors
    ///
    /// Returns an error if the edge does not exist or the value violates its
    /// type.
    pub fn set_edge_property(&self, id: EdgeId, key: &str, value: Value) -> Result<()> {
        self.direct(DirectTarget::Current)
            .set_edge_property(id, key, value)
    }

    /// Removes a property from a node.
    ///
    /// Returns true if the property existed and was removed, false otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error if a constraint requires the property (`NOT NULL`,
    /// `NODE KEY`).
    pub fn remove_node_property(&self, id: NodeId, key: &str) -> Result<bool> {
        self.direct(DirectTarget::Current)
            .remove_node_property(id, key)
    }

    /// Removes a property from an edge.
    ///
    /// Returns true if the property existed and was removed, false otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error if the edge's type requires the property.
    pub fn remove_edge_property(&self, id: EdgeId, key: &str) -> Result<bool> {
        self.direct(DirectTarget::Current)
            .remove_edge_property(id, key)
    }

    /// Creates multiple nodes in bulk, each with a single vector property.
    ///
    /// The batch is one transaction: it is created and recovered completely
    /// or not at all, and other readers see all of it or none of it.
    ///
    /// # Arguments
    ///
    /// * `label` - Label applied to all created nodes
    /// * `property` - Property name for the vector data
    /// * `vectors` - Vector data for each node
    ///
    /// # Returns
    ///
    /// Vector of created `NodeId`s in the same order as the input vectors.
    ///
    /// # Errors
    ///
    /// Returns the first node's error (for example a vector of another size
    /// than the property's vector index); nothing of the batch is created then.
    pub fn batch_create_nodes(
        &self,
        label: &str,
        property: &str,
        vectors: Vec<Vec<f32>>,
    ) -> Result<Vec<NodeId>> {
        self.direct(DirectTarget::Current)
            .batch_create_nodes(label, property, vectors)
    }

    /// Batch-creates nodes with full property maps.
    ///
    /// Each entry in `properties_list` is a complete property map for one node.
    /// The batch is one transaction: it is created and recovered completely
    /// or not at all, and other readers see all of it or none of it.
    ///
    /// # Arguments
    ///
    /// * `label` - Label for all created nodes.
    /// * `properties_list` - One property map per node to create.
    ///
    /// # Returns
    ///
    /// Vector of created `NodeId`s in the same order as the input.
    ///
    /// # Errors
    ///
    /// Returns the first node's error (for example a `UNIQUE` value that
    /// another node, or an earlier node of the batch, already has); nothing of
    /// the batch is created then.
    pub fn batch_create_nodes_with_props(
        &self,
        label: &str,
        properties_list: Vec<HashMap<PropertyKey, Value>>,
    ) -> Result<Vec<NodeId>> {
        self.batch_create_nodes_with_labels(&[label], properties_list)
    }

    /// Batch-creates nodes with full property maps, each with all of
    /// `labels` (for example a namespace label and a type label).
    ///
    /// The batch is one transaction: it is created and recovered completely
    /// or not at all, and other readers see all of it or none of it.
    ///
    /// # Errors
    ///
    /// Returns the first node's error; nothing of the batch is created then.
    pub fn batch_create_nodes_with_labels(
        &self,
        labels: &[&str],
        properties_list: Vec<HashMap<PropertyKey, Value>>,
    ) -> Result<Vec<NodeId>> {
        self.direct(DirectTarget::Current)
            .batch_create_nodes_with_labels(labels, properties_list)
    }

    /// Batch-creates edges, each with its own endpoints, type and
    /// properties, in one transaction: created and recovered completely or
    /// not at all. Returns the IDs in input order.
    ///
    /// # Errors
    ///
    /// Returns the first edge's error (an endpoint that does not exist, or a
    /// schema violation); nothing of the batch is created then.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    /// use grafeo_engine::database::BatchEdge;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let alix = db.create_node(&["Person"])?;
    /// let gus = db.create_node(&["Person"])?;
    /// let ids = db.batch_create_edges(vec![
    ///     BatchEdge::new(alix, gus, "KNOWS").with_properties([("since", 2020_i64)]),
    ///     BatchEdge::new(gus, alix, "KNOWS"),
    /// ])?;
    /// assert_eq!(ids.len(), 2);
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    pub fn batch_create_edges(&self, edges: Vec<BatchEdge>) -> Result<Vec<EdgeId>> {
        self.direct(DirectTarget::Current).batch_create_edges(edges)
    }
}

/// A version of a history as a reader at `epoch` sees it: none when it was
/// created later, and not yet deleted when its deletion is later.
fn history_at<T>(
    (created, deleted, entity): (EpochId, Option<EpochId>, T),
    epoch: EpochId,
) -> Option<(EpochId, Option<EpochId>, T)> {
    (created <= epoch).then(|| (created, deleted.filter(|deleted| *deleted <= epoch), entity))
}
