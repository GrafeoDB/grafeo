//! Node and edge CRUD operations for GrafeoDB.

#[cfg(feature = "wal")]
use grafeo_common::grafeo_warn;
use grafeo_common::utils::error::Result;
#[cfg(feature = "wal")]
use grafeo_storage::wal::WalRecord;

impl super::GrafeoDB {
    // === Node Operations ===

    /// Creates a node with the given labels and returns its ID.
    ///
    /// Labels categorize nodes - think of them like tags. A node can have
    /// multiple labels (e.g., `["Person", "Employee"]`).
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let alix = db.create_node(&["Person"]);
    /// let company = db.create_node(&["Company", "Startup"]);
    /// ```
    pub fn create_node(&self, labels: &[&str]) -> Result<grafeo_common::types::NodeId> {
        let id = self.lpg_store().create_node(labels);

        // Log to WAL if enabled
        #[cfg(feature = "wal")]
        self.log_wal(&WalRecord::CreateNode {
            id,
            labels: labels.iter().map(|s| (*s).to_string()).collect(),
        })?;

        #[cfg(feature = "cdc")]
        if self.cdc_active() {
            self.cdc_log.record_create_node(
                id,
                self.lpg_store().current_epoch(),
                None,
                Some(labels.iter().map(|s| (*s).to_string()).collect()),
            );
        }

        Ok(id)
    }

    /// Creates a new node with labels and properties.
    ///
    /// If WAL is enabled, the operation is logged for durability.
    pub fn create_node_with_props(
        &self,
        labels: &[&str],
        properties: impl IntoIterator<
            Item = (
                impl Into<grafeo_common::types::PropertyKey>,
                impl Into<grafeo_common::types::Value>,
            ),
        >,
    ) -> Result<grafeo_common::types::NodeId> {
        // Collect properties first so we can log them to WAL
        let props: Vec<(
            grafeo_common::types::PropertyKey,
            grafeo_common::types::Value,
        )> = properties
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();

        let id = self
            .lpg_store()
            .create_node_with_props(labels, props.iter().map(|(k, v)| (k.clone(), v.clone())));

        // Build CDC snapshot before WAL consumes props
        #[cfg(feature = "cdc")]
        let cdc_props: Option<
            std::collections::HashMap<String, grafeo_common::types::Value>,
        > = if self.cdc_active() {
            Some(
                props
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.clone()))
                    .collect(),
            )
        } else {
            None
        };

        // Log node creation to WAL
        #[cfg(feature = "wal")]
        {
            self.log_wal(&WalRecord::CreateNode {
                id,
                labels: labels.iter().map(|s| (*s).to_string()).collect(),
            })?;

            // Log each property to WAL for full durability
            for (key, value) in props {
                self.log_wal(&WalRecord::SetNodeProperty {
                    id,
                    key: key.to_string(),
                    value,
                })?;
            }
        }

        #[cfg(feature = "cdc")]
        if let Some(cdc_props) = cdc_props {
            self.cdc_log.record_create_node(
                id,
                self.lpg_store().current_epoch(),
                if cdc_props.is_empty() {
                    None
                } else {
                    Some(cdc_props)
                },
                Some(labels.iter().map(|s| (*s).to_string()).collect()),
            );
        }

        // Auto-insert into matching text indexes for the new node
        #[cfg(feature = "text-index")]
        if let Some(node) = self.lpg_store().get_node(id) {
            for label in &node.labels {
                for (prop_key, prop_val) in &node.properties {
                    if let grafeo_common::types::Value::String(text) = prop_val
                        && let Some(index) = self
                            .lpg_store()
                            .get_text_index(label.as_str(), prop_key.as_ref())
                    {
                        index.write().insert(id, text);
                    }
                }
            }
        }

        Ok(id)
    }

    /// Gets a node by ID.
    #[must_use]
    pub fn get_node(
        &self,
        id: grafeo_common::types::NodeId,
    ) -> Option<grafeo_core::graph::lpg::Node> {
        self.lpg_store().get_node(id)
    }

    /// Gets a node as it existed at a specific epoch.
    ///
    /// Uses pure epoch-based visibility (not transaction-aware), so the node
    /// is visible if and only if `created_epoch <= epoch` and it was not
    /// deleted at or before `epoch`.
    #[must_use]
    pub fn get_node_at_epoch(
        &self,
        id: grafeo_common::types::NodeId,
        epoch: grafeo_common::types::EpochId,
    ) -> Option<grafeo_core::graph::lpg::Node> {
        self.lpg_store().get_node_at_epoch(id, epoch)
    }

    /// Gets an edge as it existed at a specific epoch.
    ///
    /// Uses pure epoch-based visibility (not transaction-aware).
    #[must_use]
    pub fn get_edge_at_epoch(
        &self,
        id: grafeo_common::types::EdgeId,
        epoch: grafeo_common::types::EpochId,
    ) -> Option<grafeo_core::graph::lpg::Edge> {
        self.lpg_store().get_edge_at_epoch(id, epoch)
    }

    /// Returns all versions of a node with their creation/deletion epochs.
    ///
    /// Properties and labels reflect the current state (not versioned per-epoch).
    #[must_use]
    pub fn get_node_history(
        &self,
        id: grafeo_common::types::NodeId,
    ) -> Vec<(
        grafeo_common::types::EpochId,
        Option<grafeo_common::types::EpochId>,
        grafeo_core::graph::lpg::Node,
    )> {
        self.lpg_store().get_node_history(id)
    }

    /// Returns all versions of an edge with their creation/deletion epochs.
    ///
    /// Properties reflect the current state (not versioned per-epoch).
    #[must_use]
    pub fn get_edge_history(
        &self,
        id: grafeo_common::types::EdgeId,
    ) -> Vec<(
        grafeo_common::types::EpochId,
        Option<grafeo_common::types::EpochId>,
        grafeo_core::graph::lpg::Edge,
    )> {
        self.lpg_store().get_edge_history(id)
    }

    /// Returns a property value as it existed at a specific epoch.
    ///
    /// Uses the internal `VersionLog` to do a point-in-time read. Returns
    /// `None` if the property didn't exist or was deleted at that epoch.
    #[cfg(feature = "temporal")]
    #[must_use]
    pub fn get_node_property_at_epoch(
        &self,
        id: grafeo_common::types::NodeId,
        key: &str,
        epoch: grafeo_common::types::EpochId,
    ) -> Option<grafeo_common::types::Value> {
        let prop_key = grafeo_common::types::PropertyKey::new(key);
        self.lpg_store()
            .get_node_property_at_epoch(id, &prop_key, epoch)
    }

    /// Returns the full version timeline for a single property of a node.
    ///
    /// Each entry is `(epoch, value)` in ascending epoch order. Tombstones
    /// (deletions) appear as `Value::Null`.
    #[cfg(feature = "temporal")]
    #[must_use]
    pub fn get_node_property_history(
        &self,
        id: grafeo_common::types::NodeId,
        key: &str,
    ) -> Vec<(grafeo_common::types::EpochId, grafeo_common::types::Value)> {
        self.lpg_store().node_property_history_for_key(id, key)
    }

    /// Returns the full version history for ALL properties of a node.
    ///
    /// Each entry is `(property_key, Vec<(epoch, value)>)`.
    #[cfg(feature = "temporal")]
    #[must_use]
    pub fn get_all_node_property_history(
        &self,
        id: grafeo_common::types::NodeId,
    ) -> Vec<(
        grafeo_common::types::PropertyKey,
        Vec<(grafeo_common::types::EpochId, grafeo_common::types::Value)>,
    )> {
        self.lpg_store().node_property_history(id)
    }

    /// Returns the current epoch of the database.
    #[must_use]
    pub fn current_epoch(&self) -> grafeo_common::types::EpochId {
        self.lpg_store().current_epoch()
    }

    /// Deletes a node and all its edges.
    ///
    /// If WAL is enabled, the operation is logged for durability.
    pub fn delete_node(&self, id: grafeo_common::types::NodeId) -> Result<bool> {
        // Capture properties for CDC before deletion
        #[cfg(feature = "cdc")]
        let cdc_props = if self.cdc_active() {
            self.lpg_store().get_node(id).map(|node| {
                node.properties
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.clone()))
                    .collect::<std::collections::HashMap<String, grafeo_common::types::Value>>()
            })
        } else {
            None
        };

        // Collect matching vector indexes BEFORE deletion removes labels
        #[cfg(feature = "vector-index")]
        let indexes_to_clean: Vec<
            std::sync::Arc<grafeo_core::index::vector::VectorIndexKind>,
        > = self
            .lpg_store()
            .get_node(id)
            .map(|node| {
                let mut indexes = Vec::new();
                for label in &node.labels {
                    let prefix = format!("{}:", label.as_str());
                    for (key, index) in self.lpg_store().vector_index_entries() {
                        if key.starts_with(&prefix) {
                            indexes.push(index);
                        }
                    }
                }
                indexes
            })
            .unwrap_or_default();

        // Collect matching text indexes BEFORE deletion removes labels
        #[cfg(feature = "text-index")]
        let text_indexes_to_clean: Vec<
            std::sync::Arc<parking_lot::RwLock<grafeo_core::index::text::InvertedIndex>>,
        > = self
            .lpg_store()
            .get_node(id)
            .map(|node| {
                let mut indexes = Vec::new();
                for label in &node.labels {
                    let prefix = format!("{}:", label.as_str());
                    for (key, index) in self.lpg_store().text_index_entries() {
                        if key.starts_with(&prefix) {
                            indexes.push(index);
                        }
                    }
                }
                indexes
            })
            .unwrap_or_default();

        let exists = self.lpg_store().get_node(id).is_some();

        #[cfg(feature = "wal")]
        if exists {
            self.log_wal(&WalRecord::DeleteNode { id })?;
        }

        let result = if exists {
            self.lpg_store().delete_node(id)
        } else {
            false
        };

        // Remove from vector indexes after successful deletion
        #[cfg(feature = "vector-index")]
        if result {
            for index in indexes_to_clean {
                index.remove(id);
            }
        }

        // Remove from text indexes after successful deletion
        #[cfg(feature = "text-index")]
        if result {
            for index in text_indexes_to_clean {
                index.write().remove(id);
            }
        }

        #[cfg(feature = "cdc")]
        if result && self.cdc_active() {
            self.cdc_log.record_delete(
                crate::cdc::EntityId::Node(id),
                self.lpg_store().current_epoch(),
                cdc_props,
            );
        }

        Ok(result)
    }

    /// Sets a property on a node.
    ///
    /// If WAL is enabled, the operation is logged for durability.
    pub fn set_node_property(
        &self,
        id: grafeo_common::types::NodeId,
        key: &str,
        value: grafeo_common::types::Value,
    ) -> Result<()> {
        // Extract vector data before the value is moved into the store
        #[cfg(feature = "vector-index")]
        let vector_data = match &value {
            grafeo_common::types::Value::Vector(v) => Some(v.clone()),
            _ => None,
        };

        // Log to WAL first
        #[cfg(feature = "wal")]
        self.log_wal(&WalRecord::SetNodeProperty {
            id,
            key: key.to_string(),
            value: value.clone(),
        })?;

        // Capture old value for CDC before the store write
        #[cfg(feature = "cdc")]
        let cdc_active = self.cdc_active();
        #[cfg(feature = "cdc")]
        let cdc_old_value = if cdc_active {
            self.lpg_store()
                .get_node_property(id, &grafeo_common::types::PropertyKey::new(key))
        } else {
            None
        };
        #[cfg(feature = "cdc")]
        let cdc_new_value = if cdc_active {
            Some(value.clone())
        } else {
            None
        };

        self.lpg_store().set_node_property(id, key, value);

        #[cfg(feature = "cdc")]
        if let Some(cdc_new_value) = cdc_new_value {
            self.cdc_log.record_update(
                crate::cdc::EntityId::Node(id),
                self.lpg_store().current_epoch(),
                key,
                cdc_old_value,
                cdc_new_value,
            );
        }

        // Auto-insert into matching vector indexes
        #[cfg(feature = "vector-index")]
        if let Some(vec) = vector_data
            && let Some(node) = self.lpg_store().get_node(id)
        {
            for label in &node.labels {
                if let Some(index) = self.lpg_store().get_vector_index(label.as_str(), key) {
                    let accessor = grafeo_core::index::vector::PropertyVectorAccessor::new(
                        &**self.lpg_store(),
                        key,
                    );
                    index.insert(id, &vec, &accessor);
                }
            }
        }

        // Auto-update matching text indexes
        #[cfg(feature = "text-index")]
        if let Some(node) = self.lpg_store().get_node(id) {
            let text_val = node
                .properties
                .get(&grafeo_common::types::PropertyKey::new(key))
                .and_then(|v| match v {
                    grafeo_common::types::Value::String(s) => Some(s.to_string()),
                    _ => None,
                });
            for label in &node.labels {
                if let Some(index) = self.lpg_store().get_text_index(label.as_str(), key) {
                    let mut idx = index.write();
                    if let Some(ref text) = text_val {
                        idx.insert(id, text);
                    } else {
                        idx.remove(id);
                    }
                }
            }
        }

        Ok(())
    }

    /// Adds a label to an existing node.
    ///
    /// Returns `true` if the label was added, `false` if the node doesn't exist
    /// or already has the label.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let alix = db.create_node(&["Person"]);
    ///
    /// // Promote Alix to Employee
    /// let added = db.add_node_label(alix, "Employee");
    /// assert!(added);
    /// ```
    pub fn add_node_label(&self, id: grafeo_common::types::NodeId, label: &str) -> bool {
        let result = self.lpg_store().add_label(id, label);

        #[cfg(feature = "wal")]
        if result {
            // Log to WAL if enabled
            if let Err(e) = self.log_wal(&WalRecord::AddNodeLabel {
                id,
                label: label.to_string(),
            }) {
                grafeo_warn!("Failed to log AddNodeLabel to WAL: {}", e);
            }
        }

        // Auto-insert into vector indexes for the newly-added label
        #[cfg(feature = "vector-index")]
        if result {
            let prefix = format!("{label}:");
            for (key, index) in self.lpg_store().vector_index_entries() {
                if let Some(property) = key.strip_prefix(&prefix)
                    && let Some(node) = self.lpg_store().get_node(id)
                {
                    let prop_key = grafeo_common::types::PropertyKey::new(property);
                    if let Some(grafeo_common::types::Value::Vector(v)) =
                        node.properties.get(&prop_key)
                    {
                        let accessor = grafeo_core::index::vector::PropertyVectorAccessor::new(
                            &**self.lpg_store(),
                            property,
                        );
                        index.insert(id, v, &accessor);
                    }
                }
            }
        }

        // Auto-insert into text indexes for the newly-added label
        #[cfg(feature = "text-index")]
        if result && let Some(node) = self.lpg_store().get_node(id) {
            for (prop_key, prop_val) in &node.properties {
                if let grafeo_common::types::Value::String(text) = prop_val
                    && let Some(index) = self.lpg_store().get_text_index(label, prop_key.as_ref())
                {
                    index.write().insert(id, text);
                }
            }
        }

        result
    }

    /// Removes a label from a node.
    ///
    /// Returns `true` if the label was removed, `false` if the node doesn't exist
    /// or doesn't have the label.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let alix = db.create_node(&["Person", "Employee"]);
    ///
    /// // Remove Employee status
    /// let removed = db.remove_node_label(alix, "Employee");
    /// assert!(removed);
    /// ```
    pub fn remove_node_label(&self, id: grafeo_common::types::NodeId, label: &str) -> bool {
        // Collect text indexes to clean BEFORE removing the label
        #[cfg(feature = "text-index")]
        let text_indexes_to_clean: Vec<
            std::sync::Arc<parking_lot::RwLock<grafeo_core::index::text::InvertedIndex>>,
        > = {
            let prefix = format!("{label}:");
            self.lpg_store()
                .text_index_entries()
                .into_iter()
                .filter(|(key, _)| key.starts_with(&prefix))
                .map(|(_, index)| index)
                .collect()
        };

        let result = self.lpg_store().remove_label(id, label);

        #[cfg(feature = "wal")]
        if result {
            // Log to WAL if enabled
            if let Err(e) = self.log_wal(&WalRecord::RemoveNodeLabel {
                id,
                label: label.to_string(),
            }) {
                grafeo_warn!("Failed to log RemoveNodeLabel to WAL: {}", e);
            }
        }

        // Remove from text indexes for the removed label
        #[cfg(feature = "text-index")]
        if result {
            for index in text_indexes_to_clean {
                index.write().remove(id);
            }
        }

        result
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
    /// let alix = db.create_node(&["Person", "Employee"]);
    ///
    /// let labels = db.get_node_labels(alix).unwrap();
    /// assert!(labels.contains(&"Person".to_string()));
    /// assert!(labels.contains(&"Employee".to_string()));
    /// ```
    #[must_use]
    pub fn get_node_labels(&self, id: grafeo_common::types::NodeId) -> Option<Vec<String>> {
        self.lpg_store()
            .get_node(id)
            .map(|node| node.labels.iter().map(|s| s.to_string()).collect())
    }

    // === Edge Operations ===

    /// Creates an edge (relationship) between two nodes.
    ///
    /// Edges connect nodes and have a type that describes the relationship.
    /// They're directed - the order of `src` and `dst` matters.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let alix = db.create_node(&["Person"]);
    /// let gus = db.create_node(&["Person"]);
    ///
    /// // Alix knows Gus (directed: Alix -> Gus)
    /// let edge = db.create_edge(alix, gus, "KNOWS");
    /// ```
    pub fn create_edge(
        &self,
        src: grafeo_common::types::NodeId,
        dst: grafeo_common::types::NodeId,
        edge_type: &str,
    ) -> grafeo_common::types::EdgeId {
        let id = self.lpg_store().create_edge(src, dst, edge_type);

        // Log to WAL if enabled
        #[cfg(feature = "wal")]
        if let Err(e) = self.log_wal(&WalRecord::CreateEdge {
            id,
            src,
            dst,
            edge_type: edge_type.to_string(),
        }) {
            grafeo_warn!("Failed to log CreateEdge to WAL: {}", e);
        }

        #[cfg(feature = "cdc")]
        if self.cdc_active() {
            self.cdc_log.record_create_edge(
                id,
                self.lpg_store().current_epoch(),
                None,
                src.as_u64(),
                dst.as_u64(),
                edge_type.to_string(),
            );
        }

        id
    }

    /// Creates a new edge with properties.
    ///
    /// If WAL is enabled, the operation is logged for durability.
    pub fn create_edge_with_props(
        &self,
        src: grafeo_common::types::NodeId,
        dst: grafeo_common::types::NodeId,
        edge_type: &str,
        properties: impl IntoIterator<
            Item = (
                impl Into<grafeo_common::types::PropertyKey>,
                impl Into<grafeo_common::types::Value>,
            ),
        >,
    ) -> grafeo_common::types::EdgeId {
        // Collect properties first so we can log them to WAL
        let props: Vec<(
            grafeo_common::types::PropertyKey,
            grafeo_common::types::Value,
        )> = properties
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();

        let id = self.lpg_store().create_edge_with_props(
            src,
            dst,
            edge_type,
            props.iter().map(|(k, v)| (k.clone(), v.clone())),
        );

        // Build CDC snapshot before WAL consumes props
        #[cfg(feature = "cdc")]
        let cdc_props: Option<
            std::collections::HashMap<String, grafeo_common::types::Value>,
        > = if self.cdc_active() {
            Some(
                props
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.clone()))
                    .collect(),
            )
        } else {
            None
        };

        // Log edge creation to WAL
        #[cfg(feature = "wal")]
        {
            if let Err(e) = self.log_wal(&WalRecord::CreateEdge {
                id,
                src,
                dst,
                edge_type: edge_type.to_string(),
            }) {
                grafeo_warn!("Failed to log CreateEdge to WAL: {}", e);
            }

            // Log each property to WAL for full durability
            for (key, value) in props {
                if let Err(e) = self.log_wal(&WalRecord::SetEdgeProperty {
                    id,
                    key: key.to_string(),
                    value,
                }) {
                    grafeo_warn!("Failed to log SetEdgeProperty to WAL: {}", e);
                }
            }
        }

        #[cfg(feature = "cdc")]
        if let Some(cdc_props) = cdc_props {
            self.cdc_log.record_create_edge(
                id,
                self.lpg_store().current_epoch(),
                if cdc_props.is_empty() {
                    None
                } else {
                    Some(cdc_props)
                },
                src.as_u64(),
                dst.as_u64(),
                edge_type.to_string(),
            );
        }

        id
    }

    /// Gets an edge by ID.
    #[must_use]
    pub fn get_edge(
        &self,
        id: grafeo_common::types::EdgeId,
    ) -> Option<grafeo_core::graph::lpg::Edge> {
        self.lpg_store().get_edge(id)
    }

    /// Deletes an edge.
    ///
    /// If WAL is enabled, the operation is logged for durability.
    pub fn delete_edge(&self, id: grafeo_common::types::EdgeId) -> bool {
        // Capture properties for CDC before deletion
        #[cfg(feature = "cdc")]
        let cdc_props = if self.cdc_active() {
            self.lpg_store().get_edge(id).map(|edge| {
                edge.properties
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.clone()))
                    .collect::<std::collections::HashMap<String, grafeo_common::types::Value>>()
            })
        } else {
            None
        };

        let result = self.lpg_store().delete_edge(id);

        #[cfg(feature = "wal")]
        if result && let Err(e) = self.log_wal(&WalRecord::DeleteEdge { id }) {
            grafeo_warn!("Failed to log DeleteEdge to WAL: {}", e);
        }

        #[cfg(feature = "cdc")]
        if result && self.cdc_active() {
            self.cdc_log.record_delete(
                crate::cdc::EntityId::Edge(id),
                self.lpg_store().current_epoch(),
                cdc_props,
            );
        }

        result
    }

    /// Sets a property on an edge.
    ///
    /// If WAL is enabled, the operation is logged for durability.
    pub fn set_edge_property(
        &self,
        id: grafeo_common::types::EdgeId,
        key: &str,
        value: grafeo_common::types::Value,
    ) {
        // Log to WAL first
        #[cfg(feature = "wal")]
        if let Err(e) = self.log_wal(&WalRecord::SetEdgeProperty {
            id,
            key: key.to_string(),
            value: value.clone(),
        }) {
            grafeo_warn!("Failed to log SetEdgeProperty to WAL: {}", e);
        }

        // Capture old value for CDC before the store write
        #[cfg(feature = "cdc")]
        let cdc_active = self.cdc_active();
        #[cfg(feature = "cdc")]
        let cdc_old_value = if cdc_active {
            self.lpg_store()
                .get_edge_property(id, &grafeo_common::types::PropertyKey::new(key))
        } else {
            None
        };
        #[cfg(feature = "cdc")]
        let cdc_new_value = if cdc_active {
            Some(value.clone())
        } else {
            None
        };

        self.lpg_store().set_edge_property(id, key, value);

        #[cfg(feature = "cdc")]
        if let Some(cdc_new_value) = cdc_new_value {
            self.cdc_log.record_update(
                crate::cdc::EntityId::Edge(id),
                self.lpg_store().current_epoch(),
                key,
                cdc_old_value,
                cdc_new_value,
            );
        }
    }

    /// Removes a property from a node.
    ///
    /// Returns true if the property existed and was removed, false otherwise.
    pub fn remove_node_property(&self, id: grafeo_common::types::NodeId, key: &str) -> bool {
        let removed = self.lpg_store().remove_node_property(id, key).is_some();

        #[cfg(feature = "wal")]
        if removed
            && let Err(e) = self.log_wal(&WalRecord::RemoveNodeProperty {
                id,
                key: key.to_string(),
            })
        {
            grafeo_warn!("WAL log for RemoveNodeProperty failed: {e}");
        }

        // Remove from matching text indexes
        #[cfg(feature = "text-index")]
        if removed && let Some(node) = self.lpg_store().get_node(id) {
            for label in &node.labels {
                if let Some(index) = self.lpg_store().get_text_index(label.as_str(), key) {
                    index.write().remove(id);
                }
            }
        }

        removed
    }

    /// Removes a property from an edge.
    ///
    /// Returns true if the property existed and was removed, false otherwise.
    pub fn remove_edge_property(&self, id: grafeo_common::types::EdgeId, key: &str) -> bool {
        let removed = self.lpg_store().remove_edge_property(id, key).is_some();

        #[cfg(feature = "wal")]
        if removed
            && let Err(e) = self.log_wal(&WalRecord::RemoveEdgeProperty {
                id,
                key: key.to_string(),
            })
        {
            grafeo_warn!("WAL log for RemoveEdgeProperty failed: {e}");
        }

        removed
    }

    /// Creates multiple nodes in bulk, each with a single vector property.
    ///
    /// Much faster than individual `create_node_with_props` calls because it
    /// acquires internal locks once and loops in Rust rather than crossing
    /// the FFI boundary per vector.
    ///
    /// **Atomicity note:** Individual node creations within the batch are NOT
    /// atomic as a group. If a failure occurs mid-batch (e.g. WAL write error),
    /// nodes created before the failure will persist while later nodes may not.
    /// If you need all-or-nothing semantics, wrap the call in an explicit
    /// transaction.
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
    pub fn batch_create_nodes(
        &self,
        label: &str,
        property: &str,
        vectors: Vec<Vec<f32>>,
    ) -> Vec<grafeo_common::types::NodeId> {
        use grafeo_common::types::{PropertyKey, Value};

        let prop_key = PropertyKey::new(property);
        let labels: &[&str] = &[label];

        let ids: Vec<grafeo_common::types::NodeId> = vectors
            .into_iter()
            .map(|vec| {
                let value = Value::Vector(vec.into());
                let id = self.lpg_store().create_node_with_props(
                    labels,
                    std::iter::once((prop_key.clone(), value.clone())),
                );

                // Log to WAL
                #[cfg(feature = "wal")]
                {
                    if let Err(e) = self.log_wal(&WalRecord::CreateNode {
                        id,
                        labels: labels.iter().map(|s| (*s).to_string()).collect(),
                    }) {
                        grafeo_warn!("Failed to log CreateNode to WAL: {}", e);
                    }
                    if let Err(e) = self.log_wal(&WalRecord::SetNodeProperty {
                        id,
                        key: property.to_string(),
                        value,
                    }) {
                        grafeo_warn!("Failed to log SetNodeProperty to WAL: {}", e);
                    }
                }

                id
            })
            .collect();

        // Auto-insert into matching vector index if one exists
        #[cfg(feature = "vector-index")]
        if let Some(index) = self.lpg_store().get_vector_index(label, property) {
            let accessor = grafeo_core::index::vector::PropertyVectorAccessor::new(
                &**self.lpg_store(),
                property,
            );
            for &id in &ids {
                if let Some(node) = self.lpg_store().get_node(id) {
                    let pk = grafeo_common::types::PropertyKey::new(property);
                    if let Some(grafeo_common::types::Value::Vector(v)) = node.properties.get(&pk)
                        && std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            index.insert(id, v, &accessor);
                        }))
                        .is_err()
                    {
                        grafeo_warn!("Vector index insert panicked for node {}", id.as_u64());
                    }
                }
            }
        }

        ids
    }

    /// Batch-creates nodes with full property maps.
    ///
    /// Each entry in `properties_list` is a complete property map for one node.
    /// Vector values (`Value::Vector`) are automatically inserted into matching
    /// vector indexes. Text values are automatically inserted into matching text
    /// indexes.
    ///
    /// **Atomicity note:** Individual node creations within the batch are NOT
    /// atomic as a group. If a failure occurs mid-batch (e.g. WAL write error),
    /// nodes created before the failure will persist while later nodes may not.
    /// If you need all-or-nothing semantics, wrap the call in an explicit
    /// transaction.
    ///
    /// # Arguments
    ///
    /// * `label` - Label for all created nodes.
    /// * `properties_list` - One property map per node to create.
    ///
    /// # Returns
    ///
    /// Vector of created `NodeId`s in the same order as the input.
    pub fn batch_create_nodes_with_props(
        &self,
        label: &str,
        properties_list: Vec<
            std::collections::HashMap<
                grafeo_common::types::PropertyKey,
                grafeo_common::types::Value,
            >,
        >,
    ) -> Vec<grafeo_common::types::NodeId> {
        #[cfg(any(feature = "vector-index", feature = "text-index"))]
        use grafeo_common::types::Value;

        let labels: &[&str] = &[label];

        let ids: Vec<grafeo_common::types::NodeId> = properties_list
            .into_iter()
            .map(|props| {
                let id = self.lpg_store().create_node_with_props(
                    labels,
                    props.iter().map(|(k, v)| (k.clone(), v.clone())),
                );

                // Build CDC snapshot before WAL consumes props
                #[cfg(feature = "cdc")]
                let cdc_props: Option<
                    std::collections::HashMap<String, grafeo_common::types::Value>,
                > = if self.cdc_active() {
                    Some(
                        props
                            .iter()
                            .map(|(k, v)| (k.to_string(), v.clone()))
                            .collect(),
                    )
                } else {
                    None
                };

                // Log to WAL
                #[cfg(feature = "wal")]
                {
                    if let Err(e) = self.log_wal(&WalRecord::CreateNode {
                        id,
                        labels: labels.iter().map(|s| (*s).to_string()).collect(),
                    }) {
                        grafeo_warn!("Failed to log CreateNode to WAL: {}", e);
                    }
                    for (key, value) in props {
                        if let Err(e) = self.log_wal(&WalRecord::SetNodeProperty {
                            id,
                            key: key.to_string(),
                            value,
                        }) {
                            grafeo_warn!("Failed to log SetNodeProperty to WAL: {}", e);
                        }
                    }
                }

                #[cfg(feature = "cdc")]
                if let Some(cdc_props) = cdc_props {
                    self.cdc_log.record_create_node(
                        id,
                        self.lpg_store().current_epoch(),
                        if cdc_props.is_empty() {
                            None
                        } else {
                            Some(cdc_props)
                        },
                        Some(labels.iter().map(|s| (*s).to_string()).collect()),
                    );
                }

                id
            })
            .collect();

        // Auto-insert into matching vector indexes for any vector properties
        #[cfg(feature = "vector-index")]
        {
            for (key, index) in self.lpg_store().vector_index_entries() {
                // key is "label:property"
                if !key.starts_with(label) || !key[label.len()..].starts_with(':') {
                    continue;
                }
                let property = &key[label.len() + 1..];
                let accessor = grafeo_core::index::vector::PropertyVectorAccessor::new(
                    &**self.lpg_store(),
                    property,
                );
                let pk = grafeo_common::types::PropertyKey::new(property);
                for &id in &ids {
                    if let Some(node) = self.lpg_store().get_node(id) {
                        // reason: guard would be side-effecting; keep insert in arm body
                        #[allow(clippy::collapsible_match)]
                        match node.properties.get(&pk) {
                            Some(Value::Vector(v)) => {
                                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    index.insert(id, v, &accessor);
                                }))
                                .is_err()
                                {
                                    grafeo_warn!(
                                        "Vector index insert panicked for node {}",
                                        id.as_u64()
                                    );
                                }
                            }
                            Some(_other) => {
                                grafeo_warn!(
                                    "Node {} property '{}' expected Vector, skipping vector index insert",
                                    id.as_u64(),
                                    property
                                );
                            }
                            None => {} // No property, nothing to index
                        }
                    }
                }
            }
        }

        // Auto-insert into matching text indexes for any string properties
        #[cfg(feature = "text-index")]
        for &id in &ids {
            if let Some(node) = self.lpg_store().get_node(id) {
                for (prop_key, prop_val) in &node.properties {
                    if let Value::String(text) = prop_val
                        && let Some(index) =
                            self.lpg_store().get_text_index(label, prop_key.as_ref())
                    {
                        index.write().insert(id, text);
                    }
                }
            }
        }

        ids
    }
}
