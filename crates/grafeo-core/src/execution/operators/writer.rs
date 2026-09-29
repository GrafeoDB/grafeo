//! One validated write path for graph mutations.
//!
//! [`GraphWriter`] is what every mutation goes through: the CREATE, SET,
//! REMOVE, DELETE and MERGE operators, and the direct API through the
//! session. For each write it records the entity for write-conflict
//! detection, checks the schema and constraints, and writes with the
//! transaction's versioning, so the rules for a valid write live in one place.

use std::sync::Arc;

use grafeo_common::types::{
    EdgeId, EpochId, NodeId, PropertyKey, PropertyMap, TransactionId, Value,
};

use super::{ConstraintValidator, OperatorError, SharedWriteTracker};
use crate::graph::lpg::{Edge, Node};
use crate::graph::{Direction, GraphStoreMut};

/// The property name of a map assignment: `SET n = {...}` or `SET n += {...}`
/// arrive as one `("*", map)` pair.
const MAP_ASSIGNMENT: &str = "*";

/// A node or an edge, for the writes both have.
#[derive(Clone, Copy)]
enum Entity {
    Node(NodeId),
    Edge(EdgeId),
}

/// Writes to a graph store for one statement or direct call: validated,
/// tracked for write conflicts and versioned by the transaction.
#[derive(Clone)]
pub struct GraphWriter {
    store: Arc<dyn GraphStoreMut>,
    viewing_epoch: Option<EpochId>,
    transaction_id: Option<TransactionId>,
    validator: Option<Arc<dyn ConstraintValidator>>,
    write_tracker: Option<SharedWriteTracker>,
}

impl From<Arc<dyn GraphStoreMut>> for GraphWriter {
    /// A writer without a transaction, validator or write tracker.
    fn from(store: Arc<dyn GraphStoreMut>) -> Self {
        Self::new(store)
    }
}

impl GraphWriter {
    /// Creates a writer without a transaction, validator or write tracker.
    pub fn new(store: Arc<dyn GraphStoreMut>) -> Self {
        Self {
            store,
            viewing_epoch: None,
            transaction_id: None,
            validator: None,
            write_tracker: None,
        }
    }

    /// Writes as `transaction_id` (versioned, undone on rollback), reading at
    /// `epoch`.
    #[must_use]
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.viewing_epoch = Some(epoch);
        self.transaction_id = transaction_id;
        self
    }

    /// Checks every write against the schema and constraints.
    #[must_use]
    pub fn with_validator(mut self, validator: Arc<dyn ConstraintValidator>) -> Self {
        self.validator = Some(validator);
        self
    }

    /// Records every written entity for write-conflict detection.
    #[must_use]
    pub fn with_write_tracker(mut self, tracker: SharedWriteTracker) -> Self {
        self.write_tracker = Some(tracker);
        self
    }

    /// The store written to.
    #[must_use]
    pub fn store(&self) -> &Arc<dyn GraphStoreMut> {
        &self.store
    }

    /// The epoch reads see, if a transaction context was given.
    #[must_use]
    pub fn viewing_epoch(&self) -> Option<EpochId> {
        self.viewing_epoch
    }

    /// The transaction written as, if any.
    #[must_use]
    pub fn transaction_id(&self) -> Option<TransactionId> {
        self.transaction_id
    }

    fn epoch(&self) -> EpochId {
        self.viewing_epoch
            .unwrap_or_else(|| self.store.current_epoch())
    }

    fn transaction(&self) -> TransactionId {
        self.transaction_id.unwrap_or(TransactionId::SYSTEM)
    }

    /// The node as this writer's transaction sees it, its own writes included.
    fn node(&self, id: NodeId) -> Option<Node> {
        match (self.viewing_epoch, self.transaction_id) {
            (Some(epoch), Some(transaction_id)) => {
                self.store.get_node_versioned(id, epoch, transaction_id)
            }
            _ => self.store.get_node(id),
        }
    }

    /// The edge as this writer's transaction sees it, its own writes included.
    fn edge(&self, id: EdgeId) -> Option<Edge> {
        match (self.viewing_epoch, self.transaction_id) {
            (Some(epoch), Some(transaction_id)) => {
                self.store.get_edge_versioned(id, epoch, transaction_id)
            }
            _ => self.store.get_edge(id),
        }
    }

    fn record(&self, entity: Entity) -> Result<(), OperatorError> {
        if let (Some(tracker), Some(transaction_id)) = (&self.write_tracker, self.transaction_id) {
            match entity {
                Entity::Node(id) => tracker.record_node_write(transaction_id, id)?,
                Entity::Edge(id) => tracker.record_edge_write(transaction_id, id)?,
            }
        }
        Ok(())
    }

    // === Nodes ===

    /// Creates a node after checking it against the schema: allowed labels,
    /// type defaults, property types, NOT NULL, UNIQUE and NODE KEY.
    ///
    /// # Errors
    ///
    /// Returns the first constraint the node would violate; nothing is written then.
    pub fn create_node(
        &self,
        labels: &[String],
        mut properties: Vec<(String, Value)>,
    ) -> Result<NodeId, OperatorError> {
        if let Some(validator) = &self.validator {
            validator.validate_node_labels_allowed(labels)?;
            validator.inject_defaults(labels, &mut properties);
            self.check_node_values(validator.as_ref(), labels, &properties, None)?;
            validator.validate_node_complete(labels, &properties)?;
            validator.check_unique_node(labels, &properties, None)?;
        }
        let id = self.insert_node(labels)?;
        self.write_values(Entity::Node(id), &properties);
        Ok(id)
    }

    /// Creates a node whose remaining properties depend on the node itself
    /// (MERGE `ON CREATE SET` expressions that read it): writes `properties`,
    /// then the ones `derive` computes from the new id. The whole set is
    /// checked like [`create_node`](Self::create_node) before `derive`'s
    /// values are written.
    ///
    /// # Errors
    ///
    /// Returns the first constraint violated, or `derive`'s error.
    pub fn create_node_with(
        &self,
        labels: &[String],
        mut properties: Vec<(String, Value)>,
        derive: impl FnOnce(NodeId) -> Result<Vec<(String, Value)>, OperatorError>,
    ) -> Result<NodeId, OperatorError> {
        if let Some(validator) = &self.validator {
            validator.validate_node_labels_allowed(labels)?;
            validator.inject_defaults(labels, &mut properties);
            self.check_node_values(validator.as_ref(), labels, &properties, None)?;
        }
        let id = self.insert_node(labels)?;
        self.write_values(Entity::Node(id), &properties);

        let derived = derive(id)?;
        if let Some(validator) = &self.validator {
            self.check_node_values(validator.as_ref(), labels, &derived, Some(id))?;
            let all = overlay(properties, &derived);
            validator.validate_node_complete(labels, &all)?;
            validator.check_unique_node(labels, &all, Some(id))?;
        }
        self.write_values(Entity::Node(id), &derived);
        Ok(id)
    }

    /// Sets properties of a node, checked against its labels' constraints.
    ///
    /// `assignments` are `(key, value)` pairs, where a `("*", map)` pair
    /// assigns every entry of the map and a null map entry removes the
    /// property. With `replace`, a map assignment also removes the
    /// properties the map leaves out (`SET n = {...}`).
    ///
    /// # Errors
    ///
    /// Returns a write conflict or the first constraint violated.
    pub fn set_node_properties(
        &self,
        id: NodeId,
        assignments: &[(String, Value)],
        replace: bool,
    ) -> Result<(), OperatorError> {
        self.record(Entity::Node(id))?;
        if let Some(validator) = &self.validator
            && let Some(node) = self.node(id)
        {
            self.check_node_set(validator.as_ref(), &node, assignments, replace)?;
        }
        self.apply_set(Entity::Node(id), assignments, replace);
        Ok(())
    }

    /// Adds labels to a node, after checking the node against the
    /// constraints of the labels it gets. Returns how many were new.
    ///
    /// # Errors
    ///
    /// Returns a write conflict or the first constraint violated.
    pub fn add_labels(&self, id: NodeId, labels: &[String]) -> Result<usize, OperatorError> {
        self.record(Entity::Node(id))?;
        if let Some(validator) = &self.validator
            && let Some(node) = self.node(id)
        {
            let added: Vec<String> = labels
                .iter()
                .filter(|label| !node.has_label(label))
                .cloned()
                .collect();
            if !added.is_empty() {
                let mut all_labels = node_labels(&node);
                all_labels.extend(added.iter().cloned());
                validator.validate_node_labels_allowed(&all_labels)?;
                let values = property_list(&node.properties);
                // The node does not carry the new labels yet, so their UNIQUE
                // checks cannot find the node itself.
                self.check_node_values(validator.as_ref(), &added, &values, None)?;
                validator.validate_node_complete(&added, &values)?;
                validator.check_unique_node(&added, &values, Some(id))?;
            }
        }
        let mut added = 0;
        for label in labels {
            let new = match self.transaction_id {
                Some(transaction_id) => self.store.add_label_versioned(id, label, transaction_id),
                None => self.store.add_label(id, label),
            };
            added += usize::from(new);
        }
        Ok(added)
    }

    /// Removes labels from a node. Returns how many it had.
    ///
    /// # Errors
    ///
    /// Returns a write conflict.
    pub fn remove_labels(&self, id: NodeId, labels: &[String]) -> Result<usize, OperatorError> {
        self.record(Entity::Node(id))?;
        let mut removed = 0;
        for label in labels {
            let had = match self.transaction_id {
                Some(transaction_id) => {
                    self.store.remove_label_versioned(id, label, transaction_id)
                }
                None => self.store.remove_label(id, label),
            };
            removed += usize::from(had);
        }
        Ok(removed)
    }

    /// Deletes a node. With `detach` its edges go too; without, a node that
    /// still has edges is an error.
    ///
    /// # Errors
    ///
    /// Returns a write conflict, or an error for a node with edges and no `detach`.
    pub fn delete_node(&self, id: NodeId, detach: bool) -> Result<bool, OperatorError> {
        self.record(Entity::Node(id))?;
        if detach {
            let outgoing = self.store.edges_from(id, Direction::Outgoing);
            let incoming = self.store.edges_from(id, Direction::Incoming);
            for (_, edge) in outgoing.into_iter().chain(incoming) {
                self.delete_edge(edge)?;
            }
        } else {
            let degree = self.store.out_degree(id) + self.store.in_degree(id);
            if degree > 0 {
                return Err(OperatorError::ConstraintViolation(format!(
                    "Cannot delete node with {degree} connected edge(s). Use DETACH DELETE."
                )));
            }
        }
        Ok(self
            .store
            .delete_node_versioned(id, self.epoch(), self.transaction()))
    }

    // === Edges ===

    /// Creates an edge after checking it against the schema: allowed type,
    /// endpoint labels, property types and required properties.
    ///
    /// # Errors
    ///
    /// Returns the first constraint the edge would violate; nothing is written then.
    pub fn create_edge(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        properties: Vec<(String, Value)>,
    ) -> Result<EdgeId, OperatorError> {
        if let Some(validator) = &self.validator {
            self.check_new_edge(validator.as_ref(), src, dst, edge_type)?;
            for (name, value) in &properties {
                validator.validate_edge_property(edge_type, name, value)?;
            }
            validator.validate_edge_complete(edge_type, &properties)?;
        }
        let id = self.insert_edge(src, dst, edge_type)?;
        self.write_values(Entity::Edge(id), &properties);
        Ok(id)
    }

    /// Creates an edge whose remaining properties depend on the edge itself
    /// (MERGE `ON CREATE SET` expressions that read it), like
    /// [`create_node_with`](Self::create_node_with).
    ///
    /// # Errors
    ///
    /// Returns the first constraint violated, or `derive`'s error.
    pub fn create_edge_with(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        properties: Vec<(String, Value)>,
        derive: impl FnOnce(EdgeId) -> Result<Vec<(String, Value)>, OperatorError>,
    ) -> Result<EdgeId, OperatorError> {
        if let Some(validator) = &self.validator {
            self.check_new_edge(validator.as_ref(), src, dst, edge_type)?;
            for (name, value) in &properties {
                validator.validate_edge_property(edge_type, name, value)?;
            }
        }
        let id = self.insert_edge(src, dst, edge_type)?;
        self.write_values(Entity::Edge(id), &properties);

        let derived = derive(id)?;
        if let Some(validator) = &self.validator {
            for (name, value) in &derived {
                validator.validate_edge_property(edge_type, name, value)?;
            }
            validator.validate_edge_complete(edge_type, &overlay(properties, &derived))?;
        }
        self.write_values(Entity::Edge(id), &derived);
        Ok(id)
    }

    /// Sets properties of an edge, checked against its type; `assignments`
    /// and `replace` work as in [`set_node_properties`](Self::set_node_properties).
    ///
    /// # Errors
    ///
    /// Returns a write conflict or the first constraint violated.
    pub fn set_edge_properties(
        &self,
        id: EdgeId,
        assignments: &[(String, Value)],
        replace: bool,
    ) -> Result<(), OperatorError> {
        self.record(Entity::Edge(id))?;
        if let Some(validator) = &self.validator
            && let Some(edge) = self.edge(id)
        {
            let existing = property_list(&edge.properties);
            for (name, value) in expand_assignments(&existing, assignments, replace) {
                validator.validate_edge_property(edge.edge_type.as_str(), &name, &value)?;
            }
        }
        self.apply_set(Entity::Edge(id), assignments, replace);
        Ok(())
    }

    /// Deletes an edge.
    ///
    /// # Errors
    ///
    /// Returns a write conflict.
    pub fn delete_edge(&self, id: EdgeId) -> Result<bool, OperatorError> {
        self.record(Entity::Edge(id))?;
        Ok(self
            .store
            .delete_edge_versioned(id, self.epoch(), self.transaction()))
    }

    // === Checks ===

    /// Checks property values for a node with `labels`: types, NOT NULL and
    /// single-property UNIQUE. `own` is the node when it exists already: a
    /// value it has cannot make it a duplicate.
    fn check_node_values(
        &self,
        validator: &dyn ConstraintValidator,
        labels: &[String],
        values: &[(String, Value)],
        own: Option<NodeId>,
    ) -> Result<(), OperatorError> {
        for (name, value) in values {
            validator.validate_node_property(labels, name, value)?;
            let unchanged = own.is_some_and(|id| {
                self.store
                    .get_node_property(id, &PropertyKey::new(name.as_str()))
                    .as_ref()
                    == Some(value)
            });
            if !unchanged {
                validator.check_unique_node_property(labels, name, value)?;
            }
        }
        Ok(())
    }

    /// Checks a SET on `node` against the constraints of its own labels. The
    /// constraints on several properties see the node's properties after it.
    fn check_node_set(
        &self,
        validator: &dyn ConstraintValidator,
        node: &Node,
        assignments: &[(String, Value)],
        replace: bool,
    ) -> Result<(), OperatorError> {
        let labels = node_labels(node);
        let existing = property_list(&node.properties);
        let changes = expand_assignments(&existing, assignments, replace);
        self.check_node_values(validator, &labels, &changes, Some(node.id))?;
        validator.check_unique_node(&labels, &overlay(existing, &changes), Some(node.id))
    }

    /// Checks an edge's type and endpoint labels.
    fn check_new_edge(
        &self,
        validator: &dyn ConstraintValidator,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
    ) -> Result<(), OperatorError> {
        validator.validate_edge_type_allowed(edge_type)?;
        let labels_of = |id| {
            self.node(id)
                .map(|node| node_labels(&node))
                .unwrap_or_default()
        };
        validator.validate_edge_endpoints(edge_type, &labels_of(src), &labels_of(dst))
    }

    // === Store writes ===

    fn insert_node(&self, labels: &[String]) -> Result<NodeId, OperatorError> {
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let id = self
            .store
            .create_node_versioned(&label_refs, self.epoch(), self.transaction());
        self.record(Entity::Node(id))?;
        Ok(id)
    }

    fn insert_edge(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
    ) -> Result<EdgeId, OperatorError> {
        let id =
            self.store
                .create_edge_versioned(src, dst, edge_type, self.epoch(), self.transaction());
        self.record(Entity::Edge(id))?;
        Ok(id)
    }

    fn write_values(&self, entity: Entity, values: &[(String, Value)]) {
        for (name, value) in values {
            self.write_value(entity, name, value.clone());
        }
    }

    fn write_value(&self, entity: Entity, key: &str, value: Value) {
        match (entity, self.transaction_id) {
            (Entity::Node(id), Some(transaction_id)) => {
                self.store
                    .set_node_property_versioned(id, key, value, transaction_id);
            }
            (Entity::Node(id), None) => self.store.set_node_property(id, key, value),
            (Entity::Edge(id), Some(transaction_id)) => {
                self.store
                    .set_edge_property_versioned(id, key, value, transaction_id);
            }
            (Entity::Edge(id), None) => self.store.set_edge_property(id, key, value),
        }
    }

    fn remove_value(&self, entity: Entity, key: &str) {
        match (entity, self.transaction_id) {
            (Entity::Node(id), Some(transaction_id)) => {
                self.store
                    .remove_node_property_versioned(id, key, transaction_id);
            }
            (Entity::Node(id), None) => {
                self.store.remove_node_property(id, key);
            }
            (Entity::Edge(id), Some(transaction_id)) => {
                self.store
                    .remove_edge_property_versioned(id, key, transaction_id);
            }
            (Entity::Edge(id), None) => {
                self.store.remove_edge_property(id, key);
            }
        }
    }

    fn existing_keys(&self, entity: Entity) -> Vec<String> {
        let properties = match entity {
            Entity::Node(id) => self.node(id).map(|node| node.properties),
            Entity::Edge(id) => self.edge(id).map(|edge| edge.properties),
        };
        properties
            .map(|properties| {
                properties
                    .iter()
                    .map(|(key, _)| key.as_str().to_string())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Applies a SET: plain assignments write their value, map entries write
    /// theirs or remove the property for a null, and `replace` first removes
    /// every property.
    fn apply_set(&self, entity: Entity, assignments: &[(String, Value)], replace: bool) {
        for (name, value) in assignments {
            if name != MAP_ASSIGNMENT {
                self.write_value(entity, name, value.clone());
                continue;
            }
            let Value::Map(map) = value else {
                continue;
            };
            if replace {
                for key in self.existing_keys(entity) {
                    self.remove_value(entity, &key);
                }
            }
            for (key, entry) in map.iter() {
                if entry.is_null() {
                    self.remove_value(entity, key.as_str());
                } else {
                    self.write_value(entity, key.as_str(), entry.clone());
                }
            }
        }
    }
}

/// The node's labels as strings.
fn node_labels(node: &Node) -> Vec<String> {
    node.labels
        .iter()
        .map(|label| label.as_str().to_string())
        .collect()
}

/// A property map as `(key, value)` pairs.
fn property_list(properties: &PropertyMap) -> Vec<(String, Value)> {
    properties
        .iter()
        .map(|(key, value)| (key.as_str().to_string(), value.clone()))
        .collect()
}

/// The property changes a SET makes: map assignments count as one change
/// per entry, and with `replace` the existing properties the map leaves out
/// become null.
fn expand_assignments(
    existing: &[(String, Value)],
    assignments: &[(String, Value)],
    replace: bool,
) -> Vec<(String, Value)> {
    let mut changes: Vec<(String, Value)> = Vec::new();
    let mut replaces_all = false;
    for (name, value) in assignments {
        match (name.as_str(), value) {
            (MAP_ASSIGNMENT, Value::Map(map)) => {
                replaces_all |= replace;
                changes.extend(
                    map.iter()
                        .map(|(key, value)| (key.as_str().to_string(), value.clone())),
                );
            }
            (MAP_ASSIGNMENT, _) => {}
            _ => changes.push((name.clone(), value.clone())),
        }
    }
    if replaces_all {
        for (key, _) in existing {
            if !changes.iter().any(|(name, _)| name == key) {
                changes.push((key.clone(), Value::Null));
            }
        }
    }
    changes
}

/// `base` with `changes` applied on top (a change replaces the same key).
fn overlay(mut base: Vec<(String, Value)>, changes: &[(String, Value)]) -> Vec<(String, Value)> {
    for (name, value) in changes {
        base.retain(|(key, _)| key != name);
        base.push((name.clone(), value.clone()));
    }
    base
}
