//! One validated write path for graph mutations.
//!
//! [`GraphWriter`] is what every mutation goes through: the CREATE, SET,
//! REMOVE, DELETE and MERGE operators, and the direct API through the
//! session. For each write it records the entity for write-conflict
//! detection, checks the schema and constraints, and writes with the
//! transaction's versioning, so the rules for a valid write live in one place.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use grafeo_common::storage::value_codec::{MAX_PROPERTY_VALUE_DEPTH, nests_too_deep};
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

/// What the writes of one statement changed, as counts: the summary a query
/// result reports.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WriteCounters {
    /// Nodes created, by `INSERT`, `CREATE` or `MERGE`.
    pub nodes_created: u64,
    /// Nodes deleted.
    pub nodes_deleted: u64,
    /// Edges created.
    pub edges_created: u64,
    /// Edges deleted, also those `DETACH DELETE` removes.
    pub edges_deleted: u64,
    /// Property values written or removed, also those of created entities.
    pub properties_set: u64,
    /// Labels added, also those of created nodes.
    pub labels_added: u64,
    /// Labels removed.
    pub labels_removed: u64,
}

impl WriteCounters {
    /// Whether the writes changed anything.
    #[must_use]
    pub fn contains_updates(&self) -> bool {
        *self != Self::default()
    }
}

/// Counts writes as they happen, shared by every writer of one statement;
/// [`counters`](Self::counters) reads the totals.
#[derive(Debug, Default)]
pub struct WriteCounter {
    nodes_created: AtomicU64,
    nodes_deleted: AtomicU64,
    edges_created: AtomicU64,
    edges_deleted: AtomicU64,
    properties_set: AtomicU64,
    labels_added: AtomicU64,
    labels_removed: AtomicU64,
}

impl WriteCounter {
    /// The counts so far.
    #[must_use]
    pub fn counters(&self) -> WriteCounters {
        let read = |count: &AtomicU64| count.load(Ordering::Relaxed);
        WriteCounters {
            nodes_created: read(&self.nodes_created),
            nodes_deleted: read(&self.nodes_deleted),
            edges_created: read(&self.edges_created),
            edges_deleted: read(&self.edges_deleted),
            properties_set: read(&self.properties_set),
            labels_added: read(&self.labels_added),
            labels_removed: read(&self.labels_removed),
        }
    }
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
    counter: Option<Arc<WriteCounter>>,
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
            counter: None,
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

    /// Counts every write in `counter`.
    #[must_use]
    pub fn with_counter(mut self, counter: Arc<WriteCounter>) -> Self {
        self.counter = Some(counter);
        self
    }

    /// Adds `n` to the count `field` selects, if this writer counts.
    fn count(&self, field: impl Fn(&WriteCounter) -> &AtomicU64, n: usize) {
        if n > 0
            && let Some(counter) = &self.counter
        {
            field(counter).fetch_add(n as u64, Ordering::Relaxed);
        }
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
    #[must_use]
    pub fn node(&self, id: NodeId) -> Option<Node> {
        match (self.viewing_epoch, self.transaction_id) {
            (Some(epoch), Some(transaction_id)) => {
                self.store.get_node_versioned(id, epoch, transaction_id)
            }
            _ => self.store.get_node(id),
        }
    }

    /// The edge as this writer's transaction sees it, its own writes included.
    #[must_use]
    pub fn edge(&self, id: EdgeId) -> Option<Edge> {
        match (self.viewing_epoch, self.transaction_id) {
            (Some(epoch), Some(transaction_id)) => {
                self.store.get_edge_versioned(id, epoch, transaction_id)
            }
            _ => self.store.get_edge(id),
        }
    }

    /// Whether this writer's transaction sees the node, without reading its
    /// labels and properties.
    #[must_use]
    pub fn has_node(&self, id: NodeId) -> bool {
        match (self.viewing_epoch, self.transaction_id) {
            (Some(epoch), Some(transaction_id)) => {
                self.store
                    .is_node_visible_versioned(id, epoch, transaction_id)
            }
            _ => self
                .store
                .is_node_visible_at_epoch(id, self.store.current_epoch()),
        }
    }

    /// Whether this writer's transaction sees the edge, without reading its
    /// properties.
    #[must_use]
    pub fn has_edge(&self, id: EdgeId) -> bool {
        match (self.viewing_epoch, self.transaction_id) {
            (Some(epoch), Some(transaction_id)) => {
                self.store
                    .is_edge_visible_versioned(id, epoch, transaction_id)
            }
            _ => self
                .store
                .is_edge_visible_at_epoch(id, self.store.current_epoch()),
        }
    }

    /// Fails when this writer's transaction cannot see the node: one it
    /// deleted earlier, or one that does not exist. A write to it would change
    /// nothing anyone sees.
    fn require_node(&self, id: NodeId) -> Result<(), OperatorError> {
        if self.has_node(id) {
            Ok(())
        } else {
            Err(OperatorError::Execution(format!(
                "Node {} does not exist or has been deleted in this transaction",
                id.as_u64()
            )))
        }
    }

    /// Fails when this writer's transaction cannot see the edge, like
    /// [`require_node`](Self::require_node).
    fn require_edge(&self, id: EdgeId) -> Result<(), OperatorError> {
        if self.has_edge(id) {
            Ok(())
        } else {
            Err(OperatorError::Execution(format!(
                "Relationship {} does not exist or has been deleted in this transaction",
                id.as_u64()
            )))
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
        refuse_too_deep(plain_values(&properties))?;
        if let Some(validator) = &self.validator {
            validator.validate_node_labels_allowed(labels)?;
            validator.inject_defaults(labels, &mut properties);
            self.check_node_values(validator.as_ref(), labels, &properties, None)?;
            validator.validate_node_complete(labels, &properties)?;
            validator.check_unique_node(labels, &properties, None)?;
        }
        let id = self.insert_node(labels)?;
        self.write_values(Entity::Node(id), &properties)?;
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
        refuse_too_deep(plain_values(&properties))?;
        if let Some(validator) = &self.validator {
            validator.validate_node_labels_allowed(labels)?;
            validator.inject_defaults(labels, &mut properties);
            self.check_node_values(validator.as_ref(), labels, &properties, None)?;
        }
        let id = self.insert_node(labels)?;
        self.write_values(Entity::Node(id), &properties)?;

        let derived = derive(id)?;
        refuse_too_deep(plain_values(&derived))?;
        if let Some(validator) = &self.validator {
            self.check_node_values(validator.as_ref(), labels, &derived, Some(id))?;
            let all = overlay(properties, &derived);
            validator.validate_node_complete(labels, &all)?;
            validator.check_unique_node(labels, &all, Some(id))?;
        }
        self.write_values(Entity::Node(id), &derived)?;
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
    /// Returns an error for a node the transaction deleted, a write conflict
    /// or the first constraint violated.
    pub fn set_node_properties(
        &self,
        id: NodeId,
        assignments: &[(String, Value)],
        replace: bool,
    ) -> Result<(), OperatorError> {
        refuse_too_deep(assigned_values(assignments))?;
        self.require_node(id)?;
        self.record(Entity::Node(id))?;
        if let Some(validator) = &self.validator {
            let needs_node = replace
                || assigned_values(assignments)
                    .any(|(key, value)| validator.constrains_node_property(key, value));
            if !needs_node {
                for (key, value) in assigned_values(assignments) {
                    validator.validate_node_property(&[], key, value)?;
                }
            } else if let Some(node) = self.node(id) {
                self.check_node_set(validator.as_ref(), &node, assignments, replace)?;
            }
        }
        self.apply_set(Entity::Node(id), assignments, replace)?;
        Ok(())
    }

    /// Removes a property from a node, checked like setting it to null.
    /// Returns whether the node had it.
    ///
    /// # Errors
    ///
    /// Returns an error for a node the transaction deleted, a write conflict
    /// or the constraint the removal would violate (`NOT NULL`, `NODE KEY`).
    pub fn remove_node_property(&self, id: NodeId, key: &str) -> Result<bool, OperatorError> {
        self.require_node(id)?;
        self.record(Entity::Node(id))?;
        let Some(node) = self.node(id) else {
            return Ok(false);
        };
        if node.get_property(key).is_none() {
            return Ok(false);
        }
        if let Some(validator) = &self.validator {
            self.check_node_set(
                validator.as_ref(),
                &node,
                &[(key.to_string(), Value::Null)],
                false,
            )?;
        }
        self.remove_value(Entity::Node(id), key)?;
        Ok(true)
    }

    /// Adds labels to a node, after checking the node against the
    /// constraints of the labels it gets. Returns how many were new.
    ///
    /// # Errors
    ///
    /// Returns an error for a node the transaction deleted, a write conflict
    /// or the first constraint violated.
    pub fn add_labels(&self, id: NodeId, labels: &[String]) -> Result<usize, OperatorError> {
        self.require_node(id)?;
        self.record(Entity::Node(id))?;
        let Some(node) = self.node(id) else {
            return Ok(0);
        };
        if let Some(validator) = &self.validator {
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
        self.count(|c| &c.labels_added, added);
        Ok(added)
    }

    /// Removes labels from a node. Returns how many it had.
    ///
    /// # Errors
    ///
    /// Returns an error for a node the transaction deleted, or a write
    /// conflict.
    pub fn remove_labels(&self, id: NodeId, labels: &[String]) -> Result<usize, OperatorError> {
        self.require_node(id)?;
        self.record(Entity::Node(id))?;
        if self.node(id).is_none() {
            return Ok(0);
        }
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
        self.count(|c| &c.labels_removed, removed);
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
        let deleted = self
            .store
            .delete_node_versioned(id, self.epoch(), self.transaction())
            .map_err(refused)?;
        self.count(|c| &c.nodes_deleted, usize::from(deleted));
        Ok(deleted)
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
        refuse_too_deep(plain_values(&properties))?;
        if let Some(validator) = &self.validator {
            self.check_new_edge(validator.as_ref(), src, dst, edge_type)?;
            for (name, value) in &properties {
                validator.validate_edge_property(edge_type, name, value)?;
            }
            validator.validate_edge_complete(edge_type, &properties)?;
        }
        let id = self.insert_edge(src, dst, edge_type)?;
        self.write_values(Entity::Edge(id), &properties)?;
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
        refuse_too_deep(plain_values(&properties))?;
        if let Some(validator) = &self.validator {
            self.check_new_edge(validator.as_ref(), src, dst, edge_type)?;
            for (name, value) in &properties {
                validator.validate_edge_property(edge_type, name, value)?;
            }
        }
        let id = self.insert_edge(src, dst, edge_type)?;
        self.write_values(Entity::Edge(id), &properties)?;

        let derived = derive(id)?;
        refuse_too_deep(plain_values(&derived))?;
        if let Some(validator) = &self.validator {
            for (name, value) in &derived {
                validator.validate_edge_property(edge_type, name, value)?;
            }
            validator.validate_edge_complete(edge_type, &overlay(properties, &derived))?;
        }
        self.write_values(Entity::Edge(id), &derived)?;
        Ok(id)
    }

    /// Sets properties of an edge, checked against its type; `assignments`
    /// and `replace` work as in [`set_node_properties`](Self::set_node_properties).
    ///
    /// # Errors
    ///
    /// Returns an error for an edge the transaction deleted, a write conflict
    /// or the first constraint violated.
    pub fn set_edge_properties(
        &self,
        id: EdgeId,
        assignments: &[(String, Value)],
        replace: bool,
    ) -> Result<(), OperatorError> {
        refuse_too_deep(assigned_values(assignments))?;
        self.require_edge(id)?;
        self.record(Entity::Edge(id))?;
        if let Some(validator) = &self.validator
            && let Some(edge) = self.edge(id)
        {
            let existing = property_list(&edge.properties);
            for (name, value) in expand_assignments(&existing, assignments, replace) {
                validator.validate_edge_property(edge.edge_type.as_str(), &name, &value)?;
            }
        }
        self.apply_set(Entity::Edge(id), assignments, replace)?;
        Ok(())
    }

    /// Removes a property from an edge, checked like setting it to null.
    /// Returns whether the edge had it.
    ///
    /// # Errors
    ///
    /// Returns an error for an edge the transaction deleted, a write conflict
    /// or the constraint the removal would violate.
    pub fn remove_edge_property(&self, id: EdgeId, key: &str) -> Result<bool, OperatorError> {
        self.require_edge(id)?;
        self.record(Entity::Edge(id))?;
        let Some(edge) = self.edge(id) else {
            return Ok(false);
        };
        if edge.get_property(key).is_none() {
            return Ok(false);
        }
        if let Some(validator) = &self.validator {
            validator.validate_edge_property(edge.edge_type.as_str(), key, &Value::Null)?;
        }
        self.remove_value(Entity::Edge(id), key)?;
        Ok(true)
    }

    /// Deletes an edge.
    ///
    /// # Errors
    ///
    /// Returns a write conflict.
    pub fn delete_edge(&self, id: EdgeId) -> Result<bool, OperatorError> {
        self.record(Entity::Edge(id))?;
        let deleted = self
            .store
            .delete_edge_versioned(id, self.epoch(), self.transaction());
        self.count(|c| &c.edges_deleted, usize::from(deleted));
        Ok(deleted)
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
        if !validator.constrains_edge_endpoints(edge_type) {
            return Ok(());
        }
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
        self.count(|c| &c.nodes_created, 1);
        // `(:A:A)` gives the node one label.
        let distinct: std::collections::BTreeSet<&str> = label_refs.into_iter().collect();
        self.count(|c| &c.labels_added, distinct.len());
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
        self.count(|c| &c.edges_created, 1);
        Ok(id)
    }

    fn write_values(
        &self,
        entity: Entity,
        values: &[(String, Value)],
    ) -> Result<(), OperatorError> {
        for (name, value) in values {
            self.write_value(entity, name, value.clone())?;
        }
        Ok(())
    }

    /// Writes a property value; a null removes the property, since a property
    /// with a null value does not exist.
    fn write_value(&self, entity: Entity, key: &str, value: Value) -> Result<(), OperatorError> {
        if value.is_null() {
            return self.remove_value(entity, key);
        }
        match (entity, self.transaction_id) {
            (Entity::Node(id), Some(transaction_id)) => {
                self.store
                    .set_node_property_versioned(id, key, value, transaction_id)
                    .map_err(refused)?;
            }
            (Entity::Node(id), None) => self.store.set_node_property(id, key, value),
            (Entity::Edge(id), Some(transaction_id)) => {
                self.store
                    .set_edge_property_versioned(id, key, value, transaction_id);
            }
            (Entity::Edge(id), None) => self.store.set_edge_property(id, key, value),
        }
        self.count(|c| &c.properties_set, 1);
        Ok(())
    }

    fn remove_value(&self, entity: Entity, key: &str) -> Result<(), OperatorError> {
        let removed =
            match (entity, self.transaction_id) {
                (Entity::Node(id), Some(transaction_id)) => self
                    .store
                    .remove_node_property_versioned(id, key, transaction_id),
                (Entity::Node(id), None) => self.store.remove_node_property(id, key),
                (Entity::Edge(id), Some(transaction_id)) => self
                    .store
                    .remove_edge_property_versioned(id, key, transaction_id),
                (Entity::Edge(id), None) => self.store.remove_edge_property(id, key),
            }
            .map_err(refused)?;
        self.count(|c| &c.properties_set, usize::from(removed.is_some()));
        Ok(())
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
    fn apply_set(
        &self,
        entity: Entity,
        assignments: &[(String, Value)],
        replace: bool,
    ) -> Result<(), OperatorError> {
        for (name, value) in assignments {
            if name != MAP_ASSIGNMENT {
                self.write_value(entity, name, value.clone())?;
                continue;
            }
            let Value::Map(map) = value else {
                continue;
            };
            if replace {
                for key in self.existing_keys(entity) {
                    self.remove_value(entity, &key)?;
                }
            }
            for (key, entry) in map.iter() {
                if entry.is_null() {
                    self.remove_value(entity, key.as_str())?;
                } else {
                    self.write_value(entity, key.as_str(), entry.clone())?;
                }
            }
        }
        Ok(())
    }
}

/// The statement error of a write the store refused, such as a spilled
/// property value whose file cannot be read, which a rollback would lose.
fn refused(error: grafeo_common::utils::error::Error) -> OperatorError {
    OperatorError::Execution(error.to_string())
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

/// Refuses a property value nested deeper than a database can store
/// ([`MAX_PROPERTY_VALUE_DEPTH`] lists, maps and paths), before anything is
/// written, so a checkpoint never meets one. A limit on input, reported as
/// the property size limit is: a constraint violation (an invalid value).
fn refuse_too_deep<'v>(
    mut values: impl Iterator<Item = (&'v str, &'v Value)>,
) -> Result<(), OperatorError> {
    match values.find(|(_, value)| nests_too_deep(value)) {
        Some((key, _)) => Err(OperatorError::ConstraintViolation(format!(
            "property {key:?}: the value nests lists, maps and paths more than \
             {MAX_PROPERTY_VALUE_DEPTH} levels deep, deeper than a database can store"
        ))),
        None => Ok(()),
    }
}

/// The `(key, value)` pairs of a property list, as written.
fn plain_values(properties: &[(String, Value)]) -> impl Iterator<Item = (&str, &Value)> {
    properties.iter().map(|(key, value)| (key.as_str(), value))
}

/// The `(key, value)` pairs a SET writes: the entries of a map assignment,
/// and every other assignment itself.
fn assigned_values(assignments: &[(String, Value)]) -> impl Iterator<Item = (&str, &Value)> {
    assignments.iter().flat_map(|(name, value)| {
        let single = (name != MAP_ASSIGNMENT).then_some((name.as_str(), value));
        let entries = match value {
            Value::Map(map) if name == MAP_ASSIGNMENT => Some(map),
            _ => None,
        };
        single.into_iter().chain(
            entries
                .into_iter()
                .flat_map(|map| map.iter().map(|(key, value)| (key.as_str(), value))),
        )
    })
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

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use std::sync::Arc;

    use grafeo_common::storage::value_codec::MAX_PROPERTY_VALUE_DEPTH;
    use grafeo_common::types::{PropertyKey, Value};

    use super::GraphWriter;
    use crate::graph::GraphStoreMut;
    use crate::graph::lpg::LpgStore;

    /// A string inside `depth` lists.
    fn nested(depth: usize) -> Value {
        let mut value = Value::from("Prague");
        for _ in 0..depth {
            value = Value::List(Arc::from(vec![value]));
        }
        value
    }

    fn writer() -> (Arc<LpgStore>, GraphWriter) {
        let store = Arc::new(LpgStore::new().unwrap());
        let target: Arc<dyn GraphStoreMut> = Arc::clone(&store) as Arc<dyn GraphStoreMut>;
        (store, GraphWriter::new(target))
    }

    fn labels(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_string()).collect()
    }

    fn pairs(key: &str, value: &Value) -> Vec<(String, Value)> {
        vec![(key.to_string(), value.clone())]
    }

    #[test]
    fn values_nested_deeper_than_a_file_holds_are_refused_before_any_write() {
        let (store, writer) = writer();
        let too_deep = nested(MAX_PROPERTY_VALUE_DEPTH + 1);
        let trips = PropertyKey::new("trips");

        let mut properties = pairs("name", &Value::from("Alix"));
        properties.extend(pairs("trips", &too_deep));
        let error = writer
            .create_node(&labels(&["Person"]), properties)
            .unwrap_err();
        assert!(
            matches!(error, super::OperatorError::ConstraintViolation(_)),
            "an invalid value, not an internal error: {error:?}"
        );
        let error = error.to_string();
        assert!(
            error.contains("\"trips\"") && error.contains(&MAX_PROPERTY_VALUE_DEPTH.to_string()),
            "the error names the property and the limit: {error}"
        );
        assert_eq!(store.node_count(), 0, "a refused node is not created");

        let deepest = nested(MAX_PROPERTY_VALUE_DEPTH);
        let alix = writer
            .create_node(&labels(&["Person"]), pairs("trips", &deepest))
            .expect("the deepest value a file holds is accepted");
        assert!(
            writer
                .set_node_properties(alix, &pairs("trips", &too_deep), false)
                .is_err(),
            "SET n.trips"
        );
        let map = Value::Map(Arc::new(
            [(trips.clone(), too_deep.clone())].into_iter().collect(),
        ));
        assert!(
            writer
                .set_node_properties(alix, &pairs("*", &map), false)
                .is_err(),
            "SET n += {{trips: ...}}"
        );
        assert_eq!(
            store.get_node_property(alix, &trips),
            Some(deepest),
            "a refused SET leaves the value as it was"
        );
        assert!(
            writer
                .create_node_with(&labels(&["Person"]), Vec::new(), |_| Ok(pairs(
                    "trips", &too_deep
                )))
                .is_err(),
            "a derived value of MERGE ... ON CREATE SET"
        );

        let gus = writer
            .create_node(&labels(&["Person"]), Vec::new())
            .unwrap();
        let edges_before = store.edge_count();
        assert!(
            writer
                .create_edge(alix, gus, "KNOWS", pairs("route", &too_deep))
                .is_err(),
            "CREATE ()-[{{route: ...}}]->()"
        );
        assert_eq!(
            store.edge_count(),
            edges_before,
            "a refused edge is not created"
        );
        let knows = writer.create_edge(alix, gus, "KNOWS", Vec::new()).unwrap();
        assert!(
            writer
                .set_edge_properties(knows, &pairs("route", &too_deep), false)
                .is_err(),
            "SET r.route"
        );
        assert!(
            writer
                .create_edge_with(alix, gus, "KNOWS", Vec::new(), |_| Ok(pairs(
                    "route", &too_deep
                )))
                .is_err(),
            "a derived edge value"
        );
    }
}
