//! One validated write path for graph mutations.
//!
//! [`GraphWriter`] is what every mutation goes through: the CREATE, SET,
//! REMOVE, DELETE and MERGE operators, and the direct API through the
//! session. For each write it records the entity for write-conflict
//! detection, checks the schema and constraints, and writes with the
//! transaction's versioning, so the rules for a valid write live in one place.
//! A transaction's store changes run as a write in progress (see
//! [`WriteClaims::write_in_progress`]), which a checkpoint waits for and
//! which waits for a checkpoint.
//!
//! A writer given a [`Recording`] writes through the graph's change target
//! ([`ChangeTarget::apply`]) instead of the store's versioned methods, and
//! records each write that changed something in the transaction's change
//! set ([`ChangeRecorder::record`]): the op with its after-image, what it
//! replaced, and what it did to the transaction's pending version. A write
//! that changes nothing, or that a check or the store refuses, records
//! nothing. The claim of what a write changes comes before the store
//! changes, except for a node it creates: no other transaction knows the
//! new id, so it claims nothing.

use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use grafeo_common::change::{DataOp, Labels, Properties, Table};
use grafeo_common::storage::value_codec::{MAX_PROPERTY_VALUE_DEPTH, nests_too_deep};
use grafeo_common::types::{
    ArcStr, EdgeId, EpochId, NodeId, PropertyKey, PropertyMap, TransactionId, Value,
};
use grafeo_common::utils::hash::FxHashSet;

use super::{
    BulkRows, ChangeRecorder, ConstraintValidator, OperatorError, WriteClaim, WriteClaims,
    WriteInProgress,
};
use crate::graph::apply::{Applied, ApplyError, ChangeTarget, ExternalTarget, Writer};
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
///
/// Read, not built, outside this crate: later releases may add counts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
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

/// The store of one graph as a recording writer changes it.
#[derive(Clone)]
pub enum WriteTarget {
    /// A store that creates at ids reserved from it (the built-in store).
    Store(Arc<dyn ChangeTarget>),
    /// A store a database was built on, which gives the ids of what it
    /// creates itself (see [`ExternalTarget::create_node`]).
    External(Arc<ExternalTarget>),
}

impl WriteTarget {
    /// Applies `op` as `writer` (see [`ChangeTarget::apply`]).
    fn apply(&self, op: &DataOp, writer: Writer) -> Result<Applied, ApplyError> {
        match self {
            Self::Store(target) => target.apply(op, writer),
            Self::External(target) => target.apply(op, writer),
        }
    }
}

impl std::fmt::Debug for WriteTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Store(_) => f.write_str("WriteTarget::Store"),
            Self::External(_) => f.write_str("WriteTarget::External"),
        }
    }
}

/// An edge for [`GraphWriter::create_edges`] to create.
#[derive(Debug, Clone, PartialEq)]
pub struct NewEdge {
    /// The node it leaves.
    pub src: NodeId,
    /// The node it enters.
    pub dst: NodeId,
    /// Its type.
    pub edge_type: String,
    /// Its properties; a null value is no property.
    pub properties: Vec<(String, Value)>,
}

/// The rows of a bulk create that run as one write in progress (see
/// [`WriteClaims::write_in_progress`]): a checkpoint waits for at most this
/// many rows.
const BULK_RUN: usize = 1024;

/// A bulk create's number of rows, as ids to reserve.
fn row_count(rows: usize) -> Result<u64, OperatorError> {
    u64::try_from(rows)
        .map_err(|_| OperatorError::Execution(format!("{rows} rows: more ids than a store has")))
}

/// The values a create op writes.
fn op_values(op: &DataOp) -> usize {
    match op {
        DataOp::CreateNode { properties, .. } | DataOp::CreateEdge { properties, .. } => {
            properties.len()
        }
        _ => 0,
    }
}

/// What a recording writer writes through and records in (see
/// [`GraphWriter::with_recording`]): one graph's store and one
/// transaction's changes in that graph.
#[derive(Clone)]
pub struct Recording {
    /// The graph's store.
    pub target: WriteTarget,
    /// The transaction's changes in the graph, with its claims and the write
    /// freeze.
    pub recorder: Arc<dyn ChangeRecorder>,
}

/// Writes to a graph store for one statement or direct call: validated,
/// tracked for write conflicts and versioned by the transaction.
#[derive(Clone)]
pub struct GraphWriter {
    store: Arc<dyn GraphStoreMut>,
    viewing_epoch: Option<EpochId>,
    transaction_id: Option<TransactionId>,
    validator: Option<Arc<dyn ConstraintValidator>>,
    counter: Option<Arc<WriteCounter>>,
    /// Where the writes go and are recorded; `None` to write through the
    /// store's versioned methods.
    recording: Option<Recording>,
    /// The claims of a writer without a recording that writes as a
    /// transaction through the store's versioned methods.
    claims: Option<Arc<dyn WriteClaims>>,
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
            counter: None,
            recording: None,
            claims: None,
        }
    }

    /// Claims what each write changes through `claims`, and holds its write
    /// freeze, for a writer without a recording that writes as a transaction
    /// through the store's versioned methods (see
    /// [`with_transaction_context`](Self::with_transaction_context)). A
    /// recording claims through its own recorder instead.
    #[must_use]
    pub fn with_claims(mut self, claims: Arc<dyn WriteClaims>) -> Self {
        self.claims = Some(claims);
        self
    }

    /// Writes through `recording`'s target and records each write that
    /// changed something in its recorder, which also takes the claims and
    /// holds the write freeze, as the recorder's writer: a transaction reads
    /// at its snapshot, its own writes included; an immediate write reads at
    /// its epoch, as the system. The store this writer was made with stays
    /// the one it reads.
    #[must_use]
    pub fn with_recording(mut self, recording: Recording) -> Self {
        match recording.recorder.writer() {
            Writer::Transaction { id, snapshot } => {
                self.viewing_epoch = Some(snapshot);
                self.transaction_id = Some(id);
            }
            Writer::Immediate { epoch, .. } => {
                self.viewing_epoch = Some(epoch);
                self.transaction_id = None;
            }
            Writer::Replay { .. } => {}
        }
        self.recording = Some(recording);
        self
    }

    /// Writes as `transaction_id` through the store's versioned methods,
    /// reading at `epoch`, without claims or a record of what it changed:
    /// nothing stamps or undoes the versions it writes (a writer of a
    /// transaction the engine runs gets a [`Recording`] instead, see
    /// [`with_recording`](Self::with_recording)).
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

    /// Claims what the next store change writes, for write-conflict
    /// detection, through the recording's recorder or the writer's claims; a
    /// writer without either claims nothing. An edge's endpoints are claimed
    /// once the edge passed its checks, right before it is written: an edge
    /// refused claims nothing, so it holds off no delete. The delete of a
    /// node also conflicts with an edge another transaction creates to it.
    fn claim(&self, claim: WriteClaim) -> Result<(), OperatorError> {
        match (&self.recording, &self.claims) {
            (Some(recording), _) => recording.recorder.claim(claim),
            (None, Some(claims)) => claims.claim(claim),
            (None, None) => Ok(()),
        }
    }

    /// Marks this writer's store changes as in progress while the guard
    /// lives (see [`WriteClaims::write_in_progress`]): a checkpoint or a
    /// copy of the store waits for them, and they wait for one. `None` for
    /// a writer without a recording or claims.
    ///
    /// Each public write method takes it once, right before its first store
    /// change; the methods it calls never take it again (it is not
    /// reentrant), and the expressions a `derive` of
    /// [`create_node_with`](Self::create_node_with) evaluates run without it.
    fn write_in_progress(&self) -> Option<WriteInProgress<'_>> {
        match (&self.recording, &self.claims) {
            (Some(recording), _) => recording.recorder.write_in_progress(),
            (None, Some(claims)) => claims.write_in_progress(),
            (None, None) => None,
        }
    }

    /// Applies `op` through the recording's target and records it when it
    /// changed something; whether it did. A write the store refuses changes
    /// nothing and records nothing. Called with the write freeze held.
    fn apply_op(&self, recording: &Recording, op: DataOp) -> Result<bool, OperatorError> {
        let applied = recording
            .target
            .apply(&op, recording.recorder.writer())
            .map_err(store_refused)?;
        record_applied(recording, op, applied)
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
        properties: Vec<(String, Value)>,
    ) -> Result<NodeId, OperatorError> {
        let properties = self.new_node_properties(labels, properties)?;
        if let Some(validator) = &self.validator {
            validator.validate_node_properties_declared(labels, &properties)?;
            self.check_node_values(validator.as_ref(), labels, &properties, None)?;
            validator.validate_node_complete(labels, &properties)?;
            validator.check_unique_node(labels, &properties, None)?;
        }
        let _writing = self.write_in_progress();
        self.insert_node(labels, &properties)
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
        properties: Vec<(String, Value)>,
        derive: impl FnOnce(NodeId) -> Result<Vec<(String, Value)>, OperatorError>,
    ) -> Result<NodeId, OperatorError> {
        let properties = self.new_node_properties(labels, properties)?;
        if let Some(validator) = &self.validator {
            validator.validate_node_properties_declared(labels, &properties)?;
            self.check_node_values(validator.as_ref(), labels, &properties, None)?;
        }
        let id = {
            let _writing = self.write_in_progress();
            self.insert_node(labels, &properties)?
        };

        // Evaluates expressions: no write is in progress meanwhile.
        let mut derived = derive(id)?;
        refuse_too_deep(plain_values(&derived))?;
        if let Some(validator) = &self.validator {
            convert_values(&mut derived, |key, value| {
                validator.convert_node_property(labels, key, value)
            });
            validator.validate_node_properties_declared(labels, &derived)?;
            self.check_node_values(validator.as_ref(), labels, &derived, Some(id))?;
            let all = overlay(properties, &derived);
            validator.validate_node_complete(labels, &all)?;
            validator.check_unique_node(labels, &all, Some(id))?;
        }
        let _writing = self.write_in_progress();
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
        self.claim(WriteClaim::Node(id))?;
        let mut assignments = Cow::Borrowed(assignments);
        if let Some(validator) = &self.validator {
            let needs_node = replace
                || assigned_values(&assignments)
                    .any(|(key, value)| validator.constrains_node_property(key, value));
            if !needs_node {
                for (key, value) in assigned_values(&assignments) {
                    validator.validate_node_property(&[], key, value)?;
                }
            } else if let Some(node) = self.node(id) {
                let labels = node_labels(&node);
                if let Some(converted) = convert_assignments(&assignments, |key, value| {
                    validator.convert_node_property(&labels, key, value)
                }) {
                    assignments = Cow::Owned(converted);
                }
                self.check_node_set(validator.as_ref(), &node, &assignments, replace)?;
            }
        }
        let _writing = self.write_in_progress();
        self.apply_set(Entity::Node(id), &assignments, replace)?;
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
        self.claim(WriteClaim::Node(id))?;
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
        let _writing = self.write_in_progress();
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
        self.claim(WriteClaim::Node(id))?;
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
        let _writing = self.write_in_progress();
        let mut added = 0;
        for label in labels {
            let new = match (&self.recording, self.transaction_id) {
                (Some(recording), _) => self.apply_op(
                    recording,
                    DataOp::AddNodeLabel {
                        id,
                        label: ArcStr::from(label.as_str()),
                    },
                )?,
                (None, Some(transaction_id)) => {
                    self.store.add_label_versioned(id, label, transaction_id)
                }
                (None, None) => self.store.add_label(id, label),
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
        self.claim(WriteClaim::Node(id))?;
        if self.node(id).is_none() {
            return Ok(0);
        }
        let _writing = self.write_in_progress();
        let mut removed = 0;
        for label in labels {
            let had = match (&self.recording, self.transaction_id) {
                (Some(recording), _) => self.apply_op(
                    recording,
                    DataOp::RemoveNodeLabel {
                        id,
                        label: ArcStr::from(label.as_str()),
                    },
                )?,
                (None, Some(transaction_id)) => {
                    self.store.remove_label_versioned(id, label, transaction_id)
                }
                (None, None) => self.store.remove_label(id, label),
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
    /// Returns a write conflict (also with a transaction that creates an
    /// edge to the node), or an error for a node with edges and no `detach`.
    pub fn delete_node(&self, id: NodeId, detach: bool) -> Result<bool, OperatorError> {
        self.claim(WriteClaim::NodeDelete(id))?;
        let _writing = self.write_in_progress();
        if detach {
            let outgoing = self.store.edges_from(id, Direction::Outgoing);
            let incoming = self.store.edges_from(id, Direction::Incoming);
            for (_, edge) in outgoing.into_iter().chain(incoming) {
                self.remove_edge(edge)?;
            }
        } else if self.store.out_degree(id) + self.store.in_degree(id) > 0 {
            let degree = self.connected_edge_count(id);
            if degree > 0 {
                return Err(OperatorError::ConstraintViolation(format!(
                    "Cannot delete node with {degree} connected edge(s). Use DETACH DELETE."
                )));
            }
        }
        let deleted = match &self.recording {
            Some(recording) => self.apply_op(recording, DataOp::DeleteNode { id })?,
            None => self
                .store
                .delete_node_versioned(id, self.epoch(), self.transaction())
                .map_err(refused)?,
        };
        self.count(|c| &c.nodes_deleted, usize::from(deleted));
        Ok(deleted)
    }

    /// The edges of node `id` that a delete without `DETACH` refuses: those
    /// the store lists for it, less the ones this writer's transaction
    /// deleted itself. The base of a compacted store lists a transaction's
    /// deletes until it commits, as other readers still see them; an edge the
    /// transaction deleted is one visible at its snapshot that it no longer
    /// sees. Edges others created or committed after the snapshot still
    /// count.
    fn connected_edge_count(&self, id: NodeId) -> usize {
        let outgoing = self.store.edges_from(id, Direction::Outgoing);
        let incoming = self.store.edges_from(id, Direction::Incoming);
        let edges = outgoing.into_iter().chain(incoming).map(|(_, edge)| edge);
        let (Some(epoch), Some(transaction_id)) = (self.viewing_epoch, self.transaction_id) else {
            return edges.count();
        };
        edges
            .filter(|&edge| {
                let deleted_by_this_transaction = self.store.is_edge_visible_at_epoch(edge, epoch)
                    && !self
                        .store
                        .is_edge_visible_versioned(edge, epoch, transaction_id);
                !deleted_by_this_transaction
            })
            .count()
    }

    // === Edges ===

    /// Creates an edge after checking that the transaction sees both
    /// endpoints and checking the edge against the schema: allowed type,
    /// endpoint labels, type defaults, property types and required
    /// properties. Then it claims the endpoints against a concurrent delete,
    /// so an edge refused claims nothing.
    ///
    /// # Errors
    ///
    /// Returns an error for an endpoint the transaction does not see (one it
    /// deleted, or that does not exist), the first constraint the edge would
    /// violate, or a write conflict with a transaction that deletes an
    /// endpoint; nothing is written then.
    pub fn create_edge(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        properties: Vec<(String, Value)>,
    ) -> Result<EdgeId, OperatorError> {
        let mut properties = self.new_edge_properties(edge_type, properties)?;
        self.require_node(src)?;
        self.require_node(dst)?;
        if let Some(validator) = &self.validator {
            self.check_new_edge(validator.as_ref(), src, dst, edge_type)?;
            convert_values(&mut properties, |key, value| {
                validator.convert_edge_property(edge_type, key, value)
            });
            validator.validate_edge_properties_declared(edge_type, &properties)?;
            for (name, value) in &properties {
                validator.validate_edge_property(edge_type, name, value)?;
            }
            validator.validate_edge_complete(edge_type, &properties)?;
        }
        self.claim(WriteClaim::Endpoints(src, dst))?;
        let _writing = self.write_in_progress();
        self.insert_edge(src, dst, edge_type, &properties)
    }

    /// Creates an edge whose remaining properties depend on the edge itself
    /// (MERGE `ON CREATE SET` expressions that read it), like
    /// [`create_node_with`](Self::create_node_with); the endpoints are
    /// checked and claimed as [`create_edge`](Self::create_edge) does.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`create_edge`](Self::create_edge), or
    /// `derive`'s error.
    pub fn create_edge_with(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        properties: Vec<(String, Value)>,
        derive: impl FnOnce(EdgeId) -> Result<Vec<(String, Value)>, OperatorError>,
    ) -> Result<EdgeId, OperatorError> {
        let mut properties = self.new_edge_properties(edge_type, properties)?;
        self.require_node(src)?;
        self.require_node(dst)?;
        if let Some(validator) = &self.validator {
            self.check_new_edge(validator.as_ref(), src, dst, edge_type)?;
            convert_values(&mut properties, |key, value| {
                validator.convert_edge_property(edge_type, key, value)
            });
            validator.validate_edge_properties_declared(edge_type, &properties)?;
            for (name, value) in &properties {
                validator.validate_edge_property(edge_type, name, value)?;
            }
        }
        self.claim(WriteClaim::Endpoints(src, dst))?;
        let id = {
            let _writing = self.write_in_progress();
            self.insert_edge(src, dst, edge_type, &properties)?
        };

        // Evaluates expressions: no write is in progress meanwhile.
        let mut derived = derive(id)?;
        refuse_too_deep(plain_values(&derived))?;
        if let Some(validator) = &self.validator {
            convert_values(&mut derived, |key, value| {
                validator.convert_edge_property(edge_type, key, value)
            });
            validator.validate_edge_properties_declared(edge_type, &derived)?;
            for (name, value) in &derived {
                validator.validate_edge_property(edge_type, name, value)?;
            }
            validator.validate_edge_complete(edge_type, &overlay(properties, &derived))?;
        }
        let _writing = self.write_in_progress();
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
        self.claim(WriteClaim::Edge(id))?;
        let mut assignments = Cow::Borrowed(assignments);
        if let Some(validator) = &self.validator
            && let Some(edge) = self.edge(id)
        {
            let edge_type = edge.edge_type.as_str();
            if let Some(converted) = convert_assignments(&assignments, |key, value| {
                validator.convert_edge_property(edge_type, key, value)
            }) {
                assignments = Cow::Owned(converted);
            }
            let existing = property_list(&edge.properties);
            let changes = expand_assignments(&existing, &assignments, replace);
            validator.validate_edge_properties_declared(edge_type, &changes)?;
            for (name, value) in changes {
                validator.validate_edge_property(edge_type, &name, &value)?;
            }
        }
        let _writing = self.write_in_progress();
        self.apply_set(Entity::Edge(id), &assignments, replace)?;
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
        self.claim(WriteClaim::Edge(id))?;
        let Some(edge) = self.edge(id) else {
            return Ok(false);
        };
        if edge.get_property(key).is_none() {
            return Ok(false);
        }
        if let Some(validator) = &self.validator {
            validator.validate_edge_property(edge.edge_type.as_str(), key, &Value::Null)?;
        }
        let _writing = self.write_in_progress();
        self.remove_value(Entity::Edge(id), key)?;
        Ok(true)
    }

    /// Deletes an edge.
    ///
    /// # Errors
    ///
    /// Returns a write conflict.
    pub fn delete_edge(&self, id: EdgeId) -> Result<bool, OperatorError> {
        let _writing = self.write_in_progress();
        self.remove_edge(id)
    }

    // === Bulk creates ===

    /// Creates one node with `labels` per property list of `rows`, in
    /// order, each checked as [`create_node`](Self::create_node) checks it:
    /// a row sees the rows before it (a `UNIQUE` value repeated within the
    /// rows is refused). Returns the ids in row order.
    ///
    /// With a recording whose recorder takes bulk writes
    /// ([`ChangeRecorder::bulk`]) the rows are one bulk write: their ids are
    /// one range reserved from the store and recorded as one entry before
    /// the first row is applied, which the commit stamps and a rollback
    /// undoes as a range; each row is applied without a before-image, and
    /// kept for the commit only when the recorder asks for the rows. Any
    /// other writer creates row by row.
    ///
    /// # Errors
    ///
    /// Returns the first row's error. The rows before it stay written, and
    /// so do their records (the range, or one entry per row): the
    /// transaction's rollback, or the statement's, takes them back.
    pub fn create_nodes(
        &self,
        labels: &[String],
        rows: Vec<Vec<(String, Value)>>,
    ) -> Result<Vec<NodeId>, OperatorError> {
        let Some((recording, target, keep)) = self.bulk() else {
            return rows
                .into_iter()
                .map(|properties| self.create_node(labels, properties))
                .collect();
        };
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let ids = target
            .reserve_node_ids(row_count(rows.len())?)
            .map_err(store_refused)?;
        recording.recorder.record_bulk(Table::Nodes, ids.clone())?;
        let writer = recording.recorder.writer();
        let distinct = distinct_labels(labels);
        let mut kept = (keep == BulkRows::Keep).then(|| Vec::with_capacity(rows.len()));
        let mut created = Vec::with_capacity(rows.len());
        let mut writing = None;
        for (row, (properties, raw)) in rows.into_iter().zip(ids).enumerate() {
            let properties = self.new_node_properties(labels, properties)?;
            if let Some(validator) = &self.validator {
                validator.validate_node_properties_declared(labels, &properties)?;
                self.check_node_values(validator.as_ref(), labels, &properties, None)?;
                validator.validate_node_complete(labels, &properties)?;
                validator.check_unique_node(labels, &properties, None)?;
            }
            if row % BULK_RUN == 0 {
                // Released before it is asked for again: not reentrant.
                drop(writing.take());
                writing = self.write_in_progress();
            }
            let id = NodeId::new(raw);
            let op = DataOp::CreateNode {
                id,
                labels: distinct.clone(),
                properties: present_values(&properties),
            };
            target.apply_bulk_row(&op, writer).map_err(store_refused)?;
            self.count(|c| &c.nodes_created, 1);
            self.count(|c| &c.labels_added, distinct.len());
            self.count(|c| &c.properties_set, op_values(&op));
            if let Some(kept) = &mut kept {
                kept.push(op);
            }
            created.push(id);
        }
        drop(writing);
        if let Some(rows) = kept {
            recording.recorder.record_bulk_rows(rows)?;
        }
        Ok(created)
    }

    /// Creates the edges of `edges`, in order, each checked as
    /// [`create_edge`](Self::create_edge) checks it. Returns the ids in
    /// order.
    ///
    /// With a recording whose recorder takes bulk writes, the edges are one
    /// bulk write, as [`create_nodes`](Self::create_nodes) describes. Each
    /// endpoint is checked and claimed once per call, before the first edge
    /// that names it is written; the new edges are not claimed: their ids
    /// are the call's reserved range, which no other transaction can name,
    /// and a concurrent delete of an endpoint meets the endpoint's claim.
    ///
    /// # Errors
    ///
    /// Returns the first edge's error, as
    /// [`create_nodes`](Self::create_nodes) does.
    pub fn create_edges(&self, edges: Vec<NewEdge>) -> Result<Vec<EdgeId>, OperatorError> {
        let Some((recording, target, keep)) = self.bulk() else {
            return edges
                .into_iter()
                .map(|edge| self.create_edge(edge.src, edge.dst, &edge.edge_type, edge.properties))
                .collect();
        };
        if edges.is_empty() {
            return Ok(Vec::new());
        }
        let ids = target
            .reserve_edge_ids(row_count(edges.len())?)
            .map_err(store_refused)?;
        recording.recorder.record_bulk(Table::Edges, ids.clone())?;
        let writer = recording.recorder.writer();
        let mut kept = (keep == BulkRows::Keep).then(|| Vec::with_capacity(edges.len()));
        let mut created = Vec::with_capacity(edges.len());
        // The endpoints seen and claimed: the transaction's view of them
        // holds for the call, which deletes nothing.
        let mut endpoints: FxHashSet<NodeId> = FxHashSet::default();
        let mut edge_type: Option<ArcStr> = None;
        let mut writing = None;
        for (row, (edge, raw)) in edges.into_iter().zip(ids).enumerate() {
            let NewEdge {
                src,
                dst,
                edge_type: name,
                properties,
            } = edge;
            let mut properties = self.new_edge_properties(&name, properties)?;
            let new_src = !endpoints.contains(&src);
            let new_dst = !endpoints.contains(&dst);
            // A batch names its endpoints by id: one this transaction does
            // not see is "not found", as a single create through the direct
            // API reports it.
            for (new, endpoint) in [(new_src, src), (new_dst, dst)] {
                if new && !self.has_node(endpoint) {
                    return Err(OperatorError::from(
                        grafeo_common::utils::error::Error::NodeNotFound(endpoint),
                    ));
                }
            }
            if let Some(validator) = &self.validator {
                self.check_new_edge(validator.as_ref(), src, dst, &name)?;
                convert_values(&mut properties, |key, value| {
                    validator.convert_edge_property(&name, key, value)
                });
                validator.validate_edge_properties_declared(&name, &properties)?;
                for (key, value) in &properties {
                    validator.validate_edge_property(&name, key, value)?;
                }
                validator.validate_edge_complete(&name, &properties)?;
            }
            if new_src || new_dst {
                self.claim(WriteClaim::Endpoints(src, dst))?;
                endpoints.insert(src);
                endpoints.insert(dst);
            }
            if row % BULK_RUN == 0 {
                drop(writing.take());
                writing = self.write_in_progress();
            }
            // One name per type, not one per edge.
            let edge_type = match &edge_type {
                Some(shared) if shared.as_str() == name => shared.clone(),
                _ => edge_type.insert(ArcStr::from(name.as_str())).clone(),
            };
            let id = EdgeId::new(raw);
            let op = DataOp::CreateEdge {
                id,
                src,
                dst,
                edge_type,
                properties: present_values(&properties),
            };
            target.apply_bulk_row(&op, writer).map_err(store_refused)?;
            self.count(|c| &c.edges_created, 1);
            self.count(|c| &c.properties_set, op_values(&op));
            if let Some(kept) = &mut kept {
                kept.push(op);
            }
            created.push(id);
        }
        drop(writing);
        if let Some(rows) = kept {
            recording.recorder.record_bulk_rows(rows)?;
        }
        Ok(created)
    }

    /// The recording, store and row handling of a bulk write: `Some` when
    /// the writer records into a recorder that takes bulk writes, through a
    /// store that creates at ids reserved from it.
    fn bulk(&self) -> Option<(&Recording, &Arc<dyn ChangeTarget>, BulkRows)> {
        let recording = self.recording.as_ref()?;
        let WriteTarget::Store(target) = &recording.target else {
            return None;
        };
        let keep = recording.recorder.bulk()?;
        Some((recording, target, keep))
    }

    /// [`delete_edge`](Self::delete_edge), for a caller whose write is
    /// already in progress.
    fn remove_edge(&self, id: EdgeId) -> Result<bool, OperatorError> {
        self.claim(WriteClaim::Edge(id))?;
        let deleted = match &self.recording {
            Some(recording) => self.apply_op(recording, DataOp::DeleteEdge { id })?,
            None => self
                .store
                .delete_edge_versioned(id, self.epoch(), self.transaction()),
        };
        self.count(|c| &c.edges_deleted, usize::from(deleted));
        Ok(deleted)
    }

    // === Checks ===

    /// The properties a new node with `labels` gets: `properties` with the
    /// validator's type defaults added and each value converted to the type
    /// the schema declares for it, once the labels are allowed. No value may
    /// nest too deep, a default included: a custom validator's default is
    /// written like any other value.
    fn new_node_properties(
        &self,
        labels: &[String],
        mut properties: Vec<(String, Value)>,
    ) -> Result<Vec<(String, Value)>, OperatorError> {
        if let Some(validator) = &self.validator {
            validator.validate_node_labels_allowed(labels)?;
            validator.inject_defaults(labels, &mut properties);
            convert_values(&mut properties, |key, value| {
                validator.convert_node_property(labels, key, value)
            });
        }
        refuse_too_deep(plain_values(&properties))?;
        Ok(properties)
    }

    /// The properties a new edge of `edge_type` gets: `properties` with the
    /// validator's type defaults added. No value may nest too deep, a
    /// default included, as for a node.
    fn new_edge_properties(
        &self,
        edge_type: &str,
        mut properties: Vec<(String, Value)>,
    ) -> Result<Vec<(String, Value)>, OperatorError> {
        if let Some(validator) = &self.validator {
            validator.inject_edge_defaults(edge_type, &mut properties);
        }
        refuse_too_deep(plain_values(&properties))?;
        Ok(properties)
    }

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
        validator.validate_node_properties_declared(&labels, &changes)?;
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

    /// Creates a node with `labels` and `properties` (checked already); a
    /// null value is no property.
    fn insert_node(
        &self,
        labels: &[String],
        properties: &[(String, Value)],
    ) -> Result<NodeId, OperatorError> {
        if let Some(recording) = &self.recording {
            return self.create_recorded_node(recording, labels, properties);
        }
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let id = self
            .store
            .create_node_versioned(&label_refs, self.epoch(), self.transaction());
        self.claim(WriteClaim::Node(id))?;
        self.count(|c| &c.nodes_created, 1);
        // `(:A:A)` gives the node one label.
        let distinct: std::collections::BTreeSet<&str> = label_refs.into_iter().collect();
        self.count(|c| &c.labels_added, distinct.len());
        self.write_values(Entity::Node(id), properties)?;
        Ok(id)
    }

    /// Creates a node through the recording's target as one entry, with its
    /// labels and every value of `properties` that is not null: at an id
    /// reserved from the store, or the one an external store gives. It
    /// claims nothing: no other transaction knows the new id before this one
    /// commits.
    fn create_recorded_node(
        &self,
        recording: &Recording,
        labels: &[String],
        properties: &[(String, Value)],
    ) -> Result<NodeId, OperatorError> {
        let labels = distinct_labels(labels);
        let values = present_values(properties);
        let (label_count, value_count) = (labels.len(), values.len());
        let writer = recording.recorder.writer();
        let (id, op, applied) = match &recording.target {
            WriteTarget::Store(target) => {
                let id = NodeId::new(target.reserve_node_ids(1).map_err(store_refused)?.start);
                let op = DataOp::CreateNode {
                    id,
                    labels,
                    properties: values,
                };
                let applied = target.apply(&op, writer).map_err(store_refused)?;
                (id, op, applied)
            }
            WriteTarget::External(target) => {
                let (op, applied) = target
                    .create_node(labels, values, writer)
                    .map_err(store_refused)?;
                let DataOp::CreateNode { id, .. } = op else {
                    return Err(OperatorError::Internal(
                        "an external store's node create gave another op".to_string(),
                    ));
                };
                (id, op, applied)
            }
        };
        if !record_applied(recording, op, applied)? {
            return Err(OperatorError::Internal(format!(
                "the store created no node {}",
                id.as_u64()
            )));
        }
        self.count(|c| &c.nodes_created, 1);
        self.count(|c| &c.labels_added, label_count);
        self.count(|c| &c.properties_set, value_count);
        Ok(id)
    }

    /// Creates an edge (checked already, its endpoints claimed) with
    /// `properties`; a null value is no property.
    fn insert_edge(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        properties: &[(String, Value)],
    ) -> Result<EdgeId, OperatorError> {
        if let Some(recording) = &self.recording {
            return self.create_recorded_edge(recording, src, dst, edge_type, properties);
        }
        let id = self
            .store
            .create_edge_versioned(src, dst, edge_type, self.epoch(), self.transaction())
            .map_err(conflict_or_refused)?;
        self.claim(WriteClaim::Edge(id))?;
        self.count(|c| &c.edges_created, 1);
        self.write_values(Entity::Edge(id), properties)?;
        Ok(id)
    }

    /// Creates an edge through the recording's target as one entry, with
    /// every value of `properties` that is not null: at an id reserved from
    /// the store and claimed before the store changes (a concurrent
    /// `DETACH DELETE` of an endpoint lists the edge), or at the id an
    /// external store gives, claimed once it is known.
    fn create_recorded_edge(
        &self,
        recording: &Recording,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        properties: &[(String, Value)],
    ) -> Result<EdgeId, OperatorError> {
        let values = present_values(properties);
        let value_count = values.len();
        let edge_type = ArcStr::from(edge_type);
        let writer = recording.recorder.writer();
        let (id, op, applied) = match &recording.target {
            WriteTarget::Store(target) => {
                let id = EdgeId::new(target.reserve_edge_ids(1).map_err(store_refused)?.start);
                self.claim(WriteClaim::Edge(id))?;
                let op = DataOp::CreateEdge {
                    id,
                    src,
                    dst,
                    edge_type,
                    properties: values,
                };
                let applied = target.apply(&op, writer).map_err(store_refused)?;
                (id, op, applied)
            }
            WriteTarget::External(target) => {
                let (op, applied) = target
                    .create_edge(src, dst, edge_type, values, writer)
                    .map_err(store_refused)?;
                let DataOp::CreateEdge { id, .. } = op else {
                    return Err(OperatorError::Internal(
                        "an external store's edge create gave another op".to_string(),
                    ));
                };
                (id, op, applied)
            }
        };
        if !record_applied(recording, op, applied)? {
            return Err(OperatorError::Internal(format!(
                "the store created no edge {}",
                id.as_u64()
            )));
        }
        if matches!(recording.target, WriteTarget::External(_)) {
            self.claim(WriteClaim::Edge(id))?;
        }
        self.count(|c| &c.edges_created, 1);
        self.count(|c| &c.properties_set, value_count);
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
        if let Some(recording) = &self.recording {
            let key = PropertyKey::new(key);
            let op = match entity {
                Entity::Node(id) => DataOp::SetNodeProperty { id, key, value },
                Entity::Edge(id) => DataOp::SetEdgeProperty { id, key, value },
            };
            let changed = self.apply_op(recording, op)?;
            self.count(|c| &c.properties_set, usize::from(changed));
            return Ok(());
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
        if let Some(recording) = &self.recording {
            let key = PropertyKey::new(key);
            let op = match entity {
                Entity::Node(id) => DataOp::RemoveNodeProperty { id, key },
                Entity::Edge(id) => DataOp::RemoveEdgeProperty { id, key },
            };
            let removed = self.apply_op(recording, op)?;
            self.count(|c| &c.properties_set, usize::from(removed));
            return Ok(());
        }
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
    OperatorError::from(error)
}

/// A write the store refused: a write conflict stays one (a compacted store
/// refuses an edge to a node another transaction deleted), anything else is
/// [`refused`].
fn conflict_or_refused(error: grafeo_common::utils::error::Error) -> OperatorError {
    use grafeo_common::utils::error::{Error, TransactionError};

    match error {
        Error::Transaction(TransactionError::WriteConflict(message)) => {
            OperatorError::WriteConflict(message)
        }
        other => refused(other),
    }
}

/// Records in `recording` what the store reported it applied for `op`;
/// whether the write changed something. A write that changed nothing
/// records nothing.
fn record_applied(
    recording: &Recording,
    op: DataOp,
    applied: Applied,
) -> Result<bool, OperatorError> {
    match applied {
        Applied::Changed { before, version } => {
            recording.recorder.record(op, before, version)?;
            Ok(true)
        }
        Applied::Unchanged => Ok(false),
        // Committed at once without a before-image (an immediate write whose
        // entries nothing reads): nothing to record.
        Applied::Committed => Ok(true),
    }
}

/// The statement error of a write the store's change target refused: a
/// node delete while the node has edges is the constraint a user sees,
/// anything else (a spilled value that cannot be read, a store limit) an
/// execution error, as [`refused`].
fn store_refused(error: ApplyError) -> OperatorError {
    match error {
        ApplyError::HasEdges(id) => OperatorError::ConstraintViolation(format!(
            "Cannot delete node {} with connected edge(s). Use DETACH DELETE.",
            id.as_u64()
        )),
        other => OperatorError::Execution(other.to_string()),
    }
}

/// The labels of a new node, each once, in the order given: `(:A:A)` gives
/// the node one label.
fn distinct_labels(labels: &[String]) -> Labels {
    let mut distinct = Labels::new();
    for label in labels {
        if !distinct.iter().any(|seen| seen.as_str() == label) {
            distinct.push(ArcStr::from(label.as_str()));
        }
    }
    distinct
}

/// The values of a new entity's properties, in the order given: a null is
/// no property.
fn present_values(properties: &[(String, Value)]) -> Properties {
    properties
        .iter()
        .filter(|(_, value)| !value.is_null())
        .map(|(key, value)| (PropertyKey::new(key.as_str()), value.clone()))
        .collect()
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

/// Replaces each value of a property list that `convert` converts to the
/// type the schema declares for its key (see
/// [`ConstraintValidator::convert_node_property`]).
fn convert_values(
    properties: &mut [(String, Value)],
    convert: impl Fn(&str, &Value) -> Option<Value>,
) {
    for (key, value) in properties {
        if let Some(converted) = convert(key, value) {
            *value = converted;
        }
    }
}

/// The assignments of a SET with each value that `convert` converts
/// replaced, also the entries of a map assignment. `None` when no value
/// converts.
fn convert_assignments(
    assignments: &[(String, Value)],
    convert: impl Fn(&str, &Value) -> Option<Value>,
) -> Option<Vec<(String, Value)>> {
    let convert_one = |name: &str, value: &Value| -> Option<Value> {
        match value {
            Value::Map(map) if name == MAP_ASSIGNMENT => {
                let mut converted_any = false;
                let entries = map
                    .iter()
                    .map(|(key, entry)| {
                        let converted = convert(key.as_str(), entry);
                        converted_any |= converted.is_some();
                        (key.clone(), converted.unwrap_or_else(|| entry.clone()))
                    })
                    .collect();
                converted_any.then(|| Value::Map(Arc::new(entries)))
            }
            _ if name == MAP_ASSIGNMENT => None,
            _ => convert(name, value),
        }
    };
    let converted: Vec<Option<Value>> = assignments
        .iter()
        .map(|(name, value)| convert_one(name, value))
        .collect();
    if converted.iter().all(Option::is_none) {
        return None;
    }
    Some(
        assignments
            .iter()
            .zip(converted)
            .map(|((name, value), converted)| {
                (name.clone(), converted.unwrap_or_else(|| value.clone()))
            })
            .collect(),
    )
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
    use std::time::Duration;

    use grafeo_common::storage::value_codec::MAX_PROPERTY_VALUE_DEPTH;
    use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};

    use super::{GraphWriter, OperatorError};
    use crate::graph::GraphStoreMut;
    use crate::graph::lpg::LpgStore;

    /// A string inside `depth` lists.
    pub(super) fn nested(depth: usize) -> Value {
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

    pub(super) fn labels(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_string()).collect()
    }

    pub(super) fn pairs(key: &str, value: &Value) -> Vec<(String, Value)> {
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

    /// A validator that accepts every value, gives every node a `trips`
    /// default nested deeper than a database can store, as a custom
    /// [`ConstraintValidator`](super::ConstraintValidator) may, and refuses
    /// edges of type `HATES` and edge properties named `grudge`.
    pub(super) struct CustomRules;

    impl super::ConstraintValidator for CustomRules {
        fn validate_node_property(
            &self,
            _: &[String],
            _: &str,
            _: &Value,
        ) -> Result<(), OperatorError> {
            Ok(())
        }

        fn validate_node_complete(
            &self,
            _: &[String],
            _: &[(String, Value)],
        ) -> Result<(), OperatorError> {
            Ok(())
        }

        fn check_unique_node_property(
            &self,
            _: &[String],
            _: &str,
            _: &Value,
        ) -> Result<(), OperatorError> {
            Ok(())
        }

        fn validate_edge_property(
            &self,
            _: &str,
            key: &str,
            _: &Value,
        ) -> Result<(), OperatorError> {
            if key == "grudge" {
                return Err(OperatorError::ConstraintViolation("no grudges".to_string()));
            }
            Ok(())
        }

        fn validate_edge_complete(
            &self,
            _: &str,
            _: &[(String, Value)],
        ) -> Result<(), OperatorError> {
            Ok(())
        }

        fn validate_edge_type_allowed(&self, edge_type: &str) -> Result<(), OperatorError> {
            if edge_type == "HATES" {
                return Err(OperatorError::ConstraintViolation("no hate".to_string()));
            }
            Ok(())
        }

        fn inject_defaults(&self, _: &[String], properties: &mut Vec<(String, Value)>) {
            properties.push(("trips".to_string(), nested(MAX_PROPERTY_VALUE_DEPTH + 1)));
        }

        /// A `ROUTE` gets `km: 88` unless given one; a `TRAVELS` gets a
        /// `legs` default nested deeper than a database can store.
        fn inject_edge_defaults(&self, edge_type: &str, properties: &mut Vec<(String, Value)>) {
            match edge_type {
                "ROUTE" if !properties.iter().any(|(key, _)| key == "km") => {
                    properties.push(("km".to_string(), Value::Int64(88)));
                }
                "TRAVELS" => {
                    properties.push(("legs".to_string(), nested(MAX_PROPERTY_VALUE_DEPTH + 1)));
                }
                _ => {}
            }
        }
    }

    /// The validator's edge type defaults fill the properties a new edge is
    /// not given, through `create_edge` and `create_edge_with`: a value the
    /// caller gives or MERGE derives wins.
    #[test]
    fn edge_defaults_fill_what_the_caller_leaves_out() {
        let (store, writer) = writer();
        let writer = writer.with_validator(Arc::new(CustomRules));
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        let km = PropertyKey::new("km");
        let km_of = |edge| store.get_edge_property(edge, &km);

        let route = writer.create_edge(alix, gus, "ROUTE", Vec::new()).unwrap();
        assert_eq!(km_of(route), Some(Value::Int64(88)), "create_edge");
        let given = writer
            .create_edge(alix, gus, "ROUTE", pairs("km", &Value::Int64(3)))
            .unwrap();
        assert_eq!(km_of(given), Some(Value::Int64(3)), "a given value");
        let merged = writer
            .create_edge_with(alix, gus, "ROUTE", Vec::new(), |_| Ok(Vec::new()))
            .unwrap();
        assert_eq!(km_of(merged), Some(Value::Int64(88)), "create_edge_with");
        let derived = writer
            .create_edge_with(alix, gus, "ROUTE", Vec::new(), |_| {
                Ok(pairs("km", &Value::Int64(19)))
            })
            .unwrap();
        assert_eq!(km_of(derived), Some(Value::Int64(19)), "a derived value");
        let knows = writer.create_edge(alix, gus, "KNOWS", Vec::new()).unwrap();
        assert_eq!(km_of(knows), None, "another edge type has no default");
    }

    /// A default the validator adds is checked as a value the caller gives
    /// is: one nested too deep is refused before the node or edge is
    /// written.
    #[test]
    fn a_default_nested_deeper_than_a_file_holds_is_refused_before_any_write() {
        let (store, writer) = writer();
        let writer = writer.with_validator(Arc::new(CustomRules));
        let refused_trips = |result: &Result<NodeId, OperatorError>| {
            matches!(result, Err(OperatorError::ConstraintViolation(message))
                if message.contains("\"trips\""))
        };

        let created = writer.create_node(&labels(&["Person"]), pairs("name", &Value::from("Alix")));
        assert!(refused_trips(&created), "create_node: {created:?}");
        let merged = writer.create_node_with(&labels(&["Person"]), Vec::new(), |_| {
            Ok(pairs("name", &Value::from("Gus")))
        });
        assert!(refused_trips(&merged), "create_node_with: {merged:?}");
        assert_eq!(store.node_count(), 0, "no node with the default is written");

        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        let refused_legs = |result: &Result<EdgeId, OperatorError>| {
            matches!(result, Err(OperatorError::ConstraintViolation(message))
                if message.contains("\"legs\""))
        };
        let created = writer.create_edge(alix, gus, "TRAVELS", Vec::new());
        assert!(refused_legs(&created), "create_edge: {created:?}");
        let merged = writer.create_edge_with(alix, gus, "TRAVELS", Vec::new(), |_| Ok(Vec::new()));
        assert!(refused_legs(&merged), "create_edge_with: {merged:?}");
        assert_eq!(store.edge_count(), 0, "no edge with the default is written");
    }

    // === Every write method ===

    /// How long a write that should wait gets to finish anyway.
    pub(super) const BRIEFLY: Duration = Duration::from_millis(100);

    /// The committed graph a transaction writes to: Alix (in Amsterdam), who
    /// knows Gus since 3, and Vincent, who has no edges.
    pub(super) struct People {
        pub(super) alix: NodeId,
        pub(super) gus: NodeId,
        pub(super) vincent: NodeId,
        pub(super) knows: EdgeId,
    }

    /// A store holding [`People`], committed. Every call builds the same
    /// store, with the same ids.
    pub(super) fn people_store() -> (Arc<LpgStore>, People) {
        let store = Arc::new(LpgStore::new().unwrap());
        let alix = store.create_node(&["Person"]);
        store.set_node_property(alix, "city", Value::from("Amsterdam"));
        let gus = store.create_node(&["Person"]);
        let vincent = store.create_node(&["Person"]);
        let knows = store.create_edge(alix, gus, "KNOWS");
        store.set_edge_property(knows, "since", Value::Int64(3));
        let people = People {
            alix,
            gus,
            vincent,
            knows,
        };
        (store, people)
    }

    /// A write method of a transaction, called on [`People`].
    pub(super) type Write = fn(&GraphWriter, &People) -> Result<(), OperatorError>;

    /// Every write method of [`GraphWriter`], each changing what the
    /// transaction sees.
    pub(super) fn every_write() -> Vec<(&'static str, Write)> {
        vec![
            ("create_node", |writer, _| {
                writer
                    .create_node(&labels(&["Person"]), pairs("name", &Value::from("Mia")))
                    .map(drop)
            }),
            ("create_node_with", |writer, _| {
                writer
                    .create_node_with(&labels(&["Person"]), Vec::new(), |_| {
                        Ok(pairs("name", &Value::from("Jules")))
                    })
                    .map(drop)
            }),
            ("set_node_properties", |writer, people| {
                writer.set_node_properties(
                    people.alix,
                    &pairs("city", &Value::from("Prague")),
                    false,
                )
            }),
            ("remove_node_property", |writer, people| {
                writer.remove_node_property(people.alix, "city").map(drop)
            }),
            ("add_labels", |writer, people| {
                writer
                    .add_labels(people.alix, &labels(&["Traveller"]))
                    .map(drop)
            }),
            ("remove_labels", |writer, people| {
                writer
                    .remove_labels(people.alix, &labels(&["Person"]))
                    .map(drop)
            }),
            ("delete_node detaching", |writer, people| {
                writer.delete_node(people.gus, true).map(drop)
            }),
            ("delete_node", |writer, people| {
                writer.delete_node(people.vincent, false).map(drop)
            }),
            ("create_edge", |writer, people| {
                writer
                    .create_edge(
                        people.alix,
                        people.vincent,
                        "KNOWS",
                        pairs("since", &Value::Int64(19)),
                    )
                    .map(drop)
            }),
            ("create_edge_with", |writer, people| {
                writer
                    .create_edge_with(people.alix, people.vincent, "KNOWS", Vec::new(), |_| {
                        Ok(pairs("since", &Value::Int64(88)))
                    })
                    .map(drop)
            }),
            ("set_edge_properties", |writer, people| {
                writer.set_edge_properties(people.knows, &pairs("since", &Value::Int64(88)), false)
            }),
            ("remove_edge_property", |writer, people| {
                writer.remove_edge_property(people.knows, "since").map(drop)
            }),
            ("delete_edge", |writer, people| {
                writer.delete_edge(people.knows).map(drop)
            }),
        ]
    }
}

/// The writer with a recording: through a change target, into a change set.
#[cfg(all(test, feature = "lpg"))]
#[path = "writer_recording_tests.rs"]
mod recording_tests;
