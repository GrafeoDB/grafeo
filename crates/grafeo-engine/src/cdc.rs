//! Change Data Capture (CDC) for tracking entity mutations.
//!
//! When the `cdc` feature is enabled, the database records every mutation
//! (create, update, delete) with before/after property snapshots. This
//! enables audit trails, temporal queries, and downstream sync.
//!
//! # Event model
//!
//! Every write emits a [`ChangeEvent`] keyed by [`EntityId`]. Three id
//! variants cover the three supported data models: `Node(NodeId)` and
//! `Edge(EdgeId)` for LPG, `Triple(u64)` for RDF (hashed via a
//! content-stable triple hash so the log can key events by triple
//! content without maintaining a separate id registry).
//! [`ChangeKind`] distinguishes
//! `Create` / `Update` / `Delete`; the `before` and `after` property maps
//! are populated only for the variants that can carry them (`Update` has
//! both; `Create` has only `after`; `Delete` has only `before`).
//!
//! The extra edge/label/triple fields on [`ChangeEvent`] are skipped in
//! serialisation when absent so JSON consumers see a clean shape per
//! variant. Adding a new field is a non-breaking change when it is also
//! `#[serde(skip_serializing_if = "Option::is_none")]`.
//!
//! # Thread safety and ordering
//!
//! [`CdcLog`] is the authoritative in-memory store, a
//! `RwLock<HbHashMap<(graph, EntityId), Vec<ChangeEvent>>>` (entity ids
//! repeat across graphs, so the graph is part of the key). Readers (history
//! queries, retention probes) take the read lock; writers (commit-path
//! recording) take the write lock. We use [`hashbrown`]'s `HashMap` for
//! the Fx-hashed inner map, and [`parking_lot`]'s `RwLock` for cheap
//! uncontended locking; under concurrent insert+read the typical pattern
//! is short write critical sections (one `entry().or_default().push()`
//! per event) and occasional long read critical sections for
//! `history*()` snapshots that clone the event vector out of the lock.
//!
//! Per-entity event order within the `Vec<ChangeEvent>` is insertion
//! order, which is monotonic by construction: the only writer is the
//! commit path (see "Integration with the commit path" below) and each
//! event carries the MVCC `epoch` that produced it plus an HLC
//! timestamp from the embedded [`HlcClock`]. The clock guarantees
//! strictly-increasing timestamps across all threads within a process
//! (wall-clock ms in the upper 48 bits, logical counter in the lower
//! 16, with the logical bits bumped on collision).
//!
//! # Integration with the commit path
//!
//! A transaction's writes (statements, the direct API of a session or of
//! the database, batches) are recorded in its change set; nothing reaches
//! the log while it runs. Its commit builds its events from the change set
//! (one per change, folded into the creates of the entities it created)
//! and records them in the commit's ordered step, before its epoch is
//! published: commits complete one at a time in epoch order, so the events
//! carry the commit epoch and timestamps that follow the epochs, and a
//! reader that sees a commit through MVCC also sees its events. A rollback,
//! a rollback to a savepoint and a statement that fails report nothing.
//! Bulk imports report no events.
//!
//! # CDC epoch vs. MVCC epoch
//!
//! The `epoch` field on [`ChangeEvent`] is the MVCC [`EpochId`] produced
//! by the [`TransactionManager`](crate::transaction::TransactionManager)
//! at commit time. It is the same epoch that tags the resulting
//! [`LpgStore`](grafeo_core::graph::lpg::LpgStore) version, so
//! `history_since(id, epoch)` and `MATCH ... AT EPOCH` return a
//! consistent view. The CDC log has no epoch of its own: retention is
//! driven by external epoch advances through
//! [`apply_retention()`](CdcLog::apply_retention), called from
//! [`GrafeoDB::gc()`](crate::GrafeoDB::gc) under the same epoch used to
//! GC MVCC version chains.
//!
//! # Why in-memory only
//!
//! The CDC log is not persisted: it lives only in the
//! `RwLock<HashMap<_, Vec<ChangeEvent>>>` above, is allocated fresh by
//! [`CdcLog::new`] at database open, and a crash+recover cycle loses
//! the entire event history. The WAL replay path
//! ([`GrafeoDB::apply_wal_records`](crate::GrafeoDB)) rebuilds
//! [`LpgStore`](grafeo_core::graph::lpg::LpgStore) state by calling
//! mutation methods directly, bypassing the CDC recording sites in
//! `database::crud`, so no events are re-emitted. The WAL does contain
//! the same sequence of mutations a consumer would need to rebuild an
//! equivalent history, but that reconstruction is not wired up today.
//!
//! Keeping CDC in memory avoids a second persistent log with its own
//! crash-consistency semantics and keeps the commit path off the
//! disk-flush critical path. The retention limits in
//! [`CdcRetentionConfig`] therefore protect working set, not
//! durability; see [#250][] for the unbounded-growth incident that
//! motivated the retention knobs.
//!
//! The planned reactive event bus (see [#470][]) generalises this pattern:
//! CDC becomes one `MutationListener` among many, the recording
//! path moves behind a trait, and other listeners (cache invalidation,
//! replication, scoring hooks) register alongside. The in-process API
//! surface here is the one CDC binding that carries forward; the rest
//! of the file is effectively "the first listener".
//!
//! [#250]: https://github.com/GrafeoDB/grafeo/issues/250
//! [#470]: https://github.com/GrafeoDB/grafeo/issues/470
//!
//! # Example
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use grafeo_engine::GrafeoDB;
//! use grafeo_common::types::Value;
//!
//! let db = GrafeoDB::new_in_memory();
//! let id = db.create_node(&["Person"])?;
//! db.set_node_property(id, "name", Value::from("Alix"))?;
//! db.set_node_property(id, "name", Value::from("Gus"))?;
//!
//! let history = db.history(id)?;
//! assert_eq!(history.len(), 3); // create + 2 updates
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use grafeo_common::memory::buffer::{MemoryConsumer, MemoryRegion, priorities};
use grafeo_common::types::{EdgeId, EpochId, HlcClock, HlcTimestamp, NodeId, Value};
use hashbrown::HashMap as HbHashMap;
use parking_lot::RwLock;

/// The kind of mutation that occurred.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum ChangeKind {
    /// A new entity was created.
    Create,
    /// An existing entity was updated (property set or removed).
    Update,
    /// An entity was deleted.
    Delete,
}

/// A unique identifier for a graph entity (node, edge, or RDF triple).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum EntityId {
    /// A node identifier.
    Node(NodeId),
    /// An edge identifier.
    Edge(EdgeId),
    /// An RDF triple, identified by a content hash of its terms.
    Triple(u64),
}

impl From<NodeId> for EntityId {
    fn from(id: NodeId) -> Self {
        Self::Node(id)
    }
}

impl From<EdgeId> for EntityId {
    fn from(id: EdgeId) -> Self {
        Self::Edge(id)
    }
}

impl EntityId {
    /// Returns the raw u64 value for binding layers.
    #[must_use]
    pub fn as_u64(&self) -> u64 {
        match self {
            Self::Node(id) => id.as_u64(),
            Self::Edge(id) => id.as_u64(),
            Self::Triple(h) => *h,
        }
    }

    /// Returns `true` if this is a node identifier.
    #[must_use]
    pub fn is_node(&self) -> bool {
        matches!(self, Self::Node(_))
    }

    /// Returns `true` if this is an RDF triple identifier.
    #[must_use]
    pub fn is_triple(&self) -> bool {
        matches!(self, Self::Triple(_))
    }
}

/// A recorded change event with before/after property snapshots, or an RDF
/// triple insert/delete.
///
/// Later releases may add fields, so outside this crate an event is read,
/// not built: a pattern names its fields with `..`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct ChangeEvent {
    /// The entity that was changed.
    pub entity_id: EntityId,
    /// The graph the entity is in: its storage key (`name`, or `schema/name`
    /// inside a schema), `None` for the default graph. Entity ids repeat
    /// across graphs, so an id names an entity only together with this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph: Option<String>,
    /// The kind of change.
    pub kind: ChangeKind,
    /// MVCC epoch when the change occurred.
    pub epoch: EpochId,
    /// Hybrid Logical Clock timestamp for causal ordering.
    ///
    /// Encodes physical milliseconds (upper 48 bits) and a logical counter
    /// (lower 16 bits) into a `u64`. Backward-compatible: plain wall-clock
    /// values have logical counter = 0.
    pub timestamp: HlcTimestamp,
    /// Properties before the change (None for Create and for triple events).
    pub before: Option<HashMap<String, Value>>,
    /// Properties after the change (None for Delete and for triple events).
    pub after: Option<HashMap<String, Value>>,
    /// Node labels: on a Create event the node's labels, on a Delete event
    /// the labels it had, on a label change the labels after the change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub labels: Option<Vec<String>>,
    /// Node labels before a label change. Present only on label changes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before_labels: Option<Vec<String>>,
    /// Edge relationship type. Present on edge Create and Delete events.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edge_type: Option<String>,
    /// Edge source node ID. Present on edge Create and Delete events.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub src_id: Option<u64>,
    /// Edge destination node ID. Present on edge Create and Delete events.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dst_id: Option<u64>,
    /// RDF triple subject (N-Triples encoded). Present only on triple events.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub triple_subject: Option<String>,
    /// RDF triple predicate (N-Triples encoded). Present only on triple events.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub triple_predicate: Option<String>,
    /// RDF triple object (N-Triples encoded). Present only on triple events.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub triple_object: Option<String>,
    /// Named graph containing the triple. `None` means the default graph.
    /// Present only on triple events.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub triple_graph: Option<String>,
}

/// Configuration for CDC event retention.
///
/// Controls how many events the CDC log keeps in memory. When limits are
/// exceeded, the oldest events (by epoch) are pruned automatically.
///
/// Start from [`CdcRetentionConfig::default()`] (1,000 epochs and 100,000
/// events) or [`CdcRetentionConfig::unlimited()`] and set a limit with
/// [`with_max_epochs`](Self::with_max_epochs) and
/// [`with_max_events`](Self::with_max_events). Later releases may add
/// settings, so outside this crate it cannot be built with a struct literal.
///
/// # Examples
///
/// ```
/// use grafeo_engine::Config;
/// use grafeo_engine::cdc::CdcRetentionConfig;
///
/// let config = Config::in_memory()
///     .with_cdc()
///     .with_cdc_retention(CdcRetentionConfig::unlimited().with_max_events(88_000));
/// assert_eq!(config.cdc_retention.max_epochs, None);
/// ```
///
/// ```compile_fail,E0639
/// use grafeo_engine::cdc::CdcRetentionConfig;
///
/// let retention = CdcRetentionConfig {
///     max_epochs: None,
///     max_events: Some(88_000),
/// };
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CdcRetentionConfig {
    /// Maximum number of epochs to retain. Events older than
    /// `current_epoch - max_epochs` are pruned during GC.
    /// `None` disables epoch-based pruning.
    pub max_epochs: Option<u64>,
    /// Maximum total event count across all entities. Oldest events
    /// (by epoch) are pruned when this limit is exceeded.
    /// `None` disables count-based pruning.
    pub max_events: Option<usize>,
}

impl Default for CdcRetentionConfig {
    fn default() -> Self {
        Self {
            max_epochs: Some(1000),
            max_events: Some(100_000),
        }
    }
}

impl CdcRetentionConfig {
    /// Keeps every event: no limit on epochs or events. Beware of unbounded
    /// memory growth on long-running instances.
    #[must_use]
    pub fn unlimited() -> Self {
        Self {
            max_epochs: None,
            max_events: None,
        }
    }

    /// Prunes the events more than `max_epochs` epochs older than the
    /// current one (see [`max_epochs`](Self::max_epochs)).
    #[must_use]
    pub fn with_max_epochs(mut self, max_epochs: u64) -> Self {
        self.max_epochs = Some(max_epochs);
        self
    }

    /// Prunes the oldest events beyond `max_events` in total (see
    /// [`max_events`](Self::max_events)).
    #[must_use]
    pub fn with_max_events(mut self, max_events: usize) -> Self {
        self.max_events = Some(max_events);
        self
    }
}

/// The CDC log that records entity mutations.
///
/// Thread-safe: uses `RwLock<HashMap>` for concurrent access. Timestamps
/// are assigned by the embedded [`HlcClock`] to guarantee monotonicity.
///
/// Event retention is controlled by [`CdcRetentionConfig`]. Without retention
/// limits, the log grows unbounded (see [#250]).
///
/// [#250]: https://github.com/GrafeoDB/grafeo/issues/250
#[derive(Debug)]
pub struct CdcLog {
    events: RwLock<HbHashMap<(Option<String>, EntityId), Vec<ChangeEvent>>>,
    clock: Arc<HlcClock>,
    retention: CdcRetentionConfig,
}

impl CdcLog {
    /// Creates a new empty CDC log with a fresh HLC clock and default retention.
    #[must_use]
    pub fn new() -> Self {
        Self {
            events: RwLock::new(HbHashMap::new()),
            clock: Arc::new(HlcClock::new()),
            retention: CdcRetentionConfig::default(),
        }
    }

    /// Creates a new CDC log with the given retention config.
    #[must_use]
    pub fn with_retention(retention: CdcRetentionConfig) -> Self {
        Self {
            events: RwLock::new(HbHashMap::new()),
            clock: Arc::new(HlcClock::new()),
            retention,
        }
    }

    /// Returns the next HLC timestamp from this log's clock.
    pub fn next_timestamp(&self) -> HlcTimestamp {
        self.clock.now()
    }

    /// Returns a reference to the HLC clock for remote timestamp merging.
    pub fn clock(&self) -> &Arc<HlcClock> {
        &self.clock
    }

    /// Records a change event.
    pub fn record(&self, event: ChangeEvent) {
        self.events
            .write()
            .entry((event.graph.clone(), event.entity_id))
            .or_default()
            .push(event);
    }

    /// Records a batch of change events with a single write-lock acquisition.
    pub fn record_batch(&self, events: impl IntoIterator<Item = ChangeEvent>) {
        let mut guard = self.events.write();
        for event in events {
            guard
                .entry((event.graph.clone(), event.entity_id))
                .or_default()
                .push(event);
        }
    }

    /// Records a node creation.
    pub fn record_create_node(
        &self,
        id: NodeId,
        epoch: EpochId,
        props: Option<HashMap<String, Value>>,
        labels: Option<Vec<String>>,
    ) {
        self.record(ChangeEvent {
            entity_id: EntityId::Node(id),
            graph: None,
            kind: ChangeKind::Create,
            epoch,
            timestamp: self.clock.now(),
            before: None,
            after: props,
            labels,
            before_labels: None,
            edge_type: None,
            src_id: None,
            dst_id: None,
            triple_subject: None,
            triple_predicate: None,
            triple_object: None,
            triple_graph: None,
        });
    }

    /// Records an edge creation.
    pub fn record_create_edge(
        &self,
        id: EdgeId,
        epoch: EpochId,
        props: Option<HashMap<String, Value>>,
        src_id: u64,
        dst_id: u64,
        edge_type: String,
    ) {
        self.record(ChangeEvent {
            entity_id: EntityId::Edge(id),
            graph: None,
            kind: ChangeKind::Create,
            epoch,
            timestamp: self.clock.now(),
            before: None,
            after: props,
            labels: None,
            before_labels: None,
            edge_type: Some(edge_type),
            src_id: Some(src_id),
            dst_id: Some(dst_id),
            triple_subject: None,
            triple_predicate: None,
            triple_object: None,
            triple_graph: None,
        });
    }

    /// Records an RDF triple insertion.
    ///
    /// The terms must be N-Triples encoded (e.g. `<http://example.org/s>`,
    /// `"hello"`, `"42"^^<http://www.w3.org/2001/XMLSchema#integer>`).
    pub fn record_triple_insert(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        graph: Option<&str>,
        epoch: EpochId,
    ) {
        let id = triple_hash(subject, predicate, object, graph);
        self.record(ChangeEvent {
            entity_id: EntityId::Triple(id),
            graph: None,
            kind: ChangeKind::Create,
            epoch,
            timestamp: self.clock.now(),
            before: None,
            after: None,
            labels: None,
            before_labels: None,
            edge_type: None,
            src_id: None,
            dst_id: None,
            triple_subject: Some(subject.to_string()),
            triple_predicate: Some(predicate.to_string()),
            triple_object: Some(object.to_string()),
            triple_graph: graph.map(ToString::to_string),
        });
    }

    /// Records an RDF triple deletion.
    ///
    /// The terms must be N-Triples encoded.
    pub fn record_triple_delete(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        graph: Option<&str>,
        epoch: EpochId,
    ) {
        let id = triple_hash(subject, predicate, object, graph);
        self.record(ChangeEvent {
            entity_id: EntityId::Triple(id),
            graph: None,
            kind: ChangeKind::Delete,
            epoch,
            timestamp: self.clock.now(),
            before: None,
            after: None,
            labels: None,
            before_labels: None,
            edge_type: None,
            src_id: None,
            dst_id: None,
            triple_subject: Some(subject.to_string()),
            triple_predicate: Some(predicate.to_string()),
            triple_object: Some(object.to_string()),
            triple_graph: graph.map(ToString::to_string),
        });
    }

    /// Records a property update.
    pub fn record_update(
        &self,
        entity_id: EntityId,
        epoch: EpochId,
        key: &str,
        old_value: Option<Value>,
        new_value: Value,
    ) {
        let before = old_value.map(|v| {
            let mut m = HashMap::new();
            m.insert(key.to_string(), v);
            m
        });
        let mut after_map = HashMap::new();
        after_map.insert(key.to_string(), new_value);

        self.record(ChangeEvent {
            entity_id,
            graph: None,
            kind: ChangeKind::Update,
            epoch,
            timestamp: self.clock.now(),
            before,
            after: Some(after_map),
            labels: None,
            before_labels: None,
            edge_type: None,
            src_id: None,
            dst_id: None,
            triple_subject: None,
            triple_predicate: None,
            triple_object: None,
            triple_graph: None,
        });
    }

    /// Records an entity deletion.
    pub fn record_delete(
        &self,
        entity_id: EntityId,
        epoch: EpochId,
        props: Option<HashMap<String, Value>>,
    ) {
        self.record(ChangeEvent {
            entity_id,
            graph: None,
            kind: ChangeKind::Delete,
            epoch,
            timestamp: self.clock.now(),
            before: props,
            after: None,
            labels: None,
            before_labels: None,
            edge_type: None,
            src_id: None,
            dst_id: None,
            triple_subject: None,
            triple_predicate: None,
            triple_object: None,
            triple_graph: None,
        });
    }

    /// Returns all change events for an entity of the default graph, ordered
    /// by epoch.
    #[must_use]
    pub fn history(&self, entity_id: EntityId) -> Vec<ChangeEvent> {
        self.history_in(None, entity_id)
    }

    /// Returns all change events for an entity of `graph` (its storage key,
    /// `None` for the default graph), ordered by epoch.
    #[must_use]
    pub fn history_in(&self, graph: Option<&str>, entity_id: EntityId) -> Vec<ChangeEvent> {
        self.events
            .read()
            .get(&(graph.map(str::to_string), entity_id))
            .cloned()
            .unwrap_or_default()
    }

    /// Returns change events for an entity of the default graph since the
    /// given epoch.
    #[must_use]
    pub fn history_since(&self, entity_id: EntityId, since_epoch: EpochId) -> Vec<ChangeEvent> {
        self.history_since_in(None, entity_id, since_epoch)
    }

    /// Returns change events for an entity of `graph` (`None` for the default
    /// graph) since the given epoch.
    #[must_use]
    pub fn history_since_in(
        &self,
        graph: Option<&str>,
        entity_id: EntityId,
        since_epoch: EpochId,
    ) -> Vec<ChangeEvent> {
        self.events
            .read()
            .get(&(graph.map(str::to_string), entity_id))
            .map(|events| {
                events
                    .iter()
                    .filter(|e| e.epoch >= since_epoch)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Returns all change events across all entities and graphs in an epoch
    /// range; each event names its graph.
    #[must_use]
    pub fn changes_between(&self, start_epoch: EpochId, end_epoch: EpochId) -> Vec<ChangeEvent> {
        let guard = self.events.read();
        let mut results = Vec::new();
        for events in guard.values() {
            for event in events {
                if event.epoch >= start_epoch && event.epoch <= end_epoch {
                    results.push(event.clone());
                }
            }
        }
        results.sort_by_key(|e| e.epoch);
        results
    }

    /// Returns the total number of recorded events.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.read().values().map(Vec::len).sum()
    }

    /// Estimates the heap footprint of the log in bytes.
    ///
    /// Counts the entity-keyed hash map overhead, the per-entity `Vec<ChangeEvent>`
    /// capacity, and the events themselves. `ChangeEvent` contains owned `Value`
    /// payloads that can be arbitrarily large (property snapshots), but this
    /// estimate treats them as fixed-size and therefore undercounts histories
    /// that hold large strings, vectors, or bytes. Good enough to surface
    /// "CDC is dominating heap" in a memory-usage breakdown; not a precise
    /// accounting.
    #[must_use]
    pub fn heap_memory_bytes(&self) -> (usize, usize, usize) {
        let guard = self.events.read();
        let entity_count = guard.len();
        let mut event_count = 0usize;
        let mut bytes = 0usize;
        let entry_overhead = std::mem::size_of::<(EntityId, Vec<ChangeEvent>)>() + 8;
        let event_size = std::mem::size_of::<ChangeEvent>();
        for events in guard.values() {
            event_count += events.len();
            bytes += entry_overhead + events.capacity() * event_size;
        }
        (bytes, entity_count, event_count)
    }

    /// Removes all events with `epoch < min_epoch`.
    ///
    /// Entities whose entire history falls below the threshold are removed
    /// from the map entirely.
    pub fn prune_before(&self, min_epoch: EpochId) {
        let mut guard = self.events.write();
        guard.retain(|_, events| {
            events.retain(|e| e.epoch >= min_epoch);
            !events.is_empty()
        });
    }

    /// Prunes the oldest events to stay within `max_events`, if configured.
    ///
    /// Finds the epoch cutoff that brings the total count at or below the
    /// limit, then calls [`prune_before`](Self::prune_before).
    pub fn prune_to_limit(&self) {
        let Some(max) = self.retention.max_events else {
            return;
        };
        let count = self.event_count();
        if count <= max {
            return;
        }

        // Collect all epochs, sort, and find the cutoff
        let guard = self.events.read();
        let mut epochs: Vec<EpochId> = guard
            .values()
            .flat_map(|events| events.iter().map(|e| e.epoch))
            .collect();
        drop(guard);

        epochs.sort();
        let excess = count - max;
        // The cutoff epoch: remove events up to and including this epoch
        if let Some(&cutoff) = epochs.get(excess.saturating_sub(1)) {
            self.prune_before(EpochId::new(cutoff.as_u64() + 1));
        }
    }

    /// Applies epoch-based and count-based retention limits.
    ///
    /// Called from the database GC cycle.
    pub fn apply_retention(&self, current_epoch: EpochId) {
        // Epoch-based: prune events older than max_epochs
        if let Some(max_epochs) = self.retention.max_epochs {
            let cutoff = current_epoch.as_u64().saturating_sub(max_epochs);
            self.prune_before(EpochId::new(cutoff));
        }
        // Count-based: prune oldest events if over limit
        self.prune_to_limit();
    }

    /// Approximate memory usage in bytes.
    ///
    /// Each `ChangeEvent` is roughly 256 bytes including the property maps,
    /// strings, and per-entity HashMap overhead.
    #[must_use]
    pub fn approximate_memory_bytes(&self) -> usize {
        // ~256 bytes per event: struct fields + property map overhead + strings
        self.event_count() * 256
    }
}

impl Default for CdcLog {
    fn default() -> Self {
        Self::new()
    }
}

/// `MemoryConsumer` implementation for CDC: eviction prunes the oldest events.
impl MemoryConsumer for CdcLog {
    fn name(&self) -> &str {
        "cdc_log"
    }

    fn memory_usage(&self) -> usize {
        self.approximate_memory_bytes()
    }

    fn eviction_priority(&self) -> u8 {
        // CDC events are auxiliary history, not required for query correctness:
        // evicting them only loses history lookups, while indexes and graph
        // storage back live reads. Evict before those.
        priorities::QUERY_CACHE
    }

    fn region(&self) -> MemoryRegion {
        MemoryRegion::ExecutionBuffers
    }

    fn evict(&self, target_bytes: usize) -> usize {
        let before = self.approximate_memory_bytes();
        if before == 0 {
            return 0;
        }
        // Estimate how many events to remove
        let events_to_remove = target_bytes / 256;
        if events_to_remove == 0 {
            return 0;
        }

        // Find the epoch cutoff for the oldest N events
        let guard = self.events.read();
        let mut epochs: Vec<EpochId> = guard
            .values()
            .flat_map(|events| events.iter().map(|e| e.epoch))
            .collect();
        drop(guard);

        if epochs.is_empty() {
            return 0;
        }
        epochs.sort();
        let cutoff_idx = events_to_remove.min(epochs.len()).saturating_sub(1);
        let cutoff = epochs[cutoff_idx];
        self.prune_before(EpochId::new(cutoff.as_u64() + 1));

        before.saturating_sub(self.approximate_memory_bytes())
    }

    fn can_spill(&self) -> bool {
        // CDC events are transient metadata: eviction (pruning) is sufficient.
        // No disk spill needed.
        false
    }

    fn current_tier(&self) -> grafeo_common::memory::StorageTier {
        // CDC log is never spilled to disk.
        if self.memory_usage() == 0 {
            grafeo_common::memory::StorageTier::Uninitialized
        } else {
            grafeo_common::memory::StorageTier::InMemory
        }
    }
}

/// Computes a stable-within-process content hash for an RDF triple.
///
/// Used as the raw `u64` in `EntityId::Triple` so the CDC log can key events
/// by triple content without storing a separate ID registry.
fn triple_hash(subject: &str, predicate: &str, object: &str, graph: Option<&str>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    subject.hash(&mut h);
    predicate.hash(&mut h);
    object.hash(&mut h);
    graph.hash(&mut h);
    h.finish()
}

impl CdcLog {
    /// Records the events of a commit at `epoch` that changed what `set`
    /// holds (see [`events_for_commit`]). Called in the commit's ordered
    /// step, where no other commit runs: the timestamps the log's clock
    /// gives follow the commit epochs, so a later commit's events never
    /// carry an earlier timestamp.
    pub(crate) fn record_commit(&self, set: &grafeo_common::change::ChangeSet, epoch: EpochId) {
        if set.is_empty() {
            return;
        }
        self.record_batch(events_for_commit(set, epoch, &self.clock));
    }
}

/// The change events of a commit at `epoch` from its change set, one per
/// write in the order applied (see [`ChangeSet::ops`](grafeo_common::change::ChangeSet::ops):
/// each data entry, and each row a bulk write kept with its range; a range
/// without its rows, a bulk import's, has none), each naming its graph,
/// timestamped by `clock` in that order. A triple's event names its RDF
/// graph as `triple_graph`, as the RDF bulk writes report theirs. An event
/// is built from the write's op and before-image: a create carries its
/// labels or type and endpoints and the values it was created with, a
/// delete what the entity held, an update the value or the labels before
/// and after. The transaction's later changes to an entity it created are
/// then folded into the create (see [`fold_into_creates`]).
pub(crate) fn events_for_commit(
    set: &grafeo_common::change::ChangeSet,
    epoch: EpochId,
    clock: &HlcClock,
) -> Vec<ChangeEvent> {
    use grafeo_common::change::{Before, DataOp};

    let values = |properties: &[(grafeo_common::types::PropertyKey, Value)]| {
        (!properties.is_empty()).then(|| {
            properties
                .iter()
                .map(|(key, value)| (key.as_str().to_string(), value.clone()))
                .collect::<HashMap<String, Value>>()
        })
    };
    let one = |key: &grafeo_common::types::PropertyKey, value: &Value| {
        Some(HashMap::from([(key.as_str().to_string(), value.clone())]))
    };
    let names = |labels: &grafeo_common::change::Labels| -> Vec<String> {
        labels.iter().map(ToString::to_string).collect()
    };
    // Sized by the writes, not the entries: a batch is one entry for many
    // rows.
    let mut events = Vec::with_capacity(set.ops().count());
    for (graph, op, before) in set.ops() {
        let (entity_id, kind) = match op {
            DataOp::CreateNode { id, .. } => (EntityId::Node(*id), ChangeKind::Create),
            DataOp::CreateEdge { id, .. } => (EntityId::Edge(*id), ChangeKind::Create),
            DataOp::DeleteNode { id } => (EntityId::Node(*id), ChangeKind::Delete),
            DataOp::DeleteEdge { id } => (EntityId::Edge(*id), ChangeKind::Delete),
            DataOp::SetNodeProperty { id, .. }
            | DataOp::RemoveNodeProperty { id, .. }
            | DataOp::AddNodeLabel { id, .. }
            | DataOp::RemoveNodeLabel { id, .. } => (EntityId::Node(*id), ChangeKind::Update),
            DataOp::SetEdgeProperty { id, .. } | DataOp::RemoveEdgeProperty { id, .. } => {
                (EntityId::Edge(*id), ChangeKind::Update)
            }
            #[cfg(feature = "triple-store")]
            DataOp::InsertTriple { triple } => {
                events.push(triple_event(
                    set,
                    graph,
                    triple,
                    ChangeKind::Create,
                    epoch,
                    clock,
                ));
                continue;
            }
            #[cfg(feature = "triple-store")]
            DataOp::DeleteTriple { triple } => {
                events.push(triple_event(
                    set,
                    graph,
                    triple,
                    ChangeKind::Delete,
                    epoch,
                    clock,
                ));
                continue;
            }
            // A build without the triple store records no triple.
            #[cfg(not(feature = "triple-store"))]
            DataOp::InsertTriple { .. } | DataOp::DeleteTriple { .. } => continue,
        };
        let mut event = ChangeEvent {
            entity_id,
            graph: set
                .graph(graph)
                .and_then(|graph| graph.key.as_ref().map(ToString::to_string)),
            kind,
            epoch,
            timestamp: clock.now(),
            before: None,
            after: None,
            labels: None,
            before_labels: None,
            edge_type: None,
            src_id: None,
            dst_id: None,
            triple_subject: None,
            triple_predicate: None,
            triple_object: None,
            triple_graph: None,
        };
        match (op, before) {
            (
                DataOp::CreateNode {
                    labels, properties, ..
                },
                _,
            ) => {
                event.labels = Some(names(labels));
                event.after = values(properties);
            }
            (
                DataOp::CreateEdge {
                    src,
                    dst,
                    edge_type,
                    properties,
                    ..
                },
                _,
            ) => {
                event.edge_type = Some(edge_type.to_string());
                event.src_id = Some(src.as_u64());
                event.dst_id = Some(dst.as_u64());
                event.after = values(properties);
            }
            (DataOp::DeleteNode { .. }, Before::Node(image)) => {
                event.labels = Some(names(&image.labels));
                event.before = values(&image.properties);
            }
            (DataOp::DeleteEdge { .. }, Before::Edge(image)) => {
                event.edge_type = Some(image.edge_type.to_string());
                event.src_id = Some(image.src.as_u64());
                event.dst_id = Some(image.dst.as_u64());
                event.before = values(&image.properties);
            }
            (
                DataOp::SetNodeProperty { key, value, .. }
                | DataOp::SetEdgeProperty { key, value, .. },
                Before::Value(old),
            ) => {
                event.before = old.as_ref().and_then(|old| one(key, old));
                event.after = one(key, value);
            }
            (
                DataOp::RemoveNodeProperty { key, .. } | DataOp::RemoveEdgeProperty { key, .. },
                Before::Value(old),
            ) => {
                event.before = old.as_ref().and_then(|old| one(key, old));
            }
            (DataOp::AddNodeLabel { label, .. }, Before::Labels(labels)) => {
                let before = names(labels);
                let mut after = before.clone();
                after.push(label.to_string());
                event.before_labels = Some(before);
                event.labels = Some(after);
            }
            (DataOp::RemoveNodeLabel { label, .. }, Before::Labels(labels)) => {
                let before = names(labels);
                let after = before
                    .iter()
                    .filter(|name| name.as_str() != label.as_str())
                    .cloned()
                    .collect();
                event.before_labels = Some(before);
                event.labels = Some(after);
            }
            // The change set checks each entry's before-image against its
            // op (and keeps creates only with a bulk range), so no other
            // pair is recorded.
            _ => {}
        }
        events.push(event);
    }
    fold_into_creates(events)
}

/// The event of a triple entry of graph `graph` (`kind` says whether it was
/// inserted or deleted), as [`CdcLog::record_triple_insert`] and
/// [`CdcLog::record_triple_delete`] build it: the terms as N-Triples strings.
#[cfg(feature = "triple-store")]
fn triple_event(
    set: &grafeo_common::change::ChangeSet,
    graph: grafeo_common::change::GraphSlot,
    triple: &grafeo_common::storage::log_record::TripleRecord,
    kind: ChangeKind,
    epoch: EpochId,
    clock: &HlcClock,
) -> ChangeEvent {
    let (subject, predicate, object) = crate::transaction::ntriples_terms(triple);
    let graph = set
        .graph(graph)
        .and_then(|graph| graph.key.as_ref().map(ToString::to_string));
    ChangeEvent {
        entity_id: EntityId::Triple(triple_hash(&subject, &predicate, &object, graph.as_deref())),
        graph: None,
        kind,
        epoch,
        timestamp: clock.now(),
        before: None,
        after: None,
        labels: None,
        before_labels: None,
        edge_type: None,
        src_id: None,
        dst_id: None,
        triple_subject: Some(subject),
        triple_predicate: Some(predicate),
        triple_object: Some(object),
        triple_graph: graph,
    }
}

/// Folds a transaction's changes to the entities it created into their
/// create events, so a create event carries the entity as the transaction
/// left it: its final labels and properties.
///
/// Statements and the direct API both create an entity and then set its
/// properties and labels one by one; without this, a consumer saw a create
/// event without properties followed by one update per property.
/// Changes to entities that existed before the transaction stay as they are.
/// Entity ids repeat across graphs, so the folding stays within one graph.
fn fold_into_creates(events: Vec<ChangeEvent>) -> Vec<ChangeEvent> {
    let mut folded: Vec<ChangeEvent> = Vec::with_capacity(events.len());
    let mut created: HashMap<(Option<String>, EntityId), usize> = HashMap::new();
    for event in events {
        let key = (event.graph.clone(), event.entity_id);
        let Some(&at) = created.get(&key) else {
            if event.kind == ChangeKind::Create {
                created.insert(key, folded.len());
            }
            folded.push(event);
            continue;
        };
        if event.kind != ChangeKind::Update {
            folded.push(event);
            continue;
        }
        let create = &mut folded[at];
        // A label change lists the labels after it.
        if let Some(labels) = event.labels {
            create.labels = Some(labels);
        }
        match (event.before, event.after) {
            (_, Some(after)) => create.after.get_or_insert_with(HashMap::new).extend(after),
            (Some(removed), None) => {
                if let Some(properties) = create.after.as_mut() {
                    for key in removed.keys() {
                        properties.remove(key);
                    }
                }
            }
            (None, None) => {}
        }
        if create.after.as_ref().is_some_and(HashMap::is_empty) {
            create.after = None;
        }
    }
    folded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_history() {
        let log = CdcLog::new();
        let node_id = NodeId::new(1);

        log.record_create_node(node_id, EpochId(1), None, None);
        log.record_update(
            EntityId::Node(node_id),
            EpochId(2),
            "name",
            None,
            Value::from("Alix"),
        );
        log.record_update(
            EntityId::Node(node_id),
            EpochId(3),
            "name",
            Some(Value::from("Alix")),
            Value::from("Gus"),
        );

        let history = log.history(EntityId::Node(node_id));
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].kind, ChangeKind::Create);
        assert_eq!(history[1].kind, ChangeKind::Update);
        assert_eq!(history[2].kind, ChangeKind::Update);
    }

    #[test]
    fn test_history_since() {
        let log = CdcLog::new();
        let node_id = NodeId::new(1);

        log.record_create_node(node_id, EpochId(1), None, None);
        log.record_update(
            EntityId::Node(node_id),
            EpochId(5),
            "name",
            None,
            Value::from("Alix"),
        );
        log.record_update(
            EntityId::Node(node_id),
            EpochId(10),
            "name",
            Some(Value::from("Alix")),
            Value::from("Gus"),
        );

        let since_5 = log.history_since(EntityId::Node(node_id), EpochId(5));
        assert_eq!(since_5.len(), 2);
        assert_eq!(since_5[0].epoch, EpochId(5));
    }

    #[test]
    fn test_changes_between() {
        let log = CdcLog::new();

        log.record_create_node(NodeId::new(1), EpochId(1), None, None);
        log.record_create_node(NodeId::new(2), EpochId(3), None, None);
        log.record_update(
            EntityId::Node(NodeId::new(1)),
            EpochId(5),
            "x",
            None,
            Value::from(42),
        );

        let changes = log.changes_between(EpochId(2), EpochId(5));
        assert_eq!(changes.len(), 2); // epoch 3 and 5
    }

    #[test]
    fn test_delete_event() {
        let log = CdcLog::new();
        let node_id = NodeId::new(1);

        let mut props = HashMap::new();
        props.insert("name".to_string(), Value::from("Alix"));

        log.record_create_node(node_id, EpochId(1), Some(props.clone()), None);
        log.record_delete(EntityId::Node(node_id), EpochId(2), Some(props));

        let history = log.history(EntityId::Node(node_id));
        assert_eq!(history.len(), 2);
        assert_eq!(history[1].kind, ChangeKind::Delete);
        assert!(history[1].after.is_none());
        assert!(history[1].before.is_some());
    }

    #[test]
    fn test_empty_history() {
        let log = CdcLog::new();
        let history = log.history(EntityId::Node(NodeId::new(999)));
        assert!(history.is_empty());
    }

    #[test]
    fn test_event_count() {
        let log = CdcLog::new();
        assert_eq!(log.event_count(), 0);

        log.record_create_node(NodeId::new(1), EpochId(1), None, None);
        log.record_create_node(NodeId::new(2), EpochId(2), None, None);
        assert_eq!(log.event_count(), 2);
    }

    #[test]
    fn test_entity_id_conversions() {
        let node_id = NodeId::new(42);
        let entity: EntityId = node_id.into();
        assert!(entity.is_node());
        assert_eq!(entity.as_u64(), 42);

        let edge_id = EdgeId::new(7);
        let entity: EntityId = edge_id.into();
        assert!(!entity.is_node());
        assert_eq!(entity.as_u64(), 7);
    }

    #[test]
    fn test_prune_before() {
        let log = CdcLog::new();

        // Record events across 10 epochs
        for epoch in 1..=10 {
            log.record_create_node(NodeId::new(epoch), EpochId(epoch), None, None);
        }
        assert_eq!(log.event_count(), 10);

        // Prune everything before epoch 6
        log.prune_before(EpochId(6));
        assert_eq!(log.event_count(), 5);

        // Verify only epochs 6-10 remain
        let remaining = log.changes_between(EpochId(0), EpochId(100));
        assert!(remaining.iter().all(|e| e.epoch >= EpochId(6)));
    }

    #[test]
    fn test_prune_to_limit() {
        let retention = CdcRetentionConfig {
            max_epochs: None,
            max_events: Some(5),
        };
        let log = CdcLog::with_retention(retention);

        // Record 10 events
        for epoch in 1..=10 {
            log.record_create_node(NodeId::new(epoch), EpochId(epoch), None, None);
        }
        assert_eq!(log.event_count(), 10);

        // Prune to limit of 5
        log.prune_to_limit();
        assert!(log.event_count() <= 5);
    }

    #[test]
    fn test_apply_retention_epoch_based() {
        let retention = CdcRetentionConfig {
            max_epochs: Some(3),
            max_events: None,
        };
        let log = CdcLog::with_retention(retention);

        for epoch in 1..=10 {
            log.record_create_node(NodeId::new(epoch), EpochId(epoch), None, None);
        }
        assert_eq!(log.event_count(), 10);

        // Apply retention at current epoch 10 with max_epochs=3
        // Should prune events before epoch 7
        log.apply_retention(EpochId(10));

        let remaining = log.changes_between(EpochId(0), EpochId(100));
        assert!(remaining.iter().all(|e| e.epoch >= EpochId(7)));
        assert_eq!(remaining.len(), 4); // epochs 7, 8, 9, 10
    }

    #[test]
    fn test_memory_consumer_evict() {
        let log = CdcLog::new();

        // Record 100 events
        for epoch in 1..=100 {
            log.record_create_node(NodeId::new(epoch), EpochId(epoch), None, None);
        }

        let before = log.approximate_memory_bytes();
        assert!(before > 0);

        // Evict ~half the memory
        let freed = log.evict(before / 2);
        assert!(freed > 0);
        assert!(log.event_count() < 100);
    }

    #[test]
    fn test_retention_config_default() {
        let config = CdcRetentionConfig::default();
        assert_eq!(config.max_epochs, Some(1000));
        assert_eq!(config.max_events, Some(100_000));
    }

    /// `unlimited()` keeps every event, where the default drops those more
    /// than 1,000 epochs old; `with_max_epochs` and `with_max_events` each
    /// bound the history on their own.
    #[test]
    fn retention_constructors_set_the_limits() {
        let kept = |retention: CdcRetentionConfig| {
            let log = CdcLog::with_retention(retention);
            for epoch in 1..=1_003 {
                log.record_create_node(NodeId::new(epoch), EpochId(epoch), None, None);
            }
            log.apply_retention(EpochId(1_003));
            log.event_count()
        };
        assert_eq!(
            kept(CdcRetentionConfig::default()),
            1_001,
            "the default keeps epochs 3 to 1003"
        );
        assert_eq!(
            kept(CdcRetentionConfig::unlimited()),
            1_003,
            "unlimited keeps every event"
        );
        assert_eq!(
            kept(CdcRetentionConfig::unlimited().with_max_epochs(3)),
            4,
            "3 epochs back from 1003 keeps epochs 1000 to 1003"
        );
        assert_eq!(
            kept(CdcRetentionConfig::unlimited().with_max_events(19)),
            19,
            "19 events keeps the 19 newest"
        );
        let both = CdcRetentionConfig::unlimited()
            .with_max_epochs(88)
            .with_max_events(19);
        assert_eq!((both.max_epochs, both.max_events), (Some(88), Some(19)));
    }

    #[test]
    fn test_apply_retention_count_only() {
        let retention = CdcRetentionConfig {
            max_epochs: None,
            max_events: Some(4),
        };
        let log = CdcLog::with_retention(retention);

        for epoch in 1..=10 {
            log.record_create_node(NodeId::new(epoch), EpochId(epoch), None, None);
        }
        assert_eq!(log.event_count(), 10);

        // apply_retention with no epoch limit should still prune by count
        log.apply_retention(EpochId(10));
        assert!(
            log.event_count() <= 4,
            "count-based retention should prune to at most 4 events, got {}",
            log.event_count()
        );
    }

    #[test]
    fn test_apply_retention_combined_epoch_and_count() {
        // epoch limit keeps last 5 (epochs 6..=10), count limit keeps 3.
        // The stricter (count) should win after both passes.
        let retention = CdcRetentionConfig {
            max_epochs: Some(5),
            max_events: Some(3),
        };
        let log = CdcLog::with_retention(retention);

        for epoch in 1..=10 {
            log.record_create_node(NodeId::new(epoch), EpochId(epoch), None, None);
        }
        assert_eq!(log.event_count(), 10);

        log.apply_retention(EpochId(10));

        // Epoch pass prunes epochs < 5 (keeps 6..=10 = 5 events).
        // Count pass then prunes to at most 3.
        assert!(
            log.event_count() <= 3,
            "combined retention should honour the stricter limit, got {}",
            log.event_count()
        );
        // All remaining events should be recent
        let remaining = log.changes_between(EpochId(0), EpochId(100));
        assert!(remaining.iter().all(|e| e.epoch >= EpochId(6)));
    }

    #[test]
    fn test_prune_before_epoch_zero() {
        let log = CdcLog::new();

        for epoch in 1..=5 {
            log.record_create_node(NodeId::new(epoch), EpochId(epoch), None, None);
        }
        assert_eq!(log.event_count(), 5);

        // Pruning before epoch 0 should be a no-op: all events have epoch >= 1
        log.prune_before(EpochId(0));
        assert_eq!(
            log.event_count(),
            5,
            "prune_before(0) should not remove anything"
        );
    }

    #[test]
    fn test_prune_to_limit_same_epoch() {
        let retention = CdcRetentionConfig {
            max_epochs: None,
            max_events: Some(3),
        };
        let log = CdcLog::with_retention(retention);

        // All 10 events share the same epoch: the cutoff epoch equals the
        // only epoch present, so prune_before removes everything at or below
        // that epoch. This is by design: epoch-granularity pruning cannot
        // split events within the same epoch.
        for i in 1..=10 {
            log.record_create_node(NodeId::new(i), EpochId(5), None, None);
        }
        assert_eq!(log.event_count(), 10);

        log.prune_to_limit();

        // After pruning, the log should have fewer events than before.
        // With all events at the same epoch, the cutoff removes them all
        // because prune_before uses a strict < comparison on epoch+1.
        assert!(
            log.event_count() < 10,
            "prune_to_limit should have removed events, got {}",
            log.event_count()
        );
    }

    #[test]
    fn test_evict_tiny_target_is_noop() {
        let log = CdcLog::new();
        for epoch in 1..=10 {
            log.record_create_node(NodeId::new(epoch), EpochId(epoch), None, None);
        }
        assert_eq!(log.event_count(), 10);

        // target_bytes < 256 means events_to_remove rounds to 0, so nothing freed
        let freed = log.evict(100);
        assert_eq!(freed, 0, "evict with target < 256 bytes should be a no-op");
        assert_eq!(log.event_count(), 10);
    }

    #[test]
    fn test_heap_memory_bytes_scales_with_events() {
        let log = CdcLog::new();
        let (empty_bytes, empty_entities, empty_events) = log.heap_memory_bytes();
        assert_eq!(empty_entities, 0);
        assert_eq!(empty_events, 0);

        for epoch in 1..=50 {
            log.record_create_node(NodeId::new(epoch), EpochId(epoch), None, None);
        }
        let (populated_bytes, populated_entities, populated_events) = log.heap_memory_bytes();
        assert_eq!(populated_entities, 50);
        assert_eq!(populated_events, 50);
        assert!(
            populated_bytes > empty_bytes,
            "heap estimate must grow when events are recorded"
        );
    }
}
