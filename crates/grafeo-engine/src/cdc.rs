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
//! `Edge(EdgeId)` for LPG, `Triple(u64)` for RDF: the first 8 bytes of
//! SHA-256 over the triple's terms and its graph, so the log keys events by
//! triple content without a separate id registry, and the id is the same in
//! every process and release. It is a 64-bit hash, unique in practice but
//! not guaranteed: a consumer that must not conflate two triples keys on the
//! terms the event carries.
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
//! [`CdcLog`] is the authoritative in-memory store: under one `RwLock`, a
//! map from `(graph, EntityId)` to the entity's events (entity ids repeat
//! across graphs, so the graph is part of the key), and the commits held in
//! epoch order, each with the keys of the entities it has events of.
//! Readers (history queries, retention probes) take the read lock; writers
//! (commit-path recording, prunes) take the write lock. We use
//! [`hashbrown`]'s `HashMap` for the Fx-hashed inner map, and
//! [`parking_lot`]'s `RwLock` for cheap uncontended locking; under
//! concurrent insert+read the typical pattern is short write critical
//! sections (one entry lookup and push per event) and occasional long read
//! critical sections for `history*()` snapshots that clone the events out of
//! the lock. A prune pops whole commits from the front of the queue and each
//! of their entities' oldest events from the front of its list, so it costs
//! what it drops, not the size of the log.
//!
//! Per-entity event order within an entity's events is insertion
//! order, which is monotonic by construction: the only writer is the
//! commit path (see "Integration with the commit path" below) and each
//! event carries the MVCC `epoch` that produced it plus an HLC
//! timestamp from the embedded [`HlcClock`]. The clock guarantees
//! strictly-increasing timestamps across all threads within a process
//! (wall-clock ms in the upper 48 bits, logical counter in the lower
//! 16, carrying into the milliseconds past 65,535). A commit reserves one
//! run of consecutive timestamps for its writes, and write `i` (from 0, in
//! the order applied) gets the run's first plus `i`, a write folded into a
//! create included, so a later commit's timestamps are larger.
//!
//! # Integration with the commit path
//!
//! A transaction's writes (statements, the direct API of a session or of
//! the database, batches) are recorded in its change set; nothing reaches
//! the log while it runs. Its commit builds its events from the change set
//! (one per change, folded into its net changes: later changes fold into
//! the create of an entity it created, and a create folded with its delete
//! leaves nothing)
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
//! [`RowGroupStore`](grafeo_core::graph::rowgroup::RowGroupStore) version, so
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
//! [`RowGroupStore`](grafeo_core::graph::rowgroup::RowGroupStore) state by calling
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
//!
//! # Stability
//!
//! The `grafeo` crate re-exports this module. Its items are stable unless
//! their documentation has a `# Stability: Experimental` section: see
//! <https://grafeo.dev/stability/>.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use grafeo_common::memory::buffer::{
    BufferManager, MemoryConsumer, MemoryGrant, MemoryRegion, priorities,
};
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
    /// An RDF triple, identified by a hash of its terms and its graph: the
    /// same in every process and release, unique in practice but not
    /// guaranteed (key on the event's terms when that matters).
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
    /// Properties before the change: the old value of an update (None when
    /// the property was not there), and for a delete what the entity held, a
    /// map that is empty when it held none. None for a create and a triple,
    /// and when the write's before-image is too large for the WAL record that
    /// holds it (past `MAX_RECORD_BYTES` or the list rule): the record then
    /// leaves the properties out. A delete's event still has its labels, or its
    /// edge type and endpoints, which the record keeps, unless those alone
    /// pass the limits (labels of about 1 GiB); then the record has no
    /// image and the event has neither, as the durable feed reads it back.
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
    /// On an edge Delete event, the node whose detach delete removed the
    /// edge (`DETACH DELETE`, `GrafeoDB::detach_delete_node`): the node's own
    /// Delete event follows in the same commit, so a consumer can group a
    /// cascade by it. `None` for an edge deleted on its own (#465).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deleted_with: Option<u64>,
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

/// Where a commit stands in the durable change feed: the log position its
/// group starts at, and its epoch. [`CdcBatch::position`] gives it, and
/// [`GrafeoDB::acknowledge_cdc_feed`](crate::GrafeoDB::acknowledge_cdc_feed)
/// moves the feed past it. Its token, `<lsn>:<epoch>`, is what a consumer
/// keeps between runs.
///
/// # Stability: Experimental
///
/// New in 0.6.4; its token and the feed's calls may change in a patch
/// release until the feed is stable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CdcPosition {
    lsn: u64,
    epoch: EpochId,
}

impl CdcPosition {
    /// The position of the group that starts at `lsn` and commits at `epoch`.
    pub(crate) const fn new(lsn: u64, epoch: EpochId) -> Self {
        Self { lsn, epoch }
    }

    /// The log position the group starts at, which the durable feed reads.
    #[cfg(all(feature = "wal", feature = "grafeo-file"))]
    pub(crate) const fn lsn(&self) -> u64 {
        self.lsn
    }

    /// The group's epoch.
    #[must_use]
    pub const fn epoch(&self) -> EpochId {
        self.epoch
    }

    /// The position as a token, `<lsn>:<epoch>`, which
    /// [`from_token`](Self::from_token) reads back.
    #[must_use]
    pub fn to_token(&self) -> String {
        format!("{}:{}", self.lsn, self.epoch.as_u64())
    }

    /// Reads a token [`to_token`](Self::to_token) wrote.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidValue`](grafeo_common::utils::error::Error::InvalidValue)
    /// when `token` is not two decimal numbers joined by `:`.
    pub fn from_token(token: &str) -> grafeo_common::utils::error::Result<Self> {
        let number = |part: &str| {
            (!part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
                .then(|| part.parse::<u64>().ok())
                .flatten()
        };
        token
            .split_once(':')
            .and_then(|(lsn, epoch)| Some(Self::new(number(lsn)?, EpochId::new(number(epoch)?))))
            .ok_or_else(|| {
                grafeo_common::utils::error::Error::InvalidValue(format!(
                    "{token:?} is not a change feed position: a position is <lsn>:<epoch>"
                ))
            })
    }
}

impl std::fmt::Display for CdcPosition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_token())
    }
}

/// One commit of the durable change feed: its position and its events, as
/// the live log recorded them (see
/// [`GrafeoDB::read_cdc_feed`](crate::GrafeoDB::read_cdc_feed)).
///
/// # Stability: Experimental
///
/// New in 0.6.4.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CdcBatch {
    /// The commit's position: acknowledging it moves the feed past it.
    pub position: CdcPosition,
    /// The commit's events, in the order it applied its writes. A commit
    /// whose writes undid each other has none (see the net rule).
    pub events: Vec<ChangeEvent>,
}

/// Configuration for CDC event retention.
///
/// Controls how many events the CDC log keeps in memory. When limits are
/// exceeded, the oldest events (by epoch) are pruned automatically.
///
/// Start from [`CdcRetentionConfig::default()`] (1,000 epochs and 100,000
/// events) or [`CdcRetentionConfig::unlimited()`] and set a limit with
/// [`with_max_epochs`](Self::with_max_epochs),
/// [`with_max_events`](Self::with_max_events) and
/// [`with_max_bytes`](Self::with_max_bytes). Later releases may add
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
///     max_bytes: None,
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
    /// The bytes the log's events may hold, counted as they come.
    /// When the log passes the bound, its oldest whole commits are dropped
    /// until it is at or below three quarters of the bound; the newest
    /// commit is always kept.
    /// `None` here lets the database derive the bound: a sixteenth of its
    /// memory limit (see [`Config::with_cdc_memory_limit`](crate::Config::with_cdc_memory_limit)).
    /// A bound is capped at the memory limit. The eviction cycle may prune
    /// further under memory pressure, the newest commit included. Pruning
    /// the in-memory log leaves the log kept on disk for the change feed's
    /// cursor alone.
    pub max_bytes: Option<usize>,
}

impl Default for CdcRetentionConfig {
    fn default() -> Self {
        Self {
            max_epochs: Some(1000),
            max_events: Some(100_000),
            max_bytes: None,
        }
    }
}

impl CdcRetentionConfig {
    /// No limit on epochs or events. The byte bound stays: the one given
    /// with [`with_max_bytes`](Self::with_max_bytes), or the one the database
    /// derives from its memory limit.
    #[must_use]
    pub fn unlimited() -> Self {
        Self {
            max_epochs: None,
            max_events: None,
            max_bytes: None,
        }
    }

    /// Bounds the bytes the log's events hold (see
    /// [`max_bytes`](Self::max_bytes)).
    #[must_use]
    pub fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = Some(max_bytes);
        self
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
/// Event retention is controlled by [`CdcRetentionConfig`]. A log a
/// database holds is bounded in bytes too (see
/// [`CdcRetentionConfig::max_bytes`]), whatever the epoch and event limits.
#[derive(Debug)]
pub struct CdcLog {
    events: RwLock<Held>,
    clock: Arc<HlcClock>,
    retention: CdcRetentionConfig,
    /// The calls of the triple helpers that recorded nothing (see
    /// [`dropped_lines`](Self::dropped_lines)).
    dropped_lines: std::sync::atomic::AtomicU64,
    /// What [`heap_memory_bytes`](Self::heap_memory_bytes) walks, kept as
    /// events come and go, under the events' write lock (one writer: a load
    /// and a store, no read-modify-write).
    counted: AtomicUsize,
    /// The events the log holds, kept alike.
    held_events: AtomicUsize,
    /// The entities and events the prunes visited (see
    /// [`prune_visits`](Self::prune_visits)).
    prune_visits: std::sync::atomic::AtomicU64,
    /// The counts of the last byte prune (see
    /// [`last_byte_prune`](Self::last_byte_prune)).
    byte_prune: parking_lot::Mutex<Option<(usize, usize, usize)>>,
    /// The grant of the counted bytes, once a database attaches its memory.
    memory: LogMemory,
}

/// The grant of a CDC log and the manager it comes from.
#[derive(Default)]
struct LogMemory {
    manager: std::sync::OnceLock<Arc<BufferManager>>,
    grant: parking_lot::Mutex<Option<MemoryGrant>>,
    /// The grant's size, read without its lock.
    granted: AtomicUsize,
}

impl std::fmt::Debug for LogMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogMemory")
            .field("granted", &self.granted.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

/// An entity's key in the log: its graph's storage key and its id.
type EntityKey = (Option<String>, EntityId);

/// The log's events and the commits that hold them, under one lock.
#[derive(Debug, Default)]
struct Held {
    /// Each entity's events, in epoch order.
    entities: HbHashMap<EntityKey, VecDeque<ChangeEvent>>,
    /// The commits held, oldest first: the entities each has events of (ids,
    /// not events). A prune pops whole commits from the front.
    commits: VecDeque<HeldCommit>,
}

/// A commit the log holds: its epoch and the keys of the entities it has
/// events of, each once. The first key is inline, so a commit of one entity
/// takes no allocation of its own.
#[derive(Debug)]
struct HeldCommit {
    epoch: EpochId,
    first: EntityKey,
    rest: Vec<EntityKey>,
}

impl HeldCommit {
    fn new(epoch: EpochId, key: EntityKey) -> Self {
        Self {
            epoch,
            first: key,
            rest: Vec::new(),
        }
    }

    fn keys(&self) -> impl Iterator<Item = &EntityKey> {
        std::iter::once(&self.first).chain(&self.rest)
    }

    /// Adds `key`; the slots its list grew by.
    fn add(&mut self, key: EntityKey) -> usize {
        let before = self.rest.capacity();
        self.rest.push(key);
        self.rest.capacity() - before
    }
}

/// The bytes a bucket of the entities table takes: an entry's key and
/// value inline, and its control byte. The table itself is counted by its
/// allocation (`allocation_size`), which removals leave as it is.
const ENTRY_SLOT_BYTES: usize = std::mem::size_of::<(EntityKey, VecDeque<ChangeEvent>)>() + 1;

/// The bytes a bucket of an event's property map takes: an entry's key and
/// value inline, and its control byte.
const PROPERTY_SLOT_BYTES: usize = std::mem::size_of::<(String, Value)>() + 1;

/// The bytes an event's property map of `capacity` takes with buckets of
/// `slot` bytes: its buckets, by hashbrown's own rounding from a capacity
/// (4 below 4, 8 below 8, else the power of two at or above eight sevenths
/// of it), and the group of control bytes past them. Exact for a map built
/// by inserts; a removal that leaves a tombstone lowers the capacity, which
/// the rounding takes back up to the buckets. An empty map allocates nothing.
fn table_bytes(capacity: usize, slot: usize) -> usize {
    /// The control bytes past the buckets: hashbrown's group (`Group::WIDTH`),
    /// 16 where it scans with SSE2 or LSX, 8 with NEON on aarch64, and
    /// otherwise the generic group's word, u64 or u32 as in its
    /// `control/group/generic.rs`.
    const GROUP_BYTES: usize = if cfg!(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2",
        not(miri)
    )) || cfg!(all(
        target_arch = "loongarch64",
        target_feature = "lsx",
        not(miri)
    )) {
        16
    } else if cfg!(any(
        target_pointer_width = "64",
        target_arch = "aarch64",
        target_arch = "x86_64",
        target_arch = "wasm32"
    )) {
        8
    } else {
        4
    };
    let buckets = match capacity {
        0 => return 0,
        1..4 => 4,
        4..8 => 8,
        _ => (capacity * 8 / 7).next_power_of_two(),
    };
    buckets * slot + GROUP_BYTES
}

/// The bytes a key takes in a commit's list, besides its heap bytes.
const KEY_BYTES: usize = std::mem::size_of::<EntityKey>();

/// The bytes a commit takes in the queue, besides its list's heap bytes.
const COMMIT_BYTES: usize = std::mem::size_of::<HeldCommit>();

/// The heap bytes an entry's key owns besides its inline slot: the graph
/// name's string, cloned into the key when the entry is created (the event
/// keeps its own copy, counted in [`event_heap_bytes`]).
fn key_heap_bytes(key: &EntityKey) -> usize {
    key.0.as_ref().map_or(0, String::capacity)
}

/// The bytes the room of `events` takes, without what each event owns.
fn room_bytes(events: &VecDeque<ChangeEvent>) -> usize {
    events.capacity() * std::mem::size_of::<ChangeEvent>()
}

/// The heap bytes a commit's list of keys takes.
fn keys_bytes(commit: &HeldCommit) -> usize {
    commit.rest.capacity() * KEY_BYTES + commit.keys().map(key_heap_bytes).sum::<usize>()
}

/// The heap bytes `event` owns besides its place in its entity's vector:
/// its strings, its label lists and its property maps with their keys and
/// values. An event never changes in the log, so the figure at its push is
/// the figure at its prune and in the walk.
fn event_heap_bytes(event: &ChangeEvent) -> usize {
    let text = |text: &Option<String>| text.as_ref().map_or(0, String::capacity);
    let names = |names: &Option<Vec<String>>| {
        names.as_ref().map_or(0, |names| {
            names.capacity() * std::mem::size_of::<String>()
                + names.iter().map(String::capacity).sum::<usize>()
        })
    };
    let properties = |properties: &Option<HashMap<String, Value>>| {
        properties.as_ref().map_or(0, |properties| {
            table_bytes(properties.capacity(), PROPERTY_SLOT_BYTES)
                + properties
                    .iter()
                    .map(|(key, value)| key.capacity() + value.estimated_size_bytes())
                    .sum::<usize>()
        })
    };
    text(&event.graph)
        + text(&event.edge_type)
        + text(&event.triple_subject)
        + text(&event.triple_predicate)
        + text(&event.triple_object)
        + text(&event.triple_graph)
        + names(&event.labels)
        + names(&event.before_labels)
        + properties(&event.before)
        + properties(&event.after)
}

/// Lists `key` under the commit of `epoch` in `commits`, adding the commit;
/// the bytes the queue grew. A commit's events come in epoch order through
/// the commit path; the public helpers take any epoch, so an older one finds
/// its place. An entity's event recorded after a newer one of the entity
/// goes with that newer one, which stands before it in the entity's events;
/// its key may then be listed under two commits, which costs only the slot.
fn list_key(commits: &mut VecDeque<HeldCommit>, epoch: EpochId, key: EntityKey) -> usize {
    let queue_before = commits.capacity();
    let key_bytes = key_heap_bytes(&key);
    let grown_keys = match commits.back_mut() {
        Some(last) if last.epoch == epoch => last.add(key),
        Some(last) if last.epoch > epoch => {
            let at = commits.partition_point(|commit| commit.epoch < epoch);
            match commits.get_mut(at).filter(|commit| commit.epoch == epoch) {
                Some(commit) if commit.keys().any(|listed| *listed == key) => return 0,
                Some(commit) => commit.add(key),
                None => {
                    commits.insert(at, HeldCommit::new(epoch, key));
                    0
                }
            }
        }
        _ => {
            commits.push_back(HeldCommit::new(epoch, key));
            0
        }
    };
    (commits.capacity() - queue_before) * COMMIT_BYTES + grown_keys * KEY_BYTES + key_bytes
}

impl CdcLog {
    /// Creates a new empty CDC log with a fresh HLC clock and default retention.
    #[must_use]
    pub fn new() -> Self {
        Self::with_retention(CdcRetentionConfig::default())
    }

    /// Creates a new CDC log with the given retention config.
    #[must_use]
    pub fn with_retention(retention: CdcRetentionConfig) -> Self {
        Self {
            events: RwLock::new(Held::default()),
            clock: Arc::new(HlcClock::new()),
            retention,
            dropped_lines: std::sync::atomic::AtomicU64::new(0),
            counted: AtomicUsize::new(0),
            held_events: AtomicUsize::new(0),
            prune_visits: std::sync::atomic::AtomicU64::new(0),
            byte_prune: parking_lot::Mutex::new(None),
            memory: LogMemory::default(),
        }
    }

    /// How many calls of [`record_triple_insert`](Self::record_triple_insert)
    /// and [`record_triple_delete`](Self::record_triple_delete) recorded
    /// nothing: their terms were not one N-Triples triple (a blank argument
    /// included), or the build has no triple store to read them. Each also
    /// logs a warning, which a build without the `tracing` feature drops, so
    /// this count is the signal there.
    #[must_use]
    pub fn dropped_lines(&self) -> u64 {
        self.dropped_lines.load(Ordering::Relaxed)
    }

    /// How many entities and events the log's prunes have visited, all
    /// prunes together: a prune visits what it drops, not the whole log. For
    /// tests.
    #[doc(hidden)]
    #[must_use]
    pub fn prune_visits(&self) -> u64 {
        self.prune_visits.load(Ordering::Relaxed)
    }

    /// The last prune the byte bound ran, as the counted bytes when it began,
    /// before it dropped its last commit, and when it ended: `None` before
    /// the first. For tests of the rule.
    #[doc(hidden)]
    #[must_use]
    pub fn last_byte_prune(&self) -> Option<(usize, usize, usize)> {
        *self.byte_prune.lock()
    }

    /// Counts a helper call that recorded nothing.
    fn drop_line(&self) {
        self.dropped_lines.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns the next HLC timestamp from this log's clock.
    pub fn next_timestamp(&self) -> HlcTimestamp {
        self.clock.now()
    }

    /// Returns a reference to the HLC clock for remote timestamp merging.
    pub fn clock(&self) -> &Arc<HlcClock> {
        &self.clock
    }

    /// Takes the grant of the log's bytes from `manager` from now on. A log
    /// keeps the first manager it gets.
    pub fn attach_memory(&self, manager: Arc<BufferManager>) {
        if self.memory.manager.set(manager).is_ok() {
            self.settle();
        }
    }

    /// Brings the grant to the counted bytes, forced (a commit, which must
    /// not fail, records the events): two loads and a few compares while it
    /// is within the bounds of `BufferManager::settled_size`.
    fn settle(&self) {
        let Some(manager) = self.memory.manager.get() else {
            return;
        };
        let counted = self.counted.load(Ordering::Relaxed);
        let granted = self.memory.granted.load(Ordering::Relaxed);
        if manager.settled_size(granted, counted) == granted {
            return;
        }
        let mut grant = self.memory.grant.lock();
        // Read again under the grant's lock, so a settle that read an older
        // count cannot shrink what a newer one just grew.
        let counted = self.counted.load(Ordering::Relaxed);
        manager.settle_forced(&mut grant, MemoryRegion::ExecutionBuffers, counted);
        self.memory.granted.store(
            grant.as_ref().map_or(0, MemoryGrant::size),
            Ordering::Relaxed,
        );
    }

    /// [`settle`](Self::settle) without the slack: the grant keeps no more
    /// than the counted bytes' `BufferManager::grant_size`, so what an eviction
    /// pruned leaves the buffer manager's count at once (the eviction cycle
    /// reads it). The bytes the grant gave back.
    fn settle_tight(&self) -> usize {
        let Some(manager) = self.memory.manager.get() else {
            return 0;
        };
        let mut grant = self.memory.grant.lock();
        let counted = self.counted.load(Ordering::Relaxed);
        let held = grant.as_ref().map_or(0, MemoryGrant::size);
        let mut given_back = 0;
        if held < counted {
            manager.settle_forced(&mut grant, MemoryRegion::ExecutionBuffers, counted);
        } else if let Some(grant) = grant.as_mut() {
            given_back = held.saturating_sub(manager.grant_size(counted));
            grant.shrink(given_back);
        }
        self.memory.granted.store(
            grant.as_ref().map_or(0, MemoryGrant::size),
            Ordering::Relaxed,
        );
        given_back
    }

    /// Pushes `event` under `held`, counting what it grew: its entity's
    /// events and, for the commit's first event of the entity, the commit's
    /// list of keys.
    fn push(&self, held: &mut Held, event: ChangeEvent) {
        let mut grown = event_heap_bytes(&event);
        let epoch = event.epoch;
        let table_before = held.entities.allocation_size();
        let (events, before, listed) =
            match held.entities.entry((event.graph.clone(), event.entity_id)) {
                hashbrown::hash_map::Entry::Occupied(entry) => {
                    let events = entry.into_mut();
                    let listed = events.back().is_some_and(|last| last.epoch == epoch);
                    let before = room_bytes(events);
                    (events, before, listed)
                }
                hashbrown::hash_map::Entry::Vacant(entry) => {
                    grown += key_heap_bytes(entry.key());
                    // Room for the first event alone: pushes then keep at most twice
                    // the room the events need, so a prune shrinks only what it took from.
                    (entry.insert(VecDeque::with_capacity(1)), 0, false)
                }
            };
        // The commit's first event of the entity lists it under the commit.
        let key = (!listed).then(|| (event.graph.clone(), event.entity_id));
        events.push_back(event);
        grown += room_bytes(events) - before;
        // An insert grows the table or leaves it as it is.
        grown += held.entities.allocation_size() - table_before;
        if let Some(key) = key {
            grown += list_key(&mut held.commits, epoch, key);
        }
        // One writer, under the events' write lock: a load and a store.
        self.counted.store(
            self.counted.load(Ordering::Relaxed) + grown,
            Ordering::Relaxed,
        );
        self.held_events.store(
            self.held_events.load(Ordering::Relaxed) + 1,
            Ordering::Relaxed,
        );
    }

    /// Drops the oldest commit `held` holds: each of its entities' events of
    /// its epoch or older, from the front of their lists, and an entity left
    /// with none. Counts what it freed and visited; the events it dropped,
    /// `None` when no commit is held.
    fn pop_commit(&self, held: &mut Held) -> Option<usize> {
        let queue_before = held.commits.capacity();
        let commit = held.commits.pop_front()?;
        let (mut freed, mut dropped, mut visits) = (keys_bytes(&commit), 0, 0_u64);
        for key in commit.keys() {
            visits += 1;
            let Some(events) = held.entities.get_mut(key) else {
                continue;
            };
            let before = room_bytes(events);
            while events
                .front()
                .is_some_and(|event| event.epoch <= commit.epoch)
            {
                if let Some(event) = events.pop_front() {
                    freed += event_heap_bytes(&event);
                    dropped += 1;
                    visits += 1;
                }
            }
            if events.is_empty() {
                freed += key_heap_bytes(key) + before;
                held.entities.remove(key);
            } else {
                // Twice the room the events need at most: what a prune takes
                // from a long history is given back.
                if events.capacity() > 2 * events.len() {
                    events.shrink_to_fit();
                }
                freed += before - room_bytes(events);
            }
        }
        if held.commits.capacity() > 2 * held.commits.len() {
            held.commits.shrink_to_fit();
        }
        freed += (queue_before - held.commits.capacity()) * COMMIT_BYTES;
        // Under a quarter of its buckets full, or emptied, the table is cut
        // to what it holds. Each cut follows the removal of a fixed share of
        // the entries it held at its last growth or cut (about 3/7 after a
        // growth, more after a cut), so it costs O(1) a removal amortised.
        // (Its capacity is no measure here: a removal that leaves a tombstone
        // lowers it.)
        let table = held.entities.allocation_size();
        if table > 0 && held.entities.len() * 4 * ENTRY_SLOT_BYTES < table {
            held.entities.shrink_to_fit();
            freed += table - held.entities.allocation_size();
        }
        // One writer, under the events' write lock: a load and a store.
        let counted = self.counted.load(Ordering::Relaxed);
        debug_assert!(freed <= counted, "freed {freed} of {counted} counted");
        self.counted
            .store(counted.saturating_sub(freed), Ordering::Relaxed);
        self.held_events.store(
            self.held_events.load(Ordering::Relaxed) - dropped,
            Ordering::Relaxed,
        );
        self.prune_visits.fetch_add(visits, Ordering::Relaxed);
        Some(dropped)
    }

    /// Records a change event.
    pub fn record(&self, event: ChangeEvent) {
        {
            let mut held = self.events.write();
            self.push(&mut held, event);
            self.prune_to_bytes(&mut held);
        }
        self.settle();
    }

    /// Records a batch of change events with a single write-lock acquisition.
    pub fn record_batch(&self, events: impl IntoIterator<Item = ChangeEvent>) {
        {
            let mut held = self.events.write();
            for event in events {
                self.push(&mut held, event);
            }
            self.prune_to_bytes(&mut held);
        }
        self.settle();
    }

    /// The byte bound in force (see
    /// [`CdcRetentionConfig::max_bytes`]): `None` for a log no database
    /// bounded.
    #[must_use]
    pub fn max_bytes(&self) -> Option<usize> {
        self.retention.max_bytes
    }

    /// When the log passes the byte bound, drops its oldest whole commits
    /// until it is at or below three quarters of the bound; the newest commit
    /// is always kept. A prune costs what it drops.
    fn prune_to_bytes(&self, held: &mut Held) {
        let Some(max) = self.retention.max_bytes else {
            return;
        };
        if self.counted.load(Ordering::Relaxed) <= max {
            return;
        }
        let target = max - max / 4;
        let began = self.counted.load(Ordering::Relaxed);
        let mut before_last = began;
        while self.counted.load(Ordering::Relaxed) > target && held.commits.len() > 1 {
            before_last = self.counted.load(Ordering::Relaxed);
            self.pop_commit(held);
        }
        *self.byte_prune.lock() = Some((began, before_last, self.counted.load(Ordering::Relaxed)));
    }

    /// The events whose count differs from the walk, as the line's name with
    /// the counted and the walked bytes: empty when the count is right. For
    /// tests and explicit checks: it walks every entity.
    #[doc(hidden)]
    #[must_use]
    pub fn account_mismatches(&self) -> Vec<(String, usize, usize)> {
        let (walked, _, events) = self.heap_memory_bytes();
        let mut mismatches = Vec::new();
        let counted = self.counted.load(Ordering::Relaxed);
        if counted != walked {
            mismatches.push(("bytes".to_string(), counted, walked));
        }
        let held = self.held_events.load(Ordering::Relaxed);
        if held != events {
            mismatches.push(("events".to_string(), held, events));
        }
        mismatches
    }

    /// The bytes of the log's grant in the buffer manager's execution
    /// buffers: none before a database attaches its memory.
    #[must_use]
    pub fn granted_bytes(&self) -> usize {
        self.memory.granted.load(Ordering::Relaxed)
    }

    /// [`heap_memory_bytes`](Self::heap_memory_bytes) as the log counts it,
    /// without a walk: `(bytes, entities, events)`.
    #[must_use]
    pub fn counted_memory_bytes(&self) -> (usize, usize, usize) {
        let entities = self.events.read().entities.len();
        (
            self.counted.load(Ordering::Relaxed),
            entities,
            self.held_events.load(Ordering::Relaxed),
        )
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
            deleted_with: None,
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
            deleted_with: None,
            triple_subject: None,
            triple_predicate: None,
            triple_object: None,
            triple_graph: None,
        });
    }

    /// Records an RDF triple insertion.
    ///
    /// The terms must be N-Triples encoded (e.g. `<http://example.org/s>`,
    /// `"hello"`, `"42"^^<http://www.w3.org/2001/XMLSchema#integer>`). The
    /// event carries the parsed terms and the id a commit gives the triple.
    /// `graph` is the named graph's IRI without angle brackets (as SPARQL's
    /// `GRAPH <iri>` names it), or `None` for the default graph.
    ///
    /// Terms that are not one N-Triples triple (a blank argument included),
    /// or a build without the triple store (the `triple-store` feature) to
    /// read them with, record nothing: the call logs a warning that names
    /// why (in a build with `tracing`) and counts in
    /// [`dropped_lines`](Self::dropped_lines).
    pub fn record_triple_insert(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        graph: Option<&str>,
        epoch: EpochId,
    ) {
        self.record_parsed_triple(
            [subject, predicate, object],
            graph,
            ChangeKind::Create,
            epoch,
        );
    }

    /// Records an RDF triple deletion.
    ///
    /// The terms must be N-Triples encoded, and `graph` is as for
    /// [`record_triple_insert`](Self::record_triple_insert); terms that do not
    /// parse record nothing, warn and count in
    /// [`dropped_lines`](Self::dropped_lines), as there.
    pub fn record_triple_delete(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        graph: Option<&str>,
        epoch: EpochId,
    ) {
        self.record_parsed_triple(
            [subject, predicate, object],
            graph,
            ChangeKind::Delete,
            epoch,
        );
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
            deleted_with: None,
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
            deleted_with: None,
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
            .entities
            .get(&(graph.map(str::to_string), entity_id))
            .map_or_default(|events| events.iter().cloned().collect())
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
            .entities
            .get(&(graph.map(str::to_string), entity_id))
            .map_or_default(|events| {
                events
                    .iter()
                    .filter(|e| e.epoch >= since_epoch)
                    .cloned()
                    .collect()
            })
    }

    /// Returns all change events across all entities and graphs in an epoch
    /// range, in commit order (by epoch, then in the order each commit
    /// applied its writes, which their timestamps follow); each event names
    /// its graph.
    #[must_use]
    pub fn changes_between(&self, start_epoch: EpochId, end_epoch: EpochId) -> Vec<ChangeEvent> {
        let guard = self.events.read();
        let mut results = Vec::new();
        for events in guard.entities.values() {
            for event in events {
                if event.epoch >= start_epoch && event.epoch <= end_epoch {
                    results.push(event.clone());
                }
            }
        }
        results.sort_by_key(|e| (e.epoch, e.timestamp));
        results
    }

    /// Returns the total number of recorded events.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events
            .read()
            .entities
            .values()
            .map(VecDeque::len)
            .sum()
    }

    /// Estimates the heap footprint of the log in bytes.
    ///
    /// Counts the entity-keyed hash table by its allocation, the room of each
    /// entity's events, what each event owns on the heap (its strings, labels
    /// and property snapshots, a value as [`Value::estimated_size_bytes`]
    /// reads it), and the queue of commits with their lists of keys.
    #[must_use]
    pub fn heap_memory_bytes(&self) -> (usize, usize, usize) {
        let guard = self.events.read();
        let entity_count = guard.entities.len();
        let mut event_count = 0usize;
        let mut bytes = guard.commits.capacity() * COMMIT_BYTES
            + guard.commits.iter().map(keys_bytes).sum::<usize>()
            + guard.entities.allocation_size();
        for (key, events) in &guard.entities {
            event_count += events.len();
            bytes += key_heap_bytes(key)
                + room_bytes(events)
                + events.iter().map(event_heap_bytes).sum::<usize>();
        }
        (bytes, entity_count, event_count)
    }

    /// Removes all events with `epoch < min_epoch`: the commits before it,
    /// oldest first.
    ///
    /// Entities whose entire history falls below the threshold are removed
    /// from the map entirely.
    ///
    /// The commit path records in epoch order. An event the public helpers
    /// record with an epoch older than an event of its entity already held
    /// goes with that newer event (see `list_key`), so it can outlive a
    /// `min_epoch` above its own until the newer one goes.
    pub fn prune_before(&self, min_epoch: EpochId) {
        {
            let mut held = self.events.write();
            while held
                .commits
                .front()
                .is_some_and(|commit| commit.epoch < min_epoch)
            {
                self.pop_commit(&mut held);
            }
        }
        self.settle();
    }

    /// Prunes the oldest events to stay within `max_events`, if configured:
    /// whole commits, oldest first, until the log holds at most that many.
    pub fn prune_to_limit(&self) {
        let Some(max) = self.retention.max_events else {
            return;
        };
        if self.held_events.load(Ordering::Relaxed) <= max {
            return;
        }
        {
            let mut held = self.events.write();
            while self.held_events.load(Ordering::Relaxed) > max {
                if self.pop_commit(&mut held).is_none() {
                    break;
                }
            }
        }
        self.settle();
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

    /// The bytes the log counts, without a walk.
    fn memory_usage(&self) -> usize {
        self.counted.load(Ordering::Relaxed)
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
        if target_bytes == 0 {
            return 0;
        }
        // The oldest whole commits, until the count dropped by the target. The
        // count is read under the write lock, so a record cannot land between
        // the reading and the pops.
        let before = {
            let mut held = self.events.write();
            let before = self.counted.load(Ordering::Relaxed);
            if before == 0 {
                return 0;
            }
            while before - self.counted.load(Ordering::Relaxed) < target_bytes {
                if self.pop_commit(&mut held).is_none() {
                    break;
                }
            }
            before
        };
        // The caller (the eviction cycle) reads the buffer manager's count:
        // what the grant gave back is what it freed. Without a manager the
        // counted bytes are all there is.
        if self.memory.manager.get().is_some() {
            self.settle_tight()
        } else {
            before.saturating_sub(self.counted.load(Ordering::Relaxed))
        }
    }

    fn can_evict(&self) -> bool {
        true
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

impl CdcLog {
    /// Records the event of the N-Triples `terms` (subject, predicate,
    /// object) in the graph with storage key `graph` (`None` for the default
    /// graph), as a commit records it: the terms parsed, then the event
    /// [`triple_event`] builds, with the normalised terms and their
    /// [`triple_id`]. Terms that are not one triple (a blank argument
    /// included, which a shifted line could hide) record nothing and take no
    /// timestamp: the warning names why, and [`dropped_lines`](Self::dropped_lines)
    /// counts it.
    #[cfg(feature = "triple-store")]
    fn record_parsed_triple(
        &self,
        [subject, predicate, object]: [&str; 3],
        graph: Option<&str>,
        kind: ChangeKind,
        epoch: EpochId,
    ) {
        let line = format!("{subject} {predicate} {object} .");
        let refused = |reason: String| {
            grafeo_common::grafeo_warn!("no change event recorded: `{line}` {reason}");
            self.drop_line();
        };
        if [subject, predicate, object]
            .iter()
            .any(|term| term.trim().is_empty())
        {
            return refused("has a blank term".to_string());
        }
        let triple = match grafeo_core::graph::rdf::parse_ntriples_text(&line) {
            Ok(triples) => match triples.as_slice() {
                [triple] => grafeo_common::storage::log_record::TripleRecord::from(triple),
                _ => return refused("is not one N-Triples triple".to_string()),
            },
            Err(error) => return refused(format!("is no N-Triples triple: {error}")),
        };
        self.record(triple_event(graph, &triple, kind, epoch, self.clock.now()));
    }

    /// A build without the triple store has no reader for the terms, so it
    /// records nothing, warns and counts the call.
    #[cfg(not(feature = "triple-store"))]
    fn record_parsed_triple(
        &self,
        terms: [&str; 3],
        _graph: Option<&str>,
        _kind: ChangeKind,
        _epoch: EpochId,
    ) {
        grafeo_common::grafeo_warn!(
            "no change event recorded for `{} {} {}`: reading N-Triples terms needs the triple \
             store (the `triple-store` feature)",
            terms[0],
            terms[1],
            terms[2]
        );
        self.drop_line();
    }
}

/// The id of `triple` in the graph with storage key `graph` (`None` for the
/// default graph): the first 8 bytes, little endian, of SHA-256 over
/// [`canonical_triple_bytes`]. The same in every process and release. A
/// 64-bit hash: unique in practice, not guaranteed, so a consumer that must
/// not conflate two triples keys on the terms the event carries.
#[cfg(feature = "triple-store")]
pub(crate) fn triple_id(
    triple: &grafeo_common::storage::log_record::TripleRecord,
    graph: Option<&str>,
) -> u64 {
    use sha2::Digest as _;

    let digest = sha2::Sha256::digest(canonical_triple_bytes(triple, graph));
    let mut first = [0; 8];
    first.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(first)
}

/// The encoding [`triple_id`] hashes, kept apart from the log's encoding so
/// a revision of the log moves no id. For the subject, the predicate and the
/// object in turn: a tag byte (0 an IRI, 1 a blank node, 2 a literal), then,
/// each as its length (u64, little endian) and its UTF-8 bytes, the IRI, the
/// blank node's label (without `_:`), or the literal's lexical form,
/// datatype IRI and language tag (length 0 without one; a plain literal has
/// the datatype `xsd:string`). Then a tag byte for the graph, 0 for the
/// default graph and 1 for a named one, followed for a named graph by its
/// storage key the same way: a named graph with an empty name keeps ids of
/// its own.
#[cfg(feature = "triple-store")]
fn canonical_triple_bytes(
    triple: &grafeo_common::storage::log_record::TripleRecord,
    graph: Option<&str>,
) -> Vec<u8> {
    use grafeo_common::storage::log_record::TermRecord;

    fn text(out: &mut Vec<u8>, text: &str) {
        out.extend_from_slice(&(text.len() as u64).to_le_bytes());
        out.extend_from_slice(text.as_bytes());
    }
    let mut out = Vec::new();
    for term in [&triple.subject, &triple.predicate, &triple.object] {
        match term {
            TermRecord::Iri(iri) => {
                out.push(0);
                text(&mut out, iri);
            }
            TermRecord::Blank(label) => {
                out.push(1);
                text(&mut out, label);
            }
            TermRecord::Literal {
                value,
                datatype,
                language,
            } => {
                out.push(2);
                text(&mut out, value);
                text(&mut out, datatype);
                text(&mut out, language.as_deref().unwrap_or_default());
            }
        }
    }
    match graph {
        None => out.push(0),
        Some(key) => {
            out.push(1);
            text(&mut out, key);
        }
    }
    out
}

/// The timestamps a commit reserved for its writes (see
/// [`CdcLog::reserve_for`]): `writes` consecutive values from `first`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CommitStamps {
    /// The first timestamp, write 0's.
    pub first: HlcTimestamp,
    /// How many writes the commit has, and so how many timestamps it holds.
    pub writes: usize,
}

impl CdcLog {
    /// The timestamps of a commit that changed `set`, reserved once for all
    /// its writes: one per write of
    /// [`ChangeSet::ops`](grafeo_common::change::ChangeSet::ops), a write
    /// folded into a create included. `None` for a set without writes.
    /// Called where commits are serialized in epoch order, so a later
    /// commit's timestamps are larger.
    pub(crate) fn reserve_for(
        &self,
        set: &grafeo_common::change::ChangeSet,
    ) -> Option<CommitStamps> {
        let writes = set.ops().count();
        let count = std::num::NonZeroU64::new(u64::try_from(writes).ok()?)?;
        Some(CommitStamps {
            first: self.clock.reserve(count),
            writes,
        })
    }

    /// [`record_commit`](Self::record_commit) with the timestamps the commit
    /// reserved when it wrote its log group (`first`), or, without one,
    /// reserved now: a commit without a log group reserves in its publish
    /// turn, where commits are serialized in epoch order too.
    /// The verdicts of the log's image rule come with the timestamps
    /// (`logged`: the first one and the entries whose whole image is
    /// refused, with what the log keeps of each, see
    /// [`ChangeSet::refused_images`](grafeo_common::change::ChangeSet::refused_images)),
    /// or, without a log group, are taken here once: what the log leaves
    /// out, the events leave out.
    pub(crate) fn record_commit_at(
        &self,
        set: &grafeo_common::change::ChangeSet,
        epoch: EpochId,
        logged: Option<(
            HlcTimestamp,
            &[(usize, grafeo_common::storage::log_record::ImageLogged)],
        )>,
    ) {
        match logged {
            Some((first, refused)) => {
                let writes = set.ops().count();
                self.record_commit(set, epoch, CommitStamps { first, writes }, refused);
            }
            None => {
                if let Some(stamps) = self.reserve_for(set) {
                    self.record_commit(set, epoch, stamps, &set.refused_images());
                }
            }
        }
    }

    /// Records the events of a commit at `epoch` that changed what `set`
    /// holds, with the timestamps `stamps` (from
    /// [`reserve_for`](Self::reserve_for)); see [`events_for_ops`].
    pub(crate) fn record_commit(
        &self,
        set: &grafeo_common::change::ChangeSet,
        epoch: EpochId,
        stamps: CommitStamps,
        refused: &[(usize, grafeo_common::storage::log_record::ImageLogged)],
    ) {
        let ops = set.ops_logged(refused).map(|(slot, op, before, logged)| {
            let graph = set.graph(slot).and_then(|graph| graph.key.as_deref());
            (graph, op, before, logged)
        });
        self.record_batch(events_for_ops(ops, stamps.writes, epoch, stamps.first));
    }
}

/// The change events of a commit at `epoch` whose writes are `ops`, in the
/// order applied, each with its graph's storage key (`None` for the default
/// graph): one per write of [`ChangeSet::ops`](grafeo_common::change::ChangeSet::ops)
/// (each data entry, and each row a bulk write kept with its range; a range
/// without its rows, a bulk import's, has none). Write `i` (from 0) is
/// timestamped `first.plus(i)`; `writes` is how many there are (the room the
/// events take). A triple's event names its RDF graph as
/// `triple_graph`, as the RDF bulk writes report theirs. An event is built
/// from the write's op and before-image, as much of it as the log keeps
/// (`logged`): a create carries its labels or type and endpoints and the
/// values it was created with, a delete what the entity held (its labels or
/// type and endpoints alone when the log leaves its properties out), an
/// update the value or the labels before and after. The
/// events are then folded into the transaction's net changes (see
/// [`net_changes`]); a folded write's timestamp is used up.
pub(crate) fn events_for_ops<'a>(
    ops: impl Iterator<
        Item = (
            Option<&'a str>,
            &'a grafeo_common::change::DataOp,
            &'a grafeo_common::change::Before,
            grafeo_common::storage::log_record::ImageLogged,
        ),
    >,
    writes: usize,
    epoch: EpochId,
    first: HlcTimestamp,
) -> Vec<ChangeEvent> {
    use grafeo_common::change::{Before, DataOp};
    use grafeo_common::storage::log_record::ImageLogged;

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
    let mut events = Vec::with_capacity(writes);
    let mut timestamp = first;
    for (graph, op, before, logged) in ops {
        let stamp = timestamp;
        timestamp = timestamp.plus(1);
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
                    graph,
                    triple,
                    ChangeKind::Create,
                    epoch,
                    stamp,
                ));
                continue;
            }
            #[cfg(feature = "triple-store")]
            DataOp::DeleteTriple { triple } => {
                events.push(triple_event(
                    graph,
                    triple,
                    ChangeKind::Delete,
                    epoch,
                    stamp,
                ));
                continue;
            }
            // A build without the triple store records no triple.
            #[cfg(not(feature = "triple-store"))]
            DataOp::InsertTriple { .. } | DataOp::DeleteTriple { .. } => continue,
        };
        let mut event = ChangeEvent {
            entity_id,
            graph: graph.map(ToString::to_string),
            kind,
            epoch,
            timestamp: stamp,
            before: None,
            after: None,
            labels: None,
            before_labels: None,
            edge_type: None,
            src_id: None,
            dst_id: None,
            deleted_with: None,
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
            // A delete's `before` is what the entity held, an empty map
            // for none, and `None` when the log left the properties out.
            (DataOp::DeleteNode { .. }, Before::Node(image)) => {
                event.labels = Some(names(&image.labels));
                event.before = (logged == ImageLogged::Whole)
                    .then(|| values(&image.properties).unwrap_or_default());
            }
            (DataOp::DeleteEdge { .. }, Before::Edge(image)) => {
                event.edge_type = Some(image.edge_type.to_string());
                event.src_id = Some(image.src.as_u64());
                event.dst_id = Some(image.dst.as_u64());
                event.before = (logged == ImageLogged::Whole)
                    .then(|| values(&image.properties).unwrap_or_default());
                event.deleted_with = image.deleted_with.map(|node| node.as_u64());
            }
            (
                DataOp::SetNodeProperty { key, value, .. }
                | DataOp::SetEdgeProperty { key, value, .. },
                Before::Value(old),
            ) => {
                event.before = old.as_ref().and_then(|old| one(key, old));
                event.after = one(key, value);
            }
            // A set whose old value was too large to log: the new value.
            (
                DataOp::SetNodeProperty { key, value, .. }
                | DataOp::SetEdgeProperty { key, value, .. },
                Before::Absent,
            ) => {
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
    net_changes(events)
}

/// The event of a triple entry of the graph with storage key `graph`
/// (`kind` says whether it was inserted or deleted), as
/// [`CdcLog::record_triple_insert`] and [`CdcLog::record_triple_delete`]
/// build it: the terms as N-Triples strings.
#[cfg(feature = "triple-store")]
fn triple_event(
    graph: Option<&str>,
    triple: &grafeo_common::storage::log_record::TripleRecord,
    kind: ChangeKind,
    epoch: EpochId,
    timestamp: HlcTimestamp,
) -> ChangeEvent {
    let (subject, predicate, object) = crate::transaction::ntriples_terms(triple);
    let graph = graph.map(ToString::to_string);
    ChangeEvent {
        entity_id: EntityId::Triple(triple_id(triple, graph.as_deref())),
        graph: graph.clone(),
        kind,
        epoch,
        timestamp,
        before: None,
        after: None,
        labels: None,
        before_labels: None,
        edge_type: None,
        src_id: None,
        dst_id: None,
        deleted_with: None,
        triple_subject: Some(subject),
        triple_predicate: Some(predicate),
        triple_object: Some(object),
        triple_graph: graph,
    }
}

/// What a transaction's earlier events did to an entity, for [`net_changes`].
enum Earlier {
    /// It created the entity: the create's place.
    Created(usize),
    /// It updated an entity that existed before: the updates' places.
    Updated(Vec<usize>),
    /// It deleted the entity: the delete's place.
    Deleted(usize),
}

/// Folds a transaction's events into its net changes.
///
/// - Later changes to an entity the transaction created fold into the
///   create, so it carries the entity as the transaction left it: its final
///   labels and properties. Statements and the direct API both create an
///   entity and then set its properties and labels one by one.
/// - A create folded with its delete leaves nothing: an entity created and
///   deleted in the transaction (its edges with it, through `DETACH DELETE`)
///   existed at no committed epoch.
/// - Updates to an entity that existed before fold into its delete, which
///   then shows what it held before the transaction.
/// - A triple deleted and inserted again leaves nothing: it is the same
///   triple, there before and after.
/// - Updates to an entity that exists before and after the transaction are
///   not folded: each write reports its own update.
///
/// Entity ids repeat across graphs, so the folding stays within one graph. A
/// triple's id is a 64-bit hash of its terms, so two events of one id fold
/// only when their terms are equal too: two triples whose ids collide stay
/// two.
fn net_changes(events: Vec<ChangeEvent>) -> Vec<ChangeEvent> {
    let mut net: Vec<Option<ChangeEvent>> = Vec::with_capacity(events.len());
    let mut earlier: HashMap<(Option<String>, EntityId), Earlier> = HashMap::new();
    for mut event in events {
        let key = (event.graph.clone(), event.entity_id);
        let at = net.len();
        match (earlier.get_mut(&key), &event.kind) {
            (Some(Earlier::Created(create)), ChangeKind::Update) => {
                if let Some(create) = net[*create].as_mut() {
                    fold_update(create, event);
                }
            }
            (Some(Earlier::Created(create)), ChangeKind::Delete)
                if same_entity(net[*create].as_ref(), &event) =>
            {
                net[*create] = None;
                earlier.remove(&key);
            }
            (Some(Earlier::Deleted(delete)), ChangeKind::Create)
                if event.entity_id.is_triple() && same_entity(net[*delete].as_ref(), &event) =>
            {
                net[*delete] = None;
                earlier.remove(&key);
            }
            (Some(Earlier::Updated(updates)), ChangeKind::Update) => {
                updates.push(at);
                net.push(Some(event));
            }
            (Some(Earlier::Updated(updates)), ChangeKind::Delete) => {
                let updates = updates.iter().filter_map(|&update| net[update].take());
                restore_before(&mut event, updates);
                earlier.insert(key, Earlier::Deleted(at));
                net.push(Some(event));
            }
            (_, kind) => {
                let next = match kind {
                    ChangeKind::Create => Earlier::Created(at),
                    ChangeKind::Update => Earlier::Updated(vec![at]),
                    ChangeKind::Delete => Earlier::Deleted(at),
                };
                earlier.insert(key, next);
                net.push(Some(event));
            }
        }
    }
    net.into_iter().flatten().collect()
}

/// Whether `earlier`, an event of the same key as `event`, is of the same
/// entity: always for a node or an edge (their ids are given out once), and
/// for a triple when its terms are equal too.
fn same_entity(earlier: Option<&ChangeEvent>, event: &ChangeEvent) -> bool {
    earlier.is_some_and(|earlier| {
        !event.entity_id.is_triple()
            || (earlier.triple_subject == event.triple_subject
                && earlier.triple_predicate == event.triple_predicate
                && earlier.triple_object == event.triple_object
                && earlier.triple_graph == event.triple_graph)
    })
}

/// Gives `delete` what the entity held before the transaction: the first
/// value each of `updates` changed, and the labels before the first label
/// change. A delete whose properties the log left out (`before` is `None`)
/// keeps them left out: the updates' old values alone would be a partial
/// image.
fn restore_before(delete: &mut ChangeEvent, updates: impl Iterator<Item = ChangeEvent>) {
    let mut restored = std::collections::HashSet::new();
    let mut labels_restored = false;
    for update in updates {
        if let Some(labels) = update.before_labels
            && !labels_restored
        {
            delete.labels = Some(labels);
            labels_restored = true;
        }
        let Some(properties) = delete.before.as_mut() else {
            continue;
        };
        let keys = update
            .after
            .iter()
            .chain(&update.before)
            .flat_map(HashMap::keys);
        for key in keys.cloned().collect::<Vec<_>>() {
            if !restored.insert(key.clone()) {
                continue;
            }
            match update.before.as_ref().and_then(|before| before.get(&key)) {
                Some(value) => properties.insert(key, value.clone()),
                None => properties.remove(&key),
            };
        }
    }
    // A delete's `before` stays a map, empty when the entity held nothing.
}

/// Folds `update` into `create`, an event of the entity it changed.
fn fold_update(create: &mut ChangeEvent, event: ChangeEvent) {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A triple's event with the terms `object` names, under the id `id`.
    fn triple(id: u64, object: &str, kind: ChangeKind) -> ChangeEvent {
        ChangeEvent {
            entity_id: EntityId::Triple(id),
            graph: None,
            kind,
            epoch: EpochId(7),
            timestamp: HlcTimestamp::new(1, 0),
            before: None,
            after: None,
            labels: None,
            before_labels: None,
            edge_type: None,
            src_id: None,
            dst_id: None,
            deleted_with: None,
            triple_subject: Some("<http://ex.org/alix>".to_string()),
            triple_predicate: Some("<http://ex.org/knows>".to_string()),
            triple_object: Some(object.to_string()),
            triple_graph: None,
        }
    }

    /// Two triples whose 64-bit ids collide are two triples: a create of one
    /// and a delete of the other in one transaction both stay, and a delete
    /// and a create of one fold only when the terms are equal too.
    #[test]
    fn triples_fold_only_when_their_terms_are_equal() {
        let gus = "<http://ex.org/gus>";
        let mia = "<http://ex.org/mia>";
        let kept = |events: Vec<ChangeEvent>| {
            net_changes(events)
                .into_iter()
                .map(|event| (event.kind, event.triple_object.unwrap()))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            kept(vec![
                triple(42, gus, ChangeKind::Create),
                triple(42, mia, ChangeKind::Delete),
            ]),
            [
                (ChangeKind::Create, gus.to_string()),
                (ChangeKind::Delete, mia.to_string())
            ],
            "a create and a colliding delete"
        );
        assert_eq!(
            kept(vec![
                triple(42, gus, ChangeKind::Delete),
                triple(42, mia, ChangeKind::Create),
            ]),
            [
                (ChangeKind::Delete, gus.to_string()),
                (ChangeKind::Create, mia.to_string())
            ],
            "a delete and a colliding create"
        );
        assert!(
            kept(vec![
                triple(42, gus, ChangeKind::Delete),
                triple(42, gus, ChangeKind::Create),
            ])
            .is_empty(),
            "the same triple deleted and inserted again"
        );
        assert!(
            kept(vec![
                triple(42, gus, ChangeKind::Create),
                triple(42, gus, ChangeKind::Delete),
            ])
            .is_empty(),
            "the same triple inserted and deleted"
        );
    }

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
            max_bytes: None,
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
            max_bytes: None,
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
            max_bytes: None,
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
            max_bytes: None,
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
            max_bytes: None,
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

    /// An eviction of fewer bytes than one commit holds drops that one
    /// commit: it pops until what it asked for is freed.
    #[test]
    fn a_tiny_eviction_drops_one_commit() {
        let log = CdcLog::new();
        for epoch in 1..=10 {
            log.record_create_node(NodeId::new(epoch), EpochId(epoch), None, None);
        }
        assert_eq!(log.event_count(), 10);

        let freed = log.evict(100);
        assert!(freed >= 100, "{freed}");
        assert_eq!(log.event_count(), 9);
        assert_eq!(log.account_mismatches(), Vec::new());
    }

    /// An eviction drops the oldest commits until it freed what it was asked
    /// for, counted, not an estimate of 256 bytes an event: asked for one
    /// commit's bytes of 64 KiB commits, it drops one or two, not the log.
    #[test]
    fn an_eviction_frees_what_it_was_asked() {
        let log = CdcLog::new();
        for epoch in 1..=40 {
            record_named(&log, epoch, epoch, 64 * 1024);
        }
        let before = log.counted_memory_bytes().0;
        let freed = log.evict(64 * 1024);
        let (counted, _, events) = log.counted_memory_bytes();
        assert!(freed >= 64 * 1024, "{freed}");
        assert_eq!(freed, before - counted);
        assert!(events >= 38, "{events} events kept");
        assert_eq!(log.account_mismatches(), Vec::new());
    }

    /// A create of node `id` at `epoch` whose `name` holds `bytes` bytes.
    fn record_named(log: &CdcLog, id: u64, epoch: u64, bytes: usize) {
        let props = HashMap::from([("name".to_string(), Value::from("j".repeat(bytes)))]);
        log.record_create_node(NodeId::new(id), EpochId(epoch), Some(props), None);
    }

    /// The log counts what each event owns (its property snapshot, labels
    /// and strings), at its push and its prune, as the walk does.
    #[test]
    fn the_count_holds_what_each_event_owns() {
        let log = CdcLog::new();
        record_named(&log, 1, 1, 100_000);
        log.record_create_node(
            NodeId::new(2),
            EpochId(2),
            None,
            Some(vec!["Vincent".to_string(), "Mia".to_string()]),
        );
        let (counted, _, _) = log.counted_memory_bytes();
        assert!(counted >= 100_000, "the snapshot is counted: {counted}");
        assert_eq!(log.account_mismatches(), Vec::new());

        log.prune_before(EpochId(2));
        let (counted, _, events) = log.counted_memory_bytes();
        assert_eq!(events, 1);
        assert!(counted < 100_000, "the pruned snapshot is not: {counted}");
        assert_eq!(log.account_mismatches(), Vec::new());
        log.prune_before(EpochId(3));
        assert_eq!(log.counted_memory_bytes().0, 0);
    }

    /// A prune that leaves a long history short gives its room back: the
    /// count drops with it.
    #[test]
    fn a_partial_prune_gives_the_room_back() {
        let log = CdcLog::new();
        for epoch in 1..=64 {
            log.record_create_node(NodeId::new(1), EpochId(epoch), None, None);
        }
        log.prune_before(EpochId(60));
        let (counted, entities, events) = log.counted_memory_bytes();
        assert_eq!((entities, events), (1, 5));
        assert!(
            counted < 16 * std::mem::size_of::<ChangeEvent>(),
            "{counted}"
        );
        assert_eq!(log.account_mismatches(), Vec::new());
    }

    /// A prune or an eviction that empties the log gives the entities table
    /// back, so the count that reads nothing holds nothing (the allocator's
    /// view is `tests/cdc_log_in_the_account.rs`).
    #[test]
    fn an_emptied_log_gives_its_table_back() {
        for evict in [false, true] {
            let log = CdcLog::new();
            for id in 1..=10_000 {
                log.record_create_node(NodeId::new(id), EpochId(id), None, None);
            }
            assert_eq!(log.account_mismatches(), Vec::new());
            if evict {
                log.evict(usize::MAX);
            } else {
                log.prune_before(EpochId(u64::MAX));
            }
            assert_eq!(log.event_count(), 0);
            assert_eq!(
                log.events.read().entities.capacity(),
                0,
                "the table is given back (eviction: {evict})"
            );
            assert_eq!(log.counted_memory_bytes().0, 0);
            assert_eq!(log.account_mismatches(), Vec::new());
        }
    }

    /// The table shrinks as a prune empties it, in steps: the prune of one
    /// commit that leaves it under a quarter of its buckets full cuts it, and
    /// no other does; the count follows the walk at every step.
    #[test]
    fn a_draining_table_shrinks_in_steps() {
        let log = CdcLog::new();
        for id in 1..=4_096 {
            log.record_create_node(NodeId::new(id), EpochId(id), None, None);
        }
        let table = |log: &CdcLog| {
            let held = log.events.read();
            (held.entities.len(), held.entities.allocation_size())
        };
        let full = table(&log).1;
        let mut cuts = 0;
        for epoch in 2..=4_096 - 512 {
            let (_, before) = table(&log);
            log.prune_before(EpochId(epoch));
            let (len, after) = table(&log);
            if len * 4 * ENTRY_SLOT_BYTES < before {
                assert!(after < before, "epoch {epoch}: {len} entries, not cut");
                cuts += 1;
            } else {
                assert_eq!(after, before, "epoch {epoch}: {len} entries, cut");
            }
        }
        let (len, after) = table(&log);
        assert_eq!(len, 513);
        assert_eq!(cuts, 2, "cut at about 2,048 and 1,024 entries");
        assert!(after < full / 2, "{after} of {full}");
        assert!(log.events.read().entities.capacity() >= len);
        assert_eq!(log.account_mismatches(), Vec::new());
    }

    /// An entity's event with an epoch older than one it already holds (the
    /// public helpers take any epoch) keeps the count equal to the walk as it
    /// is listed, pruned and listed twice under one commit; the older event
    /// goes with the newer one, and the emptied log counts nothing.
    #[test]
    fn an_older_epoch_of_a_held_entity_keeps_the_count() {
        let log = CdcLog::new();
        let check = |what: &str| assert_eq!(log.account_mismatches(), Vec::new(), "{what}");
        log.record_create_node(NodeId::new(1), EpochId(5), None, None);
        log.record_update(
            EntityId::Node(NodeId::new(1)),
            EpochId(3),
            "w",
            None,
            Value::Int64(3),
        );
        log.record_create_node(NodeId::new(2), EpochId(3), None, None);
        check("an older epoch listed");
        // A third event of node 1 at epoch 5, after the older one: node 1 is
        // listed twice under commit 5.
        log.record_update(
            EntityId::Node(NodeId::new(1)),
            EpochId(5),
            "w",
            None,
            Value::Int64(5),
        );
        check("listed twice under one commit");
        log.prune_before(EpochId(4));
        assert_eq!(log.history(EntityId::Node(NodeId::new(2))).len(), 0);
        assert_eq!(
            log.history(EntityId::Node(NodeId::new(1))).len(),
            3,
            "the older event stays with the newer one"
        );
        check("commit 3 pruned");
        log.prune_before(EpochId(6));
        assert_eq!(log.event_count(), 0);
        check("commit 5 pruned");
        assert_eq!(log.counted_memory_bytes().0, 0);
    }

    /// An eviction gives back what it pruned from the grant at once (no slack
    /// kept), and reports the grant's drop: the eviction cycle reads the
    /// buffer manager's count.
    #[test]
    fn an_eviction_reports_the_grant_it_gave_back() {
        let manager = BufferManager::with_budget(1 << 30);
        let log = CdcLog::new();
        log.attach_memory(Arc::clone(&manager));
        for epoch in 1..=40 {
            record_named(&log, epoch, epoch, 64 * 1024);
        }
        let granted = log.granted_bytes();
        let counted = log.counted_memory_bytes().0;
        assert!(
            granted >= counted && counted > 2 * manager.grant_step(),
            "{counted}"
        );

        // The eviction drops commits until it freed what it asked for: 30
        // commits of a little over 64 KiB each, so the 30 oldest go and the
        // 10 newest (about 640 KiB) stay.
        let freed = log.evict(30 * 64 * 1024);
        let (counted, _, events) = log.counted_memory_bytes();
        assert_eq!(events, 10);
        assert_eq!(freed, granted - log.granted_bytes());
        assert!(freed > 0);
        assert!(
            log.granted_bytes() >= counted && log.granted_bytes() <= manager.grant_size(counted),
            "granted {} for {counted}",
            log.granted_bytes()
        );
        assert_eq!(log.account_mismatches(), Vec::new());
    }

    /// Writers and pruners at once leave the grant covering the count: a
    /// settle that read an older count never shrinks what a newer one grew.
    #[test]
    fn concurrent_writes_and_prunes_keep_the_grant_covering_the_count() {
        let manager = BufferManager::with_budget(1 << 30);
        let log = Arc::new(CdcLog::new());
        log.attach_memory(Arc::clone(&manager));
        std::thread::scope(|scope| {
            for writer in 0..3_u64 {
                let log = Arc::clone(&log);
                scope.spawn(move || {
                    for i in 0..200 {
                        record_named(&log, writer * 1_000 + i, 10 + i, 16 * 1024);
                    }
                });
            }
            let log = Arc::clone(&log);
            scope.spawn(move || {
                for epoch in 0..200 {
                    log.prune_before(EpochId(epoch));
                }
            });
        });
        let counted = log.counted_memory_bytes().0;
        assert!(
            log.granted_bytes() >= counted,
            "granted {} for {counted}",
            log.granted_bytes()
        );
        assert_eq!(log.account_mismatches(), Vec::new());
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

    /// A node event of `graph` for the net fold.
    fn node_event(kind: ChangeKind, graph: Option<&str>, id: u64) -> ChangeEvent {
        ChangeEvent {
            entity_id: EntityId::Node(NodeId::new(id)),
            graph: graph.map(ToString::to_string),
            kind,
            epoch: EpochId(1),
            timestamp: HlcTimestamp::new(1, 0),
            before: None,
            after: None,
            labels: None,
            before_labels: None,
            edge_type: None,
            src_id: None,
            dst_id: None,
            deleted_with: None,
            triple_subject: None,
            triple_predicate: None,
            triple_object: None,
            triple_graph: None,
        }
    }

    /// An event of node 3 of `kind`, with `before` and `after` of one key `x`
    /// and the labels `labels`.
    fn valued_node_event(
        kind: ChangeKind,
        before: Option<i64>,
        after: Option<i64>,
        labels: Option<&[&str]>,
    ) -> ChangeEvent {
        let one = |value: Option<i64>| {
            value.map(|value| HashMap::from([("x".to_string(), Value::Int64(value))]))
        };
        ChangeEvent {
            entity_id: EntityId::Node(NodeId::new(3)),
            graph: None,
            kind,
            epoch: EpochId(7),
            timestamp: HlcTimestamp::new(1, 0),
            before: one(before),
            after: one(after),
            labels: labels.map(|labels| labels.iter().map(ToString::to_string).collect()),
            before_labels: None,
            edge_type: None,
            src_id: None,
            dst_id: None,
            deleted_with: None,
            triple_subject: None,
            triple_predicate: None,
            triple_object: None,
            triple_graph: None,
        }
    }

    /// An update and then a delete of a node whose delete left its
    /// properties out (`before` is `None`: the log kept its identity alone)
    /// folds into the delete with its properties still left out, not the
    /// update's old values as a partial image; a delete with its whole image
    /// takes the updates' old values back.
    #[test]
    fn a_delete_without_its_properties_keeps_them_left_out_after_updates() {
        let update = valued_node_event(ChangeKind::Update, Some(1), Some(2), None);
        let left_out = valued_node_event(ChangeKind::Delete, None, None, Some(&["Person"]));
        let net = net_changes(vec![update.clone(), left_out]);
        assert_eq!(net.len(), 1, "{net:?}");
        assert_eq!(net[0].kind, ChangeKind::Delete);
        assert_eq!(net[0].before, None, "the properties stay left out");
        assert_eq!(net[0].labels, Some(vec!["Person".to_string()]));
        let whole = valued_node_event(ChangeKind::Delete, Some(2), None, Some(&["Person"]));
        let net = net_changes(vec![update, whole]);
        assert_eq!(
            net[0].before,
            Some(HashMap::from([("x".to_string(), Value::Int64(1))])),
            "the whole image takes the old value back"
        );
    }

    /// The `Nothing` verdict through `events_for_ops`: the log kept no image,
    /// so a delete reports neither labels nor properties, and a set keeps its
    /// new value with no old one.
    #[test]
    fn a_write_logged_without_its_image_reports_without_it() {
        use grafeo_common::change::{Before, DataOp};
        use grafeo_common::storage::log_record::ImageLogged;
        let delete = DataOp::DeleteNode { id: NodeId::new(3) };
        let set = DataOp::SetNodeProperty {
            id: NodeId::new(4),
            key: "x".into(),
            value: Value::Int64(2),
        };
        let absent = Before::Absent;
        let ops = [
            (None, &delete, &absent, ImageLogged::Nothing),
            (None, &set, &absent, ImageLogged::Nothing),
        ];
        let events = events_for_ops(ops.into_iter(), 2, EpochId(7), HlcTimestamp::new(1, 0));
        assert_eq!(events.len(), 2, "{events:?}");
        assert_eq!(events[0].kind, ChangeKind::Delete);
        assert_eq!(
            (events[0].labels.is_none(), events[0].before.is_none()),
            (true, true)
        );
        assert_eq!(events[1].kind, ChangeKind::Update);
        assert_eq!(events[1].before, None);
        assert_eq!(
            events[1].after,
            Some(HashMap::from([("x".to_string(), Value::Int64(2))]))
        );
    }

    /// The fold keys on the graph as well as the id: a create in one graph and a delete of the
    /// same id in another are two changes; an LPG delete followed by a create of its id (which
    /// the store never does) is not folded away, as a triple's is.
    #[test]
    fn the_net_fold_keeps_graphs_and_models_apart() {
        let kinds = |events: Vec<ChangeEvent>| -> Vec<(ChangeKind, Option<String>)> {
            events
                .into_iter()
                .map(|event| (event.kind, event.graph))
                .collect()
        };
        let folded = net_changes(vec![
            node_event(ChangeKind::Create, Some("g1"), 5),
            node_event(ChangeKind::Delete, Some("g2"), 5),
        ]);
        assert_eq!(
            kinds(folded),
            [
                (ChangeKind::Create, Some("g1".to_string())),
                (ChangeKind::Delete, Some("g2".to_string()))
            ]
        );
        let folded = net_changes(vec![
            node_event(ChangeKind::Delete, None, 7),
            node_event(ChangeKind::Create, None, 7),
        ]);
        assert_eq!(
            kinds(folded),
            [(ChangeKind::Delete, None), (ChangeKind::Create, None)]
        );
    }
}

#[cfg(all(test, feature = "triple-store"))]
#[path = "cdc_triple_id_tests.rs"]
mod triple_id_tests;

/// A build without the triple store reads no N-Triples: the helpers record nothing.
#[cfg(all(test, not(feature = "triple-store")))]
mod without_the_triple_store {
    use super::*;

    #[test]
    fn the_triple_helpers_record_nothing() {
        let log = CdcLog::new();
        let () = log.record_triple_insert("<a>", "<b>", "<c>", None, EpochId(1));
        let () = log.record_triple_delete("<a>", "<b>", "<c>", None, EpochId(1));
        assert_eq!(log.event_count(), 0);
        assert_eq!(log.dropped_lines(), 2, "the count is the signal");
    }
}

/// A refused line is counted in every build: without `tracing` the warning compiles to nothing,
/// so the count is what a caller sees.
#[cfg(test)]
mod refused_lines {
    use super::*;

    #[test]
    fn refused_triple_lines_are_counted() {
        eprintln!("tracing on: {}", cfg!(feature = "tracing"));
        let log = CdcLog::new();
        log.record_triple_insert("<a>", "<b>", "\"unterminated", None, EpochId(1));
        log.record_triple_delete("", "<b>", "<c>", None, EpochId(1));
        assert_eq!(log.event_count(), 0);
        assert_eq!(log.dropped_lines(), 2);
    }
}
