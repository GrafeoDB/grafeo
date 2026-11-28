//! Transaction manager.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use grafeo_common::types::{EdgeId, EpochId, HashableValue, NodeId, Timestamp, TransactionId};
use grafeo_common::utils::error::{Error, ExpiredSnapshot, Result, TransactionError};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use grafeo_core::execution::operators::WriteInProgress;
use parking_lot::{Condvar, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

use super::changes::TransactionChanges;
use super::history::{History, HistoryImage};
use crate::config::HistoryRetention;

/// Why a [`TransactionManager`] refuses commits until the database is opened
/// again (see [`TransactionManager::refuse_commits`]).
#[derive(Debug, Clone)]
#[cfg_attr(
    not(feature = "grafeo-file"),
    expect(
        dead_code,
        reason = "only a database file's checkpoint or restore refuses commits"
    )
)]
pub(crate) enum CommitRefusal {
    /// A checkpoint's outcome is unknown and the database has no WAL: a
    /// commit would be durable only at a checkpoint, and none can run.
    OutcomeUnknown(String),
    /// A restore replaced the stores but its checkpoint failed: a commit on
    /// top would replay onto the previous image. With the cause when the
    /// checkpoint's outcome is unknown, `None` when it failed in a known way.
    FailedRestore(Option<String>),
}

/// State of a transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TransactionState {
    /// Transaction is active.
    Active,
    /// The commit is decided (its epoch is assigned) and its versions, events
    /// and WAL records are being written; it becomes `Committed` when that is
    /// done. Its writes still conflict with other transactions' writes.
    Committing,
    /// Transaction is committed.
    Committed,
    /// Transaction is aborted.
    Aborted,
}

/// Transaction isolation level.
///
/// Controls the consistency guarantees and performance tradeoffs for transactions.
///
/// # Comparison
///
/// | Level | Dirty Reads | Non-Repeatable Reads | Phantom Reads | Write Skew |
/// |-------|-------------|----------------------|---------------|------------|
/// | ReadCommitted | No | Yes | Yes | Yes |
/// | SnapshotIsolation | No | No | No | Yes |
/// | Serializable | No | No | No | No, once queries record their reads (#400) |
///
/// A session refuses `Serializable` until then (`GRAFEO-Q004`): with no reads
/// recorded it could not prevent write skew.
///
/// # Performance
///
/// Higher isolation levels require more bookkeeping:
/// - `ReadCommitted`: Only tracks writes
/// - `SnapshotIsolation`: Tracks writes + snapshot versioning
/// - `Serializable`: Tracks writes + reads + SSI validation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum IsolationLevel {
    /// Read Committed: sees only committed data, but may see different
    /// versions of the same row within a transaction.
    ///
    /// Lowest overhead, highest throughput, but weaker consistency.
    ReadCommitted,

    /// Snapshot Isolation (default): each transaction sees a consistent
    /// snapshot as of transaction start. Prevents non-repeatable reads
    /// and phantom reads.
    ///
    /// Vulnerable to write skew anomaly.
    #[default]
    SnapshotIsolation,

    /// Serializable Snapshot Isolation (SSI): serializability by detecting
    /// read-write conflicts in addition to write-write conflicts.
    ///
    /// Not available yet: queries do not record their reads, so a session
    /// refuses this level (`GRAFEO-Q004`) rather than run snapshot isolation
    /// under its name (#400).
    Serializable,
}

/// Entity identifier for write tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EntityId {
    /// A node.
    Node(NodeId),
    /// An edge.
    Edge(EdgeId),
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

/// A node or edge of one graph: the unit of conflict detection.
///
/// Named graphs number their nodes and edges on their own, so the same ID
/// in two graphs is two entities, and writes to them never conflict.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphEntity {
    /// The graph's storage key; `None` for the default graph.
    pub graph: Option<Arc<str>>,
    /// The node or edge.
    pub entity: EntityId,
}

impl GraphEntity {
    /// An entity of the graph with storage key `graph` (`None`: the
    /// default graph).
    #[must_use]
    pub fn new(graph: Option<Arc<str>>, entity: impl Into<EntityId>) -> Self {
        Self {
            graph,
            entity: entity.into(),
        }
    }
}

impl From<EntityId> for GraphEntity {
    fn from(entity: EntityId) -> Self {
        Self::new(None, entity)
    }
}

impl From<NodeId> for GraphEntity {
    fn from(id: NodeId) -> Self {
        Self::new(None, id)
    }
}

impl From<EdgeId> for GraphEntity {
    fn from(id: EdgeId) -> Self {
        Self::new(None, id)
    }
}

impl std::fmt::Display for GraphEntity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.entity)?;
        if let Some(graph) = &self.graph {
            write!(f, " in graph '{graph}'")?;
        }
        Ok(())
    }
}

/// A value a transaction writes under a UNIQUE, PRIMARY KEY or NODE KEY
/// constraint: the graph, the constraint's label and properties, and the
/// values, one per property. Two transactions that write the same one
/// conflict as two writes of one entity do: between open transactions the
/// first writer wins, and at commit a transaction loses to one that wrote it
/// and committed after it began. Within one transaction a value may be
/// written any number of times.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KeyClaim {
    /// The graph's storage key; `None` for the default graph.
    pub graph: Option<Arc<str>>,
    /// The label the constraint is on.
    pub label: Arc<str>,
    /// The constraint's properties, in its order.
    pub properties: Vec<String>,
    /// The value of each property.
    pub values: Vec<HashableValue>,
}

impl std::fmt::Display for KeyClaim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, ":{}(", self.label)?;
        for (index, (property, value)) in self.properties.iter().zip(&self.values).enumerate() {
            if index > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{property} = {:?}", value.0)?;
        }
        write!(f, ")")?;
        if let Some(graph) = &self.graph {
            write!(f, " in graph '{graph}'")?;
        }
        Ok(())
    }
}

/// How a conflict message names a key or MERGE pattern claim: `on key ...`,
/// `on MERGE ...`.
enum ClaimName<'a> {
    Key(&'a KeyClaim),
    Pattern(&'a dyn std::fmt::Display),
}

impl std::fmt::Display for ClaimName<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Key(key) => write!(f, "on key {key}"),
            Self::Pattern(pattern) => write!(f, "on MERGE {pattern}"),
        }
    }
}

/// How a conflict message names the holder that kept a key or MERGE pattern
/// claim (see [`TransactionManager::record_key`]).
fn holder_kind(committing: bool) -> &'static str {
    if committing {
        "is committing"
    } else {
        "began first"
    }
}

/// The fingerprints of the MERGE patterns a transaction created (see
/// [`TransactionManager::record_pattern`]): the first few inline and the
/// rest in a set, so a statement that claims one or two, as an auto-commit
/// MERGE does, allocates nothing for them.
#[derive(Debug, Default)]
pub(crate) struct PatternSet {
    /// The first fingerprints, `inline[..len]`.
    inline: [u64; PatternSet::INLINE],
    len: usize,
    /// The fingerprints past the first `INLINE`, boxed: most transactions
    /// claim fewer, and every one carries the set.
    spilled: Option<Box<FingerprintSet>>,
}

/// A set of fingerprints, which are hashes already: the set takes each as
/// its own hash.
type FingerprintSet = hashbrown::HashSet<u64, std::hash::BuildHasherDefault<FingerprintHasher>>;

/// The hasher of a [`FingerprintSet`]: a fingerprint is its own hash.
#[derive(Default)]
struct FingerprintHasher(u64);

impl std::hash::Hasher for FingerprintHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        // Only `write_u64` is called for a `u64` key; any other input is
        // folded in, so the hasher stays correct for it.
        for byte in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(*byte);
        }
    }

    fn write_u64(&mut self, fingerprint: u64) {
        self.0 = fingerprint;
    }
}

impl PatternSet {
    /// How many fingerprints are kept inline.
    const INLINE: usize = 4;

    /// Whether the set holds `fingerprint`.
    fn contains(&self, fingerprint: u64) -> bool {
        self.inline[..self.len].contains(&fingerprint)
            || self
                .spilled
                .as_ref()
                .is_some_and(|spilled| spilled.contains(&fingerprint))
    }

    /// Adds `fingerprint`; whether it was new.
    fn insert(&mut self, fingerprint: u64) -> bool {
        if self.contains(fingerprint) {
            return false;
        }
        if self.len < Self::INLINE {
            self.inline[self.len] = fingerprint;
            self.len += 1;
        } else {
            self.spilled
                .get_or_insert_with(Box::default)
                .insert(fingerprint);
        }
        true
    }

    /// Removes `fingerprint`.
    fn remove(&mut self, fingerprint: u64) {
        if let Some(index) = self.inline[..self.len]
            .iter()
            .position(|held| *held == fingerprint)
        {
            self.inline[index] = self.inline[self.len - 1];
            self.len -= 1;
        } else if let Some(spilled) = &mut self.spilled {
            spilled.remove(&fingerprint);
        }
    }

    /// How many fingerprints the set holds.
    fn len(&self) -> usize {
        self.len + self.spilled.as_ref().map_or(0, |spilled| spilled.len())
    }

    /// Whether the set holds none.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The fingerprints, in no order.
    fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        self.inline[..self.len]
            .iter()
            .chain(self.spilled.iter().flat_map(|spilled| spilled.iter()))
            .copied()
    }
}

/// Information about an active transaction.
pub struct TransactionInfo {
    /// Transaction state.
    pub state: TransactionState,
    /// Isolation level for this transaction.
    pub isolation_level: IsolationLevel,
    /// Start epoch (snapshot epoch for reads).
    pub start_epoch: EpochId,
    /// Set of entities written by this transaction.
    pub write_set: FxHashSet<GraphEntity>,
    /// Set of entities read by this transaction (for serializable isolation).
    pub read_set: FxHashSet<GraphEntity>,
    /// The nodes this transaction deleted (in `write_set` too): another
    /// transaction's edge to one of them conflicts with the delete.
    pub delete_set: FxHashSet<GraphEntity>,
    /// The nodes the edges this transaction created end at, claimed against
    /// a delete by another transaction (see
    /// [`TransactionManager::record_endpoints`]).
    pub endpoint_set: FxHashSet<GraphEntity>,
    /// The values the transaction wrote under UNIQUE, PRIMARY KEY and NODE
    /// KEY constraints (see [`TransactionManager::record_key`]).
    pub(crate) key_set: FxHashSet<KeyClaim>,
    /// The fingerprints of the MERGE patterns the transaction created (see
    /// [`TransactionManager::record_pattern`]).
    pub(crate) pattern_set: PatternSet,
    /// The older transaction that took a key or MERGE pattern claim from
    /// this one, when one did, and the claim it took (`on key ...`, `on
    /// MERGE ...`): oldest claim wins (see [`TransactionManager::record_key`]).
    /// The transaction's key and pattern claims no longer count, and it fails
    /// at its next claim and at commit, naming the claim; a rollback to a
    /// savepoint keeps the mark.
    doomed_by: Option<(TransactionId, String)>,
    /// What the transaction changed, while it is open or committing: a
    /// checkpoint reads the committed state of what open transactions
    /// changed from it (see [`TransactionManager::open_change_sets`]).
    /// Dropped once the transaction is committed or aborted.
    pub(crate) changes: Option<Arc<TransactionChanges>>,
    /// Its RDF writes from the start of its commit (the net of each triple
    /// it wrote, per graph), for the conflict checks of the commits after
    /// it: read without its change set, whose lock its commit holds while it
    /// applies them. Kept in the committed-claims index after its publish
    /// while a transaction that began before it is open.
    #[cfg(feature = "triple-store")]
    rdf_writes: Option<KeptRdfWrites>,
    /// Inverses of new claims since the first savepoint or statement mark.
    /// Direct writes and auto-commit transactions never allocate this log.
    claim_undo: Option<Vec<ClaimUndo>>,
    /// Its snapshot as a reader of the read horizon (see [`SnapshotReader`]).
    reader: Arc<SnapshotReader>,
    /// Its commit epoch, from the start of its commit until the commit is taken back
    /// ([`CommitGuard::abort`]) or published.
    pub(crate) commit_epoch: Option<EpochId>,
}

/// A claim-log position inside one active transaction.
#[derive(Clone, Copy)]
pub(crate) struct ClaimMark(usize);

/// One change to conflict claims, undone only after storage undo succeeds.
enum ClaimUndo {
    Write(GraphEntity),
    Delete(GraphEntity),
    Endpoint(GraphEntity),
    Key(KeyClaim),
    Pattern(u64),
}

impl TransactionInfo {
    /// Creates a new transaction info with the given isolation level.
    fn new(
        start_epoch: EpochId,
        isolation_level: IsolationLevel,
        changes: Arc<TransactionChanges>,
        reader: Arc<SnapshotReader>,
    ) -> Self {
        Self {
            state: TransactionState::Active,
            isolation_level,
            start_epoch,
            write_set: FxHashSet::default(),
            read_set: FxHashSet::default(),
            delete_set: FxHashSet::default(),
            endpoint_set: FxHashSet::default(),
            key_set: FxHashSet::default(),
            pattern_set: PatternSet::default(),
            doomed_by: None,
            changes: Some(changes),
            #[cfg(feature = "triple-store")]
            rdf_writes: None,
            claim_undo: None,
            reader,
            commit_epoch: None,
        }
    }

    /// Whether `claim` of `entity` by another transaction conflicts with
    /// this transaction's writes and claims, and why.
    fn conflict_with(&self, claim: Claim, entity: &GraphEntity) -> Option<String> {
        let written = || self.write_set.contains(entity);
        match claim {
            Claim::Write | Claim::Delete if written() => Some(write_conflict(entity)),
            Claim::Delete if self.endpoint_set.contains(entity) => Some(delete_under_edge(entity)),
            Claim::Endpoint if self.delete_set.contains(entity) => Some(edge_to_deleted(entity)),
            _ => None,
        }
    }

    /// The conflict of `ours`, the RDF writes of a transaction that commits
    /// now, with this transaction's, which committed after it began: read
    /// from the writes its commit's start took out (`rdf_writes`), never
    /// from its change set.
    #[cfg(feature = "triple-store")]
    fn rdf_conflict(&self, ours: &TransactionChanges) -> Option<Error> {
        self.rdf_writes
            .as_ref()
            .and_then(|theirs| ours.rdf_conflict_with(&theirs.writes))
    }
}

/// The RDF writes a committed transaction (or a load) keeps for the
/// conflict checks of the transactions that began before it: in the memory
/// report while kept (`TransactionManager::kept_rdf_writes_bytes`), and
/// counted out when dropped.
#[cfg(feature = "triple-store")]
#[derive(Debug)]
struct KeptRdfWrites {
    writes: Box<super::rdf::PendingTriples>,
    /// Their heap bytes, measured before they were kept.
    bytes: usize,
    /// The manager's count of every kept write set's bytes.
    counted: Arc<std::sync::atomic::AtomicUsize>,
    /// Their grant from the database's buffer manager, forced (the commit
    /// holds them already), given back when they drop.
    _grant: Option<grafeo_common::memory::buffer::MemoryGrant>,
}

#[cfg(feature = "triple-store")]
impl KeptRdfWrites {
    /// Keeps `writes`, of `bytes` heap bytes, counted in `counted` and
    /// granted by `memory`.
    fn new(
        writes: super::rdf::PendingTriples,
        bytes: usize,
        counted: &Arc<std::sync::atomic::AtomicUsize>,
        memory: Option<Arc<grafeo_common::memory::buffer::BufferManager>>,
    ) -> Self {
        counted.fetch_add(bytes, Ordering::Relaxed);
        Self {
            writes: Box::new(writes),
            bytes,
            counted: Arc::clone(counted),
            _grant: memory.map(|memory| {
                memory.reserve_forced(
                    bytes,
                    grafeo_common::memory::buffer::MemoryRegion::GraphStorage,
                )
            }),
        }
    }
}

#[cfg(feature = "triple-store")]
impl Drop for KeptRdfWrites {
    fn drop(&mut self) {
        self.counted.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

/// The conflict of two writes of `entity` (a delete is a write).
fn write_conflict(entity: &GraphEntity) -> String {
    format!("Write-write conflict on entity {entity}")
}

/// The conflict of a delete of `entity` with another transaction's edge to it.
fn delete_under_edge(entity: &GraphEntity) -> String {
    format!(
        "Write conflict on entity {entity}: another transaction creates an edge to the node this \
         transaction deletes"
    )
}

/// The conflict of an edge to `entity` with another transaction's delete of it.
fn edge_to_deleted(entity: &GraphEntity) -> String {
    format!(
        "Write conflict on entity {entity}: another transaction deletes the node this transaction \
         creates an edge to"
    )
}

/// The open transactions and the claim indexes conflict detection reads (under
/// `TransactionManager::transactions`).
#[derive(Default)]
struct Registry {
    /// The active and committing transactions: a transaction leaves when it is published or
    /// aborted.
    open: FxHashMap<TransactionId, TransactionInfo>,
    /// What the open transactions, and the commits they began before, claimed.
    claims: Claims,
}

/// The open transactions that hold one claim (see [`Claims`]): most claims have one holder,
/// kept inline.
#[derive(Debug)]
struct Holders {
    first: TransactionId,
    rest: Vec<TransactionId>,
}

impl Holders {
    /// The holders, in no order.
    fn iter(&self) -> impl Iterator<Item = TransactionId> + '_ {
        std::iter::once(self.first).chain(self.rest.iter().copied())
    }

    /// Adds `holder`, once.
    fn add(&mut self, holder: TransactionId) {
        if self.first != holder && !self.rest.contains(&holder) {
            self.rest.push(holder);
        }
    }

    /// Removes `holder`; whether none is left.
    fn remove(&mut self, holder: TransactionId) -> bool {
        if self.first == holder {
            match self.rest.pop() {
                Some(next) => self.first = next,
                None => return true,
            }
        } else {
            self.rest.retain(|held| *held != holder);
        }
        false
    }
}

/// Records that `holder` holds `claim` in `index`.
fn hold<K: Eq + std::hash::Hash>(
    index: &mut FxHashMap<K, Holders>,
    claim: K,
    holder: TransactionId,
) {
    index
        .entry(claim)
        .and_modify(|holders| holders.add(holder))
        .or_insert_with(|| Holders {
            first: holder,
            rest: Vec::new(),
        });
}

/// Records that `holder` no longer holds `claim` in `index`.
fn release<K: Eq + std::hash::Hash>(
    index: &mut FxHashMap<K, Holders>,
    claim: &K,
    holder: TransactionId,
) {
    if index
        .get_mut(claim)
        .is_some_and(|holders| holders.remove(holder))
    {
        index.remove(claim);
    }
}

/// The newest commit epoch of each claim kind on one entity (see [`Claims::committed`]).
#[derive(Debug, Clone, Copy, Default)]
struct CommittedClaims {
    /// The newest commit that wrote the entity (deletes included).
    write: u64,
    /// The newest commit that deleted it.
    delete: u64,
    /// The newest commit that created an edge to it.
    endpoint: u64,
}

/// One commit in the committed-claims index, with what it claimed: the trim drops it whole
/// once no open transaction began before it.
struct CommittedEntry {
    epoch: u64,
    /// The entities it wrote (deletes included).
    writes: FxHashSet<GraphEntity>,
    deletes: FxHashSet<GraphEntity>,
    endpoints: FxHashSet<GraphEntity>,
    keys: FxHashSet<KeyClaim>,
    patterns: PatternSet,
    /// Its RDF writes, in the memory report while kept.
    #[cfg(feature = "triple-store")]
    rdf: Option<KeptRdfWrites>,
}

impl CommittedEntry {
    /// The claims of `info`, a transaction published at `epoch`, moved out of it.
    fn of(epoch: u64, info: &mut TransactionInfo) -> Self {
        Self {
            epoch,
            writes: std::mem::take(&mut info.write_set),
            deletes: std::mem::take(&mut info.delete_set),
            endpoints: std::mem::take(&mut info.endpoint_set),
            keys: std::mem::take(&mut info.key_set),
            patterns: std::mem::take(&mut info.pattern_set),
            #[cfg(feature = "triple-store")]
            rdf: info.rdf_writes.take(),
        }
    }

    /// Whether it claims nothing.
    fn is_empty(&self) -> bool {
        let empty = self.writes.is_empty()
            && self.endpoints.is_empty()
            && self.keys.is_empty()
            && self.patterns.is_empty();
        #[cfg(feature = "triple-store")]
        let empty = empty && self.rdf.is_none();
        empty
    }
}

/// The claim indexes: what conflict detection looks up, once per claim, instead of walking
/// every open transaction and every commit since the oldest one began.
#[derive(Default)]
struct Claims {
    /// Whether the open index (`open`, `open_keys`, `open_patterns`) is kept: set by the begin
    /// that makes two transactions open at once, which indexes the one already open, and
    /// cleared when none is open. A transaction alone conflicts with nobody, so a lone writer
    /// (a bulk load, every auto-commit statement while nothing else is open) indexes nothing.
    indexed: bool,
    /// The open transactions that wrote, deleted or claimed as an edge's endpoint each entity.
    open: FxHashMap<GraphEntity, Holders>,
    /// The open transactions that claimed each key value.
    open_keys: FxHashMap<KeyClaim, Holders>,
    /// The open transactions that claimed each MERGE pattern.
    open_patterns: FxHashMap<u64, Holders>,
    /// The open transactions whose commit is in progress (its epoch assigned, not yet
    /// published): a commit checks against them as against a published one. Few: one per
    /// commit between its log write and its publish.
    committing: Vec<TransactionId>,
    /// The newest commit epoch of each claim kind per entity, over the commits in `log`.
    committed: FxHashMap<GraphEntity, CommittedClaims>,
    /// The newest commit epoch of each key value, over the commits in `log`.
    committed_keys: FxHashMap<KeyClaim, u64>,
    /// The newest commit epoch of each MERGE pattern, over the commits in `log`.
    committed_patterns: FxHashMap<u64, u64>,
    /// The triple writes of the commits in `log`.
    #[cfg(feature = "triple-store")]
    committed_triples: super::rdf::CommittedTriples,
    /// The manager's count of the heap bytes kept for RDF conflict checks, which the
    /// memory report reads: `committed_triples` counts its maps into it.
    #[cfg(feature = "triple-store")]
    kept_rdf_bytes: Arc<std::sync::atomic::AtomicUsize>,
    /// The commits published while another transaction was open, in epoch order, each with
    /// what it claimed: the trim pops them in order, so it is O(what it drops).
    log: VecDeque<CommittedEntry>,
}

impl Claims {
    /// Indexes every claim of `info`, open transaction `holder`'s.
    fn index(&mut self, holder: TransactionId, info: &TransactionInfo) {
        for entity in info.write_set.iter().chain(&info.endpoint_set) {
            hold(&mut self.open, entity.clone(), holder);
        }
        for key in &info.key_set {
            hold(&mut self.open_keys, key.clone(), holder);
        }
        for fingerprint in info.pattern_set.iter() {
            hold(&mut self.open_patterns, fingerprint, holder);
        }
    }

    /// Takes every claim of `info`, open transaction `holder`'s, out of the open index.
    fn unindex(&mut self, holder: TransactionId, info: &TransactionInfo) {
        for entity in info.write_set.iter().chain(&info.endpoint_set) {
            release(&mut self.open, entity, holder);
        }
        for key in &info.key_set {
            release(&mut self.open_keys, key, holder);
        }
        for fingerprint in info.pattern_set.iter() {
            release(&mut self.open_patterns, &fingerprint, holder);
        }
    }

    /// Indexes `entity` for `holder` as its claim sets in `info` now hold it.
    fn sync_entity(&mut self, holder: TransactionId, info: &TransactionInfo, entity: GraphEntity) {
        if info.write_set.contains(&entity) || info.endpoint_set.contains(&entity) {
            hold(&mut self.open, entity, holder);
        } else {
            release(&mut self.open, &entity, holder);
        }
    }

    /// Adds `entry`, a commit published after every commit in the log, to the committed index.
    fn add_committed(&mut self, entry: CommittedEntry) {
        if entry.is_empty() {
            return;
        }
        let epoch = entry.epoch;
        debug_assert!(
            self.log.back().is_none_or(|last| last.epoch < epoch),
            "commits are published in epoch order"
        );
        // The trim drops an entity by its write epoch: every deleted entity is written too.
        debug_assert!(
            entry.deletes.is_subset(&entry.writes),
            "a commit's deletes are among its writes"
        );
        for entity in &entry.writes {
            self.committed.entry(entity.clone()).or_default().write = epoch;
        }
        for entity in &entry.deletes {
            self.committed.entry(entity.clone()).or_default().delete = epoch;
        }
        for entity in &entry.endpoints {
            self.committed.entry(entity.clone()).or_default().endpoint = epoch;
        }
        for key in &entry.keys {
            self.committed_keys.insert(key.clone(), epoch);
        }
        for fingerprint in entry.patterns.iter() {
            self.committed_patterns.insert(fingerprint, epoch);
        }
        #[cfg(feature = "triple-store")]
        if let Some(rdf) = &entry.rdf {
            self.committed_triples.add(&rdf.writes, epoch);
            self.committed_triples.recount(&self.kept_rdf_bytes);
        }
        self.log.push_back(entry);
    }

    /// Whether a trim to `horizon` would drop anything (see [`trim`](Self::trim)).
    #[cfg(feature = "triple-store")]
    fn trim_due(&self, horizon: Option<u64>) -> bool {
        self.log
            .front()
            .is_some_and(|oldest| horizon.is_none_or(|horizon| oldest.epoch <= horizon))
    }

    /// Drops the commits at or before `horizon`, which no open transaction that can still
    /// commit began before, and with them each claim no later commit in the log made; with no
    /// horizon (none open that can commit) all of them. With `any_open` false, the open index
    /// is dropped too. Returns the dropped commits, for the caller to drop after its locks.
    fn trim(&mut self, horizon: Option<u64>, any_open: bool) -> Vec<CommittedEntry> {
        if !any_open && self.indexed {
            self.indexed = false;
            self.open = FxHashMap::default();
            self.open_keys = FxHashMap::default();
            self.open_patterns = FxHashMap::default();
            self.committing.clear();
        }
        let Some(horizon) = horizon else {
            if self.log.is_empty() {
                return Vec::new();
            }
            // New maps: a reader that held many commits gives their memory back.
            self.committed = FxHashMap::default();
            self.committed_keys = FxHashMap::default();
            self.committed_patterns = FxHashMap::default();
            #[cfg(feature = "triple-store")]
            {
                self.committed_triples.clear();
                self.committed_triples.recount(&self.kept_rdf_bytes);
            }
            return self.log.drain(..).collect();
        };
        let mut dropped = Vec::new();
        while self
            .log
            .front()
            .is_some_and(|oldest| oldest.epoch <= horizon)
        {
            let Some(entry) = self.log.pop_front() else {
                break;
            };
            for entity in entry.writes.iter().chain(&entry.endpoints) {
                if self
                    .committed
                    .get(entity)
                    .is_some_and(|newest| newest.write.max(newest.endpoint) <= horizon)
                {
                    self.committed.remove(entity);
                }
            }
            for key in &entry.keys {
                if self
                    .committed_keys
                    .get(key)
                    .is_some_and(|newest| *newest <= horizon)
                {
                    self.committed_keys.remove(key);
                }
            }
            for fingerprint in entry.patterns.iter() {
                if self
                    .committed_patterns
                    .get(&fingerprint)
                    .is_some_and(|newest| *newest <= horizon)
                {
                    self.committed_patterns.remove(&fingerprint);
                }
            }
            #[cfg(feature = "triple-store")]
            if let Some(rdf) = &entry.rdf {
                self.committed_triples.remove(&rdf.writes, horizon);
                self.committed_triples.recount(&self.kept_rdf_bytes);
            }
            dropped.push(entry);
        }
        dropped
    }

    /// The conflict of `ours`, open transaction `transaction_id`'s, which began at `start`,
    /// with the commits after `start`: those in the committed index and those still
    /// committing (`open` holds them). One lookup per claim of `ours`, plus a probe of each
    /// commit in progress. The checks and their errors as before the indexes: writes, then
    /// edges and deletes, keys, MERGE patterns, RDF triples, and serializable reads.
    fn commit_conflict(
        &self,
        open: &FxHashMap<TransactionId, TransactionInfo>,
        transaction_id: TransactionId,
        ours: &TransactionInfo,
        start: u64,
    ) -> Option<Error> {
        let conflict =
            |message: String| Some(Error::Transaction(TransactionError::WriteConflict(message)));
        let committing = || {
            self.committing
                .iter()
                .filter(move |other| **other != transaction_id)
                .filter_map(|other| open.get(other))
                .filter(|other| {
                    other
                        .commit_epoch
                        .is_some_and(|epoch| epoch.as_u64() > start)
                })
        };
        let newest = |entity: &GraphEntity| self.committed.get(entity).copied().unwrap_or_default();

        if let Some(entity) = ours.write_set.iter().find(|entity| {
            newest(entity).write > start
                || committing().any(|other| other.write_set.contains(*entity))
        }) {
            return conflict(write_conflict(entity));
        }

        // A commit after our start deleted a node we link to, or linked to a node we delete
        // (neither saw the other's change, so the second commit would leave a dangling edge).
        if let Some(entity) = ours.endpoint_set.iter().find(|entity| {
            newest(entity).delete > start
                || committing().any(|other| other.delete_set.contains(*entity))
        }) {
            return conflict(edge_to_deleted(entity));
        }
        if let Some(entity) = ours.delete_set.iter().find(|entity| {
            newest(entity).endpoint > start
                || committing().any(|other| other.endpoint_set.contains(*entity))
        }) {
            return conflict(delete_under_edge(entity));
        }

        // Unique key values: neither constraint check could see the other's.
        if let Some(key) = ours.key_set.iter().find(|key| {
            self.committed_keys
                .get(*key)
                .is_some_and(|epoch| *epoch > start)
                || committing().any(|other| other.key_set.contains(*key))
        }) {
            return conflict(format!(
                "on key {key}: a transaction that committed after this one began wrote the same \
                 value of the key"
            ));
        }

        // MERGE patterns: only fingerprints are kept, so the message cannot name the pattern.
        if ours.pattern_set.iter().any(|fingerprint| {
            self.committed_patterns
                .get(&fingerprint)
                .is_some_and(|epoch| *epoch > start)
                || committing().any(|other| other.pattern_set.contains(fingerprint))
        }) {
            return conflict(
                "a transaction that committed after this one began created a pattern a MERGE of \
                 this transaction creates; retry the transaction"
                    .to_string(),
            );
        }

        // RDF triples written by both where one of the two deletes it. A commit in progress
        // keeps its writes from the start of its commit (`rdf_writes`).
        #[cfg(feature = "triple-store")]
        if let Some(our_changes) = &ours.changes
            && our_changes.has_pending_triples()
        {
            if let Some(error) =
                our_changes.rdf_conflict_with_committed(&self.committed_triples, start)
            {
                return Some(error);
            }
            if let Some(error) = committing().find_map(|other| other.rdf_conflict(our_changes)) {
                return Some(error);
            }
        }

        // Serializable: we read an entity a commit after our start wrote (an rw-antidependency,
        // which can cause write skew).
        if ours.isolation_level == IsolationLevel::Serializable
            && let Some(entity) = ours.read_set.iter().find(|entity| {
                newest(entity).write > start
                    || committing().any(|other| other.write_set.contains(*entity))
            })
        {
            return Some(Error::Transaction(TransactionError::SerializationFailure(
                format!(
                    "Read-write conflict on entity {entity}: another transaction modified data \
                     we read"
                ),
            )));
        }
        None
    }

    /// The sizes the tests read: the open index's entries (entities, keys and patterns), the
    /// committed index's (entities, keys, patterns and triples), and the log's.
    #[cfg(test)]
    fn sizes(&self) -> (usize, usize, usize) {
        let committed =
            self.committed.len() + self.committed_keys.len() + self.committed_patterns.len();
        #[cfg(feature = "triple-store")]
        let committed = committed + self.committed_triples.len();
        (
            self.open.len() + self.open_keys.len() + self.open_patterns.len(),
            committed,
            self.log.len(),
        )
    }
}

/// How a transaction claims an entity for conflict detection
/// (first-writer-wins between open transactions, and at commit against the
/// transactions that committed after it began).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Claim {
    /// A write: conflicts with another transaction's writes (deletes
    /// included).
    Write,
    /// The delete of a node: a write, which also conflicts with another
    /// transaction's claim on the node as an edge's endpoint.
    Delete,
    /// The endpoint of an edge the transaction creates: conflicts only with
    /// another transaction's delete of the node.
    Endpoint,
}

/// Manages transactions and MVCC versioning.
///
/// # A commit that does not complete
///
/// A commit stamps its versions with its epoch before the epoch is published
/// (see `CommitGuard`). When the commit code panics in between, the stamped
/// versions stay invisible only as long as no higher epoch is published: a
/// later commit would publish its own epoch, and with it the stamped part of
/// the failed commit. So the first commit that does not complete poisons the
/// manager: from then on every commit fails with an error saying a commit did
/// not complete and the database must be reopened, and so does everything
/// that would persist or copy the store, which holds the stamped versions
/// (checkpoints, saves, backups and copies of the database call
/// [`check_no_incomplete_commit`](Self::check_no_incomplete_commit)).
/// Transactions still begin, and reads see every epoch published before the
/// failed commit.
pub struct TransactionManager {
    /// Next transaction ID.
    next_transaction_id: AtomicU64,
    /// The last epoch handed out: a commit takes the next one in
    /// [`start_commit`](Self::start_commit).
    assigned_epoch: AtomicU64,
    /// The last epoch whose commit is complete, the one readers see
    /// ([`current_epoch`](Self::current_epoch)). It trails `assigned_epoch`
    /// while a commit is being written, so no snapshot holds part of one.
    published_epoch: AtomicU64,
    /// Called once, after the first commit published while it is set: what
    /// starts a database's interval checkpoints when no WAL group can.
    #[cfg(feature = "grafeo-file")]
    first_commit_hook: parking_lot::Mutex<Option<FirstCommitHook>>,
    /// Whether a first-commit hook is set: one relaxed load per commit.
    #[cfg(feature = "grafeo-file")]
    first_commit_armed: AtomicBool,
    /// The committed changes since the open: one per commit with changes
    /// (a read-only commit, or one that matched nothing, counts none) and
    /// one per standalone change. Bumped in the publish turn.
    #[cfg(feature = "grafeo-file")]
    change_count: AtomicU64,
    /// The `change_count` the last successful checkpoint covered: with
    /// `change_count` above it, a database without a WAL has changes its
    /// file lacks.
    #[cfg(feature = "grafeo-file")]
    checkpointed_count: AtomicU64,
    /// Number of currently active transactions (for fast-path conflict skip).
    active_count: AtomicU64,
    /// The open transactions and the claim indexes (see [`Registry`]).
    transactions: RwLock<Registry>,
    /// Held by a direct call that commits at once while no transaction is
    /// open (see [`idle_gate`](Self::idle_gate)), and briefly by every
    /// [`begin`](Self::begin) and [`begin_private`](Self::begin_private), so
    /// no transaction starts while such a call runs.
    ///
    /// Lock order: the idle gate before the commit lock (such a call takes
    /// both, in that order; `begin` takes only the idle gate).
    idle_gate: Mutex<()>,
    /// Held shared by each direct read of what a snapshot restore replaces,
    /// for its call, and exclusive by the restore through its refill (see
    /// [`hold_direct_read`](Self::hold_direct_read)).
    replace_gate: RwLock<()>,
    /// Held by a commit from its epoch until it is complete (see
    /// [`CommitGuard`]), by a change outside a commit (a schema statement, a
    /// graph command, a direct call that commits at once) while it runs, by
    /// a checkpoint or a copy of the store
    /// while it builds its image (see [`hold_commits`](Self::hold_commits)),
    /// and by [`close_for_writes`](Self::close_for_writes): commits complete
    /// one at a time, in epoch order, and no image holds part of one. A
    /// [`begin`](Self::begin) does not take it: its snapshot is the published
    /// epoch, which holds only complete commits.
    ///
    /// Lock order: a checkpoint takes the file's checkpoint guard before this
    /// lock, and the idle gate comes before it too; the write freeze comes
    /// after it. Nothing that holds it waits for
    /// the checkpoint timer thread (`close()` stops the timer before it takes
    /// it).
    commit_lock: Mutex<()>,
    /// The write freeze: held shared by every store change of an open
    /// transaction while it runs (a write, see
    /// [`write_in_progress`](Self::write_in_progress), and the undo of a
    /// rollback or a rollback to a savepoint), and exclusively by a
    /// checkpoint or a copy of the store while it builds its image (see
    /// [`hold_commits`](Self::hold_commits)): they read a store and change
    /// logs that do not move. Changes outside
    /// any commit ([`hold_commits_for_change`](Self::hold_commits_for_change))
    /// do not take it: they hold the commit lock, which already keeps every
    /// checkpoint out.
    ///
    /// Lock order: the commit lock before this one, so a checkpoint that
    /// holds the commit lock waits here only for store changes in progress,
    /// which never wait for the commit lock, the checkpoint guard or anything
    /// else a checkpoint holds. It is not reentrant (a fair lock): no thread
    /// asks for it while it holds it, in either mode, because a checkpoint
    /// waiting in between would block the second request.
    write_freeze: RwLock<()>,
    /// Set when a commit did not complete (its [`CommitGuard`] was dropped
    /// without `complete`): no commit can run afterwards.
    poisoned: AtomicBool,
    /// Why commits are refused until a reopen while reads, checkpoints and
    /// copies go on (see [`refuse_commits`](Self::refuse_commits)).
    commit_refusal: Mutex<Option<CommitRefusal>>,
    /// Set by [`close_for_writes`](Self::close_for_writes) when a persistent
    /// database closes: no commit and no write outside a transaction can run
    /// afterwards.
    closed: AtomicBool,
    /// The lowest epoch that is never published: the epoch of a commit that
    /// did not complete (nothing at or above it is published, see
    /// [`Publish`]), 0 after a broken invariant ([`poison`](Self::poison)),
    /// `u64::MAX` otherwise.
    unpublishable_from: AtomicU64,
    /// Whose turn it is to publish: held while a commit publishes its epoch,
    /// and while a waiter checks the published epoch, so no wake-up is lost.
    ///
    /// Lock order: after the commit lock (a hold waits for the written
    /// commits while it holds the commit lock), before the `transactions`
    /// lock.
    publish_turn: Mutex<()>,
    /// Signalled when an epoch is published, and when a commit does not
    /// complete.
    published: Condvar,
    /// The readers that hold the read horizon back, by the epoch each reads
    /// at: the snapshot of every open and committing transaction, and every
    /// read outside a transaction that runs now (see
    /// [`pin_read`](Self::pin_read)). The read horizon is its oldest epoch,
    /// or the published epoch when it holds none, so it never walks the
    /// finished transactions. It also counts which snapshots belong to
    /// transactions that can still commit: the oldest such epoch is the
    /// horizon the committed-claims index trims to.
    ///
    /// Lock order: after the `transactions` lock (a transaction's snapshot
    /// is counted and released while that lock is held); the read horizon,
    /// a pin and its drop take this one alone, and the collect horizon takes
    /// the history's locks after it.
    reads: Arc<ReadPins>,
    /// The history: its retention, the commit times and the floor of what a
    /// read may ask for (see `history`).
    history: History,
    /// Change sets of ended transactions, emptied, that the next ones take
    /// with their buffers (see [`recycle`](Self::recycle)): an autocommit
    /// write allocates no new set. At most [`SPARE_CHANGE_SETS`].
    spare_changes: Mutex<Vec<Arc<TransactionChanges>>>,
    /// The buffer manager the change sets take their grants from, once the
    /// database attaches it. Weak: the manager holds consumers that hold this
    /// manager (the vector consumer's collect horizon), and a strong handle
    /// would keep both, and the spill directory, past the database.
    memory: std::sync::OnceLock<std::sync::Weak<grafeo_common::memory::buffer::BufferManager>>,
    /// The heap bytes of the RDF writes committed transactions keep for the
    /// conflict checks of older ones (see `KeptRdfWrites`).
    #[cfg(feature = "triple-store")]
    kept_rdf_bytes: Arc<std::sync::atomic::AtomicUsize>,
    /// How many bytes of committed state the commits since the oldest open
    /// transaction's snapshot may replace before that transaction expires
    /// (see [`set_snapshot_retention_limit`](Self::set_snapshot_retention_limit));
    /// `u64::MAX` until the database sets it: no bound.
    snapshot_retention_limit: AtomicU64,
}

/// The most emptied change sets the manager keeps for new transactions.
const SPARE_CHANGE_SETS: usize = 16;

/// The epochs the readers that hold the read horizon back read at, each with
/// how many such readers: open and committing transactions by their
/// snapshot, reads outside a transaction by the epoch they pinned.
///
/// Ascending, each epoch once with a count above 0: every reader is counted
/// at the published epoch, read under the lock, which only moves forward, so
/// a new epoch goes at the back. The oldest is the front, and the buffer is
/// kept from one reader to the next (a reader that begins and ends allocates
/// nothing).
#[derive(Debug, Default)]
struct ReadPins {
    epochs: Mutex<Readers>,
    /// What runs once a reader ends, its pin or its transaction (see
    /// [`TransactionManager::on_reader_end`]), with no lock held.
    reader_end: std::sync::OnceLock<ReaderEnd>,
    /// Woken when a change that replaces the stores ends (see
    /// [`TransactionManager::hold_idle`]): the reads that waited for it pin.
    replaced_whole: Condvar,
}

/// The readers by epoch (see [`ReadPins`]), and the ledger the snapshot
/// retention limit reads: the bytes of committed state every published
/// commit replaced, the labeled property graph's and the RDF overlay's apart
/// (a history window keeps the first anyway; the overlay keeps the second for
/// the readers alone).
#[derive(Debug, Default)]
struct Readers {
    pinned: VecDeque<PinnedEpoch>,
    replaced: Replaced,
    /// The newest oldest epoch whose readers the bound expired or warned
    /// about: no transaction begins at it again, so it is not scanned again.
    handled_through: Option<u64>,
    /// The oldest epoch with a snapshot of a transaction that can still
    /// commit (`PinnedEpoch::committable` above 0): the horizon the
    /// committed-claims index trims to; `None` with none.
    committable_from: Option<u64>,
    /// The pinned reads outside a transaction (each [`ReadPin`]), which a
    /// change that replaces the stores whole waits for no one of.
    reads: usize,
    /// A change that replaces the stores whole is in progress (see
    /// [`TransactionManager::hold_idle`]): a read waits to pin until it ends.
    replacing: bool,
}

#[cfg(test)]
impl Readers {
    /// The first epoch with a snapshot of a transaction that can still commit, by a walk of
    /// the ledger: the oracle `committable_from` is checked against.
    fn first_committable(readers: &Self) -> Option<u64> {
        readers
            .pinned
            .iter()
            .find(|pinned| pinned.committable > 0)
            .map(|pinned| pinned.epoch)
    }
}

/// Bytes of committed state replaced, in total since the manager began.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Replaced {
    lpg: u64,
    rdf: u64,
}

/// The readers at one epoch, and the ledger when the first was counted.
#[derive(Debug)]
struct PinnedEpoch {
    epoch: u64,
    readers: usize,
    /// How many of `readers` are snapshots of transactions that can still
    /// commit: not ended, not expired, not doomed by a lost claim. Pins
    /// outside a transaction never count here.
    committable: usize,
    replaced_at: Replaced,
}

/// The work a reader's end may let go (see `ReadPins::reader_end`).
struct ReaderEnd(Box<dyn Fn() + Send + Sync>);

impl std::fmt::Debug for ReaderEnd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ReaderEnd")
    }
}

impl ReadPins {
    /// Runs the reader-end work, if any is set: called with no lock held.
    fn reader_ended(&self) {
        if let Some(ReaderEnd(work)) = self.reader_end.get() {
            work();
        }
    }

    /// Counts a reader at the published epoch, read while the lock is held,
    /// and returns that epoch: the read horizon reads the published epoch
    /// under the same lock, so it never passes a reader it has not counted.
    /// A `committable` reader (a transaction's snapshot) is counted as one
    /// that can still commit too.
    fn hold_published(&self, published: &AtomicU64, committable: bool) -> EpochId {
        let mut readers = self.epochs.lock();
        // A transaction begins behind the idle gate the replacing change holds.
        if !committable {
            self.wait_for_replacement(&mut readers);
            readers.reads += 1;
        }
        let epoch = published.load(Ordering::Acquire);
        let replaced = readers.replaced;
        match readers.pinned.back_mut() {
            Some(newest) if newest.epoch == epoch => newest.readers += 1,
            _ => {
                debug_assert!(
                    readers
                        .pinned
                        .back()
                        .is_none_or(|newest| newest.epoch < epoch),
                    "the published epoch only moves forward"
                );
                readers.pinned.push_back(PinnedEpoch {
                    epoch,
                    readers: 1,
                    committable: 0,
                    replaced_at: replaced,
                });
            }
        }
        if committable {
            let newest = readers.pinned.len() - 1;
            Self::count_committable(&mut readers, newest);
        }
        EpochId::new(epoch)
    }

    /// Counts the reader at `pinned[index]`, counted already, as a snapshot of
    /// a transaction that can still commit, with the lock held. The horizon
    /// becomes the older of the two epochs: a begin is at the published epoch,
    /// at or after every counted one, and the minimum keeps the horizon right
    /// for any epoch.
    fn count_committable(readers: &mut Readers, index: usize) {
        let pinned = &mut readers.pinned[index];
        pinned.committable += 1;
        let epoch = pinned.epoch;
        readers.committable_from = Some(
            readers
                .committable_from
                .map_or(epoch, |from| from.min(epoch)),
        );
    }

    /// Counts a reader at `epoch`, at or before the published epoch, in its
    /// place among the readers (ascending).
    /// A reader counted below the newest epoch takes the ledger of the epoch after it, an
    /// undercount only for pins that never expire (time-travel reads).
    fn hold_at(readers: &mut Readers, epoch: u64) {
        match readers
            .pinned
            .binary_search_by_key(&epoch, |pinned| pinned.epoch)
        {
            Ok(index) => readers.pinned[index].readers += 1,
            Err(index) => {
                let replaced_at = readers
                    .pinned
                    .get(index)
                    .map_or(readers.replaced, |next| next.replaced_at);
                readers.pinned.insert(
                    index,
                    PinnedEpoch {
                        epoch,
                        readers: 1,
                        committable: 0,
                        replaced_at,
                    },
                );
            }
        }
    }

    /// Waits, with the lock held, while a change that replaces the stores
    /// whole is in progress: the read then pins the replaced stores.
    fn wait_for_replacement(&self, readers: &mut MutexGuard<'_, Readers>) {
        while readers.replacing {
            self.replaced_whole.wait(readers);
        }
    }

    /// Releases one reader at `epoch`.
    fn release(&self, epoch: EpochId) {
        Self::release_in(&mut self.epochs.lock(), epoch);
    }

    /// Releases the pinned read at `epoch` (a [`ReadPin`]).
    fn release_read(&self, epoch: EpochId) {
        let mut readers = self.epochs.lock();
        Self::release_in(&mut readers, epoch);
        readers.reads -= 1;
    }

    /// Releases one reader at `epoch`, with the lock held.
    fn release_in(readers: &mut Readers, epoch: EpochId) {
        if let Ok(index) = readers
            .pinned
            .binary_search_by_key(&epoch.as_u64(), |pinned| pinned.epoch)
        {
            Self::release_at(readers, index);
        }
    }

    /// Releases one reader of `pinned[index]`, with the lock held.
    fn release_at(readers: &mut Readers, index: usize) {
        let pinned = &mut readers.pinned[index];
        pinned.readers -= 1;
        if pinned.readers == 0 {
            debug_assert_eq!(
                pinned.committable, 0,
                "a snapshot stops counting as committable before its reader is released"
            );
            readers.pinned.remove(index);
        }
    }

    /// A transaction's snapshot at `epoch` ends (the transaction published or
    /// aborted), with the lock held: one search of the ledger serves both its
    /// committable count, when it still counted (`committable`), and its
    /// reader, when not released already (`release`).
    fn end_snapshot(readers: &mut Readers, epoch: u64, committable: bool, release: bool) {
        let found = readers
            .pinned
            .binary_search_by_key(&epoch, |pinned| pinned.epoch);
        debug_assert!(found.is_ok(), "an ending snapshot is a counted reader");
        let Ok(index) = found else {
            return;
        };
        if committable {
            Self::uncount_at(readers, index);
        }
        if release {
            Self::release_at(readers, index);
        }
    }

    /// One transaction whose snapshot is at `epoch` can commit no more (it
    /// expired or was doomed), with the lock held (see [`uncount_at`](Self::uncount_at)).
    fn uncount_committable(readers: &mut Readers, epoch: u64) {
        let found = readers
            .pinned
            .binary_search_by_key(&epoch, |pinned| pinned.epoch);
        debug_assert!(found.is_ok(), "a committable snapshot is a counted reader");
        if let Ok(index) = found {
            Self::uncount_at(readers, index);
        }
    }

    /// One transaction whose snapshot is at `pinned[index]` can commit no more,
    /// with the lock held. When it was the last such snapshot at the oldest
    /// committable epoch, the horizon moves to the next epoch with one,
    /// walking from that epoch forward: the epochs before it hold none, and
    /// each epoch is passed once, as the horizon only moves forward.
    fn uncount_at(readers: &mut Readers, index: usize) {
        let pinned = &mut readers.pinned[index];
        pinned.committable -= 1;
        if pinned.committable == 0 && readers.committable_from == Some(pinned.epoch) {
            readers.committable_from = readers
                .pinned
                .iter()
                .skip(index + 1)
                .find(|pinned| pinned.committable > 0)
                .map(|pinned| pinned.epoch);
        }
    }

    /// The oldest epoch a reader reads at, `None` with no reader.
    fn oldest(readers: &Readers) -> Option<u64> {
        readers.pinned.front().map(|pinned| pinned.epoch)
    }
}

/// A reader that cannot expire (a pin outside a transaction, a private or statement
/// transaction) holding replaced data past the snapshot retention limit: reported once per
/// epoch by the commit that crossed it, on its statement's result and to `tracing`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HeldPastLimit {
    /// The oldest epoch, which only such readers hold.
    pub(crate) epoch: u64,
    /// The bytes the commits since it replaced.
    pub(crate) kept: u64,
    /// The snapshot retention limit.
    pub(crate) limit: u64,
}

impl HeldPastLimit {
    /// The notice on the result of the statement whose commit crossed the limit.
    pub(crate) fn query_warning(self) -> grafeo_common::utils::warning::QueryWarning {
        grafeo_common::utils::warning::QueryWarning::with_message(
            grafeo_common::utils::warning::Warning::SnapshotHeldPastLimit,
            self.message(),
        )
    }

    /// The notice's text.
    pub(crate) fn message(self) -> String {
        let Self { epoch, kept, limit } = self;
        format!(
            "a snapshot, view, stream outside a transaction, iterator or export holds epoch              {epoch}, keeping {kept} bytes of replaced data past the snapshot retention limit of              {limit} bytes; drop it when done"
        )
    }
}

/// A transaction's snapshot as a reader of the read horizon: counted from the transaction's
/// start until it is released, once, when the transaction ends or, once it expired (the
/// snapshot retention limit, see [`TransactionManager::set_snapshot_retention_limit`]), when
/// no read of it is in progress.
#[derive(Debug)]
pub(crate) struct SnapshotReader {
    pins: Arc<ReadPins>,
    epoch: EpochId,
    /// The reads in progress (low bits), [`EXPIRED`] and [`RELEASED`].
    state: AtomicU64,
    /// The bytes replaced since the snapshot when it expired, for its error.
    kept: AtomicU64,
    /// The limit when it expired.
    limit: AtomicU64,
    /// False for a private transaction (one direct call): it never expires.
    expirable: bool,
    /// Whether it counts as a snapshot of a transaction that can still commit
    /// (`PinnedEpoch::committable`); changed only under the readers' lock.
    committable: AtomicBool,
}

/// The reader's count is released.
const RELEASED: u64 = 1 << 63;
/// The reader expired.
const EXPIRED: u64 = 1 << 62;
/// The reads in progress.
const READS: u64 = EXPIRED - 1;

impl SnapshotReader {
    /// A transaction's snapshot at `epoch`, counted in `pins` as a reader and
    /// as one that can still commit.
    fn new(pins: Arc<ReadPins>, epoch: EpochId, expirable: bool) -> Self {
        Self {
            pins,
            epoch,
            state: AtomicU64::new(0),
            kept: AtomicU64::new(0),
            limit: AtomicU64::new(0),
            expirable,
            committable: AtomicBool::new(true),
        }
    }

    /// The transaction can commit no more (it ended, its snapshot expired, it
    /// lost a claim): its snapshot no longer holds the committed-claims index
    /// back. Once, with the readers' lock held.
    fn stop_committable(&self, readers: &mut Readers) {
        if self.committable.swap(false, Ordering::Relaxed) {
            ReadPins::uncount_committable(readers, self.epoch.as_u64());
        }
    }

    /// Starts a read at the snapshot, held while the returned read lives.
    ///
    /// # Errors
    ///
    /// [`TransactionError::SnapshotExpired`] once the snapshot expired.
    pub(crate) fn read(self: &Arc<Self>) -> Result<SnapshotRead> {
        let before = self.state.fetch_add(1, Ordering::AcqRel);
        if before & EXPIRED != 0 {
            self.end_read();
            return Err(self.expired_error());
        }
        Ok(SnapshotRead(Arc::clone(self)))
    }

    /// Whether the snapshot expired.
    pub(crate) fn is_expired(&self) -> bool {
        self.state.load(Ordering::Acquire) & EXPIRED != 0
    }

    /// The error of an expired snapshot, with what it kept and the limit then.
    pub(crate) fn expired_error(&self) -> Error {
        Error::Transaction(self.expired_transaction_error())
    }

    /// [`expired_error`](Self::expired_error) as the transaction error.
    fn expired_transaction_error(&self) -> TransactionError {
        TransactionError::SnapshotExpired(Box::new(ExpiredSnapshot {
            kept: self.kept.load(Ordering::Acquire),
            limit: self.limit.load(Ordering::Acquire),
        }))
    }

    /// A read ended: the last read of an expired snapshot releases its count (exactly one of
    /// it, the expiry and the transaction's end does: the compare-exchange from `EXPIRED`).
    fn end_read(&self) {
        let after = self.state.fetch_sub(1, Ordering::AcqRel) - 1;
        if after == EXPIRED
            && self
                .state
                .compare_exchange(
                    EXPIRED,
                    EXPIRED | RELEASED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
        {
            self.pins.release(self.epoch);
            self.pins.reader_ended();
        }
    }

    /// Expires the snapshot, with the readers' lock held: its count is released at once when
    /// no read is in progress (the caller runs the reader-end work after its locks), else by
    /// the last read's end.
    fn expire(&self, readers: &mut Readers, kept: u64, limit: u64) {
        self.stop_committable(readers);
        self.kept.store(kept, Ordering::Relaxed);
        self.limit.store(limit, Ordering::Relaxed);
        let before = self.state.fetch_or(EXPIRED, Ordering::AcqRel);
        if before & (EXPIRED | RELEASED) == 0
            && before & READS == 0
            && self
                .state
                .compare_exchange(
                    EXPIRED,
                    EXPIRED | RELEASED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
        {
            ReadPins::release_in(readers, self.epoch);
        }
    }

    /// The transaction ended (published or aborted): it can commit no more, and its count is
    /// released unless it is released already, both in one search of the ledger, with the
    /// readers' lock held.
    fn end_in(&self, readers: &mut Readers) {
        let committable = self.committable.swap(false, Ordering::Relaxed);
        let release = self.state.fetch_or(RELEASED, Ordering::AcqRel) & RELEASED == 0;
        if committable || release {
            ReadPins::end_snapshot(readers, self.epoch.as_u64(), committable, release);
        }
    }
}

/// A read in progress at a transaction's snapshot: while it lives, an expiry does not release
/// the snapshot, so what the read reads stays.
#[must_use = "the snapshot is held only while the read lives"]
#[derive(Debug)]
pub(crate) struct SnapshotRead(Arc<SnapshotReader>);

impl SnapshotRead {
    /// The epoch the snapshot reads at.
    pub(crate) fn epoch(&self) -> EpochId {
        self.0.epoch
    }
}

impl Drop for SnapshotRead {
    fn drop(&mut self) {
        self.0.end_read();
    }
}

/// What a read holds while it runs, released when it drops: a pin (a read outside a
/// transaction, a time-travel read), or a read at its transaction's snapshot.
#[derive(Debug)]
pub(crate) enum ReadHold {
    /// A pinned read.
    Pin {
        /// Held until the hold drops.
        _pin: ReadPin,
    },
    /// A read at the transaction's snapshot.
    Snapshot {
        /// Held until the hold drops.
        _read: SnapshotRead,
    },
}

/// A committed read, including a historical override inside a transaction,
/// pinned at its epoch until it is dropped (also by the unwind of a panic):
/// the read horizon stays at that epoch or before, so garbage collection
/// keeps every version the read may still need (every read statement
/// outside a transaction pins its epoch for its run).
#[must_use = "a read is pinned only while the pin lives"]
pub(crate) struct ReadPin {
    pins: Arc<ReadPins>,
    epoch: EpochId,
}

impl ReadPin {
    /// The epoch the read is pinned at: its selected historical epoch or
    /// the published epoch when it was pinned.
    #[must_use]
    pub(crate) fn epoch(&self) -> EpochId {
        self.epoch
    }

    /// Another pin at the same epoch, released on its own: this pin keeps
    /// the epoch at or after the history floor while the new one is made.
    pub(crate) fn again(&self) -> ReadPin {
        let mut readers = self.pins.epochs.lock();
        ReadPins::hold_at(&mut readers, self.epoch.as_u64());
        readers.reads += 1;
        ReadPin {
            pins: Arc::clone(&self.pins),
            epoch: self.epoch,
        }
    }
}

impl std::fmt::Debug for ReadPin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReadPin")
            .field("epoch", &self.epoch)
            .finish_non_exhaustive()
    }
}

impl Drop for ReadPin {
    fn drop(&mut self) {
        self.pins.release_read(self.epoch);
        self.pins.reader_ended();
    }
}

/// A direct read held for its call (see
/// [`TransactionManager::hold_direct_read`]).
pub(crate) struct DirectRead<'a> {
    _gate: RwLockReadGuard<'a, ()>,
}

#[cfg(debug_assertions)]
thread_local! {
    /// The direct reads this thread holds: at most one.
    static DIRECT_READ_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(debug_assertions)]
impl Drop for DirectRead<'_> {
    fn drop(&mut self) {
        DIRECT_READ_DEPTH.with(|depth| depth.set(depth.get() - 1));
    }
}

/// Every other piece of work held off for a change that replaces the stores
/// whole (see [`TransactionManager::hold_idle`]). Its drop lets the reads
/// that waited pin, and the transactions that waited begin.
#[must_use = "other work is held off only while the guard lives"]
pub(crate) struct IdleHold<'a> {
    reads: &'a ReadPins,
    _gate: MutexGuard<'a, ()>,
}

impl Drop for IdleHold<'_> {
    fn drop(&mut self) {
        self.reads.epochs.lock().replacing = false;
        self.reads.replaced_whole.notify_all();
    }
}

/// Refuses `doing` while `transactions` or `reads` are open (`GRAFEO-T004`),
/// with what is open: "cannot restore a snapshot while 2 transactions and 1
/// read are open: end them first". Both counts of a change that replaces the
/// stores whole go through it (see [`TransactionManager::hold_idle`]).
fn refuse_open_work(doing: &str, transactions: usize, reads: usize) -> Result<()> {
    if transactions == 0 && reads == 0 {
        return Ok(());
    }
    let count = |n: usize, what: &str| format!("{n} {what}{}", if n == 1 { "" } else { "s" });
    let open: Vec<String> = [(transactions, "transaction"), (reads, "read")]
        .into_iter()
        .filter(|&(n, _)| n > 0)
        .map(|(n, what)| count(n, what))
        .collect();
    let one = transactions + reads == 1;
    Err(Error::Transaction(TransactionError::InvalidState(format!(
        "cannot {doing} while {} {} open: end {} first",
        open.join(" and "),
        if one { "is" } else { "are" },
        if one { "it" } else { "them" }
    ))))
}

/// Commits held off (see [`TransactionManager::hold_commits`]): while this
/// lives, no commit is between its epoch and its completion, and no commit,
/// transaction start or write outside a transaction can begin. Held for a
/// checkpoint or a copy of the store, it also freezes the store: no store
/// change of an open transaction is in progress or can begin.
///
/// Held for a standalone change ([`changed`](Self::changed)), it counts one
/// committed change as it is released.
#[must_use = "commits are held off only while the guard lives"]
pub(crate) struct CommitsHeld<'a> {
    // Fields drop in order: the freeze is released before the commit lock,
    // the reverse of the order they are taken in.
    writes: Option<RwLockWriteGuard<'a, ()>>,
    commit: Option<MutexGuard<'a, ()>>,
    /// The manager that counts the change, for a hold of a standalone
    /// change.
    #[cfg(feature = "grafeo-file")]
    change: Option<&'a TransactionManager>,
}

#[cfg(feature = "grafeo-file")]
impl<'a> CommitsHeld<'a> {
    /// Counts this hold's change as committed when it is released: for a
    /// standalone change, which commits no transaction (a hold for a
    /// checkpoint, a copy or a backup changes nothing). A change that cannot
    /// tell whether it changed anything counts.
    pub(crate) fn changed(&mut self, manager: &'a TransactionManager) {
        self.change = Some(manager);
    }
}

/// The count of a standalone change: after the write freeze goes and before
/// the commit lock does, in the publish turn (the commit lock comes before
/// the turn, as for a commit that holds it), so a checkpoint sees the change
/// and its count or neither; the first-commit hook then runs with no lock
/// held.
impl Drop for CommitsHeld<'_> {
    fn drop(&mut self) {
        drop(self.writes.take());
        #[cfg(feature = "grafeo-file")]
        if let Some(manager) = self.change.take() {
            {
                let _turn = manager.publish_turn.lock();
                manager.change_count.fetch_add(1, Ordering::AcqRel);
            }
            drop(self.commit.take());
            manager.run_first_commit_hook();
        }
        drop(self.commit.take());
    }
}

/// A commit in progress, from [`TransactionManager::start_commit`] until it
/// is published. Until then the transaction is
/// [`TransactionState::Committing`]: it still counts as open, its writes
/// still conflict with other transactions' writes, and readers do not see
/// its epoch yet.
///
/// The guard holds the commit lock: no other commit takes an epoch, and no
/// [`begin`](TransactionManager::begin) runs. It ends in one of three ways:
///
/// - [`abort`](Self::abort): the commit's log write failed and nothing of it
///   remains. The epoch goes back, and the transaction is active again for
///   the caller's rollback.
/// - [`written`](Self::written): the commit's log group is written. The
///   commit lock is released, so the next commit starts, and the returned
///   [`Publish`] publishes this one in its turn.
/// - [`complete`](Self::complete): both at once, while the lock is held.
///
/// A guard dropped without any of them (only when the commit code panics)
/// does not report the transaction as committed and does not publish its
/// epoch: it stays `Committing`, so its half-written entities stay locked
/// against other writers, and only the commit lock is released. It also
/// poisons the manager: versions the commit already stamped with its epoch
/// would become visible with the epoch of any later commit, so every later
/// [`start_commit`](TransactionManager::start_commit) fails until the
/// database is reopened (see [`TransactionManager`]).
#[must_use = "a commit ends with `complete()`, `written()` or `abort()`"]
pub(crate) struct CommitGuard<'a> {
    // Fields drop in order: an unfinished commit poisons the manager before
    // the commit lock is released, so a commit waiting for the lock sees it.
    commit: Publish<'a>,
    _commit: MutexGuard<'a, ()>,
}

impl CommitGuard<'_> {
    /// The commit epoch.
    pub(crate) fn epoch(&self) -> EpochId {
        self.commit.epoch
    }

    /// When the commit happened, while the database keeps history (`None`
    /// otherwise): its WAL group carries this time, so a replay records the
    /// time the live commit recorded.
    #[cfg(feature = "wal")]
    pub(crate) fn commit_time(&self) -> Option<u64> {
        self.commit.commit_time
    }

    /// Publishes the commit while the commit lock is held: the transaction is
    /// committed, readers see its epoch, and what waited for it can run.
    ///
    /// # Errors
    ///
    /// As [`Publish::complete`].
    pub(crate) fn complete(self) -> Result<()> {
        let Self { commit, _commit } = self;
        commit.complete()
    }

    /// Takes the commit back before anything of it is written or stamped
    /// (its log write failed and was cut back): the epoch is free again,
    /// which is safe because the commit lock is still held and no later
    /// commit took one, and the transaction is active again, with its claims,
    /// for the caller to roll back. Nothing is published or poisoned.
    pub(crate) fn abort(self) {
        let Self {
            mut commit,
            _commit,
        } = self;
        let manager = commit.manager;
        {
            let mut registry = manager.transactions.write();
            if let Some(info) = registry.open.get_mut(&commit.transaction_id) {
                info.state = TransactionState::Active;
                info.commit_epoch = None;
            }
            registry
                .claims
                .committing
                .retain(|committing| *committing != commit.transaction_id);
            // The last epoch handed out is this commit's (the commit lock).
            manager
                .assigned_epoch
                .store(commit.epoch.as_u64() - 1, Ordering::SeqCst);
        }
        commit.finished = true;
    }
}

impl<'a> CommitGuard<'a> {
    /// Whether the commit has changes, the fact that decides whether the WAL
    /// records it: a commit without (a read, or a write that matched nothing)
    /// is no committed change for the checkpointer.
    #[cfg(feature = "grafeo-file")]
    pub(crate) fn set_changed(&mut self, changed: bool) {
        self.commit.changed = changed;
    }

    /// The commit's group is in the log: releases the commit lock, so the
    /// next commit can take its epoch and write, and returns the handle that
    /// publishes this one after every commit below it.
    pub(crate) fn written(self) -> Publish<'a> {
        let Self { commit, _commit } = self;
        commit
    }
}

/// A written commit that waits to be published (see
/// [`CommitGuard::written`]): its log group is written, the commit lock is
/// released, and its transaction is still [`TransactionState::Committing`].
///
/// Commits publish in the order of their epochs, so a reader that sees an
/// epoch sees every commit below it: [`complete`](Self::complete) waits for
/// the commit below this one.
///
/// A handle dropped without `complete` is a commit that did not complete, as
/// a dropped [`CommitGuard`] is: the manager is poisoned, and neither this
/// epoch nor any above it is published.
#[must_use = "a written commit is committed only after `complete()`"]
pub(crate) struct Publish<'a> {
    manager: &'a TransactionManager,
    transaction_id: TransactionId,
    epoch: EpochId,
    /// When the commit happened, while the database keeps history: recorded
    /// in the time index as it publishes, and the time of its WAL group.
    commit_time: Option<u64>,
    /// Whether the commit was published or taken back.
    finished: bool,
    /// Whether the commit has changes (what its WAL group records): only
    /// such a commit counts as a committed change. Set by
    /// [`CommitGuard::set_changed`]; a commit counts as one unless told.
    #[cfg(feature = "grafeo-file")]
    changed: bool,
}

/// A commit whose group is written, on its way to being published.
pub(crate) enum WrittenCommit<'a> {
    /// The commit lock went: the next commit writes while this one syncs.
    Released(Publish<'a>),
    /// The commit lock stays until it is published: nothing commits between a change of
    /// the catalog and its durability, or sees one half applied.
    Held(CommitGuard<'a>),
}

impl WrittenCommit<'_> {
    /// Publishes the commit as [`Publish::complete_replacing`] does.
    ///
    /// # Errors
    ///
    /// As [`Publish::complete_with`].
    pub(crate) fn complete_replacing(
        self,
        ordered: impl FnOnce() -> Result<u64>,
    ) -> Result<Option<HeldPastLimit>> {
        match self {
            Self::Released(publish) => publish.complete_replacing(ordered),
            Self::Held(CommitGuard { commit, _commit }) => commit.complete_replacing(ordered),
        }
    }
}

impl Publish<'_> {
    /// The commit epoch.
    #[cfg(test)]
    pub(crate) fn epoch(&self) -> EpochId {
        self.epoch
    }

    /// Publishes the commit once every commit below it is published: the
    /// transaction is committed, its claims are released, readers see its
    /// epoch, and what waited for it can run.
    ///
    /// # Errors
    ///
    /// As [`complete_with`](Self::complete_with).
    pub(crate) fn complete(self) -> Result<()> {
        self.complete_with(|| Ok(()))
    }

    /// Publishes the commit as [`complete`](Self::complete) does, once
    /// `ordered` ran in the commit's turn: after every commit below it is
    /// published and before any above it publishes. The commit's steps whose
    /// order must follow the epochs go there: stamping its versions, applying
    /// its RDF triples, reporting its change events, moving the store's epoch.
    /// `ordered` must not wait for anything that waits for a commit to be
    /// published (it holds the publish turn).
    ///
    /// # Errors
    ///
    /// [`TransactionError::IncompleteCommit`] when a commit at a lower epoch
    /// did not complete meanwhile (or the manager was poisoned): this one is
    /// then not published either (its epoch would show the other's stamped
    /// versions) and stays `Committing`, in the log as it was written, until
    /// the database is reopened. The error of `ordered`, which leaves the
    /// commit incomplete as a dropped handle does.
    pub(crate) fn complete_with(self, ordered: impl FnOnce() -> Result<()>) -> Result<()> {
        self.complete_replacing(|| ordered().map(|()| 0))
            .map(|_| ())
    }

    /// [`complete_with`](Self::complete_with) for a commit whose `ordered` step reports the
    /// bytes of RDF state it replaced (the overlay's growth for older readers), which the
    /// snapshot retention ledger counts beside the change set's own.
    ///
    /// # Errors
    ///
    /// As [`complete_with`](Self::complete_with).
    pub(crate) fn complete_replacing(
        mut self,
        ordered: impl FnOnce() -> Result<u64>,
    ) -> Result<Option<HeldPastLimit>> {
        let manager = self.manager;
        let mut turn = manager.publish_turn.lock();
        loop {
            if self.epoch.as_u64() >= manager.unpublishable_from.load(Ordering::Acquire) {
                // Reported here: the drop of this handle says nothing more.
                self.finished = true;
                return Err(Error::Transaction(TransactionError::IncompleteCommit));
            }
            if manager.published_epoch.load(Ordering::Acquire) + 1 >= self.epoch.as_u64() {
                break;
            }
            manager.published.wait(&mut turn);
        }
        let rdf_replaced = match ordered() {
            Ok(bytes) => bytes,
            Err(error) => {
                // Not published: the handle, dropped unfinished, poisons the
                // manager once the turn is free.
                drop(turn);
                return Err(error);
            }
        };
        let held_past_limit;
        // What the commit replaced, for the ledger and the history's time index.
        let mut lpg_replaced = 0;
        // The published transaction and the commits the trim dropped, dropped once the locks
        // are released (they may hold kept RDF writes and their grants).
        let (published, trimmed) = {
            let mut registry = manager.transactions.write();
            let Registry { open, claims } = &mut *registry;
            // Every transaction leaves at its publish, private or not, whatever else is open.
            let published = open.remove(&self.transaction_id).map(|mut info| {
                claims
                    .committing
                    .retain(|committing| *committing != self.transaction_id);
                if claims.indexed {
                    claims.unindex(self.transaction_id, &info);
                }
                // Stamped: what it changed is committed state now, and what
                // it replaced is kept for the readers older than this commit.
                if let Some(changes) = info.changes.take() {
                    lpg_replaced = changes.superseded_bytes() as u64;
                }
                // Its claims serve the commits of the transactions still open, which
                // all began before this publish (each is registered under this lock).
                if !open.is_empty() {
                    claims.add_committed(CommittedEntry::of(self.epoch.as_u64(), &mut info));
                }
                info
            });
            // Counted down under the `transactions` lock, like the count up
            // of a transaction that begins.
            manager.active_count.fetch_sub(1, Ordering::Release);
            // Under the readers' lock: a reader counted at the published
            // epoch sees the ledger that goes with it.
            let mut readers = manager.reads.epochs.lock();
            // It reads no more.
            if let Some(info) = &published {
                info.reader.end_in(&mut readers);
            }
            readers.replaced.lpg = readers.replaced.lpg.saturating_add(lpg_replaced);
            readers.replaced.rdf = readers.replaced.rdf.saturating_add(rdf_replaced);
            // Commits publish one at a time and in epoch order (the turn), so
            // the published epoch only moves forward.
            manager
                .published_epoch
                .fetch_max(self.epoch.as_u64(), Ordering::Release);
            held_past_limit = manager.bound_snapshots(open, &mut readers);
            // After the bound: a snapshot it expired needs no commit kept.
            (
                published,
                TransactionManager::trim_committed(open, claims, &readers),
            )
        };
        // In the turn: the hook's arming reads it there (see
        // `arm_first_commit_hook`).
        #[cfg(feature = "grafeo-file")]
        if self.changed {
            manager.change_count.fetch_add(1, Ordering::AcqRel);
        }
        // In the turn, so the times are recorded in epoch order, with what the commit replaced.
        if let Some(time) = self.commit_time {
            manager
                .history
                .record_replacing(self.epoch, time, lpg_replaced);
        }
        self.finished = true;
        drop(turn);
        manager.published.notify_all();
        #[cfg(feature = "grafeo-file")]
        if self.changed {
            manager.run_first_commit_hook();
        }
        // After every lock: the writes kept for nobody go, and what older
        // readers needed may go too.
        drop(published);
        drop(trimmed);
        manager.reads.reader_ended();
        if let Some(held) = held_past_limit {
            grafeo_common::grafeo_warn!("{}", held.message());
        }
        Ok(held_past_limit)
    }
}

/// What [`TransactionManager::set_first_commit_hook`] calls.
#[cfg(feature = "grafeo-file")]
pub(crate) type FirstCommitHook = std::sync::Arc<dyn Fn() + Send + Sync>;

/// The first-commit hook: only a database with a file has interval
/// checkpoints to start.
#[cfg(feature = "grafeo-file")]
impl TransactionManager {
    /// Arms `hook` for the next committed change, once, unless one came
    /// after the last checkpoint already: then it arms nothing and returns
    /// `true`, for the caller to do what the hook would. Taken in the publish
    /// turn, where every committed change is counted: a change committed
    /// meanwhile is either seen here or runs the hook. The hook runs with no
    /// lock of the commit held and must not wait for another commit.
    pub(crate) fn arm_first_commit_hook(&self, hook: FirstCommitHook) -> bool {
        let _turn = self.publish_turn.lock();
        if self.changed_since_checkpoint() {
            return true;
        }
        *self.first_commit_hook.lock() = Some(hook);
        self.first_commit_armed.store(true, Ordering::Release);
        false
    }

    /// Clears the first-commit hook.
    pub(crate) fn clear_first_commit_hook(&self) {
        *self.first_commit_hook.lock() = None;
        self.first_commit_armed.store(false, Ordering::Release);
    }

    /// Whether a change was committed after the last successful checkpoint
    /// (since the open, when none ran): what a database without a WAL has
    /// and its file lacks.
    pub(crate) fn changed_since_checkpoint(&self) -> bool {
        self.change_count.load(Ordering::Acquire) > self.checkpointed_count.load(Ordering::Acquire)
    }

    /// The committed changes counted so far: what a checkpoint taken now
    /// covers, read while it holds commits off.
    pub(crate) fn change_count(&self) -> u64 {
        self.change_count.load(Ordering::Acquire)
    }

    /// Records that a checkpoint covering `count` changes succeeded.
    pub(crate) fn checkpointed(&self, count: u64) {
        self.checkpointed_count.fetch_max(count, Ordering::AcqRel);
    }

    /// Runs and clears the first-commit hook, if one is set.
    fn run_first_commit_hook(&self) {
        if !self.first_commit_armed.load(Ordering::Relaxed)
            || !self.first_commit_armed.swap(false, Ordering::AcqRel)
        {
            return;
        }
        let hook = self.first_commit_hook.lock().take();
        if let Some(hook) = hook {
            hook();
        }
    }
}

impl Drop for Publish<'_> {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        grafeo_common::grafeo_error!(
            "commit of transaction {:?} at epoch {:?} did not complete; its writes stay \
             locked, and no transaction can commit until the database is reopened",
            self.transaction_id,
            self.epoch
        );
        self.manager.incomplete_from(self.epoch);
    }
}

impl TransactionManager {
    /// Creates a new transaction manager.
    #[must_use]
    pub fn new() -> Self {
        #[cfg(feature = "triple-store")]
        let kept_rdf_bytes: Arc<std::sync::atomic::AtomicUsize> = Arc::default();
        let registry = Registry {
            open: FxHashMap::default(),
            claims: Claims {
                #[cfg(feature = "triple-store")]
                kept_rdf_bytes: Arc::clone(&kept_rdf_bytes),
                ..Claims::default()
            },
        };
        Self {
            // Start at 2 to avoid collision with TransactionId::SYSTEM (which is 1)
            // TransactionId::INVALID = u64::MAX, TransactionId::SYSTEM = 1, user transactions start at 2
            next_transaction_id: AtomicU64::new(2),
            assigned_epoch: AtomicU64::new(0),
            published_epoch: AtomicU64::new(0),
            #[cfg(feature = "grafeo-file")]
            first_commit_hook: parking_lot::Mutex::new(None),
            #[cfg(feature = "grafeo-file")]
            first_commit_armed: AtomicBool::new(false),
            #[cfg(feature = "grafeo-file")]
            change_count: AtomicU64::new(0),
            #[cfg(feature = "grafeo-file")]
            checkpointed_count: AtomicU64::new(0),
            active_count: AtomicU64::new(0),
            transactions: RwLock::new(registry),
            idle_gate: Mutex::new(()),
            replace_gate: RwLock::new(()),
            commit_lock: Mutex::new(()),
            write_freeze: RwLock::new(()),
            poisoned: AtomicBool::new(false),
            commit_refusal: Mutex::new(None),
            closed: AtomicBool::new(false),
            unpublishable_from: AtomicU64::new(u64::MAX),
            publish_turn: Mutex::new(()),
            published: Condvar::new(),
            reads: Arc::new(ReadPins::default()),
            history: History::default(),
            spare_changes: Mutex::new(Vec::new()),
            memory: std::sync::OnceLock::new(),
            #[cfg(feature = "triple-store")]
            kept_rdf_bytes,
            snapshot_retention_limit: AtomicU64::new(u64::MAX),
        }
    }

    /// Sets the work that runs once a reader ends (a read pin drops, a
    /// transaction aborts or publishes), with no lock of the manager held:
    /// what was kept for older readers may go then. Set once; later calls
    /// keep the first.
    #[cfg(feature = "triple-store")]
    pub(crate) fn on_reader_end(&self, work: impl Fn() + Send + Sync + 'static) {
        let _ = self.reads.reader_end.set(ReaderEnd(Box::new(work)));
    }

    /// The heap bytes of the RDF writes kept for the conflict checks of
    /// transactions that began before their commits.
    #[cfg(feature = "triple-store")]
    #[must_use]
    pub(crate) fn kept_rdf_writes_bytes(&self) -> usize {
        self.kept_rdf_bytes.load(Ordering::Relaxed)
    }

    /// The count behind [`kept_rdf_writes_bytes`](Self::kept_rdf_writes_bytes),
    /// for a check at a reader's end that needs no handle on the manager.
    #[cfg(feature = "triple-store")]
    #[must_use]
    pub(crate) fn kept_rdf_writes_counter(&self) -> Arc<std::sync::atomic::AtomicUsize> {
        Arc::clone(&self.kept_rdf_bytes)
    }

    /// Lets go of the RDF writes no open transaction's commit is checked
    /// against any more: those of commits before every open transaction
    /// began (the trim of the committed-claims index, see `trim_committed`).
    /// One atomic load when none is kept; what goes is dropped after the
    /// lock.
    #[cfg(feature = "triple-store")]
    pub(crate) fn release_kept_rdf_writes(&self) {
        if self.kept_rdf_bytes.load(Ordering::Relaxed) == 0 {
            return;
        }
        // Checked under the read side first, so a reader's end while an
        // older transaction stays open takes no write lock.
        let due = {
            let registry = self.transactions.read();
            let readers = self.reads.epochs.lock();
            registry
                .claims
                .trim_due(Self::conflict_horizon_in(&readers))
        };
        if !due {
            return;
        }
        let dropped = {
            let mut registry = self.transactions.write();
            let readers = self.reads.epochs.lock();
            let Registry { open, claims } = &mut *registry;
            Self::trim_committed(open, claims, &readers)
        };
        drop(dropped);
    }

    /// The epoch the committed-claims index trims to (its conflict horizon): the oldest
    /// snapshot of a transaction that can still commit, as the readers' ledger counts them
    /// (`Readers::committable_from`); `None` with none. A reader that cannot commit (a pin
    /// outside a transaction, a doomed transaction, an expired snapshot whose read still runs)
    /// holds no commit in the index. O(1).
    fn conflict_horizon_in(readers: &Readers) -> Option<u64> {
        readers.committable_from
    }

    /// Trims the committed-claims index to the conflict horizon (see
    /// `conflict_horizon_in`, `Claims::trim`), with the `transactions` lock and the readers'
    /// lock held: at every publish and abort. Returns what it dropped, for the caller to drop
    /// after its locks.
    fn trim_committed(
        open: &FxHashMap<TransactionId, TransactionInfo>,
        claims: &mut Claims,
        readers: &Readers,
    ) -> Vec<CommittedEntry> {
        claims.trim(Self::conflict_horizon_in(readers), !open.is_empty())
    }

    /// Gives the change sets made from now on their grants from `manager`
    /// (see `TransactionChanges`). A manager keeps the first it gets.
    pub(crate) fn attach_memory(&self, manager: Arc<grafeo_common::memory::buffer::BufferManager>) {
        // A second attach changes nothing.
        let _ = self.memory.set(Arc::downgrade(&manager));
    }

    /// A manager whose commits take their times from `clock` (milliseconds
    /// since the Unix epoch) once history is on: for tests of the retention
    /// window.
    #[cfg(test)]
    pub(crate) fn with_history_clock(clock: super::history::Clock) -> Self {
        Self {
            history: History::with_clock(clock),
            ..Self::new()
        }
    }

    /// Begins a new transaction with the default isolation level (Snapshot Isolation).
    pub fn begin(&self) -> TransactionId {
        self.begin_with_isolation(IsolationLevel::default())
    }

    /// Begins a new transaction with the specified isolation level.
    pub fn begin_with_isolation(&self, isolation_level: IsolationLevel) -> TransactionId {
        // Wait for a direct call that commits at once (it needs no open
        // transaction), not for a commit in progress: the snapshot is the
        // published epoch, which holds every commit up to it, complete, and
        // none of one still being written or synced.
        let _gate = self.idle_gate.lock();
        self.register(isolation_level, false, true).0
    }

    /// Begins the transaction one statement runs in outside an explicit transaction (an
    /// auto-commit write): as [`begin`](Self::begin), except that its snapshot never expires
    /// under the snapshot retention limit, since a statement outside a transaction is bounded
    /// by its query timeout and must not fail for a limit it never asked for.
    pub(crate) fn begin_statement(&self) -> TransactionId {
        let _gate = self.idle_gate.lock();
        self.register(IsolationLevel::default(), false, false).0
    }

    /// Exclusive access for a direct call that commits at once: `Some` while
    /// no transaction is open. No transaction can begin until the guard is
    /// dropped, so such a call cannot conflict with one and needs no
    /// claims; nor with another such call, which waits for the guard.
    pub(crate) fn idle_gate(&self) -> Option<MutexGuard<'_, ()>> {
        let gate = self.idle_gate.lock();
        // Acquire pairs with the release of a commit's completion: the call
        // sees everything the last commit wrote.
        (self.active_count.load(Ordering::Acquire) == 0).then_some(gate)
    }

    /// Holds every other piece of work off for a change that replaces the
    /// stores whole (a snapshot restore), while the returned guard lives:
    /// no transaction can begin and no read can pin, and each waits until
    /// the guard drops, then sees the replaced stores whole. Taken before
    /// commits are held (lock order: the idle gate, the transactions, the
    /// readers, as in `begin`).
    ///
    /// # Errors
    ///
    /// [`TransactionError::InvalidState`] (`GRAFEO-T004`) while any other
    /// work is open: a transaction (read-only too) or a pinned read (a
    /// running query, a stream, an iterator, a snapshot or a view), the
    /// caller's own included, which `doing` names ("restore a snapshot").
    /// Nothing waits for it: the open work reads the stores it started on.
    pub(crate) fn hold_idle(&self, doing: &str) -> Result<IdleHold<'_>> {
        let gate = self.idle_gate.lock();
        let transactions = self.transactions.read().open.len();
        let mut readers = self.reads.epochs.lock();
        // An early exit: open work refuses before reads are held off.
        refuse_open_work(doing, transactions, readers.reads)?;
        readers.replacing = true;
        Ok(IdleHold {
            reads: &self.reads,
            _gate: gate,
        })
    }

    /// Counts the pinned reads again once the change that holds `_idle` holds
    /// `_commits` too. A copy pins with commits held (see
    /// [`pin_checkpoint_at`](Self::pin_checkpoint_at)) and goes on reading
    /// after it lets them go, so one that took commits between the count of
    /// [`hold_idle`](Self::hold_idle) and the change's own hold may still be
    /// reading. With both held the count is final: no transaction begins, a
    /// read waits to pin, and a copy needs commits.
    ///
    /// # Errors
    ///
    /// As [`hold_idle`](Self::hold_idle), while such a read is open.
    pub(crate) fn check_still_idle(
        &self,
        _idle: &IdleHold<'_>,
        _commits: &CommitsHeld<'_>,
        doing: &str,
    ) -> Result<()> {
        // The idle gate is held: no transaction is open.
        refuse_open_work(doing, 0, self.reads.epochs.lock().reads)
    }

    /// Holds a direct read of what a change that replaces the stores whole
    /// replaces (`get_node`, `list_graphs`, ...) for its call: the change
    /// waits for it, and it waits for the change, then reads the replaced
    /// stores whole. A direct read ends within its call, so it is waited for,
    /// not counted as open work. A direct read never holds another on the same
    /// thread: the gate prefers a waiting writer, so a nested read would wait
    /// behind the restore that waits for the outer one. Debug builds check it.
    pub(crate) fn hold_direct_read(&self) -> DirectRead<'_> {
        #[cfg(debug_assertions)]
        DIRECT_READ_DEPTH.with(|depth| {
            debug_assert_eq!(
                depth.get(),
                0,
                "a direct read inside a direct read: the replace gate is not reentrant"
            );
            depth.set(depth.get() + 1);
        });
        DirectRead {
            _gate: self.replace_gate.read(),
        }
    }

    /// Holds the direct reads off for the refill of a change that holds
    /// `_idle` and `_commits` and passed its final count (see
    /// [`check_still_idle`](Self::check_still_idle)): waits for the direct
    /// reads in progress, and the new ones wait until the guard drops.
    pub(crate) fn hold_replacement(
        &self,
        _idle: &IdleHold<'_>,
        _commits: &CommitsHeld<'_>,
    ) -> RwLockWriteGuard<'_, ()> {
        self.replace_gate.write()
    }

    /// Begins a private transaction: one without a session, for a direct
    /// call of the database or a graph handle while a transaction is open
    /// (or a batch), committed or rolled back by the call itself. Like
    /// [`begin`](Self::begin) it waits for a direct call holding the idle
    /// gate, not for a commit in progress: it reads at the published epoch.
    /// It does not wait for open transactions either: its writes are claimed
    /// like any transaction's, so it conflicts with an open (or committing)
    /// transaction that wrote the same entity first, and the other way round.
    ///
    /// # Errors
    ///
    /// Fails once the database is closed for writes (see
    /// [`check_open`](Self::check_open)): the call's commit would fail.
    pub(crate) fn begin_private(&self) -> Result<(TransactionId, Arc<TransactionChanges>)> {
        let _gate = self.idle_gate.lock();
        self.check_open()?;
        Ok(self.register(IsolationLevel::default(), true, false))
    }

    /// The writer of a bulk write that holds commits off for its whole run
    /// (an import, see
    /// [`StreamingChanges`](crate::transaction::StreamingChanges)): a
    /// transaction id no other transaction has, reading at the published
    /// epoch. Not registered: the bulk write creates only new nodes and
    /// edges, which no other transaction sees before it commits, so none
    /// conflicts with it; and it commits or rolls back itself while it holds
    /// commits off (`_commits`), so no checkpoint, copy or other commit sees
    /// it halfway.
    pub(crate) fn bulk_writer(&self, _commits: &CommitsHeld<'_>) -> (TransactionId, EpochId) {
        let transaction_id =
            TransactionId::new(self.next_transaction_id.fetch_add(1, Ordering::Relaxed));
        (transaction_id, self.current_epoch())
    }

    /// Registers a new active transaction reading at the published epoch,
    /// with its (empty) change set.
    fn register(
        &self,
        isolation_level: IsolationLevel,
        private: bool,
        expirable: bool,
    ) -> (TransactionId, Arc<TransactionChanges>) {
        let transaction_id =
            TransactionId::new(self.next_transaction_id.fetch_add(1, Ordering::Relaxed));
        // Counted while the `transactions` lock is held: a commit that
        // publishes meanwhile sees this transaction or does not, never half
        // (it keeps its claims for the transactions open then, see
        // `Publish::complete`). The snapshot is read and held under the
        // readers' lock, so the read horizon, which reads the published
        // epoch under it, never passes a snapshot it has not seen.
        let mut registry = self.transactions.write();
        let epoch = self.reads.hold_published(&self.published_epoch, true);
        let changes = self.change_set(transaction_id, epoch);
        let reader = Arc::new(SnapshotReader::new(
            Arc::clone(&self.reads),
            epoch,
            expirable && !private,
        ));
        let info = TransactionInfo::new(epoch, isolation_level, Arc::clone(&changes), reader);
        let Registry { open, claims } = &mut *registry;
        open.insert(transaction_id, info);
        // The second open transaction: from now on every claim is indexed, the
        // ones the first made alone too.
        if open.len() > 1 && !claims.indexed {
            claims.indexed = true;
            for (holder, info) in open.iter() {
                claims.index(*holder, info);
            }
        }
        self.active_count.fetch_add(1, Ordering::Release);
        drop(registry);
        (transaction_id, changes)
    }

    /// An empty change set for transaction `transaction_id`, which reads at
    /// `epoch`: one an ended transaction left (see [`recycle`](Self::recycle)),
    /// or a new one.
    fn change_set(&self, transaction_id: TransactionId, epoch: EpochId) -> Arc<TransactionChanges> {
        let spare = self.spare_changes.lock().pop();
        if let Some(mut changes) = spare
            && let Some(set) = Arc::get_mut(&mut changes)
        {
            set.reuse(transaction_id, epoch);
            set.memory = self.memory_manager();
            return changes;
        }
        let mut changes = TransactionChanges::new(transaction_id, epoch);
        changes.memory = self.memory_manager();
        Arc::new(changes)
    }

    /// The manager a new change set takes its grant from: one upgrade per
    /// transaction, none per write.
    fn memory_manager(&self) -> Option<Arc<grafeo_common::memory::buffer::BufferManager>> {
        self.memory.get().and_then(std::sync::Weak::upgrade)
    }

    /// Keeps the change set of a transaction that ended (committed or rolled
    /// back) for a next one, emptied: when nothing else holds it, its
    /// buffers are not those of a large transaction, and fewer than
    /// [`SPARE_CHANGE_SETS`] are kept. Otherwise it is dropped.
    pub(crate) fn recycle(&self, mut changes: Arc<TransactionChanges>) {
        let Some(set) = Arc::get_mut(&mut changes) else {
            return;
        };
        if !set.is_worth_keeping() {
            return;
        }
        // Emptied now, so the next transaction's begin pays nothing for
        // what this one held: it only names itself (`change_set`).
        set.empty();
        let mut spare = self.spare_changes.lock();
        if spare.len() < SPARE_CHANGE_SETS {
            spare.push(changes);
        }
    }

    /// What transaction `transaction_id` changed so far, while it is open.
    pub(crate) fn changes(&self, transaction_id: TransactionId) -> Option<Arc<TransactionChanges>> {
        self.transactions
            .read()
            .open
            .get(&transaction_id)
            .and_then(|info| info.changes.clone())
    }

    /// The change sets of the open transactions, for a checkpoint or a copy
    /// of the store that holds commits off (`_commits`): no commit is in
    /// progress, and no write or rollback of an open transaction, so the sets
    /// and the stores hold the same changes until the hold is released.
    pub(crate) fn open_change_sets(
        &self,
        _commits: &CommitsHeld<'_>,
    ) -> Vec<Arc<TransactionChanges>> {
        self.transactions
            .read()
            .open
            .values()
            .filter(|info| info.state == TransactionState::Active)
            .filter_map(|info| info.changes.clone())
            .collect()
    }

    /// The change sets of the open transactions, for a commit that holds the commit lock
    /// and has frozen writes ([`freeze_writes`](Self::freeze_writes)). Its own is not
    /// among them: it is committing.
    pub(crate) fn open_change_sets_while_frozen(&self) -> Vec<Arc<TransactionChanges>> {
        self.transactions
            .read()
            .open
            .values()
            .filter(|info| info.state == TransactionState::Active)
            .filter_map(|info| info.changes.clone())
            .collect()
    }

    /// Keeps the writes of open transactions out, for a commit that holds the commit lock
    /// and drops a graph they may be writing (see `write_freeze`): the commit lock is
    /// taken first, as [`hold_commits`](Self::hold_commits) takes it.
    pub(crate) fn freeze_writes(&self) -> RwLockWriteGuard<'_, ()> {
        self.write_freeze.write()
    }

    /// Fails when an open transaction other than `own` holds `key`, for a statement that
    /// would change it: it loses to the claim as a second transaction's write would.
    ///
    /// # Errors
    ///
    /// A write conflict naming the key.
    pub(crate) fn refuse_claimed(&self, key: &KeyClaim, own: Option<TransactionId>) -> Result<()> {
        if self.active_count.load(Ordering::Relaxed) == 0 {
            return Ok(());
        }
        let registry = self.transactions.read();
        let live = |holder: TransactionId| {
            Some(holder) != own
                && registry
                    .open
                    .get(&holder)
                    .is_some_and(|info| !Self::given_up(info))
        };
        // Not indexed: at most one transaction is open.
        let claimed = if registry.claims.indexed {
            registry
                .claims
                .open_keys
                .get(key)
                .is_some_and(|holders| holders.iter().any(live))
        } else {
            registry
                .open
                .iter()
                .any(|(holder, info)| info.key_set.contains(key) && live(*holder))
        };
        if claimed {
            return Err(Error::Transaction(TransactionError::WriteConflict(
                format!("on {key}: an open transaction changes the same name"),
            )));
        }
        Ok(())
    }

    /// Poisons the manager after a broken invariant that leaves the store
    /// in a state no commit may build on (a change applied but not
    /// recorded, an undo that failed): from now on every commit and
    /// checkpoint fails, as after a commit that did not complete.
    pub(crate) fn poison(&self, reason: &str) {
        grafeo_common::grafeo_error!(
            "{reason}; no transaction can commit until the database is reopened"
        );
        // Nothing more is published: no commit may build on this store.
        self.incomplete_from(EpochId::new(0));
    }

    /// Records that the commit at `epoch` did not complete: the manager is
    /// poisoned, no epoch at or above it is published, and everything that
    /// waits for an epoch to be published is woken to see it.
    fn incomplete_from(&self, epoch: EpochId) {
        self.poisoned.store(true, Ordering::Release);
        self.unpublishable_from
            .fetch_min(epoch.as_u64(), Ordering::AcqRel);
        self.wake_publish_waiters();
    }

    /// Wakes whatever waits for an epoch to be published. Taking the turn
    /// first means a waiter is either before its check, and sees the new
    /// state, or already waiting, and is woken.
    fn wake_publish_waiters(&self) {
        drop(self.publish_turn.lock());
        self.published.notify_all();
    }

    /// Waits until every written commit is published (see [`Publish`]):
    /// called with the commit lock held, so no further commit is written
    /// meanwhile. Returns at once when a commit did not complete: the
    /// commits at and above it are never published.
    fn wait_for_written_commits(&self) {
        let mut turn = self.publish_turn.lock();
        while !self.has_incomplete_commit()
            && self.published_epoch.load(Ordering::Acquire)
                < self.assigned_epoch.load(Ordering::Acquire)
        {
            self.published.wait(&mut turn);
        }
    }

    /// Returns the isolation level of a transaction.
    pub fn isolation_level(&self, transaction_id: TransactionId) -> Option<IsolationLevel> {
        self.transactions
            .read()
            .open
            .get(&transaction_id)
            .map(|info| info.isolation_level)
    }

    /// Fails when `transaction_id` was doomed (see
    /// [`record_key`](Self::record_key)): it lost a key or MERGE pattern
    /// claim to an older transaction, so it can claim nothing more and
    /// cannot commit.
    fn refuse_doomed(info: &TransactionInfo, transaction_id: TransactionId) -> Result<()> {
        match Self::doom_error(info, transaction_id) {
            Some(error) => Err(Error::Transaction(error)),
            None => Ok(()),
        }
    }

    /// The error of a doomed transaction (see [`refuse_doomed`](Self::refuse_doomed)).
    fn doom_error(
        info: &TransactionInfo,
        transaction_id: TransactionId,
    ) -> Option<TransactionError> {
        info.doomed_by.as_ref().map(|(older, lost)| {
            TransactionError::WriteConflict(format!(
                "{lost}: transaction {transaction_id} lost this claim to transaction {older}, \
                 which began first; retry the transaction"
            ))
        })
    }

    /// Whether the database gave up on the transaction: it was doomed, or its snapshot
    /// expired. Its key and MERGE pattern claims no longer count then (they guard no store
    /// version of it), so a writer of the same key or pattern goes on at once; it cannot
    /// commit, and its session rolls it back at its next call.
    fn given_up(info: &TransactionInfo) -> bool {
        info.doomed_by.is_some() || info.reader.is_expired()
    }

    /// Why the database gave up on the open transaction `transaction_id`, when it did: its
    /// snapshot expired (`GRAFEO-T012`), or it lost a key or MERGE pattern claim to an older
    /// transaction (`GRAFEO-T001`). Its session then rolls it back.
    pub(crate) fn given_up_on(&self, transaction_id: TransactionId) -> Option<TransactionError> {
        let registry = self.transactions.read();
        let info = registry
            .open
            .get(&transaction_id)
            .filter(|info| info.state == TransactionState::Active)?;
        Self::doom_error(info, transaction_id).or_else(|| {
            info.reader
                .is_expired()
                .then(|| info.reader.expired_transaction_error())
        })
    }

    /// Settles a key or MERGE pattern claim of `requester` against the other
    /// open transactions that hold it (`holders`, from the open index; oldest
    /// claim wins, without waiting): a holder that is committing, or that
    /// began before the requester, keeps the claim, and the requester fails;
    /// every younger holder that is still active is doomed, and the requester
    /// takes the claim. A doomed holder's claims do not count, and a doomed
    /// one keeps `claim` (`on key ...`, formatted only when one is doomed) for
    /// its error. Returns the holder that kept the claim and whether it was
    /// committing; fails first when the requester itself was doomed, which
    /// then dooms nobody.
    fn settle_claim(
        open: &mut FxHashMap<TransactionId, TransactionInfo>,
        reads: &ReadPins,
        holders: Option<&Holders>,
        requester: TransactionId,
        claim: &dyn std::fmt::Display,
    ) -> Result<std::result::Result<(), (TransactionId, bool)>> {
        if let Some(info) = open.get(&requester) {
            Self::refuse_doomed(info, requester)?;
            if info.reader.is_expired() {
                return Err(info.reader.expired_error());
            }
        }
        let mut younger = Vec::new();
        for other in holders.into_iter().flat_map(Holders::iter) {
            let Some(info) = open.get(&other).filter(|_| other != requester) else {
                continue;
            };
            if Self::given_up(info) {
                continue;
            }
            let committing = info.state == TransactionState::Committing;
            if committing || other < requester {
                return Ok(Err((other, committing)));
            }
            younger.push(other);
        }
        if !younger.is_empty() {
            let lost = claim.to_string();
            // Doomed, it cannot commit: from now on it holds the
            // committed-claims index back no more.
            let mut readers = reads.epochs.lock();
            for other in younger {
                if let Some(info) = open.get_mut(&other) {
                    info.doomed_by = Some((requester, lost.clone()));
                    info.reader.stop_committable(&mut readers);
                }
            }
        }
        Ok(Ok(()))
    }

    /// Records a write operation for the transaction.
    ///
    /// Uses first-writer-wins: if another active transaction has already
    /// written to the same entity, returns a write-write conflict error
    /// immediately (before the caller mutates the store).
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction is not active or if another
    /// active transaction has already written to the same entity.
    pub fn record_write(
        &self,
        transaction_id: TransactionId,
        entity: impl Into<GraphEntity>,
    ) -> Result<()> {
        self.record_claims(transaction_id, Claim::Write, [entity.into()])
    }

    /// Records the delete of a node: a write, as
    /// [`record_write`](Self::record_write) records one, that also conflicts
    /// with an open transaction that creates an edge to the node (see
    /// [`record_endpoints`](Self::record_endpoints)), first writer wins; at
    /// commit, with one that did and committed after this one began.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction is not active, or if another open
    /// transaction wrote the node or claimed it as an edge's endpoint.
    pub fn record_delete(
        &self,
        transaction_id: TransactionId,
        entity: impl Into<GraphEntity>,
    ) -> Result<()> {
        self.record_claims(transaction_id, Claim::Delete, [entity.into()])
    }

    /// Claims the endpoints of an edge the transaction is about to create,
    /// so no committed edge ends at a deleted node: an open transaction that
    /// deletes one of them conflicts with this one, first writer wins, and at
    /// commit so does one that deleted one and committed after this one
    /// began. A claim does not conflict with other writes of the node or
    /// with other claims on it: transactions that create edges to one node,
    /// or set its properties, go on side by side.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction is not active, or if another open
    /// transaction deletes one of the endpoints; nothing is claimed then.
    pub fn record_endpoints(
        &self,
        transaction_id: TransactionId,
        endpoints: [GraphEntity; 2],
    ) -> Result<()> {
        self.record_claims(transaction_id, Claim::Endpoint, endpoints)
    }

    /// Records `claim` of each of `entities` for `transaction_id`, after
    /// checking it against the writes and claims of the other open
    /// transactions (first-writer-wins): all of them, or none on a conflict.
    /// After a commit that did not complete nothing can commit, so a write
    /// fails with that error first, not with a conflict a retry cannot get
    /// past (the incomplete commit keeps its claims until the reopen).
    fn record_claims<const N: usize>(
        &self,
        transaction_id: TransactionId,
        claim: Claim,
        entities: [GraphEntity; N],
    ) -> Result<()> {
        self.check_no_incomplete_commit()?;
        let mut registry = self.transactions.write();
        let Registry { open, claims } = &mut *registry;

        // First-writer-wins conflict detection: one lookup per entity in the
        // open index, kept while two or more transactions are open (a lone
        // one, the common auto-commit case, conflicts with nobody). A commit
        // in progress still holds its claims.
        if claims.indexed {
            if let Some(ours) = open.get(&transaction_id) {
                Self::refuse_doomed(ours, transaction_id)?;
            }
            for entity in &entities {
                let conflict = claims.open.get(entity).and_then(|holders| {
                    holders
                        .iter()
                        .filter(|holder| *holder != transaction_id)
                        .filter_map(|holder| open.get(&holder))
                        .find_map(|other| other.conflict_with(claim, entity))
                });
                if let Some(conflict) = conflict {
                    return Err(Error::Transaction(TransactionError::WriteConflict(
                        conflict,
                    )));
                }
            }
        }

        // Single lookup: get_mut for both state check and write_set insert
        let info = open.get_mut(&transaction_id).ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(
                "Transaction not found".to_string(),
            ))
        })?;

        if info.state != TransactionState::Active {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "Transaction is not active".to_string(),
            )));
        }
        Self::refuse_doomed(info, transaction_id)?;

        if claims.indexed {
            for entity in &entities {
                hold(&mut claims.open, entity.clone(), transaction_id);
            }
        }
        for entity in entities {
            match claim {
                Claim::Write => {
                    if let Some(undo) = &mut info.claim_undo {
                        if info.write_set.insert(entity.clone()) {
                            undo.push(ClaimUndo::Write(entity));
                        }
                    } else {
                        info.write_set.insert(entity);
                    }
                }
                Claim::Delete => {
                    if let Some(undo) = &mut info.claim_undo {
                        if info.write_set.insert(entity.clone()) {
                            undo.push(ClaimUndo::Write(entity.clone()));
                        }
                        if info.delete_set.insert(entity.clone()) {
                            undo.push(ClaimUndo::Delete(entity));
                        }
                    } else {
                        info.write_set.insert(entity.clone());
                        info.delete_set.insert(entity);
                    }
                }
                Claim::Endpoint => {
                    if let Some(undo) = &mut info.claim_undo {
                        if info.endpoint_set.insert(entity.clone()) {
                            undo.push(ClaimUndo::Endpoint(entity));
                        }
                    } else {
                        info.endpoint_set.insert(entity);
                    }
                }
            }
        }
        Ok(())
    }

    /// Claims `key`, a value the transaction writes under a UNIQUE, PRIMARY
    /// KEY or NODE KEY constraint, so no two transactions commit the same
    /// one: when another open transaction claimed it, this one fails, first
    /// writer wins, and at commit it fails if one that claimed it committed
    /// after it began. A constraint check calls this once it found no
    /// committed node with the value.
    ///
    /// # Errors
    ///
    /// Returns a write conflict if another open transaction claimed `key`,
    /// and an error if the transaction is not active or an earlier commit did
    /// not complete.
    pub fn record_key(&self, transaction_id: TransactionId, key: KeyClaim) -> Result<()> {
        self.check_no_incomplete_commit()?;
        let mut registry = self.transactions.write();
        let Registry { open, claims } = &mut *registry;
        if open.len() > 1
            && let Err((holder, committing)) = Self::settle_claim(
                open,
                &self.reads,
                claims.open_keys.get(&key),
                transaction_id,
                &ClaimName::Key(&key),
            )?
        {
            return Err(Error::Transaction(TransactionError::WriteConflict(
                format!(
                    "on key {key}: transaction {holder} ({}) writes the same value of the key; \
                 retry the statement, or the transaction (transaction {transaction_id})",
                    holder_kind(committing)
                ),
            )));
        }
        let info = open.get_mut(&transaction_id).ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(
                "Transaction not found".to_string(),
            ))
        })?;
        if info.state != TransactionState::Active {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "Transaction is not active".to_string(),
            )));
        }
        Self::refuse_doomed(info, transaction_id)?;
        if info.claim_undo.is_none() && !claims.indexed {
            info.key_set.insert(key);
        } else if info.key_set.insert(key.clone()) {
            if claims.indexed {
                hold(&mut claims.open_keys, key.clone(), transaction_id);
            }
            if let Some(undo) = &mut info.claim_undo {
                undo.push(ClaimUndo::Key(key));
            }
        }
        Ok(())
    }

    /// Claims `fingerprint`, a MERGE pattern the transaction creates (the
    /// pattern `pattern` names, for the error), so two transactions that MERGE
    /// one pattern do not both create it: when another open transaction
    /// claimed it, this one fails, first writer wins, and at commit it fails
    /// if one that claimed it committed after it began. A transaction may
    /// claim one pattern any number of times. A MERGE calls this when nothing
    /// matched, before it creates.
    ///
    /// # Errors
    ///
    /// Returns a write conflict if another open transaction claimed
    /// `fingerprint`, and an error if the transaction is not active or an
    /// earlier commit did not complete.
    pub fn record_pattern(
        &self,
        transaction_id: TransactionId,
        fingerprint: u64,
        pattern: &dyn std::fmt::Display,
    ) -> Result<()> {
        self.check_no_incomplete_commit()?;
        let mut registry = self.transactions.write();
        let Registry { open, claims } = &mut *registry;
        if open.len() > 1
            && let Err((holder, committing)) = Self::settle_claim(
                open,
                &self.reads,
                claims.open_patterns.get(&fingerprint),
                transaction_id,
                &ClaimName::Pattern(pattern),
            )?
        {
            return Err(Error::Transaction(TransactionError::WriteConflict(
                format!(
                    "on MERGE {pattern}: transaction {holder} ({}) creates the same pattern; \
                 retry the statement, or the transaction (transaction {transaction_id})",
                    holder_kind(committing)
                ),
            )));
        }
        let info = open.get_mut(&transaction_id).ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(
                "Transaction not found".to_string(),
            ))
        })?;
        if info.state != TransactionState::Active {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "Transaction is not active".to_string(),
            )));
        }
        Self::refuse_doomed(info, transaction_id)?;
        if info.pattern_set.insert(fingerprint) {
            if claims.indexed {
                hold(&mut claims.open_patterns, fingerprint, transaction_id);
            }
            if let Some(undo) = &mut info.claim_undo {
                undo.push(ClaimUndo::Pattern(fingerprint));
            }
        }
        Ok(())
    }

    /// Records a read operation for the transaction (for serializable isolation).
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction is not active.
    pub fn record_read(
        &self,
        transaction_id: TransactionId,
        entity: impl Into<GraphEntity>,
    ) -> Result<()> {
        let mut registry = self.transactions.write();
        let info = registry.open.get_mut(&transaction_id).ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(
                "Transaction not found".to_string(),
            ))
        })?;

        if info.state != TransactionState::Active {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "Transaction is not active".to_string(),
            )));
        }

        info.read_set.insert(entity.into());
        Ok(())
    }

    /// Commits a transaction with conflict detection.
    ///
    /// # Conflict Detection
    ///
    /// - **All isolation levels**: Write-write conflicts (two transactions writing
    ///   to the same entity) are always detected and cause the second committer to abort.
    ///
    /// - **Serializable only**: Read-write conflicts (SSI validation) are additionally
    ///   checked. If transaction T1 read an entity that another transaction T2 wrote,
    ///   and T2 committed after T1 started, T1 will abort. This prevents write skew
    ///   for the reads recorded with [`record_read`](Self::record_read); queries
    ///   record none yet (#400), which is why a session refuses this level.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The transaction is not active
    /// - There's a write-write conflict with another committed transaction
    /// - (Serializable only) There's a read-write conflict (SSI violation)
    /// - An earlier commit did not complete (see [`TransactionManager`]): no
    ///   transaction commits until the database is reopened
    /// - The database is closed: `close()` of a persistent database started
    /// - A store fails to stamp the transaction's changes: the commit does
    ///   not complete, which poisons the manager
    ///
    /// The changes the transaction recorded (the writes of a
    /// [`QueryProcessor`](crate::query::QueryProcessor) with its transaction
    /// context) are stamped with the commit epoch, so they are visible once
    /// it is published. They are not logged: a session's commit logs its
    /// transaction.
    pub fn commit(&self, transaction_id: TransactionId) -> Result<EpochId> {
        let changes = self.changes(transaction_id);
        let commit = self.start_commit(transaction_id)?;
        let epoch = commit.epoch();
        if let Some(changes) = changes {
            // A store that fails leaves the commit half stamped: the guard,
            // dropped uncompleted, poisons the manager.
            changes.commit(epoch).map_err(|error| {
                Error::Internal(format!(
                    "the commit of transaction {transaction_id:?} could not stamp its changes: \
                     {error}"
                ))
            })?;
        }
        commit.complete()?;
        Ok(epoch)
    }

    /// Commits a transaction like [`commit`](Self::commit), but completes the
    /// commit only when the returned guard is dropped (see [`CommitGuard`]):
    /// the caller writes the commit's versions, events and WAL records first.
    ///
    /// # Errors
    ///
    /// As [`commit`](Self::commit); the transaction then stays active.
    pub(crate) fn start_commit(&self, transaction_id: TransactionId) -> Result<CommitGuard<'_>> {
        self.start_commit_after(transaction_id, false)
    }

    /// [`start_commit`](Self::start_commit) of a commit that reads what others committed
    /// (one with schema statements, whose indexes hold it): waits for the commits written
    /// before it to be published.
    ///
    /// # Errors
    ///
    /// As [`start_commit`](Self::start_commit).
    pub(crate) fn start_commit_when_quiet(
        &self,
        transaction_id: TransactionId,
    ) -> Result<CommitGuard<'_>> {
        self.start_commit_after(transaction_id, true)
    }

    fn start_commit_after(
        &self,
        transaction_id: TransactionId,
        quiet: bool,
    ) -> Result<CommitGuard<'_>> {
        // The bytes of the RDF writes the commit keeps, measured before any
        // lock of the manager is taken (it walks them): only this thread
        // changes them now.
        #[cfg(feature = "triple-store")]
        let kept_bytes = {
            let changes = self
                .transactions
                .read()
                .open
                .get(&transaction_id)
                .and_then(|info| info.changes.clone());
            changes.map_or(0, |changes| changes.rdf_writes_bytes())
        };
        let commit_lock = self.commit_lock.lock();
        if quiet {
            self.wait_for_written_commits();
        }
        // A commit that did not complete sets this before it releases the
        // commit lock: any epoch published after it would publish its
        // stamped versions too.
        self.check_no_incomplete_commit()?;
        self.check_commits_allowed()?;
        self.check_open()?;
        // One write lock: the checks, the state and the commit epoch change
        // together, so no other commit sees this one half started.
        let mut registry = self.transactions.write();
        let Registry { open, claims } = &mut *registry;

        // First, validate the transaction exists and is active
        let ours = open.get(&transaction_id).ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(
                "Transaction not found".to_string(),
            ))
        })?;
        if ours.state != TransactionState::Active {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "Transaction is not active".to_string(),
            )));
        }
        Self::refuse_doomed(ours, transaction_id)?;
        if ours.reader.is_expired() {
            return Err(ours.reader.expired_error());
        }
        let our_start_epoch = ours.start_epoch.as_u64();

        // Only a transaction that committed after ours began can conflict
        // with it, and each commit advances the epoch: if the epoch has not
        // moved since, there is nothing to check. (Commits hold the
        // `transactions` lock, so none can happen during the checks.)
        // Transactions committed before our start are part of our snapshot,
        // so overwriting their values is not a conflict.
        if self.assigned_epoch.load(Ordering::Acquire) > our_start_epoch
            && let Some(conflict) =
                claims.commit_conflict(open, transaction_id, ours, our_start_epoch)
        {
            return Err(conflict);
        }

        // Commit successful: advance epoch atomically.
        // SeqCst ensures all threads see commits in a consistent total order.
        let commit_epoch = EpochId::new(self.assigned_epoch.fetch_add(1, Ordering::SeqCst) + 1);

        // The transaction stays open, committing, until it is published or
        // its commit is taken back.
        if let Some(info) = open.get_mut(&transaction_id) {
            info.state = TransactionState::Committing;
            info.commit_epoch = Some(commit_epoch);
            // Its RDF writes, for the checks of the commits after it, which
            // then never wait on its change set while they hold this lock.
            #[cfg(feature = "triple-store")]
            if let Some(changes) = &info.changes
                && changes.has_pending_triples()
            {
                info.rdf_writes = Some(KeptRdfWrites::new(
                    changes.take_rdf_writes(),
                    kept_bytes,
                    &self.kept_rdf_bytes,
                    self.memory_manager(),
                ));
            }
        }
        claims.committing.push(transaction_id);

        Ok(CommitGuard {
            commit: Publish {
                manager: self,
                transaction_id,
                epoch: commit_epoch,
                // Under the commit lock: the times follow the epochs.
                commit_time: self.history.commit_time(),
                finished: false,
                #[cfg(feature = "grafeo-file")]
                changed: true,
            },
            _commit: commit_lock,
        })
    }

    /// Holds commits off for a checkpoint or a copy of the store: while the
    /// returned guard lives, no commit is between its epoch and its
    /// completion, and no commit, transaction start or write outside a
    /// transaction can begin. It also freezes the store (see `write_freeze`):
    /// no store change of an open transaction, a write or the undo of a
    /// rollback, is in progress or can begin, so the image is read from a
    /// store and change logs that do not move. Waits for a commit in
    /// progress to complete, then for the store changes in progress.
    ///
    /// Lock order: the commit lock, then the write freeze. The calling
    /// thread must not have a write in progress (see
    /// [`write_in_progress`](Self::write_in_progress)): it would wait for
    /// itself.
    ///
    /// # Errors
    ///
    /// Fails, once the commit lock is held, when a commit did not complete
    /// (see [`check_no_incomplete_commit`](Self::check_no_incomplete_commit)):
    /// the store holds its stamped versions, which no image may contain.
    pub(crate) fn hold_commits(&self) -> Result<CommitsHeld<'_>> {
        let commit = self.commit_lock.lock();
        self.wait_for_written_commits();
        self.check_no_incomplete_commit()?;
        let writes = self.write_freeze.write();
        Ok(CommitsHeld {
            writes: Some(writes),
            commit: Some(commit),
            #[cfg(feature = "grafeo-file")]
            change: None,
        })
    }

    /// Holds commits off for a change that takes effect at once and logs its
    /// own WAL group outside any commit (a schema statement, a graph command,
    /// an RDF update or a write outside a transaction), for as long as the
    /// guard lives: a checkpoint, a copy or `close()` sees all of it or none
    /// of it. Waits for a commit or checkpoint in progress, then fails once
    /// the database is closed (see [`check_open`](Self::check_open)) or after
    /// a commit that did not complete.
    ///
    /// It does not freeze the store (see `write_freeze`): the writes of open
    /// transactions go on meanwhile.
    ///
    /// # Errors
    ///
    /// [`TransactionError::IncompleteCommit`] or
    /// [`TransactionError::DatabaseClosed`].
    pub(crate) fn hold_commits_for_drop(&self) -> Result<CommitsHeld<'_>> {
        let held = self.hold_commits()?;
        // Under the commit lock, as a commit's check: a drop is a change the
        // log and the next checkpoint carry.
        self.check_commits_allowed()?;
        self.check_open()?;
        Ok(held)
    }

    /// Holds commits off for a change outside a transaction (see
    /// [`hold_commits`](Self::hold_commits)), refused as a commit is.
    ///
    /// # Errors
    ///
    /// As [`hold_commits_for_change`](Self::hold_commits_for_change).
    pub(crate) fn hold_commits_for_change(&self) -> Result<CommitsHeld<'_>> {
        let commit = self.commit_lock.lock();
        self.wait_for_written_commits();
        self.check_no_incomplete_commit()?;
        self.check_commits_allowed()?;
        self.check_open()?;
        Ok(CommitsHeld {
            writes: None,
            commit: Some(commit),
            #[cfg(feature = "grafeo-file")]
            change: None,
        })
    }

    /// Marks a store change of an open transaction as in progress, for as
    /// long as the returned guard lives: a write and its record in the
    /// transaction's change set (through the transaction's change recorder),
    /// or the undo of a rollback or a rollback to a savepoint. It waits while a
    /// checkpoint or a copy of the store holds commits off (see
    /// [`hold_commits`](Self::hold_commits)), and they wait for it.
    ///
    /// Not reentrant: the calling thread must not hold one already, nor
    /// hold commits off for a checkpoint or a copy (see `write_freeze`). It
    /// may hold the commit lock (taken before this), but nothing else a
    /// checkpoint waits for while it holds commits off.
    pub(crate) fn write_in_progress(&self) -> WriteInProgress<'_> {
        self.write_freeze.read()
    }

    /// Closes the database for writes: from now on every commit and every
    /// write outside a transaction fails (see [`check_open`](Self::check_open)).
    /// Waits for a commit in progress to complete. A persistent database
    /// calls this when it closes, before its final checkpoint: a write that
    /// ran after that checkpoint would be written only to the WAL the close
    /// removes. Transactions still begin, and reads still work.
    pub(crate) fn close_for_writes(&self) {
        let _commit = self.commit_lock.lock();
        self.wait_for_written_commits();
        // Set while the commit lock is held: a commit or a write outside a
        // transaction waiting for the lock sees it.
        self.closed.store(true, Ordering::Release);
    }

    /// Fails once the database is closed for writes (see
    /// [`close_for_writes`](Self::close_for_writes)). Every commit, and every
    /// write outside a transaction, calls this while it holds the commit
    /// lock.
    ///
    /// # Errors
    ///
    /// Returns [`TransactionError::DatabaseClosed`].
    pub(crate) fn check_open(&self) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::Transaction(TransactionError::DatabaseClosed));
        }
        Ok(())
    }

    /// Whether a commit did not complete (see [`TransactionManager`]): its
    /// stamped versions are in the store, so from then on no transaction
    /// commits and nothing may persist or copy the store.
    #[must_use]
    pub fn has_incomplete_commit(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    /// Fails when a commit did not complete (see
    /// [`has_incomplete_commit`](Self::has_incomplete_commit)). Every commit,
    /// checkpoint and copy of the store calls this first.
    ///
    /// # Errors
    ///
    /// Returns [`TransactionError::IncompleteCommit`], which says a commit
    /// did not complete and the database must be reopened.
    pub fn check_no_incomplete_commit(&self) -> Result<()> {
        if self.has_incomplete_commit() {
            return Err(Error::Transaction(TransactionError::IncompleteCommit));
        }
        Ok(())
    }

    /// Refuses every commit and write outside a transaction from now on,
    /// until the database is opened again; reads, checkpoints and copies go
    /// on. The first refusal stays.
    #[cfg(feature = "grafeo-file")]
    pub(crate) fn refuse_commits(&self, refusal: CommitRefusal) {
        self.commit_refusal.lock().get_or_insert(refusal);
    }

    /// Fails when commits are refused (see
    /// [`refuse_commits`](Self::refuse_commits)).
    fn check_commits_allowed(&self) -> Result<()> {
        match &*self.commit_refusal.lock() {
            None => Ok(()),
            Some(CommitRefusal::OutcomeUnknown(cause)) => Err(Error::from(
                grafeo_common::utils::error::StorageError::CheckpointOutcomeUnknown(format!(
                    "a database without a WAL takes no commits after a checkpoint whose \
                     outcome is unknown ({cause}): reopen the database"
                )),
            )),
            Some(CommitRefusal::FailedRestore(None)) => {
                Err(Error::Transaction(TransactionError::IncompleteCommit))
            }
            Some(CommitRefusal::FailedRestore(Some(cause))) => Err(Error::from(
                grafeo_common::utils::error::StorageError::CheckpointOutcomeUnknown(format!(
                    "a restore whose checkpoint outcome is unknown takes no commits ({cause}): \
                     reopen the database"
                )),
            )),
        }
    }

    /// Aborts a transaction, undoing the changes it recorded that nobody
    /// undid yet (the writes of a
    /// [`QueryProcessor`](crate::query::QueryProcessor) with its transaction
    /// context; a session and a direct call undo their own first).
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction is not active, or when a store
    /// fails to undo a change, which poisons the manager (the transaction is
    /// aborted either way).
    pub fn abort(&self, transaction_id: TransactionId) -> Result<()> {
        let undone = match self.changes(transaction_id) {
            Some(changes) => {
                // A store change in progress: a checkpoint never reads the
                // store or the change set halfway through the undo.
                let _writing = self.write_in_progress();
                changes.rollback()
            }
            None => Ok(()),
        };
        let undo_error = match undone {
            Err(error) => {
                let message = format!(
                    "the abort of transaction {transaction_id:?} could not undo its changes: \
                     {error}"
                );
                self.poison(&message);
                Some(Error::Internal(message))
            }
            Ok(()) => None,
        };
        self.abort_registered(transaction_id)?;
        // After the lock: what was kept for this reader may go.
        self.reads.reader_ended();
        undo_error.map_or(Ok(()), Err)
    }

    /// Ends transaction `transaction_id`, aborted: it leaves the registry with
    /// its claims, its snapshot is released, and the committed-claims index
    /// trims to the transactions still open.
    fn abort_registered(&self, transaction_id: TransactionId) -> Result<()> {
        // Dropped after the locks.
        let (_aborted, _trimmed) = {
            let mut registry = self.transactions.write();
            let Registry { open, claims } = &mut *registry;
            let info = open.remove(&transaction_id).ok_or_else(|| {
                Error::Transaction(TransactionError::InvalidState(
                    "Transaction not found".to_string(),
                ))
            })?;
            if info.state != TransactionState::Active {
                // A commit in progress stays.
                open.insert(transaction_id, info);
                return Err(Error::Transaction(TransactionError::InvalidState(
                    "Transaction is not active".to_string(),
                )));
            }
            if claims.indexed {
                claims.unindex(transaction_id, &info);
            }
            self.active_count.fetch_sub(1, Ordering::Relaxed);
            let mut readers = self.reads.epochs.lock();
            info.reader.end_in(&mut readers);
            (info, Self::trim_committed(open, claims, &readers))
        };
        Ok(())
    }

    /// Captures conflict claims in O(1), enabling the inverse log lazily.
    pub(crate) fn mark_claims(&self, transaction_id: TransactionId) -> Result<ClaimMark> {
        let mut registry = self.transactions.write();
        let info = registry.open.get_mut(&transaction_id).ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(
                "Transaction not found".to_string(),
            ))
        })?;
        if info.state != TransactionState::Active {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "Transaction is not active".to_string(),
            )));
        }
        Ok(ClaimMark(
            info.claim_undo.get_or_insert_with(Vec::new).len(),
        ))
    }

    /// The length of the open transaction's claim log, `None` without one (tests).
    #[cfg(test)]
    pub(crate) fn claim_log_len(&self, transaction_id: TransactionId) -> Option<usize> {
        self.transactions
            .read()
            .open
            .get(&transaction_id)
            .and_then(|info| info.claim_undo.as_ref().map(Vec::len))
    }

    /// Drops the claim log of an active transaction: no mark can rewind it any more (the
    /// session holds no statement mark and no savepoint). Its claims stay in its sets.
    pub(crate) fn trim_claims(&self, transaction_id: TransactionId) {
        if let Some(info) = self.transactions.write().open.get_mut(&transaction_id)
            && info.state == TransactionState::Active
        {
            info.claim_undo = None;
        }
    }

    /// Rewinds conflict claims after the caller has successfully undone the
    /// corresponding storage changes, in the transaction's sets and in the
    /// open index. Never hold this lock during backend undo.
    pub(crate) fn restore_claims(
        &self,
        transaction_id: TransactionId,
        mark: ClaimMark,
    ) -> Result<()> {
        let mut registry = self.transactions.write();
        let Registry { open, claims } = &mut *registry;
        let info = open.get_mut(&transaction_id).ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(
                "Transaction not found".to_string(),
            ))
        })?;
        if info.state != TransactionState::Active {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "Transaction is not active".to_string(),
            )));
        }
        if info
            .claim_undo
            .as_ref()
            .is_none_or(|undo| mark.0 > undo.len())
        {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "Invalid conflict-claim savepoint".to_string(),
            )));
        }
        while let Some(undo) = info
            .claim_undo
            .as_mut()
            .filter(|undo| undo.len() > mark.0)
            .and_then(Vec::pop)
        {
            match undo {
                ClaimUndo::Write(entity) => {
                    info.write_set.remove(&entity);
                    if claims.indexed {
                        claims.sync_entity(transaction_id, info, entity);
                    }
                }
                ClaimUndo::Delete(entity) => {
                    info.delete_set.remove(&entity);
                }
                ClaimUndo::Endpoint(entity) => {
                    info.endpoint_set.remove(&entity);
                    if claims.indexed {
                        claims.sync_entity(transaction_id, info, entity);
                    }
                }
                ClaimUndo::Key(key) => {
                    info.key_set.remove(&key);
                    if claims.indexed {
                        release(&mut claims.open_keys, &key, transaction_id);
                    }
                }
                ClaimUndo::Pattern(fingerprint) => {
                    info.pattern_set.remove(fingerprint);
                    if claims.indexed {
                        release(&mut claims.open_patterns, &fingerprint, transaction_id);
                    }
                }
            }
        }
        Ok(())
    }

    /// Returns the write set of a transaction.
    ///
    /// This returns a copy of the entities written by this transaction,
    /// used for rollback to discard uncommitted versions.
    ///
    /// # Errors
    ///
    /// Returns a `TransactionError::InvalidState` if the transaction is not found.
    pub fn get_write_set(&self, transaction_id: TransactionId) -> Result<FxHashSet<GraphEntity>> {
        let registry = self.transactions.read();
        let info = registry.open.get(&transaction_id).ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(
                "Transaction not found".to_string(),
            ))
        })?;
        Ok(info.write_set.clone())
    }

    /// Aborts all active transactions: each leaves the registry, as an abort
    /// does. A commit in progress stays.
    ///
    /// Used during database shutdown.
    pub fn abort_all_active(&self) {
        // Dropped after the locks.
        let (_aborted, _trimmed) = {
            let mut registry = self.transactions.write();
            let Registry { open, claims } = &mut *registry;
            let active: Vec<TransactionId> = open
                .iter()
                .filter(|(_, info)| info.state == TransactionState::Active)
                .map(|(id, _)| *id)
                .collect();
            let mut readers = self.reads.epochs.lock();
            let aborted: Vec<TransactionInfo> = active
                .into_iter()
                .filter_map(|id| {
                    let info = open.remove(&id)?;
                    if claims.indexed {
                        claims.unindex(id, &info);
                    }
                    info.reader.end_in(&mut readers);
                    self.active_count.fetch_sub(1, Ordering::Relaxed);
                    Some(info)
                })
                .collect();
            (aborted, Self::trim_committed(open, claims, &readers))
        };
    }

    /// Returns the state of a transaction.
    pub fn state(&self, transaction_id: TransactionId) -> Option<TransactionState> {
        self.transactions
            .read()
            .open
            .get(&transaction_id)
            .map(|info| info.state)
    }

    /// Returns the start epoch of a transaction.
    pub fn start_epoch(&self, transaction_id: TransactionId) -> Option<EpochId> {
        self.transactions
            .read()
            .open
            .get(&transaction_id)
            .map(|info| info.start_epoch)
    }

    /// Returns the current epoch: the last one whose commit is complete, so a
    /// snapshot at this epoch never holds part of a commit.
    #[must_use]
    pub fn current_epoch(&self) -> EpochId {
        EpochId::new(self.published_epoch.load(Ordering::Acquire))
    }

    /// The epoch a read asked for at `epoch` reads at: `epoch`, or the
    /// [current epoch](Self::current_epoch) when that is earlier, so a read
    /// never sees a commit that is not complete (one still in progress, or
    /// one that failed partway). Every read at a chosen epoch goes through it.
    #[must_use]
    pub fn readable_epoch(&self, epoch: EpochId) -> EpochId {
        epoch.min(self.current_epoch())
    }

    /// Synchronizes the epoch counter to at least the given value.
    ///
    /// Used after snapshot import and WAL recovery to align the
    /// TransactionManager epoch with the store epoch, and by a write outside
    /// any transaction (which runs while no commit is in progress) to publish
    /// its epoch.
    pub fn sync_epoch(&self, epoch: EpochId) {
        self.assigned_epoch
            .fetch_max(epoch.as_u64(), Ordering::SeqCst);
        self.published_epoch
            .fetch_max(epoch.as_u64(), Ordering::SeqCst);
        self.wake_publish_waiters();
    }

    /// Publishes `epoch`, a bulk RDF load's, which the caller applied while
    /// it holds commits off (`_commits`). While a transaction is open, the
    /// load is first recorded as a committed writer whose RDF writes
    /// `writes` gives (its net inserts), so the commit of a transaction open
    /// across it checks against them as against a transaction's. Recorded
    /// and published under the `transactions` lock, under which a
    /// transaction begins: none begins between the check and the publish
    /// and reads below the load unrecorded. The load itself conflicts with
    /// nothing: it applied against the newest state at its own epoch.
    #[cfg(feature = "triple-store")]
    pub(crate) fn publish_load(
        &self,
        _commits: &CommitsHeld<'_>,
        epoch: EpochId,
        writes: impl FnOnce() -> super::rdf::PendingTriples,
    ) {
        // Built before the `transactions` lock while a transaction is open,
        // which every begin and commit check takes: a load's run is walked
        // whole. Commits are held, so only a begin can come between.
        let measured = |writes: super::rdf::PendingTriples| {
            let bytes = writes.heap_bytes();
            (writes, bytes)
        };
        let mut writes = Some(writes);
        let built = (self.active_count.load(Ordering::Acquire) > 0)
            .then(|| writes.take().map(|writes| measured(writes())))
            .flatten();
        {
            let mut registry = self.transactions.write();
            // Counted under this lock (see `register`): one that began since
            // the read above is open across the load too, and its writes are
            // built here then, a rare case.
            if self.active_count.load(Ordering::Acquire) > 0 {
                let (writes, bytes) = built
                    .or_else(|| writes.take().map(|writes| measured(writes())))
                    .unwrap_or_default();
                registry.claims.add_committed(CommittedEntry {
                    epoch: epoch.as_u64(),
                    writes: FxHashSet::default(),
                    deletes: FxHashSet::default(),
                    endpoints: FxHashSet::default(),
                    keys: FxHashSet::default(),
                    patterns: PatternSet::default(),
                    rdf: Some(KeptRdfWrites::new(
                        writes,
                        bytes,
                        &self.kept_rdf_bytes,
                        self.memory_manager(),
                    )),
                });
            }
            self.assigned_epoch
                .fetch_max(epoch.as_u64(), Ordering::SeqCst);
            self.published_epoch
                .fetch_max(epoch.as_u64(), Ordering::SeqCst);
        }
        // After the `transactions` lock: a commit's publish takes the publish
        // turn first, then that lock.
        self.wake_publish_waiters();
    }

    /// The read horizon: the oldest epoch any reader may still read at, now
    /// or from now on. Garbage collection keeps every version a reader at
    /// the horizon or later can see (`gc_versions`, `collect`).
    ///
    /// It is the oldest of: the snapshot of every open transaction and of
    /// every committing one (a commit whose publish has not come, or never
    /// comes after an unknown outcome, still reads at its snapshot and its
    /// versions stay pending), the epoch of every read pinned outside a
    /// transaction (`pin_read`: every read statement outside a transaction
    /// pins for its run), and the published epoch, at which every reader
    /// that starts from now on reads.
    ///
    /// Each of them is counted by the epoch it reads at as it begins (a
    /// transaction's snapshot until its commit is published or it aborts,
    /// a pin until it is dropped), so this reads the oldest counted epoch
    /// and costs the same however many transactions finished. A reader's
    /// epoch is read and counted under the lock this reads them under, so a
    /// reader that starts meanwhile is either counted or reads at the
    /// published epoch read here or a later one.
    #[must_use]
    pub fn read_horizon(&self) -> EpochId {
        let readers = self.reads.epochs.lock();
        let published = self.published_epoch.load(Ordering::Acquire);
        EpochId::new(ReadPins::oldest(&readers).map_or(published, |oldest| oldest.min(published)))
    }

    /// Whether the bytes a write replaces matter to anyone now: a read holds an epoch (a
    /// transaction's snapshot, a snapshot, a view, a stream), or the database keeps history. A
    /// direct call that commits at once builds its before-images to count them only then (see
    /// [`add_replaced`](Self::add_replaced)); a reader that comes later reads after it.
    pub(crate) fn counts_replaced(&self) -> bool {
        self.history.is_on() || !self.reads.epochs.lock().pinned.is_empty()
    }

    /// Adds `lpg` bytes a direct call that commits at once replaced to the ledger, before its
    /// epoch is published, as a publish adds a commit's: the readers that hold older epochs
    /// count them, and one held past the snapshot retention limit is noticed. The call has no
    /// result to carry the notice, so it goes to `tracing` only.
    pub(crate) fn add_replaced(&self, lpg: u64) {
        if lpg == 0 {
            return;
        }
        let held = {
            let registry = self.transactions.read();
            let mut readers = self.reads.epochs.lock();
            readers.replaced.lpg = readers.replaced.lpg.saturating_add(lpg);
            self.bound_snapshots(&registry.open, &mut readers)
        };
        if let Some(held) = held {
            grafeo_common::grafeo_warn!("{}", held.message());
        }
    }

    /// Each time-index entry's epoch and the bytes the commits replaced up to it, for tests.
    #[cfg(test)]
    pub(crate) fn history_totals(&self) -> Vec<(u64, u64)> {
        self.history.totals()
    }

    /// The bytes of the labeled property graph's committed state the commits replaced so far,
    /// for tests.
    #[cfg(test)]
    pub(crate) fn replaced_lpg(&self) -> u64 {
        self.reads.epochs.lock().replaced.lpg
    }

    /// Sets the snapshot retention limit: how many bytes of committed state the commits since
    /// the oldest open transaction's snapshot may replace before that transaction expires
    /// (`GRAFEO-T012` from its next statement on), so a forgotten transaction cannot hold old
    /// versions without bound. `u64::MAX` sets no bound.
    pub fn set_snapshot_retention_limit(&self, bytes: u64) {
        self.snapshot_retention_limit
            .store(bytes, Ordering::Relaxed);
    }

    /// The snapshot retention limit (see
    /// [`set_snapshot_retention_limit`](Self::set_snapshot_retention_limit)).
    #[must_use]
    pub fn snapshot_retention_limit(&self) -> u64 {
        self.snapshot_retention_limit.load(Ordering::Relaxed)
    }

    /// The snapshot reader of an open transaction, for its session to keep.
    fn snapshot_reader(&self, transaction_id: TransactionId) -> Option<Arc<SnapshotReader>> {
        self.transactions
            .read()
            .open
            .get(&transaction_id)
            .filter(|info| info.state == TransactionState::Active)
            .map(|info| Arc::clone(&info.reader))
    }

    /// A read at the snapshot of transaction `transaction_id`, for a caller that runs plans in
    /// a transaction it does not own.
    ///
    /// # Errors
    ///
    /// [`TransactionError::SnapshotExpired`] once the snapshot expired. `None` for a
    /// transaction that is not active (ended, or committing), whose snapshot holds nothing for
    /// a new read.
    pub(crate) fn read_in(&self, transaction_id: TransactionId) -> Result<Option<SnapshotRead>> {
        self.snapshot_reader(transaction_id)
            .map(|reader| reader.read())
            .transpose()
    }

    /// The snapshot retention limit's check, at each publish with both locks held: while the
    /// bytes the commits since the oldest reader's epoch replaced exceed the limit, the open
    /// transactions reading at that epoch expire. Only what the reader adds beyond what is kept
    /// anyway counts: a history window keeps the labeled property graph's versions, so a reader
    /// inside it counts only the RDF overlay's. A reader that cannot expire (a pin outside a
    /// transaction, a private or statement transaction) is returned, once per epoch, for the
    /// publisher to report after its locks. O(1) unless over the limit.
    #[must_use = "the warning about a reader that cannot expire is the caller's, after its locks"]
    fn bound_snapshots(
        &self,
        transactions: &FxHashMap<TransactionId, TransactionInfo>,
        readers: &mut Readers,
    ) -> Option<HeldPastLimit> {
        let limit = self.snapshot_retention_limit.load(Ordering::Relaxed);
        if limit == u64::MAX {
            return None;
        }
        let mut held = None;
        loop {
            let Some(front) = readers.pinned.front() else {
                return held;
            };
            let (epoch, at) = (front.epoch, front.replaced_at);
            if readers
                .handled_through
                .is_some_and(|handled| handled >= epoch)
            {
                return held;
            }
            let rdf = readers.replaced.rdf.saturating_sub(at.rdf);
            let lpg = readers.replaced.lpg.saturating_sub(at.lpg);
            // The common case, before the history window is asked.
            if lpg.saturating_add(rdf) <= limit {
                return held;
            }
            let in_window = self
                .history
                .retention_horizon()
                .is_some_and(|start| start <= epoch);
            let kept = if in_window {
                rdf
            } else {
                lpg.saturating_add(rdf)
            };
            if kept <= limit {
                return held;
            }
            let mut expired = 0usize;
            for info in transactions.values() {
                if info.state == TransactionState::Active
                    && info.reader.expirable
                    && info.start_epoch.as_u64() == epoch
                    && !info.reader.is_expired()
                {
                    info.reader.expire(readers, kept, limit);
                    expired += 1;
                }
            }
            if expired == 0 {
                held = Some(HeldPastLimit { epoch, kept, limit });
            }
            readers.handled_through = Some(epoch);
            if readers.pinned.front().map(|pinned| pinned.epoch) == Some(epoch) {
                return held;
            }
        }
    }

    /// Pins a read outside any transaction at the published epoch, for as
    /// long as the returned pin lives: the [read horizon](Self::read_horizon)
    /// stays at its epoch or before. The read may read at that epoch or at
    /// any later one it loads after the pin (the published epoch only moves
    /// forward). Read statements outside a transaction pin for their run
    /// (#412 decision Q6), so collecting at the horizon never takes a version
    /// such a statement still reads.
    pub(crate) fn pin_read(&self) -> ReadPin {
        let epoch = self.reads.hold_published(&self.published_epoch, false);
        ReadPin {
            pins: Arc::clone(&self.reads),
            epoch,
        }
    }

    /// Pins a read at `epoch`, a time-travel read's (one after the
    /// published epoch reads at the published one), for as long as the
    /// returned pin lives: the collect horizon stays at its epoch or before,
    /// so the versions it reads stay while it runs.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidValue`] for an epoch before the history floor (see
    /// [`history_floor`](Self::history_floor)): "history before ... is not
    /// retained". The floor is read under the lock the collect horizon moves
    /// it under, so a read pinned here is never collected past.
    pub(crate) fn pin_read_at(&self, epoch: EpochId) -> Result<ReadPin> {
        let mut readers = self.reads.epochs.lock();
        self.reads.wait_for_replacement(&mut readers);
        self.pin_at_in(&mut readers, epoch)
    }

    /// Pins the read of a checkpoint or a copy at `epoch`, as [`pin_read_at`](Self::pin_read_at),
    /// with commits held: it never waits for a change that replaces the stores whole (see
    /// [`hold_idle`](Self::hold_idle)). Such a change replaces them only while it holds commits
    /// itself, so with commits held none is halfway, and the restore's own checkpoint pins here.
    ///
    /// # Errors
    ///
    /// As [`pin_read_at`](Self::pin_read_at).
    #[cfg(feature = "grafeo-file")]
    pub(crate) fn pin_checkpoint_at(
        &self,
        epoch: EpochId,
        _commits: &CommitsHeld<'_>,
    ) -> Result<ReadPin> {
        self.pin_at_in(&mut self.reads.epochs.lock(), epoch)
    }

    /// Pins a read at `epoch` (see [`pin_read_at`](Self::pin_read_at)), with the readers' lock
    /// held.
    fn pin_at_in(&self, readers: &mut Readers, epoch: EpochId) -> Result<ReadPin> {
        if epoch < self.history.floor() {
            return Err(self.history.before_floor(epoch));
        }
        let epoch = self.readable_epoch(epoch);
        ReadPins::hold_at(readers, epoch.as_u64());
        readers.reads += 1;
        Ok(ReadPin {
            pins: Arc::clone(&self.reads),
            epoch,
        })
    }

    /// The horizon garbage collection runs at: the older of the
    /// [read horizon](Self::read_horizon) and the start of the database's
    /// retention window (none while history is off, so it is the read
    /// horizon then, as without history). Every version a reader at it or
    /// later reads is kept, and it becomes the history floor: from now on a
    /// read before it is refused. The engine's collects run at it.
    #[must_use]
    pub fn collect_horizon(&self) -> EpochId {
        let readers = self.reads.epochs.lock();
        self.collect_horizon_in(&readers)
    }

    /// [`collect_horizon`](Self::collect_horizon) with the readers' lock held.
    fn collect_horizon_in(&self, readers: &Readers) -> EpochId {
        let published = self.published_epoch.load(Ordering::Acquire);
        let read = ReadPins::oldest(readers).map_or(published, |oldest| oldest.min(published));
        let horizon = match self.history.retention_horizon() {
            Some(start) => read.min(start),
            None => read,
        };
        let horizon = EpochId::new(horizon);
        self.history.collected_at(horizon);
        horizon
    }

    /// The oldest epoch a read may still ask for: the newest collect
    /// horizon (see [`collect_horizon`](Self::collect_horizon)). A
    /// time-travel read before it is refused.
    #[must_use]
    pub fn history_floor(&self) -> EpochId {
        self.history.floor()
    }

    /// Sets how long the database keeps history, from now on (see
    /// [`HistoryRetention`]): turned on, the commit times are recorded from
    /// the published epoch on; turned off, the time index goes, and the
    /// collect horizon is the read horizon again.
    pub fn set_history_retention(&self, retention: HistoryRetention) {
        self.history
            .set_retention(retention, self.published_epoch.load(Ordering::Acquire));
    }

    /// Sets how many bytes of replaced data the history window may keep (`None`: no limit): once
    /// the commits since its start replaced more, the start moves forward, whichever of time or
    /// bytes comes first (see [`Config::with_history_retention_limit`](crate::Config::with_history_retention_limit)).
    pub(crate) fn set_history_retention_limit(&self, bytes: Option<u64>) {
        self.history.set_retention_limit(bytes);
    }

    /// How long the database keeps history.
    #[must_use]
    pub fn history_retention(&self) -> HistoryRetention {
        self.history.retention()
    }

    /// The epoch a read at `timestamp` reads at: the last one committed at
    /// or before it (at the resolution of a millisecond).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidValue`] while the database keeps no history, and for a
    /// time before what it keeps.
    pub fn epoch_at(&self, timestamp: Timestamp) -> Result<EpochId> {
        self.history.epoch_at(timestamp)
    }

    /// Now on the clock commit times come from, in milliseconds since the
    /// Unix epoch.
    pub(crate) fn history_now(&self) -> u64 {
        self.history.now()
    }

    /// The time a write outside a transaction that commits at once takes,
    /// under the commit lock, while the database keeps history: its WAL
    /// group carries it, and [`record_commit_time`](Self::record_commit_time)
    /// records it once its epoch is published.
    pub(crate) fn commit_time(&self) -> Option<u64> {
        self.history.commit_time()
    }

    /// Records that `epoch` committed at `millis`: a write outside a
    /// transaction, once published, and a replayed WAL group. Nothing while
    /// history is off.
    /// It replaced `lpg` bytes of the labeled property graph's committed state, which the
    /// history window keeps: 0 for a replayed group, which the end of the open counts.
    pub(crate) fn record_commit_time(&self, epoch: EpochId, millis: u64, lpg: u64) {
        self.history.record_replacing(epoch, millis, lpg);
    }

    /// What a checkpoint writes of the history, `None` while it is off.
    pub(crate) fn history_image(&self) -> Option<HistoryImage> {
        self.history.image()
    }

    /// Sets the retention an open starts with, before it loads the file and
    /// replays the log (see [`anchor_history`](Self::anchor_history)).
    pub(crate) fn open_history(&self, retention: HistoryRetention) {
        self.history.open(retention);
    }

    /// Takes in what a load found of the history: the image of a file's
    /// `TIME_INDEX` section, or, for a file without one, the epoch of its
    /// checkpoint, before which nothing is retained.
    pub(crate) fn restore_history_or_floor(
        &self,
        image: Option<HistoryImage>,
        checkpoint: EpochId,
    ) {
        match image {
            Some(image) => self.history.restore(image),
            None => self.history.collected_at(checkpoint),
        }
    }

    /// After a snapshot arrived (an import, a restore): the history starts
    /// at the published epoch, now, so a read before it is refused under
    /// every retention setting (a snapshot carries no history). Taken under
    /// the read pins' lock, so no read pins below the new floor meanwhile.
    pub(crate) fn restart_history(&self) {
        let _readers = self.reads.epochs.lock();
        self.history
            .restart(EpochId::new(self.published_epoch.load(Ordering::Acquire)));
    }

    /// The epoch a checkpoint writes history from, while the database keeps it: the collect
    /// horizon now (the later of the window's starts by time and by bytes, never past an open
    /// reader), which the floor moves to first, so the time index the checkpoint writes starts
    /// where the history it writes does. `None` while history is off.
    pub(crate) fn history_checkpoint_floor(&self) -> Option<EpochId> {
        self.history.is_on().then(|| {
            // The floor moves to the horizon when it is later: the later of the two.
            let horizon = self.collect_horizon();
            horizon.max(self.history.floor())
        })
    }

    /// [`history_checkpoint_floor`](Self::history_checkpoint_floor) with a read pinned there,
    /// under the one lock the floor moves under, so no collect passes it before the image is
    /// read (`None` while history is off). The pin's epoch is where the history starts.
    #[cfg(feature = "grafeo-file")]
    pub(crate) fn pin_history_checkpoint(&self) -> Option<ReadPin> {
        if !self.history.is_on() {
            return None;
        }
        let mut readers = self.reads.epochs.lock();
        let floor = self.collect_horizon_in(&readers).max(self.history.floor());
        ReadPins::hold_at(&mut readers, floor.as_u64());
        readers.reads += 1;
        Some(ReadPin {
            pins: Arc::clone(&self.reads),
            epoch: floor,
        })
    }

    /// Whether the history's time index counts bytes: history is on and a retention limit is
    /// set (see [`set_history_retention_limit`](Self::set_history_retention_limit)).
    pub(crate) fn history_counts_bytes(&self) -> bool {
        self.history.counts_bytes()
    }

    /// Counts what the store keeps of replaced state (`kept`, see
    /// `RowGroupStore::kept_superseded`) into the history's time index, at the end of an open:
    /// the history a load restored and a replay kept counts as the commits that made it did.
    pub(crate) fn count_kept_history(&self, kept: Vec<(u64, usize)>) {
        self.history.count_kept(kept);
    }

    /// At the end of an open or a new database: a history that is on and
    /// holds no commit at or after the published epoch starts there, now,
    /// so a read at a time from now on finds its epoch.
    pub(crate) fn anchor_history(&self) {
        self.history
            .anchor(self.published_epoch.load(Ordering::Acquire));
    }

    /// Returns the number of active transactions.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.transactions
            .read()
            .open
            .values()
            .filter(|info| info.state == TransactionState::Active)
            .count()
    }

    /// Trims the committed-claims index to the transactions open now (see
    /// `trim_committed`): a finished transaction leaves the manager at its
    /// publish or abort, which trim already, so this drops only what a reader
    /// that ended since let go (a read pin outside a transaction).
    ///
    /// Returns the number of log entries dropped from the index.
    pub fn gc(&self) -> usize {
        let dropped = {
            let mut registry = self.transactions.write();
            let readers = self.reads.epochs.lock();
            let Registry { open, claims } = &mut *registry;
            Self::trim_committed(open, claims, &readers)
        };
        dropped.len()
    }

    /// Returns the last assigned transaction ID.
    ///
    /// Returns `None` if no transactions have been started yet.
    #[must_use]
    pub fn last_assigned_transaction_id(&self) -> Option<TransactionId> {
        let next = self.next_transaction_id.load(Ordering::Relaxed);
        if next > 1 {
            Some(TransactionId::new(next - 1))
        } else {
            None
        }
    }

    /// The open transactions in the registry (tests).
    #[cfg(test)]
    pub(crate) fn registered(&self) -> usize {
        self.transactions.read().open.len()
    }

    /// The sizes of the claim indexes (tests): the open index's entries (entities, keys and
    /// patterns), the committed index's (entities, keys, patterns and triples), and the
    /// commits in its log.
    #[cfg(test)]
    pub(crate) fn claim_index_sizes(&self) -> (usize, usize, usize) {
        self.transactions.read().claims.sizes()
    }

    /// The conflict horizon as the ledger caches it (`Readers::committable_from`) and as a walk
    /// of the ledger finds it, the first epoch with a committable snapshot (tests).
    #[cfg(test)]
    pub(crate) fn committable_horizons(&self) -> (Option<u64>, Option<u64>) {
        let readers = self.reads.epochs.lock();
        (
            readers.committable_from,
            Readers::first_committable(&readers),
        )
    }

    /// The epoch the committed-claims index trims to (tests, see `conflict_horizon_in`).
    #[cfg(test)]
    pub(crate) fn conflict_horizon(&self) -> Option<EpochId> {
        Self::conflict_horizon_in(&self.reads.epochs.lock()).map(EpochId::new)
    }

    /// The commit epoch of a committing transaction (tests): `None` before its commit
    /// starts, after it is taken back, and once it is published.
    #[cfg(test)]
    pub(crate) fn commit_epoch(&self, transaction_id: TransactionId) -> Option<EpochId> {
        self.transactions
            .read()
            .open
            .get(&transaction_id)
            .and_then(|info| info.commit_epoch)
    }
}

impl Default for TransactionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "claim_index_tests.rs"]
mod claim_index_tests;

#[cfg(test)]
#[path = "claim_trim_tests.rs"]
mod claim_trim_tests;

#[cfg(test)]
mod tests {
    use super::*;

    /// A direct read inside a direct read on one thread fails the debug check:
    /// the replace gate prefers a waiting writer, so it would hang a restore.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "a direct read inside a direct read")]
    fn a_nested_direct_read_fails_the_debug_check() {
        let mgr = TransactionManager::new();
        let _outer = mgr.hold_direct_read();
        let _inner = mgr.hold_direct_read();
    }

    /// Direct reads one after another on one thread, and side by side on two,
    /// pass the debug check.
    #[test]
    fn direct_reads_in_turn_and_on_two_threads_pass() {
        let mgr = Arc::new(TransactionManager::new());
        drop(mgr.hold_direct_read());
        let held = mgr.hold_direct_read();
        let other = Arc::clone(&mgr);
        std::thread::spawn(move || drop(other.hold_direct_read()))
            .join()
            .unwrap();
        drop(held);
    }

    /// Commits a transaction of its own that replaced `rdf` bytes of RDF state.
    fn commit_replacing(mgr: &TransactionManager, rdf: u64) {
        let writer = mgr.begin();
        mgr.start_commit(writer)
            .unwrap()
            .written()
            .complete_replacing(|| Ok(rdf))
            .unwrap();
    }

    fn expired(result: Result<Option<SnapshotRead>>) -> (u64, u64) {
        match result {
            Err(Error::Transaction(TransactionError::SnapshotExpired(expired))) => {
                (expired.kept, expired.limit)
            }
            other => panic!("expected the snapshot to expire, got {other:?}"),
        }
    }

    /// An open transaction's snapshot expires once the commits since it began replaced more
    /// than the limit: its next read and its commit fail with `GRAFEO-T012` naming both
    /// numbers, and its snapshot no longer holds the read horizon. Under the limit it reads on.
    #[test]
    fn a_snapshot_expires_once_the_commits_since_it_replaced_past_the_limit() {
        let mgr = TransactionManager::new();
        mgr.set_snapshot_retention_limit(1000);
        let reader = mgr.begin();
        let at = mgr.read_horizon();
        commit_replacing(&mgr, 600);
        assert!(mgr.read_in(reader).unwrap().is_some(), "under the limit");
        assert_eq!(mgr.read_horizon(), at, "the reader holds the horizon");
        commit_replacing(&mgr, 600);
        assert_eq!(expired(mgr.read_in(reader)), (1200, 1000));
        assert_eq!(
            mgr.read_horizon(),
            mgr.current_epoch(),
            "released: the horizon moved past it"
        );
        let Err(error) = mgr.start_commit(reader) else {
            panic!("an expired transaction commits");
        };
        assert_eq!(error.error_code().as_str(), "GRAFEO-T012");
        mgr.abort(reader).unwrap();
        assert_eq!(mgr.read_horizon(), mgr.current_epoch(), "released once");
    }

    /// A read in progress keeps an expired snapshot until it ends: the horizon stays, a new
    /// read fails, and the end of the last read releases it and runs the reader-end work.
    #[test]
    fn an_expiry_waits_for_the_read_in_progress() {
        let mgr = TransactionManager::new();
        // The reader-end work (pruning the RDF overlay) exists with the triple store.
        #[cfg(feature = "triple-store")]
        let ended = {
            let ended = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let counted = Arc::clone(&ended);
            mgr.on_reader_end(move || {
                counted.fetch_add(1, Ordering::Relaxed);
            });
            ended
        };
        mgr.set_snapshot_retention_limit(1000);
        let reader = mgr.begin();
        let at = mgr.read_horizon();
        let read = mgr.read_in(reader).unwrap().unwrap();
        commit_replacing(&mgr, 2000);
        assert_eq!(mgr.read_horizon(), at, "the read in progress holds it");
        expired(mgr.read_in(reader));
        #[cfg(feature = "triple-store")]
        let before = ended.load(Ordering::Relaxed);
        drop(read);
        assert_eq!(mgr.read_horizon(), mgr.current_epoch());
        #[cfg(feature = "triple-store")]
        assert_eq!(
            ended.load(Ordering::Relaxed),
            before + 1,
            "the release runs the reader-end work"
        );
        mgr.abort(reader).unwrap();
    }

    /// The readers at the oldest epoch expire together; one that began later, under the limit
    /// for its own epoch, reads on.
    #[test]
    fn the_readers_at_the_oldest_epoch_expire_together() {
        let mgr = TransactionManager::new();
        mgr.set_snapshot_retention_limit(1000);
        let first = mgr.begin();
        let second = mgr.begin();
        commit_replacing(&mgr, 600);
        let later = mgr.begin();
        commit_replacing(&mgr, 600);
        expired(mgr.read_in(first));
        expired(mgr.read_in(second));
        assert!(mgr.read_in(later).unwrap().is_some(), "600 since it began");
        for transaction in [first, second, later] {
            mgr.abort(transaction).unwrap();
        }
    }

    /// What cannot fail never expires: a private transaction (one direct call) and a read pin
    /// outside a transaction, at the oldest epoch, past the limit. A transaction that began
    /// with them expires; the pin keeps the horizon.
    #[test]
    fn a_private_transaction_and_a_pin_never_expire() {
        let mgr = TransactionManager::new();
        mgr.set_snapshot_retention_limit(1000);
        let pin = mgr.pin_read();
        let (private, _changes) = mgr.begin_private().unwrap();
        let reader = mgr.begin();
        commit_replacing(&mgr, 5000);
        expired(mgr.read_in(reader));
        assert_eq!(mgr.read_horizon(), pin.epoch(), "the pin holds the horizon");
        mgr.start_commit(private)
            .expect("a private transaction never expires")
            .complete()
            .unwrap();
        drop(pin);
        mgr.abort(reader).unwrap();
        assert_eq!(mgr.read_horizon(), mgr.current_epoch());
    }

    /// Reads that start and end on several threads while commits expire their snapshot: the
    /// count is released exactly once. A pin at the same epoch catches a second release (the
    /// horizon would pass the pin while it lives).
    #[test]
    fn reads_racing_an_expiry_release_the_snapshot_once() {
        for _ in 0..20 {
            let mgr = Arc::new(TransactionManager::new());
            mgr.set_snapshot_retention_limit(1000);
            let reader = mgr.begin();
            let pin = mgr.pin_read();
            let readers: Vec<_> = (0..4)
                .map(|_| {
                    let mgr = Arc::clone(&mgr);
                    std::thread::spawn(move || {
                        for _ in 0..2000 {
                            match mgr.read_in(reader) {
                                Ok(Some(read)) => drop(read),
                                Ok(None) => return,
                                Err(_) => {}
                            }
                        }
                    })
                })
                .collect();
            commit_replacing(&mgr, 2000);
            for thread in readers {
                thread.join().unwrap();
            }
            expired(mgr.read_in(reader));
            mgr.abort(reader).unwrap();
            assert_eq!(
                mgr.read_horizon(),
                pin.epoch(),
                "the pin still holds its epoch"
            );
            drop(pin);
            assert_eq!(mgr.read_horizon(), mgr.current_epoch());
        }
    }

    /// With no limit set (the manager's own default), nothing expires.
    #[test]
    fn without_a_limit_no_snapshot_expires() {
        let mgr = TransactionManager::new();
        let reader = mgr.begin();
        commit_replacing(&mgr, u64::MAX / 2);
        assert!(mgr.read_in(reader).unwrap().is_some());
        mgr.abort(reader).unwrap();
    }

    /// The RDF overlay keeps nothing for history: a reader inside a history window still
    /// expires on RDF bytes past the limit. (That the window keeps a reader on the labeled
    /// property graph's bytes, which take a real write, is the session test
    /// `a_reader_inside_the_history_window_keeps_its_snapshot`.)
    #[test]
    fn a_history_window_does_not_keep_the_rdf_overlay_unbounded() {
        let mgr = TransactionManager::new();
        mgr.set_history_retention(HistoryRetention::Forever);
        mgr.set_snapshot_retention_limit(1000);
        let reader = mgr.begin();
        commit_replacing(&mgr, 2000);
        assert_eq!(expired(mgr.read_in(reader)), (2000, 1000));
        mgr.abort(reader).unwrap();
    }

    /// The pattern set answers as a set past its inline fingerprints: each
    /// held once, removed from either part, and its length both parts.
    #[test]
    fn the_pattern_set_holds_each_fingerprint_once_inline_or_spilled() {
        let mut set = PatternSet::default();
        assert!(set.is_empty());
        for fingerprint in 0..10 {
            assert!(set.insert(fingerprint));
            assert!(!set.insert(fingerprint), "{fingerprint} twice");
        }
        assert_eq!(set.len(), 10);
        set.remove(1);
        set.remove(8);
        assert_eq!(set.len(), 8);
        assert!(!set.contains(1) && !set.contains(8));
        let mut held: Vec<u64> = set.iter().collect();
        held.sort_unstable();
        assert_eq!(held, [0, 2, 3, 4, 5, 6, 7, 9]);
        // A freed inline place takes the next one.
        assert!(set.insert(1));
        assert!(set.contains(1));
        assert_eq!(set.len(), 9);
    }

    #[cfg(feature = "gql")]
    mod failed_storage_undo {
        use super::*;
        use crate::{Config, GrafeoDB};
        use arcstr::ArcStr;
        use grafeo_common::change::{Change, ChangeMark, DataOp};
        use grafeo_common::types::ReadView;
        use grafeo_core::graph::apply::{Applied, ApplyError, BackendTransaction, Writer};
        use grafeo_core::graph::lpg::{Edge, Node};
        use grafeo_core::graph::rowgroup::RowGroupStore;
        use grafeo_core::graph::{Direction, GraphBackend, GraphStore};
        use grafeo_core::statistics::Statistics;
        use std::sync::atomic::AtomicBool;

        #[derive(Default)]
        struct RefusesUndo {
            rows: RowGroupStore,
            refuse: AtomicBool,
        }

        fn seed_claims(mgr: &TransactionManager, tx: TransactionId, offset: u64) {
            mgr.record_write(tx, NodeId::new(offset)).unwrap();
            mgr.record_delete(tx, NodeId::new(offset + 1)).unwrap();
            let endpoint = GraphEntity::from(NodeId::new(offset + 2));
            mgr.record_endpoints(tx, [endpoint.clone(), endpoint])
                .unwrap();
            mgr.record_key(tx, claim_key(i64::try_from(offset).unwrap()))
                .unwrap();
        }

        #[test]
        fn failed_savepoint_undo_retains_every_claim_and_poisons() {
            let store = Arc::new(RefusesUndo::default());
            let db = GrafeoDB::with_store(store.clone(), Config::in_memory()).unwrap();
            let mut session = db.session();
            session.begin_transaction().unwrap();
            session.execute("INSERT (:Item {v: 0})").unwrap();
            let tx = session.inner().unwrap().current_transaction_id().unwrap();
            let mgr = Arc::clone(&db.live().unwrap().transaction_manager);
            seed_claims(&mgr, tx, 100);
            session.savepoint("before").unwrap();
            session.execute("INSERT (:Item {v: 1})").unwrap();
            seed_claims(&mgr, tx, 200);
            let before = {
                let txns = mgr.transactions.read();
                let info = &txns.open[&tx];
                (
                    info.write_set.clone(),
                    info.delete_set.clone(),
                    info.endpoint_set.clone(),
                    info.key_set.clone(),
                    info.claim_undo.as_ref().unwrap().len(),
                )
            };
            store.refuse.store(true, Ordering::Relaxed);
            let error = session.rollback_to_savepoint("before").unwrap_err();
            assert!(
                error.to_string().contains("injected undo failure"),
                "{error}"
            );
            let txns = mgr.transactions.read();
            let info = &txns.open[&tx];
            assert_eq!(info.state, TransactionState::Active);
            assert_eq!(info.write_set, before.0);
            assert_eq!(info.delete_set, before.1);
            assert_eq!(info.endpoint_set, before.2);
            assert_eq!(info.key_set, before.3);
            assert_eq!(info.claim_undo.as_ref().unwrap().len(), before.4);
            assert!(mgr.has_incomplete_commit());
            drop(txns);
            assert!(
                session.rollback_to_savepoint("before").is_err(),
                "a retry after failed storage undo must not release its claims"
            );
            let txns = mgr.transactions.read();
            let info = &txns.open[&tx];
            assert_eq!(info.write_set, before.0);
            assert_eq!(info.delete_set, before.1);
            assert_eq!(info.endpoint_set, before.2);
            assert_eq!(info.key_set, before.3);
        }

        #[test]
        fn failed_statement_undo_retains_its_write_claim_and_poisons() {
            let store = Arc::new(RefusesUndo::default());
            let db = GrafeoDB::with_store(store.clone(), Config::in_memory()).unwrap();
            let mut session = db.session();
            session.execute("INSERT (:Item {v: 0})").unwrap();
            session.begin_transaction().unwrap();
            let tx = session.inner().unwrap().current_transaction_id().unwrap();
            let mgr = Arc::clone(&db.live().unwrap().transaction_manager);
            seed_claims(&mgr, tx, 100);
            let earlier_writes = mgr.get_write_set(tx).unwrap();
            store.refuse.store(true, Ordering::Relaxed);
            let error = session
                .execute("MATCH (n:Item) SET n.v = 1 RETURN 1 / (n.v - n.v)")
                .unwrap_err();
            assert!(
                error.to_string().contains("injected undo failure"),
                "{error}"
            );
            let txns = mgr.transactions.read();
            let info = &txns.open[&tx];
            assert_eq!(info.state, TransactionState::Active);
            assert!(
                !info.write_set.is_subset(&earlier_writes),
                "the property update's undo failed, so its new write claim must remain"
            );
            assert!(
                info.write_set
                    .contains(&GraphEntity::from(NodeId::new(100)))
            );
            assert!(
                info.delete_set
                    .contains(&GraphEntity::from(NodeId::new(101)))
            );
            assert!(
                info.endpoint_set
                    .contains(&GraphEntity::from(NodeId::new(102)))
            );
            assert!(info.key_set.contains(&claim_key(100)));
            assert!(mgr.has_incomplete_commit());
        }
        impl GraphStore for RefusesUndo {
            fn node(&self, id: NodeId, view: ReadView) -> Result<Option<Node>> {
                self.rows.node(id, view)
            }
            fn edge(&self, id: EdgeId, view: ReadView) -> Result<Option<Edge>> {
                self.rows.edge(id, view)
            }
            fn edges(
                &self,
                node: NodeId,
                direction: Direction,
                types: &[ArcStr],
                view: ReadView,
            ) -> Result<Vec<(NodeId, EdgeId)>> {
                self.rows.edges(node, direction, types, view)
            }
            fn has_backward_adjacency(&self) -> bool {
                self.rows.has_backward_adjacency()
            }
            fn node_ids(&self, view: ReadView) -> Result<Vec<NodeId>> {
                self.rows.node_ids(view)
            }
            fn all_labels(&self) -> Result<Vec<String>> {
                self.rows.all_labels()
            }
            fn all_edge_types(&self) -> Result<Vec<String>> {
                self.rows.all_edge_types()
            }
            fn all_property_keys(&self) -> Result<Vec<String>> {
                self.rows.all_property_keys()
            }
            fn statistics(&self) -> Arc<Statistics> {
                self.rows.statistics()
            }
            fn committed_epoch(&self) -> EpochId {
                self.rows.committed_epoch()
            }
        }

        impl GraphBackend for RefusesUndo {
            fn reserve_node_ids(
                &self,
                count: u64,
            ) -> std::result::Result<std::ops::Range<u64>, ApplyError> {
                self.rows.reserve_node_ids(count)
            }
            fn reserve_edge_ids(
                &self,
                count: u64,
            ) -> std::result::Result<std::ops::Range<u64>, ApplyError> {
                self.rows.reserve_edge_ids(count)
            }
            fn begin(
                &self,
                transaction: TransactionId,
                snapshot: EpochId,
            ) -> std::result::Result<BackendTransaction, ApplyError> {
                self.rows.begin(transaction, snapshot)
            }
            fn apply(
                &self,
                op: &DataOp,
                writer: Writer,
            ) -> std::result::Result<Applied, ApplyError> {
                self.rows.apply(op, writer)
            }
            fn commit(
                &self,
                transaction: BackendTransaction,
                changes: &mut dyn Iterator<Item = &Change>,
                epoch: EpochId,
            ) -> std::result::Result<(), ApplyError> {
                let changes: Vec<&Change> = changes.collect();
                self.rows
                    .commit(transaction, &mut changes.into_iter(), epoch)
            }
            fn rollback(
                &self,
                transaction: BackendTransaction,
                changes: &mut dyn DoubleEndedIterator<Item = &Change>,
            ) -> std::result::Result<(), ApplyError> {
                let changes: Vec<&Change> = changes.collect();
                self.rows.rollback(transaction, &mut changes.into_iter())
            }
            fn rollback_to(
                &self,
                transaction: BackendTransaction,
                savepoint: ChangeMark,
                changes: &mut dyn DoubleEndedIterator<Item = &Change>,
            ) -> std::result::Result<(), ApplyError> {
                if self.refuse.load(Ordering::Relaxed) {
                    return Err(ApplyError::Refused("injected undo failure".to_string()));
                }
                self.rows.rollback_to(transaction, savepoint, changes)
            }
            fn collect(&self, horizon: EpochId) {
                GraphBackend::collect(&self.rows, horizon);
            }
            fn durable_epoch(&self) -> EpochId {
                self.rows.durable_epoch()
            }
            fn flush(&self, epoch: EpochId) -> std::result::Result<(), ApplyError> {
                self.rows.flush(epoch)
            }
        }
    }

    fn claim_key(value: i64) -> KeyClaim {
        KeyClaim {
            graph: None,
            label: Arc::from("Item"),
            properties: vec!["key".to_string()],
            values: vec![HashableValue(grafeo_common::types::Value::Int64(value))],
        }
    }

    #[test]
    fn claim_marks_preserve_duplicate_roles_and_nested_boundaries() {
        let mgr = TransactionManager::new();
        let tx = mgr.begin();
        let written = GraphEntity::from(NodeId::new(1));
        let deleted = GraphEntity::from(NodeId::new(2));
        let endpoint = GraphEntity::from(NodeId::new(3));
        mgr.record_write(tx, written.clone()).unwrap();
        mgr.record_delete(tx, deleted.clone()).unwrap();
        mgr.record_endpoints(tx, [endpoint.clone(), endpoint.clone()])
            .unwrap();
        mgr.record_key(tx, claim_key(1)).unwrap();
        assert!(mgr.transactions.read().open[&tx].claim_undo.is_none());
        let outer = mgr.mark_claims(tx).unwrap();
        mgr.record_delete(tx, written.clone()).unwrap();
        mgr.record_delete(tx, deleted.clone()).unwrap();
        mgr.record_endpoints(tx, [endpoint.clone(), endpoint.clone()])
            .unwrap();
        mgr.record_key(tx, claim_key(1)).unwrap();
        let inner = mgr.mark_claims(tx).unwrap();
        mgr.record_key(tx, claim_key(2)).unwrap();
        mgr.record_write(tx, NodeId::new(4)).unwrap();
        mgr.restore_claims(tx, inner).unwrap();
        {
            let txns = mgr.transactions.read();
            let info = &txns.open[&tx];
            assert_eq!(
                info.write_set,
                FxHashSet::from_iter([written.clone(), deleted.clone()])
            );
            assert_eq!(
                info.delete_set,
                FxHashSet::from_iter([written.clone(), deleted.clone()])
            );
            assert_eq!(info.endpoint_set, FxHashSet::from_iter([endpoint.clone()]));
            assert_eq!(info.key_set, FxHashSet::from_iter([claim_key(1)]));
        }
        mgr.restore_claims(tx, outer).unwrap();
        let txns = mgr.transactions.read();
        let info = &txns.open[&tx];
        assert_eq!(
            info.write_set,
            FxHashSet::from_iter([written, deleted.clone()])
        );
        assert_eq!(info.delete_set, FxHashSet::from_iter([deleted]));
        assert_eq!(info.endpoint_set, FxHashSet::from_iter([endpoint]));
        assert_eq!(info.key_set, FxHashSet::from_iter([claim_key(1)]));
    }

    #[test]
    fn claim_marks_reject_invalid_positions_without_releasing_claims() {
        let mgr = TransactionManager::new();
        let tx = mgr.begin();
        mgr.mark_claims(tx).unwrap();
        mgr.record_write(tx, NodeId::new(1)).unwrap();
        assert!(mgr.restore_claims(tx, ClaimMark(2)).is_err());
        assert_eq!(
            mgr.get_write_set(tx).unwrap(),
            FxHashSet::from_iter([GraphEntity::from(NodeId::new(1))])
        );
        mgr.abort(tx).unwrap();
        assert!(mgr.mark_claims(tx).is_err());
        assert!(mgr.restore_claims(tx, ClaimMark(0)).is_err());
    }

    /// A finished transaction leaves the registry with its claim log (#673 A3);
    /// a committed one keeps its conflict claims in the committed index while
    /// a transaction that began before it is open: that transaction's commit
    /// still finds the conflict after the other left. An aborted one keeps none.
    #[test]
    fn completed_transactions_drop_undo_history_but_keep_conflict_claims() {
        for commit in [false, true] {
            let mgr = TransactionManager::new();
            let reader = mgr.begin();
            let tx = mgr.begin();
            mgr.mark_claims(tx).unwrap();
            mgr.record_delete(tx, NodeId::new(1)).unwrap();
            mgr.record_key(tx, claim_key(1)).unwrap();
            if commit {
                mgr.commit(tx).unwrap();
            } else {
                mgr.abort(tx).unwrap();
            }
            assert_eq!(mgr.state(tx), None);
            assert_eq!(mgr.claim_log_len(tx), None);
            mgr.record_write(reader, NodeId::new(1)).unwrap();
            let result = mgr.commit(reader);
            assert_eq!(
                is_conflict(&result),
                commit,
                "the write of node 1 against the delete, found in the committed index: {result:?}"
            );
            if result.is_err() {
                mgr.abort(reader).unwrap();
            }
        }
        for commit in [false, true] {
            let mgr = TransactionManager::new();
            let reader = mgr.begin();
            let tx = mgr.begin();
            mgr.record_key(tx, claim_key(1)).unwrap();
            if commit {
                mgr.commit(tx).unwrap();
            } else {
                mgr.abort(tx).unwrap();
            }
            mgr.record_key(reader, claim_key(1)).unwrap();
            let result = mgr.commit(reader);
            assert_eq!(is_conflict(&result), commit, "the key: {result:?}");
            if result.is_err() {
                mgr.abort(reader).unwrap();
            }
        }
        let mgr = TransactionManager::new();
        let tx = mgr.begin();
        mgr.mark_claims(tx).unwrap();
        mgr.record_write(tx, NodeId::new(1)).unwrap();
        mgr.abort_all_active();
        assert_eq!(mgr.state(tx), None);
        assert_eq!(mgr.claim_log_len(tx), None);
    }

    /// `refuse_claimed` sees a key another open transaction holds, alone or
    /// with others open (the open index), and not its own or a released one.
    #[test]
    fn a_claimed_key_is_refused_with_one_or_more_transactions_open() {
        let mgr = TransactionManager::new();
        let holder = mgr.begin();
        mgr.record_key(holder, claim_key(1)).unwrap();
        assert!(is_conflict(&mgr.refuse_claimed(&claim_key(1), None)));
        assert!(mgr.refuse_claimed(&claim_key(1), Some(holder)).is_ok());
        assert!(mgr.refuse_claimed(&claim_key(2), None).is_ok());

        let other = mgr.begin();
        assert!(is_conflict(&mgr.refuse_claimed(&claim_key(1), None)));
        assert!(is_conflict(&mgr.refuse_claimed(&claim_key(1), Some(other))));
        assert!(mgr.refuse_claimed(&claim_key(1), Some(holder)).is_ok());

        mgr.abort(holder).unwrap();
        assert!(mgr.refuse_claimed(&claim_key(1), None).is_ok());
        mgr.abort(other).unwrap();
    }

    /// The trim at the end of an older transaction keeps an entity a commit
    /// after a newer one's start wrote, though an earlier commit wrote it too.
    #[test]
    fn the_trim_keeps_an_entity_a_later_commit_wrote_too() {
        let mgr = TransactionManager::new();
        let older = mgr.begin();
        let before = mgr.begin();
        mgr.record_write(before, NodeId::new(1)).unwrap();
        mgr.commit(before).unwrap();
        let newer = mgr.begin();
        let after = mgr.begin();
        mgr.record_write(after, NodeId::new(1)).unwrap();
        mgr.commit(after).unwrap();

        mgr.abort(older).unwrap();
        assert_eq!(
            mgr.claim_index_sizes().2,
            1,
            "only the commit after `newer` began"
        );
        mgr.record_write(newer, NodeId::new(1)).unwrap();
        assert!(is_conflict(&mgr.commit(newer)));
    }

    /// A commit checks against a commit still in progress (written, not yet
    /// published) exactly as against a published one: a serializable read of
    /// the entity the other writes fails with the same error either way.
    #[test]
    fn a_commit_checks_against_a_commit_in_progress() {
        let mut errors = Vec::new();
        for published in [false, true] {
            let mgr = TransactionManager::new();
            let entity = GraphEntity::from(NodeId::new(1));
            let reader = mgr.begin_with_isolation(IsolationLevel::Serializable);
            let writer = mgr.begin();
            mgr.record_read(reader, entity.clone()).unwrap();
            mgr.record_write(writer, entity).unwrap();
            let in_progress = mgr.start_commit(writer).unwrap().written();
            let in_progress = if published {
                in_progress.complete().unwrap();
                None
            } else {
                Some(in_progress)
            };
            let error = mgr.start_commit(reader).map(|_| ()).unwrap_err();
            assert!(
                matches!(
                    error,
                    Error::Transaction(TransactionError::SerializationFailure(_))
                ),
                "published: {published}: {error}"
            );
            errors.push(error.to_string());
            if let Some(in_progress) = in_progress {
                in_progress.complete().unwrap();
            }
            mgr.abort(reader).unwrap();
        }
        assert_eq!(errors[0], errors[1], "in progress and published alike");
    }

    /// Two transactions that delete one triple: the second commit fails the
    /// same way whether the first is still in progress or published.
    #[cfg(feature = "triple-store")]
    #[test]
    fn a_triple_commit_checks_against_a_commit_in_progress() {
        use grafeo_core::graph::rdf::{Term, Triple};

        let triple = || {
            Triple::new(
                Term::iri("http://example.org/mia"),
                Term::iri("http://example.org/knows"),
                Term::iri("http://example.org/vincent"),
            )
        };
        let mut errors = Vec::new();
        for published in [false, true] {
            let mgr = TransactionManager::new();
            let first = mgr.begin();
            let second = mgr.begin();
            for tx in [first, second] {
                mgr.changes(tx)
                    .unwrap()
                    .record_triple(None, triple(), false)
                    .unwrap();
            }
            let in_progress = mgr.start_commit(first).unwrap().written();
            let in_progress = if published {
                in_progress.complete().unwrap();
                None
            } else {
                Some(in_progress)
            };
            let error = mgr.start_commit(second).map(|_| ()).unwrap_err();
            assert_eq!(
                error.error_code(),
                grafeo_common::utils::error::ErrorCode::TransactionConflict,
                "published: {published}: {error}"
            );
            errors.push(error.to_string());
            if let Some(in_progress) = in_progress {
                in_progress.complete().unwrap();
            }
            mgr.abort(second).unwrap();
        }
        assert_eq!(errors[0], errors[1], "in progress and published alike");
    }

    #[test]
    fn test_begin_commit() {
        let mgr = TransactionManager::new();

        let tx = mgr.begin();
        assert_eq!(mgr.state(tx), Some(TransactionState::Active));

        let commit_epoch = mgr.commit(tx).unwrap();
        // Published: it leaves the registry (#673 A3).
        assert_eq!(mgr.state(tx), None);
        assert!(commit_epoch.as_u64() > 0);
    }

    #[test]
    fn test_begin_abort() {
        let mgr = TransactionManager::new();

        let tx = mgr.begin();
        mgr.abort(tx).unwrap();
        // Aborted: it leaves the registry (#673 A3).
        assert_eq!(mgr.state(tx), None);
    }

    #[test]
    fn test_epoch_advancement() {
        let mgr = TransactionManager::new();

        let initial_epoch = mgr.current_epoch();

        let tx = mgr.begin();
        let commit_epoch = mgr.commit(tx).unwrap();

        assert!(mgr.current_epoch().as_u64() > initial_epoch.as_u64());
        assert!(commit_epoch.as_u64() > initial_epoch.as_u64());
    }

    /// A commit made while another transaction is open leaves the registry at
    /// once, and the open one still conflicts with it at commit (#673 A3, A4).
    #[test]
    fn test_gc_preserves_needed_write_sets() {
        let mgr = TransactionManager::new();

        let tx1 = mgr.begin();
        let tx2 = mgr.begin();
        mgr.record_write(tx1, NodeId::new(1)).unwrap();
        mgr.commit(tx1).unwrap();

        assert_eq!(mgr.active_count(), 1);
        assert_eq!(mgr.gc(), 0, "publish removed it already");
        assert_eq!(mgr.state(tx1), None);
        assert_eq!(mgr.state(tx2), Some(TransactionState::Active));

        // What tx1 wrote is still known to tx2's commit.
        mgr.record_write(tx2, NodeId::new(1)).unwrap();
        assert!(matches!(
            mgr.commit(tx2),
            Err(Error::Transaction(TransactionError::WriteConflict(_)))
        ));
    }

    /// Publish and abort remove a transaction whatever else is open: `gc` finds
    /// nothing left to drop (#673 A3).
    #[test]
    fn test_gc_finds_nothing_publish_left() {
        let mgr = TransactionManager::new();

        let tx1 = mgr.begin();
        mgr.commit(tx1).unwrap();
        let tx2 = mgr.begin();
        mgr.commit(tx2).unwrap();
        let tx3 = mgr.begin();

        assert_eq!(mgr.gc(), 0);
        assert_eq!(mgr.state(tx1), None);
        assert_eq!(mgr.state(tx2), None);
        assert_eq!(mgr.state(tx3), Some(TransactionState::Active));

        mgr.commit(tx3).unwrap();
        assert_eq!(mgr.gc(), 0);
        assert_eq!(mgr.state(tx3), None);
    }

    /// An aborted transaction leaves at its abort, with another open (#673 A3).
    #[test]
    fn test_gc_removes_aborted() {
        let mgr = TransactionManager::new();

        let tx1 = mgr.begin();
        let tx2 = mgr.begin();

        mgr.abort(tx1).unwrap();
        assert_eq!(mgr.state(tx1), None, "gone at its abort");
        assert_eq!(mgr.gc(), 0);
        assert_eq!(mgr.state(tx2), Some(TransactionState::Active));
    }

    #[test]
    fn test_write_tracking() {
        let mgr = TransactionManager::new();

        let tx = mgr.begin();

        // Record writes
        mgr.record_write(tx, NodeId::new(1)).unwrap();
        mgr.record_write(tx, NodeId::new(2)).unwrap();
        mgr.record_write(tx, EdgeId::new(100)).unwrap();

        // Should commit successfully (no conflicts)
        assert!(mgr.commit(tx).is_ok());
    }

    /// With no reader the horizon is the published epoch; with open
    /// transactions, the oldest snapshot, until it ends.
    #[test]
    fn the_read_horizon_is_the_oldest_open_snapshot() {
        let mgr = TransactionManager::new();
        assert_eq!(
            mgr.read_horizon(),
            mgr.current_epoch(),
            "no reader: the published epoch"
        );

        let tx1 = mgr.begin();
        let epoch1 = mgr.start_epoch(tx1).unwrap();
        let tx2 = mgr.begin();
        mgr.commit(tx2).unwrap();
        let _tx3 = mgr.begin();
        assert_eq!(mgr.read_horizon(), epoch1, "the earliest open snapshot");

        mgr.abort(tx1).unwrap();
        assert_eq!(mgr.read_horizon(), mgr.current_epoch());
    }

    /// A read outside any transaction holds the horizon at the epoch it
    /// pinned while later commits publish; two reads pinned at one epoch
    /// hold it until both end, and then it moves to the published epoch.
    #[test]
    fn a_pinned_read_holds_the_read_horizon_back() {
        let mgr = TransactionManager::new();
        let first = mgr.begin();
        let committed = mgr.commit(first).unwrap();
        let read = mgr.pin_read();
        let again = mgr.pin_read();
        assert_eq!((read.epoch(), again.epoch()), (committed, committed));
        for _ in 0..2 {
            let tx = mgr.begin();
            mgr.commit(tx).unwrap();
        }
        assert_eq!(mgr.current_epoch(), EpochId::new(committed.as_u64() + 2));
        assert_eq!(mgr.read_horizon(), committed, "the pinned read");
        drop(read);
        assert_eq!(mgr.read_horizon(), committed, "the second read");
        let later = mgr.pin_read();
        drop(again);
        assert_eq!(mgr.read_horizon(), later.epoch(), "the later read");
        drop(later);
        assert_eq!(mgr.read_horizon(), mgr.current_epoch());
    }

    /// A pin dropped by the unwind of a panic in its read unpins it.
    #[test]
    fn a_read_that_panics_unpins() {
        let mgr = TransactionManager::new();
        let tx = mgr.begin();
        let pinned = mgr.commit(tx).unwrap();
        let read = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _pin = mgr.pin_read();
            let tx = mgr.begin();
            mgr.commit(tx).unwrap();
            assert_eq!(mgr.read_horizon(), pinned);
            panic!("the read fails");
        }));
        assert!(read.is_err());
        assert_eq!(mgr.read_horizon(), mgr.current_epoch());
        assert!(mgr.current_epoch() > pinned);
    }

    /// A committing transaction (its epoch assigned, its publish not come)
    /// still reads at its snapshot, which holds the horizon back after a
    /// later snapshot's commit published; the horizon never passes the
    /// published epoch for an epoch assigned but not published, and moves
    /// once the commit publishes.
    #[test]
    fn a_committing_transaction_holds_the_read_horizon_back() {
        let mgr = TransactionManager::new();
        let slow = mgr.begin();
        let snapshot = mgr.start_epoch(slow).unwrap();
        let fast = mgr.begin();
        mgr.commit(fast).unwrap();
        let published = mgr.current_epoch();
        assert!(published > snapshot);

        let publish = mgr.start_commit(slow).unwrap().written();
        assert_eq!(mgr.state(slow), Some(TransactionState::Committing));
        assert_eq!(mgr.read_horizon(), snapshot, "the committing snapshot");
        let reader = mgr.begin();
        assert_eq!(mgr.start_epoch(reader), Some(published));
        mgr.abort(reader).unwrap();
        assert_eq!(mgr.read_horizon(), snapshot);

        let epoch = publish.epoch();
        publish.complete().unwrap();
        assert_eq!(mgr.read_horizon(), epoch, "published: no reader before it");
    }

    /// While a commit is written and not published, the horizon stays at
    /// the published epoch: a reader that starts meanwhile reads there.
    #[test]
    fn the_read_horizon_stays_at_the_published_epoch_during_a_commit() {
        let mgr = TransactionManager::new();
        let tx = mgr.begin();
        let published = mgr.current_epoch();
        let publish = mgr.start_commit(tx).unwrap().written();
        // The committing transaction's snapshot is the published epoch; the
        // horizon does not reach the epoch it was assigned.
        assert!(publish.epoch() > published);
        assert_eq!(mgr.read_horizon(), published);
        let read = mgr.pin_read();
        assert_eq!(read.epoch(), published, "a read reads what is published");
        publish.complete().unwrap();
        assert_eq!(mgr.read_horizon(), published, "the read is pinned");
        drop(read);
        assert_eq!(mgr.read_horizon(), mgr.current_epoch());
    }

    /// Every way a transaction finishes lets go of its snapshot, before any
    /// `gc` drops it: a commit (private or not), an abort, the abort of every
    /// active one; a commit taken back keeps the snapshot until the
    /// transaction's rollback.
    #[test]
    fn a_finished_transaction_no_longer_holds_the_read_horizon() {
        let mgr = TransactionManager::new();
        let oldest = mgr.begin();
        let snapshot = mgr.start_epoch(oldest).unwrap();
        for round in 0..100 {
            let tx = mgr.begin();
            if round % 2 == 0 {
                mgr.commit(tx).unwrap();
            } else {
                mgr.abort(tx).unwrap();
            }
            let (private, _) = mgr.begin_private().unwrap();
            mgr.commit(private).unwrap();
        }
        assert_eq!(mgr.read_horizon(), snapshot, "the oldest open snapshot");
        mgr.abort(oldest).unwrap();
        assert_eq!(mgr.read_horizon(), mgr.current_epoch(), "no reader left");

        // A commit whose log write failed is taken back: the transaction is
        // active again and still reads at its snapshot.
        let taken_back = mgr.begin();
        let snapshot = mgr.start_epoch(taken_back).unwrap();
        let other = mgr.begin();
        mgr.commit(other).unwrap();
        mgr.start_commit(taken_back).unwrap().abort();
        assert_eq!(mgr.state(taken_back), Some(TransactionState::Active));
        assert_eq!(mgr.read_horizon(), snapshot, "the commit taken back");
        mgr.abort(taken_back).unwrap();
        assert_eq!(mgr.read_horizon(), mgr.current_epoch());

        let first = mgr.begin();
        let held = mgr.start_epoch(first).unwrap();
        let tx = mgr.begin();
        mgr.commit(tx).unwrap();
        let _second = mgr.begin();
        assert_eq!(mgr.read_horizon(), held);
        mgr.abort_all_active();
        assert_eq!(
            mgr.read_horizon(),
            mgr.current_epoch(),
            "all aborted at once"
        );
    }

    #[test]
    fn test_abort_all_active() {
        let mgr = TransactionManager::new();

        let tx1 = mgr.begin();
        let tx2 = mgr.begin();
        let tx3 = mgr.begin();

        mgr.commit(tx1).unwrap();
        // tx2 and tx3 still active

        mgr.abort_all_active();

        // Every one of them has left the registry (#673 A3).
        assert_eq!(mgr.state(tx1), None);
        assert_eq!(mgr.state(tx2), None);
        assert_eq!(mgr.state(tx3), None);
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_start_epoch_snapshot() {
        let mgr = TransactionManager::new();

        // Start epoch for tx1
        let tx1 = mgr.begin();
        let start1 = mgr.start_epoch(tx1).unwrap();

        // Commit tx1, advancing epoch
        mgr.commit(tx1).unwrap();

        // Start tx2 after epoch advanced
        let tx2 = mgr.begin();
        let start2 = mgr.start_epoch(tx2).unwrap();

        // tx2 should have a later start epoch
        assert!(start2.as_u64() > start1.as_u64());
    }

    #[test]
    fn test_write_write_conflict_detection() {
        let mgr = TransactionManager::new();

        // Both transactions start at the same epoch
        let tx1 = mgr.begin();
        let tx2 = mgr.begin();

        // First writer succeeds
        let entity = NodeId::new(42);
        mgr.record_write(tx1, entity).unwrap();

        // Second writer is rejected immediately (first-writer-wins)
        let result = mgr.record_write(tx2, entity);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Write-write conflict"),
            "Expected write-write conflict error"
        );

        // First commit succeeds (no conflict at commit time either)
        let result1 = mgr.commit(tx1);
        assert!(result1.is_ok());
    }

    #[test]
    fn test_commit_epoch_monotonicity() {
        let mgr = TransactionManager::new();

        let mut epochs = Vec::new();

        // Commit multiple transactions and verify epochs are strictly increasing
        for _ in 0..10 {
            let tx = mgr.begin();
            let epoch = mgr.commit(tx).unwrap();
            epochs.push(epoch.as_u64());
        }

        // Verify strict monotonicity
        for i in 1..epochs.len() {
            assert!(
                epochs[i] > epochs[i - 1],
                "Epoch {} ({}) should be greater than epoch {} ({})",
                i,
                epochs[i],
                i - 1,
                epochs[i - 1]
            );
        }
    }

    #[test]
    fn test_concurrent_commits_via_threads() {
        use std::sync::Arc;
        use std::thread;

        let mgr = Arc::new(TransactionManager::new());
        let num_threads = 10;
        let commits_per_thread = 100;

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let mgr = Arc::clone(&mgr);
                thread::spawn(move || {
                    let mut epochs = Vec::new();
                    for _ in 0..commits_per_thread {
                        let tx = mgr.begin();
                        let epoch = mgr.commit(tx).unwrap();
                        epochs.push(epoch.as_u64());
                    }
                    epochs
                })
            })
            .collect();

        let mut all_epochs: Vec<u64> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();

        // All epochs should be unique (no duplicates)
        all_epochs.sort_unstable();
        let unique_count = all_epochs.len();
        all_epochs.dedup();
        assert_eq!(
            all_epochs.len(),
            unique_count,
            "All commit epochs should be unique"
        );

        // Final epoch should equal number of commits
        // num_threads=10, commits_per_thread=100, product is 1000
        // reason: value is non-negative by preceding validation
        #[allow(clippy::cast_sign_loss)]
        let expected_epoch = (num_threads * commits_per_thread) as u64;
        assert_eq!(
            mgr.current_epoch().as_u64(),
            expected_epoch,
            "Final epoch should equal total commits"
        );
    }

    #[test]
    fn test_isolation_level_default() {
        let mgr = TransactionManager::new();

        let tx = mgr.begin();
        assert_eq!(
            mgr.isolation_level(tx),
            Some(IsolationLevel::SnapshotIsolation)
        );
    }

    #[test]
    fn test_isolation_level_explicit() {
        let mgr = TransactionManager::new();

        let transaction_rc = mgr.begin_with_isolation(IsolationLevel::ReadCommitted);
        let transaction_si = mgr.begin_with_isolation(IsolationLevel::SnapshotIsolation);
        let transaction_ser = mgr.begin_with_isolation(IsolationLevel::Serializable);

        assert_eq!(
            mgr.isolation_level(transaction_rc),
            Some(IsolationLevel::ReadCommitted)
        );
        assert_eq!(
            mgr.isolation_level(transaction_si),
            Some(IsolationLevel::SnapshotIsolation)
        );
        assert_eq!(
            mgr.isolation_level(transaction_ser),
            Some(IsolationLevel::Serializable)
        );
    }

    #[test]
    fn test_ssi_read_write_conflict_detected() {
        let mgr = TransactionManager::new();

        // tx1 starts with Serializable isolation
        let tx1 = mgr.begin_with_isolation(IsolationLevel::Serializable);

        // tx2 starts and will modify an entity
        let tx2 = mgr.begin();

        // tx1 reads entity 42
        let entity = NodeId::new(42);
        mgr.record_read(tx1, entity).unwrap();

        // tx2 writes to the same entity and commits
        mgr.record_write(tx2, entity).unwrap();
        mgr.commit(tx2).unwrap();

        // tx1 tries to commit - should fail due to SSI read-write conflict
        let result = mgr.commit(tx1);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Serialization failure"),
            "Expected serialization failure error"
        );
    }

    #[test]
    fn test_ssi_no_conflict_when_not_serializable() {
        let mgr = TransactionManager::new();

        // tx1 starts with default Snapshot Isolation
        let tx1 = mgr.begin();

        // tx2 starts and will modify an entity
        let tx2 = mgr.begin();

        // tx1 reads entity 42
        let entity = NodeId::new(42);
        mgr.record_read(tx1, entity).unwrap();

        // tx2 writes to the same entity and commits
        mgr.record_write(tx2, entity).unwrap();
        mgr.commit(tx2).unwrap();

        // tx1 should commit successfully (SI doesn't check read-write conflicts)
        let result = mgr.commit(tx1);
        assert!(
            result.is_ok(),
            "Snapshot Isolation should not detect read-write conflicts"
        );
    }

    #[test]
    fn test_ssi_no_conflict_when_write_before_read() {
        let mgr = TransactionManager::new();

        // tx1 writes and commits first
        let tx1 = mgr.begin();
        let entity = NodeId::new(42);
        mgr.record_write(tx1, entity).unwrap();
        mgr.commit(tx1).unwrap();

        // tx2 starts AFTER tx1 committed and reads the entity
        let tx2 = mgr.begin_with_isolation(IsolationLevel::Serializable);
        mgr.record_read(tx2, entity).unwrap();

        // tx2 should commit successfully (tx1 committed before tx2 started)
        let result = mgr.commit(tx2);
        assert!(
            result.is_ok(),
            "Should not conflict when writer committed before reader started"
        );
    }

    #[test]
    fn test_write_skew_prevented_by_ssi() {
        // Classic write skew scenario:
        // Account A = 50, Account B = 50, constraint: A + B >= 0
        // T1 reads A, B, writes A = A - 100
        // T2 reads A, B, writes B = B - 100
        // Without SSI, both could commit violating the constraint.

        let mgr = TransactionManager::new();

        let account_a = NodeId::new(1);
        let account_b = NodeId::new(2);

        // T1 and T2 both start with Serializable isolation
        let tx1 = mgr.begin_with_isolation(IsolationLevel::Serializable);
        let tx2 = mgr.begin_with_isolation(IsolationLevel::Serializable);

        // Both read both accounts
        mgr.record_read(tx1, account_a).unwrap();
        mgr.record_read(tx1, account_b).unwrap();
        mgr.record_read(tx2, account_a).unwrap();
        mgr.record_read(tx2, account_b).unwrap();

        // T1 writes to A, T2 writes to B (no write-write conflict)
        mgr.record_write(tx1, account_a).unwrap();
        mgr.record_write(tx2, account_b).unwrap();

        // T1 commits first
        let result1 = mgr.commit(tx1);
        assert!(result1.is_ok(), "First commit should succeed");

        // T2 tries to commit - should fail because it read account_a which T1 wrote
        let result2 = mgr.commit(tx2);
        assert!(result2.is_err(), "Second commit should fail due to SSI");
        assert!(
            result2
                .unwrap_err()
                .to_string()
                .contains("Serialization failure"),
            "Expected serialization failure error for write skew prevention"
        );
    }

    #[test]
    fn test_read_committed_allows_non_repeatable_reads() {
        let mgr = TransactionManager::new();

        // tx1 starts with ReadCommitted isolation
        let tx1 = mgr.begin_with_isolation(IsolationLevel::ReadCommitted);
        let entity = NodeId::new(42);

        // tx1 reads entity
        mgr.record_read(tx1, entity).unwrap();

        // tx2 writes and commits
        let tx2 = mgr.begin();
        mgr.record_write(tx2, entity).unwrap();
        mgr.commit(tx2).unwrap();

        // tx1 can still commit (ReadCommitted allows non-repeatable reads)
        let result = mgr.commit(tx1);
        assert!(
            result.is_ok(),
            "ReadCommitted should allow non-repeatable reads"
        );
    }

    #[test]
    fn test_isolation_level_debug() {
        assert_eq!(
            format!("{:?}", IsolationLevel::ReadCommitted),
            "ReadCommitted"
        );
        assert_eq!(
            format!("{:?}", IsolationLevel::SnapshotIsolation),
            "SnapshotIsolation"
        );
        assert_eq!(
            format!("{:?}", IsolationLevel::Serializable),
            "Serializable"
        );
    }

    #[test]
    fn test_isolation_level_default_trait() {
        let default: IsolationLevel = Default::default();
        assert_eq!(default, IsolationLevel::SnapshotIsolation);
    }

    #[test]
    fn test_ssi_concurrent_reads_no_conflict() {
        let mgr = TransactionManager::new();

        let entity = NodeId::new(42);

        // Both transactions read the same entity
        let tx1 = mgr.begin_with_isolation(IsolationLevel::Serializable);
        let tx2 = mgr.begin_with_isolation(IsolationLevel::Serializable);

        mgr.record_read(tx1, entity).unwrap();
        mgr.record_read(tx2, entity).unwrap();

        // Both should commit successfully (read-read is not a conflict)
        assert!(mgr.commit(tx1).is_ok());
        assert!(mgr.commit(tx2).is_ok());
    }

    #[test]
    fn test_ssi_write_write_conflict() {
        let mgr = TransactionManager::new();

        let entity = NodeId::new(42);

        // Both transactions attempt to write the same entity
        let tx1 = mgr.begin_with_isolation(IsolationLevel::Serializable);
        let tx2 = mgr.begin_with_isolation(IsolationLevel::Serializable);

        // First writer succeeds
        mgr.record_write(tx1, entity).unwrap();

        // Second writer is rejected immediately (first-writer-wins)
        let result = mgr.record_write(tx2, entity);
        assert!(
            result.is_err(),
            "Second record_write should fail with write-write conflict"
        );

        // First commit succeeds
        assert!(mgr.commit(tx1).is_ok());
    }

    #[test]
    fn test_ssi_concurrent_commit_race() {
        // Regression test: with the old read-then-upgrade lock pattern,
        // two concurrent SSI commits could both succeed when one should
        // have been aborted due to a read-write conflict (write skew).
        use std::sync::Arc;

        let mgr = Arc::new(TransactionManager::new());

        // Run many iterations to exercise the race window
        for _ in 0..100 {
            let entity_a = NodeId::new(1);
            let entity_b = NodeId::new(2);

            // Classic write skew setup: both transactions read both entities,
            // then each writes to a different one.
            let tx1 = mgr.begin_with_isolation(IsolationLevel::Serializable);
            let tx2 = mgr.begin_with_isolation(IsolationLevel::Serializable);

            mgr.record_read(tx1, entity_a).unwrap();
            mgr.record_read(tx1, entity_b).unwrap();
            mgr.record_read(tx2, entity_a).unwrap();
            mgr.record_read(tx2, entity_b).unwrap();

            mgr.record_write(tx1, entity_a).unwrap();
            mgr.record_write(tx2, entity_b).unwrap();

            // Commit tx1 first so it is in the committed-claims index
            mgr.commit(tx1).unwrap();

            // tx2 should be rejected: it read entity_a which tx1 wrote
            let result = mgr.commit(tx2);
            assert!(
                result.is_err(),
                "SSI should detect read-write conflict on entity_a"
            );

            // Abort the rejected transaction so its write set is cleared
            // before the next iteration.
            let _ = mgr.abort(tx2);
            mgr.gc();
        }
    }

    #[test]
    fn test_ssi_concurrent_commit_barrier() {
        // Stress test with barrier synchronization to maximize the chance
        // of concurrent commit() calls overlapping.
        use std::sync::{Arc, Barrier};
        use std::thread;

        let mgr = Arc::new(TransactionManager::new());
        let mut both_ok_count = 0;

        for _ in 0..50 {
            let entity_a = NodeId::new(1);
            let entity_b = NodeId::new(2);

            let tx1 = mgr.begin_with_isolation(IsolationLevel::Serializable);
            let tx2 = mgr.begin_with_isolation(IsolationLevel::Serializable);

            mgr.record_read(tx1, entity_a).unwrap();
            mgr.record_read(tx1, entity_b).unwrap();
            mgr.record_read(tx2, entity_a).unwrap();
            mgr.record_read(tx2, entity_b).unwrap();

            mgr.record_write(tx1, entity_a).unwrap();
            mgr.record_write(tx2, entity_b).unwrap();

            let mgr1 = Arc::clone(&mgr);
            let mgr2 = Arc::clone(&mgr);
            let barrier = Arc::new(Barrier::new(2));
            let b1 = Arc::clone(&barrier);
            let b2 = Arc::clone(&barrier);

            let h1 = thread::spawn(move || {
                b1.wait();
                mgr1.commit(tx1)
            });
            let h2 = thread::spawn(move || {
                b2.wait();
                mgr2.commit(tx2)
            });

            let r1 = h1.join().unwrap();
            let r2 = h2.join().unwrap();

            if r1.is_ok() && r2.is_ok() {
                both_ok_count += 1;
            }

            // Clean up
            if r1.is_err() {
                let _ = mgr.abort(tx1);
            }
            if r2.is_err() {
                let _ = mgr.abort(tx2);
            }
            mgr.gc();
        }

        // At most one should succeed per iteration (write skew prevention).
        // With the fix, both_ok_count should always be 0.
        assert_eq!(
            both_ok_count, 0,
            "SSI must prevent both concurrent write-skew commits from succeeding"
        );
    }

    /// A transaction that `commit` returned for is published and gone: its
    /// epoch is readable at once, and the registry no longer holds it (#673 A1).
    #[test]
    fn test_committed_epoch_present_after_commit() {
        let mgr = TransactionManager::new();

        let tx = mgr.begin();
        mgr.record_write(tx, NodeId::new(1)).unwrap();
        let epoch = mgr.commit(tx).unwrap();

        assert_eq!(mgr.current_epoch(), epoch);
        assert_eq!(mgr.state(tx), None);
    }

    /// Until a commit is complete, a direct call that commits at once cannot
    /// start and another transaction cannot write what the committing one
    /// wrote; completing the commit allows both again.
    #[test]
    fn a_commit_holds_its_writes_until_it_is_complete() {
        let mgr = TransactionManager::new();
        let tx = mgr.begin();
        let other = mgr.begin();
        mgr.record_write(tx, NodeId::new(1)).unwrap();

        let commit = mgr.start_commit(tx).unwrap();
        assert_eq!(mgr.state(tx), Some(TransactionState::Committing));
        assert_eq!(mgr.commit_epoch(tx), Some(commit.epoch()));
        assert!(matches!(
            mgr.record_write(other, NodeId::new(1)),
            Err(Error::Transaction(TransactionError::WriteConflict(_)))
        ));
        mgr.record_write(other, NodeId::new(2)).unwrap();
        mgr.abort(other).unwrap();
        assert!(
            mgr.idle_gate().is_none(),
            "a direct call that commits at once waits for the commit"
        );

        commit.complete().unwrap();
        assert_eq!(mgr.state(tx), None, "published: it left the registry");
        assert!(mgr.changes(tx).is_none(), "a commit drops its change set");
        assert!(mgr.idle_gate().is_some());
        let next = mgr.begin();
        mgr.record_write(next, NodeId::new(1)).unwrap();
    }

    /// A private transaction (a direct call) begins during a commit without
    /// waiting for it, at the published epoch, registered with its change
    /// set; it conflicts with the committing transaction on what that one
    /// wrote, and with an open transaction that wrote an entity first.
    #[test]
    fn a_private_transaction_begins_during_a_commit_at_the_published_epoch() {
        let mgr = TransactionManager::new();
        let open = mgr.begin();
        mgr.record_write(open, NodeId::new(3)).unwrap();
        let tx = mgr.begin();
        mgr.record_write(tx, NodeId::new(19)).unwrap();
        let published = mgr.current_epoch();
        let commit = mgr.start_commit(tx).unwrap();

        let (direct, changes) = mgr.begin_private().unwrap();
        assert_eq!(mgr.start_epoch(direct), Some(published));
        assert_eq!((changes.id(), changes.snapshot()), (direct, published));
        assert!(mgr.changes(direct).is_some(), "registered with its set");
        for (node, whose) in [(3, "the open transaction"), (19, "the committing one")] {
            assert!(
                matches!(
                    mgr.record_write(direct, NodeId::new(node)),
                    Err(Error::Transaction(TransactionError::WriteConflict(_)))
                ),
                "{whose} wrote node {node} first"
            );
        }
        commit.complete().unwrap();
        mgr.abort(direct).unwrap();
        assert!(mgr.changes(direct).is_none(), "an abort drops its set");
    }

    /// A transaction begins during a commit without waiting for it: it
    /// starts at the published epoch, which holds none of the commit; one
    /// that begins once the commit is published starts at its epoch.
    #[test]
    fn begin_does_not_wait_for_a_commit_in_progress() {
        let mgr = TransactionManager::new();
        let tx = mgr.begin();
        let published = mgr.current_epoch();
        let commit = mgr.start_commit(tx).unwrap();
        let epoch = commit.epoch();

        let during = mgr.begin();
        assert_eq!(mgr.start_epoch(during), Some(published));
        let publish = commit.written();
        let written = mgr.begin();
        assert_eq!(
            mgr.start_epoch(written),
            Some(published),
            "written is not published"
        );
        publish.complete().unwrap();
        let after = mgr.begin();
        assert_eq!(mgr.start_epoch(after), Some(epoch));
    }

    /// Readers see a commit's epoch only once the commit is complete, so a
    /// snapshot never holds part of a commit.
    #[test]
    fn a_commit_publishes_its_epoch_when_complete() {
        let mgr = TransactionManager::new();
        let before = mgr.current_epoch();
        let tx = mgr.begin();
        let commit = mgr.start_commit(tx).unwrap();
        let epoch = commit.epoch();
        assert!(epoch > before);
        assert_eq!(mgr.current_epoch(), before);

        commit.complete().unwrap();
        assert_eq!(mgr.current_epoch(), epoch);
    }

    // ---- A commit in phases (W4.1) ----

    /// A commit whose log write failed is aborted: its epoch goes back (no
    /// later commit took one yet), nothing is published or poisoned, and the
    /// transaction is active again for the caller's rollback, still holding
    /// its claims until then. The next commit takes the same epoch.
    ///
    /// After `a_cancelled_prepare_releases_claims_without_publishing` of
    /// pull request #614 (temporaryfix), with the epoch given back.
    #[test]
    fn an_aborted_commit_gives_its_epoch_back_and_keeps_the_manager_usable() {
        let mgr = TransactionManager::new();
        let tx = mgr.begin();
        mgr.record_write(tx, NodeId::new(1)).unwrap();
        let commit = mgr.start_commit(tx).unwrap();
        assert_eq!(commit.epoch(), EpochId::new(1));
        commit.abort();

        assert_eq!(mgr.state(tx), Some(TransactionState::Active));
        assert_eq!(mgr.commit_epoch(tx), None);
        assert_eq!(mgr.current_epoch(), EpochId::new(0));
        assert!(!mgr.has_incomplete_commit());
        let other = mgr.begin();
        assert!(
            matches!(
                mgr.record_write(other, NodeId::new(1)),
                Err(Error::Transaction(TransactionError::WriteConflict(_)))
            ),
            "the transaction of an aborted commit holds its claim until it is rolled back"
        );
        mgr.abort(other).unwrap();
        mgr.abort(tx).unwrap();
        assert_eq!(mgr.active_count(), 0);
        assert!(mgr.idle_gate().is_some());

        // Every assigned epoch is published: a hold does not wait.
        std::thread::scope(|scope| {
            let _end = EndWaits(&mgr);
            let manager = &mgr;
            let (finished, _) = spawn_and_wait(scope, PATIENTLY, move || {
                drop(manager.hold_commits_for_change().unwrap());
                drop(manager.hold_commits().unwrap());
            });
            assert!(
                finished,
                "a hold waits for the epoch of the aborted commit, which nothing publishes"
            );
        });
        let next = mgr.begin();
        mgr.record_write(next, NodeId::new(1))
            .expect("the rolled-back transaction holds no claim");
        assert_eq!(
            mgr.commit(next).unwrap(),
            EpochId::new(1),
            "the next commit takes the epoch the aborted one gave back"
        );
    }

    /// A written commit has released the commit lock, so the next one starts,
    /// but its transaction is still committing: its claims stay, readers do
    /// not see its epoch, and a direct call that commits at once waits.
    #[test]
    fn claims_stay_until_publish() {
        let mgr = TransactionManager::new();
        let before = mgr.begin();
        let tx = mgr.begin();
        mgr.record_write(tx, NodeId::new(1)).unwrap();
        let publish = mgr.start_commit(tx).unwrap().written();
        let epoch = publish.epoch();

        assert_eq!(mgr.state(tx), Some(TransactionState::Committing));
        assert_eq!(mgr.current_epoch(), EpochId::new(0));
        let after = mgr.begin();
        assert_eq!(
            mgr.start_epoch(after),
            Some(EpochId::new(0)),
            "a transaction begins while a written commit waits to publish, below its epoch"
        );
        for other in [before, after] {
            assert!(matches!(
                mgr.record_write(other, NodeId::new(1)),
                Err(Error::Transaction(TransactionError::WriteConflict(_)))
            ));
        }
        assert!(mgr.idle_gate().is_none());

        // The next commit runs meanwhile, and publishes after this one.
        mgr.record_write(after, NodeId::new(2)).unwrap();
        let second = mgr.start_commit(after).unwrap().written();
        let second_epoch = second.epoch();
        assert_eq!(second_epoch.as_u64(), epoch.as_u64() + 1);

        publish.complete().unwrap();
        assert_eq!(mgr.state(tx), None, "published: it left the registry");
        assert_eq!(mgr.current_epoch(), epoch);
        second.complete().unwrap();
        assert_eq!(mgr.current_epoch(), second_epoch);
        mgr.abort(before).unwrap();
        let next = mgr.begin();
        mgr.record_write(next, NodeId::new(1)).unwrap();
    }

    /// Eight written commits publish in the order of their epochs, whatever
    /// order their committers finish in: none publishes before the one
    /// below it, so no reader sees an epoch with a gap under it.
    #[test]
    fn publish_follows_epoch_order_under_eight_committers() {
        let mgr = TransactionManager::new();
        let mut written: Vec<_> = (0..8)
            .map(|_| {
                let tx = mgr.begin();
                mgr.start_commit(tx).unwrap().written()
            })
            .collect();
        let epochs: Vec<u64> = written
            .iter()
            .map(|publish| publish.epoch().as_u64())
            .collect();
        assert_eq!(epochs, (1..=8).collect::<Vec<u64>>());
        let first = written.remove(0);

        let manager = &mgr;
        std::thread::scope(|scope| {
            // The highest epochs first: each waits for the one below it.
            let waiting: Vec<_> = written
                .into_iter()
                .rev()
                .map(|publish| {
                    let epoch = publish.epoch();
                    let (finished, handle) = spawn_and_wait(scope, BRIEFLY, move || {
                        publish.complete().unwrap();
                        // Published in order: nothing below is missing.
                        assert!(manager.current_epoch() >= epoch);
                    });
                    assert!(!finished, "epoch {epoch:?} was published before epoch 1");
                    handle
                })
                .collect();
            assert_eq!(mgr.current_epoch(), EpochId::new(0));
            assert!(mgr.idle_gate().is_none(), "eight commits are still open");

            first.complete().unwrap();
            for handle in waiting {
                handle.join().unwrap();
            }
        });
        assert_eq!(mgr.current_epoch(), EpochId::new(8));
        assert!(mgr.idle_gate().is_some(), "every commit is published");
        assert!(!mgr.has_incomplete_commit());
    }

    /// A checkpoint's hold, a change outside a commit and `close()` take the
    /// commit lock and then wait until every written commit is published:
    /// the committed state they see, log or close on holds every commit that
    /// is in the log.
    #[test]
    fn a_hold_waits_for_every_written_commit_to_publish() {
        type Hold = fn(&TransactionManager);
        let holds: [(&str, Hold); 3] = [
            ("hold_commits", |mgr| drop(mgr.hold_commits().unwrap())),
            ("hold_commits_for_change", |mgr| {
                drop(mgr.hold_commits_for_change().unwrap());
            }),
            ("close_for_writes", TransactionManager::close_for_writes),
        ];
        for (name, hold) in holds {
            let mgr = TransactionManager::new();
            let tx = mgr.begin();
            let publish = mgr.start_commit(tx).unwrap().written();
            let epoch = publish.epoch();
            let manager = &mgr;
            std::thread::scope(|scope| {
                let (finished, held) = spawn_and_wait(scope, BRIEFLY, move || {
                    hold(manager);
                    manager.current_epoch()
                });
                assert!(
                    !finished,
                    "{name} returned before the written commit was published"
                );
                publish.complete().unwrap();
                assert_eq!(held.join().unwrap(), epoch, "{name}");
            });
        }
    }

    /// A publish handle dropped without `complete` (the commit code panicked
    /// after its group was written) is a commit that did not complete: its
    /// epoch is never published, and neither is any epoch above it, since
    /// that would show its stamped versions. Later commits fail, a written
    /// one at its publish, and a hold fails instead of waiting for ever.
    #[test]
    fn a_dropped_publish_handle_poisons_as_an_incomplete_commit() {
        let mgr = TransactionManager::new();
        let (first, tx, later) = (mgr.begin(), mgr.begin(), mgr.begin());
        mgr.record_write(tx, NodeId::new(1)).unwrap();
        let below = mgr.start_commit(first).unwrap().written();
        let publish = mgr.start_commit(tx).unwrap().written();
        let above = mgr.start_commit(later).unwrap().written();
        let below_epoch = below.epoch();

        drop(publish);
        assert!(mgr.has_incomplete_commit());
        assert_eq!(mgr.state(tx), Some(TransactionState::Committing));
        std::thread::scope(|scope| {
            let _end = EndWaits(&mgr);
            let above = publishing(scope, above);
            assert!(matches!(
                above.recv_timeout(PATIENTLY),
                Ok(Err(Error::Transaction(TransactionError::IncompleteCommit)))
            ));
            assert_eq!(mgr.state(later), Some(TransactionState::Committing));
            assert_eq!(mgr.current_epoch(), EpochId::new(0));
            // The commit below it holds nothing of it: it is published.
            let below = publishing(scope, below);
            assert!(matches!(below.recv_timeout(PATIENTLY), Ok(Ok(()))));
        });
        assert_eq!(mgr.state(first), None, "published: it left the registry");
        assert_eq!(mgr.current_epoch(), below_epoch);

        let next = mgr.begin();
        assert!(matches!(
            mgr.start_commit(next),
            Err(Error::Transaction(TransactionError::IncompleteCommit))
        ));
        assert!(matches!(
            mgr.hold_commits(),
            Err(Error::Transaction(TransactionError::IncompleteCommit))
        ));
        assert!(matches!(
            mgr.hold_commits_for_change(),
            Err(Error::Transaction(TransactionError::IncompleteCommit))
        ));
        // Returns: the database closes on what was published.
        mgr.close_for_writes();
        assert_eq!(mgr.current_epoch(), below_epoch);
    }

    /// After a broken invariant ([`TransactionManager::poison`]) nothing is
    /// published any more, a written commit included: no commit may build
    /// on that store.
    #[test]
    fn a_poisoned_manager_publishes_nothing_more() {
        let mgr = TransactionManager::new();
        let tx = mgr.begin();
        let written = mgr.start_commit(tx).unwrap().written();
        mgr.poison("an undo failed");
        std::thread::scope(|scope| {
            let _end = EndWaits(&mgr);
            let written = publishing(scope, written);
            assert!(matches!(
                written.recv_timeout(PATIENTLY),
                Ok(Err(Error::Transaction(TransactionError::IncompleteCommit)))
            ));
        });
        assert_eq!(mgr.current_epoch(), EpochId::new(0));
    }

    /// A commit that waits to publish when a commit below it does not
    /// complete is woken and fails: it does not wait for ever.
    #[test]
    fn a_waiting_publish_fails_when_the_commit_below_it_does_not_complete() {
        let mgr = TransactionManager::new();
        let tx = mgr.begin();
        let later = mgr.begin();
        let publish = mgr.start_commit(tx).unwrap().written();
        let above = mgr.start_commit(later).unwrap().written();
        std::thread::scope(|scope| {
            let _end = EndWaits(&mgr);
            let above = publishing(scope, above);
            assert!(
                above.recv_timeout(BRIEFLY).is_err(),
                "published before the commit below it"
            );
            drop(publish);
            assert!(matches!(
                above.recv_timeout(PATIENTLY),
                Ok(Err(Error::Transaction(TransactionError::IncompleteCommit)))
            ));
        });
        assert_eq!(mgr.current_epoch(), EpochId::new(0));
    }

    /// Ends every wait for an epoch when a test fails, so a failed assertion
    /// fails the test instead of leaving a thread waiting for a commit that
    /// never publishes.
    struct EndWaits<'a>(&'a TransactionManager);

    impl Drop for EndWaits<'_> {
        fn drop(&mut self) {
            if std::thread::panicking() {
                // Both ways a wait ends: refused, or its turn has come.
                self.0.incomplete_from(EpochId::new(0));
                self.0.sync_epoch(EpochId::new(u64::MAX - 1));
            }
        }
    }

    /// Publishes `publish` on a scoped thread; the result arrives on the
    /// returned channel once it is published or refused.
    fn publishing<'scope>(
        scope: &'scope std::thread::Scope<'scope, '_>,
        publish: Publish<'scope>,
    ) -> std::sync::mpsc::Receiver<Result<()>> {
        let (done, result) = std::sync::mpsc::channel();
        scope.spawn(move || {
            let _ = done.send(publish.complete());
        });
        result
    }

    /// `complete` on the guard publishes while it holds the commit lock, as
    /// every caller did before the phases: nothing starts in between.
    #[test]
    fn a_commit_completed_under_its_lock_is_published_at_once() {
        let mgr = TransactionManager::new();
        let tx = mgr.begin();
        let commit = mgr.start_commit(tx).unwrap();
        let epoch = commit.epoch();
        commit.complete().unwrap();
        assert_eq!(mgr.current_epoch(), epoch);
        assert_eq!(mgr.state(tx), None, "published: it left the registry");
        assert!(mgr.idle_gate().is_some());
    }

    /// A commit that does not complete (its code panicked between the commit
    /// decision and `complete`) is not reported as committed and does not
    /// publish its epoch: its writes stay locked against other transactions,
    /// which get the error of the failed commit (not a retryable conflict)
    /// when they write, and the commit lock is released, so transactions
    /// still begin (no transaction commits afterwards, see the next test).
    #[test]
    fn a_commit_that_does_not_complete_keeps_its_writes_locked() {
        let mgr = TransactionManager::new();
        let before = mgr.current_epoch();
        let tx = mgr.begin();
        let other = mgr.begin();
        mgr.record_write(tx, NodeId::new(1)).unwrap();

        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _commit = mgr.start_commit(tx).unwrap();
            panic!("stamping failed");
        }));
        assert!(unwound.is_err());

        assert_eq!(mgr.state(tx), Some(TransactionState::Committing));
        assert_eq!(mgr.current_epoch(), before);
        assert!(matches!(
            mgr.record_write(other, NodeId::new(1)),
            Err(Error::Transaction(TransactionError::IncompleteCommit))
        ));
        let next = mgr.begin();
        assert_eq!(mgr.start_epoch(next), Some(before));
    }

    /// A commit that does not complete may have stamped versions with its
    /// epoch, which readers never see. A later commit would publish a higher
    /// epoch, and with it that part of the failed commit, so no transaction
    /// commits afterwards (not even one that wrote nothing) and the published
    /// epoch never moves past the failed one. Transactions still begin, at
    /// the last published epoch.
    #[test]
    fn after_a_commit_that_does_not_complete_no_transaction_commits() {
        let mgr = TransactionManager::new();
        let first = mgr.begin();
        let published = mgr.commit(first).unwrap();

        let tx = mgr.begin();
        mgr.record_write(tx, NodeId::new(3)).unwrap();
        let commit = mgr.start_commit(tx).unwrap();
        let failed = commit.epoch();
        drop(commit);

        for wrote in [true, false] {
            let next = mgr.begin();
            assert_eq!(
                mgr.start_epoch(next),
                Some(published),
                "a new transaction reads what was published before the failed commit"
            );
            if wrote {
                assert!(
                    matches!(
                        mgr.record_write(next, NodeId::new(19)),
                        Err(Error::Transaction(TransactionError::IncompleteCommit))
                    ),
                    "a write fails with the failed commit's error"
                );
            }
            let Err(error) = mgr.start_commit(next) else {
                panic!("a commit after an incomplete one (wrote: {wrote}) succeeded");
            };
            assert!(
                matches!(
                    &error,
                    Error::Transaction(TransactionError::IncompleteCommit)
                ),
                "a typed error: {error:?}"
            );
            assert_eq!(error.error_code().as_str(), "GRAFEO-T008");
            let message = error.to_string();
            assert!(
                message.contains("did not complete") && message.contains("reopen"),
                "the error says a commit did not complete and the database must be \
                 reopened: {message}"
            );
            assert!(
                mgr.commit(next).is_err(),
                "commit refuses as start_commit does"
            );
            mgr.abort(next).unwrap();
        }
        assert_eq!(
            mgr.current_epoch(),
            published,
            "the published epoch stays before the failed epoch {failed:?}"
        );
        assert_eq!(mgr.state(tx), Some(TransactionState::Committing));
    }

    /// `commit` completes at once: nothing waits for it afterwards.
    #[test]
    fn commit_completes_at_once() {
        let mgr = TransactionManager::new();
        let tx = mgr.begin();
        mgr.record_write(tx, NodeId::new(1)).unwrap();
        mgr.commit(tx).unwrap();

        assert_eq!(mgr.state(tx), None, "published: it left the registry");
        assert!(mgr.idle_gate().is_some());
        let next = mgr.begin();
        mgr.record_write(next, NodeId::new(1)).unwrap();
    }

    /// How long work that should wait gets to finish anyway.
    const BRIEFLY: std::time::Duration = std::time::Duration::from_millis(100);

    /// How long work that should not wait gets to finish: long, so a loaded
    /// machine does not fail the test, and finite, so a wait fails it
    /// instead of hanging.
    const PATIENTLY: std::time::Duration = std::time::Duration::from_secs(30);

    /// Runs `work` on a scoped thread; returns whether it finished within
    /// `wait`, and its handle.
    fn spawn_and_wait<'scope, T: Send + 'scope>(
        scope: &'scope std::thread::Scope<'scope, '_>,
        wait: std::time::Duration,
        work: impl FnOnce() -> T + Send + 'scope,
    ) -> (bool, std::thread::ScopedJoinHandle<'scope, T>) {
        let (done, finished) = std::sync::mpsc::channel();
        let handle = scope.spawn(move || {
            let result = work();
            let _ = done.send(());
            result
        });
        (finished.recv_timeout(wait).is_ok(), handle)
    }

    /// A checkpoint's hold waits for a store change of an open transaction
    /// in progress, and once it holds commits off, a store change waits for
    /// it: the image is read from a store that does not move.
    #[test]
    fn a_checkpoint_hold_and_a_write_in_progress_wait_for_each_other() {
        let mgr = TransactionManager::new();
        let manager = &mgr;
        std::thread::scope(|scope| {
            let writing = mgr.write_in_progress();
            let (finished, checkpoint) = spawn_and_wait(scope, BRIEFLY, move || {
                drop(manager.hold_commits().unwrap());
            });
            assert!(!finished, "the hold waits for the write in progress");
            drop(writing);
            checkpoint.join().unwrap();

            let held = mgr.hold_commits().unwrap();
            let (finished, write) = spawn_and_wait(scope, BRIEFLY, move || {
                drop(manager.write_in_progress());
            });
            assert!(!finished, "a write waits for the checkpoint's hold");
            drop(held);
            write.join().unwrap();
        });
    }

    /// Writes in progress do not wait for each other, and a change outside
    /// any commit (a schema statement, a graph command) holds commits off
    /// without freezing the store: writes of open
    /// transactions go on, and it does not wait for them.
    #[test]
    fn a_change_outside_a_commit_does_not_freeze_the_store() {
        let mgr = TransactionManager::new();
        let manager = &mgr;
        std::thread::scope(|scope| {
            let writing = mgr.write_in_progress();
            let (finished, other) = spawn_and_wait(scope, PATIENTLY, move || {
                drop(manager.write_in_progress());
            });
            assert!(finished, "another write runs alongside");
            other.join().unwrap();

            let (finished, change) = spawn_and_wait(scope, PATIENTLY, move || {
                drop(manager.hold_commits_for_change().unwrap());
            });
            assert!(finished, "the change does not wait for the write");
            change.join().unwrap();
            drop(writing);

            let held = mgr.hold_commits_for_change().unwrap();
            let (finished, write) = spawn_and_wait(scope, PATIENTLY, move || {
                drop(manager.write_in_progress());
            });
            assert!(finished, "a write does not wait for the change");
            write.join().unwrap();
            drop(held);
        });
    }

    // === Edge endpoints claimed against deletes ===

    /// Alix, Gus and Vincent as entities of the default graph.
    fn people() -> (GraphEntity, GraphEntity, GraphEntity) {
        (
            GraphEntity::from(NodeId::new(3)),
            GraphEntity::from(NodeId::new(19)),
            GraphEntity::from(NodeId::new(88)),
        )
    }

    /// Whether `result` is a write conflict.
    fn is_conflict(result: &Result<impl std::fmt::Debug>) -> bool {
        matches!(
            result,
            Err(Error::Transaction(TransactionError::WriteConflict(_)))
        )
    }

    /// A delete of a node that an open transaction claimed as an edge's
    /// endpoint is a conflict; a delete of another node is not.
    #[test]
    fn a_delete_conflicts_with_an_open_claim_on_the_node() {
        let mgr = TransactionManager::new();
        let (alix, gus, vincent) = people();
        let linker = mgr.begin();
        let deleter = mgr.begin();
        mgr.record_endpoints(linker, [alix, gus.clone()]).unwrap();

        let deleted = mgr.record_delete(deleter, gus);
        assert!(is_conflict(&deleted), "got {deleted:?}");
        mgr.record_delete(deleter, vincent).unwrap();
        mgr.commit(linker).unwrap();
        mgr.commit(deleter).unwrap();
    }

    /// A claim on a node an open transaction deletes is a conflict, and
    /// claims neither endpoint: once the delete is rolled back, another
    /// transaction deletes the other endpoint without a conflict.
    #[test]
    fn a_claim_conflicts_with_an_open_delete_and_claims_nothing() {
        let mgr = TransactionManager::new();
        let (alix, gus, _) = people();
        let deleter = mgr.begin();
        let linker = mgr.begin();
        mgr.record_delete(deleter, gus.clone()).unwrap();

        let claimed = mgr.record_endpoints(linker, [alix.clone(), gus]);
        assert!(is_conflict(&claimed), "got {claimed:?}");
        mgr.abort(deleter).unwrap();
        let other = mgr.begin();
        mgr.record_delete(other, alix)
            .expect("the failed claim left Alix unclaimed");
    }

    /// Claims conflict with deletes only: transactions that create edges to
    /// one node, and one that writes the node, all commit.
    #[test]
    fn claims_do_not_conflict_with_claims_or_writes() {
        let mgr = TransactionManager::new();
        let (alix, gus, vincent) = people();
        let alix_links = mgr.begin();
        let vincent_links = mgr.begin();
        let writer = mgr.begin();
        mgr.record_endpoints(alix_links, [alix, gus.clone()])
            .unwrap();
        mgr.record_endpoints(vincent_links, [vincent, gus.clone()])
            .unwrap();
        mgr.record_write(writer, gus).unwrap();
        mgr.commit(alix_links).unwrap();
        mgr.commit(vincent_links).unwrap();
        mgr.commit(writer).unwrap();
    }

    /// A transaction whose new edge ends at a node another transaction
    /// deleted and committed after it began fails its commit.
    #[test]
    fn a_claim_fails_the_commit_after_a_later_committed_delete() {
        let mgr = TransactionManager::new();
        let (alix, gus, _) = people();
        let linker = mgr.begin();
        let deleter = mgr.begin();
        mgr.record_delete(deleter, gus.clone()).unwrap();
        mgr.commit(deleter).unwrap();

        mgr.record_endpoints(linker, [alix, gus])
            .expect("the delete is committed: no open transaction holds it");
        let committed = mgr.commit(linker);
        assert!(is_conflict(&committed), "got {committed:?}");
    }

    /// The other way around: a delete fails its commit when another
    /// transaction committed an edge to the node after the delete's
    /// transaction began.
    #[test]
    fn a_delete_fails_the_commit_after_a_later_committed_claim() {
        let mgr = TransactionManager::new();
        let (alix, gus, _) = people();
        let deleter = mgr.begin();
        let linker = mgr.begin();
        mgr.record_endpoints(linker, [alix, gus.clone()]).unwrap();
        mgr.commit(linker).unwrap();

        mgr.record_delete(deleter, gus).unwrap();
        let committed = mgr.commit(deleter);
        assert!(is_conflict(&committed), "got {committed:?}");
    }

    /// A claim committed before a transaction began is part of its snapshot
    /// (it sees the edge, and deletes it with the node): no conflict.
    #[test]
    fn a_claim_committed_before_the_transaction_began_does_not_conflict() {
        let mgr = TransactionManager::new();
        let (alix, gus, _) = people();
        let linker = mgr.begin();
        mgr.record_endpoints(linker, [alix, gus.clone()]).unwrap();
        mgr.commit(linker).unwrap();

        let deleter = mgr.begin();
        mgr.record_delete(deleter, gus).unwrap();
        mgr.commit(deleter).unwrap();
    }

    /// Claims of one node id in two graphs are claims of two nodes: a delete
    /// in one graph does not conflict with an edge in the other.
    #[test]
    fn claims_in_different_graphs_do_not_conflict() {
        let mgr = TransactionManager::new();
        let gus = NodeId::new(19);
        let linker = mgr.begin();
        let deleter = mgr.begin();
        mgr.record_endpoints(
            linker,
            [
                GraphEntity::new(Some(Arc::from("Paris")), gus),
                GraphEntity::new(Some(Arc::from("Paris")), gus),
            ],
        )
        .unwrap();
        mgr.record_delete(deleter, GraphEntity::new(Some(Arc::from("Prague")), gus))
            .unwrap();
        mgr.commit(linker).unwrap();
        mgr.commit(deleter).unwrap();
    }

    /// The change set's grant covers it through the commit: the commit runs
    /// the bound backends from the set (its triples, schema ops and change
    /// events apply at the collect, still from the set). The grant goes with
    /// the set.
    #[cfg(feature = "triple-store")]
    #[test]
    fn a_commit_keeps_the_change_set_grant_until_the_set_goes() {
        use grafeo_common::memory::buffer::BufferManager;
        use grafeo_core::graph::GraphBackend;
        use grafeo_core::graph::rdf::{Term, Triple};
        use grafeo_core::graph::rowgroup::RowGroupStore;

        let memory = BufferManager::with_budget(1 << 30);
        let manager = TransactionManager::new();
        manager.attach_memory(Arc::clone(&memory));
        let (_transaction, changes) = manager.begin_private().unwrap();
        // A real backend bound to the set, so the commit runs one.
        let store: Arc<dyn GraphBackend> = Arc::new(RowGroupStore::default());
        changes.bind(None, store).unwrap();
        for i in 0..20_000 {
            changes
                .record_triple(
                    None,
                    Triple::new(
                        Term::iri(format!("http://ex.org/vincent{i}")),
                        Term::iri("http://ex.org/knows"),
                        Term::iri("http://ex.org/mia"),
                    ),
                    true,
                )
                .unwrap();
        }
        let held = memory.allocated();
        assert!(held > 0, "the set holds a grant");
        changes.commit(EpochId::new(1)).unwrap();
        assert_eq!(
            memory.allocated(),
            held,
            "the set lives on through the rest of the commit"
        );
        drop(changes);
        drop(manager);
        assert_eq!(memory.allocated(), 0, "the grant went with the set");
    }
}
