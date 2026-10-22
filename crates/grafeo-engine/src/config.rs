//! Database configuration.

use std::fmt;
use std::path::PathBuf;
use std::time::Duration;

/// Encryption-at-rest configuration.
///
/// Provides the key chain that derives the data encryption keys from a master
/// encryption key via HKDF-SHA256: one for the `.grafeo` file
/// (`"grafeo-container"`) and one for its sidecar WAL (`"grafeo-wal"`), each
/// with the database id of the file header, so every database has keys of its
/// own. See [`Config::encryption`].
///
/// Wrapped in `Arc` internally so `Config` can remain `Clone` without
/// duplicating key material. Build it with [`EncryptionConfig::new()`]:
/// later releases may add settings, so outside this crate it cannot be built
/// with a struct literal.
#[cfg(feature = "encryption")]
#[derive(Clone)]
#[non_exhaustive]
pub struct EncryptionConfig {
    /// The key chain that derives per-component encryption keys.
    /// Shared via Arc so Config can be cloned.
    pub key_chain: std::sync::Arc<grafeo_common::encryption::KeyChain>,
}

#[cfg(feature = "encryption")]
impl EncryptionConfig {
    /// Encrypts with the keys `key_chain` derives from its master key.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use grafeo_common::encryption::KeyChain;
    /// use grafeo_engine::config::EncryptionConfig;
    ///
    /// // 32 bytes from your key management (a KMS, a secrets manager, an HSM).
    /// let encryption = EncryptionConfig::new(Arc::new(KeyChain::new([19; 32])));
    /// ```
    ///
    /// A struct literal does not compile, as a later setting would break it:
    ///
    /// ```compile_fail,E0639
    /// use std::sync::Arc;
    ///
    /// use grafeo_common::encryption::KeyChain;
    /// use grafeo_engine::config::EncryptionConfig;
    ///
    /// let encryption = EncryptionConfig {
    ///     key_chain: Arc::new(KeyChain::new([19; 32])),
    /// };
    /// ```
    #[must_use]
    pub fn new(key_chain: std::sync::Arc<grafeo_common::encryption::KeyChain>) -> Self {
        Self { key_chain }
    }
}

#[cfg(feature = "encryption")]
impl fmt::Debug for EncryptionConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EncryptionConfig")
            .field("key_chain", &"[redacted]")
            .finish()
    }
}

/// The graph data model for a database.
///
/// Each database uses exactly one model, chosen at creation time and immutable
/// after that. The engine initializes only the relevant store, saving memory.
///
/// Schema variants (OWL, RDFS, JSON Schema) are a server-level concern - from
/// the engine's perspective those map to either `Lpg` or `Rdf`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GraphModel {
    /// Labeled Property Graph (default). Supports GQL, Cypher, Gremlin, GraphQL.
    #[default]
    Lpg,
    /// RDF triple store. Supports SPARQL.
    Rdf,
}

impl fmt::Display for GraphModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lpg => write!(f, "LPG"),
            Self::Rdf => write!(f, "RDF"),
        }
    }
}

/// Access mode for opening a database.
///
/// Controls whether the database is opened for full read-write access
/// (the default) or read-only access. Read-only mode uses a shared file
/// lock, allowing multiple processes to read the same `.grafeo` file
/// concurrently.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AccessMode {
    /// Full read-write access (default). Acquires an exclusive file lock.
    #[default]
    ReadWrite,
    /// Read-only access. Acquires a shared file lock, allowing concurrent
    /// readers. The database loads the last checkpoint and replays the
    /// sidecar WAL into memory, writes nothing, and allows no mutations. A
    /// build without the `wal` feature refuses a file whose sidecar WAL holds
    /// commits to replay (a non-empty log file; for a 0.5.x file, any file),
    /// as a read-write open there does.
    ReadOnly,
}

impl fmt::Display for AccessMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReadWrite => write!(f, "read-write"),
            Self::ReadOnly => write!(f, "read-only"),
        }
    }
}

/// Storage format for persistent databases.
///
/// Since 0.6 every database is a single file (its WAL is the sidecar
/// directory `<path>.wal/` while it is open), and the format only decides
/// what happens at a path where nothing exists yet. An existing path always
/// opens as what it holds, whatever the format: a 0.6 file opens, and a
/// database written by 0.5.x (a file, or a WAL directory) is migrated to a
/// single 0.6 file at the same path by a read-write open and read in place by
/// a read-only open.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StorageFormat {
    /// A new path becomes a single file, whatever its extension (`.grafeo`,
    /// `.db` or none). An existing path opens as what it holds.
    #[default]
    Auto,
    /// The 0.5.x WAL-directory format (a directory holding `wal/`). WAL
    /// directories are no longer created: on a new path the open fails, use
    /// [`StorageFormat::Auto`] instead. An existing path opens as what it
    /// holds, as with `Auto`: a 0.5.x WAL directory is migrated to a single
    /// file at the same path.
    #[deprecated(
        since = "0.6.0",
        note = "WAL directories are migrated to a single file on open and no longer created; removed in 0.7.0"
    )]
    WalDirectory,
    /// A new path becomes a single file, as with [`StorageFormat::Auto`]. An
    /// existing path opens as what it holds. Since every database is a single
    /// file, it does the same as `Auto`.
    #[deprecated(
        since = "0.6.0",
        note = "every database is a single file, so it does the same as `StorageFormat::Auto`; removed in 0.7.0"
    )]
    SingleFile,
}

impl fmt::Display for StorageFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            #[allow(
                deprecated,
                reason = "the crate names its own deprecated variant, until 0.7.0 removes it"
            )]
            Self::WalDirectory => write!(f, "wal-directory"),
            #[allow(
                deprecated,
                reason = "the crate names its own deprecated variant, until 0.7.0 removes it"
            )]
            Self::SingleFile => write!(f, "single-file"),
        }
    }
}

/// WAL durability mode controlling the trade-off between safety and speed.
///
/// This enum lives in config so that `Config` can always carry the desired
/// durability regardless of whether the `wal` feature is compiled in. When
/// WAL is enabled, the engine maps this to the adapter-level durability mode.
///
/// Build the variants with settings with [`DurabilityMode::batch()`] and
/// [`DurabilityMode::adaptive()`]: later releases may add settings to them,
/// so outside this crate they cannot be built with a struct expression, and
/// a pattern names their fields with `..`.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
///
/// use grafeo_engine::{Config, DurabilityMode};
///
/// let config = Config::persistent("amsterdam.grafeo")
///     .with_wal_durability(DurabilityMode::batch(Duration::from_millis(19), 88));
/// assert!(matches!(
///     config.wal_durability,
///     DurabilityMode::Batch { max_records: 88, .. }
/// ));
/// ```
///
/// ```compile_fail,E0639
/// use grafeo_engine::DurabilityMode;
///
/// let mode = DurabilityMode::Adaptive {
///     target_interval_ms: 19,
/// };
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DurabilityMode {
    /// Fsync after every commit. Slowest but safest.
    Sync,
    /// Batch fsync periodically. Good balance of performance and durability.
    /// Built with [`DurabilityMode::batch()`].
    #[non_exhaustive]
    Batch {
        /// Maximum time between syncs in milliseconds.
        max_delay_ms: u64,
        /// Maximum records between syncs.
        max_records: u64,
    },
    /// Adaptive sync via a background flusher thread. Built with
    /// [`DurabilityMode::adaptive()`].
    #[non_exhaustive]
    Adaptive {
        /// Target interval between flushes in milliseconds.
        target_interval_ms: u64,
    },
    /// No sync - rely on OS buffer flushing. Fastest but may lose recent data.
    NoSync,
}

impl DurabilityMode {
    /// Syncs at a commit once `max_delay` has passed since the last sync or
    /// `max_records` records were written since, whichever comes first.
    ///
    /// The delay counts in whole milliseconds: a part of a millisecond is
    /// dropped.
    #[must_use]
    pub fn batch(max_delay: Duration, max_records: u64) -> Self {
        Self::Batch {
            max_delay_ms: whole_millis(max_delay),
            max_records,
        }
    }

    /// Syncs from a background thread every `target_interval`.
    ///
    /// The interval counts in whole milliseconds: a part of a millisecond is
    /// dropped, and [`Config::validate()`] refuses an interval of zero.
    #[must_use]
    pub fn adaptive(target_interval: Duration) -> Self {
        Self::Adaptive {
            target_interval_ms: whole_millis(target_interval),
        }
    }
}

/// `duration` in whole milliseconds, `u64::MAX` when it has more.
fn whole_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

impl Default for DurabilityMode {
    fn default() -> Self {
        Self::Batch {
            max_delay_ms: 100,
            max_records: 1000,
        }
    }
}

/// Errors from [`Config::validate()`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConfigError {
    /// Memory limit must be greater than zero.
    ZeroMemoryLimit,
    /// Thread count must be greater than zero.
    ZeroThreads,
    /// `DurabilityMode::Adaptive` needs an interval greater than zero.
    ZeroAdaptiveFlushInterval,
    /// RDF graph model requires the `rdf` feature flag.
    RdfFeatureRequired,
    /// `encryption` is set without a persistent `path`: an in-memory database
    /// writes nothing to disk to encrypt.
    EncryptionRequiresPersistentPath,
    /// `encryption` is set together with a `spill_path`: spill files are not
    /// encrypted, so an encrypted database spills nothing.
    EncryptionWithSpillPath,
    /// `encryption` is set and a section is pinned to
    /// [`TierOverride::ForceDisk`](grafeo_common::storage::TierOverride::ForceDisk):
    /// an encrypted database spills nothing to disk, so the override could
    /// not be honored.
    EncryptionWithForceDisk(grafeo_common::storage::SectionType),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroMemoryLimit => write!(f, "memory_limit must be greater than zero"),
            Self::ZeroThreads => write!(f, "threads must be greater than zero"),
            Self::ZeroAdaptiveFlushInterval => write!(
                f,
                "the target interval of the adaptive durability mode must be greater than zero"
            ),
            Self::RdfFeatureRequired => {
                write!(
                    f,
                    "RDF graph model requires the `rdf` feature flag to be enabled"
                )
            }
            Self::EncryptionRequiresPersistentPath => write!(
                f,
                "encryption at rest requires a persistent database path: an in-memory \
                 database writes nothing to disk to encrypt"
            ),
            Self::EncryptionWithSpillPath => write!(
                f,
                "encryption at rest cannot be combined with a spill_path: spill files are \
                 not encrypted, so an encrypted database spills nothing to disk"
            ),
            Self::EncryptionWithForceDisk(section_type) => write!(
                f,
                "encryption at rest cannot be combined with a disk tier override for the \
                 {section_type:?} section: an encrypted database spills nothing to disk"
            ),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Database configuration.
///
/// Start from [`Config::in_memory()`], [`Config::persistent()`] or
/// [`Config::read_only()`] and change settings with the `with_*` and
/// `without_*` methods. The fields are public for reading. Later releases
/// add settings, so outside this crate a `Config` cannot be built with a
/// struct literal.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
///
/// use grafeo_engine::Config;
///
/// let config = Config::persistent("amsterdam.grafeo")
///     .with_memory_limit(88 * 1024 * 1024)
///     .with_threads(3)
///     .with_query_timeout(Duration::from_secs(19));
/// assert_eq!(config.threads, 3);
/// ```
///
/// A struct literal does not compile, as a later field would break it:
///
/// ```compile_fail,E0639
/// use grafeo_engine::Config;
///
/// let config = Config {
///     threads: 3,
///     ..Config::in_memory()
/// };
/// ```
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)] // Config structs naturally have many boolean flags
#[non_exhaustive]
pub struct Config {
    /// Graph data model (LPG or RDF). Immutable after database creation.
    pub graph_model: GraphModel,
    /// Path to the database directory (None for in-memory only).
    pub path: Option<PathBuf>,

    /// Memory limit in bytes (None for unlimited).
    pub memory_limit: Option<usize>,

    /// Path for spilling data to disk under memory pressure.
    pub spill_path: Option<PathBuf>,

    /// Number of worker threads for query execution.
    pub threads: usize,

    /// Whether new commits are logged to the sidecar WAL (`<path>.wal/`) for
    /// durability, in a build with the `wal` feature (a build without it logs
    /// no commit, whatever this says). With it off, a commit reaches the file
    /// only at the next checkpoint (`close()`, `wal_checkpoint()` or the
    /// periodic one), so a crash loses the commits since the last one.
    ///
    /// It never hides commits already in a WAL: in a build with the `wal`
    /// feature an open replays a sidecar WAL that a writer left without
    /// `close()` either way. A read-write open with it off then writes those
    /// commits to the file and removes the WAL before it returns, so no later
    /// crash can replay that WAL over newer data; a read-only open only
    /// replays them into memory.
    ///
    /// On for [`Config::persistent()`], off for [`Config::in_memory()`] and
    /// [`Config::read_only()`]; [`Config::without_wal()`] turns it off.
    pub wal_enabled: bool,

    /// Whether to maintain backward edges.
    pub backward_edges: bool,

    /// Whether to enable query logging.
    pub query_logging: bool,

    /// Adaptive execution configuration. It has no effect: no query reads
    /// it.
    #[deprecated(
        since = "0.6.0",
        note = "adaptive execution was never wired in and has no effect; removed in 0.7.0"
    )]
    #[expect(
        deprecated,
        reason = "the deprecated adaptive configuration names itself until 0.7.0 removes it"
    )]
    pub adaptive: AdaptiveConfig,

    /// Whether to use factorized execution for multi-hop queries.
    ///
    /// When enabled, consecutive MATCH expansions are executed using factorized
    /// representation which avoids Cartesian product materialization. This provides
    /// 5-100x speedup for multi-hop queries with high fan-out.
    ///
    /// Enabled by default.
    pub factorized_execution: bool,

    /// Whether every query without `ORDER BY` returns its rows in random
    /// order.
    ///
    /// Without `ORDER BY` the row order is unspecified: it can change between
    /// runs, builds and versions. This test option makes that visible, so
    /// tests find code that relies on an order anyway. A streamed result is
    /// shuffled within each chunk, so the stream keeps its bounded memory.
    /// Default: `false`.
    pub shuffle_unordered: bool,

    /// WAL durability mode. Only used when `wal_enabled` is true.
    pub wal_durability: DurabilityMode,

    /// Storage format for persistent databases: what a new path becomes.
    ///
    /// With `Auto` (the default) a new path is a single file whatever its
    /// extension; an existing path opens as what it holds (see
    /// [`StorageFormat`]).
    pub storage_format: StorageFormat,

    /// Whether to enable catalog schema constraint enforcement.
    ///
    /// When true, the catalog enforces label, edge type, and property constraints
    /// (e.g. required properties, uniqueness). The server sets this for JSON
    /// Schema databases and populates constraints after creation.
    pub schema_constraints: bool,

    /// Maximum time a single query may run before being cancelled.
    ///
    /// When set, the executor checks the deadline between operator batches and
    /// returns `QueryError::timeout()` if the wall-clock limit is exceeded.
    /// `None` means no timeout (queries may run indefinitely).
    ///
    /// Default: 30 seconds. Use `with_query_timeout()` to change or
    /// `without_query_timeout()` to disable.
    pub query_timeout: Option<Duration>,

    /// Maximum size in bytes for a single property value.
    ///
    /// When set, `set_node_property()` and `set_edge_property()` reject
    /// values whose `estimated_size_bytes()` exceeds this limit.
    /// `None` means no limit (any size is accepted).
    ///
    /// Default: 16 MiB. Use `with_max_property_size()` to change or
    /// `without_max_property_size()` to disable.
    pub max_property_size: Option<usize>,

    /// Run MVCC version garbage collection every N commits.
    ///
    /// Old versions that are no longer visible to any active transaction are
    /// pruned to reclaim memory. Set to 0 to disable automatic GC.
    pub gc_interval: usize,

    /// Access mode: read-write (default) or read-only.
    ///
    /// Read-only mode uses a shared file lock, allowing multiple processes to
    /// read the same database concurrently. Mutations are rejected at the
    /// session level.
    pub access_mode: AccessMode,

    /// Whether CDC (Change Data Capture) is enabled for new sessions by default.
    ///
    /// When `true`, sessions created via [`crate::GrafeoDB::session()`]
    /// automatically track all mutations. Individual sessions can override
    /// this via [`crate::GrafeoDB::session_with_cdc()`]. The `cdc` feature
    /// flag must be compiled in for CDC to function; this field only controls
    /// runtime activation.
    ///
    /// Default: `false` (CDC is opt-in to avoid overhead on the mutation
    /// hot path).
    pub cdc_enabled: bool,

    /// CDC event retention policy.
    ///
    /// Controls how many events the CDC log retains in memory. By default,
    /// retains up to 1,000 epochs and 100,000 events.
    /// [`CdcRetentionConfig::unlimited()`](crate::cdc::CdcRetentionConfig::unlimited)
    /// disables pruning, but beware of unbounded memory growth on
    /// long-running instances. Set it with [`Config::with_cdc_retention()`].
    #[cfg(feature = "cdc")]
    pub cdc_retention: crate::cdc::CdcRetentionConfig,

    /// Per-section memory configuration.
    ///
    /// Maps `SectionType` to `SectionMemoryConfig` for sections that need
    /// custom budgets or tier pinning. Sections not listed here use the
    /// global `memory_limit` budget with automatic management.
    pub section_configs: hashbrown::HashMap<
        grafeo_common::storage::SectionType,
        grafeo_common::storage::SectionMemoryConfig,
    >,

    /// Interval between automatic checkpoints.
    ///
    /// When set, the engine periodically flushes dirty sections to the
    /// `.grafeo` container and truncates the WAL. `None` means checkpoints
    /// only happen on explicit `wal_checkpoint()` or database close.
    pub checkpoint_interval: Option<Duration>,

    /// Encryption at rest, set with [`Config::with_encryption()`].
    ///
    /// When set, the database file (`.grafeo`) and its sidecar WAL are
    /// encrypted with AES-256-GCM, with keys the key chain derives for this
    /// database (see [`EncryptionConfig`]). A new database is created
    /// encrypted; an encrypted database opens only with the key chain it was
    /// created with, and an unencrypted one only without a key. A database
    /// written by 0.5.x (a file or a WAL directory) is migrated into an
    /// encrypted file by a read-write open (the kept `.pre-0.6` copy stays
    /// unencrypted).
    ///
    /// Requires a persistent database: [`validate`](Self::validate) refuses
    /// it without a path.
    ///
    /// An encrypted database spills nothing to disk: it gets no default spill
    /// path, and [`validate`](Self::validate) refuses an explicit
    /// [`spill_path`](Self::spill_path) and a section pinned to
    /// [`TierOverride::ForceDisk`](grafeo_common::storage::TierOverride::ForceDisk)
    /// in [`section_configs`](Self::section_configs), because spill files are
    /// not encrypted. A memory limit therefore cannot move its data to disk.
    ///
    /// Not encrypted: the bytes of
    /// [`export_snapshot`](crate::GrafeoDB::export_snapshot), and an
    /// in-memory copy made with [`to_memory`](crate::GrafeoDB::to_memory)
    /// (it has no key, so a copy saved from it is plaintext).
    #[cfg(feature = "encryption")]
    pub encryption: Option<EncryptionConfig>,
}

/// Configuration for adaptive query execution.
///
/// Adaptive execution was meant to monitor actual row counts during query
/// processing and re-optimize when estimates are significantly wrong. It was
/// never wired in: no setting here changes how a query runs.
#[derive(Debug, Clone)]
#[deprecated(
    since = "0.6.0",
    note = "adaptive execution was never wired in and has no effect; removed in 0.7.0"
)]
pub struct AdaptiveConfig {
    /// Whether adaptive execution is enabled.
    pub enabled: bool,

    /// Deviation threshold that triggers re-optimization.
    ///
    /// A value of 3.0 means re-optimization is triggered when actual cardinality
    /// is more than 3x or less than 1/3x the estimated value.
    pub threshold: f64,

    /// Minimum number of rows before considering re-optimization.
    ///
    /// Helps avoid thrashing on small result sets.
    pub min_rows: u64,

    /// Maximum number of re-optimizations allowed per query.
    pub max_reoptimizations: usize,
}

#[expect(
    deprecated,
    reason = "the deprecated adaptive configuration names itself until 0.7.0 removes it"
)]
impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: 3.0,
            min_rows: 1000,
            max_reoptimizations: 3,
        }
    }
}

#[expect(
    deprecated,
    reason = "the deprecated adaptive configuration names itself until 0.7.0 removes it"
)]
impl AdaptiveConfig {
    /// Creates a disabled adaptive config.
    #[must_use]
    #[deprecated(
        since = "0.6.0",
        note = "adaptive execution was never wired in and has no effect; removed in 0.7.0"
    )]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Sets the deviation threshold.
    #[must_use]
    #[deprecated(
        since = "0.6.0",
        note = "adaptive execution was never wired in and has no effect; removed in 0.7.0"
    )]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Sets the minimum rows before re-optimization.
    #[must_use]
    #[deprecated(
        since = "0.6.0",
        note = "adaptive execution was never wired in and has no effect; removed in 0.7.0"
    )]
    pub fn with_min_rows(mut self, min_rows: u64) -> Self {
        self.min_rows = min_rows;
        self
    }

    /// Sets the maximum number of re-optimizations.
    #[must_use]
    #[deprecated(
        since = "0.6.0",
        note = "adaptive execution was never wired in and has no effect; removed in 0.7.0"
    )]
    pub fn with_max_reoptimizations(mut self, max: usize) -> Self {
        self.max_reoptimizations = max;
        self
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            graph_model: GraphModel::default(),
            path: None,
            memory_limit: None,
            spill_path: None,
            threads: num_cpus::get(),
            wal_enabled: true,
            backward_edges: true,
            query_logging: false,
            #[expect(
                deprecated,
                reason = "the deprecated adaptive configuration names itself until 0.7.0 removes it"
            )]
            adaptive: AdaptiveConfig::default(),
            factorized_execution: true,
            shuffle_unordered: false,
            wal_durability: DurabilityMode::default(),
            storage_format: StorageFormat::default(),
            schema_constraints: false,
            query_timeout: Some(Duration::from_secs(30)),
            max_property_size: Some(16 * 1024 * 1024), // 16 MiB
            gc_interval: 100,
            access_mode: AccessMode::default(),
            cdc_enabled: false,
            #[cfg(feature = "cdc")]
            cdc_retention: crate::cdc::CdcRetentionConfig::default(),
            section_configs: hashbrown::HashMap::new(),
            checkpoint_interval: None,
            #[cfg(feature = "encryption")]
            encryption: None,
        }
    }
}

impl Config {
    /// Creates a new configuration for an in-memory database.
    #[must_use]
    pub fn in_memory() -> Self {
        Self {
            path: None,
            wal_enabled: false,
            ..Default::default()
        }
    }

    /// Creates a new configuration for a persistent database.
    #[must_use]
    pub fn persistent(path: impl Into<PathBuf>) -> Self {
        Self {
            path: Some(path.into()),
            wal_enabled: true,
            ..Default::default()
        }
    }

    /// Sets the memory limit.
    #[must_use]
    pub fn with_memory_limit(mut self, limit: usize) -> Self {
        self.memory_limit = Some(limit);
        self
    }

    /// The bytes one path search of a query may hold (the paths a
    /// variable-length or shortest-path search keeps at once): a quarter of
    /// [`memory_limit`](Self::memory_limit) when it is set, else 256 MiB. A
    /// search that would hold more fails with an error instead of growing.
    #[must_use]
    pub(crate) fn path_search_budget(&self) -> usize {
        self.memory_limit.map_or(
            grafeo_core::execution::operators::DEFAULT_PATH_SEARCH_BUDGET,
            |limit| (limit / 4).max(1),
        )
    }

    /// Sets the number of worker threads.
    #[must_use]
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// Turns the sidecar WAL off: a commit then reaches the file only at the
    /// next checkpoint (`close()`, `wal_checkpoint()` or the periodic one),
    /// so a crash loses the commits since the last one. An open still
    /// replays a WAL that a writer left without `close()` (see
    /// [`Config::wal_enabled`]).
    #[must_use]
    pub fn without_wal(mut self) -> Self {
        self.wal_enabled = false;
        self
    }

    /// Disables backward edges.
    #[must_use]
    pub fn without_backward_edges(mut self) -> Self {
        self.backward_edges = false;
        self
    }

    /// Enables query logging.
    #[must_use]
    pub fn with_query_logging(mut self) -> Self {
        self.query_logging = true;
        self
    }

    /// Sets the memory budget as a fraction of system RAM.
    #[must_use]
    pub fn with_memory_fraction(mut self, fraction: f64) -> Self {
        use grafeo_common::memory::buffer::BufferManagerConfig;
        let system_memory = BufferManagerConfig::detect_system_memory();
        // reason: product of system RAM and a 0..1 fraction is always a valid positive usize
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let budget = (system_memory as f64 * fraction) as usize;
        self.memory_limit = Some(budget);
        self
    }

    /// Sets the spill directory for out-of-core processing.
    #[must_use]
    pub fn with_spill_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.spill_path = Some(path.into());
        self
    }

    /// Sets the adaptive execution configuration, which has no effect (see
    /// [`AdaptiveConfig`]).
    #[must_use]
    #[deprecated(
        since = "0.6.0",
        note = "adaptive execution was never wired in and has no effect; removed in 0.7.0"
    )]
    #[expect(
        deprecated,
        reason = "the deprecated adaptive configuration names itself until 0.7.0 removes it"
    )]
    pub fn with_adaptive(mut self, adaptive: AdaptiveConfig) -> Self {
        self.adaptive = adaptive;
        self
    }

    /// Disables adaptive execution, which has no effect (see
    /// [`AdaptiveConfig`]).
    #[must_use]
    #[deprecated(
        since = "0.6.0",
        note = "adaptive execution was never wired in and has no effect; removed in 0.7.0"
    )]
    #[expect(
        deprecated,
        reason = "the deprecated adaptive configuration names itself until 0.7.0 removes it"
    )]
    pub fn without_adaptive(mut self) -> Self {
        self.adaptive.enabled = false;
        self
    }

    /// Disables factorized execution for multi-hop queries.
    ///
    /// This reverts to the traditional flat execution model where each expansion
    /// creates a full Cartesian product. Only use this if you encounter issues
    /// with factorized execution.
    #[must_use]
    pub fn without_factorized_execution(mut self) -> Self {
        self.factorized_execution = false;
        self
    }

    /// Sets whether queries without `ORDER BY` return their rows in random
    /// order, a test option that finds code relying on a row order that is
    /// unspecified (see [`Config::shuffle_unordered`]).
    #[must_use]
    pub fn with_shuffle_unordered(mut self, shuffle: bool) -> Self {
        self.shuffle_unordered = shuffle;
        self
    }

    /// Sets the graph data model.
    #[must_use]
    pub fn with_graph_model(mut self, model: GraphModel) -> Self {
        self.graph_model = model;
        self
    }

    /// Sets the WAL durability mode.
    #[must_use]
    pub fn with_wal_durability(mut self, mode: DurabilityMode) -> Self {
        self.wal_durability = mode;
        self
    }

    /// Sets the storage format for persistent databases.
    #[must_use]
    pub fn with_storage_format(mut self, format: StorageFormat) -> Self {
        self.storage_format = format;
        self
    }

    /// Enables catalog schema constraint enforcement.
    #[must_use]
    pub fn with_schema_constraints(mut self) -> Self {
        self.schema_constraints = true;
        self
    }

    /// Sets the maximum time a query may run before being cancelled.
    #[must_use]
    pub fn with_query_timeout(mut self, timeout: Duration) -> Self {
        self.query_timeout = Some(timeout);
        self
    }

    /// Disables the query timeout, allowing queries to run indefinitely.
    #[must_use]
    pub fn without_query_timeout(mut self) -> Self {
        self.query_timeout = None;
        self
    }

    /// Sets the maximum size in bytes for a single property value.
    #[must_use]
    pub fn with_max_property_size(mut self, size: usize) -> Self {
        self.max_property_size = Some(size);
        self
    }

    /// Disables the property value size limit.
    #[must_use]
    pub fn without_max_property_size(mut self) -> Self {
        self.max_property_size = None;
        self
    }

    /// Sets the MVCC garbage collection interval (every N commits).
    ///
    /// Set to 0 to disable automatic GC.
    #[must_use]
    pub fn with_gc_interval(mut self, interval: usize) -> Self {
        self.gc_interval = interval;
        self
    }

    /// Sets the access mode (read-write or read-only).
    #[must_use]
    pub fn with_access_mode(mut self, mode: AccessMode) -> Self {
        self.access_mode = mode;
        self
    }

    /// Shorthand for opening a persistent database in read-only mode.
    ///
    /// Uses a shared file lock, allowing multiple processes to read the same
    /// `.grafeo` file concurrently. Mutations are rejected at the session level.
    #[must_use]
    pub fn read_only(path: impl Into<PathBuf>) -> Self {
        Self {
            path: Some(path.into()),
            wal_enabled: false,
            access_mode: AccessMode::ReadOnly,
            ..Default::default()
        }
    }

    /// Enables CDC (Change Data Capture) for all new sessions by default.
    ///
    /// Sessions created via [`crate::GrafeoDB::session()`] will automatically
    /// track mutations. Individual sessions can still opt out via
    /// [`crate::GrafeoDB::session_with_cdc()`].
    ///
    /// Requires the `cdc` feature flag to be compiled in.
    #[must_use]
    pub fn with_cdc(mut self) -> Self {
        self.cdc_enabled = true;
        self
    }

    /// Sets how many CDC events the change history keeps (see
    /// [`Config::cdc_retention`]); garbage collection drops the oldest
    /// events beyond it.
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
    /// assert_eq!(config.cdc_retention.max_events, Some(88_000));
    /// ```
    #[cfg(feature = "cdc")]
    #[must_use]
    pub fn with_cdc_retention(mut self, retention: crate::cdc::CdcRetentionConfig) -> Self {
        self.cdc_retention = retention;
        self
    }

    /// Sets memory configuration for a specific section type.
    ///
    /// Use this to cap a section's RAM usage or pin it to a storage tier.
    /// Sections without explicit config use the global `memory_limit` budget.
    ///
    /// # Examples
    ///
    /// ```
    /// # use grafeo_engine::Config;
    /// use grafeo_common::storage::{SectionType, SectionMemoryConfig, TierOverride};
    ///
    /// let config = Config::in_memory()
    ///     .with_section_config(SectionType::VectorStore, SectionMemoryConfig {
    ///         max_ram: Some(500 * 1024 * 1024), // 500 MB cap
    ///         tier: TierOverride::Auto,
    ///     });
    /// ```
    #[must_use]
    pub fn with_section_config(
        mut self,
        section_type: grafeo_common::storage::SectionType,
        config: grafeo_common::storage::SectionMemoryConfig,
    ) -> Self {
        self.section_configs.insert(section_type, config);
        self
    }

    /// Pins a section to a specific storage tier (Phase 8d convenience).
    ///
    /// Shorthand for `with_section_config(section_type, SectionMemoryConfig {
    /// tier, max_ram: None })`. Pass [`TierOverride::ForceDisk`] to spill the
    /// section at database open, [`TierOverride::ForceRam`] to declare it
    /// must stay in RAM (declarative only until Phase 8g), or
    /// [`TierOverride::Auto`] (the default).
    ///
    /// # Examples
    ///
    /// ```
    /// # use grafeo_engine::Config;
    /// use grafeo_common::storage::{SectionType, TierOverride};
    ///
    /// // Force the LPG compact base to mmap mode at open.
    /// let config = Config::in_memory()
    ///     .with_section_tier(SectionType::CompactStore, TierOverride::ForceDisk);
    /// ```
    ///
    /// [`TierOverride::ForceDisk`]: grafeo_common::storage::TierOverride::ForceDisk
    /// [`TierOverride::ForceRam`]: grafeo_common::storage::TierOverride::ForceRam
    /// [`TierOverride::Auto`]: grafeo_common::storage::TierOverride::Auto
    #[must_use]
    pub fn with_section_tier(
        self,
        section_type: grafeo_common::storage::SectionType,
        tier: grafeo_common::storage::TierOverride,
    ) -> Self {
        let existing_max_ram = self
            .section_configs
            .get(&section_type)
            .and_then(|c| c.max_ram);
        self.with_section_config(
            section_type,
            grafeo_common::storage::SectionMemoryConfig {
                max_ram: existing_max_ram,
                tier,
            },
        )
    }

    /// Sets the automatic checkpoint interval.
    ///
    /// When set, the engine periodically flushes dirty sections to disk.
    /// Typical values: 30-300 seconds.
    #[must_use]
    pub fn with_checkpoint_interval(mut self, interval: Duration) -> Self {
        self.checkpoint_interval = Some(interval);
        self
    }

    /// Encrypts the database at rest with the keys `encryption` derives (see
    /// [`Config::encryption`] for what is encrypted and which settings it
    /// refuses).
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use grafeo_common::encryption::KeyChain;
    /// use grafeo_engine::Config;
    /// use grafeo_engine::config::EncryptionConfig;
    ///
    /// // 32 bytes from your key management (a KMS, a secrets manager, an HSM).
    /// let master_key = [19; 32];
    /// let config = Config::persistent("berlin.grafeo")
    ///     .with_encryption(EncryptionConfig::new(Arc::new(KeyChain::new(master_key))));
    /// assert!(config.validate().is_ok());
    /// ```
    #[cfg(feature = "encryption")]
    #[must_use]
    pub fn with_encryption(mut self, encryption: EncryptionConfig) -> Self {
        self.encryption = Some(encryption);
        self
    }

    /// Validates the configuration, returning an error for invalid combinations.
    ///
    /// Called automatically by [`GrafeoDB::with_config()`](crate::GrafeoDB::with_config).
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if any setting is invalid.
    pub fn validate(&self) -> std::result::Result<(), ConfigError> {
        if let Some(limit) = self.memory_limit
            && limit == 0
        {
            return Err(ConfigError::ZeroMemoryLimit);
        }

        if self.threads == 0 {
            return Err(ConfigError::ZeroThreads);
        }

        if self.wal_durability
            == (DurabilityMode::Adaptive {
                target_interval_ms: 0,
            })
        {
            return Err(ConfigError::ZeroAdaptiveFlushInterval);
        }

        #[cfg(not(feature = "triple-store"))]
        if self.graph_model == GraphModel::Rdf {
            return Err(ConfigError::RdfFeatureRequired);
        }

        #[cfg(feature = "encryption")]
        if self.encryption.is_some() {
            if self.path.is_none() {
                return Err(ConfigError::EncryptionRequiresPersistentPath);
            }
            if self.spill_path.is_some() {
                return Err(ConfigError::EncryptionWithSpillPath);
            }
            // The first such section in section type order, so the error
            // does not depend on the map's iteration order.
            let forced_to_disk = self
                .section_configs
                .iter()
                .filter(|(_, section)| {
                    section.tier == grafeo_common::storage::TierOverride::ForceDisk
                })
                .map(|(section_type, _)| *section_type)
                .min_by_key(|section_type| section_type.to_u8());
            if let Some(section_type) = forced_to_disk {
                return Err(ConfigError::EncryptionWithForceDisk(section_type));
            }
        }

        Ok(())
    }
}

/// Helper function to get CPU count (fallback implementation).
mod num_cpus {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn get() -> usize {
        std::thread::available_parallelism().map_or(4, |n| n.get())
    }

    #[cfg(target_arch = "wasm32")]
    pub fn get() -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert_eq!(config.graph_model, GraphModel::Lpg);
        assert!(config.path.is_none());
        assert!(config.memory_limit.is_none());
        assert!(config.spill_path.is_none());
        assert!(config.threads > 0);
        assert!(config.wal_enabled);
        assert!(config.backward_edges);
        assert!(!config.query_logging);
        assert!(config.factorized_execution);
        assert_eq!(config.wal_durability, DurabilityMode::default());
        assert!(!config.schema_constraints);
        assert_eq!(config.query_timeout, Some(Duration::from_secs(30)));
        assert_eq!(config.gc_interval, 100);
    }

    #[test]
    fn test_config_in_memory() {
        let config = Config::in_memory();
        assert!(config.path.is_none());
        assert!(!config.wal_enabled);
        assert!(config.backward_edges);
    }

    #[test]
    fn test_config_persistent() {
        let config = Config::persistent("/tmp/test_db");
        assert_eq!(
            config.path.as_deref(),
            Some(std::path::Path::new("/tmp/test_db"))
        );
        assert!(config.wal_enabled);
    }

    #[test]
    fn test_config_with_memory_limit() {
        let config = Config::in_memory().with_memory_limit(1024 * 1024);
        assert_eq!(config.memory_limit, Some(1024 * 1024));
    }

    #[test]
    fn test_config_with_threads() {
        let config = Config::in_memory().with_threads(8);
        assert_eq!(config.threads, 8);
    }

    #[test]
    fn test_config_without_backward_edges() {
        let config = Config::in_memory().without_backward_edges();
        assert!(!config.backward_edges);
    }

    #[test]
    fn test_config_with_query_logging() {
        let config = Config::in_memory().with_query_logging();
        assert!(config.query_logging);
    }

    #[test]
    fn test_config_with_spill_path() {
        let config = Config::in_memory().with_spill_path("/tmp/spill");
        assert_eq!(
            config.spill_path.as_deref(),
            Some(std::path::Path::new("/tmp/spill"))
        );
    }

    #[test]
    fn test_config_with_memory_fraction() {
        let config = Config::in_memory().with_memory_fraction(0.5);
        assert!(config.memory_limit.is_some());
        assert!(config.memory_limit.unwrap() > 0);
    }

    #[test]
    fn test_config_without_factorized_execution() {
        let config = Config::in_memory().without_factorized_execution();
        assert!(!config.factorized_execution);
    }

    #[test]
    fn test_config_builder_chaining() {
        let config = Config::persistent("/tmp/db")
            .with_memory_limit(512 * 1024 * 1024)
            .with_threads(4)
            .with_query_logging()
            .without_backward_edges()
            .with_spill_path("/tmp/spill");

        assert!(config.path.is_some());
        assert_eq!(config.memory_limit, Some(512 * 1024 * 1024));
        assert_eq!(config.threads, 4);
        assert!(config.query_logging);
        assert!(!config.backward_edges);
        assert!(config.spill_path.is_some());
    }

    // --- GraphModel tests ---

    #[test]
    fn test_graph_model_default_is_lpg() {
        assert_eq!(GraphModel::default(), GraphModel::Lpg);
    }

    #[test]
    fn test_graph_model_display() {
        assert_eq!(GraphModel::Lpg.to_string(), "LPG");
        assert_eq!(GraphModel::Rdf.to_string(), "RDF");
    }

    #[test]
    fn test_config_with_graph_model() {
        let config = Config::in_memory().with_graph_model(GraphModel::Rdf);
        assert_eq!(config.graph_model, GraphModel::Rdf);
    }

    // --- DurabilityMode tests ---

    #[test]
    fn test_durability_mode_default_is_batch() {
        let mode = DurabilityMode::default();
        assert_eq!(
            mode,
            DurabilityMode::Batch {
                max_delay_ms: 100,
                max_records: 1000
            }
        );
    }

    #[test]
    fn test_config_with_wal_durability() {
        let config = Config::persistent("/tmp/db").with_wal_durability(DurabilityMode::Sync);
        assert_eq!(config.wal_durability, DurabilityMode::Sync);
    }

    #[test]
    fn test_config_with_wal_durability_nosync() {
        let config = Config::persistent("/tmp/db").with_wal_durability(DurabilityMode::NoSync);
        assert_eq!(config.wal_durability, DurabilityMode::NoSync);
    }

    #[test]
    fn test_config_with_wal_durability_adaptive() {
        let config = Config::persistent("/tmp/db").with_wal_durability(DurabilityMode::Adaptive {
            target_interval_ms: 50,
        });
        assert_eq!(
            config.wal_durability,
            DurabilityMode::Adaptive {
                target_interval_ms: 50
            }
        );
    }

    // --- max_property_size tests ---

    #[test]
    fn test_config_default_max_property_size() {
        let config = Config::in_memory();
        assert_eq!(config.max_property_size, Some(16 * 1024 * 1024));
    }

    #[test]
    fn test_config_with_max_property_size() {
        let config = Config::in_memory().with_max_property_size(1024);
        assert_eq!(config.max_property_size, Some(1024));
    }

    #[test]
    fn test_config_without_max_property_size() {
        let config = Config::in_memory().without_max_property_size();
        assert!(config.max_property_size.is_none());
    }

    // --- schema_constraints tests ---

    #[test]
    fn test_config_with_schema_constraints() {
        let config = Config::in_memory().with_schema_constraints();
        assert!(config.schema_constraints);
    }

    // --- query_timeout tests ---

    #[test]
    fn test_config_with_query_timeout() {
        let config = Config::in_memory().with_query_timeout(Duration::from_mins(1));
        assert_eq!(config.query_timeout, Some(Duration::from_mins(1)));
    }

    #[test]
    fn test_config_without_query_timeout() {
        let config = Config::in_memory().without_query_timeout();
        assert!(config.query_timeout.is_none());
    }

    #[test]
    fn test_config_default_query_timeout() {
        let config = Config::in_memory();
        assert_eq!(config.query_timeout, Some(Duration::from_secs(30)));
    }

    // --- gc_interval tests ---

    #[test]
    fn test_config_with_gc_interval() {
        let config = Config::in_memory().with_gc_interval(50);
        assert_eq!(config.gc_interval, 50);
    }

    #[test]
    fn test_config_gc_disabled() {
        let config = Config::in_memory().with_gc_interval(0);
        assert_eq!(config.gc_interval, 0);
    }

    // --- validate() tests ---

    #[test]
    fn test_validate_default_config() {
        assert!(Config::default().validate().is_ok());
    }

    #[test]
    fn test_validate_in_memory_config() {
        assert!(Config::in_memory().validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_zero_memory_limit() {
        let config = Config::in_memory().with_memory_limit(0);
        assert_eq!(config.validate(), Err(ConfigError::ZeroMemoryLimit));
    }

    #[test]
    fn test_validate_rejects_zero_threads() {
        let config = Config::in_memory().with_threads(0);
        assert_eq!(config.validate(), Err(ConfigError::ZeroThreads));
    }

    /// A zero interval made the adaptive WAL flusher sync in a busy loop.
    #[test]
    fn test_validate_rejects_zero_adaptive_interval() {
        let config = Config::in_memory().with_wal_durability(DurabilityMode::Adaptive {
            target_interval_ms: 0,
        });
        assert_eq!(
            config.validate(),
            Err(ConfigError::ZeroAdaptiveFlushInterval)
        );
    }

    /// The constructors build the variants with their limits in whole
    /// milliseconds; the default is a batch of 100 ms and 1,000 records.
    #[test]
    fn durability_constructors_build_the_variants() {
        assert_eq!(
            DurabilityMode::batch(Duration::from_millis(100), 1000),
            DurabilityMode::default()
        );
        assert_eq!(
            DurabilityMode::batch(Duration::from_millis(19), 88),
            DurabilityMode::Batch {
                max_delay_ms: 19,
                max_records: 88
            }
        );
        assert_eq!(
            DurabilityMode::adaptive(Duration::from_millis(3)),
            DurabilityMode::Adaptive {
                target_interval_ms: 3
            }
        );
        assert_eq!(
            DurabilityMode::adaptive(Duration::from_micros(88_999)),
            DurabilityMode::Adaptive {
                target_interval_ms: 88
            },
            "a part of a millisecond is dropped"
        );
        assert_eq!(
            DurabilityMode::batch(Duration::MAX, u64::MAX),
            DurabilityMode::Batch {
                max_delay_ms: u64::MAX,
                max_records: u64::MAX
            },
            "a delay beyond u64 milliseconds saturates"
        );
    }

    /// An interval under a millisecond is zero milliseconds, which would sync
    /// in a busy loop: `validate` refuses it.
    #[test]
    fn an_adaptive_interval_under_a_millisecond_is_invalid() {
        let config = Config::persistent("paris.grafeo")
            .with_wal_durability(DurabilityMode::adaptive(Duration::from_micros(880)));
        assert_eq!(
            config.validate(),
            Err(ConfigError::ZeroAdaptiveFlushInterval)
        );
    }

    #[cfg(not(feature = "triple-store"))]
    #[test]
    fn test_validate_rejects_rdf_without_feature() {
        let config = Config::in_memory().with_graph_model(GraphModel::Rdf);
        assert_eq!(config.validate(), Err(ConfigError::RdfFeatureRequired));
    }

    #[test]
    fn test_config_error_display() {
        assert_eq!(
            ConfigError::ZeroMemoryLimit.to_string(),
            "memory_limit must be greater than zero"
        );
        assert_eq!(
            ConfigError::ZeroThreads.to_string(),
            "threads must be greater than zero"
        );
        assert_eq!(
            ConfigError::RdfFeatureRequired.to_string(),
            "RDF graph model requires the `rdf` feature flag to be enabled"
        );
    }

    /// Encryption needs a path to write the encrypted file to: an in-memory
    /// configuration with a key is invalid, a persistent or read-only one is
    /// not.
    #[cfg(all(feature = "encryption", not(miri)))]
    #[test]
    fn encryption_without_a_persistent_path_is_invalid() {
        let encryption = EncryptionConfig::new(std::sync::Arc::new(
            grafeo_common::encryption::KeyChain::new([3; 32]),
        ));
        let mut in_memory = Config::in_memory();
        in_memory.encryption = Some(encryption.clone());
        assert_eq!(
            in_memory.validate(),
            Err(ConfigError::EncryptionRequiresPersistentPath)
        );
        assert!(
            ConfigError::EncryptionRequiresPersistentPath
                .to_string()
                .contains("requires a persistent database path")
        );
        for mut config in [
            Config::persistent("amsterdam.grafeo"),
            Config::read_only("amsterdam.grafeo"),
        ] {
            config.encryption = Some(encryption.clone());
            assert_eq!(config.validate(), Ok(()), "{:?}", config.path);
        }
    }

    /// Spill files are not encrypted: a spill path with encryption is invalid.
    #[cfg(all(feature = "encryption", not(miri)))]
    #[test]
    fn encryption_with_a_spill_path_is_invalid() {
        let mut config = Config::persistent("berlin.grafeo").with_spill_path("berlin.spill");
        assert_eq!(config.validate(), Ok(()), "a spill path alone is fine");
        config.encryption = Some(EncryptionConfig::new(std::sync::Arc::new(
            grafeo_common::encryption::KeyChain::new([19; 32]),
        )));
        assert_eq!(config.validate(), Err(ConfigError::EncryptionWithSpillPath));
        assert!(
            ConfigError::EncryptionWithSpillPath
                .to_string()
                .contains("spill files are not encrypted")
        );
    }

    /// An encrypted database spills nothing to disk, so a section pinned to
    /// disk could not be honored: `TierOverride::ForceDisk` with encryption
    /// is invalid, while `Auto` and `ForceRam` are fine.
    #[cfg(all(feature = "encryption", not(miri)))]
    #[test]
    fn encryption_with_a_section_forced_to_disk_is_invalid() {
        use grafeo_common::storage::{SectionType, TierOverride};

        let encryption = EncryptionConfig::new(std::sync::Arc::new(
            grafeo_common::encryption::KeyChain::new([88; 32]),
        ));
        let mut config = Config::persistent("prague.grafeo")
            .with_section_tier(SectionType::VectorStore, TierOverride::ForceRam)
            .with_section_tier(SectionType::TextIndex, TierOverride::Auto);
        config.encryption = Some(encryption);
        assert_eq!(
            config.validate(),
            Ok(()),
            "Auto and ForceRam keep data in RAM"
        );

        let mut config =
            config.with_section_tier(SectionType::CompactStore, TierOverride::ForceDisk);
        let error = config
            .validate()
            .expect_err("a section forced to disk on an encrypted database")
            .to_string();
        assert!(
            error.contains("disk tier override") && error.contains("CompactStore"),
            "the error names the override and the section: {error}"
        );
        assert_eq!(
            config.validate(),
            Err(ConfigError::EncryptionWithForceDisk(
                SectionType::CompactStore
            ))
        );
        config.encryption = None;
        assert_eq!(config.validate(), Ok(()), "ForceDisk alone is fine");
    }

    // --- Builder chaining with new fields ---

    #[test]
    fn test_config_full_builder_chaining() {
        let config = Config::persistent("/tmp/db")
            .with_graph_model(GraphModel::Lpg)
            .with_memory_limit(512 * 1024 * 1024)
            .with_threads(4)
            .with_query_logging()
            .with_wal_durability(DurabilityMode::Sync)
            .with_schema_constraints()
            .without_backward_edges()
            .with_spill_path("/tmp/spill")
            .with_query_timeout(Duration::from_mins(1));

        assert_eq!(config.graph_model, GraphModel::Lpg);
        assert!(config.path.is_some());
        assert_eq!(config.memory_limit, Some(512 * 1024 * 1024));
        assert_eq!(config.threads, 4);
        assert!(config.query_logging);
        assert_eq!(config.wal_durability, DurabilityMode::Sync);
        assert!(config.schema_constraints);
        assert!(!config.backward_edges);
        assert!(config.spill_path.is_some());
        assert_eq!(config.query_timeout, Some(Duration::from_mins(1)));
        assert!(config.validate().is_ok());
    }

    /// Every setting can be changed with a constructor or a `with_*` or
    /// `without_*` method, so code outside this crate never needs a struct
    /// literal. The pattern names every field: a new setting does not
    /// compile here until it is listed with the method that sets it.
    #[test]
    fn every_setting_has_a_method() {
        use grafeo_common::storage::{SectionType, TierOverride};

        let config = Config::persistent("amsterdam.grafeo")
            .with_graph_model(GraphModel::Rdf)
            .with_memory_limit(88 * 1024 * 1024)
            .with_spill_path("amsterdam.spill")
            .with_threads(3)
            .without_wal()
            .without_backward_edges()
            .with_query_logging()
            .without_factorized_execution()
            .with_shuffle_unordered(true)
            .with_wal_durability(DurabilityMode::batch(Duration::from_millis(19), 88))
            .with_storage_format(StorageFormat::Auto)
            .with_schema_constraints()
            .with_query_timeout(Duration::from_secs(19))
            .with_max_property_size(88)
            .with_gc_interval(3)
            .with_access_mode(AccessMode::ReadOnly)
            .with_cdc()
            .with_section_tier(SectionType::VectorStore, TierOverride::ForceRam)
            .with_checkpoint_interval(Duration::from_secs(88));
        #[cfg(feature = "cdc")]
        let config = config.with_cdc_retention(
            crate::cdc::CdcRetentionConfig::unlimited()
                .with_max_epochs(19)
                .with_max_events(88),
        );
        #[cfg(feature = "encryption")]
        let config = config.with_encryption(EncryptionConfig::new(std::sync::Arc::new(
            grafeo_common::encryption::KeyChain::new([3; 32]),
        )));

        #[expect(
            deprecated,
            reason = "the pattern names every field, the deprecated adaptive one too, until 0.7.0 removes it"
        )]
        let Config {
            graph_model,
            path,
            memory_limit,
            spill_path,
            threads,
            wal_enabled,
            backward_edges,
            query_logging,
            // Deprecated: adaptive execution never changed a query.
            adaptive: _,
            factorized_execution,
            shuffle_unordered,
            wal_durability,
            storage_format,
            schema_constraints,
            query_timeout,
            max_property_size,
            gc_interval,
            access_mode,
            cdc_enabled,
            #[cfg(feature = "cdc")]
            cdc_retention,
            section_configs,
            checkpoint_interval,
            #[cfg(feature = "encryption")]
            encryption,
        } = config;

        assert_eq!(graph_model, GraphModel::Rdf);
        assert_eq!(
            path.as_deref(),
            Some(std::path::Path::new("amsterdam.grafeo"))
        );
        assert_eq!(memory_limit, Some(88 * 1024 * 1024));
        assert_eq!(
            spill_path.as_deref(),
            Some(std::path::Path::new("amsterdam.spill"))
        );
        assert_eq!(threads, 3);
        assert!(!wal_enabled, "without_wal turns the WAL off");
        assert!(!backward_edges);
        assert!(query_logging);
        assert!(!factorized_execution);
        assert!(shuffle_unordered);
        assert_eq!(
            wal_durability,
            DurabilityMode::Batch {
                max_delay_ms: 19,
                max_records: 88
            }
        );
        // The only storage format that is not deprecated is the default.
        assert_eq!(storage_format, StorageFormat::Auto);
        assert!(schema_constraints);
        assert_eq!(query_timeout, Some(Duration::from_secs(19)));
        assert_eq!(max_property_size, Some(88));
        assert_eq!(gc_interval, 3);
        assert_eq!(access_mode, AccessMode::ReadOnly);
        assert!(cdc_enabled);
        #[cfg(feature = "cdc")]
        assert_eq!(
            (cdc_retention.max_epochs, cdc_retention.max_events),
            (Some(19), Some(88)),
            "with_cdc_retention sets the retention"
        );
        assert_eq!(
            section_configs
                .get(&SectionType::VectorStore)
                .map(|section| section.tier),
            Some(TierOverride::ForceRam)
        );
        assert_eq!(checkpoint_interval, Some(Duration::from_secs(88)));
        #[cfg(feature = "encryption")]
        assert!(encryption.is_some(), "with_encryption sets the key chain");
    }

    // --- AccessMode tests ---

    #[test]
    fn test_access_mode_default_is_read_write() {
        assert_eq!(AccessMode::default(), AccessMode::ReadWrite);
    }

    #[test]
    fn test_access_mode_display() {
        assert_eq!(AccessMode::ReadWrite.to_string(), "read-write");
        assert_eq!(AccessMode::ReadOnly.to_string(), "read-only");
    }

    #[test]
    fn test_config_with_access_mode() {
        let config = Config::persistent("/tmp/db").with_access_mode(AccessMode::ReadOnly);
        assert_eq!(config.access_mode, AccessMode::ReadOnly);
    }

    #[test]
    fn test_config_read_only() {
        let config = Config::read_only("/tmp/db.grafeo");
        assert_eq!(config.access_mode, AccessMode::ReadOnly);
        assert!(config.path.is_some());
        assert!(!config.wal_enabled);
    }

    #[test]
    fn test_config_default_is_read_write() {
        let config = Config::default();
        assert_eq!(config.access_mode, AccessMode::ReadWrite);
    }

    // --- StorageFormat tests ---

    #[test]
    fn test_storage_format_default_is_auto() {
        assert_eq!(StorageFormat::default(), StorageFormat::Auto);
    }

    #[test]
    #[allow(
        deprecated,
        reason = "the crate names its own deprecated variant, until 0.7.0 removes it"
    )]
    fn test_storage_format_display() {
        assert_eq!(StorageFormat::Auto.to_string(), "auto");
        assert_eq!(StorageFormat::WalDirectory.to_string(), "wal-directory");
        assert_eq!(StorageFormat::SingleFile.to_string(), "single-file");
    }

    #[test]
    #[allow(
        deprecated,
        reason = "the crate names its own deprecated variant, until 0.7.0 removes it"
    )]
    fn test_config_with_storage_format() {
        let config = Config::in_memory().with_storage_format(StorageFormat::SingleFile);
        assert_eq!(config.storage_format, StorageFormat::SingleFile);

        let config2 = Config::in_memory().with_storage_format(StorageFormat::WalDirectory);
        assert_eq!(config2.storage_format, StorageFormat::WalDirectory);
    }

    // --- CDC config tests ---

    #[test]
    fn test_config_with_cdc() {
        let config = Config::in_memory().with_cdc();
        assert!(config.cdc_enabled);
    }

    #[test]
    fn test_config_cdc_default_false() {
        let config = Config::default();
        assert!(!config.cdc_enabled);
    }

    // --- ConfigError as std::error::Error ---

    #[test]
    fn test_config_error_is_std_error() {
        let err = ConfigError::ZeroMemoryLimit;
        // Ensure it implements std::error::Error (no source)
        let dyn_err: &dyn std::error::Error = &err;
        assert!(dyn_err.source().is_none());
        assert!(
            !dyn_err.to_string().is_empty(),
            "dyn_err.to_string() is empty"
        );
    }

    // --- Validate accepts non-zero memory limit ---

    #[test]
    fn test_validate_accepts_nonzero_memory_limit() {
        let config = Config::in_memory().with_memory_limit(1);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_accepts_none_memory_limit() {
        let config = Config::in_memory();
        assert!(config.memory_limit.is_none());
        assert!(config.validate().is_ok());
    }

    // --- DurabilityMode variants ---

    #[test]
    fn test_durability_mode_debug() {
        let sync = DurabilityMode::Sync;
        let debug = format!("{sync:?}");
        assert_eq!(debug, "Sync");

        let no_sync = DurabilityMode::NoSync;
        let debug = format!("{no_sync:?}");
        assert_eq!(debug, "NoSync");
    }

    // --- read_only config ---

    #[test]
    fn test_read_only_config_full() {
        let config = Config::read_only("/tmp/data.grafeo");
        assert_eq!(config.access_mode, AccessMode::ReadOnly);
        assert!(!config.wal_enabled);
        assert!(config.path.is_some());
        // Other defaults should still apply
        assert!(config.backward_edges);
        assert_eq!(config.graph_model, GraphModel::Lpg);
    }
}
