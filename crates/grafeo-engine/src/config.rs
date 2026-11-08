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

/// How long a database keeps history: the committed states a time-travel
/// read (`execute_at_epoch`, `execute_as_of`, [`GrafeoDB::as_of`]) can still ask
/// for. A stored setting (#551): a database file keeps it, and
/// [`GrafeoDB::set_stored_config`] changes it, from then on (what was
/// collected before stays gone; history the old setting kept stays until the
/// new one lets it go).
///
/// Off by default, which costs nothing: garbage collection keeps only what
/// open snapshots read. With a retention window it keeps every version a
/// reader at an epoch committed within the window reads, in memory, in a
/// checkpoint (as the file's `History` chunks, with an index from commit
/// times to epochs) and across a crash. A read before what the database
/// keeps fails with an error, never with a silent answer.
///
/// [`GrafeoDB::as_of`]: crate::GrafeoDB::as_of
/// [`GrafeoDB::set_stored_config`]: crate::GrafeoDB::set_stored_config
///
/// ```
/// use std::time::Duration;
///
/// use grafeo_engine::{Config, HistoryRetention};
///
/// let config = Config::in_memory()
///     .with_history_retention(HistoryRetention::Duration(Duration::from_secs(3_600)));
/// assert_eq!(
///     config.history_retention(),
///     HistoryRetention::Duration(Duration::from_secs(3_600))
/// );
/// assert_eq!(Config::in_memory().history_retention(), HistoryRetention::Off);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum HistoryRetention {
    /// No history: only what open snapshots read is kept.
    #[default]
    Off,
    /// The states committed within this long before now.
    Duration(Duration),
    /// Every state from when history was turned on.
    Forever,
}

/// The cap of everything one open spills to disk at once: sorts' runs,
/// aggregates' partitions and vector indexes' embedding columns together.
///
/// A spill past it is refused with `GRAFEO-S001`, naming the spill limit. Per
/// open, never stored in the database file. Whatever the cap, a spill that
/// would leave less free space on the spill volume than the database needs
/// for its next checkpoint is refused too. An open without a spill directory
/// (in memory without [`Config::with_spill_path`], or encrypted) spills
/// nothing, and the setting does nothing there.
///
/// # Examples
///
/// ```
/// use grafeo_engine::{Config, SpillLimit};
///
/// assert_eq!(Config::in_memory().spill_limit(), SpillLimit::Default);
/// assert_eq!(
///     Config::in_memory().with_spill_limit(1 << 30).spill_limit(),
///     SpillLimit::Bytes(1 << 30)
/// );
/// assert_eq!(
///     Config::in_memory().with_unlimited_spill().spill_limit(),
///     SpillLimit::Unlimited
/// );
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum SpillLimit {
    /// Half the spill volume's free space when the database opens; the
    /// memory limit when that cannot be read (with a warning).
    #[default]
    Default,
    /// At most this many bytes (0: nothing spills).
    Bytes(u64),
    /// No cap: the room for the next checkpoint alone bounds.
    Unlimited,
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
    /// [`Config::with_query_memory_limit`] must be greater than zero.
    ZeroQueryMemoryLimit,
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
    /// A section is pinned to
    /// [`TierOverride::ForceDisk`](grafeo_common::storage::TierOverride::ForceDisk)
    /// that cannot be moved to disk: the override would have no effect.
    ForceDiskOnUnspillableSection(grafeo_common::storage::SectionType),
    /// `wal_segment_size` is outside [`MIN_WAL_SEGMENT_SIZE`] to
    /// [`MAX_WAL_SEGMENT_SIZE`]; holds the size given.
    WalSegmentSizeOutOfBounds(u64),
    /// `checkpoint_wal_size` is below [`MIN_CHECKPOINT_WAL_SIZE`]: a smaller
    /// one would checkpoint after nearly every commit. Holds the size given.
    CheckpointWalSizeTooSmall(u64),
    /// [`Config::with_memory_fraction`] was given an infinite fraction, which
    /// would make the memory limit unbounded.
    MemoryFractionNotFinite,
}

/// Whether a section of `section_type` can be moved to disk: whether its
/// memory consumer really spills it, so that a disk tier override does what
/// it says. Only the vector store does (its embedding columns go to spill
/// files), in a build with `wal` and `vector-index`, whose consumer writes
/// them; a mappable section whose consumer cannot spill (the text and
/// property indexes, the RDF ring, which a spill would copy back onto the
/// heap) or that has no consumer (the compact store, read only from older
/// files) cannot.
pub(crate) fn can_spill(section_type: grafeo_common::storage::SectionType) -> bool {
    cfg!(all(feature = "wal", feature = "vector-index"))
        && matches!(
            section_type,
            grafeo_common::storage::SectionType::VectorStore
        )
}

/// The section types that can be moved to disk, in section type order, as
/// a list for an error message: `A, B and C`.
fn spillable_sections() -> String {
    let names: Vec<String> = (0..=u8::MAX)
        .filter_map(grafeo_common::storage::SectionType::from_u8)
        .filter(|section_type| can_spill(*section_type))
        .map(|section_type| format!("{section_type:?}"))
        .collect();
    match names.split_last() {
        Some((last, [])) => last.clone(),
        Some((last, rest)) => format!("{} and {last}", rest.join(", ")),
        None => "none in this build".to_string(),
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroMemoryLimit => write!(f, "memory_limit must be greater than zero"),
            Self::MemoryFractionNotFinite => write!(f, "memory_fraction must be a finite number"),
            Self::ZeroQueryMemoryLimit => {
                write!(f, "query_memory_limit must be greater than zero")
            }
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
            Self::ForceDiskOnUnspillableSection(section_type) => write!(
                f,
                "the {section_type:?} section cannot be moved to disk, so it takes no disk \
                 tier override; the sections that can are {}",
                spillable_sections()
            ),
            Self::WalSegmentSizeOutOfBounds(bytes) => write!(
                f,
                "wal_segment_size must be between 1 MiB and 1 GiB, and {bytes} bytes is not"
            ),
            Self::CheckpointWalSizeTooSmall(0) => write!(
                f,
                "checkpoint_wal_size must be at least 1 MiB: zero would checkpoint after every \
                 commit; turn automatic checkpoints off with \
                 Config::without_checkpoint_wal_size() instead"
            ),
            Self::CheckpointWalSizeTooSmall(bytes) => write!(
                f,
                "checkpoint_wal_size must be at least 1 MiB, and {bytes} bytes is not; turn \
                 automatic checkpoints off with Config::without_checkpoint_wal_size() instead"
            ),
        }
    }
}

/// Refuses a `checkpoint_wal_size` below [`MIN_CHECKPOINT_WAL_SIZE`]: the one
/// check of [`Config::validate`] and of `GrafeoDB::set_checkpoint_triggers`,
/// so the floor is the same at open and on an open database. `None` (off)
/// passes.
///
/// # Errors
///
/// [`ConfigError::CheckpointWalSizeTooSmall`].
pub(crate) fn check_checkpoint_wal_size(bytes: Option<u64>) -> Result<(), ConfigError> {
    match bytes {
        Some(bytes) if bytes < MIN_CHECKPOINT_WAL_SIZE => {
            Err(ConfigError::CheckpointWalSizeTooSmall(bytes))
        }
        _ => Ok(()),
    }
}

impl std::error::Error for ConfigError {}

/// How much WAL a checkpoint keeps at most beyond the log it covers for
/// incremental backups, by default: 1 GiB (see
/// [`Config::with_wal_retention_limit`]).
pub const DEFAULT_WAL_RETENTION_LIMIT: u64 = 1 << 30;

/// The WAL size past the last checkpoint at which a file database
/// checkpoints by itself, by default: 256 MiB, four segments of the default
/// size (see [`Config::with_checkpoint_wal_size`]).
pub const DEFAULT_CHECKPOINT_WAL_SIZE: u64 = 256 << 20;

/// The smallest `checkpoint_wal_size`: 1 MiB.
pub const MIN_CHECKPOINT_WAL_SIZE: u64 = 1 << 20;

/// The size a WAL segment grows to before the log goes on in a new one, by
/// default: 64 MiB (see [`Config::with_wal_segment_size`]).
pub const DEFAULT_WAL_SEGMENT_SIZE: u64 = 64 << 20;

/// The smallest `wal_segment_size`: 1 MiB.
pub const MIN_WAL_SEGMENT_SIZE: u64 = 1 << 20;

/// The largest `wal_segment_size`: 1 GiB.
pub const MAX_WAL_SEGMENT_SIZE: u64 = 1 << 30;

#[cfg(feature = "wal")]
const _: () = assert!(
    DEFAULT_WAL_SEGMENT_SIZE == grafeo_storage::wal::DEFAULT_SEGMENT_BYTES,
    "the engine's default segment size is the WAL's"
);

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
///     .with_query_timeout(Duration::from_secs(19));
/// assert_eq!(config.memory_limit(), Some(88 * 1024 * 1024));
/// ```
///
/// A struct literal does not compile, as a later field would break it:
///
/// ```compile_fail,E0639
/// use grafeo_engine::Config;
///
/// let config = Config {
///     memory_limit: Some(88),
///     ..Config::in_memory()
/// };
/// ```
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)] // Config structs naturally have many boolean flags
#[non_exhaustive]
pub struct Config {
    /// Graph data model (LPG or RDF). Immutable after database creation.
    #[deprecated(
        since = "0.6.0",
        note = "read it with `Config::graph_model()` and set it with its builder; private in 0.7.0"
    )]
    pub graph_model: GraphModel,
    /// Path to the database directory (None for in-memory only).
    #[deprecated(
        since = "0.6.0",
        note = "read it with `Config::path()` and set it with its builder; private in 0.7.0"
    )]
    pub path: Option<PathBuf>,

    /// Memory limit in bytes (`None`: three quarters of the detected memory, see
    /// [`memory_limit`](Self::memory_limit)).
    #[deprecated(
        since = "0.6.0",
        note = "read it with `Config::memory_limit()` and set it with its builder; private in 0.7.0"
    )]
    pub memory_limit: Option<usize>,

    /// Path for spilling data to disk under memory pressure.
    #[deprecated(
        since = "0.6.0",
        note = "read it with `Config::spill_path()` and set it with its builder; private in 0.7.0"
    )]
    pub spill_path: Option<PathBuf>,

    /// Number of worker threads for query execution. It has no effect:
    /// nothing sizes a pool from it.
    #[deprecated(
        since = "0.6.0",
        note = "nothing sizes a pool from the thread count, so it has no effect; removed in 0.7.0"
    )]
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
    #[deprecated(
        since = "0.6.0",
        note = "read it with `Config::wal_enabled()` and set it with its builder; private in 0.7.0"
    )]
    pub wal_enabled: bool,

    /// Whether to maintain backward edges.
    #[deprecated(
        since = "0.6.0",
        note = "read it with `Config::backward_edges()` and set it with its builder; private in 0.7.0"
    )]
    pub backward_edges: bool,

    /// Whether to enable query logging. It has no effect: no query is
    /// logged.
    #[deprecated(
        since = "0.6.0",
        note = "no query is logged, so it has no effect; removed in 0.7.0"
    )]
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
    #[deprecated(
        since = "0.6.0",
        note = "read it with `Config::factorized_execution()` and set it with its builder; private in 0.7.0"
    )]
    pub factorized_execution: bool,

    /// Whether every query without `ORDER BY` returns its rows in random
    /// order.
    ///
    /// Without `ORDER BY` the row order is unspecified: it can change between
    /// runs, builds and versions. This test option makes that visible, so
    /// tests find code that relies on an order anyway. A streamed result is
    /// shuffled within each chunk, so the stream keeps its bounded memory.
    /// Default: `false`.
    #[deprecated(
        since = "0.6.0",
        note = "read it with `Config::shuffle_unordered()` and set it with its builder; private in 0.7.0"
    )]
    pub shuffle_unordered: bool,

    /// WAL durability mode. Only used when `wal_enabled` is true.
    #[deprecated(
        since = "0.6.0",
        note = "read it with `Config::wal_durability()` and set it with its builder; private in 0.7.0"
    )]
    pub wal_durability: DurabilityMode,

    /// Storage format for persistent databases: what a new path becomes.
    ///
    /// With `Auto` (the default) a new path is a single file whatever its
    /// extension; an existing path opens as what it holds (see
    /// [`StorageFormat`]).
    #[deprecated(
        since = "0.6.0",
        note = "read it with `Config::storage_format()` and set it with its builder; private in 0.7.0"
    )]
    pub storage_format: StorageFormat,

    /// Whether to enable catalog schema constraint enforcement. It has no
    /// effect: the catalog enforces the constraints a graph type or `CREATE
    /// CONSTRAINT` declares either way. A database file still stores it.
    #[deprecated(
        since = "0.6.0",
        note = "the catalog enforces declared constraints either way, so it has no effect; removed in 0.7.0"
    )]
    pub schema_constraints: bool,

    /// Maximum time a single query may run before being cancelled.
    ///
    /// When set, the deadline is checked between the chunks a query returns
    /// and once per chunk inside the operators that read or make many rows
    /// before they return one (aggregates, sorts, joins, filters, `UNWIND`),
    /// so a query fails with a timeout error within about one chunk of work
    /// past the limit. `None` means no timeout (queries may run indefinitely).
    ///
    /// Default: 30 seconds. Use `with_query_timeout()` to change or
    /// `without_query_timeout()` to disable.
    #[deprecated(
        since = "0.6.0",
        note = "read it with `Config::query_timeout()` and set it with its builder; private in 0.7.0"
    )]
    pub query_timeout: Option<Duration>,

    /// Maximum size in bytes for a single property value.
    ///
    /// When set, `set_node_property()` and `set_edge_property()` reject
    /// values whose `estimated_size_bytes()` exceeds this limit.
    /// `None` means no limit (any size is accepted).
    ///
    /// Default: 16 MiB. Use `with_max_property_size()` to change or
    /// `without_max_property_size()` to disable.
    #[deprecated(
        since = "0.6.0",
        note = "read it with `Config::max_property_size()` and set it with its builder; private in 0.7.0"
    )]
    pub max_property_size: Option<usize>,

    /// Run MVCC version garbage collection every N commits.
    ///
    /// Old versions that are no longer visible to any active transaction are
    /// pruned to reclaim memory. Set to 0 to disable automatic GC.
    #[deprecated(
        since = "0.6.0",
        note = "read it with `Config::gc_interval()` and set it with its builder; private in 0.7.0"
    )]
    pub gc_interval: usize,

    /// Access mode: read-write (default) or read-only.
    ///
    /// Read-only mode uses a shared file lock, allowing multiple processes to
    /// read the same database concurrently. Mutations are rejected at the
    /// session level.
    #[deprecated(
        since = "0.6.0",
        note = "read it with `Config::access_mode()` and set it with its builder; private in 0.7.0"
    )]
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
    #[deprecated(
        since = "0.6.0",
        note = "read it with `Config::cdc_enabled()` and set it with its builder; private in 0.7.0"
    )]
    pub cdc_enabled: bool,

    /// CDC event retention policy.
    ///
    /// Controls how many events the CDC log retains in memory. By default,
    /// retains up to 1,000 epochs and 100,000 events.
    /// [`CdcRetentionConfig::unlimited()`](crate::cdc::CdcRetentionConfig::unlimited)
    /// disables pruning, but beware of unbounded memory growth on
    /// long-running instances. Set it with [`Config::with_cdc_retention()`].
    #[cfg(feature = "cdc")]
    #[deprecated(
        since = "0.6.0",
        note = "read it with `Config::cdc_retention()` and set it with its builder; private in 0.7.0"
    )]
    pub cdc_retention: crate::cdc::CdcRetentionConfig,

    /// Per-section memory configuration.
    ///
    /// Maps `SectionType` to `SectionMemoryConfig` for sections that need
    /// custom budgets or tier pinning. Sections not listed here use the
    /// global `memory_limit` budget with automatic management.
    #[deprecated(
        since = "0.6.0",
        note = "read it with `Config::section_configs()` and set it with its builder; private in 0.7.0"
    )]
    pub section_configs: hashbrown::HashMap<
        grafeo_common::storage::SectionType,
        grafeo_common::storage::SectionMemoryConfig,
    >,

    /// Interval between automatic checkpoints.
    ///
    /// When set, the engine periodically flushes dirty sections to the
    /// `.grafeo` container and truncates the WAL. `None` means checkpoints
    /// only happen on explicit `wal_checkpoint()` or database close.
    #[deprecated(
        since = "0.6.0",
        note = "read it with `Config::checkpoint_interval()` and set it with its builder; private in 0.7.0"
    )]
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
    #[deprecated(
        since = "0.6.0",
        note = "read it with `Config::encryption()` and set it with its builder; private in 0.7.0"
    )]
    pub encryption: Option<EncryptionConfig>,
    /// Whether an open that finds damage in the WAL keeps the commits before
    /// it instead of failing (see [`Config::with_wal_salvage`]).
    wal_salvage: bool,
    /// Checkpoint once the WAL holds this many bytes since the last
    /// checkpoint (see [`Config::with_checkpoint_wal_size`]).
    checkpoint_wal_size: Option<u64>,
    /// The size a WAL segment grows to (see
    /// [`Config::with_wal_segment_size`]).
    wal_segment_size: u64,
    /// The WAL a checkpoint keeps beyond its log position for a backup at
    /// most, in bytes (see [`Config::with_wal_retention_limit`]).
    wal_retention_limit: u64,
    /// How long the database keeps history (see
    /// [`Config::with_history_retention`]).
    history_retention: HistoryRetention,
    /// The settings a builder set: they win over the values a database file
    /// stores (see `GrafeoDB::open`), where a default does not.
    set: SetFields,

    /// The memory limit as a share of the memory detected at open
    /// ([`with_memory_fraction`](Self::with_memory_fraction)): per open, never stored. Set only
    /// while `memory_limit` is `None`.
    memory_fraction: Option<f64>,

    /// `memory_fraction`'s bytes, taken of the memory detected when the database opened, so
    /// [`memory_limit`](Self::memory_limit) reads the figure the open's budget has.
    memory_fraction_bytes: Option<usize>,

    /// The cap of what the open spills ([`with_spill_limit`](Self::with_spill_limit),
    /// [`with_unlimited_spill`](Self::with_unlimited_spill)): per open, never stored.
    spill_limit: SpillLimit,

    /// The memory one query's operators may hold
    /// ([`with_query_memory_limit`](Self::with_query_memory_limit)): per open, never stored.
    query_memory_limit: Option<usize>,
}

/// The settings of a [`Config`] a database file can store, as a set: a
/// builder adds the ones it sets (see [`Config::was_set`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct SetFields(u16);

impl SetFields {
    pub(crate) const GRAPH_MODEL: Self = Self(1);
    pub(crate) const MEMORY_LIMIT: Self = Self(1 << 1);
    pub(crate) const SPILL_PATH: Self = Self(1 << 2);
    pub(crate) const WAL_ENABLED: Self = Self(1 << 3);
    pub(crate) const BACKWARD_EDGES: Self = Self(1 << 4);
    pub(crate) const WAL_DURABILITY: Self = Self(1 << 5);
    pub(crate) const SCHEMA_CONSTRAINTS: Self = Self(1 << 6);
    pub(crate) const SECTION_CONFIGS: Self = Self(1 << 7);
    pub(crate) const HISTORY_RETENTION: Self = Self(1 << 8);

    fn insert(&mut self, field: Self) {
        self.0 |= field.0;
    }

    fn remove(&mut self, field: Self) {
        self.0 &= !field.0;
    }

    #[cfg(feature = "grafeo-file")]
    fn contains(self, field: Self) -> bool {
        self.0 & field.0 == field.0
    }
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

#[allow(
    deprecated,
    reason = "the configuration sets and reads its own fields, deprecated for callers until 0.7.0 makes them private"
)]
impl Default for Config {
    fn default() -> Self {
        Self {
            graph_model: GraphModel::default(),
            path: None,
            memory_limit: None,
            memory_fraction: None,
            memory_fraction_bytes: None,
            spill_limit: SpillLimit::Default,
            query_memory_limit: None,
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
            wal_salvage: false,
            checkpoint_wal_size: Some(DEFAULT_CHECKPOINT_WAL_SIZE),
            wal_segment_size: DEFAULT_WAL_SEGMENT_SIZE,
            wal_retention_limit: DEFAULT_WAL_RETENTION_LIMIT,
            history_retention: HistoryRetention::Off,
            set: SetFields::default(),
        }
    }
}

#[allow(
    deprecated,
    reason = "the configuration sets and reads its own fields, deprecated for callers until 0.7.0 makes them private"
)]
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

    /// Sets the memory limit, in bytes: a hard limit on what the database's
    /// structures hold (the store, its indexes, RDF, change sets, query
    /// operators).
    ///
    /// Below it, the buffer manager evicts and spills (the indexed vector
    /// store's embedding columns, query operators with a spill directory);
    /// each path search may hold a
    /// quarter of it. A write that would pass it is refused before it
    /// changes anything, with `GRAFEO-S001` naming the bytes used and the
    /// limit; reads go on, and an open that reaches or exceeds the limit holds
    /// succeeds with a warning ([`GrafeoDB::open_warnings`](crate::GrafeoDB::open_warnings)).
    /// The limit bounds what the database's structures hold, and the
    /// process's private memory runs about a third above it (allocator
    /// bookkeeping, partial pages, thread stacks and caches it does not
    /// count), so leave that margin, for example a 2.5 GiB limit for a 4 GiB
    /// container (see [what the limit counts](https://grafeo.dev/architecture/memory/#what-the-limit-counts)).
    #[must_use]
    pub fn with_memory_limit(mut self, limit: usize) -> Self {
        self.memory_limit = Some(limit);
        self.memory_fraction = None;
        self.memory_fraction_bytes = None;
        self.set.insert(SetFields::MEMORY_LIMIT);
        self
    }

    /// The bytes one path search of a query may hold (the paths a
    /// variable-length or shortest-path search keeps at once): a quarter of
    /// [`memory_limit`](Self::memory_limit) when it is set, else 256 MiB. A
    /// search that would hold more fails with an error instead of growing.
    #[must_use]
    #[cfg(query_language)]
    pub(crate) fn path_search_budget(&self) -> usize {
        self.memory_limit().map_or(
            grafeo_core::execution::operators::DEFAULT_PATH_SEARCH_BUDGET,
            |limit| (limit / 4).max(1),
        )
    }

    /// Sets the number of worker threads, which has no effect: nothing sizes
    /// a pool from it. Code that denies deprecated items no longer
    /// builds with it:
    ///
    /// ```compile_fail
    /// #![deny(deprecated)]
    /// let _config = grafeo_engine::Config::in_memory().with_threads(3);
    /// ```
    #[must_use]
    #[deprecated(
        since = "0.6.1",
        note = "nothing sizes a pool from the thread count, so it has no effect; removed in 0.7.0"
    )]
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
        self.set.insert(SetFields::WAL_ENABLED);
        self
    }

    /// Turns the sidecar WAL on, the default for a persistent database: set
    /// it to override a database stored with the WAL off (see
    /// [`without_wal`](Self::without_wal)).
    #[must_use]
    pub fn with_wal(mut self) -> Self {
        self.wal_enabled = true;
        self.set.insert(SetFields::WAL_ENABLED);
        self
    }

    /// Salvages a damaged WAL at this open, instead of failing it.
    ///
    /// By default an open refuses a WAL with damage before its end (a record
    /// whose checksum or authentication fails, with whole records after it):
    /// replaying the commits after it on top of the missing ones would
    /// build a state that never existed, so the open fails with
    /// `GRAFEO-S002` naming the file and offset. A salvaging open replays
    /// the commits before the damage and leaves the rest out; nothing is
    /// deleted: what it did not replay moves to `<file>.wal.damaged` (or
    /// `<file>.wal.damaged-<n>`). A torn tail, the end of a write a crash
    /// cut off, is never damage: every open cuts it.
    ///
    /// For this open only: the database file does not store it.
    #[must_use]
    pub fn with_wal_salvage(mut self) -> Self {
        self.wal_salvage = true;
        self
    }

    /// Disables backward edges.
    #[must_use]
    pub fn without_backward_edges(mut self) -> Self {
        self.backward_edges = false;
        self.set.insert(SetFields::BACKWARD_EDGES);
        self
    }

    /// Enables query logging, which has no effect: no query is logged. Code that denies deprecated items no longer
    /// builds with it:
    ///
    /// ```compile_fail
    /// #![deny(deprecated)]
    /// let _config = grafeo_engine::Config::in_memory().with_query_logging();
    /// ```
    #[must_use]
    #[deprecated(
        since = "0.6.1",
        note = "no query is logged, so it has no effect; removed in 0.7.0"
    )]
    pub fn with_query_logging(mut self) -> Self {
        self.query_logging = true;
        self
    }

    /// Sets the memory limit to `fraction` of the memory this process may use, detected when
    /// the database opens ([`detect_memory`](grafeo_common::memory::detect_memory): the smaller
    /// of the machine's memory and a container's or job's limit).
    ///
    /// The share applies to this open only: a database file stores no limit for it, so a file
    /// moved to a smaller machine does not carry another machine's bytes. An explicit
    /// [`with_memory_limit`](Self::with_memory_limit) is stored, and the later of the two calls
    /// wins. A fraction that is infinite, not above zero, or so small its share comes to no
    /// byte fails [`validate`](Self::validate).
    #[must_use]
    pub fn with_memory_fraction(mut self, fraction: f64) -> Self {
        self.memory_limit = None;
        self.memory_fraction = Some(fraction);
        self.memory_fraction_bytes = None;
        // An explicit limit set before is replaced: nothing is left to store.
        self.set.remove(SetFields::MEMORY_LIMIT);
        self
    }

    /// Sets the spill directory for out-of-core processing.
    #[must_use]
    pub fn with_spill_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.spill_path = Some(path.into());
        self.set.insert(SetFields::SPILL_PATH);
        self
    }

    /// Caps what the open spills to disk at once at `bytes`, all its spills
    /// together (see [`SpillLimit`]); 0 spills nothing. Per open, never
    /// stored. Of this and [`with_unlimited_spill`](Self::with_unlimited_spill),
    /// the later call wins.
    #[must_use]
    pub fn with_spill_limit(mut self, bytes: u64) -> Self {
        self.spill_limit = SpillLimit::Bytes(bytes);
        self
    }

    /// Lifts the cap of what the open spills (see [`SpillLimit`]): the room
    /// the database keeps for its next checkpoint still bounds. Per open,
    /// never stored. Of this and [`with_spill_limit`](Self::with_spill_limit),
    /// the later call wins.
    #[must_use]
    pub fn with_unlimited_spill(mut self) -> Self {
        self.spill_limit = SpillLimit::Unlimited;
        self
    }

    /// Caps the memory one query's operators (sorts, aggregates) may hold at
    /// `bytes`; past it they spill, or fail with `GRAFEO-S001` without a
    /// spill directory. Per open, never stored; 0 fails
    /// [`validate`](Self::validate). Not enforced yet: operators take their
    /// grants against it in a later patch of this release.
    #[must_use]
    pub fn with_query_memory_limit(mut self, bytes: usize) -> Self {
        self.query_memory_limit = Some(bytes);
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
        self.set.insert(SetFields::GRAPH_MODEL);
        self
    }

    /// Sets the WAL durability mode.
    #[must_use]
    pub fn with_wal_durability(mut self, mode: DurabilityMode) -> Self {
        self.wal_durability = mode;
        self.set.insert(SetFields::WAL_DURABILITY);
        self
    }

    /// Sets the storage format for persistent databases.
    #[must_use]
    pub fn with_storage_format(mut self, format: StorageFormat) -> Self {
        self.storage_format = format;
        self
    }

    /// Enables catalog schema constraint enforcement, which has no effect:
    /// the catalog enforces the constraints a graph type or `CREATE
    /// CONSTRAINT` declares either way. Code that denies deprecated items no longer
    /// builds with it:
    ///
    /// ```compile_fail
    /// #![deny(deprecated)]
    /// let _config = grafeo_engine::Config::in_memory().with_schema_constraints();
    /// ```
    #[must_use]
    #[deprecated(
        since = "0.6.1",
        note = "the catalog enforces declared constraints either way, so it has no effect; removed in 0.7.0"
    )]
    pub fn with_schema_constraints(self) -> Self {
        self.with_stored_schema_constraints(true)
    }

    /// Turns catalog schema constraint enforcement off, the default, which
    /// has no effect (see
    /// [`with_schema_constraints`](Self::with_schema_constraints)).
    #[must_use]
    #[deprecated(
        since = "0.6.1",
        note = "the catalog enforces declared constraints either way, so it has no effect; removed in 0.7.0"
    )]
    pub fn without_schema_constraints(self) -> Self {
        self.with_stored_schema_constraints(false)
    }

    /// Sets the schema-constraints flag a database file stores, as its
    /// deprecated builders do: the file format keeps the flag until 0.7.0,
    /// so an open still carries a stored value over.
    #[must_use]
    pub(crate) fn with_stored_schema_constraints(mut self, on: bool) -> Self {
        self.schema_constraints = on;
        self.set.insert(SetFields::SCHEMA_CONSTRAINTS);
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
    /// assert_eq!(config.cdc_retention().max_events, Some(88_000));
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
        self.set.insert(SetFields::SECTION_CONFIGS);
        self
    }

    /// Pins a section to a specific storage tier (Phase 8d convenience).
    ///
    /// Shorthand for `with_section_config(section_type, SectionMemoryConfig {
    /// tier, max_ram: None })`. Pass [`TierOverride::ForceDisk`] to spill the
    /// section at database open, [`TierOverride::ForceRam`] to declare it
    /// must stay in RAM (declarative only until Phase 8g), or
    /// [`TierOverride::Auto`] (the default). Only `VectorStore` can be
    /// moved to disk (its embedding columns, in a build with `wal` and
    /// `vector-index`): [`validate`](Self::validate) refuses `ForceDisk` for
    /// any other section.
    ///
    /// # Examples
    ///
    /// ```
    /// # use grafeo_engine::Config;
    /// use grafeo_common::storage::{SectionType, TierOverride};
    ///
    /// // Spill the embedding columns of the vector indexes at open.
    /// let config = Config::persistent("lyon.grafeo")
    ///     .with_section_tier(SectionType::VectorStore, TierOverride::ForceDisk);
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

    /// Sets the automatic checkpoint interval: the database checkpoints by
    /// itself when this long has passed since its last automatic checkpoint.
    /// Typical values: 30-300 seconds. Off by default; see
    /// [`CheckpointTriggers`](crate::admin::CheckpointTriggers), and
    /// `GrafeoDB::set_checkpoint_triggers` to change it on an open database.
    #[must_use]
    pub fn with_checkpoint_interval(mut self, interval: Duration) -> Self {
        self.checkpoint_interval = Some(interval);
        self
    }

    /// Checkpoints once the WAL holds `bytes` since the last checkpoint,
    /// which bounds the WAL on disk and the log a reopen replays: 256 MiB by
    /// default ([`DEFAULT_CHECKPOINT_WAL_SIZE`]), at least 1 MiB
    /// ([`validate`](Self::validate) refuses less). The checkpoint runs on a
    /// background thread, which a commit that takes the WAL past the size
    /// wakes; see [`CheckpointTriggers`](crate::admin::CheckpointTriggers).
    /// Without a WAL it never fires.
    #[must_use]
    pub fn with_checkpoint_wal_size(mut self, bytes: u64) -> Self {
        self.checkpoint_wal_size = Some(bytes);
        self
    }

    /// Turns automatic checkpoints by WAL size off: the WAL then grows until
    /// the database closes or [`wal_checkpoint`](crate::GrafeoDB::wal_checkpoint)
    /// runs, unless [`with_checkpoint_interval`](Self::with_checkpoint_interval)
    /// sets an interval.
    #[must_use]
    pub fn without_checkpoint_wal_size(mut self) -> Self {
        self.checkpoint_wal_size = None;
        self
    }

    /// Sets the size, in bytes, a WAL segment grows to before the log goes on
    /// in a new one: 64 MiB by default ([`DEFAULT_WAL_SEGMENT_SIZE`]), from
    /// 1 MiB to 1 GiB ([`validate`](Self::validate) refuses other sizes).
    /// A checkpoint removes whole segments and keeps one segment of free
    /// disk space for the WAL to grow in, so smaller segments free the disk
    /// sooner and need less room. A WAL written at one size reopens at
    /// another: the segments it holds keep their size, and new ones get the
    /// new one.
    #[must_use]
    pub fn with_wal_segment_size(mut self, bytes: u64) -> Self {
        self.wal_segment_size = bytes;
        self
    }

    /// Sets how much WAL, in bytes, a checkpoint keeps at most beyond the
    /// log it covers for incremental backups: 1 GiB by default
    /// ([`DEFAULT_WAL_RETENTION_LIMIT`]).
    ///
    /// A checkpoint keeps the WAL from where the last backup ended, so the
    /// next incremental backup can copy it. When backups stop, that WAL
    /// would grow until the disk is full: past this limit a checkpoint
    /// removes the oldest segments, so the WAL kept stays within it, and
    /// warns once. The next incremental backup then fails with
    /// `GRAFEO-S005`
    /// ([`ErrorCode::BackupChainBroken`](grafeo_common::utils::error::ErrorCode::BackupChainBroken)),
    /// and a full backup starts a new chain. Raise it when more than this
    /// much is written between two incremental backups.
    #[must_use]
    pub fn with_wal_retention_limit(mut self, bytes: u64) -> Self {
        self.wal_retention_limit = bytes;
        self
    }

    /// Sets how long the database keeps history (see [`HistoryRetention`]):
    /// a stored setting, which a database file keeps.
    #[must_use]
    pub fn with_history_retention(mut self, retention: HistoryRetention) -> Self {
        self.history_retention = retention;
        self.set.insert(SetFields::HISTORY_RETENTION);
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

    /// The graph data model (LPG or RDF), fixed when the database is created.
    #[must_use]
    pub fn graph_model(&self) -> GraphModel {
        self.graph_model
    }

    /// The path of the database file, `None` for an in-memory database.
    #[must_use]
    pub fn path(&self) -> Option<&std::path::Path> {
        self.path.as_deref()
    }

    /// The memory limit in bytes: the one set, or
    /// [`with_memory_fraction`](Self::with_memory_fraction)'s share of the memory detected (when
    /// the database opened, for a database's own configuration; now, before an open); `None`
    /// for the default, [`DEFAULT_MEMORY_FRACTION`](grafeo_common::memory::DEFAULT_MEMORY_FRACTION)
    /// of the detected memory.
    #[must_use]
    pub fn memory_limit(&self) -> Option<usize> {
        self.memory_limit
            .or(self.memory_fraction_bytes)
            .or_else(|| {
                self.memory_fraction.map(|fraction| {
                    grafeo_common::memory::system::fraction_of(
                        grafeo_common::memory::detect_memory().bytes,
                        fraction,
                    )
                })
            })
    }

    /// The memory limit set with [`with_memory_limit`](Self::with_memory_limit), the one a
    /// database file stores; `None` for the default or a fraction.
    #[must_use]
    pub(crate) fn explicit_memory_limit(&self) -> Option<usize> {
        self.memory_limit
    }

    /// The share of the detected memory set with
    /// [`with_memory_fraction`](Self::with_memory_fraction), when no explicit limit replaced it.
    #[must_use]
    pub(crate) fn memory_fraction(&self) -> Option<f64> {
        self.memory_fraction
    }

    /// The configuration with the bytes the open took for its fraction, read back by
    /// [`memory_limit`](Self::memory_limit); unchanged without a fraction.
    #[must_use]
    pub(crate) fn with_open_memory_budget(mut self, bytes: usize) -> Self {
        if self.memory_fraction.is_some() {
            self.memory_fraction_bytes = Some(bytes);
        }
        self
    }

    /// The spill directory for out-of-core processing, when one is set.
    #[must_use]
    pub fn spill_path(&self) -> Option<&std::path::Path> {
        self.spill_path.as_deref()
    }

    /// The cap of what the open spills (see [`SpillLimit`]).
    #[must_use]
    pub fn spill_limit(&self) -> SpillLimit {
        self.spill_limit
    }

    /// The memory one query's operators may hold; `None` for the memory
    /// limit.
    #[must_use]
    pub fn query_memory_limit(&self) -> Option<usize> {
        self.query_memory_limit
    }

    /// The number of worker threads, which has no effect.
    #[must_use]
    #[deprecated(
        since = "0.6.1",
        note = "nothing sizes a pool from the thread count, so it has no effect; removed in 0.7.0"
    )]
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Whether commits are logged to the sidecar WAL.
    #[must_use]
    pub fn wal_enabled(&self) -> bool {
        self.wal_enabled
    }

    /// Whether the store keeps backward edges (incoming adjacency).
    #[must_use]
    pub fn backward_edges(&self) -> bool {
        self.backward_edges
    }

    /// Whether queries are logged, which has no effect.
    #[must_use]
    #[deprecated(
        since = "0.6.1",
        note = "no query is logged, so it has no effect; removed in 0.7.0"
    )]
    pub fn query_logging(&self) -> bool {
        self.query_logging
    }

    /// Whether multi-hop queries run factorized.
    #[must_use]
    pub fn factorized_execution(&self) -> bool {
        self.factorized_execution
    }

    /// Whether queries without `ORDER BY` return their rows in random order
    /// (a test option).
    #[must_use]
    pub fn shuffle_unordered(&self) -> bool {
        self.shuffle_unordered
    }

    /// When the WAL is synced to disk.
    #[must_use]
    pub fn wal_durability(&self) -> DurabilityMode {
        self.wal_durability
    }

    /// Whether an open salvages a damaged WAL (see
    /// [`with_wal_salvage`](Self::with_wal_salvage)).
    #[must_use]
    pub fn wal_salvage(&self) -> bool {
        self.wal_salvage
    }

    /// The storage format for a persistent database.
    #[must_use]
    pub fn storage_format(&self) -> StorageFormat {
        self.storage_format
    }

    /// Whether the catalog enforces schema constraints, which has no effect.
    #[must_use]
    #[deprecated(
        since = "0.6.1",
        note = "the catalog enforces declared constraints either way, so it has no effect; removed in 0.7.0"
    )]
    pub fn schema_constraints(&self) -> bool {
        self.schema_constraints
    }

    /// The schema-constraints flag a database file stores (see
    /// [`with_stored_schema_constraints`](Self::with_stored_schema_constraints)).
    #[cfg(feature = "grafeo-file")]
    #[must_use]
    pub(crate) fn stored_schema_constraints(&self) -> bool {
        self.schema_constraints
    }

    /// The longest a query may run, `None` for no limit.
    #[must_use]
    pub fn query_timeout(&self) -> Option<Duration> {
        self.query_timeout
    }

    /// The deadline of a statement that starts now under `timeout`, `None` for no limit: the one
    /// reckoning of a query's deadline, a session's and the bindings' algorithm calls alike. A
    /// timeout too long to reckon from now (`Duration::MAX`) is no limit too.
    #[must_use]
    pub fn deadline_after(timeout: Option<Duration>) -> Option<grafeo_common::clock::Instant> {
        timeout.and_then(|limit| grafeo_common::clock::Instant::now().checked_add(limit))
    }

    /// The largest property value in bytes, `None` for no limit.
    #[must_use]
    pub fn max_property_size(&self) -> Option<usize> {
        self.max_property_size
    }

    /// Every how many commits old versions are collected (0: never).
    #[must_use]
    pub fn gc_interval(&self) -> usize {
        self.gc_interval
    }

    /// Read-write or read-only.
    #[must_use]
    pub fn access_mode(&self) -> AccessMode {
        self.access_mode
    }

    /// Whether new sessions record change data capture events.
    #[must_use]
    pub fn cdc_enabled(&self) -> bool {
        self.cdc_enabled
    }

    /// How many change data capture events are kept.
    #[cfg(feature = "cdc")]
    #[must_use]
    pub fn cdc_retention(&self) -> &crate::cdc::CdcRetentionConfig {
        &self.cdc_retention
    }

    /// The memory configuration of each section that has one.
    #[must_use]
    pub fn section_configs(
        &self,
    ) -> &hashbrown::HashMap<
        grafeo_common::storage::SectionType,
        grafeo_common::storage::SectionMemoryConfig,
    > {
        &self.section_configs
    }

    /// How often the database checkpoints on its own, when it does.
    #[must_use]
    pub fn checkpoint_interval(&self) -> Option<Duration> {
        self.checkpoint_interval
    }

    /// The WAL size that starts an automatic checkpoint (see
    /// [`with_checkpoint_wal_size`](Self::with_checkpoint_wal_size)).
    #[must_use]
    pub fn checkpoint_wal_size(&self) -> Option<u64> {
        self.checkpoint_wal_size
    }

    /// The size a WAL segment grows to (see
    /// [`with_wal_segment_size`](Self::with_wal_segment_size)).
    #[must_use]
    pub fn wal_segment_size(&self) -> u64 {
        self.wal_segment_size
    }

    /// How much WAL a checkpoint keeps at most beyond the log it covers for
    /// incremental backups (see
    /// [`with_wal_retention_limit`](Self::with_wal_retention_limit)).
    #[must_use]
    pub fn wal_retention_limit(&self) -> u64 {
        self.wal_retention_limit
    }

    /// How long the database keeps history (see
    /// [`with_history_retention`](Self::with_history_retention)).
    #[must_use]
    pub fn history_retention(&self) -> HistoryRetention {
        self.history_retention
    }

    /// The automatic checkpoint triggers this configuration sets.
    #[must_use]
    pub fn checkpoint_triggers(&self) -> crate::admin::CheckpointTriggers {
        let mut triggers = crate::admin::CheckpointTriggers::none();
        triggers.interval = self.checkpoint_interval;
        triggers.wal_bytes = self.checkpoint_wal_size;
        triggers
    }

    /// The encryption at rest, when the database is encrypted.
    #[cfg(feature = "encryption")]
    #[must_use]
    pub fn encryption(&self) -> Option<&EncryptionConfig> {
        self.encryption.as_ref()
    }

    /// Whether a builder set `field` (see [`SetFields`]): what a database
    /// file stores yields to it.
    #[cfg(feature = "grafeo-file")]
    #[must_use]
    pub(crate) fn was_set(&self, field: SetFields) -> bool {
        self.set.contains(field)
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
        if self.query_memory_limit == Some(0) {
            return Err(ConfigError::ZeroQueryMemoryLimit);
        }
        if let Some(fraction) = self.memory_fraction {
            if fraction.is_infinite() {
                return Err(ConfigError::MemoryFractionNotFinite);
            }
            // NaN and fractions not above zero, and those whose share of the
            // detected memory comes to no byte, as a zero limit.
            if fraction.is_nan()
                || fraction <= 0.0
                || self.memory_fraction_bytes.unwrap_or_else(|| {
                    grafeo_common::memory::system::fraction_of(
                        grafeo_common::memory::detect_memory().bytes,
                        fraction,
                    )
                }) == 0
            {
                return Err(ConfigError::ZeroMemoryLimit);
            }
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

        self.check_section_tiers()?;

        if !(MIN_WAL_SEGMENT_SIZE..=MAX_WAL_SEGMENT_SIZE).contains(&self.wal_segment_size) {
            return Err(ConfigError::WalSegmentSizeOutOfBounds(
                self.wal_segment_size,
            ));
        }
        check_checkpoint_wal_size(self.checkpoint_wal_size)?;

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

    /// Refuses a disk tier override on a section that cannot be moved to
    /// disk: the open would keep it in memory without a word. Also checked
    /// for a change of the stored configuration, which a later open applies.
    ///
    /// # Errors
    ///
    /// [`ConfigError::ForceDiskOnUnspillableSection`] for the first such
    /// section in section type order, so the error does not depend on the
    /// map's iteration order.
    pub(crate) fn check_section_tiers(&self) -> std::result::Result<(), ConfigError> {
        let unspillable = self
            .section_configs
            .iter()
            .filter(|(section_type, section)| {
                section.tier == grafeo_common::storage::TierOverride::ForceDisk
                    && !can_spill(**section_type)
            })
            .map(|(section_type, _)| *section_type)
            .min_by_key(|section_type| section_type.to_u8());
        match unspillable {
            Some(section_type) => Err(ConfigError::ForceDiskOnUnspillableSection(section_type)),
            None => Ok(()),
        }
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
        assert_eq!(config.graph_model(), GraphModel::Lpg);
        assert!(config.path().is_none());
        assert!(config.memory_limit().is_none());
        assert!(config.spill_path().is_none());
        assert!(config.wal_enabled());
        assert!(config.backward_edges());
        assert!(config.factorized_execution());
        assert_eq!(config.wal_durability(), DurabilityMode::default());
        assert_eq!(config.query_timeout(), Some(Duration::from_secs(30)));
        assert_eq!(config.gc_interval(), 100);
    }

    #[test]
    fn test_config_in_memory() {
        let config = Config::in_memory();
        assert!(config.path().is_none());
        assert!(!config.wal_enabled());
        assert!(config.backward_edges());
    }

    #[test]
    fn test_config_persistent() {
        let config = Config::persistent("/tmp/test_db");
        assert_eq!(config.path(), Some(std::path::Path::new("/tmp/test_db")));
        assert!(config.wal_enabled());
    }

    #[test]
    fn test_config_with_memory_limit() {
        let config = Config::in_memory().with_memory_limit(1024 * 1024);
        assert_eq!(config.memory_limit(), Some(1024 * 1024));
    }

    /// `threads`, `query_logging` and `schema_constraints` have no effect,
    /// but grafeo-server forwards them, so until 0.7.0 removes them their
    /// builders still set what their getters read, and validation still
    /// refuses zero threads.
    #[test]
    #[expect(
        deprecated,
        reason = "the deprecated no-op settings still read back what their builders set, until 0.7.0 removes them"
    )]
    fn deprecated_no_op_settings_read_back() {
        let config = Config::in_memory();
        assert_eq!(config.threads(), num_cpus::get());
        assert!(!config.query_logging());
        assert!(!config.schema_constraints());

        let config = config
            .with_threads(8)
            .with_query_logging()
            .with_schema_constraints();
        assert_eq!(config.threads(), 8);
        assert!(config.query_logging());
        assert!(config.schema_constraints());
        assert!(!config.without_schema_constraints().schema_constraints());

        assert_eq!(
            Config::in_memory().with_threads(0).validate(),
            Err(ConfigError::ZeroThreads)
        );
    }

    #[test]
    fn test_config_without_backward_edges() {
        let config = Config::in_memory().without_backward_edges();
        assert!(!config.backward_edges());
    }

    #[test]
    fn test_config_with_spill_path() {
        let config = Config::in_memory().with_spill_path("/tmp/spill");
        assert_eq!(
            config.spill_path(),
            Some(std::path::Path::new("/tmp/spill"))
        );
    }

    #[test]
    fn test_config_with_memory_fraction() {
        let config = Config::in_memory().with_memory_fraction(0.5);
        assert!(config.memory_limit().is_some());
        assert!(config.memory_limit().unwrap() > 0);
    }

    #[test]
    fn test_config_without_factorized_execution() {
        let config = Config::in_memory().without_factorized_execution();
        assert!(!config.factorized_execution());
    }

    #[test]
    fn test_config_builder_chaining() {
        let config = Config::persistent("/tmp/db")
            .with_memory_limit(512 * 1024 * 1024)
            .without_backward_edges()
            .with_spill_path("/tmp/spill");

        assert!(config.path().is_some());
        assert_eq!(config.memory_limit(), Some(512 * 1024 * 1024));
        assert!(!config.backward_edges());
        assert!(config.spill_path().is_some());
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
        assert_eq!(config.graph_model(), GraphModel::Rdf);
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
        assert_eq!(config.wal_durability(), DurabilityMode::Sync);
    }

    #[test]
    fn test_config_with_wal_durability_nosync() {
        let config = Config::persistent("/tmp/db").with_wal_durability(DurabilityMode::NoSync);
        assert_eq!(config.wal_durability(), DurabilityMode::NoSync);
    }

    #[test]
    fn test_config_with_wal_durability_adaptive() {
        let config = Config::persistent("/tmp/db").with_wal_durability(DurabilityMode::Adaptive {
            target_interval_ms: 50,
        });
        assert_eq!(
            config.wal_durability(),
            DurabilityMode::Adaptive {
                target_interval_ms: 50
            }
        );
    }

    // --- max_property_size tests ---

    #[test]
    fn test_config_default_max_property_size() {
        let config = Config::in_memory();
        assert_eq!(config.max_property_size(), Some(16 * 1024 * 1024));
    }

    #[test]
    fn test_config_with_max_property_size() {
        let config = Config::in_memory().with_max_property_size(1024);
        assert_eq!(config.max_property_size(), Some(1024));
    }

    #[test]
    fn test_config_without_max_property_size() {
        let config = Config::in_memory().without_max_property_size();
        assert!(config.max_property_size().is_none());
    }

    // --- query_timeout tests ---

    #[test]
    fn test_config_with_query_timeout() {
        let config = Config::in_memory().with_query_timeout(Duration::from_mins(1));
        assert_eq!(config.query_timeout(), Some(Duration::from_mins(1)));
    }

    #[test]
    fn test_config_without_query_timeout() {
        let config = Config::in_memory().without_query_timeout();
        assert!(config.query_timeout().is_none());
    }

    #[test]
    fn test_config_default_query_timeout() {
        let config = Config::in_memory();
        assert_eq!(config.query_timeout(), Some(Duration::from_secs(30)));
    }

    // --- gc_interval tests ---

    #[test]
    fn test_config_with_gc_interval() {
        let config = Config::in_memory().with_gc_interval(50);
        assert_eq!(config.gc_interval(), 50);
    }

    #[test]
    fn test_config_gc_disabled() {
        let config = Config::in_memory().with_gc_interval(0);
        assert_eq!(config.gc_interval(), 0);
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
        let in_memory = Config::in_memory().with_encryption(encryption.clone());
        assert_eq!(
            in_memory.validate(),
            Err(ConfigError::EncryptionRequiresPersistentPath)
        );
        assert!(
            ConfigError::EncryptionRequiresPersistentPath
                .to_string()
                .contains("requires a persistent database path")
        );
        for config in [
            Config::persistent("amsterdam.grafeo"),
            Config::read_only("amsterdam.grafeo"),
        ] {
            let config = config.with_encryption(encryption.clone());
            assert_eq!(config.validate(), Ok(()), "{:?}", config.path());
        }
    }

    /// Spill files are not encrypted: a spill path with encryption is invalid.
    #[cfg(all(feature = "encryption", not(miri)))]
    #[test]
    fn encryption_with_a_spill_path_is_invalid() {
        let config = Config::persistent("berlin.grafeo").with_spill_path("berlin.spill");
        assert_eq!(config.validate(), Ok(()), "a spill path alone is fine");
        let config = config.with_encryption(EncryptionConfig::new(std::sync::Arc::new(
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
    #[cfg(all(
        feature = "encryption",
        feature = "wal",
        feature = "vector-index",
        not(miri)
    ))]
    #[test]
    #[allow(
        deprecated,
        reason = "the test unsets the key, which no builder does, to check the other half of the rule"
    )]
    fn encryption_with_a_section_forced_to_disk_is_invalid() {
        use grafeo_common::storage::{SectionType, TierOverride};

        let encryption = EncryptionConfig::new(std::sync::Arc::new(
            grafeo_common::encryption::KeyChain::new([88; 32]),
        ));
        let config = Config::persistent("prague.grafeo")
            .with_section_tier(SectionType::TextIndex, TierOverride::ForceRam)
            .with_section_tier(SectionType::RdfRing, TierOverride::Auto)
            .with_encryption(encryption);
        assert_eq!(
            config.validate(),
            Ok(()),
            "Auto and ForceRam keep data in RAM"
        );

        let mut config =
            config.with_section_tier(SectionType::VectorStore, TierOverride::ForceDisk);
        let error = config
            .validate()
            .expect_err("a section forced to disk on an encrypted database")
            .to_string();
        assert!(
            error.contains("disk tier override") && error.contains("VectorStore"),
            "the error names the override and the section: {error}"
        );
        assert_eq!(
            config.validate(),
            Err(ConfigError::EncryptionWithForceDisk(
                SectionType::VectorStore
            ))
        );
        config.encryption = None;
        assert_eq!(config.validate(), Ok(()), "ForceDisk alone is fine");
    }

    /// Only some sections can move to disk; a disk override on any other
    /// section would have no effect, so it is refused with the section and
    /// the ones that can spill.
    #[test]
    fn force_disk_on_a_section_that_cannot_spill_is_invalid() {
        use grafeo_common::storage::{SectionType, TierOverride};

        for section_type in [
            SectionType::Catalog,
            SectionType::LpgStore,
            SectionType::RdfStore,
            SectionType::OverlayDeletions,
            SectionType::TimeIndex,
            // Mappable, but nothing moves them to disk: a disk override would do nothing. The
            // compact store is only read from a 0.5 file and folded into row groups; the ring's
            // swap copies its mapped bytes back onto the heap.
            SectionType::TextIndex,
            SectionType::PropertyIndex,
            SectionType::CompactStore,
            SectionType::RdfRing,
        ] {
            let config = Config::persistent("berlin.grafeo")
                .with_section_tier(section_type, TierOverride::ForceDisk);
            assert_eq!(
                config.validate(),
                Err(ConfigError::ForceDiskOnUnspillableSection(section_type)),
                "{section_type:?} cannot spill"
            );
            let spillable = if cfg!(all(feature = "wal", feature = "vector-index")) {
                "VectorStore"
            } else {
                "none in this build"
            };
            assert_eq!(
                ConfigError::ForceDiskOnUnspillableSection(section_type).to_string(),
                format!(
                    "the {section_type:?} section cannot be moved to disk, so it takes no disk \
                     tier override; the sections that can are {spillable}"
                )
            );
            for tier in [TierOverride::Auto, TierOverride::ForceRam] {
                let config =
                    Config::persistent("berlin.grafeo").with_section_tier(section_type, tier);
                assert_eq!(config.validate(), Ok(()), "{section_type:?} with {tier:?}");
            }
        }

        let config = Config::persistent("berlin.grafeo")
            .with_section_tier(SectionType::VectorStore, TierOverride::ForceDisk);
        if cfg!(all(feature = "wal", feature = "vector-index")) {
            assert_eq!(config.validate(), Ok(()), "the vector store spills");
        } else {
            // Nothing writes the embedding columns to disk in this build.
            assert_eq!(
                config.validate(),
                Err(ConfigError::ForceDiskOnUnspillableSection(
                    SectionType::VectorStore
                ))
            );
        }
    }

    // --- Builder chaining with new fields ---

    #[test]
    fn test_config_full_builder_chaining() {
        let config = Config::persistent("/tmp/db")
            .with_graph_model(GraphModel::Lpg)
            .with_memory_limit(512 * 1024 * 1024)
            .with_wal_durability(DurabilityMode::Sync)
            .without_backward_edges()
            .with_spill_path("/tmp/spill")
            .with_query_timeout(Duration::from_mins(1));

        assert_eq!(config.graph_model(), GraphModel::Lpg);
        assert!(config.path().is_some());
        assert_eq!(config.memory_limit(), Some(512 * 1024 * 1024));
        assert_eq!(config.wal_durability(), DurabilityMode::Sync);
        assert!(!config.backward_edges());
        assert!(config.spill_path().is_some());
        assert_eq!(config.query_timeout(), Some(Duration::from_mins(1)));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn the_wal_sizes_have_their_defaults_and_bounds() {
        let config = Config::persistent("lyon.grafeo");
        assert_eq!(
            config.checkpoint_wal_size(),
            Some(DEFAULT_CHECKPOINT_WAL_SIZE)
        );
        assert_eq!(config.wal_segment_size(), DEFAULT_WAL_SEGMENT_SIZE);
        assert_eq!(
            Config::in_memory().checkpoint_wal_size(),
            Some(DEFAULT_CHECKPOINT_WAL_SIZE),
            "a setting; only a database with a WAL acts on it"
        );
        assert_eq!(
            config
                .clone()
                .without_checkpoint_wal_size()
                .checkpoint_wal_size(),
            None
        );
        for (bytes, accepted) in [
            (MIN_WAL_SEGMENT_SIZE - 1, false),
            (MIN_WAL_SEGMENT_SIZE, true),
            (MAX_WAL_SEGMENT_SIZE, true),
            (MAX_WAL_SEGMENT_SIZE + 1, false),
        ] {
            assert_eq!(
                config
                    .clone()
                    .with_wal_segment_size(bytes)
                    .validate()
                    .is_ok(),
                accepted,
                "{bytes}"
            );
        }
        assert_eq!(
            config
                .clone()
                .with_checkpoint_wal_size(MIN_CHECKPOINT_WAL_SIZE - 1)
                .validate(),
            Err(ConfigError::CheckpointWalSizeTooSmall(
                MIN_CHECKPOINT_WAL_SIZE - 1
            ))
        );
        assert!(
            config
                .with_checkpoint_wal_size(MIN_CHECKPOINT_WAL_SIZE)
                .validate()
                .is_ok()
        );
    }

    /// The setter on an open database applies the same check, so its error
    /// is the one an open gives; zero says why it is refused.
    #[test]
    fn the_checkpoint_size_floor_is_one_check() {
        assert_eq!(check_checkpoint_wal_size(None), Ok(()));
        assert_eq!(
            check_checkpoint_wal_size(Some(MIN_CHECKPOINT_WAL_SIZE)),
            Ok(())
        );
        let zero = check_checkpoint_wal_size(Some(0)).unwrap_err().to_string();
        assert!(zero.contains("zero") && zero.contains("1 MiB"), "{zero}");
        let small = check_checkpoint_wal_size(Some(19)).unwrap_err().to_string();
        assert!(
            small.contains("19 bytes") && small.contains("1 MiB"),
            "{small}"
        );
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
            .without_wal()
            .without_backward_edges()
            .without_factorized_execution()
            .with_shuffle_unordered(true)
            .with_wal_durability(DurabilityMode::batch(Duration::from_millis(19), 88))
            .with_storage_format(StorageFormat::Auto)
            .with_query_timeout(Duration::from_secs(19))
            .with_max_property_size(88)
            .with_gc_interval(3)
            .with_access_mode(AccessMode::ReadOnly)
            .with_cdc()
            .with_section_tier(SectionType::VectorStore, TierOverride::ForceRam)
            .with_checkpoint_interval(Duration::from_secs(88))
            .with_checkpoint_wal_size(19 * 1024 * 1024)
            .with_history_retention(HistoryRetention::Duration(Duration::from_secs(88)))
            .with_wal_retention_limit(19 * 1024 * 1024)
            .with_wal_segment_size(3 * 1024 * 1024)
            .with_wal_salvage();
        #[expect(
            deprecated,
            reason = "the deprecated no-op settings still have their builders until 0.7.0 removes them"
        )]
        let config = config
            .with_threads(3)
            .with_query_logging()
            .with_schema_constraints();
        #[cfg(feature = "cdc")]
        let config = config.with_cdc_retention(
            crate::cdc::CdcRetentionConfig::unlimited()
                .with_max_epochs(19)
                .with_max_events(88),
        );
        let config = config
            .with_spill_limit(88 << 20)
            .with_query_memory_limit(19 << 20);
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
            wal_salvage,
            checkpoint_wal_size,
            history_retention,
            wal_retention_limit,
            wal_segment_size,
            // What the builders set: see the stored-configuration tests.
            set: _,
            memory_fraction,
            memory_fraction_bytes,
            spill_limit,
            query_memory_limit,
        } = config;

        assert_eq!(
            spill_limit,
            SpillLimit::Bytes(88 << 20),
            "with_spill_limit sets the cap"
        );
        assert_eq!(
            query_memory_limit,
            Some(19 << 20),
            "with_query_memory_limit sets the limit"
        );

        assert_eq!(
            wal_segment_size,
            3 * 1024 * 1024,
            "with_wal_segment_size sets the size"
        );
        assert_eq!(checkpoint_wal_size, Some(19 * 1024 * 1024));
        assert_eq!(
            wal_retention_limit,
            19 * 1024 * 1024,
            "with_wal_retention_limit sets the limit"
        );
        assert_eq!(
            history_retention,
            HistoryRetention::Duration(Duration::from_secs(88)),
            "with_history_retention sets the retention"
        );

        assert!(wal_salvage, "with_wal_salvage salvages at this open");

        assert_eq!(graph_model, GraphModel::Rdf);
        assert_eq!(
            path.as_deref(),
            Some(std::path::Path::new("amsterdam.grafeo"))
        );
        assert_eq!(memory_limit, Some(88 * 1024 * 1024));
        // `with_memory_fraction` is the other way to set the limit; the explicit one is set here.
        assert_eq!((memory_fraction, memory_fraction_bytes), (None, None));
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
        assert_eq!(config.access_mode(), AccessMode::ReadOnly);
    }

    #[test]
    fn test_config_read_only() {
        let config = Config::read_only("/tmp/db.grafeo");
        assert_eq!(config.access_mode(), AccessMode::ReadOnly);
        assert!(config.path().is_some());
        assert!(!config.wal_enabled());
    }

    #[test]
    fn test_config_default_is_read_write() {
        let config = Config::default();
        assert_eq!(config.access_mode(), AccessMode::ReadWrite);
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
        assert!(config.cdc_enabled());
    }

    #[test]
    fn test_config_cdc_default_false() {
        let config = Config::default();
        assert!(!config.cdc_enabled());
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
        assert!(config.memory_limit().is_none());
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
        assert_eq!(config.access_mode(), AccessMode::ReadOnly);
        assert!(!config.wal_enabled());
        assert!(config.path().is_some());
        // Other defaults should still apply
        assert!(config.backward_edges());
        assert_eq!(config.graph_model(), GraphModel::Lpg);
    }
}
