//! The main database struct and operations.
//!
//! Start here with [`GrafeoDB`] - it's your handle to everything.
//!
//! Operations are split across focused submodules:
//! - `query` - Query execution (execute, execute_cypher, etc.)
//! - `crud` - Node/edge CRUD operations
//! - `index` - Property, vector, and text index management
//! - `search` - Vector, text, and hybrid search
//! - `embed` - Embedding model management
//! - `persistence` - Save, load, snapshots, iteration
//! - `admin` - Stats, introspection, diagnostics, CDC

#[cfg(feature = "lpg")]
mod admin;
#[cfg(feature = "arrow-export")]
pub mod arrow;
#[cfg(all(feature = "async-storage", feature = "lpg"))]
mod async_ops;
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
pub mod backup;
#[cfg(feature = "lpg")]
pub(crate) mod catalog_records;
#[cfg(feature = "lpg")]
pub(crate) mod catalog_section;
#[cfg(all(feature = "grafeo-file", feature = "lpg"))]
mod checkpoint_timer;
#[cfg(feature = "lpg")]
mod crud;
#[cfg(feature = "lpg")]
pub(crate) mod direct;
#[cfg(feature = "embed")]
mod embed;
#[cfg(feature = "grafeo-file")]
pub(crate) mod encryption;
#[cfg(feature = "grafeo-file")]
pub(crate) mod flush;
#[cfg(feature = "lpg")]
mod graph_handle;
#[cfg(feature = "lpg")]
pub use direct::BatchEdge;
#[cfg(feature = "lpg")]
pub use graph_handle::GraphHandle;
#[cfg(all(feature = "lpg", feature = "gql"))]
pub use upsert::{EdgeUpsertOptions, UpsertSummary};
#[cfg(feature = "lpg")]
mod import;
#[cfg(feature = "lpg")]
pub(crate) mod index;
#[cfg(all(feature = "lpg", feature = "grafeo-file", feature = "vector-index"))]
mod legacy_spill;
#[cfg(feature = "grafeo-file")]
mod migration;
#[cfg(feature = "lpg")]
mod persistence;
#[cfg(all(test, feature = "lpg", feature = "gql"))]
mod processor_claims_tests;
mod query;
#[cfg(feature = "triple-store")]
mod rdf_ops;
#[cfg(all(feature = "wal", feature = "lpg"))]
mod schema_replay;
#[cfg(feature = "lpg")]
mod search;
pub(crate) mod section_consumer;
mod sections;
mod spill_directory;
#[cfg(feature = "lpg")]
pub(crate) mod standalone;
#[cfg(all(test, feature = "lpg", not(feature = "temporal")))]
pub(crate) mod test_backing;
#[cfg(all(feature = "lpg", feature = "gql"))]
mod upsert;
#[cfg(all(
    feature = "lpg",
    feature = "vector-index",
    feature = "mmap",
    not(feature = "temporal")
))]
mod vector_spill;

use grafeo_common::grafeo_error;
#[cfg(feature = "wal")]
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use parking_lot::RwLock;

use grafeo_common::memory::buffer::{BufferManager, BufferManagerConfig};
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};
#[cfg(feature = "lpg")]
use grafeo_core::graph::lpg::LpgStore;
#[cfg(feature = "triple-store")]
use grafeo_core::graph::rdf::RdfStore;
use grafeo_core::graph::{GraphStoreMut, GraphStoreSearch};
#[cfg(feature = "grafeo-file")]
use grafeo_storage::file::GrafeoFileManager;
#[cfg(feature = "grafeo-file")]
use grafeo_storage::file::detect::{OnDisk, detect};
#[cfg(feature = "grafeo-file")]
use grafeo_storage::file::v3::header::new_database_id;
#[cfg(feature = "wal")]
use grafeo_storage::wal::{DurabilityMode as WalDurabilityMode, LpgWal, WalConfig};
#[cfg(all(feature = "wal", feature = "lpg"))]
use grafeo_storage::wal::{WalRecord, WalRecovery};

use crate::catalog::Catalog;
use crate::config::Config;
use crate::query::cache::QueryCache;
use crate::session::Session;
use crate::transaction::TransactionManager;

/// Your handle to a Grafeo database.
///
/// Start here. Create one with [`new_in_memory()`](Self::new_in_memory) for
/// quick experiments, or [`open()`](Self::open) for persistent storage.
/// Then grab a [`session()`](Self::session) to start querying.
///
/// # Examples
///
/// ```
/// use grafeo_engine::GrafeoDB;
///
/// // Quick in-memory database
/// let db = GrafeoDB::new_in_memory();
///
/// // Add some data
/// db.create_node(&["Person"]);
///
/// // Query it
/// let session = db.session();
/// let result = session.execute("MATCH (p:Person) RETURN p")?;
/// # Ok::<(), grafeo_common::utils::error::Error>(())
/// ```
pub struct GrafeoDB {
    /// Database configuration.
    pub(super) config: Config,
    /// The underlying graph store (None when using an external store). Read
    /// it through `root_store` or `lpg_store`.
    #[cfg(feature = "lpg")]
    pub(super) store: Option<Arc<LpgStore>>,
    /// Schema and metadata catalog shared across sessions.
    pub(super) catalog: Arc<Catalog>,
    /// RDF triple store (if RDF feature is enabled).
    #[cfg(feature = "triple-store")]
    pub(super) rdf_store: Arc<RdfStore>,
    /// Transaction manager.
    pub(super) transaction_manager: Arc<TransactionManager>,
    /// Unified buffer manager.
    pub(super) buffer_manager: Arc<BufferManager>,
    /// Write-ahead log manager (if durability is enabled).
    #[cfg(feature = "wal")]
    pub(super) wal: Option<Arc<LpgWal>>,
    /// Query cache for parsed and optimized plans.
    pub(super) query_cache: Arc<QueryCache>,
    /// Shared commit counter for auto-GC across sessions.
    pub(super) commit_counter: Arc<AtomicUsize>,
    /// Whether the database is open.
    pub(super) is_open: RwLock<bool>,
    /// Change data capture log for tracking mutations.
    #[cfg(feature = "cdc")]
    pub(super) cdc_log: Arc<crate::cdc::CdcLog>,
    /// Whether CDC is active for new sessions and direct CRUD (runtime-mutable).
    #[cfg(feature = "cdc")]
    cdc_enabled: std::sync::atomic::AtomicBool,
    /// Registered embedding models for text-to-vector conversion.
    #[cfg(feature = "embed")]
    pub(super) embedding_models:
        RwLock<hashbrown::HashMap<String, Arc<dyn crate::embedding::EmbeddingModel>>>,
    /// Single-file database manager (when using `.grafeo` format).
    #[cfg(feature = "grafeo-file")]
    pub(super) file_manager: Option<Arc<GrafeoFileManager>>,
    /// Periodic checkpoint timer (when `checkpoint_interval` is configured).
    /// Wrapped in Mutex because `close()` takes `&self` but needs to stop the timer.
    #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
    checkpoint_timer: parking_lot::Mutex<Option<checkpoint_timer::CheckpointTimer>>,
    /// Syncs the WAL in the background under `DurabilityMode::Adaptive`.
    /// Wrapped in Mutex because `close()` takes `&self` but stops it.
    #[cfg(feature = "wal")]
    wal_flusher: parking_lot::Mutex<Option<grafeo_storage::wal::AdaptiveFlusher>>,
    /// The spill directory this open owns (a read-only open's temp
    /// directory), removed with the database once nothing spills into it.
    _spill_root: Option<Arc<spill_directory::SpillDirectory>>,
    /// Where spilled vector columns write their cache files.
    #[cfg(all(
        feature = "lpg",
        feature = "vector-index",
        feature = "mmap",
        not(feature = "temporal")
    ))]
    vector_cache: Option<Arc<spill_directory::SpillDirectory>>,
    /// External read-only graph store (when using with_store() or with_read_store()).
    /// When set, sessions route queries through this store instead of the built-in LpgStore.
    pub(super) external_read_store: Option<Arc<dyn GraphStoreSearch>>,
    /// External writable graph store (when using with_store()).
    /// None for read-only databases created via with_read_store().
    pub(super) external_write_store: Option<Arc<dyn GraphStoreMut>>,
    /// Metrics registry shared across all sessions.
    #[cfg(feature = "metrics")]
    pub(crate) metrics: Option<Arc<crate::metrics::MetricsRegistry>>,
    /// Persistent graph context for one-shot `execute()` calls.
    /// When set, each call to `session()` pre-configures the session to this graph.
    /// Updated after every one-shot `execute()` to reflect `USE GRAPH` / `SESSION RESET`.
    current_graph: RwLock<Option<String>>,
    /// Persistent schema context for one-shot `execute()` calls.
    /// When set, each call to `session()` pre-configures the session to this schema.
    /// Updated after every one-shot `execute()` to reflect `SESSION SET SCHEMA` / `SESSION RESET`.
    current_schema: RwLock<Option<String>>,
    /// Whether this database is open in read-only mode.
    /// When true, sessions automatically enforce read-only transactions.
    read_only: bool,
    /// The recorder of the direct calls that commit at once (while no
    /// transaction is open), made on the first one.
    #[cfg(feature = "lpg")]
    immediate_writes: std::sync::OnceLock<Arc<direct::ImmediateRecorder>>,
    /// Named graph projections (virtual subgraphs), shared with sessions.
    projections:
        Arc<RwLock<std::collections::HashMap<String, Arc<grafeo_core::graph::GraphProjection>>>>,
}

impl GrafeoDB {
    /// Returns a reference to the built-in LPG store.
    ///
    /// # Panics
    ///
    /// Panics if the database was created with [`with_store()`](Self::with_store) or
    /// [`with_read_store()`](Self::with_read_store), which use an external store
    /// instead of the built-in LPG store.
    #[cfg(feature = "lpg")]
    fn lpg_store(&self) -> Arc<LpgStore> {
        self.root_store().expect(
            "no built-in LpgStore: this GrafeoDB was created with an external store \
             (with_store / with_read_store). Use session() or graph_store() instead.",
        )
    }

    /// The built-in LPG store, or `None` with an external store.
    #[cfg(feature = "lpg")]
    fn root_store(&self) -> Option<Arc<LpgStore>> {
        self.store.clone()
    }

    /// Returns a borrowed reference to the active graph store: the external
    /// store, or else the built-in `LpgStore`.
    ///
    /// Unlike [`graph_store()`](Self::graph_store) (which clones an `Arc`),
    /// this borrows from `self`, suitable for constructing accessors that
    /// need `&'a dyn GraphStore` tied to the database lifetime.
    #[cfg(feature = "vector-index")]
    fn graph_store_ref(&self) -> &dyn grafeo_core::graph::GraphStore {
        if let Some(ref ext_read) = self.external_read_store {
            ext_read.as_ref()
        } else {
            #[cfg(feature = "lpg")]
            {
                &**self
                    .store
                    .as_ref()
                    .expect("no graph store: neither an external nor the built-in store")
            }
            #[cfg(not(feature = "lpg"))]
            unreachable!("no graph store available: enable the `lpg` feature or use with_store()")
        }
    }

    /// Returns whether CDC is active (runtime check).
    #[cfg(feature = "cdc")]
    #[inline]
    pub(super) fn cdc_active(&self) -> bool {
        self.cdc_enabled.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Creates an in-memory database, fast to create, gone when dropped.
    ///
    /// Use this for tests, experiments, or when you don't need persistence.
    /// For data that survives restarts, use [`open()`](Self::open) instead.
    ///
    /// # Panics
    ///
    /// Panics if the internal arena allocator cannot be initialized (out of memory).
    /// Use [`with_config()`](Self::with_config) for a fallible alternative.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    /// session.execute("INSERT (:Person {name: 'Alix'})")?;
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    #[must_use]
    pub fn new_in_memory() -> Self {
        Self::with_config(Config::in_memory()).expect("In-memory database creation should not fail")
    }

    /// Opens a database at the given path, creating it if it doesn't exist.
    ///
    /// If you've used this path before, Grafeo recovers your data from the
    /// write-ahead log automatically. First open on a new path creates an
    /// empty database: a single file, whatever the path's extension, with its
    /// write-ahead log in the sidecar directory `<path>.wal/` while it is open.
    ///
    /// A database written by 0.5.x is migrated to the 0.6 file format first:
    /// a `.grafeo` file, or a WAL directory (a directory holding `wal/`),
    /// which becomes a single file at the same path. The old files are kept,
    /// byte for byte: the file (or the whole directory) as `<path>.pre-0.6`,
    /// a file's sidecar WAL as `<path>.pre-0.6.wal`, a checkpoint image
    /// 0.5.44 left pending as `<path>.pre-0.6.checkpoint` and the spill
    /// directory as `<path>.pre-0.6.spill` (embeddings spilled before the
    /// database was closed are read into the 0.6 file). The migration
    /// refuses to start while one of these names is taken (it never replaces
    /// a kept copy), and an open waits up to five seconds for a migration
    /// another process is running before it fails. A kept copy itself (a
    /// 0.5.x database whose name ends in `.pre-0.6`) is never migrated: its
    /// read-write open fails, and a read-only open reads it.
    ///
    /// # Errors
    ///
    /// Returns an error if the path isn't writable or recovery fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::open("./my_social_network")?;
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    #[cfg(feature = "wal")]
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::with_config(Config::persistent(path.as_ref()))
    }

    /// Opens an existing database in read-only mode: it does not allow
    /// mutations, and writes nothing to the file.
    ///
    /// A file written by 0.6 or later is held under a shared file lock until
    /// the database is closed, so multiple processes can read the same
    /// `.grafeo` file concurrently. The database loads the last checkpoint
    /// and replays the sidecar WAL `<path>.wal/` into memory, where a writer
    /// that exited without `close()` left its last commits. Nothing is
    /// written: the WAL stays for the next read-write open. A build without
    /// the `wal` feature cannot replay it, and refuses a file whose sidecar
    /// WAL holds files.
    ///
    /// A database written by 0.5.x is different: it is read once into
    /// memory, a file with its sidecar WAL replayed, a WAL directory by
    /// replaying its WAL, and no lock is held once it is loaded. It is not
    /// migrated, and nothing in it changes.
    ///
    /// # Errors
    ///
    /// Returns an error if the database doesn't exist or can't be read, and,
    /// in a build without the `wal` feature, if its sidecar WAL holds commits
    /// to replay (a non-empty log file; for a 0.5.x file, any file); also
    /// if the database holds data this build cannot read (by 0.5.x too): RDF
    /// triples without `triple-store`, vector or text indexes without
    /// `vector-index` or `text-index`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::open_read_only("./my_graph.grafeo")?;
    /// let session = db.session();
    /// let result = session.execute("MATCH (n) RETURN n LIMIT 10")?;
    /// // Mutations will return an error:
    /// // session.execute("INSERT (:Person)") => Err(ReadOnly)
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    #[cfg(feature = "grafeo-file")]
    pub fn open_read_only(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::with_config(Config::read_only(path.as_ref()))
    }

    /// Creates a database with custom configuration.
    ///
    /// Use this when you need fine-grained control over memory limits,
    /// thread counts, or persistence settings. For most cases,
    /// [`new_in_memory()`](Self::new_in_memory) or [`open()`](Self::open)
    /// are simpler.
    ///
    /// With `Config::encryption` (feature `encryption`), a new database is
    /// created encrypted, and an existing one opens only with the key chain
    /// it was created with.
    ///
    /// # Errors
    ///
    /// Returns an error if the database can't be created or recovery fails,
    /// if the deprecated `StorageFormat::WalDirectory` is given for a path
    /// where nothing exists (WAL directories are no longer created), or if
    /// the path holds neither a database file nor a 0.5.x WAL directory; with
    /// encryption, also if an encrypted database is opened without its key or
    /// with another one, or an unencrypted one is opened with a key; in a
    /// build without the `wal` feature, also if the sidecar WAL of the
    /// database file holds commits only a build with `wal` can replay (a
    /// non-empty log file; for a 0.5.x file, any file); and if the database
    /// holds data this build cannot read, which it would open without and
    /// its next checkpoint drop (by 0.5.x too): RDF triples (in the file or
    /// its WAL) without `triple-store`, vector or text indexes without
    /// `vector-index` or `text-index`.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::{GrafeoDB, Config};
    ///
    /// // In-memory with a 512MB limit
    /// let config = Config::in_memory()
    ///     .with_memory_limit(512 * 1024 * 1024);
    ///
    /// let db = GrafeoDB::with_config(config)?;
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    pub fn with_config(config: Config) -> Result<Self> {
        Self::with_config_and_spill(config, true)
    }

    /// [`with_config`](Self::with_config), where `derive_spill_path` false
    /// leaves a persistent database without the `<path>.spill` default when
    /// `config.spill_path` is `None`: the migration with a key reads the old
    /// database this way, so none of its data reaches a plaintext spill file.
    pub(crate) fn with_config_and_spill(config: Config, derive_spill_path: bool) -> Result<Self> {
        // One spelling of the path for every decision and every side-file
        // name: `db/` (a natural way to name a 0.5.x WAL directory) is `db`,
        // and `.` is the directory it names, so `<path>.pre-0.6` and the
        // other side files go next to it, never inside it. `config.path`
        // keeps the caller's spelling, which `path()` reports.
        let database_path = config.path.as_deref().map(normalize_path).transpose()?;

        // Validate configuration before proceeding
        config.validate().map_err(config_error)?;

        // An encrypted database spills nothing: a spill file would hold its
        // data in plaintext (`validate` refuses an explicit spill path).
        #[cfg(feature = "encryption")]
        let derive_spill_path = derive_spill_path && config.encryption.is_none();

        #[cfg(feature = "lpg")]
        let store = Arc::new(LpgStore::new()?);
        #[cfg(feature = "triple-store")]
        let rdf_store = Arc::new(RdfStore::new());
        let transaction_manager = Arc::new(TransactionManager::new());

        let is_read_only = config.access_mode == crate::config::AccessMode::ReadOnly;

        // Where this open spills (see `spill_directory`): a read-only open
        // writes nothing beside its file, and databases sharing an explicit
        // spill path keep their vector caches apart.
        let spill_layout = spill_directory::SpillLayout::for_open(
            config.spill_path.as_deref(),
            database_path.as_deref().filter(|_| derive_spill_path),
            is_read_only,
        );
        // Where an older build may have spilled embeddings (#594).
        #[cfg(all(feature = "lpg", feature = "grafeo-file", feature = "vector-index"))]
        let legacy_spill_directories =
            legacy_spill::directories(database_path.as_deref(), config.spill_path.as_deref());
        // Whether this open reads a 0.5.x database in place: only then are old
        // spill files in a configured spill path its own, and every vector
        // index is rebuilt (see `legacy_spill`).
        #[cfg(all(feature = "lpg", feature = "grafeo-file", feature = "vector-index"))]
        let mut reads_a_0_5_database = false;

        // Create buffer manager with configured limits
        let buffer_config = BufferManagerConfig {
            budget: config.memory_limit.unwrap_or_else(|| {
                // reason: product of system RAM and 0.75 is always a valid positive usize
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let b = (BufferManagerConfig::detect_system_memory() as f64 * 0.75) as usize;
                b
            }),
            spill_path: spill_layout.root.clone(),
            ..BufferManagerConfig::default()
        };
        let buffer_manager = BufferManager::new(buffer_config);

        // Create catalog early so WAL replay can restore schema definitions
        let catalog = Arc::new(Catalog::new());

        // The keys of an encrypted database (`Config::encryption`), derived
        // once the database id is known: from the file header of an existing
        // database, or picked here for a new one.
        #[cfg(feature = "grafeo-file")]
        let keys = encryption::DatabaseKeys::from_config(&config);
        // Only a single-file database is encrypted.
        #[cfg(all(feature = "encryption", not(feature = "grafeo-file")))]
        if config.encryption.is_some() {
            return Err(Error::InvalidValue(
                "encryption at rest requires a single-file database, which this build \
                 cannot open: enable the `grafeo-file` feature"
                    .to_string(),
            ));
        }

        // What loading a v2 section file leaves for the built database: the
        // indexes to build.
        #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
        let mut loaded_sections = sections::LoadedSections::default();

        // What WAL recovery found at the end of the log, applied once the WAL
        // is open (#411): a torn tail must be sealed before anything new is
        // logged, and a log that ends inside a named graph must switch back
        // to the default graph, where every new group starts.
        #[cfg(all(feature = "wal", feature = "lpg"))]
        let mut wal_torn_tail = false;
        #[cfg(all(feature = "wal", feature = "lpg"))]
        let mut wal_in_named_graph = false;
        // A sidecar WAL a read-write open with `wal_enabled` off replayed:
        // nothing of this handle ever marks or trims it, so the open retires
        // it before it returns (`Some(true)` when it held records).
        #[cfg(all(feature = "wal", feature = "lpg"))]
        let mut retire_sidecar_wal: Option<bool> = None;

        // The migration lock while a read-write open creates a database at a
        // missing path, released once the database exists (see
        // `migration::decide_read_write`).
        #[cfg(feature = "grafeo-file")]
        let mut create_lock: Option<migration::MigrateLock> = None;

        // --- Single-file format (.grafeo) ---
        #[cfg(feature = "grafeo-file")]
        let file_manager: Option<Arc<GrafeoFileManager>> = if is_read_only {
            // Read-only mode: load the file (and replay its sidecar WAL)
            // under a shared lock and write nothing; no WAL is opened.
            let Some(ref db_path) = database_path else {
                return Err(grafeo_common::utils::error::Error::InvalidValue(
                    "read-only mode requires a database path".to_string(),
                ));
            };
            // Only a read-write open finishes a migration a crash cut off.
            migration::check_read_only(db_path)?;
            match detect(db_path)? {
                // A 0.5.x database is read into memory as it is: a file with
                // its sidecar WAL, a WAL directory by replaying its WAL.
                // Nothing keeps it open, and nothing migrates it.
                on_disk @ (OnDisk::LegacyFile | OnDisk::WalDirectory) => {
                    // The 0.5.x reader loads into the LPG store: a build
                    // without it would open an empty database.
                    #[cfg(not(feature = "lpg"))]
                    {
                        let _ = on_disk;
                        return Err(legacy_database_needs_lpg(db_path));
                    }
                    #[cfg(feature = "lpg")]
                    {
                        // It is never encrypted, and only a read-write open
                        // migrates it into an encrypted file.
                        if keys.is_encrypted() {
                            return Err(Error::InvalidValue(format!(
                                "the database is not encrypted, open it without a key: {} \
                                 (written by 0.5.x; a read-write open with the key migrates \
                                 it into an encrypted 0.6 file)",
                                db_path.display()
                            )));
                        }
                        grafeo_common::grafeo_info!(
                            "{} was written by Grafeo 0.5.x and is read in place; a read-write \
                             open migrates it to the 0.6 format (and keeps the 0.5.x files as \
                             {}.pre-0.6)",
                            db_path.display(),
                            db_path.display()
                        );
                        #[cfg(feature = "vector-index")]
                        {
                            reads_a_0_5_database = true;
                        }
                        if on_disk == OnDisk::WalDirectory {
                            Self::load_legacy_directory(
                                db_path,
                                &store,
                                &catalog,
                                #[cfg(feature = "triple-store")]
                                &rdf_store,
                            )?;
                        } else {
                            Self::load_legacy_file(
                                db_path,
                                &store,
                                &catalog,
                                #[cfg(feature = "triple-store")]
                                &rdf_store,
                                &mut loaded_sections,
                            )?;
                        }
                        None
                    }
                }
                // The manager says what is wrong with a file it cannot read.
                OnDisk::Current | OnDisk::Unknown if db_path.is_file() => {
                    let fm = GrafeoFileManager::open_read_only_with_cipher_for(db_path, |id| {
                        keys.container_cipher(id)
                    })?;
                    // Without the `wal` feature its commits since the last
                    // checkpoint cannot be replayed: the file alone would
                    // show the database without them.
                    #[cfg(not(feature = "wal"))]
                    Self::refuse_unreplayable_sidecar_wal(db_path)?;
                    #[cfg(feature = "lpg")]
                    {
                        // Refused when it holds data this build cannot
                        // read: it would show the database without it.
                        loaded_sections = fm.read_image(|image| {
                            sections::load_sections(
                                image,
                                Some(db_path),
                                &store,
                                &catalog,
                                #[cfg(feature = "triple-store")]
                                &rdf_store,
                            )
                        })?;
                        Self::continue_from_checkpoint(&fm, &store);
                    }
                    // A writer that exited without `close()` left its commits
                    // since the last checkpoint in the sidecar WAL. They are
                    // replayed into memory, as a read-write open replays them,
                    // and nothing is written (a torn tail stays for the next
                    // read-write open, which seals it, or with `wal_enabled`
                    // off removes it with the WAL). The shared lock keeps writers
                    // out, so the WAL does not change while it is read.
                    #[cfg(all(feature = "wal", feature = "lpg"))]
                    if fm.has_sidecar_wal() {
                        let recovered = WalRecovery::with_cipher(
                            fm.sidecar_wal_path(),
                            keys.wal_cipher(fm.database_id()),
                        )
                        .recover_with_tail()?;
                        Self::apply_wal_records(
                            &fm.sidecar_wal_path(),
                            &store,
                            &catalog,
                            #[cfg(feature = "triple-store")]
                            &rdf_store,
                            &recovered.records,
                            loaded_sections.unbuilt_mut(),
                        )?;
                    }
                    Some(Arc::new(fm))
                }
                OnDisk::Missing => {
                    return Err(grafeo_common::utils::error::Error::InvalidValue(format!(
                        "read-only open requires an existing database: {} does not exist",
                        db_path.display()
                    )));
                }
                _ => return Err(not_a_database(db_path)),
            }
        } else if let Some(ref db_path) = database_path {
            // An existing path opens as what it holds, whatever the storage
            // format; the format only decides about a missing path, which is
            // always created as a single file. A migration a crash cut off is
            // finished first: until then the database file can be missing,
            // and the path would look new. A decision that creates a database
            // at a missing path is taken holding the migration lock, which a
            // migration holds while the file is missing between its renames.
            // Builds without `lpg` follow the same protocol: they finish a
            // cut-off migration (that only renames), and refuse to migrate.
            #[allow(
                deprecated,
                reason = "the crate names its own deprecated variant, until 0.7.0 removes it"
            )]
            let creates_nothing =
                config.storage_format == crate::config::StorageFormat::WalDirectory;
            // Refused before the migration lock creates the directory its
            // file goes in.
            if creates_nothing && migration::nothing_at(db_path)? {
                return Err(wal_directories_are_no_longer_created(db_path));
            }
            let (on_disk, lock) = migration::decide_read_write(db_path)?;
            create_lock = lock;
            let fm = match on_disk {
                OnDisk::Missing => {
                    if creates_nothing {
                        return Err(wal_directories_are_no_longer_created(db_path));
                    }
                    Self::refuse_leftover_sidecar_wal(db_path)?;
                    let database_id = new_database_id();
                    GrafeoFileManager::create_with_id(
                        db_path,
                        database_id,
                        keys.container_cipher(database_id),
                    )?
                }
                // A 0.5.x database (a file, or a WAL directory) is migrated
                // to a single file in the 0.6 format at the same path
                // (encrypted when a key is configured), then opened.
                #[cfg(feature = "lpg")]
                OnDisk::LegacyFile | OnDisk::WalDirectory => {
                    // Taken only for a path that was missing: the migration
                    // takes the lock itself.
                    drop(create_lock.take());
                    migration::migrate(db_path, &config)?;
                    GrafeoFileManager::open_with_cipher_for(db_path, |id| {
                        keys.container_cipher(id)
                    })?
                }
                // The migration reads the old database into the LPG store,
                // so a build without it cannot migrate.
                #[cfg(not(feature = "lpg"))]
                OnDisk::LegacyFile | OnDisk::WalDirectory => {
                    return Err(legacy_database_needs_lpg(db_path));
                }
                // For any other file the manager says what is wrong.
                OnDisk::Current | OnDisk::Unknown if db_path.is_file() => {
                    let fm = GrafeoFileManager::open_with_cipher_for(db_path, |id| {
                        keys.container_cipher(id)
                    })?;
                    // Without the `wal` feature its commits since the last
                    // checkpoint cannot be replayed, and `close()` would
                    // remove the WAL that holds them.
                    #[cfg(not(feature = "wal"))]
                    Self::refuse_unreplayable_sidecar_wal(db_path)?;
                    fm
                }
                _ => return Err(not_a_database(db_path)),
            };

            #[cfg(feature = "lpg")]
            {
                // Refused when it holds data this build cannot read: the
                // next checkpoint would write the file without it.
                loaded_sections = fm.read_image(|image| {
                    sections::load_sections(
                        image,
                        Some(db_path),
                        &store,
                        &catalog,
                        #[cfg(feature = "triple-store")]
                        &rdf_store,
                    )
                })?;
                Self::continue_from_checkpoint(&fm, &store);
            }

            // A sidecar WAL holds the commits since the last checkpoint that a
            // writer left without `close()`: they are replayed whatever
            // `wal_enabled` says (it decides only whether new commits are
            // logged), and reach the file at the next checkpoint, before
            // `close()` removes the WAL.
            #[cfg(all(feature = "wal", feature = "lpg"))]
            if fm.has_sidecar_wal() {
                let recovery = WalRecovery::with_cipher(
                    fm.sidecar_wal_path(),
                    keys.wal_cipher(fm.database_id()),
                );
                let recovered = recovery.recover_with_tail()?;
                Self::apply_wal_records(
                    &fm.sidecar_wal_path(),
                    &store,
                    &catalog,
                    #[cfg(feature = "triple-store")]
                    &rdf_store,
                    &recovered.records,
                    loaded_sections.unbuilt_mut(),
                )?;
                wal_torn_tail = recovered.torn_tail;
                wal_in_named_graph = Self::ends_in_named_graph(&recovered.records);
                if !config.wal_enabled {
                    retire_sidecar_wal = Some(!recovered.records.is_empty());
                }
            }

            Some(Arc::new(fm))
        } else {
            None
        };

        // The new database file exists now: other opens may look at the
        // path again.
        #[cfg(feature = "grafeo-file")]
        drop(create_lock);

        // The WAL is the sidecar directory of the database file. A read-only
        // open, and one with `wal_enabled` off, logs nothing: it opens no WAL
        // (an existing sidecar WAL was replayed above).
        #[cfg(feature = "wal")]
        let wal = match file_manager {
            Some(ref fm) if !is_read_only && config.wal_enabled => {
                let wal_path = fm.sidecar_wal_path();
                std::fs::create_dir_all(&wal_path)?;

                // Open/create WAL manager with configured durability
                let wal_durability = match config.wal_durability {
                    crate::config::DurabilityMode::Sync => WalDurabilityMode::Sync,
                    crate::config::DurabilityMode::Batch {
                        max_delay_ms,
                        max_records,
                    } => WalDurabilityMode::Batch {
                        max_delay_ms,
                        max_records,
                    },
                    crate::config::DurabilityMode::Adaptive { target_interval_ms } => {
                        WalDurabilityMode::Adaptive { target_interval_ms }
                    }
                    crate::config::DurabilityMode::NoSync => WalDurabilityMode::NoSync,
                };
                let wal_config = WalConfig {
                    durability: wal_durability,
                    ..WalConfig::default()
                };
                // The sidecar WAL of an encrypted database is encrypted from
                // its first record on.
                let wal_cipher = keys.wal_cipher(fm.database_id());
                let wal_manager =
                    LpgWal::with_config_and_cipher(&wal_path, wal_config, wal_cipher)?;
                #[cfg(feature = "lpg")]
                {
                    if wal_torn_tail {
                        wal_manager.seal_torn_tail()?;
                    }
                    if wal_in_named_graph {
                        wal_manager.log(&WalRecord::SwitchGraph { name: None })?;
                    }
                }
                Some(Arc::new(wal_manager))
            }
            _ => None,
        };

        // `Adaptive` leaves syncing the WAL to a background flusher.
        #[cfg(feature = "wal")]
        let wal_flusher = match (&wal, config.wal_durability) {
            (Some(wal), crate::config::DurabilityMode::Adaptive { target_interval_ms }) => {
                let wal = Arc::clone(wal);
                Some(grafeo_storage::wal::AdaptiveFlusher::with_sync(
                    move || wal.sync(),
                    target_interval_ms,
                )?)
            }
            _ => None,
        };

        // Create query cache with default capacity (1000 queries)
        let query_cache = Arc::new(QueryCache::default());

        // After all snapshot/WAL recovery, the database continues at the
        // highest epoch the file, its WAL or any graph reached.
        #[cfg(feature = "lpg")]
        Self::continue_epochs(&store, &transaction_manager);

        #[cfg(feature = "cdc")]
        let cdc_enabled_val = config.cdc_enabled;
        #[cfg(feature = "cdc")]
        let cdc_retention = config.cdc_retention.clone();

        let mut db = Self {
            config,
            #[cfg(feature = "lpg")]
            store: Some(store),
            catalog,
            #[cfg(feature = "triple-store")]
            rdf_store,
            transaction_manager,
            buffer_manager,
            #[cfg(feature = "wal")]
            wal,
            query_cache,
            commit_counter: Arc::new(AtomicUsize::new(0)),
            is_open: RwLock::new(true),
            #[cfg(feature = "cdc")]
            cdc_log: Arc::new(crate::cdc::CdcLog::with_retention(cdc_retention)),
            #[cfg(feature = "cdc")]
            cdc_enabled: std::sync::atomic::AtomicBool::new(cdc_enabled_val),
            #[cfg(feature = "embed")]
            embedding_models: RwLock::new(hashbrown::HashMap::new()),
            #[cfg(feature = "grafeo-file")]
            file_manager,
            #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
            checkpoint_timer: parking_lot::Mutex::new(None),
            #[cfg(feature = "wal")]
            wal_flusher: parking_lot::Mutex::new(wal_flusher),
            _spill_root: spill_layout.root_guard.clone(),
            #[cfg(all(
                feature = "lpg",
                feature = "vector-index",
                feature = "mmap",
                not(feature = "temporal")
            ))]
            vector_cache: spill_layout.vector_cache.clone(),
            external_read_store: None,
            external_write_store: None,
            #[cfg(feature = "metrics")]
            metrics: Some(Arc::new(crate::metrics::MetricsRegistry::new())),
            current_graph: RwLock::new(None),
            current_schema: RwLock::new(None),
            read_only: is_read_only,
            projections: Arc::new(RwLock::new(std::collections::HashMap::new())),
            #[cfg(feature = "lpg")]
            immediate_writes: std::sync::OnceLock::new(),
        };

        // Register storage sections as memory consumers for pressure tracking
        db.register_section_consumers();

        // The indexes the sections did not hold are built from all the data,
        // now that WAL recovery is done.
        #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
        db.finish_load(loaded_sections);

        // With `wal_enabled` off, no WAL of this handle marks or trims the
        // sidecar WAL it replayed: left in place, a crash after a later
        // checkpoint would replay it over the newer image and revert what
        // changed since. The replayed commits are written to the file now,
        // and the WAL is removed. A crash before the removal replays the same
        // commits again, which the file already holds.
        #[cfg(all(feature = "wal", feature = "lpg"))]
        if let (Some(held_records), Some(fm)) = (retire_sidecar_wal, db.file_manager.clone()) {
            if held_records {
                db.checkpoint_to_file(&fm)?;
            }
            grafeo_common::testing::crash::maybe_crash("open:before_remove_replayed_wal");
            fm.remove_sidecar_wal()?;
        }

        // Embeddings an older build spilled come back into their columns
        // (#594), before the checkpoint timer could write the store. A 0.5.x
        // database read in place gets every vector index rebuilt: one that
        // reloaded its spilled embeddings before it closed left no old file,
        // and an index that missed the embeddings set while spilled.
        #[cfg(all(feature = "lpg", feature = "grafeo-file", feature = "vector-index"))]
        db.fold_in_legacy_spill(
            &legacy_spill_directories
                .into_iter()
                .filter(|directory| directory.derived || reads_a_0_5_database)
                .collect::<Vec<_>>(),
            reads_a_0_5_database,
        )?;

        // Start periodic checkpoint timer if configured.
        #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
        db.start_checkpoint_timer();

        // A spill cache is never data: what a previous read-write open (or a
        // crash) left of it goes, under this open's exclusive lock.
        if let Some(ref stale) = spill_layout.stale_cache {
            spill_directory::remove_stale_cache(stale);
        }

        // Phase 8a: apply per-section ForceDisk overrides. Each section
        // type configured as ForceDisk triggers a targeted spill of its
        // matching consumer; sections with Auto/ForceRam are left alone.
        // Must happen after register_section_consumers() which creates
        // the consumers we're about to spill.
        db.apply_force_disk_overrides();

        Ok(db)
    }

    /// Creates a database backed by a custom [`GraphStoreMut`] implementation.
    ///
    /// The external store handles all data persistence. WAL, CDC, and index
    /// management are the responsibility of the store implementation.
    ///
    /// Query execution (all 6 languages, optimizer, planner) works through the
    /// provided store. Admin operations (schema introspection, persistence,
    /// vector/text indexes) are not available on external stores.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use grafeo_engine::{GrafeoDB, Config};
    /// use grafeo_core::graph::GraphStoreMut;
    ///
    /// fn example(store: Arc<dyn GraphStoreMut>) -> grafeo_common::utils::error::Result<()> {
    ///     let db = GrafeoDB::with_store(store, Config::in_memory())?;
    ///     let result = db.execute("MATCH (n) RETURN count(n)")?;
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if config validation fails.
    ///
    /// [`GraphStoreMut`]: grafeo_core::graph::GraphStoreMut
    pub fn with_store(store: Arc<dyn GraphStoreMut>, config: Config) -> Result<Self> {
        config.validate().map_err(config_error)?;

        // Commits continue from the epoch the store is at.
        let transaction_manager = Arc::new(TransactionManager::new());
        transaction_manager.sync_epoch(store.current_epoch());

        let buffer_config = BufferManagerConfig {
            budget: config.memory_limit.unwrap_or_else(|| {
                // reason: product of system RAM and 0.75 is always a valid positive usize
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let b = (BufferManagerConfig::detect_system_memory() as f64 * 0.75) as usize;
                b
            }),
            spill_path: None,
            ..BufferManagerConfig::default()
        };
        let buffer_manager = BufferManager::new(buffer_config);

        let query_cache = Arc::new(QueryCache::default());

        #[cfg(feature = "cdc")]
        let cdc_enabled_val = config.cdc_enabled;

        Ok(Self {
            config,
            #[cfg(feature = "lpg")]
            store: None,
            catalog: Arc::new(Catalog::new()),
            #[cfg(feature = "triple-store")]
            rdf_store: Arc::new(RdfStore::new()),
            transaction_manager,
            buffer_manager,
            #[cfg(feature = "wal")]
            wal: None,
            query_cache,
            commit_counter: Arc::new(AtomicUsize::new(0)),
            is_open: RwLock::new(true),
            #[cfg(feature = "cdc")]
            cdc_log: Arc::new(crate::cdc::CdcLog::new()),
            #[cfg(feature = "cdc")]
            cdc_enabled: std::sync::atomic::AtomicBool::new(cdc_enabled_val),
            #[cfg(feature = "embed")]
            embedding_models: RwLock::new(hashbrown::HashMap::new()),
            #[cfg(feature = "grafeo-file")]
            file_manager: None,
            #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
            checkpoint_timer: parking_lot::Mutex::new(None),
            #[cfg(feature = "wal")]
            wal_flusher: parking_lot::Mutex::new(None),
            _spill_root: None,
            #[cfg(all(
                feature = "lpg",
                feature = "vector-index",
                feature = "mmap",
                not(feature = "temporal")
            ))]
            vector_cache: None,
            external_read_store: Some(Arc::clone(&store) as Arc<dyn GraphStoreSearch>),
            external_write_store: Some(store),
            #[cfg(feature = "metrics")]
            metrics: Some(Arc::new(crate::metrics::MetricsRegistry::new())),
            current_graph: RwLock::new(None),
            current_schema: RwLock::new(None),
            read_only: false,
            projections: Arc::new(RwLock::new(std::collections::HashMap::new())),
            #[cfg(feature = "lpg")]
            immediate_writes: std::sync::OnceLock::new(),
        })
    }

    /// Creates a database backed by a read-only [`GraphStore`].
    ///
    /// The database is set to read-only mode. Write queries (CREATE, SET,
    /// DELETE) will return `TransactionError::ReadOnly`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use grafeo_engine::{GrafeoDB, Config};
    /// use grafeo_core::graph::GraphStoreSearch;
    ///
    /// fn example(store: Arc<dyn GraphStoreSearch>) -> grafeo_common::utils::error::Result<()> {
    ///     let db = GrafeoDB::with_read_store(store, Config::in_memory())?;
    ///     let result = db.execute("MATCH (n) RETURN count(n)")?;
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if config validation fails.
    ///
    /// [`GraphStore`]: grafeo_core::graph::GraphStore
    pub fn with_read_store(store: Arc<dyn GraphStoreSearch>, config: Config) -> Result<Self> {
        config.validate().map_err(config_error)?;

        // Commits continue from the epoch the store is at.
        let transaction_manager = Arc::new(TransactionManager::new());
        transaction_manager.sync_epoch(store.current_epoch());

        let buffer_config = BufferManagerConfig {
            budget: config.memory_limit.unwrap_or_else(|| {
                // reason: product of system RAM and 0.75 is always a valid positive usize
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let b = (BufferManagerConfig::detect_system_memory() as f64 * 0.75) as usize;
                b
            }),
            spill_path: None,
            ..BufferManagerConfig::default()
        };
        let buffer_manager = BufferManager::new(buffer_config);

        let query_cache = Arc::new(QueryCache::default());

        #[cfg(feature = "cdc")]
        let cdc_enabled_val = config.cdc_enabled;

        Ok(Self {
            config,
            #[cfg(feature = "lpg")]
            store: None,
            catalog: Arc::new(Catalog::new()),
            #[cfg(feature = "triple-store")]
            rdf_store: Arc::new(RdfStore::new()),
            transaction_manager,
            buffer_manager,
            #[cfg(feature = "wal")]
            wal: None,
            query_cache,
            commit_counter: Arc::new(AtomicUsize::new(0)),
            is_open: RwLock::new(true),
            #[cfg(feature = "cdc")]
            cdc_log: Arc::new(crate::cdc::CdcLog::new()),
            #[cfg(feature = "cdc")]
            cdc_enabled: std::sync::atomic::AtomicBool::new(cdc_enabled_val),
            #[cfg(feature = "embed")]
            embedding_models: RwLock::new(hashbrown::HashMap::new()),
            #[cfg(feature = "grafeo-file")]
            file_manager: None,
            #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
            checkpoint_timer: parking_lot::Mutex::new(None),
            #[cfg(feature = "wal")]
            wal_flusher: parking_lot::Mutex::new(None),
            _spill_root: None,
            #[cfg(all(
                feature = "lpg",
                feature = "vector-index",
                feature = "mmap",
                not(feature = "temporal")
            ))]
            vector_cache: None,
            external_read_store: Some(store),
            external_write_store: None,
            #[cfg(feature = "metrics")]
            metrics: Some(Arc::new(crate::metrics::MetricsRegistry::new())),
            current_graph: RwLock::new(None),
            current_schema: RwLock::new(None),
            read_only: true,
            projections: Arc::new(RwLock::new(std::collections::HashMap::new())),
            #[cfg(feature = "lpg")]
            immediate_writes: std::sync::OnceLock::new(),
        })
    }

    /// Compacts the database: writes a checkpoint of a persistent database
    /// (as [`wal_checkpoint()`](Self::wal_checkpoint) does), drops the old
    /// versions no open transaction can see any more, in every graph (as
    /// [`gc()`](Self::gc) does), and reports what it did. An in-memory or
    /// read-only database writes no checkpoint.
    ///
    /// The database keeps its one store throughout: writes after `compact()`
    /// go through the same path, and to the WAL, as before it, and open
    /// transactions keep their snapshots and their changes.
    ///
    /// # Errors
    ///
    /// The errors of [`wal_checkpoint()`](Self::wal_checkpoint): a failed
    /// checkpoint, a commit that did not complete, and the database-closed
    /// error after `close()` of a persistent database. No version is dropped
    /// when the checkpoint fails.
    pub fn compact(&mut self) -> Result<crate::admin::CompactReport> {
        let started = std::time::Instant::now();
        #[cfg(feature = "lpg")]
        let checkpointed = self.checkpoint_now()?;
        // A build without the LPG model has no checkpoint of its own to write.
        #[cfg(not(feature = "lpg"))]
        let checkpointed = false;
        let versions_collected = u64::try_from(self.collect_garbage()).unwrap_or(u64::MAX);
        Ok(crate::admin::CompactReport {
            checkpointed,
            versions_collected,
            duration_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
        })
    }

    /// Compacts the database: the same as [`compact()`](Self::compact).
    ///
    /// # Errors
    ///
    /// The errors of [`compact()`](Self::compact).
    #[deprecated(since = "0.6.0", note = "use `compact()`, which this calls")]
    pub fn recompact(&mut self) -> Result<crate::admin::CompactReport> {
        self.compact()
    }

    /// Moves the root store, every named graph and the transaction manager
    /// to one epoch, as a live commit keeps them on one: the highest epoch
    /// a store reached. A load leaves them apart: the root store holds the
    /// epoch of the file's checkpoint and of the last commit its WAL logs,
    /// and with `temporal` replay advances the graph each commit lands in (a
    /// 0.5.43 WAL never switched back to the default graph, so its named
    /// graphs can be ahead of the root store). Reads then see every loaded
    /// and replayed version, a checkpoint records that epoch, and new
    /// commits continue above it.
    #[cfg(feature = "lpg")]
    fn continue_epochs(store: &LpgStore, transaction_manager: &TransactionManager) {
        let graphs: Vec<Arc<LpgStore>> = store
            .graph_names()
            .iter()
            .filter_map(|name| store.graph(name))
            .collect();
        let epoch = graphs
            .iter()
            .map(|graph| graph.current_epoch())
            .fold(store.current_epoch(), std::cmp::max);
        store.sync_epoch(epoch);
        for graph in &graphs {
            graph.sync_epoch(epoch);
        }
        transaction_manager.sync_epoch(epoch);
    }

    /// Raises the root store to the epoch of the file's last checkpoint, which
    /// its database header holds (the LPG section holds it only with
    /// `temporal`), before the commits its WAL logs after it are replayed.
    #[cfg(all(feature = "lpg", feature = "grafeo-file"))]
    fn continue_from_checkpoint(fm: &GrafeoFileManager, store: &LpgStore) {
        store.sync_epoch(grafeo_common::types::EpochId::new(fm.active_header().epoch));
    }

    /// Whether replaying `records` leaves the graph cursor on a named graph.
    ///
    /// Logs written before 0.5.44 could end inside a named graph; new groups
    /// assume they start in the default graph.
    #[cfg(all(feature = "wal", feature = "lpg"))]
    fn ends_in_named_graph(records: &[WalRecord]) -> bool {
        records
            .iter()
            .rev()
            .find_map(|record| match record {
                WalRecord::SwitchGraph { name } => Some(name.is_some()),
                _ => None,
            })
            .unwrap_or(false)
    }

    /// Applies WAL records, those of the WAL `wal`, to restore the database
    /// state.
    ///
    /// Data mutation records are routed through a graph cursor that tracks
    /// `SwitchGraph` context markers, replaying mutations into the correct
    /// named graph (or the default graph when cursor is `None`).
    ///
    /// Graph commands and catalog changes (standalone changes) are applied
    /// as the statements that logged them applied them (see
    /// [`standalone::apply`]); the vector and text indexes they put are left
    /// in `unbuilt`, which also loses those they drop, built from the data
    /// once the database is built.
    ///
    /// # Errors
    ///
    /// Returns an error if a record cannot be applied, naming the WAL for a
    /// standalone change, and at data this build cannot read (an RDF record
    /// without the `triple-store` feature, a vector or text index without
    /// its feature): it would be lost (see [`sections::FeatureData`]).
    #[cfg(all(feature = "wal", feature = "lpg"))]
    fn apply_wal_records(
        wal: &std::path::Path,
        store: &Arc<LpgStore>,
        catalog: &Catalog,
        #[cfg(feature = "triple-store")] rdf_store: &Arc<RdfStore>,
        records: &[WalRecord],
        unbuilt: &mut Vec<catalog_section::GraphIndexes>,
    ) -> Result<()> {
        use grafeo_common::change::StandaloneOp;
        use grafeo_common::utils::error::Error;
        use standalone::Applying;

        // Graph cursor: tracks which named graph receives data mutations.
        // `None` means the default graph.
        let mut current_graph: Option<String> = None;
        let mut target_store: Arc<LpgStore> = Arc::clone(store);

        for record in records {
            match record {
                // --- Standalone changes: as the statement applied them ---
                WalRecord::CreateNamedGraph { name } => {
                    let op = StandaloneOp::CreateGraph { name: name.clone() };
                    standalone::apply(
                        &op,
                        None,
                        store,
                        catalog,
                        &mut Applying::Replay { unbuilt },
                    )?;
                }
                WalRecord::DropNamedGraph { name } => {
                    let op = StandaloneOp::DropGraph { name: name.clone() };
                    standalone::apply(
                        &op,
                        None,
                        store,
                        catalog,
                        &mut Applying::Replay { unbuilt },
                    )?;
                    // Reset cursor if the dropped graph was active
                    if current_graph.as_deref() == Some(name.as_str()) {
                        current_graph = None;
                        target_store = Arc::clone(store);
                    }
                }
                WalRecord::SwitchGraph { name } => {
                    current_graph.clone_from(name);
                    target_store = match &current_graph {
                        None => Arc::clone(store),
                        Some(graph_name) => store
                            .graph_or_create(graph_name)
                            .map_err(|e| Error::Internal(e.to_string()))?,
                    };
                }

                // --- Data mutations: routed through target_store ---
                //
                // Replay can see records the checkpoint container already
                // holds, so every record must be safe to apply twice.
                // Properties and labels are set operations; creating a node
                // or edge that exists would duplicate its label and adjacency
                // entries and counters, so those are skipped (#417).
                WalRecord::CreateNode { id, labels } => {
                    if target_store.get_node(*id).is_none() {
                        let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
                        target_store.create_node_with_id(*id, &label_refs)?;
                    }
                }
                WalRecord::DeleteNode { id } => {
                    target_store.delete_node(*id);
                }
                WalRecord::CreateEdge {
                    id,
                    src,
                    dst,
                    edge_type,
                } => {
                    if target_store.get_edge(*id).is_none() {
                        target_store.create_edge_with_id(*id, *src, *dst, edge_type)?;
                    }
                }
                WalRecord::DeleteEdge { id } => {
                    target_store.delete_edge(*id);
                }
                WalRecord::SetNodeProperty { id, key, value } => {
                    target_store.set_node_property(*id, key, value.clone());
                }
                WalRecord::SetEdgeProperty { id, key, value } => {
                    target_store.set_edge_property(*id, key, value.clone());
                }
                WalRecord::AddNodeLabel { id, label } => {
                    target_store.add_label(*id, label);
                }
                WalRecord::RemoveNodeLabel { id, label } => {
                    target_store.remove_label(*id, label);
                }
                WalRecord::RemoveNodeProperty { id, key } => {
                    target_store.remove_node_property(*id, key)?;
                }
                WalRecord::RemoveEdgeProperty { id, key } => {
                    target_store.remove_edge_property(*id, key)?;
                }

                // --- Schema DDL replay (always on root catalog) ---
                WalRecord::CreateNodeType { .. }
                | WalRecord::DropNodeType { .. }
                | WalRecord::CreateEdgeType { .. }
                | WalRecord::DropEdgeType { .. }
                | WalRecord::CreateGraphType { .. }
                | WalRecord::DropGraphType { .. }
                | WalRecord::CreateSchema { .. }
                | WalRecord::DropSchema { .. }
                | WalRecord::AlterNodeType { .. }
                | WalRecord::AlterEdgeType { .. }
                | WalRecord::AlterGraphType { .. }
                | WalRecord::CreateProcedure { .. }
                | WalRecord::DropProcedure { .. }
                | WalRecord::CreateConstraint { .. }
                | WalRecord::DropConstraint { .. } => {
                    schema_replay::apply_schema_record(catalog, record)?;
                }
                WalRecord::CreateIndex { .. } | WalRecord::DropIndex { .. } => {
                    // Logged by 0.5.x without what the index needs (its graph,
                    // its vector parameters): never replayed. This release logs
                    // an index as a standalone change (`Standalone` below).
                }

                // --- RDF triple replay ---
                #[cfg(feature = "triple-store")]
                WalRecord::InsertRdfTriple { .. }
                | WalRecord::DeleteRdfTriple { .. }
                | WalRecord::ClearRdfGraph { .. }
                | WalRecord::CreateRdfGraph { .. }
                | WalRecord::DropRdfGraph { .. } => {
                    rdf_ops::replay_rdf_wal_record(rdf_store, record)?;
                }
                // This build cannot replay them: the next checkpoint would
                // write the file without the triples, and remove the WAL.
                #[cfg(not(feature = "triple-store"))]
                WalRecord::InsertRdfTriple { .. }
                | WalRecord::DeleteRdfTriple { .. }
                | WalRecord::ClearRdfGraph { .. }
                | WalRecord::CreateRdfGraph { .. }
                | WalRecord::DropRdfGraph { .. } => {
                    return Err(sections::refusal(
                        wal,
                        &[(&sections::RDF_TRIPLES, "WAL records".to_string())],
                    ));
                }

                WalRecord::TransactionCommit { .. } => {
                    // In temporal mode, advance the store epoch on each committed
                    // transaction so that subsequent property/label operations
                    // are recorded at the correct epoch in their VersionLogs.
                    #[cfg(feature = "temporal")]
                    {
                        target_store.new_epoch();
                    }
                }
                WalRecord::TransactionAbort { .. } | WalRecord::Checkpoint { .. } => {
                    // Transaction control records don't need replay action
                    // (recovery already filtered to only committed transactions)
                }
                WalRecord::EpochAdvance { epoch } => {
                    // The epoch the commit before took: the database goes
                    // on at or above the last one the WAL logs, also
                    // without `temporal` (the epoch only moves up).
                    store.sync_epoch(*epoch);
                }
                WalRecord::Standalone { record } => {
                    standalone::replay(
                        record,
                        wal,
                        store,
                        #[cfg(feature = "triple-store")]
                        rdf_store,
                        catalog,
                        unbuilt,
                    )?;
                }
            }
        }
        Ok(())
    }

    // =========================================================================
    // Single-file format helpers
    // =========================================================================

    /// Whether the directory `dir` exists and holds at least one file.
    #[cfg(feature = "grafeo-file")]
    fn holds_files(dir: &std::path::Path) -> Result<bool> {
        let entries = match std::fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(error) if !dir.is_dir() => {
                return Err(Error::Io(std::io::Error::new(
                    error.kind(),
                    format!("cannot inspect {}: {error}", dir.display()),
                )));
            }
            Err(error) => {
                return Err(Error::Io(std::io::Error::new(
                    error.kind(),
                    format!("cannot list {}: {error}", dir.display()),
                )));
            }
        };
        for entry in entries {
            let entry = entry.map_err(|error| {
                Error::Io(std::io::Error::new(
                    error.kind(),
                    format!("cannot list {}: {error}", dir.display()),
                ))
            })?;
            if entry.path().is_file() {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Refuses to create a database at `db_path` next to a sidecar WAL that
    /// holds files. They belong to another database (one that was at this
    /// path, or a WAL moved here): the new database would replay them as its
    /// own or, encrypted under its new key, fail to decrypt them, take them
    /// for a torn tail and delete them at its next checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an error naming the sidecar WAL in that case, or if it cannot
    /// be inspected.
    #[cfg(feature = "grafeo-file")]
    fn refuse_leftover_sidecar_wal(db_path: &std::path::Path) -> Result<()> {
        let wal = grafeo_storage::file::detect::sidecar_wal_path(db_path);
        if !Self::holds_files(&wal)? {
            return Ok(());
        }
        Err(Error::InvalidValue(format!(
            "cannot create a database at {}: the sidecar WAL {} holds files of another \
             database; move it away, or open it with the database it belongs to",
            db_path.display(),
            wal.display()
        )))
    }

    /// Refuses an open (read-write or read-only) of the 0.6 file at `db_path`
    /// in a build without the `wal` feature when its sidecar WAL holds a log
    /// file with records (a non-empty `*.log`, the files a replay reads): they
    /// hold commits since the last checkpoint (a writer exited without
    /// `close()`), which only a build with the `wal` feature can replay. The
    /// file alone would lack them, and a read-write `close()` would remove the
    /// WAL with them. A missing sidecar WAL, or one with nothing to replay
    /// (only `checkpoint.meta` or empty logs), is no obstacle.
    ///
    /// # Errors
    ///
    /// Returns an error naming the database, its WAL, the log files found
    /// and the feature in that case, or if the WAL cannot be inspected.
    #[cfg(all(feature = "grafeo-file", not(feature = "wal")))]
    fn refuse_unreplayable_sidecar_wal(db_path: &std::path::Path) -> Result<()> {
        let wal = grafeo_storage::file::detect::sidecar_wal_path(db_path);
        let cannot_list = |error: std::io::Error| {
            Error::Io(std::io::Error::new(
                error.kind(),
                format!("cannot list {}: {error}", wal.display()),
            ))
        };
        let entries = match std::fs::read_dir(&wal) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(cannot_list(error)),
        };
        let mut logs = Vec::new();
        for entry in entries {
            let entry = entry.map_err(cannot_list)?;
            let path = entry.path();
            if path.extension().is_some_and(|extension| extension == "log")
                && entry.metadata().map_err(cannot_list)?.len() > 0
            {
                logs.push(entry.file_name().to_string_lossy().into_owned());
            }
        }
        if logs.is_empty() {
            return Ok(());
        }
        logs.sort();
        Err(Error::Query(
            grafeo_common::utils::error::QueryError::unsupported(format!(
                "{} has records in its sidecar WAL {} (log files {}), commits that only a build with \
             the `wal` feature can replay: open it with such a build",
                db_path.display(),
                wal.display(),
                logs.join(", ")
            )),
        ))
    }

    /// Loads a 0.5.x WAL-directory database (the directory `path` holding its
    /// WAL in `wal/`) into the stores without writing anything, by replaying
    /// every file of its WAL. The WAL is the only copy of the data there, so
    /// every file is replayed even when an older version left checkpoint
    /// metadata, which would otherwise skip the files below its sequence
    /// (#419). A corrupt or torn record ends the replay of its file, and the
    /// replay goes on with the next file (the recovery is best effort).
    ///
    /// No lock is taken: the directory's `LOCK` has no shared mode, so a
    /// read-only open reads the WAL while a 0.5.44 writer may still append to
    /// it, and sees the commits that reached it. A migration locks the
    /// directory itself before it loads it. A migration in another process
    /// can move the directory away while it is read: afterwards the path is
    /// checked to still be the directory, so such a read fails instead of
    /// returning part of the database.
    ///
    /// # Errors
    ///
    /// Returns an error if `wal/` cannot be listed or a WAL file cannot be
    /// read, if the directory was migrated while it was read, or if a record
    /// cannot be replayed (an RDF record in a build without the
    /// `triple-store` feature); in a build without the `wal` feature, always
    /// (it cannot replay a WAL).
    #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
    fn load_legacy_directory(
        path: &std::path::Path,
        store: &Arc<LpgStore>,
        catalog: &Arc<Catalog>,
        #[cfg(feature = "triple-store")] rdf_store: &Arc<RdfStore>,
    ) -> Result<()> {
        #[cfg(not(feature = "wal"))]
        {
            let _ = (store, catalog);
            #[cfg(feature = "triple-store")]
            let _ = rdf_store;
            Err(Error::Query(
                grafeo_common::utils::error::QueryError::unsupported(format!(
                    "{} is a 0.5.x WAL-directory database, whose data only a build with the `wal` \
                 feature can replay: open or migrate it with such a build",
                    path.display()
                )),
            ))
        }
        #[cfg(feature = "wal")]
        {
            let recovered = WalRecovery::new(path.join("wal")).recover_all_with_tail();
            // Checked first: a recovery error may only be the directory
            // moving away while it was read.
            Self::refuse_migrated_while_read(path, OnDisk::WalDirectory)?;
            Self::apply_wal_records(
                &path.join("wal"),
                store,
                catalog,
                #[cfg(feature = "triple-store")]
                rdf_store,
                &recovered?.records,
                // A 0.5.x log puts no index of its own: its indexes are
                // rebuilt from the data.
                &mut Vec::new(),
            )
        }
    }

    /// Fails if `path` is no longer the 0.5.x database (`expected`, a file or
    /// a WAL directory) that was just read without a lock: a migration in
    /// another process moved it away meanwhile (a migration only moves the old
    /// database away, never back), and what was read may lack part of it.
    ///
    /// # Errors
    ///
    /// Returns an error saying so in that case, or if `path` cannot be
    /// inspected.
    #[cfg(all(feature = "wal", feature = "lpg"))]
    fn refuse_migrated_while_read(path: &std::path::Path, expected: OnDisk) -> Result<()> {
        if detect(path)? == expected {
            return Ok(());
        }
        Err(Error::Internal(format!(
            "{} was migrated to the 0.6 format by another process while it was read; open it \
             again",
            path.display()
        )))
    }

    /// Loads a database file written by 0.5.x into the stores without
    /// writing anything: its sections (container v2) or its snapshot
    /// (container v1), then the records of its sidecar WAL, where a 0.5.x
    /// process that exited without `close()` left its last changes.
    ///
    /// The file is held under a shared lock only while it is read.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be locked or read, lists a section
    /// twice, a section or the snapshot cannot be decoded, the WAL cannot be
    /// recovered, or the file or its WAL holds data this build cannot read
    /// (see [`sections::FeatureData`]); in a build without the `wal`
    /// feature, also if the sidecar WAL holds files.
    #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
    fn load_legacy_file(
        path: &std::path::Path,
        store: &Arc<LpgStore>,
        catalog: &Arc<Catalog>,
        #[cfg(feature = "triple-store")] rdf_store: &Arc<RdfStore>,
        loaded: &mut sections::LoadedSections,
    ) -> Result<()> {
        use grafeo_common::storage::{MemoryImage, ServedOnce};
        use grafeo_storage::file::legacy::{LegacyContents, LegacyFile};

        // Without the `wal` feature the sidecar WAL cannot be replayed, and
        // the file alone would lack its changes.
        migration::refuse_unreplayable_wal(path, cfg!(feature = "wal"))?;
        let file = LegacyFile::open(path, None)?;
        match file.contents()? {
            LegacyContents::Empty => {}
            LegacyContents::Snapshot(data) => Self::apply_snapshot_data(
                path,
                store,
                catalog,
                #[cfg(feature = "triple-store")]
                rdf_store,
                &data,
            )?,
            // Each section is served once and freed as soon as it is
            // loaded: a migration of a large 0.5.x file does not hold the
            // whole file's sections until the load returns.
            LegacyContents::Sections(stored) => {
                // A file with data this build cannot read (a compacted
                // base, triples, vector or text indexes) is neither read
                // nor migrated.
                *loaded = sections::load_sections(
                    &ServedOnce::new(MemoryImage::from_raw(stored)?),
                    Some(path),
                    store,
                    catalog,
                    #[cfg(feature = "triple-store")]
                    rdf_store,
                )?;
            }
        }
        // The epoch of its last checkpoint. 0.5.x started the epochs over at
        // every open, so its WAL can log lower ones after it (replay keeps
        // the highest), and a migration continues above both.
        store.sync_epoch(grafeo_common::types::EpochId::new(file.header().epoch));

        #[cfg(feature = "wal")]
        {
            let wal_path = file.sidecar_wal_path();
            let recovered = wal_path
                .exists()
                .then(|| WalRecovery::new(&wal_path).recover_with_tail())
                .transpose();
            // A migration in another process moves the file and then its
            // sidecar WAL: a WAL read (or found missing) meanwhile may lack
            // records, and a recovery error may only be that move.
            Self::refuse_migrated_while_read(path, OnDisk::LegacyFile)?;
            if let Some(recovered) = recovered? {
                Self::apply_wal_records(
                    &wal_path,
                    store,
                    catalog,
                    #[cfg(feature = "triple-store")]
                    rdf_store,
                    &recovered.records,
                    loaded.unbuilt_mut(),
                )?;
            }
        }
        Ok(())
    }

    /// Applies the snapshot blob of the 0.5.x container v1 file `path` to
    /// restore the store and catalog.
    #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
    fn apply_snapshot_data(
        path: &std::path::Path,
        store: &Arc<LpgStore>,
        catalog: &Arc<crate::catalog::Catalog>,
        #[cfg(feature = "triple-store")] rdf_store: &Arc<RdfStore>,
        data: &[u8],
    ) -> Result<()> {
        // v1 blob format: pass through to legacy loader
        persistence::load_snapshot_into_store(
            path,
            store,
            catalog,
            #[cfg(feature = "triple-store")]
            rdf_store,
            data,
        )
    }

    // =========================================================================
    // Session & Configuration
    // =========================================================================

    /// Opens a new session for running queries.
    ///
    /// Sessions are cheap to create: spin up as many as you need. Each
    /// gets its own transaction context, so concurrent sessions won't
    /// block each other on reads.
    ///
    /// # Panics
    ///
    /// Panics if the database was configured with an external graph store and
    /// the internal arena allocator cannot be initialized (out of memory).
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    ///
    /// // Run queries through the session
    /// let result = session.execute("MATCH (n) RETURN count(n)")?;
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    #[must_use]
    pub fn session(&self) -> Session {
        self.create_session_inner(None)
    }

    /// Creates a session scoped to the given identity.
    ///
    /// The identity determines what operations the session is allowed to
    /// perform. A [`Role::ReadOnly`](crate::auth::Role::ReadOnly) identity
    /// creates a read-only session; a [`Role::ReadWrite`](crate::auth::Role::ReadWrite)
    /// identity allows data mutations but not schema DDL; a
    /// [`Role::Admin`](crate::auth::Role::Admin) identity has full access.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::{GrafeoDB, auth::{Identity, Role}};
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let identity = Identity::new("app-service", [Role::ReadWrite]);
    /// let session = db.session_with_identity(identity);
    /// ```
    #[must_use]
    pub fn session_with_identity(&self, identity: crate::auth::Identity) -> Session {
        let force_read_only = !identity.can_write();
        self.create_session_inner_full(None, force_read_only, identity)
    }

    /// Creates a session scoped to a single role.
    ///
    /// Convenience shorthand for
    /// `session_with_identity(Identity::new("anonymous", [role]))`.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::{GrafeoDB, auth::Role};
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let reader = db.session_with_role(Role::ReadOnly);
    /// ```
    #[must_use]
    pub fn session_with_role(&self, role: crate::auth::Role) -> Session {
        self.session_with_identity(crate::auth::Identity::new("anonymous", [role]))
    }

    /// Creates a session with an explicit CDC override.
    ///
    /// When `cdc_enabled` is `true`, mutations in this session are tracked
    /// regardless of the database default. When `false`, mutations are not
    /// tracked regardless of the database default.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    ///
    /// // Opt in to CDC for just this session
    /// let tracked = db.session_with_cdc(true);
    /// tracked.execute("INSERT (:Person {name: 'Alix'})")?;
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    #[cfg(feature = "cdc")]
    #[must_use]
    pub fn session_with_cdc(&self, cdc_enabled: bool) -> Session {
        self.create_session_inner(Some(cdc_enabled))
    }

    /// Creates a read-only session regardless of the database's access mode.
    ///
    /// Mutations executed through this session will fail with
    /// `TransactionError::ReadOnly`. Useful for replication replicas where
    /// the database itself must remain writable (for applying CDC changes)
    /// but client-facing queries must be read-only.
    ///
    /// **Deprecated**: Use `session_with_role(Role::ReadOnly)` instead.
    #[deprecated(
        since = "0.5.36",
        note = "use session_with_role(Role::ReadOnly) instead"
    )]
    #[must_use]
    pub fn session_read_only(&self) -> Session {
        self.session_with_role(crate::auth::Role::ReadOnly)
    }

    /// Shared session creation logic.
    ///
    /// `cdc_override` overrides the database-wide `cdc_enabled` default when
    /// `Some`. `None` falls back to the database default.
    #[allow(unused_variables)] // cdc_override unused when cdc feature is off
    fn create_session_inner(&self, cdc_override: Option<bool>) -> Session {
        self.create_session_inner_full(cdc_override, false, crate::auth::Identity::anonymous())
    }

    /// Shared session creation with all overrides.
    #[allow(unused_variables)]
    fn create_session_inner_full(
        &self,
        cdc_override: Option<bool>,
        force_read_only: bool,
        identity: crate::auth::Identity,
    ) -> Session {
        let session_cfg = || crate::session::SessionConfig {
            transaction_manager: Arc::clone(&self.transaction_manager),
            query_cache: Arc::clone(&self.query_cache),
            catalog: Arc::clone(&self.catalog),
            factorized_execution: self.config.factorized_execution,
            shuffle_unordered: self.config.shuffle_unordered,
            graph_model: self.config.graph_model,
            query_timeout: self.config.query_timeout,
            max_property_size: self.config.max_property_size,
            path_search_budget: self.config.path_search_budget(),
            #[cfg(feature = "spill")]
            buffer_manager: Some(Arc::clone(&self.buffer_manager)),
            commit_counter: Arc::clone(&self.commit_counter),
            gc_interval: self.config.gc_interval,
            read_only: self.read_only || force_read_only,
            identity: identity.clone(),
            #[cfg(feature = "lpg")]
            projections: Arc::clone(&self.projections),
        };

        if let Some(ref ext_read) = self.external_read_store {
            return Session::with_external_store(
                Arc::clone(ext_read),
                self.external_write_store.as_ref().map(Arc::clone),
                session_cfg(),
            )
            .expect("arena allocation for external store session");
        }

        #[cfg(all(feature = "lpg", feature = "triple-store"))]
        let mut session =
            Session::with_rdf_store(self.lpg_store(), Arc::clone(&self.rdf_store), session_cfg());
        #[cfg(all(feature = "lpg", not(feature = "triple-store")))]
        let mut session = Session::with_store(self.lpg_store(), session_cfg());
        #[cfg(not(feature = "lpg"))]
        let mut session =
            Session::with_external_store(self.graph_store(), self.graph_store_mut(), session_cfg())
                .expect("session creation for non-lpg build");

        #[cfg(all(feature = "wal", feature = "lpg"))]
        if let Some(ref wal) = self.wal {
            session.set_wal(Arc::clone(wal));
        }

        #[cfg(feature = "cdc")]
        {
            let should_enable = cdc_override.unwrap_or_else(|| self.cdc_active());
            if should_enable {
                session.set_cdc_log(Arc::clone(&self.cdc_log));
            }
        }

        #[cfg(feature = "metrics")]
        {
            if let Some(ref m) = self.metrics {
                session.set_metrics(Arc::clone(m));
                m.session_created
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                m.session_active
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }

        // Propagate persistent graph context to the new session
        if let Some(ref graph) = *self.current_graph.read() {
            session.use_graph(graph);
        }

        // Propagate persistent schema context to the new session
        if let Some(ref schema) = *self.current_schema.read() {
            session.set_schema(schema);
        }

        // Suppress unused_mut when cdc/wal are disabled
        let _ = &mut session;

        session
    }

    /// Returns the current graph name, if any.
    ///
    /// This is the persistent graph context used by one-shot `execute()` calls.
    /// It is updated whenever `execute()` encounters `USE GRAPH`, `SESSION SET GRAPH`,
    /// or `SESSION RESET`.
    #[must_use]
    pub fn current_graph(&self) -> Option<String> {
        self.current_graph.read().clone()
    }

    /// Sets the current graph context for subsequent one-shot `execute()` calls.
    ///
    /// This is equivalent to running `USE GRAPH <name>` but without creating a session.
    /// Pass `None` to reset to the default graph.
    ///
    /// # Errors
    ///
    /// Returns an error if the named graph does not exist.
    pub fn set_current_graph(&self, name: Option<&str>) -> Result<()> {
        #[cfg(feature = "lpg")]
        if let Some(name) = name
            && !name.eq_ignore_ascii_case("default")
            && let Some(store) = self.root_store()
            && store.graph(name).is_none()
        {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!("Graph '{name}' does not exist"),
            )));
        }
        *self.current_graph.write() = name.map(ToString::to_string);
        Ok(())
    }

    /// Returns the current schema name, if any.
    ///
    /// This is the persistent schema context used by one-shot `execute()` calls.
    /// It is updated whenever `execute()` encounters `SESSION SET SCHEMA` or `SESSION RESET`.
    #[must_use]
    pub fn current_schema(&self) -> Option<String> {
        self.current_schema.read().clone()
    }

    /// Sets the current schema context for subsequent one-shot `execute()` calls.
    ///
    /// This is equivalent to running `SESSION SET SCHEMA <name>` but without creating
    /// a session. Pass `None` to clear the schema context.
    ///
    /// # Errors
    ///
    /// Returns an error if the named schema does not exist.
    pub fn set_current_schema(&self, name: Option<&str>) -> Result<()> {
        if let Some(name) = name
            && !self.catalog.schema_exists(name)
        {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!("Schema '{name}' does not exist"),
            )));
        }
        *self.current_schema.write() = name.map(ToString::to_string);
        Ok(())
    }

    /// Returns `true` if this database was opened in read-only mode.
    #[must_use]
    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Returns the graph data model of this database.
    #[must_use]
    pub fn graph_model(&self) -> crate::config::GraphModel {
        self.config.graph_model
    }

    /// Returns the configured memory limit in bytes, if any.
    #[must_use]
    pub fn memory_limit(&self) -> Option<usize> {
        self.config.memory_limit
    }

    /// Returns a point-in-time snapshot of all metrics.
    ///
    /// If the `metrics` feature is disabled or the registry is not
    /// initialized, returns a default (all-zero) snapshot.
    #[cfg(feature = "metrics")]
    #[must_use]
    pub fn metrics(&self) -> crate::metrics::MetricsSnapshot {
        let mut snapshot = self
            .metrics
            .as_ref()
            .map_or_else(crate::metrics::MetricsSnapshot::default, |m| m.snapshot());

        // Augment with cache stats from the query cache (not tracked in the registry)
        let cache_stats = self.query_cache.stats();
        snapshot.cache_hits = cache_stats.parsed_hits + cache_stats.optimized_hits;
        snapshot.cache_misses = cache_stats.parsed_misses + cache_stats.optimized_misses;
        snapshot.cache_size = cache_stats.parsed_size + cache_stats.optimized_size;
        snapshot.cache_invalidations = cache_stats.invalidations;

        snapshot
    }

    /// Returns all metrics in Prometheus text exposition format.
    ///
    /// The output is ready to serve from an HTTP `/metrics` endpoint.
    #[cfg(feature = "metrics")]
    #[must_use]
    pub fn metrics_prometheus(&self) -> String {
        self.metrics
            .as_ref()
            .map_or_else(String::new, |m| m.to_prometheus())
    }

    /// Resets all metrics counters and histograms to zero.
    #[cfg(feature = "metrics")]
    pub fn reset_metrics(&self) {
        if let Some(ref m) = self.metrics {
            m.reset();
        }
        self.query_cache.reset_stats();
    }

    /// Returns the underlying (default) store.
    ///
    /// This provides direct access to the LPG store for algorithm implementations
    /// and admin operations (index management, schema introspection, MVCC internals).
    ///
    /// For code that only needs read/write graph operations, prefer
    /// [`graph_store()`](Self::graph_store) which returns the trait interface.
    ///
    /// # Panics
    ///
    /// Panics if the database uses an external store
    /// ([`with_store`](Self::with_store), [`with_read_store`](Self::with_read_store)).
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn store(&self) -> Arc<LpgStore> {
        self.lpg_store()
    }

    // === Named Graph Management ===

    /// Creates a named graph. Returns `true` if created, `false` if it already exists.
    ///
    /// The graph exists at once, outside any transaction (also while one is
    /// open, whose rollback keeps it), and is logged as a change of its own:
    /// commits are held off meanwhile, so a checkpoint or `close()` sees all
    /// of it or none of it.
    ///
    /// # Errors
    ///
    /// Returns an error if arena allocation fails, or if the database uses an
    /// external store ([`with_store`](Self::with_store),
    /// [`with_read_store`](Self::with_read_store)), which has no named graphs;
    /// the database-closed error after `close()` of a persistent database,
    /// and the incomplete-commit error after a commit that did not complete.
    #[cfg(feature = "lpg")]
    pub fn create_graph(&self, name: &str) -> Result<bool> {
        if self.root_store().is_none() {
            return Err(grafeo_common::utils::error::Error::Query(
                grafeo_common::utils::error::QueryError::new(
                    grafeo_common::utils::error::QueryErrorKind::Semantic,
                    "Named graphs need the built-in store; this database uses an external store",
                ),
            ));
        }
        let held = self.hold_for_standalone(false)?;
        // Checked under the hold, as every standalone change.
        if self.lpg_store().graph(name).is_some() {
            return Ok(false);
        }
        let mut change = crate::transaction::StandaloneChange::new();
        change.push(grafeo_common::change::StandaloneOp::CreateGraph {
            name: name.to_string(),
        });
        self.commit_standalone(change, &held)?;
        Ok(true)
    }

    /// Drops a named graph. Returns `true` if dropped, `false` if it did not
    /// exist (also on an external store, which has no named graphs).
    ///
    /// If the dropped graph was the active graph context, the context is reset
    /// to the default graph. As [`create_graph`](Self::create_graph), it holds
    /// commits off while it runs, and it also waits for the writes of open
    /// transactions in progress: a graph an open transaction has changes in
    /// is not dropped (the transaction would commit into a graph that no
    /// longer exists), and a write that resolved the graph before the drop
    /// fails after it.
    ///
    /// # Errors
    ///
    /// Returns a write conflict while an open transaction has changes in the
    /// graph (drop it once that transaction commits or rolls back); the
    /// database-closed error after `close()` of a persistent database, and
    /// the incomplete-commit error after a commit that did not complete.
    #[cfg(feature = "lpg")]
    pub fn drop_graph(&self, name: &str) -> Result<bool> {
        if self.root_store().is_none() {
            return Ok(false);
        }
        let held = self.hold_for_standalone(true)?;
        // Checked under the hold, as every standalone change.
        if self.lpg_store().graph(name).is_none() {
            return Ok(false);
        }
        standalone::refuse_drop_with_open_changes(&self.transaction_manager, &held, name)?;
        let mut change = crate::transaction::StandaloneChange::new();
        change.push(grafeo_common::change::StandaloneOp::DropGraph {
            name: name.to_string(),
        });
        self.commit_standalone(change, &held)?;
        let mut current = self.current_graph.write();
        if current
            .as_deref()
            .is_some_and(|g| g.eq_ignore_ascii_case(name))
        {
            *current = None;
        }
        Ok(true)
    }

    /// Returns all named graph names (none on an external store).
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn list_graphs(&self) -> Vec<String> {
        self.root_store()
            .map(|store| store.graph_names())
            .unwrap_or_default()
    }

    // === Graph Projections ===

    /// Creates a named graph projection (virtual subgraph).
    ///
    /// The projection filters the graph selected now (see
    /// [`set_current_graph`](Self::set_current_graph); the default graph when
    /// none is selected) to the nodes with the specified labels and the edges
    /// with the specified types, and keeps reading that graph whatever is
    /// selected later. Returns `true` if created, `false` if a projection with
    /// that name already exists.
    ///
    /// # Errors
    ///
    /// Returns an error if the selected graph no longer exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    /// use grafeo_core::graph::ProjectionSpec;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let spec = ProjectionSpec::new()
    ///     .with_node_labels(["Person", "City"])
    ///     .with_edge_types(["LIVES_IN"]);
    /// assert!(db.create_projection("social", spec)?);
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    pub fn create_projection(
        &self,
        name: impl Into<String>,
        spec: grafeo_core::graph::ProjectionSpec,
    ) -> Result<bool> {
        use grafeo_core::graph::GraphProjection;
        use std::collections::hash_map::Entry;

        let store = self.selected_graph_store()?;
        let projection = Arc::new(GraphProjection::new(store, spec));
        let mut projections = self.projections.write();
        Ok(match projections.entry(name.into()) {
            Entry::Occupied(_) => false,
            Entry::Vacant(e) => {
                e.insert(projection);
                true
            }
        })
    }

    /// Drops a named graph projection. Returns `true` if it existed.
    pub fn drop_projection(&self, name: &str) -> bool {
        self.projections.write().remove(name).is_some()
    }

    /// Returns the names of all graph projections.
    #[must_use]
    pub fn list_projections(&self) -> Vec<String> {
        self.projections.read().keys().cloned().collect()
    }

    /// Returns a named projection as a [`GraphStoreSearch`] trait object.
    #[must_use]
    pub fn projection(&self, name: &str) -> Option<Arc<dyn GraphStoreSearch>> {
        self.projections
            .read()
            .get(name)
            .map(|p| Arc::clone(p) as Arc<dyn GraphStoreSearch>)
    }

    /// The store of the graph [`set_current_graph`](Self::set_current_graph) and
    /// [`set_current_schema`](Self::set_current_schema) select, or of the default
    /// graph when they select none, as the direct API reads it. Graph algorithms
    /// read this store.
    ///
    /// # Errors
    ///
    /// Returns an error if the selected graph no longer exists.
    pub fn selected_graph_store(&self) -> Result<Arc<dyn GraphStoreSearch>> {
        #[cfg(feature = "lpg")]
        {
            self.read_store(direct::DirectTarget::Current)
        }
        // Without the LPG model there are no named graphs to select.
        #[cfg(not(feature = "lpg"))]
        {
            Ok(self.graph_store())
        }
    }

    /// Returns the graph store as a trait object.
    ///
    /// Returns a read-only trait object for the active graph store.
    ///
    /// This provides the [`GraphStoreSearch`] interface (graph-structure reads
    /// plus text/vector search capabilities) for code that only needs read
    /// operations. For write access, use [`graph_store_mut()`](Self::graph_store_mut).
    #[must_use]
    pub fn graph_store(&self) -> Arc<dyn GraphStoreSearch> {
        if let Some(ref ext_read) = self.external_read_store {
            Arc::clone(ext_read)
        } else {
            #[cfg(feature = "lpg")]
            {
                self.lpg_store() as Arc<dyn GraphStoreSearch>
            }
            #[cfg(not(feature = "lpg"))]
            unreachable!("no graph store available: enable the `lpg` feature or use with_store()")
        }
    }

    /// Returns the writable graph store, if available.
    ///
    /// Returns `None` for read-only databases created via
    /// [`with_read_store()`](Self::with_read_store).
    #[must_use]
    pub fn graph_store_mut(&self) -> Option<Arc<dyn GraphStoreMut>> {
        if self.external_read_store.is_some() {
            self.external_write_store.as_ref().map(Arc::clone)
        } else {
            #[cfg(feature = "lpg")]
            {
                Some(self.lpg_store() as Arc<dyn GraphStoreMut>)
            }
            #[cfg(not(feature = "lpg"))]
            {
                None
            }
        }
    }

    /// Garbage collects old MVCC versions that are no longer visible.
    ///
    /// Determines the minimum epoch required by active transactions and prunes
    /// the versions older than that threshold, in the default graph and every
    /// named graph. Also cleans up completed transaction metadata in the
    /// transaction manager, and prunes the CDC event log according to its
    /// retention policy.
    pub fn gc(&self) {
        self.collect_garbage();
    }

    /// What [`gc()`](Self::gc) does; returns how many versions it dropped.
    fn collect_garbage(&self) -> usize {
        #[cfg(feature = "lpg")]
        let versions = self.root_store().map_or(0, |store| {
            let min_epoch = self.transaction_manager.min_active_epoch();
            let named: usize = store
                .graph_names()
                .iter()
                .filter_map(|name| store.graph(name))
                .map(|graph| graph.gc_versions(min_epoch))
                .sum();
            store.gc_versions(min_epoch) + named
        });
        #[cfg(not(feature = "lpg"))]
        let versions = 0;
        #[cfg(all(feature = "lpg", feature = "cdc"))]
        let current_epoch = self.transaction_manager.current_epoch();
        self.transaction_manager.gc();

        // Prune CDC events based on retention config (epoch + count limits)
        #[cfg(feature = "cdc")]
        if self.cdc_enabled.load(std::sync::atomic::Ordering::Relaxed) {
            #[cfg(feature = "lpg")]
            self.cdc_log.apply_retention(current_epoch);
        }
        versions
    }

    /// Returns the buffer manager for memory-aware operations.
    #[must_use]
    pub fn buffer_manager(&self) -> &Arc<BufferManager> {
        &self.buffer_manager
    }

    /// Returns the query cache.
    #[must_use]
    pub fn query_cache(&self) -> &Arc<QueryCache> {
        &self.query_cache
    }

    /// Clears all cached query plans.
    ///
    /// This is called automatically after DDL operations, but can also be
    /// invoked manually after external schema changes (e.g., WAL replay,
    /// import) or when you want to force re-optimization of all queries.
    pub fn clear_plan_cache(&self) {
        self.query_cache.clear();
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    /// Closes the database, flushing all pending writes.
    ///
    /// For persistent databases, this ensures everything is safely on disk.
    /// Called automatically when the database is dropped, but you can call
    /// it explicitly if you need to guarantee durability at a specific point.
    ///
    /// Once `close()` of a persistent database starts, these fail with
    /// [`TransactionError::DatabaseClosed`](grafeo_common::utils::error::TransactionError::DatabaseClosed):
    ///
    /// - commits, and statements that write, in every query language (SPARQL
    ///   updates included), schema statements and graph commands
    ///   (`CREATE GRAPH`, `DROP GRAPH`);
    /// - the direct writes: the node, edge, property and label calls (on the
    ///   database, a session or a graph handle), [`create_graph`](Self::create_graph),
    ///   [`drop_graph`](Self::drop_graph), the property, vector and text index
    ///   calls, the bulk imports, `batch_insert_rdf` and `restore_snapshot`;
    /// - what persists: `wal_checkpoint`, `save`, `backup_full`,
    ///   `backup_incremental`, `async_write_snapshot`, `compact` and
    ///   `recompact` (the file is released: another handle may have written
    ///   it since).
    ///
    /// `close()` waits for one already in progress, which completes and is
    /// written: a commit, a statement, a schema statement or graph command, an
    /// RDF update, a direct write, a direct graph or index call (a vector or
    /// text index is built first and waited for only once it installs), a
    /// bulk import or `batch_insert_rdf` (its input is read first, and waited
    /// for only once it changes the store), `restore_snapshot`, and a
    /// checkpoint, save or backup. Reads, `to_memory`, `export_snapshot` and
    /// projections (`CREATE PROJECTION`, which change only a session) still
    /// work. An in-memory database keeps taking writes.
    ///
    /// A transaction still open when `close()` runs is left out of the file:
    /// the final checkpoint, like every checkpoint, writes the committed
    /// state (what the transaction deleted is kept, the values and labels it
    /// changed are written as they were committed, nothing it created is
    /// written, and when it changed the default graph, the vector and text
    /// indexes are left out and the next open builds them from the data). Its
    /// commit then fails with `DatabaseClosed`; it can only roll back.
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL can't be flushed (check disk space/permissions).
    /// After a commit that did not complete (see
    /// [`TransactionManager`]), it
    /// writes no checkpoint, keeps the WAL, releases the database, and
    /// returns that commit's error. The next open shows the database without
    /// the failed commit when it failed before its WAL records were written;
    /// after them, the WAL holds the whole commit, and the next open replays
    /// it whole.
    pub fn close(&self) -> Result<()> {
        let mut is_open = self.is_open.write();
        if !*is_open {
            return Ok(());
        }

        // A closed handle still reads its spilled columns, but spills no
        // more: once the lock is released, `<file>.spill/cache/` belongs to
        // the next read-write open (#594).
        #[cfg(all(
            feature = "lpg",
            feature = "vector-index",
            feature = "mmap",
            not(feature = "temporal")
        ))]
        if let Some(ref cache) = self.vector_cache {
            cache.close_for_writes();
        }

        // From here on, commits, writes outside a transaction and schema
        // changes fail: one that ran after the final checkpoint below would be
        // written only to the WAL this close removes (or, without a WAL,
        // nowhere). A commit or checkpoint in progress completes first (the
        // commit lock), and the final checkpoint holds it. Taken before the
        // timer stops: the lock is released again before the timer thread is
        // joined. An in-memory database has nothing to persist and keeps
        // taking writes.
        if self.config.path.is_some() && !self.read_only {
            self.transaction_manager.close_for_writes();
        }

        // Stop the periodic checkpoint timer first, before any early return,
        // so it does not race with the closed file manager.
        #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
        if let Some(mut timer) = self.checkpoint_timer.lock().take() {
            timer.stop();
        }

        // Stop the WAL flusher before the WAL is synced and closed below; its
        // shutdown syncs once more.
        #[cfg(feature = "wal")]
        if let Some(mut flusher) = self.wal_flusher.lock().take()
            && let Err(e) = flusher.shutdown()
        {
            grafeo_common::grafeo_warn!("failed to stop the WAL flusher: {e}");
        }

        // Read-only databases: just release the shared lock, no checkpointing
        if self.read_only {
            #[cfg(feature = "grafeo-file")]
            if let Some(ref fm) = self.file_manager {
                fm.close()?;
            }
            *is_open = false;
            return Ok(());
        }

        // After a commit that did not complete, the store holds its stamped
        // part, which no checkpoint may write. The WAL holds every complete
        // commit, and the failed one only as a complete group (when it failed
        // after writing it): it stays for the next open to replay, the file
        // keeps its last checkpoint, and both are released.
        if self.transaction_manager.has_incomplete_commit() {
            #[cfg(feature = "wal")]
            if let Some(ref wal) = self.wal {
                if let Err(e) = wal.sync() {
                    grafeo_common::grafeo_warn!("failed to sync the WAL while closing: {e}");
                }
                wal.close_active_log();
            }
            #[cfg(feature = "grafeo-file")]
            if let Some(ref fm) = self.file_manager
                && let Err(e) = fm.close()
            {
                grafeo_common::grafeo_warn!("failed to release the database file: {e}");
            }
            *is_open = false;
            return self.transaction_manager.check_no_incomplete_commit();
        }

        // Checkpoint to the database file, then remove the sidecar WAL.
        #[cfg(feature = "grafeo-file")]
        if let Some(ref fm) = self.file_manager {
            // Flush WAL first so all records are on disk before we snapshot
            #[cfg(feature = "wal")]
            if let Some(ref wal) = self.wal {
                wal.sync()?;
            }
            let flush_result = self.checkpoint_to_file(fm)?;

            // Release WAL file handles before removing sidecar directory.
            // On Windows, open handles prevent directory deletion.
            #[cfg(feature = "wal")]
            if let Some(ref wal) = self.wal {
                wal.close_active_log();
            }

            // Only remove the sidecar WAL after verifying the checkpoint wrote
            // data to the container. If nothing was written and the WAL had
            // records, keep the sidecar so the next open can recover from it.
            #[cfg(feature = "wal")]
            let has_wal_records = self.wal.as_ref().is_some_and(|wal| wal.record_count() > 0);
            #[cfg(not(feature = "wal"))]
            let has_wal_records = false;

            if flush_result.sections_written > 0 || !has_wal_records {
                {
                    use grafeo_common::testing::crash::maybe_crash;
                    maybe_crash("close:before_remove_sidecar_wal");
                }
                fm.remove_sidecar_wal()?;
            } else {
                grafeo_common::grafeo_warn!(
                    "keeping sidecar WAL for recovery: checkpoint wrote 0 sections but WAL has records"
                );
            }
            fm.close()?;
        }

        *is_open = false;
        Ok(())
    }

    /// Holds `close()` off while a checkpoint, backup or save runs (it waits
    /// for the returned guard), and fails once the database is closed.
    ///
    /// After `close()` the file of a persistent database is released and its
    /// WAL removed: another handle may have opened and written it since, so
    /// nothing is persisted from this one any more (an in-memory database has
    /// no file, and is not refused). Lock order: this guard, then the
    /// checkpoint guard, then the commit lock, as in `close()`.
    ///
    /// # Errors
    ///
    /// Returns [`TransactionError::DatabaseClosed`](grafeo_common::utils::error::TransactionError::DatabaseClosed)
    /// once `close()` of a database with a path has run.
    #[cfg(any(feature = "lpg", feature = "triple-store"))]
    pub(crate) fn hold_open(&self) -> Result<parking_lot::RwLockReadGuard<'_, bool>> {
        let open = self.is_open.read();
        if !*open && self.config.path.is_some() {
            return Err(Error::Transaction(
                grafeo_common::utils::error::TransactionError::DatabaseClosed,
            ));
        }
        Ok(open)
    }

    /// Fails when an import or an RDF batch insert would be refused now, with
    /// the errors of [`hold_commits_for_import`](Self::hold_commits_for_import),
    /// without holding commits off: an import calls this before it reads or
    /// parses its input (or pulls the caller's iterator), so a refused call
    /// does no work first. The answer can change before the import holds
    /// commits off (another thread may close the database or fail a commit
    /// meanwhile): `hold_commits_for_import` checks again, under the hold.
    ///
    /// # Errors
    ///
    /// The database-closed error after `close()` of a persistent database
    /// (read-only or not), the read-only error on a read-only database, and
    /// the incomplete-commit error after a commit that did not complete.
    #[cfg(any(feature = "lpg", feature = "triple-store"))]
    fn check_import_allowed(&self) -> Result<()> {
        // A read-only `close()` sets no closed state for `check_open`: the
        // open state refuses a closed handle of either kind.
        drop(self.hold_open()?);
        if self.read_only {
            return Err(Error::Transaction(
                grafeo_common::utils::error::TransactionError::ReadOnly,
            ));
        }
        self.transaction_manager.check_no_incomplete_commit()?;
        self.transaction_manager.check_open()
    }

    /// Holds commits off while an import or an RDF batch insert changes the
    /// store, for as long as the guard lives (see
    /// [`TransactionManager::hold_commits_for_change`](crate::transaction::TransactionManager)):
    /// it writes its WAL records as it goes, through
    /// [`streaming_changes`](Self::streaming_changes), as one group that
    /// nothing else may write into, and a checkpoint or `close()` waits and
    /// holds all of the change or none of it. Meanwhile commits, new
    /// transactions, writes outside a transaction and checkpoints wait; the
    /// input is parsed before, outside the hold, once
    /// [`check_import_allowed`](Self::check_import_allowed) has passed.
    ///
    /// Nothing would persist the change on a read-only database or after
    /// `close()`, so it fails with the read-only error on a read-only
    /// database and with the database-closed error after `close()` of a
    /// persistent one (read-only or not), as `restore_snapshot` does.
    /// Otherwise it waits for a commit in progress, then fails once a
    /// `close()` has started, and after a commit that did not complete (the
    /// change would be stamped at an epoch that is never published, and no
    /// checkpoint runs any more).
    #[cfg(any(feature = "lpg", feature = "triple-store"))]
    fn hold_commits_for_import(&self) -> Result<crate::transaction::CommitsHeld<'_>> {
        self.check_import_allowed()?;
        // Checks the incomplete-commit and closed states again, under the
        // hold: they may have changed since.
        let held = self.transaction_manager.hold_commits_for_change()?;
        // Tests start a checkpoint or `close()` here, which must wait.
        #[cfg(feature = "testing-statement-injection")]
        grafeo_common::testing::commit_hook::run_during_held_change();
        Ok(held)
    }

    /// The changes of an import or an RDF batch insert that holds commits
    /// off (`held`), written to the WAL as it goes (see
    /// [`StreamingChanges`](crate::transaction::StreamingChanges)).
    #[cfg(any(feature = "lpg", feature = "triple-store"))]
    fn streaming_changes(
        &self,
        held: &crate::transaction::CommitsHeld<'_>,
    ) -> crate::transaction::StreamingChanges {
        let changes = crate::transaction::StreamingChanges::new(
            Arc::clone(&self.transaction_manager),
            self.transaction_manager.bulk_writer(held),
        );
        #[cfg(feature = "wal")]
        let changes = changes.logged_to(self.wal.clone());
        changes
    }

    /// Returns the typed WAL if available.
    #[cfg(feature = "wal")]
    #[must_use]
    pub fn wal(&self) -> Option<&Arc<LpgWal>> {
        self.wal.as_ref()
    }

    /// Holds commits off for a standalone change (a graph command or an
    /// index call, see [`standalone`]) for as long as the guard lives, so a
    /// checkpoint or `close()` sees all of it or none of it. With
    /// `writes_too` (a graph drop) it also waits for the writes of open
    /// transactions in progress and keeps new ones out meanwhile, so what
    /// their change sets hold is what the stores hold.
    ///
    /// # Errors
    ///
    /// The database-closed error after `close()` of a persistent database,
    /// and the incomplete-commit error after a commit that did not complete.
    #[cfg(feature = "lpg")]
    pub(crate) fn hold_for_standalone(
        &self,
        writes_too: bool,
    ) -> Result<crate::transaction::CommitsHeld<'_>> {
        standalone::hold(&self.transaction_manager, writes_too)
    }

    /// Logs `change` as a WAL group of its own and applies it (see
    /// [`standalone::commit`]), holding commits off (`held`).
    ///
    /// # Errors
    ///
    /// As [`standalone::commit`].
    #[cfg(feature = "lpg")]
    pub(crate) fn commit_standalone(
        &self,
        change: crate::transaction::StandaloneChange,
        held: &crate::transaction::CommitsHeld<'_>,
    ) -> Result<()> {
        standalone::commit(
            change,
            held,
            #[cfg(feature = "wal")]
            self.wal.as_deref(),
            &self.lpg_store(),
            &self.catalog,
            &self.transaction_manager,
        )
    }

    /// Registers storage sections as [`MemoryConsumer`]s with the BufferManager.
    ///
    /// Each section reports its memory usage to the buffer manager, enabling
    /// accurate pressure tracking. Called once after database construction.
    fn register_section_consumers(&mut self) {
        // LPG store section
        #[cfg(feature = "lpg")]
        let store_ref = self.store.as_ref();
        #[cfg(feature = "lpg")]
        if let Some(store) = store_ref {
            let lpg = grafeo_core::graph::lpg::LpgStoreSection::new(Arc::clone(store));
            self.buffer_manager.register_consumer(Arc::new(
                section_consumer::SectionConsumer::new(Arc::new(lpg)),
            ));
        }

        // RDF store: only when data exists
        #[cfg(feature = "triple-store")]
        if !self.rdf_store.is_empty() || self.rdf_store.graph_count() > 0 {
            let rdf = grafeo_core::graph::rdf::RdfStoreSection::new(Arc::clone(&self.rdf_store));
            self.buffer_manager.register_consumer(Arc::new(
                section_consumer::SectionConsumer::new(Arc::new(rdf)),
            ));
        }

        // Ring Index: only when Ring has been built
        #[cfg(feature = "ring-index")]
        if self.rdf_store.ring().is_some() {
            let ring = grafeo_core::index::ring::RdfRingSection::new(Arc::clone(&self.rdf_store));
            // Ring is the first index section to opt into spill wiring.
            // The Ring's `swap_to_mmap` still returns `NotSupported` until
            // its packed disk format lands, so a spill attempt today fails
            // cleanly without leaking files; once the format is in place,
            // no engine-side change is required to enable real eviction.
            let consumer = match self.buffer_manager.config().spill_path.clone() {
                Some(path) => section_consumer::SectionConsumer::with_spill(Arc::new(ring), path),
                None => section_consumer::SectionConsumer::new(Arc::new(ring)),
            };
            self.buffer_manager.register_consumer(Arc::new(consumer));
        }

        // Vector indexes: dynamic consumer that re-queries the store on each
        // memory_usage() call, so dropped indexes are freed and new ones tracked.
        #[cfg(all(
            feature = "lpg",
            feature = "vector-index",
            feature = "mmap",
            not(feature = "temporal")
        ))]
        if let Some(store) = store_ref {
            let consumer = Arc::new(section_consumer::VectorIndexConsumer::new(
                store,
                self.vector_cache.clone(),
            ));
            self.buffer_manager.register_consumer(consumer);
        }

        // Text indexes: same dynamic approach as vector indexes.
        #[cfg(all(feature = "lpg", feature = "text-index"))]
        if let Some(store) = store_ref {
            self.buffer_manager
                .register_consumer(Arc::new(section_consumer::TextIndexConsumer::new(store)));
        }

        // CDC log: register as memory consumer so the buffer manager can
        // prune events under memory pressure.
        #[cfg(feature = "cdc")]
        self.buffer_manager.register_consumer(
            Arc::clone(&self.cdc_log) as Arc<dyn grafeo_common::memory::MemoryConsumer>
        );
    }

    /// Applies `TierOverride::ForceDisk` and `TierOverride::ForceRam`
    /// overrides at database open time.
    ///
    /// For each section type in [`Config::section_configs`]:
    ///
    /// - `ForceDisk`: spills the matching registered consumer (named
    ///   `"section:<TypeName>"`) once.
    /// - `ForceRam` (Phase 8g): pins the matching consumer in the buffer
    ///   manager so subsequent spill loops (pressure-driven, explicit, or
    ///   targeted) skip it.
    /// - `Auto`: no action; the BufferManager applies its default policy.
    ///
    /// Must be called after [`Self::register_section_consumers`].
    fn apply_force_disk_overrides(&self) {
        use grafeo_common::storage::TierOverride;

        for (section_type, mem_config) in &self.config.section_configs {
            let consumer_name = format!("section:{section_type:?}");
            match mem_config.tier {
                TierOverride::ForceDisk => {
                    #[cfg(feature = "tracing")]
                    tracing::info!(
                        target: "grafeo::tier",
                        section = ?section_type,
                        tier = "ForceDisk",
                        "applying tier override at db open"
                    );
                    self.buffer_manager.spill_consumer_by_name(&consumer_name);
                }
                TierOverride::ForceRam => {
                    #[cfg(feature = "tracing")]
                    tracing::info!(
                        target: "grafeo::tier",
                        section = ?section_type,
                        tier = "ForceRam",
                        "pinning consumer to RAM"
                    );
                    self.buffer_manager.mark_force_ram(&consumer_name);
                }
                TierOverride::Auto => {}
                _ => {}
            }
        }
    }

    /// Reloads spilled consumers back into RAM up to a target memory fraction.
    ///
    /// Phase 9a: closes the spill / reload loop. After memory pressure drops
    /// (e.g. a workload finishes, or a checkpoint freed mutation overlay
    /// state), call this to bring spilled section data back into RAM for
    /// faster subsequent reads.
    ///
    /// Walks consumers currently reporting `StorageTier::OnDisk`, in priority
    /// order (highest first), reloading each as long as projected memory
    /// usage stays below `target_fraction * memory_limit`.
    ///
    /// Returns the number of consumers successfully reloaded.
    ///
    /// `target_fraction` is clamped to `[0.0, 1.0]`. A typical value is `0.7`
    /// (matching the default `soft_limit_fraction`).
    pub fn reload_eligible(&self, target_fraction: f64) -> usize {
        self.buffer_manager.reload_eligible(target_fraction)
    }

    /// Returns the current [`StorageTier`] of every registered section consumer.
    ///
    /// The map keys are the [`SectionType`]s parsed from each consumer's name
    /// (consumers whose names don't follow the `"section:<TypeName>"` convention
    /// are skipped). Tier classification is best-effort: a consumer reporting
    /// zero `memory_usage()` and `can_spill() == true` is reported as `OnDisk`,
    /// otherwise `InMemory` (or `Uninitialized` if both are zero).
    ///
    /// Useful for tests, observability, and binding-side introspection.
    ///
    /// [`StorageTier`]: grafeo_common::memory::buffer::StorageTier
    /// [`SectionType`]: grafeo_common::storage::SectionType
    #[must_use]
    pub fn storage_tiers(
        &self,
    ) -> hashbrown::HashMap<
        grafeo_common::storage::SectionType,
        grafeo_common::memory::buffer::StorageTier,
    > {
        use grafeo_common::storage::SectionType;
        let snapshot = self.buffer_manager.snapshot_consumer_tiers();
        let mut out = hashbrown::HashMap::new();
        for (name, tier) in snapshot {
            let Some(suffix) = name.strip_prefix("section:") else {
                continue;
            };
            let section_type = match suffix {
                "LpgStore" => SectionType::LpgStore,
                "RdfStore" => SectionType::RdfStore,
                "CompactStore" => SectionType::CompactStore,
                "VectorStore" => SectionType::VectorStore,
                "TextIndex" => SectionType::TextIndex,
                "RdfRing" => SectionType::RdfRing,
                "PropertyIndex" => SectionType::PropertyIndex,
                "Catalog" => SectionType::Catalog,
                _ => continue,
            };
            out.insert(section_type, tier);
        }
        out
    }

    /// What a checkpoint writes: the complete current state (see
    /// [`sections::CheckpointSources`]).
    fn checkpoint_sources(&self) -> sections::CheckpointSources {
        sections::CheckpointSources {
            #[cfg(feature = "lpg")]
            store: self.store.clone(),
            #[cfg(feature = "lpg")]
            catalog: Arc::clone(&self.catalog),
            transaction_manager: Arc::clone(&self.transaction_manager),
            #[cfg(feature = "triple-store")]
            rdf_store: Arc::clone(&self.rdf_store),
        }
    }

    /// Starts the periodic checkpoint timer when one is configured, replacing
    /// a running one.
    #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
    fn start_checkpoint_timer(&self) {
        self.stop_checkpoint_timer();
        // A file opened read-only stays so; a closed database has released
        // its file.
        if let (Some(interval), Some(fm)) = (self.config.checkpoint_interval, &self.file_manager)
            && !self.read_only
            && !fm.is_read_only()
            && *self.is_open.read()
        {
            *self.checkpoint_timer.lock() = Some(checkpoint_timer::CheckpointTimer::start(
                interval,
                Arc::clone(fm),
                self.checkpoint_sources(),
                #[cfg(feature = "wal")]
                self.wal.clone(),
            ));
        }
    }

    /// Stops the periodic checkpoint timer, if one is running.
    #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
    fn stop_checkpoint_timer(&self) {
        let running = self.checkpoint_timer.lock().take();
        if let Some(mut timer) = running {
            timer.stop();
        }
    }

    // =========================================================================
    // Backup API
    // =========================================================================

    /// Creates a full backup of the database in the given directory.
    ///
    /// Checkpoints the database, copies the `.grafeo` file, and creates a
    /// backup manifest. Subsequent incremental backups will use this as the
    /// base.
    ///
    /// # Errors
    ///
    /// Returns an error if the database has no file manager or I/O fails,
    /// and the database-closed error after `close()`.
    #[cfg(all(feature = "wal", feature = "grafeo-file", feature = "lpg"))]
    pub fn backup_full(&self, backup_dir: &std::path::Path) -> Result<backup::BackupSegment> {
        let _open = self.hold_open()?;
        let fm = self.file_manager.as_ref().ok_or_else(|| {
            Error::Query(grafeo_common::utils::error::QueryError::unsupported(
                "a backup needs a persistent database",
            ))
        })?;

        // Checkpoint to ensure the container has the latest data.
        // Skip for read-only databases: the on-disk file is already a valid
        // snapshot and the file manager rejects writes.
        if !self.read_only {
            let _ = self.checkpoint_to_file(fm)?;
        }

        backup::do_backup_full(backup_dir, fm, self.wal.as_deref())
    }

    /// Creates an incremental backup containing WAL records since the last backup.
    ///
    /// Requires a prior full backup in the backup directory.
    ///
    /// # Errors
    ///
    /// Returns an error if no full backup exists, or if the WAL has no new
    /// records, and the database-closed error after `close()`.
    #[cfg(all(feature = "wal", feature = "grafeo-file", feature = "lpg"))]
    pub fn backup_incremental(
        &self,
        backup_dir: &std::path::Path,
    ) -> Result<backup::BackupSegment> {
        let _open = self.hold_open()?;
        let wal = self.wal.as_ref().ok_or_else(|| {
            Error::Query(grafeo_common::utils::error::QueryError::unsupported(
                "an incremental backup needs a database with a WAL",
            ))
        })?;

        let current_epoch = self.transaction_manager.current_epoch();
        backup::do_backup_incremental(backup_dir, wal, current_epoch)
    }

    /// Returns the backup manifest for a backup directory, if one exists.
    ///
    /// # Errors
    ///
    /// Returns an error if the manifest file exists but cannot be parsed.
    #[cfg(all(feature = "wal", feature = "grafeo-file"))]
    pub fn read_backup_manifest(
        backup_dir: &std::path::Path,
    ) -> Result<Option<backup::BackupManifest>> {
        backup::read_manifest(backup_dir)
    }

    /// Returns the current backup cursor (last backed-up position), if any.
    #[cfg(all(feature = "wal", feature = "grafeo-file"))]
    #[must_use]
    pub fn backup_cursor(&self) -> Option<backup::BackupCursor> {
        self.wal
            .as_ref()
            .and_then(|wal| backup::read_backup_cursor(wal.dir()).ok().flatten())
    }

    /// Restores a database from a backup chain to a specific epoch.
    ///
    /// Copies the full backup to `output_path`, then replays incremental
    /// WAL segments up to `target_epoch`. The restored database can be
    /// opened with [`GrafeoDB::open`].
    ///
    /// The backups of an encrypted database are encrypted: a restore that
    /// needs only the full backup copies its encrypted file (open it with its
    /// key), and one that replays incremental segments needs the key, see
    /// `restore_to_epoch_with` (feature `encryption`).
    ///
    /// # Errors
    ///
    /// Returns an error if `output_path` or its `<output_path>.wal` sidecar
    /// already exists (restores never overwrite a database), if the backup
    /// chain does not cover the target epoch, segment checksums fail, or I/O
    /// fails; also, before anything is written, if the backup is encrypted
    /// and segments are to be replayed.
    #[cfg(all(feature = "wal", feature = "grafeo-file"))]
    pub fn restore_to_epoch(
        backup_dir: &std::path::Path,
        target_epoch: grafeo_common::types::EpochId,
        output_path: &std::path::Path,
    ) -> Result<()> {
        backup::do_restore_to_epoch(
            backup_dir,
            target_epoch,
            output_path,
            &encryption::DatabaseKeys::none(),
        )
    }

    /// Restores an encrypted database from a backup chain to a specific
    /// epoch, as [`restore_to_epoch`](Self::restore_to_epoch) does, with the
    /// key chain the database was encrypted with.
    ///
    /// The key is checked against the full backup before anything is
    /// written. The incremental segments are decrypted with the WAL key of
    /// the backup's database, and the WAL the restore writes next to the
    /// restored file is encrypted with that same key: open the restored
    /// database with the same `encryption`.
    ///
    /// # Errors
    ///
    /// The errors of [`restore_to_epoch`](Self::restore_to_epoch); also, before
    /// anything is written, if the backup was encrypted with another key or is
    /// not encrypted.
    #[cfg(all(feature = "wal", feature = "grafeo-file", feature = "encryption"))]
    pub fn restore_to_epoch_with(
        backup_dir: &std::path::Path,
        target_epoch: grafeo_common::types::EpochId,
        output_path: &std::path::Path,
        encryption: &crate::config::EncryptionConfig,
    ) -> Result<()> {
        backup::do_restore_to_epoch(
            backup_dir,
            target_epoch,
            output_path,
            &encryption::DatabaseKeys::from_encryption(encryption),
        )
    }

    /// Writes the current database state to the `.grafeo` file using the unified flush.
    ///
    /// Does NOT remove the sidecar WAL: callers that want to clean up
    /// the sidecar (e.g. `close()`) should call `fm.remove_sidecar_wal()`
    /// separately after this returns.
    #[cfg(feature = "grafeo-file")]
    fn checkpoint_to_file(&self, fm: &GrafeoFileManager) -> Result<flush::FlushResult> {
        flush::flush(
            fm,
            &self.checkpoint_sources(),
            #[cfg(feature = "wal")]
            self.wal.as_deref(),
        )
    }

    /// Returns the file manager if using single-file format.
    #[cfg(feature = "grafeo-file")]
    #[must_use]
    pub fn file_manager(&self) -> Option<&Arc<GrafeoFileManager>> {
        self.file_manager.as_ref()
    }
}

/// The spelling of a database path that decisions and side-file names use:
/// `path` without trailing separators and inner `.` components (`db/` and
/// `data/./db` name the same databases as `db` and `data/db`). A path that
/// ends in `.` or `..` (or is a root) is resolved to the directory it names:
/// side-file names (`<path>.wal`, `<path>.pre-0.6`, ...) are appended to the
/// path, and `..pre-0.6` would name a file inside the directory `.` names.
/// Its longest existing prefix is resolved through the file system (so
/// `link/..` is the parent of the symlink's target, as the file system reads
/// it, also in `link/../new/..`), and only the components past that prefix,
/// which do not exist, lexically.
///
/// # Errors
///
/// Returns an error if such a path cannot be made absolute (the current
/// directory is unknown, or the path is empty).
pub(crate) fn normalize_path(path: &std::path::Path) -> Result<std::path::PathBuf> {
    use std::path::Component;

    if matches!(path.components().next_back(), Some(Component::Normal(_))) {
        return Ok(path.components().collect());
    }
    let absolute = std::path::absolute(path).map_err(|error| {
        Error::InvalidValue(format!(
            "cannot resolve the database path {:?}: {error}",
            path.display()
        ))
    })?;
    Ok(resolve_through_longest_existing_prefix(
        &absolute,
        |prefix| std::fs::canonicalize(prefix),
    ))
}

/// `absolute` with its longest prefix that `canonicalize` resolves replaced
/// by that resolution, and the components past it resolved lexically (`.`
/// dropped, `..` removing the component before it). A component past that
/// prefix does not exist, so the lexical reading is the only one there is.
fn resolve_through_longest_existing_prefix(
    absolute: &std::path::Path,
    canonicalize: impl Fn(&std::path::Path) -> std::io::Result<std::path::PathBuf>,
) -> std::path::PathBuf {
    use std::path::{Component, PathBuf};

    let components: Vec<Component<'_>> = absolute.components().collect();
    let (mut resolved, rest) = (1..=components.len())
        .rev()
        .find_map(|split| {
            let prefix: PathBuf = components[..split].iter().collect();
            canonicalize(&prefix)
                .ok()
                .map(|resolved| (resolved, &components[split..]))
        })
        .unwrap_or((PathBuf::new(), &components[..]));
    for component in rest {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                resolved.pop();
            }
            other => resolved.push(other),
        }
    }
    resolved
}

/// The error of a build without the `lpg` feature for a database written by
/// 0.5.x (a file or a WAL directory): it can neither read it (the 0.5.x
/// reader loads into the LPG store) nor migrate it.
#[cfg(all(feature = "grafeo-file", not(feature = "lpg")))]
fn legacy_database_needs_lpg(path: &std::path::Path) -> Error {
    Error::InvalidValue(format!(
        "{} was written by Grafeo 0.5.x, and this build cannot read or migrate it (it lacks \
         the `lpg` feature): open it read-write once with a build that has the `lpg` feature, \
         which migrates it to the 0.6 file format",
        path.display()
    ))
}

/// The error of an open with the deprecated `StorageFormat::WalDirectory` at
/// a path where nothing exists: WAL directories are no longer created.
#[cfg(feature = "grafeo-file")]
fn wal_directories_are_no_longer_created(path: &std::path::Path) -> Error {
    Error::InvalidValue(format!(
        "cannot create a database at {}: WAL directories are no longer created (since 0.6 a \
         database is a single file); use `StorageFormat::Auto`, which creates a single file at \
         any path",
        path.display()
    ))
}

/// The error of a configuration [`Config::validate`] refuses: a setting the
/// caller gave that does not fit is invalid input, and a graph model this
/// build cannot run is unsupported.
fn config_error(error: crate::config::ConfigError) -> Error {
    match error {
        crate::config::ConfigError::RdfFeatureRequired => Error::Query(
            grafeo_common::utils::error::QueryError::unsupported(error.to_string()),
        ),
        other => Error::InvalidValue(other.to_string()),
    }
}

/// The error of an open of a path that holds neither a database file nor a
/// 0.5.x WAL directory: a directory without `wal/` inside, or something that
/// is neither a file nor a directory.
#[cfg(feature = "grafeo-file")]
fn not_a_database(path: &std::path::Path) -> Error {
    Error::InvalidValue(format!(
        "cannot open {}: it is not a database file, and not a 0.5.x WAL-directory database (a \
         directory holding `wal/`); nothing was changed",
        path.display()
    ))
}

impl Drop for GrafeoDB {
    fn drop(&mut self) {
        if let Err(e) = self.close() {
            grafeo_error!("Error closing database: {}", e);
        }
    }
}

#[cfg(feature = "lpg")]
impl crate::admin::AdminService for GrafeoDB {
    fn info(&self) -> crate::admin::DatabaseInfo {
        self.info()
    }

    fn detailed_stats(&self) -> crate::admin::DatabaseStats {
        self.detailed_stats()
    }

    fn schema(&self) -> crate::admin::SchemaInfo {
        self.schema()
    }

    fn validate(&self) -> crate::admin::ValidationResult {
        self.validate()
    }

    fn wal_status(&self) -> crate::admin::WalStatus {
        self.wal_status()
    }

    fn wal_checkpoint(&self) -> Result<()> {
        self.wal_checkpoint()
    }
}

// =========================================================================
// Query Result Types
// =========================================================================

pub use grafeo_core::execution::operators::WriteCounters;

/// The result of running a query.
///
/// Contains rows and columns, like a table. Use [`iter()`](Self::iter) to
/// loop through rows, or [`scalar()`](Self::scalar) if you expect a single value.
///
/// # Examples
///
/// ```
/// use grafeo_engine::GrafeoDB;
///
/// let db = GrafeoDB::new_in_memory();
/// db.create_node(&["Person"]);
///
/// let result = db.execute("MATCH (p:Person) RETURN count(p) AS total")?;
///
/// // Check what we got
/// println!("Columns: {:?}", result.columns);
/// println!("Rows: {}", result.row_count());
///
/// // Iterate through results
/// for row in result.iter() {
///     println!("{:?}", row);
/// }
/// # Ok::<(), grafeo_common::utils::error::Error>(())
/// ```
#[derive(Debug)]
pub struct QueryResult {
    /// Column names from the RETURN clause.
    pub columns: Vec<String>,
    /// Column types - useful for distinguishing NodeId/EdgeId from plain integers.
    pub column_types: Vec<grafeo_common::types::LogicalType>,
    /// The actual result rows.
    ///
    /// Use [`rows()`](Self::rows) for borrowed access or
    /// [`into_rows()`](Self::into_rows) to take ownership.
    pub(crate) rows: Vec<Vec<grafeo_common::types::Value>>,
    /// Query execution time in milliseconds (if timing was enabled).
    pub execution_time_ms: Option<f64>,
    /// Number of rows scanned during query execution (estimate).
    pub rows_scanned: Option<u64>,
    /// Status message for DDL and session commands (e.g., "Created node type 'Person'").
    pub status_message: Option<String>,
    /// GQLSTATUS code per ISO/IEC 39075:2024, sec 23: `00001` (omitted
    /// result) for a statement without a result (a write without `RETURN`,
    /// `FINISH`, a schema or transaction command), `00000` for one with a
    /// result, also when it has no rows.
    pub gql_status: grafeo_common::utils::GqlStatus,
    /// What the statement's writes changed: nodes and edges created and
    /// deleted, properties set, labels added and removed.
    pub counters: WriteCounters,
}

impl QueryResult {
    /// Checks that no column name repeats (#371). Bindings read rows by column
    /// name, so a repeated name would silently drop values. Used by every
    /// constructor that takes columns and by the streaming open path.
    ///
    /// # Errors
    ///
    /// Returns [`QueryErrorKind::Semantic`] if any column name appears more than
    /// once in `columns`.
    pub(crate) fn validate_unique_columns(columns: &[String]) -> Result<()> {
        // Column lists are short; report the first duplicate in column order.
        for (idx, name) in columns.iter().enumerate() {
            if columns[..idx].contains(name) {
                return Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    format!(
                        "duplicate column name '{name}' in query result: a result cannot carry \
                         two columns with the same name because name-addressed consumers would \
                         silently drop one; give the projections distinct aliases (e.g. `AS {name}_2`)"
                    ),
                )));
            }
        }
        Ok(())
    }

    /// The GQLSTATUS of a successful statement whose result has `columns`:
    /// none means the statement has no result (an omitted result, `00001`),
    /// as a write without `RETURN` and `FINISH` plan it.
    fn success_status(columns: &[String]) -> grafeo_common::utils::GqlStatus {
        if columns.is_empty() {
            grafeo_common::utils::GqlStatus::SUCCESS_OMITTED_RESULT
        } else {
            grafeo_common::utils::GqlStatus::SUCCESS
        }
    }

    /// Creates a fully empty query result (no columns, no rows).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            column_types: Vec::new(),
            rows: Vec::new(),
            execution_time_ms: None,
            rows_scanned: None,
            status_message: None,
            gql_status: grafeo_common::utils::GqlStatus::SUCCESS,
            counters: Default::default(),
        }
    }

    /// Creates a query result with only a status message (for DDL commands),
    /// an omitted result (GQLSTATUS `00001`).
    #[must_use]
    pub fn status(msg: impl Into<String>) -> Self {
        Self {
            columns: Vec::new(),
            column_types: Vec::new(),
            rows: Vec::new(),
            execution_time_ms: None,
            rows_scanned: None,
            status_message: Some(msg.into()),
            gql_status: grafeo_common::utils::GqlStatus::SUCCESS_OMITTED_RESULT,
            counters: Default::default(),
        }
    }

    /// Creates a new empty query result.
    ///
    /// # Errors
    ///
    /// Returns a semantic error if `columns` contains a repeated column name.
    pub fn new(columns: Vec<String>) -> Result<Self> {
        Self::validate_unique_columns(&columns)?;
        let gql_status = Self::success_status(&columns);
        let len = columns.len();
        Ok(Self {
            columns,
            column_types: vec![grafeo_common::types::LogicalType::Any; len],
            rows: Vec::new(),
            execution_time_ms: None,
            rows_scanned: None,
            status_message: None,
            gql_status,
            counters: Default::default(),
        })
    }

    /// Creates a new empty query result with column types.
    ///
    /// # Errors
    ///
    /// Returns a semantic error if `columns` contains a repeated column name.
    pub fn with_types(
        columns: Vec<String>,
        column_types: Vec<grafeo_common::types::LogicalType>,
    ) -> Result<Self> {
        Self::validate_unique_columns(&columns)?;
        let gql_status = Self::success_status(&columns);
        Ok(Self {
            columns,
            column_types,
            rows: Vec::new(),
            execution_time_ms: None,
            rows_scanned: None,
            status_message: None,
            gql_status,
            counters: Default::default(),
        })
    }

    /// Creates a query result with pre-populated rows.
    ///
    /// # Errors
    ///
    /// Returns a semantic error if `columns` contains a repeated column name.
    pub fn from_rows(
        columns: Vec<String>,
        rows: Vec<Vec<grafeo_common::types::Value>>,
    ) -> Result<Self> {
        Self::validate_unique_columns(&columns)?;
        let gql_status = Self::success_status(&columns);
        let len = columns.len();
        Ok(Self {
            columns,
            column_types: vec![grafeo_common::types::LogicalType::Any; len],
            rows,
            execution_time_ms: None,
            rows_scanned: None,
            status_message: None,
            gql_status,
            counters: Default::default(),
        })
    }

    /// Appends a row to this result.
    pub fn push_row(&mut self, row: Vec<grafeo_common::types::Value>) {
        self.rows.push(row);
    }

    /// Sets the execution metrics on this result.
    pub fn with_metrics(mut self, execution_time_ms: f64, rows_scanned: u64) -> Self {
        self.execution_time_ms = Some(execution_time_ms);
        self.rows_scanned = Some(rows_scanned);
        self
    }

    /// Returns the execution time in milliseconds, if available.
    #[must_use]
    pub fn execution_time_ms(&self) -> Option<f64> {
        self.execution_time_ms
    }

    /// Returns the number of rows scanned, if available.
    #[must_use]
    pub fn rows_scanned(&self) -> Option<u64> {
        self.rows_scanned
    }

    /// Returns the number of rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Returns the number of columns.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Returns true if the result is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Extracts a single value from the result.
    ///
    /// Use this when your query returns exactly one row with one column,
    /// like `RETURN count(n)` or `RETURN sum(p.amount)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the result has multiple rows or columns.
    pub fn scalar<T: FromValue>(&self) -> Result<T> {
        if self.rows.len() != 1 || self.columns.len() != 1 {
            return Err(grafeo_common::utils::error::Error::InvalidValue(
                "Expected single value".to_string(),
            ));
        }
        T::from_value(&self.rows[0][0])
    }

    /// Returns a slice of all result rows.
    #[must_use]
    pub fn rows(&self) -> &[Vec<grafeo_common::types::Value>] {
        &self.rows
    }

    /// Takes ownership of all result rows.
    #[must_use]
    pub fn into_rows(self) -> Vec<Vec<grafeo_common::types::Value>> {
        self.rows
    }

    /// Returns an iterator over the rows.
    pub fn iter(&self) -> impl Iterator<Item = &Vec<grafeo_common::types::Value>> {
        self.rows.iter()
    }

    /// Converts this query result to an Arrow [`RecordBatch`](arrow_array::RecordBatch).
    ///
    /// Each column in the result becomes an Arrow array. Type mapping:
    /// - `Int64` / `Float64` / `Bool` / `String` / `Bytes`: direct Arrow equivalents
    /// - `Timestamp` / `ZonedDatetime`: `Timestamp(Microsecond, UTC)`
    /// - `Date`: `Date32`, `Time`: `Time64(Nanosecond)`
    /// - `Vector`: `FixedSizeList(Float32, dim)`
    /// - `Duration` / `List` / `Map` / `Path`: serialized as `Utf8`
    ///
    /// Heterogeneous columns (mixed types) fall back to `Utf8`.
    ///
    /// # Errors
    ///
    /// Returns [`ArrowExportError`](arrow::ArrowExportError) if Arrow array construction fails.
    #[cfg(feature = "arrow-export")]
    pub fn to_record_batch(
        &self,
    ) -> std::result::Result<arrow_array::RecordBatch, arrow::ArrowExportError> {
        arrow::query_result_to_record_batch(&self.columns, &self.column_types, &self.rows)
    }

    /// Serializes this query result as Arrow IPC stream bytes.
    ///
    /// The returned bytes can be read by any Arrow implementation:
    /// - Python: `pyarrow.ipc.open_stream(buf).read_all()`
    /// - Polars: `pl.read_ipc(buf)`
    /// - Node.js: `apache-arrow` `RecordBatchStreamReader`
    ///
    /// # Errors
    ///
    /// Returns [`ArrowExportError`](arrow::ArrowExportError) on conversion or serialization failure.
    #[cfg(feature = "arrow-export")]
    pub fn to_arrow_ipc(&self) -> std::result::Result<Vec<u8>, arrow::ArrowExportError> {
        let batch = self.to_record_batch()?;
        arrow::record_batch_to_ipc_stream(&batch)
    }
}

impl std::fmt::Display for QueryResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let table = grafeo_common::fmt::format_result_table(
            &self.columns,
            &self.rows,
            self.execution_time_ms,
            self.status_message.as_deref(),
        );
        f.write_str(&table)
    }
}

/// Converts a [`grafeo_common::types::Value`] to a concrete Rust type.
///
/// Implemented for common types like `i64`, `f64`, `String`, and `bool`.
/// Used by [`QueryResult::scalar()`] to extract typed values.
pub trait FromValue: Sized {
    /// Attempts the conversion, returning an error on type mismatch.
    ///
    /// # Errors
    ///
    /// Returns `Error::TypeMismatch` if the value is not the expected type.
    fn from_value(value: &grafeo_common::types::Value) -> Result<Self>;
}

impl FromValue for i64 {
    fn from_value(value: &grafeo_common::types::Value) -> Result<Self> {
        value
            .as_int64()
            .ok_or_else(|| grafeo_common::utils::error::Error::TypeMismatch {
                expected: "INT64".to_string(),
                found: value.type_name().to_string(),
            })
    }
}

impl FromValue for f64 {
    fn from_value(value: &grafeo_common::types::Value) -> Result<Self> {
        value
            .as_float64()
            .ok_or_else(|| grafeo_common::utils::error::Error::TypeMismatch {
                expected: "FLOAT64".to_string(),
                found: value.type_name().to_string(),
            })
    }
}

impl FromValue for String {
    fn from_value(value: &grafeo_common::types::Value) -> Result<Self> {
        value.as_str().map(String::from).ok_or_else(|| {
            grafeo_common::utils::error::Error::TypeMismatch {
                expected: "STRING".to_string(),
                found: value.type_name().to_string(),
            }
        })
    }
}

impl FromValue for bool {
    fn from_value(value: &grafeo_common::types::Value) -> Result<Self> {
        value
            .as_bool()
            .ok_or_else(|| grafeo_common::utils::error::Error::TypeMismatch {
                expected: "BOOL".to_string(),
                found: value.type_name().to_string(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 0.5.x loader passes the vector and text sections of a 0.5.44 file
    /// on: their indexes come back from them, before anything is built from
    /// the data.
    #[cfg(all(
        feature = "grafeo-file",
        feature = "lpg",
        feature = "vector-index",
        feature = "text-index"
    ))]
    #[test]
    fn a_0_5_file_restores_its_search_indexes_from_their_sections() {
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/released/0.5.44/closed.grafeo");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("closed.grafeo");
        std::fs::copy(&fixture, &path).unwrap();
        let store = Arc::new(LpgStore::new().unwrap());
        let catalog = Arc::new(Catalog::new());
        #[cfg(feature = "triple-store")]
        let rdf_store = Arc::new(RdfStore::new());
        let mut loaded = sections::LoadedSections::default();
        GrafeoDB::load_legacy_file(
            &path,
            &store,
            &catalog,
            #[cfg(feature = "triple-store")]
            &rdf_store,
            &mut loaded,
        )
        .unwrap();
        assert!(
            store.get_vector_index("Document", "embedding").is_some(),
            "the vector index comes from its section"
        );
        assert!(
            store.get_text_index("Document", "content").is_some(),
            "the text index comes from its section"
        );
    }

    /// `DurabilityMode::Adaptive` syncs the WAL from a background flusher.
    /// None was started, so an adaptive database never synced its WAL.
    #[cfg(all(feature = "wal", feature = "grafeo-file"))]
    #[test]
    fn adaptive_durability_syncs_the_wal_in_the_background() {
        let dir = tempfile::tempdir().unwrap();
        let db = GrafeoDB::with_config(
            Config::persistent(dir.path().join("db.grafeo"))
                .with_storage_format(crate::config::StorageFormat::Auto)
                .with_wal_durability(crate::config::DurabilityMode::Adaptive {
                    target_interval_ms: 10,
                }),
        )
        .unwrap();
        db.create_node(&["Person"]).unwrap();
        // Waits for a flush, with a deadline long enough for a loaded
        // machine or an instrumented build (one fixed short sleep made this
        // test fail under load).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while db
            .wal_flusher
            .lock()
            .as_ref()
            .expect("a flusher runs")
            .stats()
            .flush_count
            == 0
            && std::time::Instant::now() < deadline
        {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let mut flusher = db.wal_flusher.lock().take().expect("a flusher runs");
        let stats = flusher.shutdown().unwrap();
        assert!(stats.flush_count > 0, "{stats:?}");
        db.close().unwrap();
    }

    /// `compact()` reports what it did: an in-memory database writes no
    /// checkpoint, a persistent one does; a read-only one writes none.
    #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
    #[test]
    fn compact_reports_whether_it_checkpointed() {
        let mut memory = GrafeoDB::new_in_memory();
        memory.create_node(&["Person"]).unwrap();
        let report = memory.compact().unwrap();
        assert!(!report.checkpointed, "no file to write: {report:?}");

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        let mut db = GrafeoDB::open(&path).unwrap();
        db.create_node(&["Person"]).unwrap();
        let report = db.compact().unwrap();
        assert!(report.checkpointed, "the file is written: {report:?}");
        db.close().unwrap();

        let mut read_only = GrafeoDB::open_read_only(&path).unwrap();
        let report = read_only.compact().unwrap();
        assert!(
            !report.checkpointed,
            "a read-only file is left alone: {report:?}"
        );
    }

    /// `compact()` drops the old versions of every graph, the named ones
    /// too, and a second `compact()` finds none left. Only builds that keep
    /// property history hold old versions.
    #[cfg(all(feature = "temporal", feature = "lpg"))]
    #[test]
    fn compact_collects_old_versions_in_every_graph() {
        use grafeo_common::types::Value;

        let mut db = GrafeoDB::new_in_memory();
        let alix = db.create_node(&["Person"]).unwrap();
        db.create_graph("trips").unwrap();
        let trips = db.graph("trips").unwrap();
        trips.execute("INSERT (:City {name: 'Paris'})").unwrap();
        for age in [30, 31, 32] {
            db.set_node_property(alix, "age", Value::Int64(age))
                .unwrap();
        }
        for name in ["Prague", "Berlin"] {
            trips
                .execute(&format!("MATCH (c:City) SET c.name = '{name}'"))
                .unwrap();
        }
        let after_default_only = {
            // The default graph alone, as `gc()` collected before 0.6.0.
            let store = db.lpg_store();
            store.gc_versions(db.transaction_manager.min_active_epoch())
        };
        assert!(after_default_only > 0, "Alix's older ages are collectable");

        let report = db.compact().unwrap();
        assert!(
            report.versions_collected > 0,
            "the named graph's older city names are collected too: {report:?}"
        );
        assert_eq!(
            db.compact().unwrap().versions_collected,
            0,
            "nothing is left to collect"
        );
        assert_eq!(
            db.get_node(alix).unwrap().properties.get(&"age".into()),
            Some(&Value::Int64(32)),
            "the current value stays"
        );
    }

    /// `recompact()` is a deprecated alias of `compact()`.
    #[test]
    #[expect(deprecated, reason = "the deprecated alias is what this tests")]
    fn recompact_is_compact() {
        let mut db = GrafeoDB::new_in_memory();
        let report = db.recompact().unwrap();
        assert!(!report.checkpointed);
        assert_eq!(report.versions_collected, 0);
    }

    /// `compact()` of a database opened read-only changes nothing, and no
    /// checkpoint timer may run against its file.
    #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
    #[test]
    fn compacting_a_read_only_database_starts_no_checkpoint_timer() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        {
            let db = GrafeoDB::open(&path).unwrap();
            db.create_node(&["Person"]).unwrap();
            db.close().unwrap();
        }
        let mut db = GrafeoDB::with_config(
            Config::read_only(&path).with_checkpoint_interval(std::time::Duration::from_secs(60)),
        )
        .unwrap();
        db.compact().unwrap();
        assert!(db.checkpoint_timer.lock().is_none());
    }

    /// `compact()` after `close()` fails and starts no checkpoint timer: its
    /// checkpoints would write a file the database no longer holds.
    #[cfg(all(feature = "wal", feature = "lpg"))]
    #[test]
    fn compacting_a_closed_database_starts_no_checkpoint_timer() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = GrafeoDB::with_config(
            Config::persistent(dir.path().join("db.grafeo"))
                .with_checkpoint_interval(std::time::Duration::from_secs(60)),
        )
        .unwrap();
        assert!(
            db.checkpoint_timer.lock().is_some(),
            "an open database runs its timer"
        );
        db.close().unwrap();

        let error = db.compact().expect_err("compact() after close() fails");
        assert!(
            matches!(
                error,
                Error::Transaction(grafeo_common::utils::error::TransactionError::DatabaseClosed)
            ),
            "{error:?}"
        );
        assert!(db.checkpoint_timer.lock().is_none(), "no timer was started");
    }

    #[test]
    fn test_create_in_memory_database() {
        let db = GrafeoDB::new_in_memory();
        assert_eq!(db.node_count(), 0);
        assert_eq!(db.edge_count(), 0);
    }

    #[test]
    fn test_database_config() {
        let config = Config::in_memory().with_threads(4).with_query_logging();

        let db = GrafeoDB::with_config(config).unwrap();
        assert_eq!(db.config().threads, 4);
        assert!(db.config().query_logging);
    }

    #[test]
    fn test_database_session() {
        let db = GrafeoDB::new_in_memory();
        let _session = db.session();
        // Session should be created successfully
    }

    #[cfg(all(feature = "wal", feature = "grafeo-file"))]
    #[test]
    fn test_persistent_database_recovery() {
        use grafeo_common::types::Value;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db.grafeo");

        // Create database and add some data
        {
            let db = GrafeoDB::open(&db_path).unwrap();

            let alix = db.create_node(&["Person"]).unwrap();
            db.set_node_property(alix, "name", Value::from("Alix"))
                .unwrap();

            let gus = db.create_node(&["Person"]).unwrap();
            db.set_node_property(gus, "name", Value::from("Gus"))
                .unwrap();

            let _edge = db.create_edge(alix, gus, "KNOWS").unwrap();

            // close() checkpoints: the file holds everything written
            db.close().unwrap();
        }

        // Reopen and verify the data was read back from the file
        {
            let db = GrafeoDB::open(&db_path).unwrap();

            assert_eq!(db.node_count(), 2);
            assert_eq!(db.edge_count(), 1);

            // Verify nodes exist
            let node0 = db.get_node(grafeo_common::types::NodeId::new(0));
            assert!(node0.is_some());

            let node1 = db.get_node(grafeo_common::types::NodeId::new(1));
            assert!(node1.is_some());
        }
    }

    #[cfg(all(feature = "wal", feature = "grafeo-file"))]
    #[test]
    fn test_wal_logging() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("wal_test_db.grafeo");

        let db = GrafeoDB::open(&db_path).unwrap();

        // Create some data
        let node = db.create_node(&["Test"]).unwrap();
        db.delete_node(node).unwrap();

        // WAL should have records
        if let Some(wal) = db.wal() {
            assert!(wal.record_count() > 0);
        }

        db.close().unwrap();
    }

    #[cfg(all(feature = "wal", feature = "grafeo-file"))]
    #[test]
    fn test_wal_recovery_multiple_sessions() {
        // Tests that WAL recovery works correctly across multiple open/crash
        // cycles: sessions 1 and 2 run in child processes that exit without
        // close(), so nothing is checkpointed and each open replays the WAL.
        use grafeo_common::types::Value;
        use tempfile::tempdir;

        const SESSION_VAR: &str = "GRAFEO_MULTI_SESSION_TEST_SESSION";
        const PATH_VAR: &str = "GRAFEO_MULTI_SESSION_TEST_PATH";

        if let (Ok(session), Some(path)) = (std::env::var(SESSION_VAR), std::env::var_os(PATH_VAR))
        {
            let db = GrafeoDB::open(&path).unwrap();
            match session.as_str() {
                // Session 1: Create initial data
                "1" => {
                    let alix = db.create_node(&["Person"]).unwrap();
                    db.set_node_property(alix, "name", Value::from("Alix"))
                        .unwrap();
                }
                // Session 2: Add more data
                "2" => {
                    assert_eq!(db.node_count(), 1); // Previous data recovered
                    let gus = db.create_node(&["Person"]).unwrap();
                    db.set_node_property(gus, "name", Value::from("Gus"))
                        .unwrap();
                }
                other => panic!("unknown session {other}"),
            }
            // Crash: no close(), no checkpoint, no destructors.
            std::process::exit(0);
        }

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("multi_session_db.grafeo");
        for session in ["1", "2"] {
            let status = grafeo_common::testing::child_process::run(
                std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "database::tests::test_wal_recovery_multiple_sessions",
                        "--nocapture",
                    ])
                    .env(SESSION_VAR, session)
                    .env(PATH_VAR, &db_path),
            )
            .unwrap();
            assert!(status.success(), "session {session} failed");
        }

        // Session 3: Verify all data
        {
            let db = GrafeoDB::open(&db_path).unwrap();
            assert_eq!(db.node_count(), 2);

            // Verify properties were recovered correctly
            let node0 = db.get_node(grafeo_common::types::NodeId::new(0)).unwrap();
            assert!(node0.labels.iter().any(|l| l.as_str() == "Person"));

            let node1 = db.get_node(grafeo_common::types::NodeId::new(1)).unwrap();
            assert!(node1.labels.iter().any(|l| l.as_str() == "Person"));
        }
    }

    #[cfg(all(feature = "wal", feature = "grafeo-file"))]
    #[test]
    fn test_database_consistency_after_mutations() {
        // Tests that database remains consistent after a series of create/delete operations
        use grafeo_common::types::Value;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("consistency_db.grafeo");

        {
            let db = GrafeoDB::open(&db_path).unwrap();

            // Create nodes
            let a = db.create_node(&["Node"]).unwrap();
            let b = db.create_node(&["Node"]).unwrap();
            let c = db.create_node(&["Node"]).unwrap();

            // Create edges
            let e1 = db.create_edge(a, b, "LINKS").unwrap();
            let e2 = db.create_edge(b, c, "LINKS").unwrap();

            // Delete the middle node's edges, then the node
            db.delete_edge(e1).unwrap();
            db.delete_edge(e2).unwrap();
            db.delete_node(b).unwrap();

            // Set properties on remaining nodes
            db.set_node_property(a, "value", Value::Int64(1)).unwrap();
            db.set_node_property(c, "value", Value::Int64(3)).unwrap();

            db.close().unwrap();
        }

        // Reopen and verify consistency
        {
            let db = GrafeoDB::open(&db_path).unwrap();

            // Should have 2 nodes (a and c), b was deleted
            // Note: node_count includes deleted nodes in some implementations
            // What matters is that the non-deleted nodes are accessible
            let node_a = db.get_node(grafeo_common::types::NodeId::new(0));
            assert!(node_a.is_some());

            let node_c = db.get_node(grafeo_common::types::NodeId::new(2));
            assert!(node_c.is_some());

            // Middle node should be deleted
            let node_b = db.get_node(grafeo_common::types::NodeId::new(1));
            assert!(node_b.is_none());
        }
    }

    #[cfg(all(feature = "wal", feature = "grafeo-file"))]
    #[test]
    fn test_close_is_idempotent() {
        // Calling close() multiple times should not cause errors
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("close_test_db.grafeo");

        let db = GrafeoDB::open(&db_path).unwrap();
        db.create_node(&["Test"]).unwrap();

        // First close should succeed
        assert!(db.close().is_ok());

        // Second close should also succeed (idempotent)
        assert!(db.close().is_ok());
    }

    #[test]
    fn test_with_store_external_backend() {
        use grafeo_core::graph::lpg::LpgStore;

        let external = Arc::new(LpgStore::new().unwrap());

        // Seed data on the external store directly
        let n1 = external.create_node(&["Person"]);
        external.set_node_property(n1, "name", grafeo_common::types::Value::from("Alix"));

        let db = GrafeoDB::with_store(
            Arc::clone(&external) as Arc<dyn GraphStoreMut>,
            Config::in_memory(),
        )
        .unwrap();

        let session = db.session();

        // Session should see data from the external store via execute
        #[cfg(feature = "gql")]
        {
            let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
            assert_eq!(result.rows.len(), 1);
        }
    }

    #[test]
    fn test_with_config_custom_memory_limit() {
        let config = Config::in_memory().with_memory_limit(64 * 1024 * 1024); // 64 MB

        let db = GrafeoDB::with_config(config).unwrap();
        assert_eq!(db.config().memory_limit, Some(64 * 1024 * 1024));
        assert_eq!(db.node_count(), 0);
    }

    #[cfg(feature = "metrics")]
    #[test]
    fn test_database_metrics_registry() {
        let db = GrafeoDB::new_in_memory();

        // Perform some operations
        db.create_node(&["Person"]).unwrap();
        db.create_node(&["Person"]).unwrap();

        // Check that metrics snapshot returns data
        let snap = db.metrics();
        // Session created counter should reflect at least 0 (metrics is initialized)
        assert_eq!(snap.query_count, 0); // No queries executed yet
    }

    #[test]
    fn test_query_result_has_metrics() {
        // Verifies that query results include execution metrics
        let db = GrafeoDB::new_in_memory();
        db.create_node(&["Person"]).unwrap();
        db.create_node(&["Person"]).unwrap();

        #[cfg(feature = "gql")]
        {
            let result = db.execute("MATCH (n:Person) RETURN n").unwrap();

            // Metrics should be populated
            assert!(result.execution_time_ms.is_some());
            assert!(result.rows_scanned.is_some());
            assert!(result.execution_time_ms.unwrap() >= 0.0);
            assert_eq!(result.rows_scanned.unwrap(), 2);
        }
    }

    #[test]
    fn test_empty_query_result_metrics() {
        // Verifies metrics are correct for queries returning no results
        let db = GrafeoDB::new_in_memory();
        db.create_node(&["Person"]).unwrap();

        #[cfg(feature = "gql")]
        {
            // Query that matches nothing
            let result = db.execute("MATCH (n:NonExistent) RETURN n").unwrap();

            assert!(result.execution_time_ms.is_some());
            assert!(result.rows_scanned.is_some());
            assert_eq!(result.rows_scanned.unwrap(), 0);
        }
    }

    #[cfg(feature = "cdc")]
    mod cdc_integration {
        use super::*;

        /// Helper: creates an in-memory database with CDC enabled.
        fn cdc_db() -> GrafeoDB {
            GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap()
        }

        #[test]
        fn test_node_lifecycle_history() {
            let db = cdc_db();

            // Create
            let id = db.create_node(&["Person"]).unwrap();
            // Update
            db.set_node_property(id, "name", "Alix".into()).unwrap();
            db.set_node_property(id, "name", "Gus".into()).unwrap();
            // Delete
            db.delete_node(id).unwrap();

            let history = db.history(id).unwrap();
            assert_eq!(history.len(), 4); // create + 2 updates + delete
            assert_eq!(history[0].kind, crate::cdc::ChangeKind::Create);
            assert_eq!(history[1].kind, crate::cdc::ChangeKind::Update);
            assert!(history[1].before.is_none()); // first set_node_property has no prior value
            assert_eq!(history[2].kind, crate::cdc::ChangeKind::Update);
            assert!(history[2].before.is_some()); // second update has prior "Alix"
            assert_eq!(history[3].kind, crate::cdc::ChangeKind::Delete);
        }

        #[test]
        fn test_edge_lifecycle_history() {
            let db = cdc_db();

            let alix = db.create_node(&["Person"]).unwrap();
            let gus = db.create_node(&["Person"]).unwrap();
            let edge = db.create_edge(alix, gus, "KNOWS").unwrap();
            db.set_edge_property(edge, "since", 2024i64.into()).unwrap();
            db.delete_edge(edge).unwrap();

            let history = db.history(edge).unwrap();
            assert_eq!(history.len(), 3); // create + update + delete
            assert_eq!(history[0].kind, crate::cdc::ChangeKind::Create);
            assert_eq!(history[1].kind, crate::cdc::ChangeKind::Update);
            assert_eq!(history[2].kind, crate::cdc::ChangeKind::Delete);
        }

        #[test]
        fn test_create_node_with_props_cdc() {
            let db = cdc_db();

            let id = db
                .create_node_with_props(
                    &["Person"],
                    vec![
                        ("name", grafeo_common::types::Value::from("Alix")),
                        ("age", grafeo_common::types::Value::from(30i64)),
                    ],
                )
                .unwrap();

            let history = db.history(id).unwrap();
            assert_eq!(history.len(), 1);
            assert_eq!(history[0].kind, crate::cdc::ChangeKind::Create);
            // Props should be captured
            let after = history[0].after.as_ref().unwrap();
            assert_eq!(after.len(), 2);
        }

        #[test]
        fn test_changes_between() {
            let db = cdc_db();

            let id1 = db.create_node(&["A"]).unwrap();
            let _id2 = db.create_node(&["B"]).unwrap();
            db.set_node_property(id1, "x", 1i64.into()).unwrap();

            // All events should be at the same epoch (in-memory, epoch doesn't advance without tx)
            let changes = db
                .changes_between(
                    grafeo_common::types::EpochId(0),
                    grafeo_common::types::EpochId(u64::MAX),
                )
                .unwrap();
            assert_eq!(changes.len(), 3); // 2 creates + 1 update
        }

        #[test]
        fn test_cdc_disabled_by_default() {
            let db = GrafeoDB::new_in_memory();
            assert!(!db.is_cdc_enabled());

            let id = db.create_node(&["Person"]).unwrap();
            db.set_node_property(id, "name", "Alix".into()).unwrap();

            let history = db.history(id).unwrap();
            assert!(history.is_empty(), "CDC off by default: no events recorded");
        }

        #[test]
        fn test_session_with_cdc_override_on() {
            // Database default is off, but session opts in
            let db = GrafeoDB::new_in_memory();
            let session = db.session_with_cdc(true);
            session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
            // The CDC log should have events from the opted-in session
            let changes = db
                .changes_between(
                    grafeo_common::types::EpochId(0),
                    grafeo_common::types::EpochId(u64::MAX),
                )
                .unwrap();
            assert!(
                !changes.is_empty(),
                "session_with_cdc(true) should record events"
            );
        }

        #[test]
        fn test_session_with_cdc_override_off() {
            // Database default is on, but session opts out
            let db = cdc_db();
            let session = db.session_with_cdc(false);
            session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
            let changes = db
                .changes_between(
                    grafeo_common::types::EpochId(0),
                    grafeo_common::types::EpochId(u64::MAX),
                )
                .unwrap();
            assert!(
                changes.is_empty(),
                "session_with_cdc(false) should not record events"
            );
        }

        #[test]
        fn test_set_cdc_enabled_runtime() {
            let db = GrafeoDB::new_in_memory();
            assert!(!db.is_cdc_enabled());

            // Enable at runtime
            db.set_cdc_enabled(true);
            assert!(db.is_cdc_enabled());

            let id = db.create_node(&["Person"]).unwrap();
            let history = db.history(id).unwrap();
            assert_eq!(history.len(), 1, "CDC enabled at runtime records events");

            // Disable again
            db.set_cdc_enabled(false);
            let id2 = db.create_node(&["Person"]).unwrap();
            let history2 = db.history(id2).unwrap();
            assert!(
                history2.is_empty(),
                "CDC disabled at runtime stops recording"
            );
        }
    }

    #[test]
    fn test_with_store_basic() {
        use grafeo_core::graph::lpg::LpgStore;

        let store = Arc::new(LpgStore::new().unwrap());
        let n1 = store.create_node(&["Person"]);
        store.set_node_property(n1, "name", "Alix".into());

        let graph_store = Arc::clone(&store) as Arc<dyn GraphStoreMut>;
        let db = GrafeoDB::with_store(graph_store, Config::in_memory()).unwrap();

        let result = db.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn test_with_store_session() {
        use grafeo_core::graph::lpg::LpgStore;

        let store = Arc::new(LpgStore::new().unwrap());
        let graph_store = Arc::clone(&store) as Arc<dyn GraphStoreMut>;
        let db = GrafeoDB::with_store(graph_store, Config::in_memory()).unwrap();

        let session = db.session();
        let result = session.execute("MATCH (n) RETURN count(n)").unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn test_with_store_mutations() {
        use grafeo_core::graph::lpg::LpgStore;

        let store = Arc::new(LpgStore::new().unwrap());
        let graph_store = Arc::clone(&store) as Arc<dyn GraphStoreMut>;
        let db = GrafeoDB::with_store(graph_store, Config::in_memory()).unwrap();

        let mut session = db.session();

        // Use an explicit transaction so INSERT and MATCH share the same
        // transaction context. With PENDING epochs, uncommitted versions are
        // only visible to the owning transaction.
        session.begin_transaction().unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.rows.len(), 1);

        session.commit().unwrap();
    }

    // =========================================================================
    // QueryResult tests
    // =========================================================================

    #[test]
    fn test_query_result_empty() {
        let result = QueryResult::empty();
        assert!(result.is_empty());
        assert_eq!(result.row_count(), 0);
        assert_eq!(result.column_count(), 0);
        assert!(result.execution_time_ms().is_none());
        assert!(result.rows_scanned().is_none());
        assert!(result.status_message.is_none());
    }

    #[test]
    fn test_query_result_status() {
        let result = QueryResult::status("Created node type 'Person'");
        assert!(result.is_empty());
        assert_eq!(result.column_count(), 0);
        assert_eq!(
            result.status_message.as_deref(),
            Some("Created node type 'Person'")
        );
    }

    #[test]
    fn test_query_result_new_with_columns() {
        let result = QueryResult::new(vec!["name".into(), "age".into()]).unwrap();
        assert_eq!(result.column_count(), 2);
        assert_eq!(result.row_count(), 0);
        assert!(result.is_empty());
        // Column types should default to Any
        assert_eq!(
            result.column_types,
            vec![
                grafeo_common::types::LogicalType::Any,
                grafeo_common::types::LogicalType::Any
            ]
        );
    }

    #[test]
    fn test_query_result_with_types() {
        use grafeo_common::types::LogicalType;
        let result = QueryResult::with_types(
            vec!["name".into(), "age".into()],
            vec![LogicalType::String, LogicalType::Int64],
        )
        .unwrap();
        assert_eq!(result.column_count(), 2);
        assert_eq!(result.column_types[0], LogicalType::String);
        assert_eq!(result.column_types[1], LogicalType::Int64);
    }

    #[test]
    fn test_query_result_rejects_duplicate_columns() {
        // The fail-closed result-column uniqueness invariant. A result must
        // never carry two columns with the same name (name-addressed FFI
        // consumers would silently drop one), so the column-bearing constructors
        // fail closed with a structured error that names the duplicate.
        let dup_new = QueryResult::new(vec!["x".into(), "x".into()]);
        assert!(dup_new.is_err());
        assert!(
            dup_new
                .unwrap_err()
                .to_string()
                .to_lowercase()
                .contains("duplicate column"),
            "error should name the duplicate column"
        );

        use grafeo_common::types::LogicalType;
        assert!(
            QueryResult::with_types(
                vec!["a".into(), "a".into()],
                vec![LogicalType::Any, LogicalType::Any],
            )
            .is_err()
        );
        assert!(QueryResult::from_rows(vec!["c".into(), "c".into()], vec![]).is_err());

        // Distinct column names still construct successfully.
        assert!(QueryResult::new(vec!["a".into(), "b".into()]).is_ok());
        assert!(QueryResult::from_rows(vec!["a".into(), "b".into()], vec![]).is_ok());
        // The zero/one-column cases are trivially unique.
        assert!(QueryResult::new(vec![]).is_ok());
        assert!(QueryResult::new(vec!["only".into()]).is_ok());
    }

    #[test]
    fn test_query_result_with_metrics() {
        let result = QueryResult::new(vec!["x".into()])
            .unwrap()
            .with_metrics(42.5, 100);
        assert_eq!(result.execution_time_ms(), Some(42.5));
        assert_eq!(result.rows_scanned(), Some(100));
    }

    #[test]
    fn test_query_result_scalar_success() {
        use grafeo_common::types::Value;
        let mut result = QueryResult::new(vec!["count".into()]).unwrap();
        result.rows.push(vec![Value::Int64(42)]);

        let val: i64 = result.scalar().unwrap();
        assert_eq!(val, 42);
    }

    #[test]
    fn test_query_result_scalar_wrong_shape() {
        use grafeo_common::types::Value;
        // Multiple rows
        let mut result = QueryResult::new(vec!["x".into()]).unwrap();
        result.rows.push(vec![Value::Int64(1)]);
        result.rows.push(vec![Value::Int64(2)]);
        assert!(result.scalar::<i64>().is_err());

        // Multiple columns
        let mut result2 = QueryResult::new(vec!["a".into(), "b".into()]).unwrap();
        result2.rows.push(vec![Value::Int64(1), Value::Int64(2)]);
        assert!(result2.scalar::<i64>().is_err());

        // Empty
        let result3 = QueryResult::new(vec!["x".into()]).unwrap();
        assert!(result3.scalar::<i64>().is_err());
    }

    #[test]
    fn test_query_result_iter() {
        use grafeo_common::types::Value;
        let mut result = QueryResult::new(vec!["x".into()]).unwrap();
        result.rows.push(vec![Value::Int64(1)]);
        result.rows.push(vec![Value::Int64(2)]);

        let collected: Vec<_> = result.iter().collect();
        assert_eq!(collected.len(), 2);
    }

    #[test]
    fn test_query_result_display() {
        use grafeo_common::types::Value;
        let mut result = QueryResult::new(vec!["name".into()]).unwrap();
        result.rows.push(vec![Value::from("Alix")]);
        let display = result.to_string();
        assert!(display.contains("name"));
        assert!(display.contains("Alix"));
    }

    // =========================================================================
    // FromValue error paths
    // =========================================================================

    #[test]
    fn test_from_value_i64_type_mismatch() {
        use grafeo_common::types::Value;
        let val = Value::from("not a number");
        assert!(i64::from_value(&val).is_err());
    }

    #[test]
    fn test_from_value_f64_type_mismatch() {
        use grafeo_common::types::Value;
        let val = Value::from("not a float");
        assert!(f64::from_value(&val).is_err());
    }

    #[test]
    fn test_from_value_string_type_mismatch() {
        use grafeo_common::types::Value;
        let val = Value::Int64(42);
        assert!(String::from_value(&val).is_err());
    }

    #[test]
    fn test_from_value_bool_type_mismatch() {
        use grafeo_common::types::Value;
        let val = Value::Int64(1);
        assert!(bool::from_value(&val).is_err());
    }

    #[test]
    fn test_from_value_all_success() {
        use grafeo_common::types::Value;
        assert_eq!(i64::from_value(&Value::Int64(99)).unwrap(), 99);
        assert!((f64::from_value(&Value::Float64(2.72)).unwrap() - 2.72).abs() < f64::EPSILON);
        assert_eq!(String::from_value(&Value::from("hello")).unwrap(), "hello");
        assert!(bool::from_value(&Value::Bool(true)).unwrap());
    }

    // =========================================================================
    // GrafeoDB accessor tests
    // =========================================================================

    #[test]
    fn test_database_is_read_only_false_by_default() {
        let db = GrafeoDB::new_in_memory();
        assert!(!db.is_read_only());
    }

    #[test]
    fn test_database_graph_model() {
        let db = GrafeoDB::new_in_memory();
        assert_eq!(db.graph_model(), crate::config::GraphModel::Lpg);
    }

    #[test]
    fn test_database_memory_limit_none_by_default() {
        let db = GrafeoDB::new_in_memory();
        assert!(db.memory_limit().is_none());
    }

    #[test]
    fn test_database_memory_limit_custom() {
        let config = Config::in_memory().with_memory_limit(128 * 1024 * 1024);
        let db = GrafeoDB::with_config(config).unwrap();
        assert_eq!(db.memory_limit(), Some(128 * 1024 * 1024));
    }

    #[test]
    fn test_database_buffer_manager() {
        let db = GrafeoDB::new_in_memory();
        let _bm = db.buffer_manager();
        // Just verify it doesn't panic
    }

    #[test]
    fn test_database_query_cache() {
        let db = GrafeoDB::new_in_memory();
        let _qc = db.query_cache();
    }

    #[test]
    fn test_database_clear_plan_cache() {
        let db = GrafeoDB::new_in_memory();
        // Execute a query to populate the cache
        #[cfg(feature = "gql")]
        {
            let _ = db.execute("MATCH (n) RETURN count(n)");
        }
        db.clear_plan_cache();
        // No panic means success
    }

    #[test]
    fn test_database_gc() {
        let db = GrafeoDB::new_in_memory();
        db.create_node(&["Person"]).unwrap();
        db.gc();
        // Verify no panic, node still accessible
        assert_eq!(db.node_count(), 1);
    }

    // =========================================================================
    // Named graph management
    // =========================================================================

    #[test]
    fn test_create_and_list_graphs() {
        let db = GrafeoDB::new_in_memory();
        let created = db.create_graph("social").unwrap();
        assert!(created);

        // Creating same graph again returns false
        let created_again = db.create_graph("social").unwrap();
        assert!(!created_again);

        let names = db.list_graphs();
        assert!(names.contains(&"social".to_string()));
    }

    #[test]
    fn test_drop_graph() {
        let db = GrafeoDB::new_in_memory();
        db.create_graph("temp").unwrap();
        assert!(db.drop_graph("temp").unwrap());
        assert!(!db.drop_graph("temp").unwrap()); // Already dropped
    }

    #[test]
    fn test_drop_graph_resets_current_graph() {
        let db = GrafeoDB::new_in_memory();
        db.create_graph("active").unwrap();
        db.set_current_graph(Some("active")).unwrap();
        assert_eq!(db.current_graph(), Some("active".to_string()));

        db.drop_graph("active").unwrap();
        assert_eq!(db.current_graph(), None);
    }

    // =========================================================================
    // Current graph / schema context
    // =========================================================================

    #[test]
    fn test_current_graph_default_none() {
        let db = GrafeoDB::new_in_memory();
        assert_eq!(db.current_graph(), None);
    }

    #[test]
    fn test_set_current_graph_valid() {
        let db = GrafeoDB::new_in_memory();
        db.create_graph("social").unwrap();
        db.set_current_graph(Some("social")).unwrap();
        assert_eq!(db.current_graph(), Some("social".to_string()));
    }

    #[test]
    fn test_set_current_graph_nonexistent() {
        let db = GrafeoDB::new_in_memory();
        let result = db.set_current_graph(Some("nonexistent"));
        assert!(result.is_err());
    }

    #[test]
    fn test_set_current_graph_none_resets() {
        let db = GrafeoDB::new_in_memory();
        db.create_graph("social").unwrap();
        db.set_current_graph(Some("social")).unwrap();
        db.set_current_graph(None).unwrap();
        assert_eq!(db.current_graph(), None);
    }

    #[test]
    fn test_set_current_graph_default_keyword() {
        let db = GrafeoDB::new_in_memory();
        // "default" is a special case that always succeeds
        db.set_current_graph(Some("default")).unwrap();
        assert_eq!(db.current_graph(), Some("default".to_string()));
    }

    #[test]
    fn test_current_schema_default_none() {
        let db = GrafeoDB::new_in_memory();
        assert_eq!(db.current_schema(), None);
    }

    #[test]
    fn test_set_current_schema_nonexistent() {
        let db = GrafeoDB::new_in_memory();
        let result = db.set_current_schema(Some("nonexistent"));
        assert!(result.is_err());
    }

    #[test]
    fn test_set_current_schema_none_resets() {
        let db = GrafeoDB::new_in_memory();
        db.set_current_schema(None).unwrap();
        assert_eq!(db.current_schema(), None);
    }

    // =========================================================================
    // graph_store / graph_store_mut
    // =========================================================================

    #[test]
    fn test_graph_store_returns_lpg_by_default() {
        let db = GrafeoDB::new_in_memory();
        db.create_node(&["Person"]).unwrap();
        let store = db.graph_store();
        assert_eq!(store.node_count(), 1);
    }

    #[test]
    fn test_graph_store_mut_returns_some_by_default() {
        let db = GrafeoDB::new_in_memory();
        assert!(db.graph_store_mut().is_some());
    }

    #[test]
    fn test_with_read_store() {
        use grafeo_core::graph::lpg::LpgStore;

        let store = Arc::new(LpgStore::new().unwrap());
        store.create_node(&["Person"]);

        let read_store = Arc::clone(&store) as Arc<dyn GraphStoreSearch>;
        let db = GrafeoDB::with_read_store(read_store, Config::in_memory()).unwrap();

        assert!(db.is_read_only());
        assert!(db.graph_store_mut().is_none());

        // Read queries should work
        let gs = db.graph_store();
        assert_eq!(gs.node_count(), 1);
    }

    #[test]
    fn test_with_store_graph_store_methods() {
        use grafeo_core::graph::lpg::LpgStore;

        let store = Arc::new(LpgStore::new().unwrap());
        store.create_node(&["Person"]);

        let db = GrafeoDB::with_store(
            Arc::clone(&store) as Arc<dyn GraphStoreMut>,
            Config::in_memory(),
        )
        .unwrap();

        assert!(!db.is_read_only());
        assert!(db.graph_store_mut().is_some());
        assert_eq!(db.graph_store().node_count(), 1);
    }

    // =========================================================================
    // session_read_only
    // =========================================================================

    #[test]
    #[allow(deprecated)]
    fn test_session_read_only() {
        let db = GrafeoDB::new_in_memory();
        db.create_node(&["Person"]).unwrap();

        let session = db.session_read_only();
        // Read queries should work
        #[cfg(feature = "gql")]
        {
            let result = session.execute("MATCH (n) RETURN count(n)").unwrap();
            assert_eq!(result.rows.len(), 1);
        }
    }

    // =========================================================================
    // close on in-memory database
    // =========================================================================

    #[test]
    fn test_close_in_memory_database() {
        let db = GrafeoDB::new_in_memory();
        db.create_node(&["Person"]).unwrap();
        assert!(db.close().is_ok());
        // Second close should also be fine (idempotent)
        assert!(db.close().is_ok());
    }

    // =========================================================================
    // with_config validation failure
    // =========================================================================

    #[test]
    fn test_with_config_invalid_config_zero_threads() {
        let config = Config::in_memory().with_threads(0);
        let result = GrafeoDB::with_config(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_with_config_invalid_config_zero_memory_limit() {
        let config = Config::in_memory().with_memory_limit(0);
        let result = GrafeoDB::with_config(config);
        assert!(result.is_err());
    }

    // =========================================================================
    // StorageFormat display (for config.rs coverage)
    // =========================================================================

    #[test]
    #[allow(
        deprecated,
        reason = "the crate names its own deprecated variant, until 0.7.0 removes it"
    )]
    fn test_storage_format_display() {
        use crate::config::StorageFormat;
        assert_eq!(StorageFormat::Auto.to_string(), "auto");
        assert_eq!(StorageFormat::WalDirectory.to_string(), "wal-directory");
        assert_eq!(StorageFormat::SingleFile.to_string(), "single-file");
    }

    #[test]
    fn test_storage_format_default() {
        use crate::config::StorageFormat;
        assert_eq!(StorageFormat::default(), StorageFormat::Auto);
    }

    #[test]
    #[allow(
        deprecated,
        reason = "the crate names its own deprecated variant, until 0.7.0 removes it"
    )]
    fn test_config_with_storage_format() {
        use crate::config::StorageFormat;
        let config = Config::in_memory().with_storage_format(StorageFormat::SingleFile);
        assert_eq!(config.storage_format, StorageFormat::SingleFile);
    }

    // =========================================================================
    // Config CDC
    // =========================================================================

    #[test]
    fn test_config_with_cdc() {
        let config = Config::in_memory().with_cdc();
        assert!(config.cdc_enabled);
    }

    #[test]
    fn test_config_cdc_default_false() {
        let config = Config::in_memory();
        assert!(!config.cdc_enabled);
    }

    // =========================================================================
    // ConfigError as std::error::Error
    // =========================================================================

    #[test]
    fn test_config_error_is_error_trait() {
        use crate::config::ConfigError;
        let err: Box<dyn std::error::Error> = Box::new(ConfigError::ZeroMemoryLimit);
        assert!(err.source().is_none());
    }

    // =========================================================================
    // Metrics tests
    // =========================================================================

    #[cfg(feature = "metrics")]
    #[test]
    fn test_metrics_prometheus_output() {
        let db = GrafeoDB::new_in_memory();
        let prom = db.metrics_prometheus();
        // Should contain at least some metric names
        assert!(!prom.is_empty(), "prom is empty");
    }

    #[cfg(feature = "metrics")]
    #[test]
    fn test_reset_metrics() {
        let db = GrafeoDB::new_in_memory();
        // Execute something to generate metrics
        let _session = db.session();
        db.reset_metrics();
        let snap = db.metrics();
        assert_eq!(snap.query_count, 0);
    }

    // =========================================================================
    // drop_graph on external store
    // =========================================================================

    #[test]
    fn test_drop_graph_on_external_store() {
        use grafeo_core::graph::lpg::LpgStore;

        let store = Arc::new(LpgStore::new().unwrap());
        let read_store = Arc::clone(&store) as Arc<dyn GraphStoreSearch>;
        let db = GrafeoDB::with_read_store(read_store, Config::in_memory()).unwrap();

        // drop_graph with external store (no built-in store) returns false
        assert!(!db.drop_graph("anything").unwrap());
    }
}

#[cfg(test)]
mod normalize_path_tests {
    use std::path::{Path, PathBuf};

    /// A path ending in `..` names the directory the file system reads:
    /// through a symlink, the parent of its target, not the lexical parent.
    #[cfg(unix)]
    #[test]
    fn a_path_ending_in_parent_follows_a_symlink_as_the_file_system_does() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("amsterdam").join("berlin");
        std::fs::create_dir_all(&target).unwrap();
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&target, &link).unwrap();

        let resolved = super::normalize_path(&link.join("..")).unwrap();
        assert_eq!(
            resolved,
            std::fs::canonicalize(dir.path().join("amsterdam")).unwrap(),
            "link/.. is the parent of the link's target"
        );
    }

    /// A component that does not exist after a symlink does not make the
    /// whole path lexical: `link/..` still goes through the link, and only
    /// `new/..` (which the file system cannot read) resolves lexically.
    #[cfg(unix)]
    #[test]
    fn a_missing_component_after_a_symlink_keeps_the_link_resolved() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("amsterdam").join("berlin");
        std::fs::create_dir_all(&target).unwrap();
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&target, &link).unwrap();

        let resolved = super::normalize_path(&link.join("..").join("paris").join("..")).unwrap();
        assert_eq!(
            resolved,
            std::fs::canonicalize(dir.path().join("amsterdam")).unwrap(),
            "link/../paris/.. is the parent of the link's target, paris does not exist"
        );
    }

    /// Windows reads `..` lexically, before a junction is followed, so
    /// `junction\..\paris\..` is the directory that holds the junction.
    #[cfg(windows)]
    #[test]
    fn windows_reads_parent_before_a_junction() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("amsterdam").join("berlin");
        std::fs::create_dir_all(&target).unwrap();
        let junction = dir.path().join("junction");
        // A junction needs no elevation, so a failure here is a broken setup,
        // never a reason to pass without the check.
        let output = std::process::Command::new("cmd")
            .args(["/C", "mklink", "/J"])
            .arg(&junction)
            .arg(&target)
            .output()
            .expect("cmd runs mklink /J");
        assert!(
            output.status.success(),
            "mklink /J could not create the junction ({}): {}{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        let resolved =
            super::normalize_path(&junction.join("..").join("paris").join("..")).unwrap();
        assert_eq!(
            resolved,
            std::fs::canonicalize(dir.path()).unwrap(),
            "junction\\..\\paris\\.. is the directory that holds the junction"
        );
    }

    /// The part of the path that exists is resolved by the file system (on
    /// Windows that gives the `\\?\` form, on macOS `/private/var` for
    /// `/var`), the rest lexically: the spelling is the same whether or not
    /// the last components exist.
    #[test]
    fn a_missing_tail_is_appended_to_the_resolved_existing_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let existing = dir.path().join("amsterdam");
        std::fs::create_dir_all(&existing).unwrap();

        let resolved =
            super::normalize_path(&existing.join("paris").join("prague").join("..")).unwrap();
        assert_eq!(
            resolved,
            std::fs::canonicalize(&existing).unwrap().join("paris"),
            "amsterdam is resolved by the file system, paris/prague/.. lexically"
        );
    }

    /// The prefix logic, with a file system that holds one link: only the
    /// components past the longest prefix it can resolve are resolved
    /// lexically.
    #[test]
    fn only_the_part_past_the_longest_resolvable_prefix_is_lexical() {
        let path = |text: &str| -> PathBuf { Path::new(text).components().collect() };
        // `/data/link` links to `/real/amsterdam/berlin`; nothing else exists.
        let canonicalize = |prefix: &Path| -> std::io::Result<PathBuf> {
            if prefix == path("/data/link/..") {
                Ok(path("/real/amsterdam"))
            } else if prefix == path("/data/link") {
                Ok(path("/real/amsterdam/berlin"))
            } else if prefix == path("/data") || prefix == path("/") {
                Ok(prefix.to_path_buf())
            } else {
                Err(std::io::ErrorKind::NotFound.into())
            }
        };
        let resolve =
            |text: &str| super::resolve_through_longest_existing_prefix(&path(text), canonicalize);

        assert_eq!(resolve("/data/link/../paris/.."), path("/real/amsterdam"));
        assert_eq!(
            resolve("/data/link/paris/../.."),
            path("/real/amsterdam"),
            "paris does not exist: the rest pops from the link's target"
        );
        assert_eq!(resolve("/data/gone/prague/.."), path("/data/gone"));
        assert_eq!(resolve("/data/.."), path("/"));
    }
}
