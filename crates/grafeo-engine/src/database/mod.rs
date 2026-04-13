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
#[cfg(all(feature = "async-storage", feature = "lpg"))]
pub(crate) mod async_wal_store;
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
pub mod backup;
#[cfg(feature = "lpg")]
pub(crate) mod catalog_section;
#[cfg(feature = "cdc")]
pub(crate) mod cdc_store;
#[cfg(all(feature = "grafeo-file", feature = "lpg"))]
mod checkpoint_timer;
#[cfg(feature = "lpg")]
mod crud;
#[cfg(feature = "embed")]
mod embed;
#[cfg(feature = "grafeo-file")]
pub(crate) mod flush;
#[cfg(feature = "lpg")]
mod import;
#[cfg(feature = "lpg")]
mod index;
#[cfg(feature = "lpg")]
mod persistence;
mod query;
#[cfg(feature = "triple-store")]
mod rdf_ops;
#[cfg(feature = "lpg")]
mod search;
pub(crate) mod section_consumer;
#[cfg(all(feature = "wal", feature = "lpg"))]
pub(crate) mod wal_store;

use grafeo_common::{grafeo_error, grafeo_warn};
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
use grafeo_core::graph::{GraphStore, GraphStoreMut};
#[cfg(feature = "grafeo-file")]
use grafeo_storage::file::GrafeoFileManager;
#[cfg(all(feature = "wal", feature = "lpg"))]
use grafeo_storage::wal::WalRecovery;
#[cfg(feature = "wal")]
use grafeo_storage::wal::{DurabilityMode as WalDurabilityMode, LpgWal, WalConfig, WalRecord};

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
    /// The underlying graph store (None when using an external store).
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
    /// Shared WAL graph context tracker. Tracks which named graph was last
    /// written to the WAL, so concurrent sessions can emit `SwitchGraph`
    /// records only when the context actually changes.
    #[cfg(feature = "wal")]
    pub(super) wal_graph_context: Arc<parking_lot::Mutex<Option<String>>>,
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
    /// Shared registry of spilled vector storages.
    /// Used by the search path to create `SpillableVectorAccessor` instances.
    #[cfg(all(feature = "vector-index", feature = "mmap", not(feature = "temporal")))]
    vector_spill_storages: Option<
        Arc<
            parking_lot::RwLock<
                std::collections::HashMap<String, Arc<grafeo_core::index::vector::MmapStorage>>,
            >,
        >,
    >,
    /// External read-only graph store (when using with_store() or with_read_store()).
    /// When set, sessions route queries through this store instead of the built-in LpgStore.
    pub(super) external_read_store: Option<Arc<dyn GraphStore>>,
    /// External writable graph store (when using with_store()).
    /// None for read-only databases created via with_read_store().
    pub(super) external_write_store: Option<Arc<dyn GraphStoreMut>>,
    /// For hybrid stores: the overlay LpgStore that should receive MVCC finalization
    /// on commit (finalize_version_epochs, sync_epoch). Sessions use this as their
    /// internal store so that auto-committed writes become visible across sessions.
    #[cfg(all(feature = "lpg", feature = "compact-store"))]
    pub(super) hybrid_overlay: Option<Arc<LpgStore>>,
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
    fn lpg_store(&self) -> &Arc<LpgStore> {
        self.store.as_ref().expect(
            "no built-in LpgStore: this GrafeoDB was created with an external store \
             (with_store / with_read_store). Use session() or graph_store() instead.",
        )
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
    /// empty database.
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

    /// Opens an existing database in read-only mode.
    ///
    /// Uses a shared file lock, so multiple processes can read the same
    /// `.grafeo` file concurrently. The database loads the last checkpoint
    /// snapshot but does **not** replay the WAL or allow mutations.
    ///
    /// Currently only supports the single-file (`.grafeo`) format.
    ///
    /// # Errors
    ///
    /// Returns an error if the file doesn't exist or can't be read.
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
    /// # Errors
    ///
    /// Returns an error if the database can't be created or recovery fails.
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
        // Validate configuration before proceeding
        config
            .validate()
            .map_err(|e| grafeo_common::utils::error::Error::Internal(e.to_string()))?;

        #[cfg(feature = "lpg")]
        let store = Arc::new(LpgStore::new()?);
        #[cfg(feature = "triple-store")]
        let rdf_store = Arc::new(RdfStore::new());
        let transaction_manager = Arc::new(TransactionManager::new());

        // Create buffer manager with configured limits
        let buffer_config = BufferManagerConfig {
            budget: config.memory_limit.unwrap_or_else(|| {
                (BufferManagerConfig::detect_system_memory() as f64 * 0.75) as usize
            }),
            spill_path: config.spill_path.clone().or_else(|| {
                config.path.as_ref().and_then(|p| {
                    let parent = p.parent()?;
                    let name = p.file_name()?.to_str()?;
                    Some(parent.join(format!("{name}.spill")))
                })
            }),
            ..BufferManagerConfig::default()
        };
        let buffer_manager = BufferManager::new(buffer_config);

        // Create catalog early so WAL replay can restore schema definitions
        let catalog = Arc::new(Catalog::new());

        let is_read_only = config.access_mode == crate::config::AccessMode::ReadOnly;

        // --- Single-file format (.grafeo) ---
        #[cfg(feature = "grafeo-file")]
        let file_manager: Option<Arc<GrafeoFileManager>> = if is_read_only {
            // Read-only mode: open with shared lock, load snapshot, skip WAL
            if let Some(ref db_path) = config.path {
                if db_path.exists() && db_path.is_file() {
                    let fm = GrafeoFileManager::open_read_only(db_path)?;
                    // Try v2 section-based format first
                    #[cfg(feature = "lpg")]
                    if fm.read_section_directory()?.is_some() {
                        Self::load_from_sections(
                            &fm,
                            &store,
                            &catalog,
                            #[cfg(feature = "triple-store")]
                            &rdf_store,
                        )?;
                    } else {
                        // Fall back to v1 blob format
                        let snapshot_data = fm.read_snapshot()?;
                        if !snapshot_data.is_empty() {
                            Self::apply_snapshot_data(
                                &store,
                                &catalog,
                                #[cfg(feature = "triple-store")]
                                &rdf_store,
                                &snapshot_data,
                            )?;
                        }
                    }
                    Some(Arc::new(fm))
                } else {
                    return Err(grafeo_common::utils::error::Error::Internal(format!(
                        "read-only open requires an existing .grafeo file: {}",
                        db_path.display()
                    )));
                }
            } else {
                return Err(grafeo_common::utils::error::Error::Internal(
                    "read-only mode requires a database path".to_string(),
                ));
            }
        } else if let Some(ref db_path) = config.path {
            // Initialize the file manager whenever single-file format is selected,
            // regardless of whether WAL is enabled. Without this, a database opened
            // with wal_enabled:false + StorageFormat::SingleFile would produce no
            // output at all (the file manager was previously gated behind wal_enabled).
            if Self::should_use_single_file(db_path, config.storage_format) {
                let fm = if db_path.exists() && db_path.is_file() {
                    GrafeoFileManager::open(db_path)?
                } else if !db_path.exists() {
                    GrafeoFileManager::create(db_path)?
                } else {
                    // Path exists but is not a file (directory, etc.)
                    return Err(grafeo_common::utils::error::Error::Internal(format!(
                        "path exists but is not a file: {}",
                        db_path.display()
                    )));
                };

                // Load data: try v2 section-based format, fall back to v1 blob
                #[cfg(feature = "lpg")]
                if fm.read_section_directory()?.is_some() {
                    Self::load_from_sections(
                        &fm,
                        &store,
                        &catalog,
                        #[cfg(feature = "triple-store")]
                        &rdf_store,
                    )?;
                } else {
                    let snapshot_data = fm.read_snapshot()?;
                    if !snapshot_data.is_empty() {
                        Self::apply_snapshot_data(
                            &store,
                            &catalog,
                            #[cfg(feature = "triple-store")]
                            &rdf_store,
                            &snapshot_data,
                        )?;
                    }
                }

                // Recover sidecar WAL if WAL is enabled and a sidecar exists
                #[cfg(all(feature = "wal", feature = "lpg"))]
                if config.wal_enabled && fm.has_sidecar_wal() {
                    let recovery = WalRecovery::new(fm.sidecar_wal_path());
                    let records = recovery.recover()?;
                    Self::apply_wal_records(
                        &store,
                        &catalog,
                        #[cfg(feature = "triple-store")]
                        &rdf_store,
                        &records,
                    )?;
                }

                Some(Arc::new(fm))
            } else {
                None
            }
        } else {
            None
        };

        // Determine whether to use the WAL directory path (legacy) or sidecar
        // Read-only mode skips WAL entirely (no recovery, no creation).
        #[cfg(feature = "wal")]
        let wal = if is_read_only {
            None
        } else if config.wal_enabled {
            if let Some(ref db_path) = config.path {
                // When using single-file format, the WAL is a sidecar directory
                #[cfg(feature = "grafeo-file")]
                let wal_path = if let Some(ref fm) = file_manager {
                    let p = fm.sidecar_wal_path();
                    std::fs::create_dir_all(&p)?;
                    p
                } else {
                    // Legacy: WAL inside the database directory
                    std::fs::create_dir_all(db_path)?;
                    db_path.join("wal")
                };

                #[cfg(not(feature = "grafeo-file"))]
                let wal_path = {
                    std::fs::create_dir_all(db_path)?;
                    db_path.join("wal")
                };

                // For legacy WAL directory format, check if WAL exists and recover
                #[cfg(all(feature = "lpg", feature = "grafeo-file"))]
                let is_single_file = file_manager.is_some();
                #[cfg(all(feature = "lpg", not(feature = "grafeo-file")))]
                let is_single_file = false;

                #[cfg(feature = "lpg")]
                if !is_single_file && wal_path.exists() {
                    let recovery = WalRecovery::new(&wal_path);
                    let records = recovery.recover()?;
                    Self::apply_wal_records(
                        &store,
                        &catalog,
                        #[cfg(feature = "triple-store")]
                        &rdf_store,
                        &records,
                    )?;
                }

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
                let wal_manager = LpgWal::with_config(&wal_path, wal_config)?;
                Some(Arc::new(wal_manager))
            } else {
                None
            }
        } else {
            None
        };

        // Create query cache with default capacity (1000 queries)
        let query_cache = Arc::new(QueryCache::default());

        // After all snapshot/WAL recovery, sync TransactionManager epoch
        // with the store so queries use the correct viewing epoch.
        #[cfg(all(feature = "temporal", feature = "lpg"))]
        transaction_manager.sync_epoch(store.current_epoch());

        #[cfg(feature = "cdc")]
        let cdc_enabled_val = config.cdc_enabled;
        #[cfg(feature = "cdc")]
        let cdc_retention = config.cdc_retention.clone();

        // Clone Arcs for the checkpoint timer before moving originals into the struct.
        // The timer captures its own references and runs in a background thread.
        #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
        let checkpoint_interval = config.checkpoint_interval;
        #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
        let timer_store = Arc::clone(&store);
        #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
        let timer_catalog = Arc::clone(&catalog);
        #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
        let timer_tm = Arc::clone(&transaction_manager);
        #[cfg(all(feature = "grafeo-file", feature = "lpg", feature = "triple-store"))]
        let timer_rdf = Arc::clone(&rdf_store);
        #[cfg(all(feature = "grafeo-file", feature = "lpg", feature = "wal"))]
        let timer_wal = wal.clone();

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
            #[cfg(feature = "wal")]
            wal_graph_context: Arc::new(parking_lot::Mutex::new(None)),
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
            #[cfg(all(feature = "vector-index", feature = "mmap", not(feature = "temporal")))]
            vector_spill_storages: None,
            external_read_store: None,
            external_write_store: None,
            #[cfg(all(feature = "lpg", feature = "compact-store"))]
            hybrid_overlay: None,
            #[cfg(feature = "metrics")]
            metrics: Some(Arc::new(crate::metrics::MetricsRegistry::new())),
            current_graph: RwLock::new(None),
            current_schema: RwLock::new(None),
            read_only: is_read_only,
            projections: Arc::new(RwLock::new(std::collections::HashMap::new())),
        };

        // Register storage sections as memory consumers for pressure tracking
        db.register_section_consumers();

        // Start periodic checkpoint timer if configured
        #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
        if let (Some(interval), Some(fm)) = (checkpoint_interval, &db.file_manager)
            && !is_read_only
        {
            *db.checkpoint_timer.lock() = Some(checkpoint_timer::CheckpointTimer::start(
                interval,
                Arc::clone(fm),
                timer_store,
                timer_catalog,
                timer_tm,
                #[cfg(feature = "triple-store")]
                timer_rdf,
                #[cfg(feature = "wal")]
                timer_wal,
            ));
        }

        // Discover existing spill files from a previous session.
        // If vectors were spilled before close, the spill files persist on disk
        // and need to be re-mapped so search can read from them.
        #[cfg(all(
            feature = "lpg",
            feature = "vector-index",
            feature = "mmap",
            not(feature = "temporal")
        ))]
        db.restore_spill_files();

        // If VectorStore is configured as ForceDisk, immediately spill embeddings.
        // This must happen after register_section_consumers() which creates the consumer.
        #[cfg(all(feature = "vector-index", feature = "mmap", not(feature = "temporal")))]
        if db
            .config
            .section_configs
            .get(&grafeo_common::storage::SectionType::VectorStore)
            .is_some_and(|c| c.tier == grafeo_common::storage::TierOverride::ForceDisk)
        {
            db.buffer_manager.spill_all();
        }

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
        config
            .validate()
            .map_err(|e| grafeo_common::utils::error::Error::Internal(e.to_string()))?;

        let transaction_manager = Arc::new(TransactionManager::new());

        let buffer_config = BufferManagerConfig {
            budget: config.memory_limit.unwrap_or_else(|| {
                (BufferManagerConfig::detect_system_memory() as f64 * 0.75) as usize
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
            #[cfg(feature = "wal")]
            wal_graph_context: Arc::new(parking_lot::Mutex::new(None)),
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
            #[cfg(all(feature = "vector-index", feature = "mmap", not(feature = "temporal")))]
            vector_spill_storages: None,
            external_read_store: Some(Arc::clone(&store) as Arc<dyn GraphStore>),
            external_write_store: Some(store),
            #[cfg(all(feature = "lpg", feature = "compact-store"))]
            hybrid_overlay: None,
            #[cfg(feature = "metrics")]
            metrics: Some(Arc::new(crate::metrics::MetricsRegistry::new())),
            current_graph: RwLock::new(None),
            current_schema: RwLock::new(None),
            read_only: false,
            projections: Arc::new(RwLock::new(std::collections::HashMap::new())),
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
    /// use grafeo_core::graph::GraphStore;
    ///
    /// fn example(store: Arc<dyn GraphStore>) -> grafeo_common::utils::error::Result<()> {
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
    pub fn with_read_store(store: Arc<dyn GraphStore>, config: Config) -> Result<Self> {
        config
            .validate()
            .map_err(|e| grafeo_common::utils::error::Error::Internal(e.to_string()))?;

        let transaction_manager = Arc::new(TransactionManager::new());

        let buffer_config = BufferManagerConfig {
            budget: config.memory_limit.unwrap_or_else(|| {
                (BufferManagerConfig::detect_system_memory() as f64 * 0.75) as usize
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
            #[cfg(feature = "wal")]
            wal_graph_context: Arc::new(parking_lot::Mutex::new(None)),
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
            #[cfg(all(feature = "vector-index", feature = "mmap", not(feature = "temporal")))]
            vector_spill_storages: None,
            external_read_store: Some(store),
            external_write_store: None,
            #[cfg(all(feature = "lpg", feature = "compact-store"))]
            hybrid_overlay: None,
            #[cfg(feature = "metrics")]
            metrics: Some(Arc::new(crate::metrics::MetricsRegistry::new())),
            current_graph: RwLock::new(None),
            current_schema: RwLock::new(None),
            read_only: true,
            projections: Arc::new(RwLock::new(std::collections::HashMap::new())),
        })
    }

    /// Converts the database to a read-only [`CompactStore`] for faster queries.
    ///
    /// Takes a snapshot of all nodes and edges from the current store, builds
    /// a columnar `CompactStore` with CSR adjacency, and switches the database
    /// to read-only mode. The original store is dropped to free memory.
    ///
    /// After calling this, all write queries will fail with
    /// `TransactionError::ReadOnly`. Read queries (across all supported
    /// languages) continue to work and benefit from ~60x memory reduction
    /// and 100x+ traversal speedup.
    ///
    /// # Errors
    ///
    /// Returns an error if the conversion fails (e.g. more than 32,767
    /// distinct labels or edge types).
    ///
    /// [`CompactStore`]: grafeo_core::graph::compact::CompactStore
    #[cfg(feature = "compact-store")]
    pub fn compact(&mut self) -> Result<()> {
        use grafeo_core::graph::compact::from_graph_store;

        let current_store = self.graph_store();
        let compact = from_graph_store(current_store.as_ref())
            .map_err(|e| grafeo_common::utils::error::Error::Internal(e.to_string()))?;

        self.external_read_store = Some(Arc::new(compact) as Arc<dyn GraphStore>);
        self.external_write_store = None;
        #[cfg(feature = "lpg")]
        {
            self.store = None;
        }
        // Clear any hybrid overlay from a prior compact_to_hybrid() call.  The
        // overlay Arc held here is the only owner keeping the old HybridStore
        // overlay alive after the external stores are replaced; not clearing it
        // would leak that LpgStore and cause new sessions to receive a stale
        // overlay reference that has no corresponding write store.
        #[cfg(all(feature = "lpg", feature = "compact-store"))]
        {
            self.hybrid_overlay = None;
        }
        self.read_only = true;
        self.query_cache = Arc::new(QueryCache::default());
        // Projections hold Arc refs to the old store: clear them so they don't
        // serve stale data or prevent the old store's memory from being freed.
        self.projections.write().clear();

        Ok(())
    }

    /// Applies WAL records to restore the database state.
    ///
    /// Data mutation records are routed through a graph cursor that tracks
    /// `SwitchGraph` context markers, replaying mutations into the correct
    /// named graph (or the default graph when cursor is `None`).
    #[cfg(all(feature = "wal", feature = "lpg"))]
    fn apply_wal_records(
        store: &Arc<LpgStore>,
        catalog: &Catalog,
        #[cfg(feature = "triple-store")] rdf_store: &Arc<RdfStore>,
        records: &[WalRecord],
    ) -> Result<()> {
        use crate::catalog::{
            EdgeTypeDefinition, NodeTypeDefinition, PropertyDataType, TypeConstraint, TypedProperty,
        };
        use grafeo_common::utils::error::Error;

        // Graph cursor: tracks which named graph receives data mutations.
        // `None` means the default graph.
        let mut current_graph: Option<String> = None;
        let mut target_store: Arc<LpgStore> = Arc::clone(store);

        for record in records {
            match record {
                // --- Named graph lifecycle ---
                WalRecord::CreateNamedGraph { name } => {
                    let _ = store.create_graph(name);
                }
                WalRecord::DropNamedGraph { name } => {
                    store.drop_graph(name);
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
                WalRecord::CreateNode { id, labels } => {
                    let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
                    target_store.create_node_with_id(*id, &label_refs)?;
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
                    target_store.create_edge_with_id(*id, *src, *dst, edge_type)?;
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
                    target_store.remove_node_property(*id, key);
                }
                WalRecord::RemoveEdgeProperty { id, key } => {
                    target_store.remove_edge_property(*id, key);
                }

                // --- Schema DDL replay (always on root catalog) ---
                WalRecord::CreateNodeType {
                    name,
                    properties,
                    constraints,
                } => {
                    let def = NodeTypeDefinition {
                        name: name.clone(),
                        properties: properties
                            .iter()
                            .map(|(n, t, nullable)| TypedProperty {
                                name: n.clone(),
                                data_type: PropertyDataType::from_type_name(t),
                                nullable: *nullable,
                                default_value: None,
                            })
                            .collect(),
                        constraints: constraints
                            .iter()
                            .map(|(kind, props)| match kind.as_str() {
                                "unique" => TypeConstraint::Unique(props.clone()),
                                "primary_key" => TypeConstraint::PrimaryKey(props.clone()),
                                "not_null" if !props.is_empty() => {
                                    TypeConstraint::NotNull(props[0].clone())
                                }
                                _ => TypeConstraint::Unique(props.clone()),
                            })
                            .collect(),
                        parent_types: Vec::new(),
                    };
                    let _ = catalog.register_node_type(def);
                }
                WalRecord::DropNodeType { name } => {
                    let _ = catalog.drop_node_type(name);
                }
                WalRecord::CreateEdgeType {
                    name,
                    properties,
                    constraints,
                } => {
                    let def = EdgeTypeDefinition {
                        name: name.clone(),
                        properties: properties
                            .iter()
                            .map(|(n, t, nullable)| TypedProperty {
                                name: n.clone(),
                                data_type: PropertyDataType::from_type_name(t),
                                nullable: *nullable,
                                default_value: None,
                            })
                            .collect(),
                        constraints: constraints
                            .iter()
                            .map(|(kind, props)| match kind.as_str() {
                                "unique" => TypeConstraint::Unique(props.clone()),
                                "primary_key" => TypeConstraint::PrimaryKey(props.clone()),
                                "not_null" if !props.is_empty() => {
                                    TypeConstraint::NotNull(props[0].clone())
                                }
                                _ => TypeConstraint::Unique(props.clone()),
                            })
                            .collect(),
                        source_node_types: Vec::new(),
                        target_node_types: Vec::new(),
                    };
                    let _ = catalog.register_edge_type_def(def);
                }
                WalRecord::DropEdgeType { name } => {
                    let _ = catalog.drop_edge_type_def(name);
                }
                WalRecord::CreateIndex { .. } | WalRecord::DropIndex { .. } => {
                    // Index recreation is handled by the store on startup
                    // (indexes are rebuilt from data, not WAL)
                }
                WalRecord::CreateConstraint { .. } | WalRecord::DropConstraint { .. } => {
                    // Constraint definitions are part of type definitions
                    // and replayed via CreateNodeType/CreateEdgeType
                }
                WalRecord::CreateGraphType {
                    name,
                    node_types,
                    edge_types,
                    open,
                } => {
                    use crate::catalog::GraphTypeDefinition;
                    let def = GraphTypeDefinition {
                        name: name.clone(),
                        allowed_node_types: node_types.clone(),
                        allowed_edge_types: edge_types.clone(),
                        open: *open,
                    };
                    let _ = catalog.register_graph_type(def);
                }
                WalRecord::DropGraphType { name } => {
                    let _ = catalog.drop_graph_type(name);
                }
                WalRecord::CreateSchema { name } => {
                    let _ = catalog.register_schema_namespace(name.clone());
                }
                WalRecord::DropSchema { name } => {
                    let _ = catalog.drop_schema_namespace(name);
                }

                WalRecord::AlterNodeType { name, alterations } => {
                    for (action, prop_name, type_name, nullable) in alterations {
                        match action.as_str() {
                            "add" => {
                                let prop = TypedProperty {
                                    name: prop_name.clone(),
                                    data_type: PropertyDataType::from_type_name(type_name),
                                    nullable: *nullable,
                                    default_value: None,
                                };
                                let _ = catalog.alter_node_type_add_property(name, prop);
                            }
                            "drop" => {
                                let _ = catalog.alter_node_type_drop_property(name, prop_name);
                            }
                            _ => {}
                        }
                    }
                }
                WalRecord::AlterEdgeType { name, alterations } => {
                    for (action, prop_name, type_name, nullable) in alterations {
                        match action.as_str() {
                            "add" => {
                                let prop = TypedProperty {
                                    name: prop_name.clone(),
                                    data_type: PropertyDataType::from_type_name(type_name),
                                    nullable: *nullable,
                                    default_value: None,
                                };
                                let _ = catalog.alter_edge_type_add_property(name, prop);
                            }
                            "drop" => {
                                let _ = catalog.alter_edge_type_drop_property(name, prop_name);
                            }
                            _ => {}
                        }
                    }
                }
                WalRecord::AlterGraphType { name, alterations } => {
                    for (action, type_name) in alterations {
                        match action.as_str() {
                            "add_node" => {
                                let _ =
                                    catalog.alter_graph_type_add_node_type(name, type_name.clone());
                            }
                            "drop_node" => {
                                let _ = catalog.alter_graph_type_drop_node_type(name, type_name);
                            }
                            "add_edge" => {
                                let _ =
                                    catalog.alter_graph_type_add_edge_type(name, type_name.clone());
                            }
                            "drop_edge" => {
                                let _ = catalog.alter_graph_type_drop_edge_type(name, type_name);
                            }
                            _ => {}
                        }
                    }
                }

                WalRecord::CreateProcedure {
                    name,
                    params,
                    returns,
                    body,
                } => {
                    use crate::catalog::ProcedureDefinition;
                    let def = ProcedureDefinition {
                        name: name.clone(),
                        params: params.clone(),
                        returns: returns.clone(),
                        body: body.clone(),
                    };
                    let _ = catalog.register_procedure(def);
                }
                WalRecord::DropProcedure { name } => {
                    let _ = catalog.drop_procedure(name);
                }

                // --- RDF triple replay ---
                #[cfg(feature = "triple-store")]
                WalRecord::InsertRdfTriple { .. }
                | WalRecord::DeleteRdfTriple { .. }
                | WalRecord::ClearRdfGraph { .. }
                | WalRecord::CreateRdfGraph { .. }
                | WalRecord::DropRdfGraph { .. } => {
                    rdf_ops::replay_rdf_wal_record(rdf_store, record);
                }
                #[cfg(not(feature = "triple-store"))]
                WalRecord::InsertRdfTriple { .. }
                | WalRecord::DeleteRdfTriple { .. }
                | WalRecord::ClearRdfGraph { .. }
                | WalRecord::CreateRdfGraph { .. }
                | WalRecord::DropRdfGraph { .. } => {}

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
                WalRecord::EpochAdvance { .. } => {
                    // Metadata record: no store mutation needed.
                    // Used by incremental backup and point-in-time recovery.
                }
            }
        }
        Ok(())
    }

    // =========================================================================
    // Single-file format helpers
    // =========================================================================

    /// Returns `true` if the given path should use single-file format.
    #[cfg(feature = "grafeo-file")]
    fn should_use_single_file(
        path: &std::path::Path,
        configured: crate::config::StorageFormat,
    ) -> bool {
        use crate::config::StorageFormat;
        match configured {
            StorageFormat::SingleFile => true,
            StorageFormat::WalDirectory => false,
            StorageFormat::Auto => {
                // Existing file: check magic bytes
                if path.is_file() {
                    if let Ok(mut f) = std::fs::File::open(path) {
                        use std::io::Read;
                        let mut magic = [0u8; 4];
                        if f.read_exact(&mut magic).is_ok() && magic == grafeo_storage::file::MAGIC
                        {
                            return true;
                        }
                    }
                    return false;
                }
                // Existing directory: legacy format
                if path.is_dir() {
                    return false;
                }
                // New path: check extension
                path.extension().is_some_and(|ext| ext == "grafeo")
            }
        }
    }

    /// Applies snapshot data (from a `.grafeo` file) to restore the store and catalog.
    ///
    /// Supports both v1 (monolithic blob) and v2 (section-based) formats.
    #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
    fn apply_snapshot_data(
        store: &Arc<LpgStore>,
        catalog: &Arc<crate::catalog::Catalog>,
        #[cfg(feature = "triple-store")] rdf_store: &Arc<RdfStore>,
        data: &[u8],
    ) -> Result<()> {
        // v1 blob format: pass through to legacy loader
        persistence::load_snapshot_into_store(
            store,
            catalog,
            #[cfg(feature = "triple-store")]
            rdf_store,
            data,
        )
    }

    /// Loads from a section-based `.grafeo` file (v2 format).
    ///
    /// Reads the section directory, then deserializes each section independently.
    #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
    fn load_from_sections(
        fm: &GrafeoFileManager,
        store: &Arc<LpgStore>,
        catalog: &Arc<crate::catalog::Catalog>,
        #[cfg(feature = "triple-store")] rdf_store: &Arc<RdfStore>,
    ) -> Result<()> {
        use grafeo_common::storage::{Section, SectionType};

        let dir = fm.read_section_directory()?.ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(
                "expected v2 section directory but found none".to_string(),
            )
        })?;

        // Load catalog section first (schema defs needed before data)
        if let Some(entry) = dir.find(SectionType::Catalog) {
            let data = fm.read_section_data(entry)?;
            let tm = Arc::new(crate::transaction::TransactionManager::new());
            let mut section = catalog_section::CatalogSection::new(
                Arc::clone(catalog),
                Arc::clone(store),
                move || tm.current_epoch().as_u64(),
            );
            section.deserialize(&data)?;
        }

        // Load LPG store
        if let Some(entry) = dir.find(SectionType::LpgStore) {
            let data = fm.read_section_data(entry)?;
            let mut section = grafeo_core::graph::lpg::LpgStoreSection::new(Arc::clone(store));
            section.deserialize(&data)?;
        }

        // Load RDF store
        #[cfg(feature = "triple-store")]
        if let Some(entry) = dir.find(SectionType::RdfStore) {
            let data = fm.read_section_data(entry)?;
            let mut section = grafeo_core::graph::rdf::RdfStoreSection::new(Arc::clone(rdf_store));
            section.deserialize(&data)?;
        }

        // Restore Ring Index (if persisted)
        #[cfg(feature = "ring-index")]
        if let Some(entry) = dir.find(SectionType::RdfRing) {
            let data = fm.read_section_data(entry)?;
            let mut section = grafeo_core::index::ring::RdfRingSection::new(Arc::clone(rdf_store));
            section.deserialize(&data)?;
        }

        // Restore HNSW topology (if vector indexes exist in both catalog and section)
        #[cfg(feature = "vector-index")]
        if let Some(entry) = dir.find(SectionType::VectorStore) {
            let data = fm.read_section_data(entry)?;
            let indexes = store.vector_index_entries();
            if !indexes.is_empty() {
                let mut section = grafeo_core::index::vector::VectorStoreSection::new(indexes);
                section.deserialize(&data)?;
            }
        }

        // Restore BM25 postings (if text indexes exist in both catalog and section)
        #[cfg(feature = "text-index")]
        if let Some(entry) = dir.find(SectionType::TextIndex) {
            let data = fm.read_section_data(entry)?;
            let indexes = store.text_index_entries();
            if !indexes.is_empty() {
                let mut section = grafeo_core::index::text::TextIndexSection::new(indexes);
                section.deserialize(&data)?;
            }
        }

        Ok(())
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
            adaptive_config: self.config.adaptive.clone(),
            factorized_execution: self.config.factorized_execution,
            graph_model: self.config.graph_model,
            query_timeout: self.config.query_timeout,
            commit_counter: Arc::clone(&self.commit_counter),
            gc_interval: self.config.gc_interval,
            read_only: self.read_only || force_read_only,
            identity: identity.clone(),
            #[cfg(feature = "lpg")]
            projections: Arc::clone(&self.projections),
            #[cfg(all(feature = "lpg", feature = "compact-store"))]
            hybrid_overlay: self.hybrid_overlay.as_ref().map(Arc::clone),
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
        let mut session = Session::with_rdf_store_and_adaptive(
            Arc::clone(self.lpg_store()),
            Arc::clone(&self.rdf_store),
            session_cfg(),
        );
        #[cfg(all(feature = "lpg", not(feature = "triple-store")))]
        let mut session = Session::with_adaptive(Arc::clone(self.lpg_store()), session_cfg());
        #[cfg(not(feature = "lpg"))]
        let mut session =
            Session::with_external_store(self.graph_store(), self.graph_store_mut(), session_cfg())
                .expect("session creation for non-lpg build");

        #[cfg(all(feature = "wal", feature = "lpg"))]
        if let Some(ref wal) = self.wal {
            session.set_wal(Arc::clone(wal), Arc::clone(&self.wal_graph_context));
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
            && let Some(store) = &self.store
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

    /// Returns the adaptive execution configuration.
    #[must_use]
    pub fn adaptive_config(&self) -> &crate::config::AdaptiveConfig {
        &self.config.adaptive
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
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn store(&self) -> &Arc<LpgStore> {
        self.lpg_store()
    }

    // === Named Graph Management ===

    /// Creates a named graph. Returns `true` if created, `false` if it already exists.
    ///
    /// # Errors
    ///
    /// Returns an error if arena allocation fails.
    #[cfg(feature = "lpg")]
    pub fn create_graph(&self, name: &str) -> Result<bool> {
        Ok(self.lpg_store().create_graph(name)?)
    }

    /// Drops a named graph. Returns `true` if dropped, `false` if it did not exist.
    ///
    /// If the dropped graph was the active graph context, the context is reset
    /// to the default graph.
    #[cfg(feature = "lpg")]
    pub fn drop_graph(&self, name: &str) -> bool {
        let Some(store) = &self.store else {
            return false;
        };
        let dropped = store.drop_graph(name);
        if dropped {
            let mut current = self.current_graph.write();
            if current
                .as_deref()
                .is_some_and(|g| g.eq_ignore_ascii_case(name))
            {
                *current = None;
            }
        }
        dropped
    }

    /// Returns all named graph names.
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn list_graphs(&self) -> Vec<String> {
        self.lpg_store().graph_names()
    }

    // === Graph Projections ===

    /// Creates a named graph projection (virtual subgraph).
    ///
    /// The projection filters the graph store to only include nodes with the
    /// specified labels and edges with the specified types. Returns `true` if
    /// created, `false` if a projection with that name already exists.
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
    /// assert!(db.create_projection("social", spec));
    /// ```
    pub fn create_projection(
        &self,
        name: impl Into<String>,
        spec: grafeo_core::graph::ProjectionSpec,
    ) -> bool {
        use grafeo_core::graph::GraphProjection;
        use std::collections::hash_map::Entry;

        let store = self.graph_store();
        let projection = Arc::new(GraphProjection::new(store, spec));
        let mut projections = self.projections.write();
        match projections.entry(name.into()) {
            Entry::Occupied(_) => false,
            Entry::Vacant(e) => {
                e.insert(projection);
                true
            }
        }
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

    /// Returns a named projection as a [`GraphStore`] trait object.
    #[must_use]
    pub fn projection(&self, name: &str) -> Option<Arc<dyn GraphStore>> {
        self.projections
            .read()
            .get(name)
            .map(|p| Arc::clone(p) as Arc<dyn GraphStore>)
    }

    /// Returns the graph store as a trait object.
    ///
    /// Returns a read-only trait object for the active graph store.
    ///
    /// This provides the [`GraphStore`] interface for code that only needs
    /// read operations. For write access, use [`graph_store_mut()`](Self::graph_store_mut).
    ///
    /// [`GraphStore`]: grafeo_core::graph::GraphStore
    #[must_use]
    pub fn graph_store(&self) -> Arc<dyn GraphStore> {
        if let Some(ref ext_read) = self.external_read_store {
            Arc::clone(ext_read)
        } else {
            #[cfg(feature = "lpg")]
            {
                Arc::clone(self.lpg_store()) as Arc<dyn GraphStore>
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
                Some(Arc::clone(self.lpg_store()) as Arc<dyn GraphStoreMut>)
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
    /// version chains older than that threshold. Also cleans up completed
    /// transaction metadata in the transaction manager, and prunes the CDC
    /// event log according to its retention policy.
    pub fn gc(&self) {
        #[cfg(feature = "lpg")]
        let current_epoch = {
            let min_epoch = self.transaction_manager.min_active_epoch();
            self.lpg_store().gc_versions(min_epoch);
            self.transaction_manager.current_epoch()
        };
        self.transaction_manager.gc();

        // Prune CDC events based on retention config (epoch + count limits)
        #[cfg(feature = "cdc")]
        if self.cdc_enabled.load(std::sync::atomic::Ordering::Relaxed) {
            #[cfg(feature = "lpg")]
            self.cdc_log.apply_retention(current_epoch);
        }
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
    /// # Errors
    ///
    /// Returns an error if the WAL can't be flushed (check disk space/permissions).
    pub fn close(&self) -> Result<()> {
        let mut is_open = self.is_open.write();
        if !*is_open {
            return Ok(());
        }

        // Stop the periodic checkpoint timer first, even for read-only databases.
        // compact() can switch a writable DB to read-only after the timer started,
        // so the timer must be stopped before any early return to avoid racing
        // with the closed file manager.
        #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
        if let Some(mut timer) = self.checkpoint_timer.lock().take() {
            timer.stop();
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

        // For single-file format: checkpoint to .grafeo file, then clean up sidecar WAL.
        // We must do this BEFORE the WAL close path because checkpoint_to_file
        // removes the sidecar WAL directory.
        #[cfg(feature = "grafeo-file")]
        let is_single_file = self.file_manager.is_some();
        #[cfg(not(feature = "grafeo-file"))]
        let is_single_file = false;

        #[cfg(feature = "grafeo-file")]
        if let Some(ref fm) = self.file_manager {
            // Flush WAL first so all records are on disk before we snapshot
            #[cfg(feature = "wal")]
            if let Some(ref wal) = self.wal {
                wal.sync()?;
            }
            let flush_result = self.checkpoint_to_file(fm, flush::FlushReason::Explicit)?;

            // Safety check: if WAL has records but the checkpoint was a no-op
            // (zero sections written), the container file may not contain the
            // latest data. This can happen when sections are not marked dirty
            // despite mutations going through the WAL. Force-dirty all sections
            // and retry before removing the sidecar.
            #[cfg(feature = "wal")]
            let flush_result = if flush_result.sections_written == 0 {
                if let Some(ref wal) = self.wal {
                    if wal.record_count() > 0 {
                        grafeo_warn!(
                            "WAL has {} records but checkpoint wrote 0 sections; retrying with forced flush",
                            wal.record_count()
                        );
                        self.checkpoint_to_file(fm, flush::FlushReason::Explicit)?
                    } else {
                        flush_result
                    }
                } else {
                    flush_result
                }
            } else {
                flush_result
            };

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
                grafeo_warn!(
                    "keeping sidecar WAL for recovery: checkpoint wrote 0 sections but WAL has records"
                );
            }
            fm.close()?;
        }

        // Commit and sync WAL (legacy directory format only).
        // We intentionally do NOT call wal.checkpoint() here. Directory format
        // has no snapshot: the WAL files are the sole source of truth. Writing
        // checkpoint.meta would cause recovery to skip older WAL files, losing
        // all data that predates the current log sequence.
        #[cfg(feature = "wal")]
        if !is_single_file && let Some(ref wal) = self.wal {
            // Use the last assigned transaction ID, or create one for the commit record
            let commit_tx = self
                .transaction_manager
                .last_assigned_transaction_id()
                .unwrap_or_else(|| self.transaction_manager.begin());

            // Log a TransactionCommit to mark all pending records as committed
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: commit_tx,
            })?;

            wal.sync()?;
        }

        *is_open = false;
        Ok(())
    }

    /// Returns the typed WAL if available.
    #[cfg(feature = "wal")]
    #[must_use]
    pub fn wal(&self) -> Option<&Arc<LpgWal>> {
        self.wal.as_ref()
    }

    /// Logs a WAL record if WAL is enabled.
    #[cfg(feature = "wal")]
    pub(super) fn log_wal(&self, record: &WalRecord) -> Result<()> {
        if let Some(ref wal) = self.wal {
            wal.log(record)?;
        }
        Ok(())
    }

    /// Registers storage sections as [`MemoryConsumer`]s with the BufferManager.
    ///
    /// Each section reports its memory usage to the buffer manager, enabling
    /// accurate pressure tracking. Called once after database construction.
    fn register_section_consumers(&mut self) {
        #[cfg(feature = "lpg")]
        let store_ref = self.store.as_ref();
        #[cfg(not(feature = "lpg"))]
        // LPG store section
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
            self.buffer_manager.register_consumer(Arc::new(
                section_consumer::SectionConsumer::new(Arc::new(ring)),
            ));
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
            let spill_path = self.buffer_manager.config().spill_path.clone();
            let consumer = Arc::new(section_consumer::VectorIndexConsumer::new(
                store, spill_path,
            ));
            // Share the spill registry with the search path
            self.vector_spill_storages = Some(Arc::clone(consumer.spilled_storages()));
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

    /// Discovers and re-opens spill files from a previous session.
    ///
    /// When the database was closed with spilled vector embeddings, the
    /// `vectors_*.bin` files persist in the spill directory. This method
    /// scans for them, opens each as `MmapStorage`, and registers them
    /// in the `vector_spill_storages` map so search can read from them.
    #[cfg(all(
        feature = "lpg",
        feature = "vector-index",
        feature = "mmap",
        not(feature = "temporal")
    ))]
    fn restore_spill_files(&mut self) {
        use grafeo_core::index::vector::MmapStorage;

        let spill_dir = match self.buffer_manager.config().spill_path {
            Some(ref path) => path.clone(),
            None => return,
        };

        if !spill_dir.exists() {
            return;
        }

        let spill_map = match self.vector_spill_storages {
            Some(ref map) => Arc::clone(map),
            None => return,
        };

        let Ok(entries) = std::fs::read_dir(&spill_dir) else {
            return;
        };

        let Some(ref store) = self.store else {
            return;
        };

        for entry in entries.flatten() {
            let path = entry.path();
            let file_name = match path.file_name().and_then(|n| n.to_str()) {
                Some(name) => name.to_string(),
                None => continue,
            };

            // Match pattern: vectors_{key}.bin where key is percent-encoded
            if !file_name.starts_with("vectors_")
                || !std::path::Path::new(&file_name)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("bin"))
            {
                continue;
            }

            // Extract and decode key: "vectors_Label%3Aembedding.bin" -> "Label:embedding"
            let key_part = &file_name["vectors_".len()..file_name.len() - ".bin".len()];

            // Percent-decode: %3A -> ':', %25 -> '%'
            let key = key_part.replace("%3A", ":").replace("%25", "%");

            // Key must contain ':' (label:property format)
            if !key.contains(':') {
                // Legacy file with old encoding, skip (will be re-created on next spill)
                continue;
            }

            // Only restore if the corresponding vector index exists
            if store.get_vector_index_by_key(&key).is_none() {
                // Stale spill file (index was dropped), clean it up
                let _ = std::fs::remove_file(&path);
                continue;
            }

            // Open the MmapStorage
            match MmapStorage::open(&path) {
                Ok(mmap_storage) => {
                    // Mark the property column as spilled so get() returns None
                    let property = key.split(':').nth(1).unwrap_or("");
                    let prop_key = grafeo_common::types::PropertyKey::new(property);
                    store.node_properties_mark_spilled(&prop_key);

                    spill_map.write().insert(key, Arc::new(mmap_storage));
                }
                Err(e) => {
                    eprintln!("failed to restore spill file {}: {e}", path.display());
                    // Remove corrupt spill file
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
    }

    /// Builds section objects for the current database state.
    #[cfg(feature = "grafeo-file")]
    fn build_sections(&self) -> Vec<Box<dyn grafeo_common::storage::Section>> {
        let mut sections: Vec<Box<dyn grafeo_common::storage::Section>> = Vec::new();

        // LPG sections: store, catalog, vector indexes, text indexes
        #[cfg(feature = "lpg")]
        if let Some(store) = self.store.as_ref() {
            let lpg = grafeo_core::graph::lpg::LpgStoreSection::new(Arc::clone(store));

            let catalog = catalog_section::CatalogSection::new(
                Arc::clone(&self.catalog),
                Arc::clone(store),
                {
                    let tm = Arc::clone(&self.transaction_manager);
                    move || tm.current_epoch().as_u64()
                },
            );

            sections.push(Box::new(catalog));
            sections.push(Box::new(lpg));

            // Vector indexes: persist HNSW topology to avoid rebuild on load
            #[cfg(feature = "vector-index")]
            {
                let indexes = store.vector_index_entries();
                if !indexes.is_empty() {
                    let vector = grafeo_core::index::vector::VectorStoreSection::new(indexes);
                    sections.push(Box::new(vector));
                }
            }

            // Text indexes: persist BM25 postings to avoid rebuild on load
            #[cfg(feature = "text-index")]
            {
                let indexes = store.text_index_entries();
                if !indexes.is_empty() {
                    let text = grafeo_core::index::text::TextIndexSection::new(indexes);
                    sections.push(Box::new(text));
                }
            }
        }

        #[cfg(feature = "triple-store")]
        if !self.rdf_store.is_empty() || self.rdf_store.graph_count() > 0 {
            let rdf = grafeo_core::graph::rdf::RdfStoreSection::new(Arc::clone(&self.rdf_store));
            sections.push(Box::new(rdf));
        }

        #[cfg(feature = "ring-index")]
        if self.rdf_store.ring().is_some() {
            let ring = grafeo_core::index::ring::RdfRingSection::new(Arc::clone(&self.rdf_store));
            sections.push(Box::new(ring));
        }

        sections
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
    /// Returns an error if the database has no file manager or I/O fails.
    #[cfg(all(feature = "wal", feature = "grafeo-file", feature = "lpg"))]
    pub fn backup_full(&self, backup_dir: &std::path::Path) -> Result<backup::BackupSegment> {
        let fm = self
            .file_manager
            .as_ref()
            .ok_or_else(|| Error::Internal("backup requires a persistent database".to_string()))?;

        // Checkpoint to ensure the container has the latest data.
        // Skip for read-only databases: the on-disk file is already a valid
        // snapshot and the file manager rejects writes.
        if !self.read_only {
            let _ = self.checkpoint_to_file(fm, flush::FlushReason::Explicit)?;
        }

        let current_epoch = self.transaction_manager.current_epoch();
        backup::do_backup_full(backup_dir, fm, self.wal.as_deref(), current_epoch)
    }

    /// Creates an incremental backup containing WAL records since the last backup.
    ///
    /// Requires a prior full backup in the backup directory.
    ///
    /// # Errors
    ///
    /// Returns an error if no full backup exists, or if the WAL has no new records.
    #[cfg(all(feature = "wal", feature = "grafeo-file", feature = "lpg"))]
    pub fn backup_incremental(
        &self,
        backup_dir: &std::path::Path,
    ) -> Result<backup::BackupSegment> {
        let wal = self
            .wal
            .as_ref()
            .ok_or_else(|| Error::Internal("incremental backup requires WAL".to_string()))?;

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
    /// # Errors
    ///
    /// Returns an error if the backup chain does not cover the target epoch,
    /// segment checksums fail, or I/O fails.
    #[cfg(all(feature = "wal", feature = "grafeo-file"))]
    pub fn restore_to_epoch(
        backup_dir: &std::path::Path,
        target_epoch: grafeo_common::types::EpochId,
        output_path: &std::path::Path,
    ) -> Result<()> {
        backup::do_restore_to_epoch(backup_dir, target_epoch, output_path)
    }

    /// Writes the current database state to the `.grafeo` file using the unified flush.
    ///
    /// Does NOT remove the sidecar WAL: callers that want to clean up
    /// the sidecar (e.g. `close()`) should call `fm.remove_sidecar_wal()`
    /// separately after this returns.
    #[cfg(feature = "grafeo-file")]
    fn checkpoint_to_file(
        &self,
        fm: &GrafeoFileManager,
        reason: flush::FlushReason,
    ) -> Result<flush::FlushResult> {
        let sections = self.build_sections();
        let section_refs: Vec<&dyn grafeo_common::storage::Section> =
            sections.iter().map(|s| s.as_ref()).collect();
        #[cfg(feature = "lpg")]
        let context = flush::build_context(self.lpg_store(), &self.transaction_manager);
        #[cfg(not(feature = "lpg"))]
        let context = flush::build_context_minimal(&self.transaction_manager);

        flush::flush(
            fm,
            &section_refs,
            &context,
            reason,
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
    /// GQLSTATUS code per ISO/IEC 39075:2024, sec 23.
    pub gql_status: grafeo_common::utils::GqlStatus,
}

impl QueryResult {
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
        }
    }

    /// Creates a query result with only a status message (for DDL commands).
    #[must_use]
    pub fn status(msg: impl Into<String>) -> Self {
        Self {
            columns: Vec::new(),
            column_types: Vec::new(),
            rows: Vec::new(),
            execution_time_ms: None,
            rows_scanned: None,
            status_message: Some(msg.into()),
            gql_status: grafeo_common::utils::GqlStatus::SUCCESS,
        }
    }

    /// Creates a new empty query result.
    #[must_use]
    pub fn new(columns: Vec<String>) -> Self {
        let len = columns.len();
        Self {
            columns,
            column_types: vec![grafeo_common::types::LogicalType::Any; len],
            rows: Vec::new(),
            execution_time_ms: None,
            rows_scanned: None,
            status_message: None,
            gql_status: grafeo_common::utils::GqlStatus::SUCCESS,
        }
    }

    /// Creates a new empty query result with column types.
    #[must_use]
    pub fn with_types(
        columns: Vec<String>,
        column_types: Vec<grafeo_common::types::LogicalType>,
    ) -> Self {
        Self {
            columns,
            column_types,
            rows: Vec::new(),
            execution_time_ms: None,
            rows_scanned: None,
            status_message: None,
            gql_status: grafeo_common::utils::GqlStatus::SUCCESS,
        }
    }

    /// Creates a query result with pre-populated rows.
    #[must_use]
    pub fn from_rows(columns: Vec<String>, rows: Vec<Vec<grafeo_common::types::Value>>) -> Self {
        let len = columns.len();
        Self {
            columns,
            column_types: vec![grafeo_common::types::LogicalType::Any; len],
            rows,
            execution_time_ms: None,
            rows_scanned: None,
            status_message: None,
            gql_status: grafeo_common::utils::GqlStatus::SUCCESS,
        }
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

// =============================================================================
// HybridStore integration
// =============================================================================

#[cfg(feature = "compact-store")]
impl GrafeoDB {
    /// Creates a database in hybrid mode from a frozen [`CompactStore`].
    ///
    /// The [`HybridStore`] combines a read-only columnar base (the supplied
    /// `compact`) with a mutable MVCC overlay. Reads merge both layers
    /// transparently; writes go to the overlay only. The compact base is
    /// never mutated after construction.
    ///
    /// # Errors
    ///
    /// Returns an error if config validation fails or the overlay store
    /// cannot be allocated.
    ///
    /// [`CompactStore`]: grafeo_core::graph::compact::CompactStore
    /// [`HybridStore`]: grafeo_core::graph::hybrid::HybridStore
    pub fn with_hybrid(
        compact: grafeo_core::graph::compact::CompactStore,
        config: Config,
    ) -> Result<Self> {
        use grafeo_core::graph::hybrid::HybridStore;

        let hybrid_store = HybridStore::open(compact)
            .map_err(|e| grafeo_common::utils::error::Error::Internal(e.to_string()))?;

        // Extract the overlay before wrapping in Arc so we can pass it to
        // sessions for MVCC finalization (finalize_version_epochs, sync_epoch).
        let overlay = Arc::clone(hybrid_store.overlay());

        let hybrid = Arc::new(hybrid_store);

        // HybridStore implements both GraphStore and GraphStoreMut, so we can
        // delegate directly to with_store().
        let mut db = Self::with_store(hybrid, config)?;

        // Store the overlay so sessions can call finalize_version_epochs on
        // commit, making auto-committed writes visible across sessions.
        db.hybrid_overlay = Some(overlay);

        Ok(db)
    }

    /// Freezes the current store data into a [`CompactStore`] and switches
    /// to hybrid mode.
    ///
    /// Takes a snapshot of all nodes and edges from the current store,
    /// builds a columnar [`CompactStore`], wraps it in a [`HybridStore`],
    /// and replaces the active stores. After this call the database accepts
    /// both read and write queries. The original store is dropped to free
    /// memory.
    ///
    /// This is the read-write counterpart to [`compact()`](Self::compact),
    /// which produces a read-only store.
    ///
    /// # Errors
    ///
    /// Returns an error if the conversion fails (e.g. more than 32,767
    /// distinct labels or edge types) or the new overlay cannot be allocated.
    ///
    /// [`CompactStore`]: grafeo_core::graph::compact::CompactStore
    /// [`HybridStore`]: grafeo_core::graph::hybrid::HybridStore
    pub fn compact_to_hybrid(&mut self) -> Result<()> {
        use grafeo_core::graph::compact::from_graph_store;
        use grafeo_core::graph::hybrid::HybridStore;

        let current_store = self.graph_store();
        let compact = from_graph_store(current_store.as_ref())
            .map_err(|e| grafeo_common::utils::error::Error::Internal(e.to_string()))?;

        let hybrid_store = HybridStore::open(compact)
            .map_err(|e| grafeo_common::utils::error::Error::Internal(e.to_string()))?;

        // Capture overlay for MVCC finalization in sessions.
        let overlay = Arc::clone(hybrid_store.overlay());
        let hybrid = Arc::new(hybrid_store);

        self.external_read_store = Some(Arc::clone(&hybrid) as Arc<dyn GraphStore>);
        self.external_write_store = Some(hybrid as Arc<dyn GraphStoreMut>);
        self.hybrid_overlay = Some(overlay);
        #[cfg(feature = "lpg")]
        {
            self.store = None;
        }
        self.read_only = false;
        self.query_cache = Arc::new(QueryCache::default());
        // Projections hold Arc refs to the old store: clear them so they don't
        // serve stale data or prevent the old store's memory from being freed.
        self.projections.write().clear();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[cfg(feature = "wal")]
    #[test]
    fn test_persistent_database_recovery() {
        use grafeo_common::types::Value;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        // Create database and add some data
        {
            let db = GrafeoDB::open(&db_path).unwrap();

            let alix = db.create_node(&["Person"]);
            db.set_node_property(alix, "name", Value::from("Alix"));

            let gus = db.create_node(&["Person"]);
            db.set_node_property(gus, "name", Value::from("Gus"));

            let _edge = db.create_edge(alix, gus, "KNOWS");

            // Explicitly close to flush WAL
            db.close().unwrap();
        }

        // Reopen and verify data was recovered
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

    #[cfg(feature = "wal")]
    #[test]
    fn test_wal_logging() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("wal_test_db");

        let db = GrafeoDB::open(&db_path).unwrap();

        // Create some data
        let node = db.create_node(&["Test"]);
        db.delete_node(node);

        // WAL should have records
        if let Some(wal) = db.wal() {
            assert!(wal.record_count() > 0);
        }

        db.close().unwrap();
    }

    #[cfg(feature = "wal")]
    #[test]
    fn test_wal_recovery_multiple_sessions() {
        // Tests that WAL recovery works correctly across multiple open/close cycles
        use grafeo_common::types::Value;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("multi_session_db");

        // Session 1: Create initial data
        {
            let db = GrafeoDB::open(&db_path).unwrap();
            let alix = db.create_node(&["Person"]);
            db.set_node_property(alix, "name", Value::from("Alix"));
            db.close().unwrap();
        }

        // Session 2: Add more data
        {
            let db = GrafeoDB::open(&db_path).unwrap();
            assert_eq!(db.node_count(), 1); // Previous data recovered
            let gus = db.create_node(&["Person"]);
            db.set_node_property(gus, "name", Value::from("Gus"));
            db.close().unwrap();
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

    #[cfg(feature = "wal")]
    #[test]
    fn test_database_consistency_after_mutations() {
        // Tests that database remains consistent after a series of create/delete operations
        use grafeo_common::types::Value;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("consistency_db");

        {
            let db = GrafeoDB::open(&db_path).unwrap();

            // Create nodes
            let a = db.create_node(&["Node"]);
            let b = db.create_node(&["Node"]);
            let c = db.create_node(&["Node"]);

            // Create edges
            let e1 = db.create_edge(a, b, "LINKS");
            let _e2 = db.create_edge(b, c, "LINKS");

            // Delete middle node and its edge
            db.delete_edge(e1);
            db.delete_node(b);

            // Set properties on remaining nodes
            db.set_node_property(a, "value", Value::Int64(1));
            db.set_node_property(c, "value", Value::Int64(3));

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

    #[cfg(feature = "wal")]
    #[test]
    fn test_close_is_idempotent() {
        // Calling close() multiple times should not cause errors
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("close_test_db");

        let db = GrafeoDB::open(&db_path).unwrap();
        db.create_node(&["Test"]);

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
        db.create_node(&["Person"]);
        db.create_node(&["Person"]);

        // Check that metrics snapshot returns data
        let snap = db.metrics();
        // Session created counter should reflect at least 0 (metrics is initialized)
        assert_eq!(snap.query_count, 0); // No queries executed yet
    }

    #[test]
    fn test_query_result_has_metrics() {
        // Verifies that query results include execution metrics
        let db = GrafeoDB::new_in_memory();
        db.create_node(&["Person"]);
        db.create_node(&["Person"]);

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
        db.create_node(&["Person"]);

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
            let id = db.create_node(&["Person"]);
            // Update
            db.set_node_property(id, "name", "Alix".into());
            db.set_node_property(id, "name", "Gus".into());
            // Delete
            db.delete_node(id);

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

            let alix = db.create_node(&["Person"]);
            let gus = db.create_node(&["Person"]);
            let edge = db.create_edge(alix, gus, "KNOWS");
            db.set_edge_property(edge, "since", 2024i64.into());
            db.delete_edge(edge);

            let history = db.history(edge).unwrap();
            assert_eq!(history.len(), 3); // create + update + delete
            assert_eq!(history[0].kind, crate::cdc::ChangeKind::Create);
            assert_eq!(history[1].kind, crate::cdc::ChangeKind::Update);
            assert_eq!(history[2].kind, crate::cdc::ChangeKind::Delete);
        }

        #[test]
        fn test_create_node_with_props_cdc() {
            let db = cdc_db();

            let id = db.create_node_with_props(
                &["Person"],
                vec![
                    ("name", grafeo_common::types::Value::from("Alix")),
                    ("age", grafeo_common::types::Value::from(30i64)),
                ],
            );

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

            let id1 = db.create_node(&["A"]);
            let _id2 = db.create_node(&["B"]);
            db.set_node_property(id1, "x", 1i64.into());

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

            let id = db.create_node(&["Person"]);
            db.set_node_property(id, "name", "Alix".into());

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

            let id = db.create_node(&["Person"]);
            let history = db.history(id).unwrap();
            assert_eq!(history.len(), 1, "CDC enabled at runtime records events");

            // Disable again
            db.set_cdc_enabled(false);
            let id2 = db.create_node(&["Person"]);
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
        let result = QueryResult::new(vec!["name".into(), "age".into()]);
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
        );
        assert_eq!(result.column_count(), 2);
        assert_eq!(result.column_types[0], LogicalType::String);
        assert_eq!(result.column_types[1], LogicalType::Int64);
    }

    #[test]
    fn test_query_result_with_metrics() {
        let result = QueryResult::new(vec!["x".into()]).with_metrics(42.5, 100);
        assert_eq!(result.execution_time_ms(), Some(42.5));
        assert_eq!(result.rows_scanned(), Some(100));
    }

    #[test]
    fn test_query_result_scalar_success() {
        use grafeo_common::types::Value;
        let mut result = QueryResult::new(vec!["count".into()]);
        result.rows.push(vec![Value::Int64(42)]);

        let val: i64 = result.scalar().unwrap();
        assert_eq!(val, 42);
    }

    #[test]
    fn test_query_result_scalar_wrong_shape() {
        use grafeo_common::types::Value;
        // Multiple rows
        let mut result = QueryResult::new(vec!["x".into()]);
        result.rows.push(vec![Value::Int64(1)]);
        result.rows.push(vec![Value::Int64(2)]);
        assert!(result.scalar::<i64>().is_err());

        // Multiple columns
        let mut result2 = QueryResult::new(vec!["a".into(), "b".into()]);
        result2.rows.push(vec![Value::Int64(1), Value::Int64(2)]);
        assert!(result2.scalar::<i64>().is_err());

        // Empty
        let result3 = QueryResult::new(vec!["x".into()]);
        assert!(result3.scalar::<i64>().is_err());
    }

    #[test]
    fn test_query_result_iter() {
        use grafeo_common::types::Value;
        let mut result = QueryResult::new(vec!["x".into()]);
        result.rows.push(vec![Value::Int64(1)]);
        result.rows.push(vec![Value::Int64(2)]);

        let collected: Vec<_> = result.iter().collect();
        assert_eq!(collected.len(), 2);
    }

    #[test]
    fn test_query_result_display() {
        use grafeo_common::types::Value;
        let mut result = QueryResult::new(vec!["name".into()]);
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
    fn test_database_adaptive_config() {
        let db = GrafeoDB::new_in_memory();
        let adaptive = db.adaptive_config();
        assert!(adaptive.enabled);
        assert!((adaptive.threshold - 3.0).abs() < f64::EPSILON);
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
        db.create_node(&["Person"]);
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
        assert!(db.drop_graph("temp"));
        assert!(!db.drop_graph("temp")); // Already dropped
    }

    #[test]
    fn test_drop_graph_resets_current_graph() {
        let db = GrafeoDB::new_in_memory();
        db.create_graph("active").unwrap();
        db.set_current_graph(Some("active")).unwrap();
        assert_eq!(db.current_graph(), Some("active".to_string()));

        db.drop_graph("active");
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
        db.create_node(&["Person"]);
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

        let read_store = Arc::clone(&store) as Arc<dyn GraphStore>;
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
        db.create_node(&["Person"]);

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
        db.create_node(&["Person"]);
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
        assert!(!prom.is_empty());
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
        let read_store = Arc::clone(&store) as Arc<dyn GraphStore>;
        let db = GrafeoDB::with_read_store(read_store, Config::in_memory()).unwrap();

        // drop_graph with external store (no built-in store) returns false
        assert!(!db.drop_graph("anything"));
    }
}
