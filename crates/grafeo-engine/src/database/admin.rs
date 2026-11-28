//! Admin, introspection, and diagnostic operations for GrafeoDB.

use std::path::Path;
use std::sync::Arc;

use grafeo_common::types::{EdgeId, ReadView};
use grafeo_common::utils::error::Result;
use grafeo_core::graph::GraphStore;
use grafeo_core::graph::rowgroup::RowGroupStore;

impl super::Open {
    // =========================================================================
    // ADMIN API: Counts
    // =========================================================================

    pub(crate) fn node_count(&self) -> Result<usize> {
        let snapshot = self.default_snapshot();
        snapshot.store().node_count(snapshot.view())
    }

    pub(crate) fn edge_count(&self) -> Result<usize> {
        let snapshot = self.default_snapshot();
        snapshot.store().edge_count(snapshot.view())
    }

    pub(crate) fn label_count(&self) -> Result<usize> {
        let snapshot = self.default_snapshot();
        Ok(snapshot.store().labels_in_use(snapshot.view())?.len())
    }

    pub(crate) fn property_key_count(&self) -> Result<usize> {
        // A snapshot restore waits for it, or it for the restore.
        let _read = self.transaction_manager.hold_direct_read();
        Ok(self.graph_store().all_property_keys()?.len())
    }

    pub(crate) fn edge_type_count(&self) -> Result<usize> {
        let snapshot = self.default_snapshot();
        Ok(snapshot.store().edge_types_in_use(snapshot.view())?.len())
    }

    // =========================================================================
    // ADMIN API: Introspection
    // =========================================================================

    #[must_use]
    pub(crate) fn is_persistent(&self) -> bool {
        self.config.path().is_some()
    }

    pub(crate) fn info(&self) -> Result<crate::admin::DatabaseInfo> {
        // One pinned state for both counts.
        let snapshot = self.default_snapshot();
        let (store, view) = (snapshot.store(), snapshot.view());
        Ok(crate::admin::DatabaseInfo {
            mode: crate::admin::DatabaseMode::Lpg,
            node_count: store.node_count(view)?,
            edge_count: store.edge_count(view)?,
            is_persistent: self.is_persistent(),
            path: self.config.path().map(Path::to_path_buf),
            wal_enabled: self.config.wal_enabled(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            features: {
                let mut f = vec!["gql".into()];
                if cfg!(feature = "cypher") {
                    f.push("cypher".into());
                }
                if cfg!(feature = "sparql") {
                    f.push("sparql".into());
                }
                if cfg!(feature = "gremlin") {
                    f.push("gremlin".into());
                }
                if cfg!(feature = "graphql") {
                    f.push("graphql".into());
                }
                if cfg!(feature = "sql-pgq") {
                    f.push("sql-pgq".into());
                }
                if cfg!(feature = "triple-store") {
                    f.push("rdf".into());
                }
                if cfg!(feature = "algos") {
                    f.push("algos".into());
                }
                if cfg!(feature = "vector-index") {
                    f.push("vector-index".into());
                }
                if cfg!(feature = "text-index") {
                    f.push("text-index".into());
                }
                if cfg!(feature = "hybrid-search") {
                    f.push("hybrid-search".into());
                }
                if cfg!(feature = "cdc") {
                    f.push("cdc".into());
                }
                f
            },
        })
    }

    #[must_use]
    pub(crate) fn memory_usage(&self) -> crate::memory_usage::MemoryUsage {
        use crate::memory_usage::{BufferManagerMemory, CacheMemory, MemoryUsage};
        use grafeo_common::memory::MemoryRegion;

        let (store, indexes, mvcc, string_pool) = self
            .root_store()
            .map_or_default(|store| store.counted_breakdown());

        let (parsed_bytes, optimized_bytes, cached_plan_count) =
            self.query_cache.heap_memory_bytes();
        let mut caches = CacheMemory {
            parsed_plan_cache_bytes: parsed_bytes,
            optimized_plan_cache_bytes: optimized_bytes,
            cached_plan_count,
            plan_cache_limit_bytes: self.query_cache.limit(),
            session_plan_cache_bytes: self.query_cache.session_bytes(),
            ..Default::default()
        };
        caches.compute_total();

        let bm_stats = self.buffer_manager.stats();
        let buffer_manager = BufferManagerMemory {
            budget_bytes: bm_stats.budget,
            allocated_bytes: bm_stats.total_allocated,
            hard_limit_bytes: self.buffer_manager.hard_limit(),
            graph_storage_bytes: bm_stats.region_usage(MemoryRegion::GraphStorage),
            index_buffers_bytes: bm_stats.region_usage(MemoryRegion::IndexBuffers),
            execution_buffers_bytes: bm_stats.region_usage(MemoryRegion::ExecutionBuffers),
            spill_staging_bytes: bm_stats.region_usage(MemoryRegion::SpillStaging),
        };

        let mut usage = MemoryUsage {
            store,
            indexes,
            mvcc,
            caches,
            string_pool,
            buffer_manager,
            ..Default::default()
        };

        #[cfg(feature = "triple-store")]
        {
            use crate::memory_usage::RdfMemory;
            let (
                triple_count,
                triples_and_indexes_bytes,
                term_dictionary_bytes,
                ring_index_bytes,
                named_graph_count,
            ) = self.rdf_store.counted_memory_bytes();
            usage.rdf = RdfMemory {
                triple_count,
                triples_and_indexes_bytes,
                term_dictionary_bytes,
                ring_index_bytes,
                named_graph_count,
                snapshot_bytes: self.rdf_store.snapshot_bytes()
                    + self.transaction_manager.kept_rdf_writes_bytes(),
                total_bytes: 0,
            };
            usage.rdf.compute_total();
        }

        #[cfg(feature = "cdc")]
        {
            use crate::memory_usage::CdcMemory;
            let (total_bytes, entity_count, event_count) = self.cdc_log.counted_memory_bytes();
            usage.cdc = CdcMemory {
                total_bytes,
                entity_count,
                event_count,
                granted_bytes: self.cdc_log.granted_bytes(),
                limit_bytes: self.cdc_log.max_bytes().unwrap_or(0),
            };
        }

        usage.limit_bytes = self.buffer_manager.hard_limit();
        usage.query_limit_bytes = self
            .config
            .query_memory_limit()
            .map_or(usage.limit_bytes, |query| query.min(usage.limit_bytes));
        usage.spill = self.spill_memory();
        usage.compute_total();
        usage
    }

    pub(crate) fn detailed_stats(&self) -> Result<crate::admin::DatabaseStats> {
        #[cfg(feature = "wal")]
        let disk_bytes = self.config.path().as_ref().and_then(|p| {
            // The spelling the files are named after: the caller's (`db/`)
            // no longer exists once a migrated directory became a file.
            let path = super::normalize_path(p).ok()?;
            if path.exists() {
                Self::calculate_disk_usage(&path).ok()
            } else {
                None
            }
        });
        #[cfg(not(feature = "wal"))]
        let disk_bytes: Option<usize> = None;

        // One pinned state for every count.
        let snapshot = self.default_snapshot();
        let (store, view) = (snapshot.store(), snapshot.view());
        Ok(crate::admin::DatabaseStats {
            node_count: store.node_count(view)?,
            edge_count: store.edge_count(view)?,
            label_count: store.labels_in_use(view)?.len(),
            edge_type_count: store.edge_types_in_use(view)?.len(),
            property_key_count: store.all_property_keys()?.len(),
            index_count: self.catalog.index_count(),
            memory_bytes: self.memory_usage().total_bytes,
            disk_bytes,
        })
    }

    /// Calculates the disk usage of the database at `path`: its file and its
    /// sidecar WAL directory `<path>.wal/`, or every file of a 0.5.x WAL
    /// directory read in place.
    #[cfg(feature = "wal")]
    fn calculate_disk_usage(path: &Path) -> Result<usize> {
        // The spelling the database's files are named after (`path()` keeps
        // the caller's).
        let path = &super::normalize_path(path)?;
        if path.is_dir() {
            return Self::directory_size(path);
        }
        let file = if path.is_file() {
            Self::file_size(path, &std::fs::metadata(path)?)?
        } else {
            0
        };
        let wal = Self::directory_size(&grafeo_storage::file::detect::sidecar_wal_path(path))?;
        Ok(file.saturating_add(wal))
    }

    /// The total size of the files under `dir` (0 when it does not exist),
    /// without the WAL segments still being created (`*.log.tmp`), which are
    /// never part of the database.
    #[cfg(feature = "wal")]
    fn directory_size(dir: &Path) -> Result<usize> {
        let mut total = 0usize;
        if dir.is_dir() {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                if entry
                    .file_name()
                    .to_str()
                    .is_some_and(grafeo_storage::wal::is_segment_temp_file_name)
                {
                    continue;
                }
                // From the file itself: on Windows a listing's size is stale
                // while the file is open (the active WAL segment).
                let metadata = std::fs::metadata(entry.path())?;
                if metadata.is_file() {
                    total = total.saturating_add(Self::file_size(&entry.path(), &metadata)?);
                } else if metadata.is_dir() {
                    total = total.saturating_add(Self::directory_size(&entry.path())?);
                }
            }
        }
        Ok(total)
    }

    /// The size of the file at `path` as a `usize`.
    #[cfg(feature = "wal")]
    fn file_size(path: &Path, metadata: &std::fs::Metadata) -> Result<usize> {
        usize::try_from(metadata.len()).map_err(|_| {
            grafeo_common::utils::error::Error::Internal(format!(
                "the size of {} ({} bytes) does not fit in a usize on this platform",
                path.display(),
                metadata.len()
            ))
        })
    }

    pub(crate) fn schema(&self) -> Result<crate::admin::SchemaInfo> {
        let snapshot = self.default_snapshot();
        let summary = crate::admin::LpgSchemaInfo::in_use(snapshot.store(), snapshot.view())?;
        Ok(crate::admin::SchemaInfo::Lpg(summary))
    }

    #[must_use]
    pub(crate) fn list_indexes(&self) -> Vec<crate::admin::IndexInfo> {
        // A snapshot restore waits for it, or it for the restore.
        let _read = self.transaction_manager.hold_direct_read();
        let store = self.current_lpg_store().ok();
        super::index::listed_indexes(&self.catalog, store.as_deref(), |target| {
            store
                .as_deref()
                .is_some_and(|store| super::index::store_has(store, target))
        })
        .into_iter()
        .map(|index| crate::admin::IndexInfo {
            name: index.name.unwrap_or_default(),
            index_type: format!("{:?}", index.kind),
            target: format!("{}:{}", index.label.unwrap_or_default(), index.property),
            unique: false,
            cardinality: None,
            size_bytes: None,
        })
        .collect()
    }

    #[doc(hidden)]
    #[must_use]
    pub(crate) fn graph_type_scope_warnings(&self) -> Vec<crate::admin::ValidationWarning> {
        // A snapshot restore waits for it, or it for the restore.
        let _read = self.transaction_manager.hold_direct_read();
        type Listed<'a> = &'a dyn Fn(&crate::catalog::GraphTypeDefinition) -> bool;
        type Held<'a> = &'a dyn Fn(&grafeo_core::graph::rowgroup::RowGroupStore) -> bool;

        let mut warnings = Vec::new();
        let graph_types = self.catalog.all_graph_type_defs();
        let Some(root) = self.root_store().filter(|_| !graph_types.is_empty()) else {
            return warnings;
        };
        let bindings = self.catalog.all_graph_type_bindings();
        let graph_names = root.graph_names();
        // Whether `graph` has a graph type that lists the type.
        let bound = |graph: &str, listed: Listed<'_>| {
            bindings.iter().any(|(bound_graph, type_name)| {
                bound_graph == graph
                    && graph_types
                        .iter()
                        .any(|def| def.name == *type_name && listed(def))
            })
        };
        // The first graph of the type's schema, not bound to it, holding what `held` counts.
        let unbound_holder = |name: &str, listed: Listed<'_>, held: Held<'_>| {
            let in_schema = name.rsplit_once('/').map(|(schema, _)| schema);
            if in_schema.is_none() && held(&root) {
                return Some("the default graph".to_string());
            }
            graph_names.iter().find_map(|graph| {
                let holds = graph.rsplit_once('/').map(|(schema, _)| schema) == in_schema
                    && !bound(graph, listed)
                    && root.graph(graph).is_some_and(|store| held(&store));
                holds.then(|| format!("graph '{graph}'"))
            })
        };
        let mut warn = |kind: &str, name: &str, graph_type: &str, graph: String| {
            let short = crate::catalog::unqualified(name);
            warnings.push(crate::admin::ValidationWarning {
                code: "GRAPH_TYPE_NOT_BOUND".to_string(),
                message: format!(
                    "{kind} type '{short}' is listed by graph type '{graph_type}', so its rules \
                     apply only in graphs bound to that type, and {graph} holds some: to keep \
                     them enforced there, use a graph typed by it, or take the type out of the \
                     graph type, declare it with CREATE {} TYPE and list it again, which keeps it \
                     a rule by its label; validate() names the nodes that would break it",
                    kind.to_uppercase()
                ),
                context: Some(format!("{kind}_type:{short}")),
            });
        };
        let mut warned = std::collections::HashSet::new();
        for def in &graph_types {
            if crate::catalog::is_current_graph_type(&def.name) {
                continue;
            }
            for name in &def.allowed_node_types {
                // A standalone type a graph type lists applies in every graph (#764).
                if !warned.insert(("node", name.clone()))
                    || self
                        .catalog
                        .get_node_type(name)
                        .is_some_and(|def| def.standalone)
                {
                    continue;
                }
                let label = crate::catalog::unqualified(name);
                let holder = unbound_holder(
                    name,
                    &|other| other.allowed_node_types.contains(name),
                    &|store| store.label_node_estimate(label) > 0,
                );
                if let Some(graph) = holder {
                    warn("node", name, &def.name, graph);
                }
            }
            for name in &def.allowed_edge_types {
                if !warned.insert(("edge", name.clone()))
                    || self
                        .catalog
                        .get_edge_type_def(name)
                        .is_some_and(|def| def.standalone)
                {
                    continue;
                }
                let edge_type = crate::catalog::unqualified(name);
                let holder = unbound_holder(
                    name,
                    &|other| other.allowed_edge_types.contains(name),
                    &|store| {
                        store
                            .statistics()
                            .get_edge_type(edge_type)
                            .is_some_and(|counted| counted.edge_count > 0)
                    },
                );
                if let Some(graph) = holder {
                    warn("edge", name, &def.name, graph);
                }
            }
        }
        warnings
    }

    /// The nodes that would break a listed node type in a graph not bound to a graph type that
    /// lists it (#764): for each type and graph that `GRAPH_TYPE_NOT_BOUND` covers, its nodes of
    /// the type's label checked against the type, one warning with the count, the first ids and
    /// the steps to make the type a rule by its label again. A file of an older release cannot
    /// tell a standalone type a graph type took over from one it declared, so both are checked.
    /// On demand: O(nodes of the label) per such graph, read at `view`.
    ///
    /// # Errors
    ///
    /// A read that fails.
    fn listed_type_row_warnings(
        &self,
        view: ReadView,
    ) -> Result<Vec<crate::admin::ValidationWarning>> {
        /// The ids a warning names at most.
        const NAMED: usize = 10;

        let mut warnings = Vec::new();
        let mut graph_types: Vec<_> = self
            .catalog
            .all_graph_type_defs()
            .into_iter()
            .filter(|def| !crate::catalog::is_current_graph_type(&def.name))
            .collect();
        let Some(root) = self.root_store().filter(|_| !graph_types.is_empty()) else {
            return Ok(warnings);
        };
        graph_types.sort_by(|a, b| a.name.cmp(&b.name));
        let bindings = self.catalog.all_graph_type_bindings();
        let mut graph_names = root.graph_names();
        graph_names.sort();
        let mut names: Vec<&String> = graph_types
            .iter()
            .flat_map(|def| &def.allowed_node_types)
            .collect();
        names.sort();
        names.dedup();
        for name in names {
            let Some(def) = self
                .catalog
                .resolved_node_type(name)
                .filter(|def| !def.standalone)
            else {
                continue;
            };
            let listing: Vec<&str> = graph_types
                .iter()
                .filter(|graph_type| graph_type.allowed_node_types.contains(name))
                .map(|graph_type| graph_type.name.as_str())
                .collect();
            let bound = |graph: &str| {
                bindings.iter().any(|(bound_graph, graph_type)| {
                    bound_graph == graph && listing.contains(&graph_type.as_str())
                })
            };
            let in_schema = name.rsplit_once('/').map(|(schema, _)| schema);
            let mut graphs: Vec<(String, Arc<RowGroupStore>)> = Vec::new();
            if in_schema.is_none() {
                graphs.push(("the default graph".to_string(), Arc::clone(&root)));
            }
            for graph in &graph_names {
                if graph.rsplit_once('/').map(|(schema, _)| schema) == in_schema
                    && !bound(graph)
                    && let Some(store) = root.graph(graph)
                {
                    graphs.push((format!("graph '{graph}'"), store));
                }
            }
            let label = crate::catalog::unqualified(name);
            let graph_type = crate::catalog::unqualified(listing[0]);
            for (graph, store) in graphs {
                let mut breaking = Vec::new();
                for id in store.nodes_by_label(label, view)? {
                    let Some(node) = store.node(id, view)? else {
                        continue;
                    };
                    let property = |key: &str| {
                        node.properties
                            .get(&grafeo_common::types::PropertyKey::from(key))
                    };
                    if crate::catalog::breaks_node_type(&def, property) {
                        breaking.push(id.0);
                    }
                }
                if breaking.is_empty() {
                    continue;
                }
                let count = breaking.len();
                breaking.sort_unstable();
                let ids: Vec<String> = breaking
                    .iter()
                    .take(NAMED)
                    .map(ToString::to_string)
                    .collect();
                let more = if count > NAMED { ", ..." } else { "" };
                warnings.push(crate::admin::ValidationWarning {
                    code: "LISTED_TYPE_ROWS".to_string(),
                    message: format!(
                        "{count} nodes of {graph} would break node type '{label}' if it applied \
                         there (ids {}{more}): graph type '{graph_type}' lists it, so it applies \
                         only in graphs bound to that type. To make it a rule by its label in \
                         every graph again: fix or remove these nodes; drop or unbind the graphs \
                         typed by '{graph_type}'; DROP GRAPH TYPE {graph_type} (it drops '{label}' \
                         when no other graph type lists it); CREATE NODE TYPE {label} with its \
                         properties; CREATE GRAPH TYPE {graph_type} listing NODE TYPE {label} \
                         again; bind the graphs again",
                        ids.join(", ")
                    ),
                    context: Some(format!("node_type:{label}")),
                });
            }
        }
        Ok(warnings)
    }

    pub(crate) fn validate(&self) -> Result<crate::admin::ValidationResult> {
        let mut result = crate::admin::ValidationResult::default();
        // The graph as queries see it at the current epoch, one pinned state
        // for every check.
        let snapshot = self.default_snapshot();
        let (store, view) = (snapshot.store(), snapshot.view());

        // Check for dangling edge references: every edge row of the built-in
        // store the view sees, against the nodes it sees. An external store's
        // reads list only the edges whose ends it sees, so its edges cannot
        // be checked this way: the report says so.
        match self.root_store() {
            Some(root) => {
                for id in 0..root.next_edge_id() {
                    let Some(edge) = root.edge(EdgeId::new(id), view)? else {
                        continue;
                    };
                    if !store.is_node_visible(edge.src, view)? {
                        result.errors.push(crate::admin::ValidationError {
                            code: "DANGLING_SRC".to_string(),
                            message: format!(
                                "Edge {} references non-existent source node {}",
                                edge.id.0, edge.src.0
                            ),
                            context: Some(format!("edge:{}", edge.id.0)),
                        });
                    }
                    if !store.is_node_visible(edge.dst, view)? {
                        result.errors.push(crate::admin::ValidationError {
                            code: "DANGLING_DST".to_string(),
                            message: format!(
                                "Edge {} references non-existent destination node {}",
                                edge.id.0, edge.dst.0
                            ),
                            context: Some(format!("edge:{}", edge.id.0)),
                        });
                    }
                }
            }
            None => result.warnings.push(crate::admin::ValidationWarning {
                code: "EXTERNAL_EDGES_UNCHECKED".to_string(),
                message: "the edges of an external store are not checked for dangling ends: its reads list only the edges whose ends it sees"
                    .to_string(),
                context: None,
            }),
        }

        // Add warnings for potential issues
        if store.node_count(view)? > 0 && store.edge_count(view)? == 0 {
            result.warnings.push(crate::admin::ValidationWarning {
                code: "NO_EDGES".to_string(),
                message: "Database has nodes but no edges".to_string(),
                context: None,
            });
        }
        result.warnings.extend(self.graph_type_scope_warnings());
        result.warnings.extend(self.listed_type_row_warnings(view)?);

        Ok(result)
    }

    /// When the file's last checkpoint was written, in seconds since the Unix
    /// epoch; `None` before the first one, and without a file.
    fn last_checkpoint_seconds(&self) -> Option<u64> {
        #[cfg(feature = "grafeo-file")]
        {
            let header = self.file_manager.as_ref()?.active_header();
            (header.iteration > 0).then_some(header.timestamp_ms / 1000)
        }
        #[cfg(not(feature = "grafeo-file"))]
        None
    }

    pub(crate) fn set_checkpoint_triggers(
        &self,
        triggers: crate::admin::CheckpointTriggers,
    ) -> Result<()> {
        // The floor `Config::validate` applies at open, with its error.
        crate::config::check_checkpoint_wal_size(triggers.wal_bytes)
            .map_err(super::config_error)?;
        if triggers.interval.is_some_and(|interval| interval.is_zero()) {
            return Err(grafeo_common::utils::error::Error::InvalidValue(
                "a checkpoint trigger of zero would checkpoint without pause: leave it unset \
                 to turn it off"
                    .to_string(),
            ));
        }
        let _open = self.hold_open()?;
        if self.read_only {
            return Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::ReadOnly,
            ));
        }
        #[cfg(feature = "grafeo-file")]
        if self.file_manager.is_some() {
            *self.checkpoint_triggers.lock() = triggers;
            self.restart_checkpoint_timer();
            return Ok(());
        }
        Err(grafeo_common::utils::error::Error::InvalidValue(
            "a database without a file has nothing to checkpoint".to_string(),
        ))
    }

    #[must_use]
    pub(crate) fn checkpoint_status(&self) -> crate::admin::CheckpointStatus {
        let status = crate::admin::CheckpointStatus {
            last_checkpoint: self.last_checkpoint_seconds(),
            ..crate::admin::CheckpointStatus::default()
        };
        #[cfg(feature = "grafeo-file")]
        let status = {
            let failures = self.checkpoint_failures.snapshot();
            crate::admin::CheckpointStatus {
                triggers: *self.checkpoint_triggers.lock(),
                failures: failures.failures,
                last_error: failures.last_error,
                last_failure: failures.last_failure,
                stopped: failures.stopped,
                ..status
            }
        };
        status
    }

    #[must_use]
    pub(crate) fn wal_status(&self) -> crate::admin::WalStatus {
        #[cfg(feature = "wal")]
        if let Some(ref wal) = self.wal {
            return crate::admin::WalStatus {
                enabled: true,
                // The sidecar WAL of the database file, `<path>.wal/`.
                path: Some(wal.dir().to_path_buf()),
                // The log after the last checkpoint: what a reopen replays.
                size_bytes: usize::try_from(wal.end_lsn().saturating_sub(self.checkpoint_lsn()))
                    .unwrap_or(usize::MAX),
                record_count: usize::try_from(
                    wal.records_written().saturating_sub(
                        self.wal_records_at_checkpoint
                            .load(std::sync::atomic::Ordering::Acquire),
                    ),
                )
                .unwrap_or(usize::MAX),
                last_checkpoint: self.last_checkpoint_seconds(),
                current_epoch: self.read_epoch().as_u64(),
            };
        }

        crate::admin::WalStatus {
            enabled: false,
            path: None,
            size_bytes: 0,
            record_count: 0,
            last_checkpoint: None,
            current_epoch: self.read_epoch().as_u64(),
        }
    }

    pub(crate) fn wal_checkpoint(&self) -> Result<()> {
        self.checkpoint_now().map(|_| ())
    }

    /// Writes a checkpoint of the database file, as
    /// [`wal_checkpoint()`](Self::wal_checkpoint) does, and returns whether
    /// it wrote one: not for a read-only database, nor for one without a
    /// file.
    ///
    /// # Errors
    ///
    /// The errors of [`wal_checkpoint()`](Self::wal_checkpoint).
    pub(super) fn checkpoint_now(&self) -> Result<bool> {
        // `close()` waits for the checkpoint, and none runs after it.
        let _open = self.hold_open()?;
        // Read-only databases have no WAL and the on-disk file is already a
        // valid snapshot: nothing to checkpoint.
        if self.read_only {
            return Ok(false);
        }
        // The store holds the stamped part of a commit that did not complete.
        self.transaction_manager.check_no_incomplete_commit()?;

        // A database on a backend of its own checkpoints through the backend,
        // and drops the log the backend holds durably.
        #[cfg(feature = "wal")]
        if let Some(backend) = self.external_write_store.as_ref() {
            if self.wal.is_none() {
                return Ok(false);
            }
            self.checkpoint_backend(backend)?;
            return Ok(true);
        }

        // Flush all sections to the .grafeo file. The flush marks and truncates
        // the WAL only once the file is durable (#417).
        #[cfg(feature = "grafeo-file")]
        if let Some(ref fm) = self.file_manager {
            self.checkpoint_to_file(fm)?;
            return Ok(true);
        }

        Ok(false)
    }

    // =========================================================================
    // ADMIN API: Change Data Capture
    // =========================================================================

    /// What the open has spilled, from its spill account: no walk.
    fn spill_memory(&self) -> crate::memory_usage::SpillMemory {
        use grafeo_core::execution::spill::{SpillCap, SpillOwner};

        let mut spill = crate::memory_usage::SpillMemory::default();
        match &self.spill_account {
            Some(account) => {
                spill.total_bytes = account.used();
                spill.limit_bytes = match account.cap() {
                    SpillCap::Bytes(bytes) => Some(bytes),
                    SpillCap::Unlimited => None,
                };
                spill.sort_bytes = account.used_by(SpillOwner::Sort);
                spill.aggregate_bytes = account.used_by(SpillOwner::Aggregate);
                spill.vector_column_bytes = account.used_by(SpillOwner::VectorColumns);
            }
            // No spill directory: nothing spills.
            None => spill.limit_bytes = Some(0),
        }
        spill
    }

    #[doc(hidden)]
    #[cfg(feature = "cdc")]
    #[must_use]
    pub(crate) fn cdc_log(&self) -> &std::sync::Arc<crate::cdc::CdcLog> {
        &self.cdc_log
    }

    #[cfg(feature = "cdc")]
    #[must_use]
    pub(crate) fn is_cdc_enabled(&self) -> bool {
        self.cdc_active()
    }

    #[cfg(feature = "cdc")]
    pub(crate) fn set_cdc_enabled(&self, enabled: bool) {
        self.cdc_enabled
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    #[cfg(feature = "cdc")]
    pub(crate) fn set_cdc(&self, enabled: bool) -> Result<()> {
        let _open = self.hold_open()?;
        if self.read_only {
            return Err(grafeo_common::utils::error::Error::from(
                grafeo_common::utils::error::QueryError::unsupported(
                    "a database opened read-only cannot change its change data capture setting",
                ),
            ));
        }
        if self.root_store().is_none() {
            return Err(super::config_error(
                crate::config::ConfigError::CdcWithoutFile,
            ));
        }
        #[cfg(feature = "grafeo-file")]
        if let Some(fm) = self.file_manager.as_ref() {
            // A stored feed is durable only through the WAL: without one,
            // commits would leave a gap in it.
            #[cfg(feature = "wal")]
            let without_wal = self.wal.is_none();
            #[cfg(not(feature = "wal"))]
            let without_wal = true;
            if enabled && without_wal {
                return Err(super::config_error(
                    crate::config::ConfigError::CdcWithoutWal,
                ));
            }
            let sources = self.checkpoint_sources()?;
            // Commits are held from the change to its checkpoint at log
            // position L: every group after L was written after the change,
            // so the cursor at L misses nothing and holds nothing from before.
            let _checkpoint = fm.checkpoint_guard();
            // After a checkpoint whose outcome is unknown nothing more is
            // stored: refused before anything changes.
            fm.check_checkpoint_allowed()?;
            let commits = self.transaction_manager.hold_commits()?;
            // Turning it on writes the cursor first, at L: the WAL's end while
            // commits are held, where the checkpoint starts its new segment.
            // A cursor that fails to be written leaves everything as it was.
            // One that is there while the setting is stored on stays where it
            // is: it holds the changes its consumer has not read yet.
            #[cfg(feature = "wal")]
            let wrote_cursor = match self.wal.as_ref() {
                Some(wal)
                    if enabled
                        && !(self
                            .catalog
                            .cdc_config()
                            .is_some_and(|record| record.enabled)
                            && super::cdc_cursor::default_exists(wal.dir())) =>
                {
                    super::cdc_cursor::write_default(wal.dir(), wal.end_lsn(), false)?;
                    true
                }
                _ => false,
            };
            // The clock is filled in when the record is read (`cdc_config`).
            let record =
                grafeo_common::storage::catalog_record::CdcConfigRecord { enabled, clock: 0 };
            let previous_record = self.catalog.set_cdc_config(Some(record));
            // Capture goes on at once, and off only once the setting is
            // stored: the commits the checkpoint lets go on while it writes
            // are captured whichever way it ends.
            let previous = if enabled {
                self.cdc_enabled
                    .swap(true, std::sync::atomic::Ordering::Relaxed)
            } else {
                self.cdc_enabled.load(std::sync::atomic::Ordering::Relaxed)
            };
            if let Err(error) = super::flush::flush_held(fm, &sources, None, commits) {
                // A checkpoint whose outcome is unknown may have stored it:
                // the new setting stays, the side that keeps the WAL (on
                // keeps its cursor; off removes none), as a reopen may find it.
                if super::flush::is_outcome_unknown(&error) {
                    self.cdc_enabled
                        .store(enabled, std::sync::atomic::Ordering::Relaxed);
                    return Err(error);
                }
                // Nothing was stored: the setting goes back to what the file
                // holds, and the cursor written for it goes too (one left
                // behind is removed at the next open).
                self.catalog.set_cdc_config(previous_record);
                self.cdc_enabled
                    .store(previous, std::sync::atomic::Ordering::Relaxed);
                #[cfg(feature = "wal")]
                if wrote_cursor
                    && let Some(wal) = self.wal.as_ref()
                    && let Err(cursor_error) = super::cdc_cursor::remove_default(wal.dir())
                {
                    grafeo_common::grafeo_warn!(
                        "the CDC cursor of a change that was not stored stays until the next \
                         open: {cursor_error}"
                    );
                }
                return Err(error);
            }
            self.cdc_enabled
                .store(enabled, std::sync::atomic::Ordering::Relaxed);
            grafeo_common::testing::crash::maybe_crash("cdc:after_toggle_checkpoint");
            // Turning it off removes the cursor once the setting is stored
            // (again when it is off: a removal that failed before is retried).
            // Turning it on has no step after the checkpoint: its cursor was
            // written first.
            #[cfg(feature = "wal")]
            if !enabled && let Some(wal) = self.wal.as_ref() {
                super::cdc_cursor::remove_default(wal.dir()).map_err(|error| {
                    grafeo_common::utils::error::Error::from(
                        grafeo_common::utils::error::StorageError::CdcCursorPending(
                            error.to_string(),
                        ),
                    )
                })?;
            }
            return Ok(());
        }
        // An in-memory database has nothing to reopen: the setting lasts while it is open.
        self.set_cdc_enabled(enabled);
        Ok(())
    }

    #[cfg(feature = "cdc")]
    pub(crate) fn history(
        &self,
        entity_id: impl Into<crate::cdc::EntityId>,
    ) -> Result<Vec<crate::cdc::ChangeEvent>> {
        let epoch = self.read_epoch();
        let mut events = self.cdc_log.history(entity_id.into());
        events.retain(|event| event.epoch <= epoch);
        Ok(events)
    }

    #[cfg(feature = "cdc")]
    pub(crate) fn history_since(
        &self,
        entity_id: impl Into<crate::cdc::EntityId>,
        since_epoch: grafeo_common::types::EpochId,
    ) -> Result<Vec<crate::cdc::ChangeEvent>> {
        let epoch = self.read_epoch();
        let mut events = self.cdc_log.history_since(entity_id.into(), since_epoch);
        events.retain(|event| event.epoch <= epoch);
        Ok(events)
    }

    #[cfg(feature = "cdc")]
    pub(crate) fn changes_between(
        &self,
        start_epoch: grafeo_common::types::EpochId,
        end_epoch: grafeo_common::types::EpochId,
    ) -> Result<Vec<crate::cdc::ChangeEvent>> {
        let end_epoch = end_epoch.min(self.read_epoch());
        Ok(self.cdc_log.changes_between(start_epoch, end_epoch))
    }
}

// The database's public methods: each reaches the open state for its call (#712).
impl super::GrafeoDB {
    /// Returns the number of nodes in the database, as of the current epoch
    /// (see [`current_epoch`](Self::current_epoch)).
    ///
    /// # Errors
    ///
    /// The error of a read the store cannot do (an external store's own).
    pub fn node_count(&self) -> Result<usize> {
        self.live()?.node_count()
    }

    /// Returns the number of edges in the database, as of the current epoch
    /// (see [`current_epoch`](Self::current_epoch)).
    ///
    /// # Errors
    ///
    /// As [`node_count`](Self::node_count).
    pub fn edge_count(&self) -> Result<usize> {
        self.live()?.edge_count()
    }

    /// Returns the number of distinct labels in the database.
    ///
    /// # Errors
    ///
    /// As [`node_count`](Self::node_count).
    pub fn label_count(&self) -> Result<usize> {
        self.live()?.label_count()
    }

    /// Returns the number of distinct property keys ever written to the
    /// database, of nodes and edges together: not the keys in use now. A key
    /// stays counted after every value of it is removed or deleted, and
    /// after the transaction that wrote it rolled back.
    ///
    /// # Errors
    ///
    /// As [`node_count`](Self::node_count).
    pub fn property_key_count(&self) -> Result<usize> {
        self.live()?.property_key_count()
    }

    /// Returns the number of distinct edge types in the database.
    ///
    /// # Errors
    ///
    /// As [`node_count`](Self::node_count).
    pub fn edge_type_count(&self) -> Result<usize> {
        self.live()?.edge_type_count()
    }

    /// Returns true if this database is backed by a file (persistent).
    ///
    /// In-memory databases return false.
    #[must_use]
    pub fn is_persistent(&self) -> bool {
        // From the configuration it was opened with: it answers after `close()` too.
        self.config.path().is_some()
    }

    /// Returns the database file path, if persistent.
    ///
    /// In-memory databases return None.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.config.path()
    }

    /// Returns high-level database information.
    ///
    /// Includes node/edge counts, persistence status, and mode (LPG/RDF).
    ///
    /// # Errors
    ///
    /// As [`node_count`](Self::node_count).
    pub fn info(&self) -> Result<crate::admin::DatabaseInfo> {
        self.live()?.info()
    }

    /// Returns a hierarchical memory usage breakdown. After `close()` it returns a report of zero
    /// bytes; it returns a `Result` from 0.7.0.
    ///
    /// The store, its indexes, its versions and its dictionaries come from
    /// the store's account (counted as they grow, so the report takes time
    /// independent of the store's size), the rest from the caches, string
    /// pools and buffer manager, each as estimated heap bytes. Safe to call
    /// concurrently with queries. On a database on a backend of its own the
    /// store, its indexes and its versions are the backend's, which the
    /// engine does not measure: they count as zero, and the rest (caches,
    /// buffers, the catalog) is the engine's own.
    #[must_use]
    pub fn memory_usage(&self) -> crate::memory_usage::MemoryUsage {
        self.live_if()
            .map_or_else(Default::default, |open| open.memory_usage())
    }

    /// Returns detailed database statistics.
    ///
    /// Includes counts, memory usage, and index information.
    ///
    /// # Errors
    ///
    /// As [`node_count`](Self::node_count).
    pub fn detailed_stats(&self) -> Result<crate::admin::DatabaseStats> {
        self.live()?.detailed_stats()
    }

    /// Returns schema information: the labels, edge types and property keys
    /// in use, each label and edge type with its count and its properties'
    /// value types, and the labels each edge type connects.
    ///
    /// It reads the whole graph, as a committed reader at the current epoch
    /// sees it, all of the report in one pinned state.
    ///
    /// # Errors
    ///
    /// As [`node_count`](Self::node_count).
    pub fn schema(&self) -> Result<crate::admin::SchemaInfo> {
        self.live()?.schema()
    }

    /// Returns detailed information about the current graph's indexes: the
    /// ones `CREATE INDEX` made, by the name it gave them, as `SHOW INDEXES`
    /// lists them, and the ones made through the API, whose `name` is empty.
    /// A name whose index only another graph has is not listed. The target
    /// of a property index made through the API, which covers every label,
    /// has an empty label (`:age`). After `close()` it returns an empty list; it returns a `Result`
    /// from 0.7.0.
    #[must_use]
    pub fn list_indexes(&self) -> Vec<crate::admin::IndexInfo> {
        self.live_if()
            .map_or_else(Default::default, |open| open.list_indexes())
    }

    /// The types whose rules an older release applied in graphs not bound to their graph
    /// type: since 0.6.2 a type a graph type lists applies only in its graphs, so a database
    /// written before may hold elements those rules no longer check. One warning per type,
    /// from the counts the store keeps, so no scan. For [`validate`](Self::validate), and
    /// for a binding to tell at open. After `close()` it returns an empty list; it returns a
    /// `Result` from 0.7.0.
    #[doc(hidden)]
    #[must_use]
    pub fn graph_type_scope_warnings(&self) -> Vec<crate::admin::ValidationWarning> {
        self.live_if()
            .map_or_else(Default::default, |open| open.graph_type_scope_warnings())
    }

    /// Validates database integrity.
    ///
    /// Checks for:
    /// - Dangling edge references (edges pointing to non-existent nodes)
    /// - Internal index consistency
    ///
    /// Returns a list of errors and warnings. Empty errors = valid. Reads
    /// the database as of the current epoch.
    ///
    /// # Errors
    ///
    /// As [`node_count`](Self::node_count): a read that fails is an error,
    /// not a finding.
    pub fn validate(&self) -> Result<crate::admin::ValidationResult> {
        self.live()?.validate()
    }

    /// Changes when this database checkpoints by itself (#426): when the
    /// interval has passed since its last automatic checkpoint, or once the
    /// WAL holds the given bytes since the last checkpoint. The new triggers
    /// replace the ones in force (from
    /// [`Config::with_checkpoint_interval`](crate::Config::with_checkpoint_interval)
    /// and [`Config::with_checkpoint_wal_size`](crate::Config::with_checkpoint_wal_size),
    /// or an earlier call), until the database closes; the file does not
    /// store them. [`CheckpointTriggers::none`](crate::admin::CheckpointTriggers::none)
    /// turns automatic checkpoints off.
    ///
    /// # Errors
    ///
    /// A zero interval, a size below 1 MiB (the error an open gives for
    /// such a `Config::with_checkpoint_wal_size`), a database without a file
    /// (in memory) or opened read-only, which never checkpoints, and the
    /// database-closed error after `close()`.
    pub fn set_checkpoint_triggers(
        &self,
        triggers: crate::admin::CheckpointTriggers,
    ) -> Result<()> {
        self.live()?.set_checkpoint_triggers(triggers)
    }

    /// How the automatic checkpoints fare: the triggers in force, the last
    /// checkpoint of any kind, and the automatic ones that failed since the
    /// last that succeeded, with the last error. A failed automatic
    /// checkpoint is also logged as a warning, and retried after a pause. After `close()` it
    /// returns the default status (no checkpoint, no failure, not stopped); it returns a `Result`
    /// from 0.7.0.
    #[must_use]
    pub fn checkpoint_status(&self) -> crate::admin::CheckpointStatus {
        self.live_if()
            .map_or_else(Default::default, |open| open.checkpoint_status())
    }

    /// Returns WAL (Write-Ahead Log) status. After `close()` it returns a status of a disabled WAL
    /// (no path, size and records 0); it returns a `Result` from 0.7.0.
    ///
    /// Returns None if WAL is not enabled.
    #[must_use]
    pub fn wal_status(&self) -> crate::admin::WalStatus {
        self.live_if()
            .map_or_else(Default::default, |open| open.wal_status())
    }

    /// Forces a WAL checkpoint.
    ///
    /// Flushes all pending WAL records to the database file.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint fails, after a commit that did not
    /// complete (see [`TransactionManager`](crate::transaction::TransactionManager)),
    /// and the database-closed error after `close()`
    /// (its file is released: another handle may have written it since).
    /// A checkpoint that fails between writing its header and syncing it
    /// returns `GRAFEO-S008`: its outcome is unknown, the next open finds the
    /// new image or the previous one, and until then every checkpoint is
    /// refused with the same error (reopen the database).
    pub fn wal_checkpoint(&self) -> Result<()> {
        self.live()?.wal_checkpoint()
    }

    /// The database's change log, for tests and explicit checks of its
    /// memory account (`CdcLog::account_mismatches`).
    ///
    /// Since 0.6.4 it returns an owned handle, and after `close()` an empty one: holding it keeps
    /// that part of the database alive past `close()`.
    #[doc(hidden)]
    #[cfg(feature = "cdc")]
    #[must_use]
    pub fn cdc_log(&self) -> std::sync::Arc<crate::cdc::CdcLog> {
        self.live_if().map_or_else(
            || std::sync::Arc::new(crate::cdc::CdcLog::new()),
            |open| std::sync::Arc::clone(open.cdc_log()),
        )
    }

    /// Returns whether CDC is enabled by default for new sessions. After `close()` it returns
    /// `false`; it returns a `Result` from 0.7.0.
    #[cfg(feature = "cdc")]
    #[must_use]
    pub fn is_cdc_enabled(&self) -> bool {
        self.live_if()
            .map_or_else(Default::default, |open| open.is_cdc_enabled())
    }

    /// Turns change data capture on or off for new sessions and stores the
    /// setting with the database, so a reopen has it as it was: the setting
    /// changes, then a checkpoint writes it. An in-memory database has
    /// nothing to reopen: it changes the setting for as long as it is open.
    ///
    /// Does not affect sessions that were already created.
    /// [`Config::with_cdc`](crate::Config::with_cdc) turns capture on for one
    /// open only; this stores it.
    ///
    /// # Errors
    ///
    /// A database opened read-only refuses the change. A database on a store
    /// of its own ([`with_store`](Self::with_store)) has no file to store it
    /// in, and an open without the WAL keeps no durable feed (below): both
    /// refuse it
    /// (turning capture on, for the second) with a configuration error
    /// (`GRAFEO-V001`) that names the remedy. While capture is stored on, a
    /// writable open without the WAL is refused the same way. A checkpoint
    /// that fails (no room on the disk, an I/O error) leaves the setting and
    /// the stored record as they were. After a checkpoint whose outcome is
    /// unknown (`GRAFEO-S008`: it failed between writing its header and
    /// syncing it), the setting keeps its new value, as the next open may
    /// find it, and the call is refused until a reopen. The database-closed
    /// error after `close()`.
    ///
    /// While the setting is stored on, the database keeps the cursor of its
    /// durable change feed ([`read_cdc_feed`](Self::read_cdc_feed)) in its
    /// WAL directory, and checkpoints and `close()` keep the WAL from it, up
    /// to [`Config::with_wal_retention_limit`](crate::Config::with_wal_retention_limit):
    /// `<path>.wal` stays after `close()`. Turning it off removes the feed's
    /// cursor once the setting is stored:
    /// when that removal fails, the error is `GRAFEO-S007`
    /// ([`StorageError::CdcCursorPending`](grafeo_common::utils::error::StorageError::CdcCursorPending)).
    /// The setting is then stored off and the cursor keeps no log; the next
    /// open, or `set_cdc(false)` again, removes it.
    #[cfg(feature = "cdc")]
    pub fn set_cdc(&self, enabled: bool) -> Result<()> {
        self.live()?.set_cdc(enabled)
    }

    /// Sets whether CDC is enabled by default for new sessions, for as long
    /// as the database is open: nothing is stored. After `close()` it does
    /// nothing.
    ///
    /// Does not affect sessions that were already created.
    #[cfg(feature = "cdc")]
    #[deprecated(
        since = "0.6.4",
        note = "use `set_cdc` for a database with a file, which stores the setting and reports a \
                failure, or `Config::with_cdc()` for one without; removed in 0.7.0"
    )]
    pub fn set_cdc_enabled(&self, enabled: bool) {
        if let Some(open) = self.live_if() {
            open.set_cdc_enabled(enabled);
        }
    }

    /// Returns the full change history for an entity (node or edge) of the
    /// default graph (a session's `history` reads its current graph), up to
    /// the current epoch: a commit's events are recorded before it is
    /// complete, and returned once it is.
    ///
    /// Events are ordered chronologically by epoch.
    ///
    /// # Errors
    ///
    /// Returns an error if the CDC feature is not enabled.
    #[cfg(feature = "cdc")]
    pub fn history(
        &self,
        entity_id: impl Into<crate::cdc::EntityId>,
    ) -> Result<Vec<crate::cdc::ChangeEvent>> {
        self.live()?.history(entity_id)
    }

    /// Returns change events for an entity of the default graph since the
    /// given epoch, up to the current epoch.
    ///
    /// # Errors
    ///
    /// Currently infallible, but returns `Result` for forward compatibility.
    #[cfg(feature = "cdc")]
    pub fn history_since(
        &self,
        entity_id: impl Into<crate::cdc::EntityId>,
        since_epoch: grafeo_common::types::EpochId,
    ) -> Result<Vec<crate::cdc::ChangeEvent>> {
        self.live()?.history_since(entity_id, since_epoch)
    }

    /// Returns all change events across all entities and graphs in an epoch
    /// range, up to the current epoch; each event names its graph.
    ///
    /// # Errors
    ///
    /// Currently infallible, but returns `Result` for forward compatibility.
    #[cfg(feature = "cdc")]
    pub fn changes_between(
        &self,
        start_epoch: grafeo_common::types::EpochId,
        end_epoch: grafeo_common::types::EpochId,
    ) -> Result<Vec<crate::cdc::ChangeEvent>> {
        self.live()?.changes_between(start_epoch, end_epoch)
    }
}
