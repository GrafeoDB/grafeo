//! Admin, introspection, and diagnostic operations for GrafeoDB.

use std::path::Path;

use grafeo_common::types::{ArcStr, EdgeId, EpochId};
use grafeo_common::utils::error::Result;
use grafeo_common::utils::hash::FxHashMap;
use grafeo_core::graph::{Direction, GraphStoreSearch};

impl super::GrafeoDB {
    // =========================================================================
    // ADMIN API: Counts
    // =========================================================================

    /// Returns the number of nodes in the database, as of the current epoch
    /// (see [`current_epoch`](Self::current_epoch)).
    #[must_use]
    pub fn node_count(&self) -> usize {
        let epoch = self.read_epoch();
        let store = self.graph_store();
        counted_at_own_epoch(&*store, epoch, || store.node_count()).unwrap_or_else(|| {
            store
                .filter_visible_node_ids(&store.all_node_ids(), epoch)
                .len()
        })
    }

    /// Returns the number of edges in the database, as of the current epoch
    /// (see [`current_epoch`](Self::current_epoch)).
    #[must_use]
    pub fn edge_count(&self) -> usize {
        let epoch = self.read_epoch();
        let store = self.graph_store();
        counted_at_own_epoch(&*store, epoch, || store.edge_count())
            .unwrap_or_else(|| self.edge_count_at(&*store, epoch))
    }

    /// The edges of `store` visible at `epoch`. The database's own store
    /// numbers its edges densely, so each is looked up; another store's
    /// edges are found through the nodes that have them, which misses an
    /// edge that a commit not complete yet deleted (its adjacency entry is
    /// gone at once).
    fn edge_count_at(&self, store: &dyn GraphStoreSearch, epoch: EpochId) -> usize {
        if self.external_read_store.is_none() {
            let lpg = self.lpg_store();
            return (0..lpg.next_edge_id())
                .filter(|&id| lpg.is_edge_visible_at_epoch(EdgeId::new(id), epoch))
                .count();
        }
        store
            .all_node_ids()
            .into_iter()
            .flat_map(|node| store.edges_from(node, Direction::Outgoing))
            .filter(|&(_, edge)| store.is_edge_visible_at_epoch(edge, epoch))
            .count()
    }

    /// Returns the number of distinct labels in the database.
    #[must_use]
    pub fn label_count(&self) -> usize {
        self.graph_store().all_labels().len()
    }

    /// Returns the number of distinct property keys in the database, of
    /// nodes and edges together.
    #[must_use]
    pub fn property_key_count(&self) -> usize {
        self.graph_store().all_property_keys().len()
    }

    /// Returns the number of distinct edge types in the database.
    #[must_use]
    pub fn edge_type_count(&self) -> usize {
        self.graph_store().all_edge_types().len()
    }

    // =========================================================================
    // ADMIN API: Introspection
    // =========================================================================

    /// Returns true if this database is backed by a file (persistent).
    ///
    /// In-memory databases return false.
    #[must_use]
    pub fn is_persistent(&self) -> bool {
        self.config.path.is_some()
    }

    /// Returns the database file path, if persistent.
    ///
    /// In-memory databases return None.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.config.path.as_deref()
    }

    /// Returns high-level database information.
    ///
    /// Includes node/edge counts, persistence status, and mode (LPG/RDF).
    #[must_use]
    pub fn info(&self) -> crate::admin::DatabaseInfo {
        crate::admin::DatabaseInfo {
            mode: crate::admin::DatabaseMode::Lpg,
            node_count: self.node_count(),
            edge_count: self.edge_count(),
            is_persistent: self.is_persistent(),
            path: self.config.path.clone(),
            wal_enabled: self.config.wal_enabled,
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
        }
    }

    /// Returns a hierarchical memory usage breakdown.
    ///
    /// Walks all internal structures (store, indexes, MVCC chains, caches,
    /// string pools, buffer manager) and returns estimated heap bytes for each.
    /// Safe to call concurrently with queries.
    #[must_use]
    pub fn memory_usage(&self) -> crate::memory_usage::MemoryUsage {
        use crate::memory_usage::{BufferManagerMemory, CacheMemory, MemoryUsage};
        use grafeo_common::memory::MemoryRegion;

        let (store, indexes, mvcc, string_pool) = self.lpg_store().memory_breakdown();

        let (parsed_bytes, optimized_bytes, cached_plan_count) =
            self.query_cache.heap_memory_bytes();
        let mut caches = CacheMemory {
            parsed_plan_cache_bytes: parsed_bytes,
            optimized_plan_cache_bytes: optimized_bytes,
            cached_plan_count,
            ..Default::default()
        };
        caches.compute_total();

        let bm_stats = self.buffer_manager.stats();
        let buffer_manager = BufferManagerMemory {
            budget_bytes: bm_stats.budget,
            allocated_bytes: bm_stats.total_allocated,
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
            ) = self.rdf_store.heap_memory_bytes();
            usage.rdf = RdfMemory {
                triple_count,
                triples_and_indexes_bytes,
                term_dictionary_bytes,
                ring_index_bytes,
                named_graph_count,
                total_bytes: 0,
            };
            usage.rdf.compute_total();
        }

        #[cfg(feature = "cdc")]
        {
            use crate::memory_usage::CdcMemory;
            let (total_bytes, entity_count, event_count) = self.cdc_log.heap_memory_bytes();
            usage.cdc = CdcMemory {
                total_bytes,
                entity_count,
                event_count,
            };
        }

        usage.compute_total();
        usage
    }

    /// Returns detailed database statistics.
    ///
    /// Includes counts, memory usage, and index information.
    #[must_use]
    pub fn detailed_stats(&self) -> crate::admin::DatabaseStats {
        #[cfg(feature = "wal")]
        let disk_bytes = self.config.path.as_ref().and_then(|p| {
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

        crate::admin::DatabaseStats {
            node_count: self.node_count(),
            edge_count: self.edge_count(),
            label_count: self.label_count(),
            edge_type_count: self.edge_type_count(),
            property_key_count: self.property_key_count(),
            index_count: self.catalog.index_count(),
            memory_bytes: self.memory_usage().total_bytes,
            disk_bytes,
        }
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

    /// The total size of the files under `dir` (0 when it does not exist).
    #[cfg(feature = "wal")]
    fn directory_size(dir: &Path) -> Result<usize> {
        let mut total = 0usize;
        if dir.is_dir() {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let metadata = entry.metadata()?;
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

    /// Returns schema information (labels, edge types, property keys).
    ///
    /// For LPG mode, returns label and edge type information.
    /// For RDF mode, returns predicate and named graph information.
    #[must_use]
    pub fn schema(&self) -> crate::admin::SchemaInfo {
        let store = self.graph_store();
        // The label index holds every node with the label, also those of a
        // transaction that has not committed: count the nodes that have the
        // label at the current epoch.
        let epoch = self.read_epoch();
        let labels = store
            .all_labels()
            .into_iter()
            .map(|name| crate::admin::LabelInfo {
                count: store
                    .nodes_by_label(&name)
                    .into_iter()
                    .filter(|&id| {
                        store
                            .get_node_at_epoch(id, epoch)
                            .is_some_and(|node| node.has_label(&name))
                    })
                    .count(),
                name,
            })
            .collect();

        // One pass over the edges counts every type, at the current epoch.
        let mut edges_per_type: FxHashMap<ArcStr, usize> = FxHashMap::default();
        for node in store.all_node_ids() {
            for (_, edge) in store.edges_from(node, Direction::Outgoing) {
                if store.is_edge_visible_at_epoch(edge, epoch)
                    && let Some(edge_type) = store.edge_type(edge)
                {
                    *edges_per_type.entry(edge_type).or_default() += 1;
                }
            }
        }
        let edge_types = store
            .all_edge_types()
            .into_iter()
            .map(|name| crate::admin::EdgeTypeInfo {
                count: edges_per_type.get(name.as_str()).copied().unwrap_or(0),
                name,
            })
            .collect();

        let property_keys = store.all_property_keys();

        crate::admin::SchemaInfo::Lpg(crate::admin::LpgSchemaInfo {
            labels,
            edge_types,
            property_keys,
        })
    }

    /// Returns detailed information about all indexes.
    #[must_use]
    pub fn list_indexes(&self) -> Vec<crate::admin::IndexInfo> {
        self.catalog
            .all_indexes()
            .into_iter()
            .map(|def| {
                let label_name = self
                    .catalog
                    .get_label_name(def.label)
                    .unwrap_or_else(|| "?".into());
                let prop_name = self
                    .catalog
                    .get_property_key_name(def.property_key)
                    .unwrap_or_else(|| "?".into());
                crate::admin::IndexInfo {
                    name: format!("idx_{}_{}", label_name, prop_name),
                    index_type: format!("{:?}", def.index_type),
                    target: format!("{}:{}", label_name, prop_name),
                    unique: false,
                    cardinality: None,
                    size_bytes: None,
                }
            })
            .collect()
    }

    /// Validates database integrity.
    ///
    /// Checks for:
    /// - Dangling edge references (edges pointing to non-existent nodes)
    /// - Internal index consistency
    ///
    /// Returns a list of errors and warnings. Empty errors = valid. Reads
    /// the database as of the current epoch.
    #[must_use]
    pub fn validate(&self) -> crate::admin::ValidationResult {
        let mut result = crate::admin::ValidationResult::default();
        // Nodes as queries see them, at the current epoch: after `compact()`
        // an edge of the overlay can end at a node of the compacted base.
        let epoch = self.read_epoch();
        let store = self.graph_store();

        // Check for dangling edge references
        for edge in self.iter_edges() {
            if store.get_node_at_epoch(edge.src, epoch).is_none() {
                result.errors.push(crate::admin::ValidationError {
                    code: "DANGLING_SRC".to_string(),
                    message: format!(
                        "Edge {} references non-existent source node {}",
                        edge.id.0, edge.src.0
                    ),
                    context: Some(format!("edge:{}", edge.id.0)),
                });
            }
            if store.get_node_at_epoch(edge.dst, epoch).is_none() {
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

        // Add warnings for potential issues
        if self.node_count() > 0 && self.edge_count() == 0 {
            result.warnings.push(crate::admin::ValidationWarning {
                code: "NO_EDGES".to_string(),
                message: "Database has nodes but no edges".to_string(),
                context: None,
            });
        }

        result
    }

    /// Returns WAL (Write-Ahead Log) status.
    ///
    /// Returns None if WAL is not enabled.
    #[must_use]
    pub fn wal_status(&self) -> crate::admin::WalStatus {
        #[cfg(feature = "wal")]
        if let Some(ref wal) = self.wal {
            return crate::admin::WalStatus {
                enabled: true,
                // The sidecar WAL of the database file, `<path>.wal/`.
                path: Some(wal.dir().to_path_buf()),
                size_bytes: wal.size_bytes(),
                // reason: WAL record count fits usize on 64-bit targets
                #[allow(clippy::cast_possible_truncation)]
                record_count: wal.record_count() as usize,
                last_checkpoint: wal.last_checkpoint_timestamp(),
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

    /// Forces a WAL checkpoint.
    ///
    /// Flushes all pending WAL records to the database file.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint fails, or after a commit that did
    /// not complete (see [`TransactionManager`](crate::transaction::TransactionManager)).
    pub fn wal_checkpoint(&self) -> Result<()> {
        // Read-only databases have no WAL and the on-disk file is already a
        // valid snapshot: nothing to checkpoint.
        if self.read_only {
            return Ok(());
        }
        // The store holds the stamped part of a commit that did not complete.
        self.transaction_manager.check_no_incomplete_commit()?;

        // Flush all sections to the .grafeo file. The flush marks and truncates
        // the WAL only once the file is durable (#417).
        #[cfg(feature = "grafeo-file")]
        if let Some(ref fm) = self.file_manager {
            let _ = self.checkpoint_to_file(fm)?;
        }

        Ok(())
    }

    // =========================================================================
    // ADMIN API: Change Data Capture
    // =========================================================================

    /// Returns whether CDC is enabled by default for new sessions.
    #[cfg(feature = "cdc")]
    #[must_use]
    pub fn is_cdc_enabled(&self) -> bool {
        self.cdc_active()
    }

    /// Sets whether CDC is enabled by default for new sessions.
    ///
    /// Does not affect sessions that were already created.
    #[cfg(feature = "cdc")]
    pub fn set_cdc_enabled(&self, enabled: bool) {
        self.cdc_enabled
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
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
        let epoch = self.read_epoch();
        let mut events = self.cdc_log.history(entity_id.into());
        events.retain(|event| event.epoch <= epoch);
        Ok(events)
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
        let epoch = self.read_epoch();
        let mut events = self.cdc_log.history_since(entity_id.into(), since_epoch);
        events.retain(|event| event.epoch <= epoch);
        Ok(events)
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
        let end_epoch = end_epoch.min(self.read_epoch());
        Ok(self.cdc_log.changes_between(start_epoch, end_epoch))
    }
}

/// `count`, which counts at the store's own epoch, when that is the count at
/// `epoch`: the store's epoch is not ahead of `epoch` (no version is newer
/// than the store's epoch) and does not move during the count. `None` when
/// it is ahead: a commit has stamped its versions and is not complete (or
/// never completes).
fn counted_at_own_epoch(
    store: &dyn GraphStoreSearch,
    epoch: EpochId,
    count: impl FnOnce() -> usize,
) -> Option<usize> {
    let store_epoch = store.current_epoch();
    if store_epoch > epoch {
        return None;
    }
    let counted = count();
    (store.current_epoch() == store_epoch).then_some(counted)
}
