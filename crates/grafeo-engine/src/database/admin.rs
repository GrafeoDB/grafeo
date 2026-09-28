//! Admin, introspection, and diagnostic operations for GrafeoDB.

use std::path::Path;

use grafeo_common::utils::error::Result;

impl super::GrafeoDB {
    // =========================================================================
    // ADMIN API: Counts
    // =========================================================================

    /// Returns the number of nodes in the database.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.lpg_store().node_count()
    }

    /// Returns the number of edges in the database.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.lpg_store().edge_count()
    }

    /// Returns the number of distinct labels in the database.
    #[must_use]
    pub fn label_count(&self) -> usize {
        self.lpg_store().label_count()
    }

    /// Returns the number of distinct property keys in the database.
    #[must_use]
    pub fn property_key_count(&self) -> usize {
        self.lpg_store().property_key_count()
    }

    /// Returns the number of distinct edge types in the database.
    #[must_use]
    pub fn edge_type_count(&self) -> usize {
        self.lpg_store().edge_type_count()
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
            node_count: self.lpg_store().node_count(),
            edge_count: self.lpg_store().edge_count(),
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
            if p.exists() {
                Self::calculate_disk_usage(p).ok()
            } else {
                None
            }
        });
        #[cfg(not(feature = "wal"))]
        let disk_bytes: Option<usize> = None;

        crate::admin::DatabaseStats {
            node_count: self.lpg_store().node_count(),
            edge_count: self.lpg_store().edge_count(),
            label_count: self.lpg_store().label_count(),
            edge_type_count: self.lpg_store().edge_type_count(),
            property_key_count: self.lpg_store().property_key_count(),
            index_count: self.catalog.index_count(),
            memory_bytes: self.memory_usage().total_bytes,
            disk_bytes,
        }
    }

    /// Calculates total disk usage for the database directory.
    #[cfg(feature = "wal")]
    fn calculate_disk_usage(path: &Path) -> Result<usize> {
        let mut total = 0usize;
        if path.is_dir() {
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                let metadata = entry.metadata()?;
                if metadata.is_file() {
                    // reason: file sizes fit usize on 64-bit targets
                    #[allow(clippy::cast_possible_truncation)]
                    let file_len = metadata.len() as usize;
                    total += file_len;
                } else if metadata.is_dir() {
                    total += Self::calculate_disk_usage(&entry.path())?;
                }
            }
        }
        Ok(total)
    }

    /// Returns schema information (labels, edge types, property keys).
    ///
    /// For LPG mode, returns label and edge type information.
    /// For RDF mode, returns predicate and named graph information.
    #[must_use]
    pub fn schema(&self) -> crate::admin::SchemaInfo {
        let labels = self
            .lpg_store()
            .all_labels()
            .into_iter()
            .map(|name| crate::admin::LabelInfo {
                name: name.clone(),
                count: self.lpg_store().nodes_with_label(&name).count(),
            })
            .collect();

        let edge_types = self
            .lpg_store()
            .all_edge_types()
            .into_iter()
            .map(|name| crate::admin::EdgeTypeInfo {
                name: name.clone(),
                count: self.lpg_store().edges_with_type(&name).count(),
            })
            .collect();

        let property_keys = self.lpg_store().all_property_keys();

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
    /// Returns a list of errors and warnings. Empty errors = valid.
    #[must_use]
    pub fn validate(&self) -> crate::admin::ValidationResult {
        let mut result = crate::admin::ValidationResult::default();

        // Check for dangling edge references
        for edge in self.lpg_store().all_edges() {
            if self.lpg_store().get_node(edge.src).is_none() {
                result.errors.push(crate::admin::ValidationError {
                    code: "DANGLING_SRC".to_string(),
                    message: format!(
                        "Edge {} references non-existent source node {}",
                        edge.id.0, edge.src.0
                    ),
                    context: Some(format!("edge:{}", edge.id.0)),
                });
            }
            if self.lpg_store().get_node(edge.dst).is_none() {
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
        if self.lpg_store().node_count() > 0 && self.lpg_store().edge_count() == 0 {
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
                path: self.config.path.as_ref().map(|p| p.join("wal")),
                size_bytes: wal.size_bytes(),
                // reason: WAL record count fits usize on 64-bit targets
                #[allow(clippy::cast_possible_truncation)]
                record_count: wal.record_count() as usize,
                last_checkpoint: wal.last_checkpoint_timestamp(),
                current_epoch: self.lpg_store().current_epoch().as_u64(),
            };
        }

        crate::admin::WalStatus {
            enabled: false,
            path: None,
            size_bytes: 0,
            record_count: 0,
            last_checkpoint: None,
            current_epoch: self.lpg_store().current_epoch().as_u64(),
        }
    }

    /// Forces a WAL checkpoint.
    ///
    /// Flushes all pending WAL records to the main storage. In WAL-directory
    /// databases the WAL is the only copy of the data, so this only syncs it.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint fails.
    pub fn wal_checkpoint(&self) -> Result<()> {
        // Read-only databases have no WAL and the on-disk file is already a
        // valid snapshot: nothing to checkpoint.
        if self.read_only {
            return Ok(());
        }

        // WAL-directory mode: a checkpoint record would make recovery skip,
        // and truncation delete, WAL files that exist nowhere else (#419).
        #[cfg(feature = "wal")]
        {
            #[cfg(feature = "grafeo-file")]
            let has_snapshot = self.file_manager.is_some();
            #[cfg(not(feature = "grafeo-file"))]
            let has_snapshot = false;

            if !has_snapshot {
                if let Some(ref wal) = self.wal {
                    wal.sync()?;
                }
                return Ok(());
            }
        }

        // Flush all sections to the .grafeo file. The flush marks and truncates
        // the WAL only once the file is durable (#417).
        #[cfg(feature = "grafeo-file")]
        if let Some(ref fm) = self.file_manager {
            let _ = self.checkpoint_to_file(fm, super::flush::FlushReason::Explicit)?;
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

    /// Returns the full change history for an entity (node or edge).
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
        Ok(self.cdc_log.history(entity_id.into()))
    }

    /// Returns change events for an entity since the given epoch.
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
        Ok(self.cdc_log.history_since(entity_id.into(), since_epoch))
    }

    /// Returns all change events across all entities in an epoch range.
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
        Ok(self.cdc_log.changes_between(start_epoch, end_epoch))
    }
}
