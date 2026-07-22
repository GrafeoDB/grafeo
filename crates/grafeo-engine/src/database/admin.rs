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

    /// Returns the current committed epoch of the database.
    ///
    /// This is the same committed epoch consumed by [`GrafeoDB::backup_full`]
    /// and [`GrafeoDB::backup_incremental`]: it is read from the transaction
    /// manager, which advances the epoch atomically only when a transaction
    /// commits successfully. Pending/uncommitted transactions do not affect
    /// the returned value, and a failed or rolled-back transaction is never
    /// presented as a committed epoch.
    ///
    /// The call is read-only and side-effect free: it performs no backup,
    /// WAL checkpoint, epoch increment, or storage mutation. It is safe to
    /// call concurrently under the engine's existing concurrent-read contract.
    ///
    /// After successful transactions the value is monotonic.
    #[must_use]
    pub fn current_epoch(&self) -> grafeo_common::types::EpochId {
        self.transaction_manager.current_epoch()
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
    /// Flushes all pending WAL records to the main storage.
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

        #[cfg(feature = "wal")]
        if let Some(ref wal) = self.wal {
            let epoch = self.lpg_store().current_epoch();
            let transaction_id = self
                .transaction_manager
                .last_assigned_transaction_id()
                .unwrap_or_else(|| self.transaction_manager.begin());
            wal.checkpoint(transaction_id, epoch)?;
            wal.sync()?;
        }

        // Flush all sections to .grafeo file (explicit checkpoint)
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

#[cfg(test)]
mod tests {
    use crate::GrafeoDB;

    /// A freshly opened database reports its defined initial committed epoch.
    #[test]
    fn test_current_epoch_initial() {
        let db = GrafeoDB::new_in_memory();
        assert_eq!(
            db.current_epoch(),
            grafeo_common::types::EpochId::INITIAL,
            "fresh DB must report the initial committed epoch"
        );
    }

    /// A successful committed transaction advances the epoch monotonically.
    #[test]
    fn test_current_epoch_advances_on_commit() {
        let db = GrafeoDB::new_in_memory();
        let initial = db.current_epoch();

        let mut session = db.session();
        session.begin_transaction().expect("begin");
        session
            .execute("INSERT (:Person {name: 'Alix'})")
            .expect("insert");
        session.commit().expect("commit");

        let after = db.current_epoch();
        assert!(
            after > initial,
            "committed transaction must advance the epoch: initial={initial:?} after={after:?}"
        );

        // A second committed transaction advances it further (monotonic).
        let mut session2 = db.session();
        session2.begin_transaction().expect("begin2");
        session2
            .execute("INSERT (:Person {name: 'Boi'})")
            .expect("insert2");
        session2.commit().expect("commit2");
        let after2 = db.current_epoch();
        assert!(
            after2 > after,
            "epoch must be monotonic across commits: after={after:?} after2={after2:?}"
        );
    }

    /// A rolled-back transaction must not be reported as a committed advance.
    #[test]
    fn test_current_epoch_rollback_no_advance() {
        let db = GrafeoDB::new_in_memory();
        let initial = db.current_epoch();

        let mut session = db.session();
        session.begin_transaction().expect("begin");
        session
            .execute("INSERT (:Person {name: 'Cy'})")
            .expect("insert");
        session.rollback().expect("rollback");

        assert_eq!(
            db.current_epoch(),
            initial,
            "rolled-back transaction must not advance the committed epoch"
        );
    }

    /// Pending/uncommitted work is excluded from the reported epoch.
    #[test]
    fn test_current_epoch_excludes_pending() {
        let db = GrafeoDB::new_in_memory();
        let initial = db.current_epoch();

        let mut session = db.session();
        session.begin_transaction().expect("begin");
        session
            .execute("INSERT (:Person {name: 'Dee'})")
            .expect("insert");

        // The transaction is still open: the epoch must not have advanced.
        assert_eq!(
            db.current_epoch(),
            initial,
            "an open (uncommitted) transaction must not advance the committed epoch"
        );

        session.rollback().expect("cleanup rollback");
    }

    /// Calling the API repeatedly is side-effect free: it creates no backup
    /// artifacts and does not mutate a watched empty backup directory.
    #[cfg(all(feature = "wal", feature = "grafeo-file", feature = "lpg"))]
    #[test]
    fn test_current_epoch_no_backup_side_effects() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let backup_dir = tmp.path().join("backups");
        std::fs::create_dir_all(&backup_dir).expect("create backup dir");

        let db = GrafeoDB::new_in_memory();

        // Snapshot the directory listing before the calls.
        let before: Vec<_> = std::fs::read_dir(&backup_dir)
            .expect("read dir")
            .map(|e| e.expect("entry").file_name())
            .collect();

        // Repeated calls must be side-effect free.
        let e1 = db.current_epoch();
        let e2 = db.current_epoch();
        let e3 = db.current_epoch();
        assert_eq!(e1, e2);
        assert_eq!(e2, e3);

        let after: Vec<_> = std::fs::read_dir(&backup_dir)
            .expect("read dir")
            .map(|e| e.expect("entry").file_name())
            .collect();

        assert_eq!(
            before, after,
            "current_epoch() must not create backup artifacts or mutate the backup dir"
        );
        assert!(
            after.is_empty(),
            "no backup manifest/segment files may be created by current_epoch()"
        );
    }
}
