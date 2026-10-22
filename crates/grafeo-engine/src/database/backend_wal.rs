//! The write-ahead log of a database on a backend of its own
//! (`GrafeoDB::with_store` with a path): opened and replayed at the open,
//! and checkpointed through the backend (see the store contract's
//! durability, `GraphBackend::durable_epoch` and `GraphBackend::flush`).

use std::sync::Arc;

use grafeo_common::utils::error::{Error, Result};
use grafeo_core::graph::GraphBackend;
use grafeo_storage::file::v3::header::new_database_id;

use super::{GrafeoDB, backup, catalog_section, encryption, replay, wal_log};

impl GrafeoDB {
    /// Opens the WAL of a database on a backend of its own with a path: the
    /// sidecar directory `<path>.wal`, held by this process, after its
    /// groups were replayed (the catalog, and the backend's commits after its
    /// durable epoch); the database goes on above the last epoch the log or
    /// the backend holds. Nothing without a path or with the WAL turned off.
    ///
    /// # Errors
    ///
    /// The errors of locking, scanning and replaying the log (one that does
    /// not fit the backend fails the open) and of opening its writer.
    pub(super) fn open_backend_wal(&mut self, store: &Arc<dyn GraphBackend>) -> Result<()> {
        let Some(path) = self.config.path().map(std::path::Path::to_path_buf) else {
            return Ok(());
        };
        if !self.config.wal_enabled() {
            return Ok(());
        }
        let dir = grafeo_storage::file::detect::sidecar_wal_path(&path);
        std::fs::create_dir_all(&dir)?;
        let lock = grafeo_storage::wal::lock_wal_directory(&dir).map_err(Error::from)?;
        let keys = encryption::DatabaseKeys::from_config(&self.config);
        let header = grafeo_storage::wal::first_segment_header(&dir).map_err(Error::from)?;
        let durable = store.durable_epoch();
        let mut last = durable.max(store.committed_epoch());
        let (database_id, start_lsn) = match header {
            None => (new_database_id(), 0),
            Some(header) => {
                let replayed = replay::replay_into_backend(
                    &dir,
                    grafeo_storage::wal::ScanOptions {
                        cipher_for_salt: keys.wal_segment_ciphers(header.database_id),
                        salvage: self.config.wal_salvage(),
                        ..grafeo_storage::wal::ScanOptions::new(
                            header.database_id,
                            header.first_lsn,
                        )
                    },
                    &**store,
                    &self.catalog,
                    durable,
                    Some(&self.transaction_manager),
                )?;
                // Cut what no commit finished before the writer goes on.
                replayed.end.cut_torn_tail()?;
                last = last.max(replayed.last_epoch);
                (header.database_id, replayed.end.end_lsn)
            }
        };
        self.transaction_manager.sync_epoch(last);
        self.open_wal_at(&dir, database_id, &keys, start_lsn)?;
        *self.backend_wal_lock.lock() = Some(lock);
        Ok(())
    }

    /// The checkpoint of a database on a backend of its own: the backend
    /// makes every commit up to the committed epoch durable
    /// ([`GraphBackend::flush`]); when it then reports that epoch durable, a
    /// new log starts with the catalog's records as its first group, and the
    /// older groups go, unless a backup chain still reads them. A backend
    /// that keeps nothing across a restart keeps the whole log. Holds commits
    /// off while it runs. Whether it dropped the older groups.
    ///
    /// # Errors
    ///
    /// The backend's flush error (the log keeps every commit), and the
    /// errors of writing the catalog group and removing the old segments.
    pub(super) fn checkpoint_backend(&self, backend: &Arc<dyn GraphBackend>) -> Result<bool> {
        use grafeo_common::storage::log_record::Origin;
        use grafeo_common::testing::crash::maybe_crash;

        let Some(wal) = self.wal.as_ref() else {
            return Ok(false);
        };
        let _held = self.transaction_manager.hold_commits()?;
        let epoch = self.transaction_manager.current_epoch();
        backend
            .flush(epoch)
            .map_err(|error| Error::Internal(format!("the backend did not flush: {error}")))?;
        if backend.durable_epoch() < epoch {
            return Ok(false);
        }
        let dir = wal.dir().to_path_buf();
        if Self::backup_chain_needs_backend_wal(&dir) {
            return Ok(false);
        }
        // The new log starts with the catalog, so dropping the older groups
        // loses no schema.
        let start = wal.rotate().map_err(Error::from)?;
        let mut change = crate::transaction::StandaloneChange::new();
        for record in catalog_section::catalog_records(&self.catalog, Vec::new(), &[], false)? {
            change.push(grafeo_common::change::StandaloneOp::PutCatalog(record));
        }
        let mut group = wal
            .begin_group(grafeo_common::types::TransactionId::SYSTEM)
            .map_err(wal_log::commit_error)?;
        wal_log::write_standalone(&mut group, &change, epoch, Origin::Schema)?;
        group.finish().map_err(wal_log::commit_error)?;
        wal.sync().map_err(Error::from)?;
        maybe_crash("backend_checkpoint:after_catalog_group");
        wal.remove_segments_before(start).map_err(Error::from)?;
        maybe_crash("backend_checkpoint:after_old_groups");
        Ok(true)
    }

    /// Whether a backup chain needs the log of a database on a backend of its
    /// own, in `dir`: it holds a backup cursor (see `backup_chain_needs_wal`).
    fn backup_chain_needs_backend_wal(dir: &std::path::Path) -> bool {
        match backup::read_backup_cursor(dir) {
            Ok(cursor) => cursor.is_some(),
            Err(error) => {
                grafeo_common::grafeo_warn!(
                    "keeping the WAL at a checkpoint: cannot read the backup cursor: {error}"
                );
                true
            }
        }
    }
}
