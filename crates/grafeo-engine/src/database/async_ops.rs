//! Async engine operations for `GrafeoDB`.
//!
//! These methods offload I/O-bound operations (WAL checkpoint, snapshot writes)
//! to blocking tasks via [`tokio::task::spawn_blocking`], avoiding blocking the
//! tokio runtime. Query execution stays sync (CPU-bound, rayon).
//!
//! All methods take `self: &Arc<Self>` so the `GrafeoDB` can be cheaply cloned
//! into the blocking closure. This matches how `grafeo-server` already holds
//! databases in `Arc<GrafeoDB>`.

use std::sync::Arc;

use grafeo_common::utils::error::{Error, Result};

use super::GrafeoDB;

impl GrafeoDB {
    /// Asynchronous WAL checkpoint.
    ///
    /// Offloads the sync [`wal_checkpoint()`](Self::wal_checkpoint) to a blocking
    /// task so the tokio runtime is not blocked during fsync.
    ///
    /// Takes `&Arc<Self>` because the blocking closure must be `'static`.
    /// This matches the `Arc<GrafeoDB>` pattern used by `grafeo-server`.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint fails or the blocking task panics.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example() -> grafeo_common::utils::error::Result<()> {
    /// use std::sync::Arc;
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = Arc::new(GrafeoDB::open("./my_db")?);
    /// db.async_wal_checkpoint().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn async_wal_checkpoint(self: &Arc<Self>) -> Result<()> {
        let db = Arc::clone(self);
        tokio::task::spawn_blocking(move || db.wal_checkpoint())
            .await
            .map_err(|e| Error::Internal(format!("async checkpoint task failed: {e}")))?
    }

    /// Asynchronous snapshot write to the `.grafeo` file.
    ///
    /// Offloads the full snapshot export and file write to a blocking task.
    /// This is the most I/O-intensive operation (serialization + fsync) and
    /// benefits most from not blocking the tokio runtime.
    ///
    /// Requires the `grafeo-file` feature.
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot write fails or the blocking task
    /// panics, and the database-closed error after `close()`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example() -> grafeo_common::utils::error::Result<()> {
    /// use std::sync::Arc;
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = Arc::new(GrafeoDB::open("./my_db")?);
    /// db.async_write_snapshot().await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "grafeo-file")]
    pub async fn async_write_snapshot(self: &Arc<Self>) -> Result<()> {
        let db = Arc::clone(self);
        tokio::task::spawn_blocking(move || {
            // `close()` waits for the write, and none runs after it.
            let _open = db.hold_open()?;
            // Read-only databases have nothing to flush.
            if db.read_only {
                return Ok(());
            }
            let Some(ref fm) = db.file_manager else {
                return Err(Error::Internal(
                    "no file manager configured for snapshot write".to_string(),
                ));
            };
            db.checkpoint_to_file(fm).map(|_| ())
        })
        .await
        .map_err(|e| Error::Internal(format!("async snapshot task failed: {e}")))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn async_wal_checkpoint_in_memory() {
        // In-memory databases have no WAL, so this should be a no-op success.
        let db = Arc::new(GrafeoDB::new_in_memory());
        db.async_wal_checkpoint().await.unwrap();
    }

    #[tokio::test]
    async fn async_wal_checkpoint_with_data() {
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(GrafeoDB::open(dir.path().join("test.grafeo")).unwrap());

        // Insert some data
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        drop(session);

        // Async checkpoint should succeed
        db.async_wal_checkpoint().await.unwrap();
    }

    #[tokio::test]
    async fn async_wal_checkpoint_does_not_block_runtime() {
        // Verify that async_wal_checkpoint yields to the runtime.
        // Run a concurrent task alongside the checkpoint.
        let db = Arc::new(GrafeoDB::new_in_memory());

        let (checkpoint_result, concurrent_result) =
            tokio::join!(db.async_wal_checkpoint(), async { 42 },);

        checkpoint_result.unwrap();
        assert_eq!(concurrent_result, 42);
    }

    /// Regression: async_write_snapshot on a read-only database should be a
    /// no-op, not an error.
    #[cfg(feature = "grafeo-file")]
    #[tokio::test]
    async fn async_write_snapshot_on_read_only_is_no_op() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ro_async.grafeo");

        {
            let db = GrafeoDB::open(&path).unwrap();
            let session = db.session();
            session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
            db.close().unwrap();
        }

        let db = Arc::new(GrafeoDB::open_read_only(&path).unwrap());
        db.async_write_snapshot()
            .await
            .expect("async_write_snapshot on read-only should be a no-op");
        assert_eq!(db.node_count(), 1);
    }

    /// The snapshot itself must reach the file: the roundtrip test below
    /// passes through `close()`, which checkpoints anyway.
    #[cfg(feature = "grafeo-file")]
    #[tokio::test]
    async fn async_write_snapshot_writes_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.grafeo");
        let db = Arc::new(GrafeoDB::open(&path).unwrap());
        db.session()
            .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
            .unwrap();

        db.async_write_snapshot().await.unwrap();

        let header = db.file_manager.as_ref().unwrap().active_header();
        assert_eq!((header.node_count, header.edge_count), (2, 1));
    }

    /// After `close()` the file is released (another handle may have written
    /// it since): the snapshot write fails with the database-closed error and
    /// writes nothing.
    #[cfg(feature = "grafeo-file")]
    #[tokio::test]
    async fn async_write_snapshot_after_close_fails_and_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("berlin.grafeo");
        let db = Arc::new(GrafeoDB::open(&path).unwrap());
        db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        db.close().unwrap();
        let before = std::fs::read(&path).unwrap();

        let error = db
            .async_write_snapshot()
            .await
            .expect_err("a snapshot write after close() fails");
        assert!(
            matches!(
                error,
                Error::Transaction(grafeo_common::utils::error::TransactionError::DatabaseClosed)
            ),
            "{error:?}"
        );
        assert!(
            std::fs::read(&path).unwrap() == before,
            "the file is unchanged"
        );
    }

    #[cfg(feature = "grafeo-file")]
    #[tokio::test]
    async fn async_write_snapshot_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.grafeo");
        let db = Arc::new(GrafeoDB::open(&path).unwrap());

        // Insert data
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        drop(session);

        // Write snapshot
        db.async_write_snapshot().await.unwrap();

        // Verify data survives by opening a fresh instance
        drop(db);
        let db2 = GrafeoDB::open(&path).unwrap();
        let session2 = db2.session();
        let result = session2.execute("MATCH (p:Person) RETURN p.name").unwrap();
        assert_eq!(result.rows.len(), 1);
    }
}
