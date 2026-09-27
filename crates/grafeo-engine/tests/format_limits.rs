//! Data that exceeds a storage format limit is refused, not written corrupt
//! (#392).
//!
//! The LPG section stores some counts in fixed-width fields (here: labels per
//! node in a `u16`). Writing more must fail loudly at checkpoint, keep the
//! sidecar WAL, and leave the database recoverable on reopen.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test format_limits
//! ```

#![allow(missing_docs)]

#[cfg(feature = "grafeo-file")]
mod tests {
    use grafeo_engine::config::StorageFormat;
    use grafeo_engine::{Config, GrafeoDB};

    #[test]
    fn checkpoint_over_a_format_limit_fails_and_data_survives_in_the_wal() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("limits.grafeo");
        let config = || Config::persistent(&path).with_storage_format(StorageFormat::SingleFile);

        let labels: Vec<String> = (0..=usize::from(u16::MAX))
            .map(|i| format!("L{i}"))
            .collect();
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();

        {
            let db = GrafeoDB::with_config(config()).unwrap();
            // Committed in a transaction so the WAL holds a commit marker;
            // direct `db.create_node` writes are covered by the WAL tests.
            let mut session = db.session();
            session.begin_transaction().unwrap();
            let node = session.create_node(&label_refs);
            session.create_node(&["Person"]);
            session.commit().unwrap();

            let err = db
                .close()
                .expect_err("the snapshot cannot hold 65,536 labels on one node");
            assert!(
                err.to_string().contains("LPG labels per node: 65536"),
                "{err}"
            );
            assert!(
                dir.path().join("limits.grafeo.wal").exists(),
                "the sidecar WAL must be kept when the checkpoint fails"
            );
            assert_eq!(db.get_node(node).unwrap().labels.len(), labels.len());
        }

        // Reopening replays the WAL: nothing was lost.
        let db = GrafeoDB::with_config(config()).unwrap();
        assert_eq!(db.node_count(), 2);
        let wide = db
            .session()
            .execute("MATCH (n:L65535) RETURN count(n) AS c")
            .unwrap();
        assert_eq!(wide.rows()[0][0], grafeo_common::types::Value::Int64(1));
    }
}
