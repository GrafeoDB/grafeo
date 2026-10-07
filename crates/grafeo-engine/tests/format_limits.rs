//! Data past the limits of the 0.5.x formats is written whole (#392).
//!
//! The 0.5.x LPG section stored some counts in fixed-width fields (labels per
//! node in a `u16`), so a checkpoint of more failed. The chunked section of
//! 0.6 has no such field: a node with 65,536 labels checkpoints, and a reopen
//! reads it back from the file alone.
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
    fn a_node_with_more_labels_than_a_u16_survives_a_checkpoint_and_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("limits.grafeo");
        let config = || Config::persistent(&path).with_storage_format(StorageFormat::Auto);

        let labels: Vec<String> = (0..=usize::from(u16::MAX))
            .map(|i| format!("L{i}"))
            .collect();
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();

        let node = {
            let db = GrafeoDB::with_config(config()).unwrap();
            let mut session = db.session();
            session.begin_transaction().unwrap();
            let node = session.create_node(&label_refs).unwrap();
            session.create_node(&["Person"]).unwrap();
            session.commit().unwrap();
            db.close()
                .expect("the checkpoint holds 65,536 labels on one node");
            node
        };
        assert!(
            !dir.path().join("limits.grafeo.wal").exists(),
            "a checkpoint that succeeds removes the sidecar WAL: the reopen reads the file"
        );

        let db = GrafeoDB::with_config(config()).unwrap();
        assert_eq!(db.node_count(), 2);
        assert_eq!(db.get_node(node).unwrap().labels.len(), labels.len());
        let wide = db
            .session()
            .execute("MATCH (n:L65535) RETURN count(n) AS c")
            .unwrap();
        assert_eq!(wide.rows()[0][0], grafeo_common::types::Value::Int64(1));
    }

    /// A string inside `depth` lists.
    #[cfg(feature = "wal")]
    fn nested(depth: usize) -> grafeo_common::types::Value {
        let mut value = grafeo_common::types::Value::from("Prague");
        for _ in 0..depth {
            value = grafeo_common::types::Value::List(vec![value].into());
        }
        value
    }

    /// A 0.5.43 WAL directory at `path` holding Alix, whose `trips` is
    /// `value`.
    #[cfg(feature = "wal")]
    fn wal_directory_with(path: &std::path::Path, value: grafeo_common::types::Value) {
        use grafeo_common::types::{NodeId, TransactionId, Value};
        use grafeo_storage::wal::{WalManager, WalRecord};

        let wal = WalManager::open(path.join("wal")).unwrap();
        for record in [
            WalRecord::CreateNode {
                id: NodeId::new(0),
                labels: vec!["Person".to_string()],
            },
            WalRecord::SetNodeProperty {
                id: NodeId::new(0),
                key: "name".to_string(),
                value: Value::from("Alix"),
            },
            WalRecord::SetNodeProperty {
                id: NodeId::new(0),
                key: "trips".to_string(),
                value,
            },
            WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            },
        ] {
            wal.log(&record).unwrap();
        }
        wal.sync().unwrap();
    }

    /// A 0.5.x database could hold a value nested deeper than 0.6 writes take
    /// (128 lists, maps and paths). Its migration fails naming the node and
    /// the property, and leaves the database as it was; a value at the limit
    /// migrates and reads back.
    #[cfg(feature = "wal")]
    #[test]
    fn a_0_5_value_nested_too_deep_fails_the_migration_naming_it() {
        use grafeo_common::storage::MAX_PROPERTY_VALUE_DEPTH;
        use grafeo_common::types::NodeId;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("deep");
        wal_directory_with(&path, nested(MAX_PROPERTY_VALUE_DEPTH + 1));
        let error = GrafeoDB::open(&path)
            .err()
            .expect("the migration refuses the value")
            .to_string();
        assert!(
            error.contains("node 0, property \"trips\""),
            "the error names the node and the property: {error}"
        );
        assert!(
            path.join("wal").is_dir() && !dir.path().join("deep.pre-0.6").exists(),
            "the 0.5.x database stays where it was"
        );

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("deepest");
        let deepest = nested(MAX_PROPERTY_VALUE_DEPTH);
        wal_directory_with(&path, deepest.clone());
        GrafeoDB::open(&path).unwrap().close().unwrap();
        let db = GrafeoDB::open(&path).unwrap();
        let trips = db
            .get_node(NodeId::new(0))
            .unwrap()
            .properties
            .to_btree_map()
            .get(&grafeo_common::types::PropertyKey::new("trips"))
            .cloned();
        assert_eq!(trips, Some(deepest));
        db.close().unwrap();
    }
}
