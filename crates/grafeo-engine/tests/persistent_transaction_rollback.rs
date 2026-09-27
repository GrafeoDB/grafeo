//! Rolling back a transaction on a persistent database restores property and
//! label changes, the same as in memory.
//!
//! Persistent sessions write through a WAL-logging store wrapper. It must keep
//! the transactional (undo-recording) path for property and label changes,
//! otherwise a rolled-back `SET` / `REMOVE` stays applied.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test persistent_transaction_rollback
//! ```

#![allow(missing_docs)]

#[cfg(feature = "wal")]
mod tests {
    use grafeo_common::types::Value;
    use grafeo_engine::config::StorageFormat;
    use grafeo_engine::{Config, GrafeoDB};
    use std::path::Path;

    fn configs(dir: &Path) -> Vec<(&'static str, Config)> {
        let mut configs = vec![
            (
                "wal directory",
                Config::persistent(dir.join("dir-db"))
                    .with_storage_format(StorageFormat::WalDirectory),
            ),
            (
                "wal directory + cdc",
                Config::persistent(dir.join("dir-cdc-db"))
                    .with_storage_format(StorageFormat::WalDirectory)
                    .with_cdc(),
            ),
        ];
        #[cfg(feature = "grafeo-file")]
        configs.push((
            "single file",
            Config::persistent(dir.join("single.grafeo"))
                .with_storage_format(StorageFormat::SingleFile),
        ));
        configs
    }

    /// Node and edge state as `(labels, n.v, n.w, n.x, r.weight, r.note)`.
    fn state(db: &GrafeoDB) -> Vec<Value> {
        let result = db
            .session()
            .execute(
                "MATCH (n:Person {name: 'Alix'})-[r:KNOWS]->() \
                 RETURN labels(n), n.v, n.w, n.x, r.weight, r.note",
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
        let mut row = result.rows()[0].clone();
        if let Value::List(labels) = &row[0] {
            let mut sorted: Vec<Value> = labels.to_vec();
            sorted.sort_by_key(|v| v.to_string());
            row[0] = Value::List(sorted.into());
        }
        row
    }

    fn seed(db: &GrafeoDB) {
        db.session()
            .execute(
                "INSERT (:Person:Base {name: 'Alix', v: 1, x: 'keep'})\
                 -[:KNOWS {weight: 5, note: 'n'}]->(:Person {name: 'Gus'})",
            )
            .unwrap();
    }

    fn change_everything(session: &grafeo_engine::session::Session) {
        for query in [
            "MATCH (n:Person {name: 'Alix'}) SET n.v = 2",
            "MATCH (n:Person {name: 'Alix'}) SET n.w = 3",
            "MATCH (n:Person {name: 'Alix'}) REMOVE n.x",
            "MATCH (n:Person {name: 'Alix'}) SET n:Extra",
            "MATCH (n:Person {name: 'Alix'}) REMOVE n:Base",
            "MATCH (:Person {name: 'Alix'})-[r:KNOWS]->() SET r.weight = 9",
            "MATCH (:Person {name: 'Alix'})-[r:KNOWS]->() REMOVE r.note",
        ] {
            session
                .execute(query)
                .unwrap_or_else(|e| panic!("{query}: {e}"));
        }
    }

    #[test]
    fn rollback_restores_properties_and_labels() {
        let dir = tempfile::tempdir().unwrap();
        for (name, config) in configs(dir.path()) {
            let db = GrafeoDB::with_config(config).unwrap();
            seed(&db);
            let before = state(&db);

            let mut session = db.session();
            session.begin_transaction().unwrap();
            change_everything(&session);
            assert_ne!(
                state(&db),
                before,
                "{name}: changes are applied in the transaction"
            );
            session.rollback().unwrap();

            assert_eq!(state(&db), before, "{name}: rollback restores everything");
        }
    }

    /// Control: the same rollback on an in-memory database.
    #[test]
    fn rollback_restores_properties_and_labels_in_memory() {
        let db = GrafeoDB::new_in_memory();
        seed(&db);
        let before = state(&db);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        change_everything(&session);
        session.rollback().unwrap();
        assert_eq!(state(&db), before);
    }

    #[test]
    fn rollback_is_not_replayed_after_reopen() {
        let dir = tempfile::tempdir().unwrap();
        for (name, config) in configs(dir.path()) {
            let before = {
                let db = GrafeoDB::with_config(config.clone()).unwrap();
                seed(&db);
                let before = state(&db);
                let mut session = db.session();
                session.begin_transaction().unwrap();
                change_everything(&session);
                session.rollback().unwrap();
                db.close().unwrap();
                before
            };

            let db = GrafeoDB::with_config(config).unwrap();
            assert_eq!(state(&db), before, "{name}: reopened state");
        }
    }

    #[test]
    fn commit_persists_changes() {
        let dir = tempfile::tempdir().unwrap();
        for (name, config) in configs(dir.path()) {
            let after = {
                let db = GrafeoDB::with_config(config.clone()).unwrap();
                seed(&db);
                let mut session = db.session();
                session.begin_transaction().unwrap();
                change_everything(&session);
                session.commit().unwrap();
                let after = state(&db);
                db.close().unwrap();
                after
            };

            let db = GrafeoDB::with_config(config).unwrap();
            assert_eq!(
                state(&db),
                after,
                "{name}: committed changes survive reopen"
            );
        }
    }
}
