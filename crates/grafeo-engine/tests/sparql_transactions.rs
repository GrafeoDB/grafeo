//! SPARQL updates inside explicit transactions.
//!
//! Every update form (`INSERT DATA`, `DELETE DATA`, `INSERT ... WHERE`,
//! `DELETE WHERE`, `DELETE/INSERT ... WHERE` with or without `WITH`) is
//! buffered in the open transaction: the transaction sees its own writes,
//! other sessions do not, commit applies them in the default graph and in
//! named graphs, and rollback discards them, also across a reopen.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test sparql_transactions
//! ```

#![allow(missing_docs)]

#[cfg(all(feature = "sparql", feature = "triple-store"))]
mod tests {
    use grafeo_common::types::Value;
    use grafeo_engine::session::Session;
    use grafeo_engine::{Config, GrafeoDB, GraphModel};

    fn rdf_db() -> GrafeoDB {
        GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap()
    }

    /// Sorted `?s ?o` pairs for predicate `<http://ex.org/p>` in the default graph.
    fn default_graph(session: &Session) -> Vec<(String, String)> {
        pairs(session, "SELECT ?s ?o WHERE { ?s <http://ex.org/p> ?o }")
    }

    /// Sorted `?s ?o` pairs for predicate `<http://ex.org/p>` in graph `<g>`.
    fn named_graph(session: &Session) -> Vec<(String, String)> {
        pairs(
            session,
            "SELECT ?s ?o WHERE { GRAPH <http://ex.org/g> { ?s <http://ex.org/p> ?o } }",
        )
    }

    fn pairs(session: &Session, query: &str) -> Vec<(String, String)> {
        let result = session.execute_sparql(query).unwrap();
        let text = |value: &Value| match value {
            Value::String(s) => s.to_string(),
            other => format!("{other:?}"),
        };
        let mut rows: Vec<(String, String)> = result
            .rows()
            .iter()
            .map(|row| (text(&row[0]), text(&row[1])))
            .collect();
        rows.sort();
        rows
    }

    fn pair(s: &str, o: &str) -> (String, String) {
        (format!("http://ex.org/{s}"), o.to_string())
    }

    const SEED: &str = r#"INSERT DATA {
        <http://ex.org/alix> <http://ex.org/p> "1" .
        <http://ex.org/gus> <http://ex.org/p> "2" .
        GRAPH <http://ex.org/g> { <http://ex.org/vincent> <http://ex.org/p> "3" . }
    }"#;

    /// Runs every update form in one transaction.
    fn run_all_update_forms(session: &Session) {
        for update in [
            r#"INSERT DATA { <http://ex.org/mia> <http://ex.org/p> "4" . }"#,
            r#"DELETE DATA { <http://ex.org/gus> <http://ex.org/p> "2" . }"#,
            r#"INSERT { ?s <http://ex.org/p> "copy" } WHERE { ?s <http://ex.org/p> "1" }"#,
            r#"DELETE WHERE { <http://ex.org/alix> <http://ex.org/p> "1" }"#,
            r#"WITH <http://ex.org/g>
               DELETE { ?s <http://ex.org/p> "3" }
               INSERT { ?s <http://ex.org/p> "30" }
               WHERE  { ?s <http://ex.org/p> "3" }"#,
            r#"INSERT DATA { GRAPH <http://ex.org/g> { <http://ex.org/butch> <http://ex.org/p> "5" . } }"#,
        ] {
            session
                .execute_sparql(update)
                .unwrap_or_else(|e| panic!("{update}: {e}"));
        }
    }

    fn expected_default_after_updates() -> Vec<(String, String)> {
        vec![pair("alix", "copy"), pair("mia", "4")]
    }

    fn expected_named_after_updates() -> Vec<(String, String)> {
        vec![pair("butch", "5"), pair("vincent", "30")]
    }

    #[test]
    fn rollback_discards_every_update_form() {
        let db = rdf_db();
        let mut session = db.session();
        session.execute_sparql(SEED).unwrap();
        let default_before = default_graph(&session);
        let named_before = named_graph(&session);

        session.begin_transaction().unwrap();
        run_all_update_forms(&session);
        session.rollback().unwrap();

        assert_eq!(default_graph(&session), default_before);
        assert_eq!(named_graph(&session), named_before);
    }

    #[test]
    fn commit_applies_every_update_form() {
        let db = rdf_db();
        let mut session = db.session();
        session.execute_sparql(SEED).unwrap();

        session.begin_transaction().unwrap();
        run_all_update_forms(&session);
        session.commit().unwrap();

        // Also seen by a fresh session, so it was applied, not just buffered.
        let other = db.session();
        assert_eq!(default_graph(&other), expected_default_after_updates());
        assert_eq!(named_graph(&other), expected_named_after_updates());
    }

    #[test]
    fn transaction_reads_its_own_writes_and_others_do_not() {
        let db = rdf_db();
        let mut session = db.session();
        session.execute_sparql(SEED).unwrap();
        let other = db.session();
        let default_before = default_graph(&other);
        let named_before = named_graph(&other);

        session.begin_transaction().unwrap();
        run_all_update_forms(&session);

        // Same result as the committed state in `commit_applies_every_update_form`.
        assert_eq!(default_graph(&session), expected_default_after_updates());
        assert_eq!(named_graph(&session), expected_named_after_updates());
        let any_graph = pairs(
            &session,
            "SELECT ?s ?o WHERE { GRAPH ?g { ?s <http://ex.org/p> ?o } }",
        );
        assert_eq!(any_graph, expected_named_after_updates());

        // Another session still sees the committed state.
        assert_eq!(default_graph(&other), default_before);
        assert_eq!(named_graph(&other), named_before);
        session.rollback().unwrap();
    }

    #[test]
    fn later_update_sees_earlier_write_in_same_transaction() {
        // INSERT DATA then DELETE WHERE on the same triple: the delete must see
        // the pending insert, so nothing is left after commit.
        let db = rdf_db();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute_sparql(r#"INSERT DATA { <http://ex.org/alix> <http://ex.org/p> "1" . }"#)
            .unwrap();
        session
            .execute_sparql(r#"DELETE WHERE { <http://ex.org/alix> <http://ex.org/p> ?o }"#)
            .unwrap();
        assert!(default_graph(&session).is_empty());
        session.commit().unwrap();
        assert!(default_graph(&db.session()).is_empty());
    }

    #[cfg(feature = "wal")]
    mod durability {
        use super::*;
        use grafeo_engine::config::StorageFormat;
        use std::path::Path;

        fn open(path: &Path) -> GrafeoDB {
            GrafeoDB::with_config(
                Config::persistent(path)
                    .with_graph_model(GraphModel::Rdf)
                    .with_storage_format(StorageFormat::WalDirectory),
            )
            .unwrap()
        }

        /// Seeds in an explicit transaction, so the WAL holds a commit marker
        /// for the seed independent of the auto-commit path.
        fn seed_committed(session: &mut Session) {
            session.begin_transaction().unwrap();
            session.execute_sparql(SEED).unwrap();
            session.commit().unwrap();
        }

        #[test]
        fn rolled_back_updates_stay_gone_after_reopen() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("db");
            let (default_before, named_before) = {
                let db = open(&path);
                let mut session = db.session();
                seed_committed(&mut session);
                let before = (default_graph(&session), named_graph(&session));
                session.begin_transaction().unwrap();
                run_all_update_forms(&session);
                session.rollback().unwrap();
                db.close().unwrap();
                before
            };

            let db = open(&path);
            let session = db.session();
            assert_eq!(default_graph(&session), default_before);
            assert_eq!(named_graph(&session), named_before);
        }

        /// An auto-committed write followed by a rolled-back transaction must
        /// survive a reopen. Recovery keeps one pending buffer for the whole
        /// WAL, and auto-commit writes carry no commit marker, so the abort
        /// discards them too.
        #[test]
        #[ignore = "WAL recovery groups records globally, not per transaction; fixed by the WAL commit-grouping work"]
        fn auto_committed_write_survives_later_rollback_and_reopen() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("db");
            let before = {
                let db = open(&path);
                let mut session = db.session();
                session.execute_sparql(SEED).unwrap();
                let before = default_graph(&session);
                session.begin_transaction().unwrap();
                run_all_update_forms(&session);
                session.rollback().unwrap();
                db.close().unwrap();
                before
            };

            let db = open(&path);
            assert_eq!(default_graph(&db.session()), before);
        }

        #[test]
        fn committed_updates_survive_reopen() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("db");
            {
                let db = open(&path);
                let mut session = db.session();
                seed_committed(&mut session);
                session.begin_transaction().unwrap();
                run_all_update_forms(&session);
                session.commit().unwrap();
                db.close().unwrap();
            }

            let db = open(&path);
            let session = db.session();
            assert_eq!(default_graph(&session), expected_default_after_updates());
            assert_eq!(named_graph(&session), expected_named_after_updates());
        }
    }
}
