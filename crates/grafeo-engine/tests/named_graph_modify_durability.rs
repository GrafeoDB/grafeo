//! Durability tests for SPARQL DELETE/INSERT WHERE (MODIFY).
//!
//! `RdfModifyOperator` previously emitted no WAL record at all, so a
//! DELETE/INSERT-WHERE MODIFY was not durable even on the default graph and was
//! lost on `wal-directory` replay. These tests open a DB with
//! `StorageFormat::WalDirectory`, run a MODIFY, close, reopen, and assert the
//! mutation survived -- for both a named graph and the default graph.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test named_graph_modify_durability
//! ```

#![allow(missing_docs)]

#[cfg(all(feature = "wal", feature = "sparql", feature = "triple-store"))]
mod durability {
    use grafeo_engine::config::StorageFormat;
    use grafeo_engine::{Config, GrafeoDB, GraphModel};
    use std::path::Path;

    fn open(path: &Path) -> GrafeoDB {
        GrafeoDB::with_config(
            Config::persistent(path)
                .with_graph_model(GraphModel::Rdf)
                .with_storage_format(StorageFormat::WalDirectory),
        )
        .expect("open wal-directory rdf db")
    }

    /// A named-graph MODIFY (delete old value, insert new) must survive a
    /// close/reopen WAL replay, with the resolved graph carried in the record.
    #[test]
    fn named_graph_modify_survives_reopen() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("db");

        {
            let db = open(&path);
            let session = db.session();
            session
                .execute_sparql(
                    r#"INSERT DATA {
                           GRAPH <http://ex.org/g> {
                               <http://ex.org/alix> <http://ex.org/status> "active" .
                           }
                       }"#,
                )
                .expect("seed named graph");
            session
                .execute_sparql(
                    r#"WITH <http://ex.org/g>
                       DELETE { <http://ex.org/alix> <http://ex.org/status> ?s }
                       INSERT { <http://ex.org/alix> <http://ex.org/status> "archived" }
                       WHERE  { <http://ex.org/alix> <http://ex.org/status> ?s }"#,
                )
                .expect("named MODIFY");
            db.close().expect("close");
        }

        let db = open(&path);
        let session = db.session();
        let result = session
            .execute_sparql(
                r#"SELECT ?s WHERE {
                       GRAPH <http://ex.org/g> { <http://ex.org/alix> <http://ex.org/status> ?s }
                   }"#,
            )
            .expect("query after reopen");
        assert_eq!(
            result.row_count(),
            1,
            "named MODIFY result was lost on reopen (RdfModifyOperator emitted no WAL)"
        );
        assert_eq!(
            result.rows()[0][0].to_string().trim_matches('"'),
            "archived",
            "named MODIFY new value was not durable across reopen"
        );
    }

    /// The same hole existed on the default graph: a default-graph MODIFY must
    /// also survive close/reopen.
    #[test]
    fn default_graph_modify_survives_reopen() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("db");

        {
            let db = open(&path);
            let session = db.session();
            session
                .execute_sparql(
                    r#"INSERT DATA { <http://ex.org/item> <http://ex.org/version> "1" . }"#,
                )
                .expect("seed default graph");
            session
                .execute_sparql(
                    r#"DELETE { <http://ex.org/item> <http://ex.org/version> ?v }
                       INSERT { <http://ex.org/item> <http://ex.org/version> "2" }
                       WHERE  { <http://ex.org/item> <http://ex.org/version> ?v }"#,
                )
                .expect("default MODIFY");
            db.close().expect("close");
        }

        let db = open(&path);
        let session = db.session();
        let result = session
            .execute_sparql(
                r#"SELECT ?v WHERE { <http://ex.org/item> <http://ex.org/version> ?v }"#,
            )
            .expect("query after reopen");
        assert_eq!(
            result.row_count(),
            1,
            "default-graph MODIFY result was lost on reopen"
        );
        assert_eq!(
            result.rows()[0][0].to_string().trim_matches('"'),
            "2",
            "default-graph MODIFY new value was not durable across reopen"
        );
    }
}
