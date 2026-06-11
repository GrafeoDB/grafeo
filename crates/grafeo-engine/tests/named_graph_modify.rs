//! Integration tests for named-graph SPARQL DELETE/INSERT WHERE (MODIFY) routing.
//!
//! Companion to the declarative oracles in
//! `tests/spec/rdf/sparql/update_advanced.gtest`. These cover behaviours the
//! `.gtest` harness cannot express: fail-closed loud errors for unsupported
//! variants (USING, variable graph targets), and the white-box invariant that a
//! zero-match named delete does not create an empty named sub-store.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test named_graph_modify
//! ```

#![allow(missing_docs)]

#[cfg(all(feature = "sparql", feature = "triple-store"))]
mod tests {
    use grafeo_engine::{Config, GrafeoDB, GraphModel};

    fn rdf_db() -> GrafeoDB {
        GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap()
    }

    // ==================== mis-write / routing (store-level) ====================

    /// The headline corruption: a GRAPH-wrapped INSERT-via-Modify must write to
    /// the named graph and must NOT leak into the default graph.
    #[test]
    fn named_insert_via_modify_writes_named_not_default() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(r#"INSERT DATA { <http://ex.org/trigger> <http://ex.org/p> "go" . }"#)
            .unwrap();
        session
            .execute_sparql(
                r#"INSERT { GRAPH <http://ex.org/g> { <http://ex.org/s> <http://ex.org/p> "v" } }
                   WHERE  { <http://ex.org/trigger> <http://ex.org/p> ?x }"#,
            )
            .unwrap();

        // The named graph received the triple ...
        let named = session
            .execute_sparql(
                r#"SELECT ?s WHERE { GRAPH <http://ex.org/g> { ?s <http://ex.org/p> "v" } }"#,
            )
            .unwrap();
        assert_eq!(
            named.row_count(),
            1,
            "named INSERT-via-Modify must land in the named graph"
        );

        // ... and the default graph did NOT (no silent mis-write).
        let default = session
            .execute_sparql(r#"SELECT ?s WHERE { ?s <http://ex.org/p> "v" }"#)
            .unwrap();
        assert_eq!(
            default.row_count(),
            0,
            "named INSERT-via-Modify must not leak into the default graph"
        );
    }

    // ==================== non-creating delete target ====================

    /// A DELETE WHERE targeting a graph that does not exist must be a clean
    /// no-op that does not materialise an empty named sub-store.
    #[test]
    fn named_delete_where_does_not_create_empty_substore() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"DELETE WHERE {
                       GRAPH <http://ex.org/missing> { <http://ex.org/a> <http://ex.org/p> ?o }
                   }"#,
            )
            .unwrap();
        let rdf = db.rdf_store();
        assert!(
            rdf.graph("http://ex.org/missing").is_none(),
            "a zero-match named DELETE WHERE must not create an empty named sub-store"
        );
    }

    /// A WITH-scoped Modify whose target graph does not exist must not create it
    /// on the delete path.
    #[test]
    fn named_modify_delete_does_not_create_empty_substore() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(r#"INSERT DATA { <http://ex.org/trigger> <http://ex.org/p> "go" . }"#)
            .unwrap();
        session
            .execute_sparql(
                r#"WITH <http://ex.org/missing>
                   DELETE { <http://ex.org/a> <http://ex.org/p> ?o }
                   WHERE  { <http://ex.org/a> <http://ex.org/p> ?o }"#,
            )
            .unwrap();
        let rdf = db.rdf_store();
        assert!(
            rdf.graph("http://ex.org/missing").is_none(),
            "a WITH-scoped delete against a missing graph must not create it"
        );
    }

    // ==================== M5: fail-closed loud errors ====================

    /// USING / USING NAMED is parsed but cannot be honoured; it must error
    /// loudly rather than silently drop the clause and mis-scope the update.
    #[test]
    fn modify_using_clause_errors_loudly() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(r#"INSERT DATA { <http://ex.org/a> <http://ex.org/p> "1" . }"#)
            .unwrap();
        let result = session.execute_sparql(
            r#"DELETE { <http://ex.org/a> <http://ex.org/p> ?o }
               USING <http://ex.org/g>
               WHERE  { <http://ex.org/a> <http://ex.org/p> ?o }"#,
        );
        assert!(
            result.is_err(),
            "USING in a SPARQL Modify must error loudly, not silently no-op"
        );
    }

    /// A variable graph target (`GRAPH ?g`) in a DELETE/INSERT template cannot
    /// be routed to a concrete graph; fail closed rather than misroute.
    #[test]
    fn modify_variable_graph_template_errors_loudly() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                       GRAPH <http://ex.org/g> { <http://ex.org/a> <http://ex.org/p> "1" . }
                   }"#,
            )
            .unwrap();
        let result = session.execute_sparql(
            r#"DELETE { GRAPH ?g { <http://ex.org/a> <http://ex.org/p> ?o } }
               WHERE  { GRAPH ?g { <http://ex.org/a> <http://ex.org/p> ?o } }"#,
        );
        assert!(
            result.is_err(),
            "a variable graph target must error loudly (fail-closed)"
        );
    }

    /// The variable-graph guard must fire even when the WHERE binds zero rows,
    /// so an unsupported template can never degrade to a silent no-op.
    #[test]
    fn modify_variable_graph_template_errors_even_with_zero_bindings() {
        let db = rdf_db();
        let session = db.session();
        let result = session.execute_sparql(
            r#"DELETE { GRAPH ?g { <http://ex.org/a> <http://ex.org/p> ?o } }
               WHERE  { GRAPH ?g { <http://ex.org/a> <http://ex.org/p> ?o } }"#,
        );
        assert!(
            result.is_err(),
            "variable-graph guard must fire even with zero WHERE bindings (fail-closed)"
        );
    }

    /// The same fail-closed guard for the pure `DELETE WHERE { GRAPH ?g { .. } }`
    /// form, which flows through the translator extraction + pattern executor.
    #[test]
    fn delete_where_variable_graph_errors_loudly() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                       GRAPH <http://ex.org/g> { <http://ex.org/a> <http://ex.org/p> "1" . }
                   }"#,
            )
            .unwrap();
        let result = session.execute_sparql(
            r#"DELETE WHERE { GRAPH ?g { <http://ex.org/a> <http://ex.org/p> ?o } }"#,
        );
        assert!(
            result.is_err(),
            "DELETE WHERE with a variable graph must error loudly (fail-closed)"
        );
    }
}
