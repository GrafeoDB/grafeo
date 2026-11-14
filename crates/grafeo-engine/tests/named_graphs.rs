//! Integration tests for SPARQL named graphs (RDF Datasets).
//!
//! Exercises: SPARQL translator graph_context_stack, RdfTripleScanOperator
//! graph-aware mode, RdfCreateGraphOperator, RdfDropGraphOperator,
//! RdfClearGraphOperator, and RdfStore named graph API.

#[cfg(all(feature = "sparql", feature = "triple-store"))]
mod tests {
    use grafeo_engine::{Config, GrafeoDB, GraphModel};

    fn rdf_db() -> GrafeoDB {
        GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap()
    }

    // ==================== CREATE / DROP GRAPH ====================

    #[test]
    fn create_graph_succeeds() {
        let db = rdf_db();
        let session = db.session();
        let result = session.execute_sparql("CREATE GRAPH <http://ex.org/g1>");
        assert!(result.is_ok());
    }

    #[test]
    fn create_duplicate_graph_errors() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql("CREATE GRAPH <http://ex.org/g1>")
            .unwrap();
        // Non-silent CREATE of existing graph should fail
        let result = session.execute_sparql("CREATE GRAPH <http://ex.org/g1>");
        assert!(result.is_err());
    }

    #[test]
    fn create_duplicate_graph_silent_ok() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql("CREATE GRAPH <http://ex.org/g1>")
            .unwrap();
        let result = session.execute_sparql("CREATE SILENT GRAPH <http://ex.org/g1>");
        assert!(result.is_ok());
    }

    #[test]
    fn drop_graph_succeeds() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql("CREATE GRAPH <http://ex.org/g1>")
            .unwrap();
        let result = session.execute_sparql("DROP GRAPH <http://ex.org/g1>");
        assert!(result.is_ok());
    }

    #[test]
    fn drop_nonexistent_graph_errors() {
        let db = rdf_db();
        let session = db.session();
        let result = session.execute_sparql("DROP GRAPH <http://ex.org/nope>");
        assert!(result.is_err());
    }

    #[test]
    fn drop_nonexistent_graph_silent_ok() {
        let db = rdf_db();
        let session = db.session();
        let result = session.execute_sparql("DROP SILENT GRAPH <http://ex.org/nope>");
        assert!(result.is_ok());
    }

    #[test]
    fn drop_default_clears_default_graph() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/a> <http://ex.org/p> "1" .
                }"#,
            )
            .unwrap();

        session.execute_sparql("DROP DEFAULT").unwrap();

        let result = session
            .execute_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
            .unwrap();
        assert_eq!(
            result.row_count(),
            0,
            "DROP DEFAULT should clear the default graph"
        );
    }

    // ==================== CLEAR GRAPH ====================

    #[test]
    fn clear_default_graph() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/a> <http://ex.org/p> "1" .
                    <http://ex.org/b> <http://ex.org/p> "2" .
                }"#,
            )
            .unwrap();

        session.execute_sparql("CLEAR DEFAULT").unwrap();

        let result = session
            .execute_sparql("SELECT ?s WHERE { ?s ?p ?o }")
            .unwrap();
        assert_eq!(result.row_count(), 0);
    }

    #[test]
    fn clear_named_graph_preserves_default() {
        let db = rdf_db();
        let session = db.session();

        // Insert into default graph
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/a> <http://ex.org/p> "1" .
                }"#,
            )
            .unwrap();

        // Create a named graph and populate it via store API
        let rdf = db.rdf_store();
        rdf.graph_or_create("http://ex.org/g1");
        if let Some(g) = rdf.graph_latest("http://ex.org/g1") {
            use grafeo_core::graph::rdf::{Literal, Term, Triple};
            g.insert(Triple::new(
                Term::iri("http://ex.org/b"),
                Term::iri("http://ex.org/p"),
                Term::Literal(Literal::simple("2")),
            ));
        }

        // Clear named graph only
        session
            .execute_sparql("CLEAR GRAPH <http://ex.org/g1>")
            .unwrap();

        // Default graph still has data
        let result = session
            .execute_sparql("SELECT ?s WHERE { ?s ?p ?o }")
            .unwrap();
        assert_eq!(result.row_count(), 1, "Default graph should be untouched");

        // Named graph is empty
        let named = session
            .execute_sparql(
                r#"SELECT ?s WHERE {
                    GRAPH <http://ex.org/g1> { ?s ?p ?o }
                }"#,
            )
            .unwrap();
        assert_eq!(named.row_count(), 0, "Named graph should be cleared");
    }

    // ==================== GRAPH clause queries ====================

    #[test]
    fn graph_clause_queries_named_graph() {
        let db = rdf_db();

        // Populate named graph via store API
        let rdf = db.rdf_store();
        let g = rdf.graph_or_create("http://ex.org/g1");
        {
            use grafeo_core::graph::rdf::{Literal, Term, Triple};
            g.insert(Triple::new(
                Term::iri("http://ex.org/alix"),
                Term::iri("http://ex.org/name"),
                Term::Literal(Literal::simple("Alix")),
            ));
        }

        let session = db.session();
        let result = session
            .execute_sparql(
                r#"SELECT ?s ?name WHERE {
                    GRAPH <http://ex.org/g1> { ?s <http://ex.org/name> ?name }
                }"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn graph_variable_scans_all_graphs() {
        let db = rdf_db();
        let session = db.session();

        // Insert into default graph
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                }"#,
            )
            .unwrap();

        // Populate named graphs via store API
        let rdf = db.rdf_store();
        {
            use grafeo_core::graph::rdf::{Literal, Term, Triple};
            let g1 = rdf.graph_or_create("http://ex.org/g1");
            g1.insert(Triple::new(
                Term::iri("http://ex.org/gus"),
                Term::iri("http://ex.org/name"),
                Term::Literal(Literal::simple("Gus")),
            ));
            let g2 = rdf.graph_or_create("http://ex.org/g2");
            g2.insert(Triple::new(
                Term::iri("http://ex.org/harm"),
                Term::iri("http://ex.org/name"),
                Term::Literal(Literal::simple("Harm")),
            ));
        }

        // GRAPH ?g scans named graphs only (per SPARQL 1.1 spec sec 13.3)
        let result = session
            .execute_sparql(
                r#"SELECT ?g ?s ?name WHERE {
                    GRAPH ?g { ?s <http://ex.org/name> ?name }
                }"#,
            )
            .unwrap();

        assert!(
            result.row_count() >= 2,
            "Expected >= 2 rows from named graphs, got {}",
            result.row_count()
        );
    }

    // ==================== INSERT DATA + GRAPH clause ====================

    #[test]
    fn insert_data_graph_routes_to_named_graph() {
        let db = rdf_db();
        let session = db.session();

        // INSERT DATA into a named graph via SPARQL
        session
            .execute_sparql(
                r#"INSERT DATA {
                    GRAPH <http://ex.org/g1> {
                        <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    }
                }"#,
            )
            .unwrap();

        // Default graph should be empty
        let default_result = session
            .execute_sparql("SELECT ?s WHERE { ?s ?p ?o }")
            .unwrap();
        assert_eq!(
            default_result.row_count(),
            0,
            "Default graph should be empty after INSERT DATA GRAPH"
        );

        // Named graph should have the triple
        let named_result = session
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    GRAPH <http://ex.org/g1> {
                        <http://ex.org/alix> <http://ex.org/name> ?name
                    }
                }"#,
            )
            .unwrap();
        assert_eq!(
            named_result.row_count(),
            1,
            "Named graph should have the inserted triple"
        );
    }

    #[test]
    fn delete_data_graph_removes_from_named_graph() {
        let db = rdf_db();
        let session = db.session();

        // Insert into named graph
        session
            .execute_sparql(
                r#"INSERT DATA {
                    GRAPH <http://ex.org/g1> {
                        <http://ex.org/alix> <http://ex.org/name> "Alix" .
                        <http://ex.org/gus> <http://ex.org/name> "Gus" .
                    }
                }"#,
            )
            .unwrap();

        // Delete one triple from the named graph
        session
            .execute_sparql(
                r#"DELETE DATA {
                    GRAPH <http://ex.org/g1> {
                        <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    }
                }"#,
            )
            .unwrap();

        // Only Gus should remain
        let result = session
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    GRAPH <http://ex.org/g1> { ?s <http://ex.org/name> ?name }
                }"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 1, "Should have 1 triple after delete");
    }

    #[test]
    fn insert_data_graph_does_not_leak_to_default() {
        let db = rdf_db();
        let session = db.session();

        // Insert into default and named
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/default> <http://ex.org/p> "default" .
                }"#,
            )
            .unwrap();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    GRAPH <http://ex.org/g1> {
                        <http://ex.org/named> <http://ex.org/p> "named" .
                    }
                }"#,
            )
            .unwrap();

        // Default graph has exactly 1 triple
        let default_result = session
            .execute_sparql("SELECT ?s WHERE { ?s ?p ?o }")
            .unwrap();
        assert_eq!(default_result.row_count(), 1, "Default graph: 1 triple");

        // Named graph has exactly 1 triple
        let named_result = session
            .execute_sparql(
                r#"SELECT ?s WHERE {
                    GRAPH <http://ex.org/g1> { ?s ?p ?o }
                }"#,
            )
            .unwrap();
        assert_eq!(named_result.row_count(), 1, "Named graph: 1 triple");
    }

    // ==================== Graph clause isolation ====================

    #[test]
    fn graph_clause_isolates_from_default() {
        let db = rdf_db();
        let session = db.session();

        // Insert into default graph
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                }"#,
            )
            .unwrap();

        // Populate named graph via store API
        let rdf = db.rdf_store();
        {
            use grafeo_core::graph::rdf::{Literal, Term, Triple};
            let g = rdf.graph_or_create("http://ex.org/g1");
            g.insert(Triple::new(
                Term::iri("http://ex.org/gus"),
                Term::iri("http://ex.org/name"),
                Term::Literal(Literal::simple("Gus")),
            ));
        }

        // Default graph: only Alix
        let default_result = session
            .execute_sparql(r#"SELECT ?s WHERE { ?s <http://ex.org/name> ?name }"#)
            .unwrap();
        assert_eq!(
            default_result.row_count(),
            1,
            "Default graph should have 1 triple"
        );

        // Named graph: only Gus
        let named_result = session
            .execute_sparql(
                r#"SELECT ?s WHERE {
                    GRAPH <http://ex.org/g1> { ?s <http://ex.org/name> ?name }
                }"#,
            )
            .unwrap();
        assert_eq!(
            named_result.row_count(),
            1,
            "Named graph should have 1 triple"
        );
    }

    // ==================== COPY / MOVE / ADD ====================

    #[test]
    fn copy_named_graph_preserves_source() {
        let db = rdf_db();
        let session = db.session();

        session
            .execute_sparql(
                r#"INSERT DATA {
                    GRAPH <http://ex.org/src> {
                        <http://ex.org/a> <http://ex.org/p> "1" .
                        <http://ex.org/b> <http://ex.org/q> "2" .
                    }
                }"#,
            )
            .unwrap();

        session
            .execute_sparql("COPY <http://ex.org/src> TO <http://ex.org/dst>")
            .unwrap();

        // Source retained
        let src = session
            .execute_sparql(
                r#"SELECT ?s WHERE {
                    GRAPH <http://ex.org/src> { ?s ?p ?o }
                }"#,
            )
            .unwrap();
        assert_eq!(src.row_count(), 2, "Source should still have 2 triples");

        // Destination has copy
        let dst = session
            .execute_sparql(
                r#"SELECT ?s WHERE {
                    GRAPH <http://ex.org/dst> { ?s ?p ?o }
                }"#,
            )
            .unwrap();
        assert_eq!(dst.row_count(), 2, "Dest should have 2 triples");
    }

    #[test]
    fn move_named_graph_removes_source() {
        let db = rdf_db();
        let session = db.session();

        session
            .execute_sparql(
                r#"INSERT DATA {
                    GRAPH <http://ex.org/src> {
                        <http://ex.org/a> <http://ex.org/p> "val" .
                    }
                }"#,
            )
            .unwrap();

        session
            .execute_sparql("MOVE <http://ex.org/src> TO <http://ex.org/dst>")
            .unwrap();

        // Source gone
        let src = session
            .execute_sparql(
                r#"SELECT ?s WHERE {
                    GRAPH <http://ex.org/src> { ?s ?p ?o }
                }"#,
            )
            .unwrap();
        assert_eq!(src.row_count(), 0, "Source should be empty after MOVE");

        // Destination has data
        let dst = session
            .execute_sparql(
                r#"SELECT ?s WHERE {
                    GRAPH <http://ex.org/dst> { ?s ?p ?o }
                }"#,
            )
            .unwrap();
        assert_eq!(dst.row_count(), 1, "Dest should have 1 triple");
    }

    #[test]
    fn add_merges_into_destination() {
        let db = rdf_db();
        let session = db.session();

        session
            .execute_sparql(
                r#"INSERT DATA {
                    GRAPH <http://ex.org/g1> {
                        <http://ex.org/a> <http://ex.org/p> "from-g1" .
                    }
                    GRAPH <http://ex.org/g2> {
                        <http://ex.org/b> <http://ex.org/q> "from-g2" .
                    }
                }"#,
            )
            .unwrap();

        session
            .execute_sparql("ADD <http://ex.org/g1> TO <http://ex.org/g2>")
            .unwrap();

        // g1 unchanged
        let g1 = session
            .execute_sparql(
                r#"SELECT ?s WHERE {
                    GRAPH <http://ex.org/g1> { ?s ?p ?o }
                }"#,
            )
            .unwrap();
        assert_eq!(g1.row_count(), 1, "g1 should still have 1 triple");

        // g2 has union
        let g2 = session
            .execute_sparql(
                r#"SELECT ?s WHERE {
                    GRAPH <http://ex.org/g2> { ?s ?p ?o }
                }"#,
            )
            .unwrap();
        assert_eq!(g2.row_count(), 2, "g2 should have 2 triples after ADD");
    }

    #[test]
    fn copy_nonexistent_source_errors() {
        let db = rdf_db();
        let session = db.session();
        let result = session.execute_sparql("COPY <http://ex.org/nope> TO <http://ex.org/dst>");
        assert!(result.is_err(), "COPY from nonexistent graph should error");
    }

    #[test]
    fn copy_silent_nonexistent_source_ok() {
        let db = rdf_db();
        let session = db.session();
        let result =
            session.execute_sparql("COPY SILENT <http://ex.org/nope> TO <http://ex.org/dst>");
        assert!(
            result.is_ok(),
            "COPY SILENT from nonexistent graph should succeed"
        );
    }
}
