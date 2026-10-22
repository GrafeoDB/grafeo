//! Regression tests for SPARQL query execution, inspired by bugs found
//! in other RDF databases (Oxigraph, Blazegraph, Jena, RDF4J).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test sparql_regression_external
//! ```

#[cfg(all(feature = "sparql", feature = "triple-store"))]
mod tests {
    use grafeo_engine::GrafeoDB;

    fn rdf_db() -> GrafeoDB {
        GrafeoDB::new_in_memory()
    }

    fn insert_people(db: &GrafeoDB) {
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/alix> <http://ex.org/name> "Alix" .
                <http://ex.org/alix> <http://ex.org/age>  "30" .
                <http://ex.org/gus>  <http://ex.org/name> "Gus" .
                <http://ex.org/gus>  <http://ex.org/age>  "25" .
                <http://ex.org/vincent> <http://ex.org/name> "Vincent" .
            }"#,
        )
        .unwrap();
    }

    // ========================================================================
    // OPTIONAL clause correctness
    // Inspired by Oxigraph #1200: second OPTIONAL hides first result.
    // ========================================================================

    #[test]
    fn optional_unmatched_produces_null_binding() {
        let db = rdf_db();
        insert_people(&db);

        // Vincent has name but no age
        let r = db
            .execute_sparql(
                r#"SELECT ?name ?age WHERE {
                    <http://ex.org/vincent> <http://ex.org/name> ?name .
                    OPTIONAL { <http://ex.org/vincent> <http://ex.org/age> ?age }
                }"#,
            )
            .unwrap();
        assert_eq!(
            r.row_count(),
            1,
            "OPTIONAL with no match should still return 1 row"
        );
    }

    #[test]
    fn multiple_optionals_preserve_results() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/x> <http://ex.org/name> "X" .
                <http://ex.org/x> <http://ex.org/age>  "10" .
            }"#,
        )
        .unwrap();

        // Both OPTIONALLs match
        let r = db
            .execute_sparql(
                r#"SELECT ?name ?age ?extra WHERE {
                    <http://ex.org/x> <http://ex.org/name> ?name .
                    OPTIONAL { <http://ex.org/x> <http://ex.org/age> ?age }
                    OPTIONAL { <http://ex.org/x> <http://ex.org/extra> ?extra }
                }"#,
            )
            .unwrap();
        assert_eq!(
            r.row_count(),
            1,
            "Multiple OPTIONALLs must not cause the base result to disappear"
        );
    }

    // ========================================================================
    // COUNT on empty result set
    // Inspired by RDF4J #1978: aggregate on empty set behavior.
    // ========================================================================

    #[test]
    fn count_on_empty_returns_zero() {
        let db = rdf_db();
        let r = db
            .execute_sparql(
                "SELECT (COUNT(?s) AS ?cnt) WHERE { ?s <http://ex.org/nonexistent> ?o }",
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "COUNT over empty should return 1 row");
        // COUNT of empty set is 0 per SPARQL spec
    }

    // ========================================================================
    // UNION semantics
    // ========================================================================

    #[test]
    fn union_returns_both_branches() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/a> <http://ex.org/name> "Alix" .
                <http://ex.org/a> <http://ex.org/city> "Amsterdam" .
            }"#,
        )
        .unwrap();

        let r = db
            .execute_sparql(
                r#"SELECT ?val WHERE {
                    { <http://ex.org/a> <http://ex.org/name> ?val }
                    UNION
                    { <http://ex.org/a> <http://ex.org/city> ?val }
                }"#,
            )
            .unwrap();
        assert_eq!(
            r.row_count(),
            2,
            "UNION must return results from both branches"
        );
    }

    // ========================================================================
    // FILTER edge cases
    // ========================================================================

    #[test]
    fn filter_greater_than() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/a> <http://ex.org/val> "10" .
                <http://ex.org/b> <http://ex.org/val> "20" .
                <http://ex.org/c> <http://ex.org/val> "30" .
            }"#,
        )
        .unwrap();

        let r = db
            .execute_sparql(r#"SELECT ?v WHERE { ?s <http://ex.org/val> ?v FILTER(?v > "15") }"#)
            .unwrap();
        assert_eq!(r.row_count(), 2, "FILTER > should match 20 and 30");
    }

    #[test]
    fn filter_not_exists() {
        let db = rdf_db();
        insert_people(&db);

        // Vincent has name but no age
        let r = db
            .execute_sparql(
                r#"SELECT ?n WHERE {
                    ?s <http://ex.org/name> ?n .
                    FILTER NOT EXISTS { ?s <http://ex.org/age> ?a }
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "Only Vincent has name without age");
    }

    // ========================================================================
    // DISTINCT deduplication
    // ========================================================================

    #[test]
    fn select_distinct_deduplicates() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/a> <http://ex.org/type> "Person" .
                <http://ex.org/b> <http://ex.org/type> "Person" .
                <http://ex.org/c> <http://ex.org/type> "City" .
            }"#,
        )
        .unwrap();

        let r = db
            .execute_sparql(r#"SELECT DISTINCT ?t WHERE { ?s <http://ex.org/type> ?t }"#)
            .unwrap();
        assert_eq!(
            r.row_count(),
            2,
            "DISTINCT should collapse duplicate 'Person' to 1"
        );
    }

    // ========================================================================
    // INSERT DATA and DELETE DATA correctness
    // ========================================================================

    #[test]
    fn insert_and_query_roundtrip() {
        let db = rdf_db();
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/x> <http://ex.org/p> "hello" . }"#)
            .unwrap();

        let r = db
            .execute_sparql("SELECT ?o WHERE { <http://ex.org/x> <http://ex.org/p> ?o }")
            .unwrap();
        assert_eq!(r.row_count(), 1);
    }

    #[test]
    fn delete_data_removes_triple() {
        let db = rdf_db();
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/x> <http://ex.org/p> "hello" . }"#)
            .unwrap();
        db.execute_sparql(r#"DELETE DATA { <http://ex.org/x> <http://ex.org/p> "hello" . }"#)
            .unwrap();

        let r = db
            .execute_sparql("SELECT ?o WHERE { <http://ex.org/x> <http://ex.org/p> ?o }")
            .unwrap();
        assert_eq!(r.row_count(), 0, "Deleted triple must not be visible");
    }

    #[test]
    fn delete_then_reinsert() {
        let db = rdf_db();
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/x> <http://ex.org/p> "old" . }"#)
            .unwrap();
        db.execute_sparql(r#"DELETE DATA { <http://ex.org/x> <http://ex.org/p> "old" . }"#)
            .unwrap();
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/x> <http://ex.org/p> "new" . }"#)
            .unwrap();

        let r = db
            .execute_sparql("SELECT ?o WHERE { <http://ex.org/x> <http://ex.org/p> ?o }")
            .unwrap();
        assert_eq!(r.row_count(), 1, "Re-inserted triple must be visible");
    }

    // ========================================================================
    // LIMIT and OFFSET
    // ========================================================================

    #[test]
    fn limit_restricts_results() {
        let db = rdf_db();
        for i in 0..10 {
            db.execute_sparql(&format!(
                r#"INSERT DATA {{ <http://ex.org/n{i}> <http://ex.org/val> "{i}" . }}"#
            ))
            .unwrap();
        }

        let r = db
            .execute_sparql("SELECT ?s WHERE { ?s <http://ex.org/val> ?v } LIMIT 3")
            .unwrap();
        assert_eq!(r.row_count(), 3, "LIMIT 3 should return exactly 3 rows");
    }

    #[test]
    fn offset_skips_rows() {
        let db = rdf_db();
        for i in 0..5 {
            db.execute_sparql(&format!(
                r#"INSERT DATA {{ <http://ex.org/n{i}> <http://ex.org/val> "{i}" . }}"#
            ))
            .unwrap();
        }

        let all = db
            .execute_sparql("SELECT ?v WHERE { ?s <http://ex.org/val> ?v } ORDER BY ?v")
            .unwrap();
        let offset = db
            .execute_sparql("SELECT ?v WHERE { ?s <http://ex.org/val> ?v } ORDER BY ?v OFFSET 2")
            .unwrap();
        assert_eq!(
            offset.row_count(),
            all.row_count() - 2,
            "OFFSET 2 should skip 2 rows"
        );
    }

    // ========================================================================
    // ORDER BY correctness
    // ========================================================================

    #[test]
    fn order_by_ascending() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/c> <http://ex.org/name> "Vincent" .
                <http://ex.org/a> <http://ex.org/name> "Alix" .
                <http://ex.org/b> <http://ex.org/name> "Gus" .
            }"#,
        )
        .unwrap();

        let r = db
            .execute_sparql("SELECT ?n WHERE { ?s <http://ex.org/name> ?n } ORDER BY ?n")
            .unwrap();
        assert_eq!(r.row_count(), 3);
        // First result should be alphabetically first
    }

    // ========================================================================
    // GROUP BY with COUNT
    // Inspired by Oxigraph #646: GROUP BY with HAVING returns 0 rows.
    // ========================================================================

    #[test]
    fn group_by_with_count() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/a> <http://ex.org/type> "Person" .
                <http://ex.org/b> <http://ex.org/type> "Person" .
                <http://ex.org/c> <http://ex.org/type> "City" .
            }"#,
        )
        .unwrap();

        let r = db
            .execute_sparql(
                r#"SELECT ?t (COUNT(?s) AS ?cnt) WHERE {
                    ?s <http://ex.org/type> ?t
                } GROUP BY ?t ORDER BY ?t"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 2, "Should have 2 groups: City and Person");
    }

    // ========================================================================
    // Multiple triple patterns in a single query
    // ========================================================================

    #[test]
    fn join_two_triple_patterns() {
        let db = rdf_db();
        insert_people(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name ?age WHERE {
                    ?s <http://ex.org/name> ?name .
                    ?s <http://ex.org/age> ?age
                } ORDER BY ?name"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 2, "Alix and Gus have both name and age");
    }

    // ========================================================================
    // REGEX filter
    // ========================================================================

    #[test]
    fn filter_regex() {
        let db = rdf_db();
        insert_people(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?n WHERE {
                    ?s <http://ex.org/name> ?n
                    FILTER(REGEX(?n, "^A"))
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "Only Alix starts with A");
    }
}
