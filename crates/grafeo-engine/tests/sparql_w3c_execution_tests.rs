//! End-to-end execution tests aligned with the W3C SPARQL 1.1 Query Language
//! specification. Each section tests actual query execution against an in-memory
//! RDF store, verifying both result counts and value correctness.
//!
//! Spec reference: <https://www.w3.org/TR/sparql11-query/>
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test sparql_w3c_execution_tests
//! ```

#[cfg(all(feature = "sparql", feature = "triple-store"))]
mod tests {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    fn rdf_db() -> GrafeoDB {
        GrafeoDB::new_in_memory()
    }

    /// Insert a common set of triples used by many tests.
    fn insert_foaf_data(db: &GrafeoDB) {
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/alix> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                <http://ex.org/alix> <http://xmlns.com/foaf/0.1/name> "Alix" .
                <http://ex.org/alix> <http://xmlns.com/foaf/0.1/age> "30" .
                <http://ex.org/alix> <http://xmlns.com/foaf/0.1/knows> <http://ex.org/gus> .
                <http://ex.org/alix> <http://xmlns.com/foaf/0.1/mbox> "alix@example.org" .

                <http://ex.org/gus> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                <http://ex.org/gus> <http://xmlns.com/foaf/0.1/name> "Gus" .
                <http://ex.org/gus> <http://xmlns.com/foaf/0.1/age> "25" .
                <http://ex.org/gus> <http://xmlns.com/foaf/0.1/knows> <http://ex.org/alix> .

                <http://ex.org/vincent> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                <http://ex.org/vincent> <http://xmlns.com/foaf/0.1/name> "Vincent" .

                <http://ex.org/amsterdam> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://ex.org/City> .
                <http://ex.org/amsterdam> <http://xmlns.com/foaf/0.1/name> "Amsterdam" .

                <http://ex.org/alix> <http://ex.org/livesIn> <http://ex.org/amsterdam> .
                <http://ex.org/gus> <http://ex.org/livesIn> <http://ex.org/amsterdam> .
            }"#,
        )
        .unwrap();
    }

    // ====================================================================
    // 2 - Making Simple Queries
    // ====================================================================

    #[test]
    fn sec2_select_wildcard() {
        let db = rdf_db();
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/a> <http://ex.org/p> "hello" . }"#)
            .unwrap();

        let r = db.execute_sparql("SELECT * WHERE { ?s ?p ?o }").unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(r.columns.len(), 3);
    }

    #[test]
    fn sec2_select_specific_vars() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person>
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 3, "Should find Alix, Gus, Vincent");
    }

    // ====================================================================
    // 5 - Graph Patterns
    // ====================================================================

    #[test]
    fn sec5_basic_graph_pattern_join() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name ?age WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://xmlns.com/foaf/0.1/age> ?age
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 2, "Only Alix and Gus have both name and age");
    }

    #[test]
    fn sec5_empty_where_clause() {
        let db = rdf_db();
        insert_foaf_data(&db);

        // Empty WHERE should return one empty solution
        let r = db.execute_sparql("SELECT (1 AS ?one) WHERE { }");
        // Depending on implementation, may return 1 row or error
        // The spec says empty BGP matches with one empty solution
        assert!(r.is_ok(), "Empty WHERE clause: {r:?}");
    }

    // ====================================================================
    // 6 - OPTIONAL
    // ====================================================================

    #[test]
    fn sec6_optional_preserves_non_matching() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name ?age WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                    OPTIONAL { ?s <http://xmlns.com/foaf/0.1/age> ?age }
                }"#,
            )
            .unwrap();
        assert_eq!(
            r.row_count(),
            3,
            "All 3 people returned; Vincent has NULL age"
        );
    }

    #[test]
    fn sec6_nested_optional() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name ?age ?mbox WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                    OPTIONAL {
                        ?s <http://xmlns.com/foaf/0.1/age> ?age .
                        OPTIONAL { ?s <http://xmlns.com/foaf/0.1/mbox> ?mbox }
                    }
                }"#,
            )
            .unwrap();
        assert_eq!(
            r.row_count(),
            3,
            "All 3 people; only Alix has both age and mbox"
        );
    }

    // ====================================================================
    // 7 - UNION
    // ====================================================================

    #[test]
    fn sec7_union_disjoint_patterns() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?thing ?name WHERE {
                    {
                        ?thing <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                        ?thing <http://xmlns.com/foaf/0.1/name> ?name
                    }
                    UNION
                    {
                        ?thing <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://ex.org/City> .
                        ?thing <http://xmlns.com/foaf/0.1/name> ?name
                    }
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 4, "3 persons + 1 city = 4 results");
    }

    // ====================================================================
    // 8 - Negation
    // ====================================================================

    #[test]
    fn sec8_filter_not_exists_execution() {
        let db = rdf_db();
        insert_foaf_data(&db);

        // Find people who have no age
        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                    FILTER NOT EXISTS { ?s <http://xmlns.com/foaf/0.1/age> ?age }
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "Only Vincent has no age");
    }

    #[test]
    fn sec8_minus_execution() {
        let db = rdf_db();
        insert_foaf_data(&db);

        // All person names MINUS those who know someone
        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                    MINUS { ?s <http://xmlns.com/foaf/0.1/knows> ?other }
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "Only Vincent does not know anyone");
    }

    // ====================================================================
    // 10 - BIND
    // ====================================================================

    #[test]
    fn sec10_bind_concat() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/alix> <http://ex.org/first> "Alix" .
                <http://ex.org/alix> <http://ex.org/last> "Vega" .
            }"#,
        )
        .unwrap();

        let r = db
            .execute_sparql(
                r#"SELECT ?full WHERE {
                    ?s <http://ex.org/first> ?f .
                    ?s <http://ex.org/last> ?l .
                    BIND(CONCAT(?f, " ", ?l) AS ?full)
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        let val = r.rows()[0][0].to_string();
        assert!(
            val.contains("Alix") && val.contains("Vega"),
            "BIND CONCAT should produce 'Alix Vega', got: {val}"
        );
    }

    // ====================================================================
    // 10.2 - VALUES
    // ====================================================================

    #[test]
    fn sec10_values_filter() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    VALUES ?s { <http://ex.org/alix> <http://ex.org/gus> }
                    ?s <http://xmlns.com/foaf/0.1/name> ?name
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 2, "VALUES restricts to Alix and Gus");
    }

    // ====================================================================
    // 11 - Aggregates
    // ====================================================================

    #[test]
    fn sec11_count_star_execution() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT (COUNT(*) AS ?total) WHERE {
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person>
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "COUNT(*) returns 1 row");
    }

    #[test]
    fn sec11_count_with_group_by() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?type (COUNT(?s) AS ?cnt)
                WHERE { ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?type }
                GROUP BY ?type
                ORDER BY ?type"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 2, "Two types: City and Person");
    }

    #[test]
    fn sec11_sum_aggregate() {
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
            .execute_sparql("SELECT (SUM(?v) AS ?total) WHERE { ?s <http://ex.org/val> ?v }")
            .unwrap();
        assert_eq!(r.row_count(), 1, "SUM should return 1 row");
    }

    #[test]
    fn sec11_min_max_aggregate() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT (MIN(?age) AS ?youngest) (MAX(?age) AS ?oldest)
                WHERE { ?s <http://xmlns.com/foaf/0.1/age> ?age }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "MIN/MAX should return 1 row");
    }

    #[test]
    fn sec11_avg_aggregate() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT (AVG(?age) AS ?avgAge)
                WHERE { ?s <http://xmlns.com/foaf/0.1/age> ?age }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "AVG should return 1 row");
    }

    #[test]
    fn sec11_group_concat_aggregate() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT (GROUP_CONCAT(?name; SEPARATOR=", ") AS ?names)
                WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person>
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "GROUP_CONCAT returns 1 row");
        let val = r.rows()[0][0].to_string();
        assert!(
            val.contains("Alix"),
            "GROUP_CONCAT should contain Alix, got: {val}"
        );
        assert!(
            val.contains("Gus"),
            "GROUP_CONCAT should contain Gus, got: {val}"
        );
    }

    #[test]
    fn sec11_sample_aggregate() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT (SAMPLE(?name) AS ?example)
                WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person>
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "SAMPLE returns 1 row");
    }

    #[test]
    fn sec11_having_filters_groups() {
        let db = rdf_db();
        insert_foaf_data(&db);

        // Only types with more than 1 instance
        let r = db
            .execute_sparql(
                r#"SELECT ?type (COUNT(?s) AS ?cnt)
                WHERE { ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?type }
                GROUP BY ?type
                HAVING (COUNT(?s) > 1)"#,
            )
            .unwrap();
        assert_eq!(
            r.row_count(),
            1,
            "Only Person has >1 instances (City has 1)"
        );
    }

    #[test]
    fn sec11_count_on_empty_set() {
        let db = rdf_db();
        let r = db
            .execute_sparql(
                "SELECT (COUNT(?s) AS ?cnt) WHERE { ?s <http://ex.org/nonexistent> ?o }",
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "COUNT over empty set returns 1 row");
    }

    // ====================================================================
    // 15 - Solution Modifiers
    // ====================================================================

    #[test]
    fn sec15_distinct_execution() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT DISTINCT ?city WHERE {
                    ?s <http://ex.org/livesIn> ?city
                }"#,
            )
            .unwrap();
        assert_eq!(
            r.row_count(),
            1,
            "Both Alix and Gus live in Amsterdam; DISTINCT collapses to 1"
        );
    }

    #[test]
    fn sec15_order_by_ascending() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person>
                } ORDER BY ?name"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 3);
        let names: Vec<String> = r.rows().iter().map(|row| row[0].to_string()).collect();
        assert!(
            names[0] <= names[1] && names[1] <= names[2],
            "ORDER BY ASC should be sorted: {names:?}"
        );
    }

    #[test]
    fn sec15_order_by_descending() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person>
                } ORDER BY DESC(?name)"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 3);
        let names: Vec<String> = r.rows().iter().map(|row| row[0].to_string()).collect();
        assert!(
            names[0] >= names[1] && names[1] >= names[2],
            "ORDER BY DESC should be reverse sorted: {names:?}"
        );
    }

    #[test]
    fn sec15_order_by_multiple_keys() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/a> <http://ex.org/dept> "Engineering" .
                <http://ex.org/a> <http://ex.org/name> "Vincent" .
                <http://ex.org/b> <http://ex.org/dept> "Engineering" .
                <http://ex.org/b> <http://ex.org/name> "Alix" .
                <http://ex.org/c> <http://ex.org/dept> "Design" .
                <http://ex.org/c> <http://ex.org/name> "Gus" .
            }"#,
        )
        .unwrap();

        let r = db
            .execute_sparql(
                r#"SELECT ?dept ?name WHERE {
                    ?s <http://ex.org/dept> ?dept .
                    ?s <http://ex.org/name> ?name
                } ORDER BY ?dept ?name"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 3);
        // Design < Engineering, and within Engineering: Alix < Vincent
        let first_dept = r.rows()[0][0].to_string();
        assert!(
            first_dept.contains("Design"),
            "First dept should be Design, got: {first_dept}"
        );
    }

    #[test]
    fn sec15_limit_execution() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name
                } LIMIT 2"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 2, "LIMIT 2 returns at most 2 rows");
    }

    #[test]
    fn sec15_offset_execution() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let all = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person>
                } ORDER BY ?name"#,
            )
            .unwrap();
        let offset = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person>
                } ORDER BY ?name OFFSET 1"#,
            )
            .unwrap();
        assert_eq!(
            offset.row_count(),
            all.row_count() - 1,
            "OFFSET 1 skips first row"
        );
    }

    #[test]
    fn sec15_limit_offset_combined() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person>
                } ORDER BY ?name LIMIT 1 OFFSET 1"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "LIMIT 1 OFFSET 1 returns exactly 1 row");
    }

    // ====================================================================
    // 16 - Query Forms
    // ====================================================================

    #[test]
    fn sec16_ask_true() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"ASK {
                    ?s <http://xmlns.com/foaf/0.1/name> "Alix"
                }"#,
            )
            .unwrap();
        // One row with one Boolean, in the column the SPARQL 1.1 Query
        // Results JSON Format names an ASK result (section 4.2).
        assert_eq!(r.columns, ["boolean"]);
        assert_eq!(r.rows(), [[Value::Bool(true)]]);
    }

    #[test]
    fn sec16_ask_false() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"ASK {
                    ?s <http://xmlns.com/foaf/0.1/name> "Nonexistent"
                }"#,
            )
            .unwrap();
        assert_eq!(r.columns, ["boolean"]);
        assert_eq!(r.rows(), [[Value::Bool(false)]]);
    }

    #[test]
    fn sec16_construct_returns_triples() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"CONSTRUCT {
                    ?s <http://ex.org/hasName> ?name
                } WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person>
                }"#,
            )
            .unwrap();
        assert!(
            r.row_count() >= 3,
            "CONSTRUCT should produce at least 3 triples (one per person)"
        );
    }

    #[test]
    fn sec16_describe_returns_triples() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db.execute_sparql("DESCRIBE <http://ex.org/alix>").unwrap();
        assert!(
            r.row_count() >= 1,
            "DESCRIBE should return at least 1 triple about Alix"
        );
    }

    // ====================================================================
    // SPARQL Update Operations
    // ====================================================================

    #[test]
    fn update_insert_data_and_query() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/mia> <http://xmlns.com/foaf/0.1/name> "Mia" .
                <http://ex.org/mia> <http://xmlns.com/foaf/0.1/age> "28"
            }"#,
        )
        .unwrap();

        let r = db
            .execute_sparql(
                r#"SELECT ?name ?age WHERE {
                    <http://ex.org/mia> <http://xmlns.com/foaf/0.1/name> ?name .
                    <http://ex.org/mia> <http://xmlns.com/foaf/0.1/age> ?age
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
    }

    #[test]
    fn update_delete_data_and_verify() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/x> <http://ex.org/p> "val1" .
                <http://ex.org/x> <http://ex.org/q> "val2"
            }"#,
        )
        .unwrap();

        db.execute_sparql(r#"DELETE DATA { <http://ex.org/x> <http://ex.org/p> "val1" . }"#)
            .unwrap();

        let r = db
            .execute_sparql("SELECT ?p ?o WHERE { <http://ex.org/x> ?p ?o }")
            .unwrap();
        assert_eq!(r.row_count(), 1, "Only q/val2 remains after delete");
    }

    #[test]
    fn update_delete_where() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/a> <http://ex.org/status> "draft" .
                <http://ex.org/b> <http://ex.org/status> "published" .
                <http://ex.org/c> <http://ex.org/status> "draft" .
            }"#,
        )
        .unwrap();

        db.execute_sparql(
            r#"DELETE WHERE {
                ?s <http://ex.org/status> "draft"
            }"#,
        )
        .unwrap();

        let r = db
            .execute_sparql("SELECT ?s WHERE { ?s <http://ex.org/status> ?o }")
            .unwrap();
        assert_eq!(
            r.row_count(),
            1,
            "Only the 'published' triple should remain"
        );
    }

    #[test]
    fn update_modify_delete_insert() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/a> <http://ex.org/status> "draft" .
                <http://ex.org/b> <http://ex.org/status> "draft" .
            }"#,
        )
        .unwrap();

        db.execute_sparql(
            r#"DELETE { ?s <http://ex.org/status> "draft" }
               INSERT { ?s <http://ex.org/status> "published" }
               WHERE  { ?s <http://ex.org/status> "draft" }"#,
        )
        .unwrap();

        let r = db
            .execute_sparql(r#"SELECT ?s WHERE { ?s <http://ex.org/status> "published" }"#)
            .unwrap();
        assert_eq!(
            r.row_count(),
            2,
            "Both triples should be updated to published"
        );

        let r2 = db
            .execute_sparql(r#"SELECT ?s WHERE { ?s <http://ex.org/status> "draft" }"#)
            .unwrap();
        assert_eq!(r2.row_count(), 0, "No draft triples should remain");
    }

    // ====================================================================
    // FILTER expressions (more comprehensive)
    // ====================================================================

    #[test]
    fn filter_equality() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?s WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    FILTER(?name = "Alix")
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
    }

    #[test]
    fn filter_inequality() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                    FILTER(?name != "Alix")
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 2, "Gus and Vincent (not Alix)");
    }

    #[test]
    fn filter_and_or_combined() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://xmlns.com/foaf/0.1/age> ?age .
                    FILTER(?age >= "25" && ?age <= "30")
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 2, "Alix (30) and Gus (25) both in range");
    }

    #[test]
    fn filter_or_logic() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                    FILTER(?name = "Alix" || ?name = "Gus")
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 2, "Alix or Gus");
    }

    #[test]
    fn filter_regex_case_insensitive() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                    FILTER(REGEX(?name, "^alix$", "i"))
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "Case-insensitive regex should match Alix");
    }

    #[test]
    fn filter_contains() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                    FILTER(CONTAINS(?name, "us"))
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "Only Gus contains 'us'");
    }

    #[test]
    fn filter_strstarts() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                    FILTER(STRSTARTS(?name, "V"))
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "Only Vincent starts with V");
    }

    // ====================================================================
    // String Functions (execution)
    // ====================================================================

    #[test]
    fn func_strlen() {
        let db = rdf_db();
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/x> <http://ex.org/name> "Alix" }"#)
            .unwrap();

        let r = db
            .execute_sparql(
                r#"SELECT (STRLEN(?name) AS ?len) WHERE {
                    <http://ex.org/x> <http://ex.org/name> ?name
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
    }

    #[test]
    fn func_ucase() {
        let db = rdf_db();
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/x> <http://ex.org/name> "Alix" }"#)
            .unwrap();

        let r = db
            .execute_sparql(
                r#"SELECT (UCASE(?name) AS ?upper) WHERE {
                    <http://ex.org/x> <http://ex.org/name> ?name
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        let upper = r.rows()[0][0].to_string();
        assert!(
            upper.contains("ALIX"),
            "UCASE should produce ALIX, got: {upper}"
        );
    }

    #[test]
    fn func_lcase() {
        let db = rdf_db();
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/x> <http://ex.org/name> "Alix" }"#)
            .unwrap();

        let r = db
            .execute_sparql(
                r#"SELECT (LCASE(?name) AS ?lower) WHERE {
                    <http://ex.org/x> <http://ex.org/name> ?name
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        let lower = r.rows()[0][0].to_string();
        assert!(
            lower.contains("alix"),
            "LCASE should produce alix, got: {lower}"
        );
    }

    // ====================================================================
    // Named Graph Operations
    // ====================================================================

    #[test]
    fn named_graph_insert_and_query() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                GRAPH <http://ex.org/graph1> {
                    <http://ex.org/alix> <http://ex.org/name> "Alix"
                }
            }"#,
        )
        .unwrap();

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    GRAPH <http://ex.org/graph1> {
                        ?s <http://ex.org/name> ?name
                    }
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "Should find Alix in named graph");
    }

    #[test]
    fn named_graph_isolation() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                GRAPH <http://ex.org/g1> {
                    <http://ex.org/a> <http://ex.org/p> "in-g1"
                }
                GRAPH <http://ex.org/g2> {
                    <http://ex.org/b> <http://ex.org/p> "in-g2"
                }
            }"#,
        )
        .unwrap();

        let r1 = db
            .execute_sparql(
                r#"SELECT ?o WHERE {
                    GRAPH <http://ex.org/g1> { ?s <http://ex.org/p> ?o }
                }"#,
            )
            .unwrap();
        assert_eq!(r1.row_count(), 1, "g1 has 1 triple");

        let r2 = db
            .execute_sparql(
                r#"SELECT ?o WHERE {
                    GRAPH <http://ex.org/g2> { ?s <http://ex.org/p> ?o }
                }"#,
            )
            .unwrap();
        assert_eq!(r2.row_count(), 1, "g2 has 1 triple");
    }

    // ====================================================================
    // EXPLAIN (Grafeo extension)
    // ====================================================================

    #[test]
    fn explain_shows_plan() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                "EXPLAIN SELECT ?name WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?name }",
            )
            .unwrap();
        assert_eq!(r.columns, vec!["plan"]);
        assert_eq!(r.row_count(), 1);
        let plan = r.rows()[0][0].to_string();
        assert!(
            plan.contains("TripleScan"),
            "Plan should contain TripleScan, got: {plan}"
        );
    }

    // ====================================================================
    // PREFIX resolution in execution
    // ====================================================================

    #[test]
    fn prefix_resolution() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                SELECT ?name WHERE {
                    ?s foaf:name ?name .
                    ?s a foaf:Person
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 3, "PREFIX + 'a' shorthand should work");
    }

    // ====================================================================
    // Multiple INSERT DATA + complex query
    // ====================================================================

    #[test]
    fn multi_insert_complex_query() {
        let db = rdf_db();

        // Insert in multiple batches
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/jules> <http://ex.org/name> "Jules" .
                <http://ex.org/jules> <http://ex.org/role> "hitman"
            }"#,
        )
        .unwrap();

        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/butch> <http://ex.org/name> "Butch" .
                <http://ex.org/butch> <http://ex.org/role> "boxer"
            }"#,
        )
        .unwrap();

        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/mia> <http://ex.org/name> "Mia" .
                <http://ex.org/mia> <http://ex.org/role> "actress"
            }"#,
        )
        .unwrap();

        // Complex query with FILTER, ORDER BY, LIMIT
        let r = db
            .execute_sparql(
                r#"SELECT ?name ?role WHERE {
                    ?s <http://ex.org/name> ?name .
                    ?s <http://ex.org/role> ?role .
                    FILTER(?role != "hitman")
                } ORDER BY ?name LIMIT 2"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 2, "2 of 3 match filter, LIMIT 2");
    }

    // ====================================================================
    // Edge cases and boundary conditions
    // ====================================================================

    #[test]
    fn empty_result_set() {
        let db = rdf_db();
        let r = db
            .execute_sparql("SELECT ?s WHERE { ?s <http://ex.org/nonexistent> ?o }")
            .unwrap();
        assert_eq!(r.row_count(), 0, "Query on empty store returns 0 rows");
    }

    #[test]
    fn insert_duplicate_triple() {
        let db = rdf_db();
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/x> <http://ex.org/p> "val" }"#)
            .unwrap();
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/x> <http://ex.org/p> "val" }"#)
            .unwrap();

        let r = db
            .execute_sparql("SELECT ?o WHERE { <http://ex.org/x> <http://ex.org/p> ?o }")
            .unwrap();
        assert_eq!(
            r.row_count(),
            1,
            "RDF set semantics: duplicate triple should not produce extra row"
        );
    }

    #[test]
    fn large_insert_and_query() {
        let db = rdf_db();
        use std::fmt::Write;
        let mut triples = String::from("INSERT DATA {\n");
        for i in 0..100 {
            let _ = writeln!(
                triples,
                "    <http://ex.org/n{i}> <http://ex.org/val> \"{i}\" ."
            );
        }
        triples.push('}');
        db.execute_sparql(&triples).unwrap();

        let r = db
            .execute_sparql("SELECT ?s WHERE { ?s <http://ex.org/val> ?v } LIMIT 50")
            .unwrap();
        assert_eq!(r.row_count(), 50, "LIMIT 50 on 100 triples");

        let all = db
            .execute_sparql("SELECT (COUNT(?s) AS ?cnt) WHERE { ?s <http://ex.org/val> ?v }")
            .unwrap();
        assert_eq!(all.row_count(), 1, "COUNT returns 1 row");
    }

    // ====================================================================
    // FILTER with IN / NOT IN (if supported at execution level)
    // ====================================================================

    #[test]
    fn filter_in_operator() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                    FILTER(?name IN ("Alix", "Gus"))
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 2, "IN should match Alix and Gus");
    }

    #[test]
    fn filter_not_in_operator() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                    FILTER(?name NOT IN ("Alix", "Gus"))
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "NOT IN should match only Vincent");
    }

    // ====================================================================
    // BOUND function
    // ====================================================================

    #[test]
    fn filter_bound_function() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                    OPTIONAL { ?s <http://xmlns.com/foaf/0.1/age> ?age }
                    FILTER(BOUND(?age))
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 2, "BOUND(?age) should match Alix and Gus");
    }

    #[test]
    fn filter_not_bound_function() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                    OPTIONAL { ?s <http://xmlns.com/foaf/0.1/age> ?age }
                    FILTER(!BOUND(?age))
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "!BOUND(?age) should match only Vincent");
    }

    // ====================================================================
    // COUNT(*) Fast-Path (0.5.37 - Optimizer Foundation)
    // ====================================================================

    #[test]
    fn count_star_fully_unbound() {
        let db = rdf_db();
        insert_foaf_data(&db);
        // 15 triples in the foaf data
        let r = db
            .execute_sparql("SELECT (COUNT(*) AS ?cnt) WHERE { ?s ?p ?o }")
            .unwrap();
        assert_eq!(r.row_count(), 1);
        let count = r.rows()[0][0].as_int64().unwrap();
        assert_eq!(count, 15);
    }

    #[test]
    fn count_star_predicate_bound() {
        let db = rdf_db();
        insert_foaf_data(&db);
        let r = db
            .execute_sparql(
                "SELECT (COUNT(*) AS ?cnt) WHERE { ?s <http://xmlns.com/foaf/0.1/knows> ?o }",
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        let count = r.rows()[0][0].as_int64().unwrap();
        assert_eq!(count, 2, "alix knows gus, gus knows alix");
    }

    #[test]
    fn count_star_with_group_by_not_fast_path() {
        let db = rdf_db();
        insert_foaf_data(&db);
        // GROUP BY prevents fast-path, should still work via normal aggregate
        let r = db
            .execute_sparql(
                r#"SELECT ?s (COUNT(*) AS ?cnt) WHERE { ?s ?p ?o } GROUP BY ?s ORDER BY ?s"#,
            )
            .unwrap();
        // Each subject should have its own count
        assert!(r.row_count() > 1, "GROUP BY should produce multiple rows");
    }

    #[test]
    fn count_star_with_distinct_not_fast_path() {
        let db = rdf_db();
        insert_foaf_data(&db);
        // DISTINCT prevents fast-path, should still work
        let r = db
            .execute_sparql("SELECT (COUNT(DISTINCT ?s) AS ?cnt) WHERE { ?s ?p ?o }")
            .unwrap();
        assert_eq!(r.row_count(), 1);
        let count = r.rows()[0][0].as_int64().unwrap();
        // Unique subjects: alix, gus, vincent, amsterdam
        assert!(count > 0);
    }

    #[test]
    fn count_star_per_named_graph() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                GRAPH <http://ex.org/g1> {
                    <http://ex.org/alix> <http://xmlns.com/foaf/0.1/name> "Alix" .
                    <http://ex.org/gus> <http://xmlns.com/foaf/0.1/name> "Gus" .
                }
            }"#,
        )
        .unwrap();

        let r = db
            .execute_sparql(
                "SELECT (COUNT(*) AS ?cnt) WHERE { GRAPH <http://ex.org/g1> { ?s ?p ?o } }",
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        let count = r.rows()[0][0].as_int64().unwrap();
        assert_eq!(count, 2);
    }

    // ====================================================================
    // Optimizer: Join Conditions + TripleScan in DPccp (0.5.37)
    // ====================================================================

    #[test]
    fn two_hop_join_produces_correct_results() {
        let db = rdf_db();
        insert_foaf_data(&db);
        // Two-hop: ?a knows ?b, ?b has name ?name
        let r = db
            .execute_sparql(
                r#"SELECT ?a ?name WHERE {
                    ?a <http://xmlns.com/foaf/0.1/knows> ?b .
                    ?b <http://xmlns.com/foaf/0.1/name> ?name .
                }"#,
            )
            .unwrap();
        // alix knows gus (name "Gus"), gus knows alix (name "Alix")
        assert_eq!(r.row_count(), 2);
    }

    #[test]
    fn three_hop_join_produces_correct_results() {
        let db = rdf_db();
        insert_foaf_data(&db);
        // Three-hop: ?a knows ?b, ?b knows ?c, ?c has name ?name
        let r = db
            .execute_sparql(
                r#"SELECT ?a ?name WHERE {
                    ?a <http://xmlns.com/foaf/0.1/knows> ?b .
                    ?b <http://xmlns.com/foaf/0.1/knows> ?c .
                    ?c <http://xmlns.com/foaf/0.1/name> ?name .
                }"#,
            )
            .unwrap();
        // alix->gus->alix (name "Alix"), gus->alix->gus (name "Gus")
        assert_eq!(r.row_count(), 2);
    }

    #[test]
    fn selective_predicate_join_correct() {
        let db = rdf_db();
        insert_foaf_data(&db);
        // Join with selective predicate: livesIn (2 triples) + name (4 triples)
        let r = db
            .execute_sparql(
                r#"SELECT ?name ?city WHERE {
                    ?s <http://ex.org/livesIn> ?c .
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?c <http://xmlns.com/foaf/0.1/name> ?city .
                }"#,
            )
            .unwrap();
        // alix livesIn amsterdam, gus livesIn amsterdam
        assert_eq!(r.row_count(), 2);
    }

    // ====================================================================
    // Phase 3: SPARQL Spec Compliance (0.5.37)
    // ====================================================================

    #[test]
    fn bind_typed_literal_integer() {
        let db = rdf_db();
        insert_foaf_data(&db);
        let r = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                SELECT ?x WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> "Alix" .
                    BIND("42"^^xsd:integer AS ?x)
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        let val = &r.rows()[0][0];
        // Should be integer 42, not empty string
        assert!(
            val.as_int64() == Some(42) || val.to_string() == "42",
            "expected 42, got: {val:?}"
        );
    }

    #[test]
    fn bind_typed_literal_double() {
        let db = rdf_db();
        insert_foaf_data(&db);
        let r = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                SELECT ?x WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> "Alix" .
                    BIND("3.14"^^xsd:double AS ?x)
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        let val = &r.rows()[0][0];
        assert!(
            val.to_string().starts_with("3.14"),
            "expected 3.14, got: {val:?}"
        );
    }

    #[test]
    fn filter_equality_pushdown_into_scan() {
        let db = rdf_db();
        insert_foaf_data(&db);
        // FILTER(?name = "Alix") should be pushed into the TripleScan
        // so only matching triples are scanned (index lookup, not full scan + filter)
        let r = db
            .execute_sparql(
                r#"SELECT ?s WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    FILTER(?name = "Alix")
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        assert!(r.rows()[0][0].to_string().contains("alix"));
    }

    #[test]
    fn filter_equality_on_predicate() {
        let db = rdf_db();
        insert_foaf_data(&db);
        // FILTER on a predicate variable
        let r = db
            .execute_sparql(
                r#"SELECT ?s ?o WHERE {
                    ?s ?p ?o .
                    FILTER(?p = <http://xmlns.com/foaf/0.1/name>)
                }"#,
            )
            .unwrap();
        // 4 names: Alix, Gus, Vincent, Amsterdam
        assert_eq!(r.row_count(), 4);
    }

    #[test]
    fn describe_without_where_clause() {
        let db = rdf_db();
        insert_foaf_data(&db);
        // DESCRIBE <iri> without WHERE should return CBD for that resource
        let r = db.execute_sparql("DESCRIBE <http://ex.org/alix>").unwrap();
        // alix has 5 triples: type, name, age, knows, mbox
        assert!(
            r.row_count() >= 5,
            "DESCRIBE without WHERE should return triples, got {} rows",
            r.row_count()
        );
    }

    #[test]
    fn select_reduced_returns_results() {
        let db = rdf_db();
        insert_foaf_data(&db);
        let r = db
            .execute_sparql(
                "SELECT REDUCED ?type WHERE { ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?type }",
            )
            .unwrap();
        // REDUCED may or may not eliminate duplicates (spec allows either).
        // Our implementation treats REDUCED as a no-op, returning all rows.
        assert!(
            r.row_count() >= 2,
            "REDUCED should return at least the distinct count"
        );
    }

    #[test]
    fn physical_explain_shows_operator_names() {
        let db = rdf_db();
        insert_foaf_data(&db);
        let r = db
            .execute_sparql(
                r#"EXPLAIN SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person>
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        let plan = r.rows()[0][0].to_string();
        // Physical plan should contain physical operator names
        assert!(
            plan.contains("RdfTripleScan") || plan.contains("HashJoin") || plan.contains("Project"),
            "EXPLAIN should show physical operator names, got: {plan}"
        );
    }

    #[test]
    fn explain_analyze_shows_timing() {
        let db = rdf_db();
        insert_foaf_data(&db);
        let r = db
            .execute_sparql(
                r#"EXPLAIN ANALYZE SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name
                }"#,
            )
            .unwrap();
        assert!(r.row_count() >= 1);
        let profile = r.rows()[0][0].to_string();
        // EXPLAIN ANALYZE should show timing or operator stats
        assert!(
            profile.contains("time") || profile.contains("rows") || profile.contains("ms"),
            "EXPLAIN ANALYZE should show execution stats, got: {profile}"
        );
    }
}
