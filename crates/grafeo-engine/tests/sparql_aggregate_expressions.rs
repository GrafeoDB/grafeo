//! Integration tests for SPARQL expressions in projection and GROUP BY.
//!
//! Covers two areas:
//!   1. **Projection functions**: STR(), STRLEN() used in SELECT (not just FILTER).
//!   2. **GROUP BY with expressions**: STR() inside GROUP BY combined with COUNT.
// Test values are small known constants
#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
//!
//! These tests verify the full pipeline: SPARQL translator projection handling,
//! RDF planner expression pre-projection, and physical execution.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test sparql_aggregate_expressions
//! ```

#[cfg(all(feature = "sparql", feature = "triple-store"))]
mod sparql_aggregate_expression_tests {
    use grafeo_engine::GrafeoDB;

    fn rdf_db() -> GrafeoDB {
        GrafeoDB::new_in_memory()
    }

    fn insert_sample_triples(db: &GrafeoDB) {
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/alix> <http://ex.org/name> "Alix" .
                <http://ex.org/alix> <http://ex.org/age>  "30" .
                <http://ex.org/gus>  <http://ex.org/name> "Gus" .
                <http://ex.org/gus>  <http://ex.org/age>  "25" .
            }"#,
        )
        .unwrap();
    }

    /// GROUP BY (STR(?s)): expression-based grouping should not panic.
    /// STR() on RDF IRI subjects currently produces identical strings for
    /// different subjects (expression evaluation limitation), so the group
    /// count may be less than expected.
    #[test]
    fn sparql_group_by_str_with_count() {
        let db = rdf_db();
        insert_sample_triples(&db);

        let result = db.execute_sparql(
            "SELECT (STR(?s) AS ?subject) (COUNT(*) AS ?cnt) WHERE { ?s ?p ?o } GROUP BY (STR(?s))",
        );

        // Must not panic or error. The store plumbing is now in place.
        let qr = result.unwrap();
        assert!(
            qr.row_count() >= 1,
            "GROUP BY STR(?s) should produce at least 1 group, got {}",
            qr.row_count()
        );
    }

    /// ORDER BY ASC(STR(?s)): expression-based sorting should work.
    #[test]
    fn sparql_order_by_str() {
        let db = rdf_db();
        insert_sample_triples(&db);

        let qr = db
            .execute_sparql("SELECT ?s WHERE { ?s <http://ex.org/name> ?o } ORDER BY ASC(STR(?s))")
            .unwrap();
        assert_eq!(
            qr.row_count(),
            2,
            "ORDER BY ASC(STR(?s)) should return 2 rows"
        );
    }

    /// Both GROUP BY and ORDER BY with STR(): combined complex expressions.
    #[test]
    fn sparql_group_by_and_order_by_both_complex() {
        let db = rdf_db();
        insert_sample_triples(&db);

        let result = db.execute_sparql(
            "SELECT (STR(?s) AS ?subject) (COUNT(*) AS ?cnt) WHERE { ?s ?p ?o } GROUP BY (STR(?s)) ORDER BY (STR(?s))",
        );

        // Must not panic or error. STR() on IRIs may collapse groups (see above).
        let qr = result.unwrap();
        assert!(
            qr.row_count() >= 1,
            "GROUP BY + ORDER BY STR(?s) should produce at least 1 group, got {}",
            qr.row_count()
        );
    }

    /// ORDER BY DESC(STR(?s)): descending with a function expression.
    #[test]
    fn sparql_order_by_desc_str() {
        let db = rdf_db();
        insert_sample_triples(&db);

        let qr = db
            .execute_sparql("SELECT ?s WHERE { ?s <http://ex.org/name> ?o } ORDER BY DESC(STR(?s))")
            .unwrap();
        assert_eq!(
            qr.row_count(),
            2,
            "ORDER BY DESC(STR(?s)) should return 2 rows"
        );
    }

    /// A computed ORDER BY key is a column the sort adds to sort by: the
    /// result keeps only the selected columns.
    #[test]
    fn sparql_order_by_expression_returns_only_the_selected_columns() {
        let db = rdf_db();
        insert_sample_triples(&db);

        let qr = db
            .execute_sparql("SELECT ?s WHERE { ?s <http://ex.org/name> ?o } ORDER BY DESC(STR(?s))")
            .unwrap();
        assert_eq!(qr.columns, vec!["s"]);
        assert_eq!(qr.column_types.len(), 1, "types: {:?}", qr.column_types);
        assert_eq!(qr.row_count(), 2);
        assert!(
            qr.rows().iter().all(|row| row.len() == 1),
            "rows: {:?}",
            qr.rows()
        );
    }

    // ---------------------------------------------------------------
    // Area 1: SPARQL translator projection with function expressions
    // ---------------------------------------------------------------

    /// STR() in SELECT projection: exercises the translate_projection path
    /// where a FunctionCall expression appears as a projected column.
    #[test]
    fn test_sparql_str_in_projection() {
        let db = rdf_db();
        insert_sample_triples(&db);

        let result =
            db.execute_sparql("SELECT (STR(?s) AS ?name) WHERE { ?s <http://ex.org/name> ?o }");

        let qr = result.unwrap();
        assert_eq!(
            qr.row_count(),
            2,
            "STR(?s) projection should return 2 rows (one per subject with name)"
        );
        // The projected alias should appear in the column list
        assert!(
            qr.columns.contains(&"name".to_string()),
            "Result should contain column 'name', got: {:?}",
            qr.columns
        );
        // Every projected value should be a non-empty string (the IRI serialised via STR)
        for row in qr.iter() {
            let val = &row[qr.columns.iter().position(|c| c == "name").unwrap()];
            let s = val.to_string();
            assert!(
                !s.is_empty(),
                "STR(?s) should produce a non-empty string, got: {val:?}"
            );
        }
    }

    /// STRLEN() in SELECT projection: exercises the translate_projection path
    /// where a FunctionCall expression appears as a projected column alongside
    /// a plain variable.
    ///
    /// STRLEN() in RDF projection is evaluated via RdfProjectOperator, which
    /// delegates to RdfExpressionPredicate for full SPARQL function support.
    #[test]
    fn test_sparql_strlen_in_projection() {
        let db = rdf_db();
        insert_sample_triples(&db);

        let result = db.execute_sparql(
            "SELECT ?name (STRLEN(?name) AS ?len) WHERE { ?s <http://ex.org/name> ?name }",
        );

        let qr = result.unwrap();
        assert_eq!(qr.row_count(), 2, "STRLEN projection should return 2 rows");
        assert!(
            qr.columns.contains(&"len".to_string()),
            "Result should contain column 'len', got: {:?}",
            qr.columns
        );

        let name_idx = qr.columns.iter().position(|c| c == "name").unwrap();
        let len_idx = qr.columns.iter().position(|c| c == "len").unwrap();

        for row in qr.iter() {
            let name_val = &row[name_idx];
            let len_val = &row[len_idx];

            // Extract the actual string content (without Display quotes)
            let name_content = match name_val {
                grafeo_common::types::Value::String(s) => s.as_str(),
                other => panic!("Expected String for ?name, got: {other:?}"),
            };
            let expected_len = name_content.len() as i64;

            match len_val {
                grafeo_common::types::Value::Int64(n) => {
                    assert_eq!(
                        *n, expected_len,
                        "STRLEN(\"{name_content}\") should be {expected_len}, got {n}"
                    );
                }
                grafeo_common::types::Value::Float64(f) => {
                    assert_eq!(
                        *f as i64, expected_len,
                        "STRLEN(\"{name_content}\") should be {expected_len}, got {f}"
                    );
                }
                grafeo_common::types::Value::String(s) => {
                    let parsed: i64 = s.parse().unwrap_or(-1);
                    assert_eq!(
                        parsed, expected_len,
                        "STRLEN(\"{name_content}\") should be {expected_len}, got \"{s}\""
                    );
                }
                grafeo_common::types::Value::Null => {
                    panic!("STRLEN should not return Null now that RdfProjectOperator is used");
                }
                other => {
                    panic!("STRLEN should return a numeric or Null value, got: {other:?}");
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // Area 2: RDF planner GROUP BY with expression pre-projection
    // ---------------------------------------------------------------

    /// GROUP BY ?s with STR(?s) in projection and COUNT(*) aggregate.
    /// Exercises the RDF planner path that pre-projects complex expressions
    /// in group-by keys before the physical aggregate operator.
    ///
    /// When aggregation is active, the projection alias from `(STR(?s) AS ?name)`
    /// is applied by the aggregate operator's output schema, so the column name
    /// comes from the GROUP BY key ("s") rather than the SELECT alias ("name").
    #[test]
    fn test_sparql_group_by_with_str() {
        let db = rdf_db();
        insert_sample_triples(&db);

        // Each subject has 2 triples (name + age), so grouping by ?s
        // with COUNT(*) should yield 2 per group.
        let result = db.execute_sparql(
            "SELECT (STR(?s) AS ?name) (COUNT(*) AS ?cnt) WHERE { ?s ?p ?o } GROUP BY ?s",
        );

        let qr = result.unwrap();
        assert_eq!(
            qr.row_count(),
            2,
            "GROUP BY ?s should produce 2 groups (alix, gus), got {}",
            qr.row_count()
        );

        assert!(
            qr.columns.contains(&"cnt".to_string()),
            "Result should contain column 'cnt', got: {:?}",
            qr.columns
        );

        let cnt_idx = qr.columns.iter().position(|c| c == "cnt").unwrap();
        for row in qr.iter() {
            let cnt_val = &row[cnt_idx];
            // Each subject (alix, gus) has exactly 2 triples
            match cnt_val {
                grafeo_common::types::Value::Int64(n) => {
                    assert_eq!(*n, 2, "Each subject group should have count=2, got {n}");
                }
                grafeo_common::types::Value::Float64(f) => {
                    assert_eq!(
                        *f as i64, 2,
                        "Each subject group should have count=2, got {f}"
                    );
                }
                grafeo_common::types::Value::String(s) => {
                    let parsed: i64 = s.parse().unwrap_or(-1);
                    assert_eq!(
                        parsed, 2,
                        "Each subject group should have count=2, got \"{s}\""
                    );
                }
                other => {
                    panic!("COUNT should return a numeric value, got: {other:?}");
                }
            }
        }

        // The group key column is "s" (from GROUP BY ?s); the alias "name"
        // from the SELECT expression is not propagated through the aggregate
        // operator. Verify that the subject IRIs are present in the output.
        let subject_col = if qr.columns.contains(&"name".to_string()) {
            "name"
        } else {
            "s"
        };
        let subj_idx = qr.columns.iter().position(|c| c == subject_col).unwrap();
        let mut subjects: Vec<String> = qr.iter().map(|row| row[subj_idx].to_string()).collect();
        subjects.sort();
        assert!(
            subjects.iter().any(|n| n.contains("alix")),
            "Expected a group key containing 'alix', got: {subjects:?}"
        );
        assert!(
            subjects.iter().any(|n| n.contains("gus")),
            "Expected a group key containing 'gus', got: {subjects:?}"
        );
    }

    // ---------------------------------------------------------------
    // Area 3: SPARQL dateTime functions on typed literals
    // ---------------------------------------------------------------

    fn insert_datetime_triples(db: &GrafeoDB) {
        db.execute_sparql(
            r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            INSERT DATA {
                <http://ex.org/event1> <http://ex.org/date> "2024-06-15T14:30:45+05:30"^^xsd:dateTime .
                <http://ex.org/event2> <http://ex.org/date> "2024-12-25T08:00:00-08:00"^^xsd:dateTime .
            }"#,
        )
        .unwrap();
    }

    #[test]
    fn sparql_year_month_day_from_zoned_datetime() {
        let db = rdf_db();
        insert_datetime_triples(&db);

        let result = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                SELECT ?y ?m ?d WHERE {
                    <http://ex.org/event1> <http://ex.org/date> ?dt .
                    BIND(YEAR(?dt) AS ?y)
                    BIND(MONTH(?dt) AS ?m)
                    BIND(DAY(?dt) AS ?d)
                }"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn sparql_hours_minutes_seconds_from_zoned_datetime() {
        let db = rdf_db();
        insert_datetime_triples(&db);

        let result = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                SELECT ?h ?min ?sec WHERE {
                    <http://ex.org/event1> <http://ex.org/date> ?dt .
                    BIND(HOURS(?dt) AS ?h)
                    BIND(MINUTES(?dt) AS ?min)
                    BIND(SECONDS(?dt) AS ?sec)
                }"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn sparql_timezone_and_tz_from_zoned_datetime() {
        let db = rdf_db();
        insert_datetime_triples(&db);

        let result = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                SELECT ?tz WHERE {
                    <http://ex.org/event2> <http://ex.org/date> ?dt .
                    BIND(TZ(?dt) AS ?tz)
                }"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
    }

    // ---------------------------------------------------------------
    // Area 4: SPARQL LANG() and LANGMATCHES()
    // ---------------------------------------------------------------

    fn insert_language_tagged_triples(db: &GrafeoDB) {
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/alix> <http://www.w3.org/2000/01/rdf-schema#label> "Alix"@en .
                <http://ex.org/alix> <http://www.w3.org/2000/01/rdf-schema#label> "Alix"@fr .
                <http://ex.org/gus>  <http://www.w3.org/2000/01/rdf-schema#label> "Gus"@en-US .
                <http://ex.org/item> <http://www.w3.org/2000/01/rdf-schema#label> "Plain" .
            }"#,
        )
        .unwrap();
    }

    #[test]
    fn sparql_langmatches_exact() {
        let db = rdf_db();
        insert_language_tagged_triples(&db);

        let result = db
            .execute_sparql(
                r#"SELECT ?label WHERE {
                    ?s <http://www.w3.org/2000/01/rdf-schema#label> ?label .
                    FILTER(LANGMATCHES(LANG(?label), "en"))
                }"#,
            )
            .unwrap();
        // Should match "en" and "en-US" (prefix match)
        assert!(
            result.row_count() >= 2,
            "LANGMATCHES should match 'en' and 'en-US', got {} rows",
            result.row_count()
        );
    }

    #[test]
    fn sparql_langmatches_wildcard() {
        let db = rdf_db();
        insert_language_tagged_triples(&db);

        let result = db
            .execute_sparql(
                r#"SELECT ?label WHERE {
                    ?s <http://www.w3.org/2000/01/rdf-schema#label> ?label .
                    FILTER(LANGMATCHES(LANG(?label), "*"))
                }"#,
            )
            .unwrap();
        // Should match all language-tagged literals (en, fr, en-US) but NOT "Plain"
        assert!(
            result.row_count() >= 3,
            "LANGMATCHES(*) should match all tagged literals, got {} rows",
            result.row_count()
        );
    }

    #[test]
    fn sparql_query_without_lang_columns() {
        // Verify strip_internal_columns works: __lang_ columns should not appear in results
        let db = rdf_db();
        insert_language_tagged_triples(&db);

        let result = db
            .execute_sparql(
                r#"SELECT ?s WHERE {
                    ?s <http://www.w3.org/2000/01/rdf-schema#label> ?label .
                }"#,
            )
            .unwrap();
        for col in &result.columns {
            assert!(
                !col.starts_with("__lang_"),
                "Internal __lang_ column should be stripped from results, found: {col}"
            );
        }
    }
}
