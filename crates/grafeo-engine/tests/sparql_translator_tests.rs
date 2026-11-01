//! Integration tests for SPARQL translator.

#[cfg(all(feature = "sparql", feature = "triple-store"))]
mod tests {
    use grafeo_engine::query::translators::sparql::translate;

    #[test]
    fn test_translate_simple_select() {
        let query = "SELECT ?x WHERE { ?x ?y ?z }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_select_with_filter() {
        let query = "SELECT ?x WHERE { ?x ?y ?z FILTER(?z > 10) }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_select_with_limit_offset() {
        let query = "SELECT ?x WHERE { ?x ?y ?z } LIMIT 10 OFFSET 5";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_select_distinct() {
        let query = "SELECT DISTINCT ?x WHERE { ?x ?y ?z }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_order_by() {
        let query = "SELECT ?x ?y WHERE { ?x ?p ?y } ORDER BY ?y";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_aggregation() {
        let query = r#"
            SELECT ?type (COUNT(?x) AS ?count)
            WHERE {
                ?x <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?type
            }
            GROUP BY ?type
        "#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_union() {
        let query = r#"
            SELECT ?name
            WHERE {
                { ?x <http://xmlns.com/foaf/0.1/name> ?name }
                UNION
                { ?x <http://xmlns.com/foaf/0.1/nick> ?name }
            }
        "#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_ask() {
        let query = "ASK { ?x <http://xmlns.com/foaf/0.1/knows> ?y }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_construct() {
        let query = r#"
            CONSTRUCT { ?s ?p ?o }
            WHERE { ?s ?p ?o }
        "#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_prefix_resolution() {
        let query = r#"
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            SELECT ?name
            WHERE { ?x foaf:name ?name }
        "#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_optional() {
        let query = r#"
            SELECT ?name ?email
            WHERE {
                ?x <http://xmlns.com/foaf/0.1/name> ?name
                OPTIONAL { ?x <http://xmlns.com/foaf/0.1/mbox> ?email }
            }
        "#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_multiple_patterns() {
        let query = r#"
            SELECT ?name ?age
            WHERE {
                ?person <http://xmlns.com/foaf/0.1/name> ?name .
                ?person <http://xmlns.com/foaf/0.1/age> ?age
            }
        "#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_bind() {
        let query = r#"
            SELECT ?x ?doubled
            WHERE {
                ?x <http://example.org/value> ?val
                BIND(?val * 2 AS ?doubled)
            }
        "#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_nested_filter() {
        let query = r#"
            SELECT ?x
            WHERE {
                ?x <http://example.org/value> ?v
                FILTER(?v > 10 && ?v < 100 || ?v = 0)
            }
        "#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_explain_prefix_sets_flag() {
        let query = "EXPLAIN SELECT ?x WHERE { ?x ?y ?z }";
        let plan = translate(query).unwrap();
        assert!(plan.explain);
    }

    #[test]
    fn test_explain_prefix_case_insensitive() {
        let query = "explain SELECT ?x WHERE { ?x ?y ?z }";
        let plan = translate(query).unwrap();
        assert!(plan.explain);
    }

    #[test]
    fn test_no_explain_prefix() {
        let query = "SELECT ?x WHERE { ?x ?y ?z }";
        let plan = translate(query).unwrap();
        assert!(!plan.explain);
    }

    /// Each form SPARQL takes before the prologue (`EXPLAIN`, `EXPLAIN
    /// ANALYZE`, `PROFILE`, any case) sets the same flags written after the
    /// `BASE` and `PREFIX` lines, and the query reads the prefixes.
    #[test]
    fn each_explain_form_reads_the_same_after_the_prologue() {
        let body = "SELECT ?n WHERE { ?s ex:name ?n }";
        for (keyword, explain, profile) in [
            ("EXPLAIN", true, false),
            ("explain", true, false),
            ("EXPLAIN ANALYZE", false, true),
            ("Explain  analyze", false, true),
            ("EXPLAIN\nANALYZE", false, true),
            ("PROFILE", false, true),
            ("profile", false, true),
        ] {
            for prologue in [
                "PREFIX ex: <http://example.org/>",
                "BASE <http://example.org/> PREFIX ex: <http://example.org/>",
                "PREFIX ex: <http://example.org/>\nPREFIX foaf: <http://xmlns.com/foaf/0.1/>\n",
            ] {
                for query in [
                    format!("{keyword} {prologue} {body}"),
                    format!("{prologue} {keyword} {body}"),
                ] {
                    let plan = translate(&query).unwrap_or_else(|e| panic!("{query}: {e}"));
                    assert_eq!((plan.explain, plan.profile), (explain, profile), "{query}");
                }
            }
        }
        // A query after the prologue that only starts with the keyword's
        // letters is no EXPLAIN.
        let plan = translate(
            "PREFIX ex: <http://example.org/> SELECT ?explain WHERE { ?s ex:name ?explain }",
        )
        .unwrap();
        assert_eq!((plan.explain, plan.profile), (false, false));
    }
}

#[cfg(all(feature = "sparql", feature = "triple-store"))]
mod explain_integration {
    use grafeo_engine::GrafeoDB;

    #[test]
    fn test_sparql_explain_returns_plan() {
        let db = GrafeoDB::new_in_memory();
        db.execute_sparql(
            r#"INSERT DATA { <http://example.org/alix> <http://xmlns.com/foaf/0.1/name> "Alix" }"#,
        )
        .unwrap();

        let result = db
            .execute_sparql(
                "EXPLAIN SELECT ?name WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?name }",
            )
            .unwrap();

        assert_eq!(result.columns, vec!["plan"]);
        assert_eq!(result.rows().len(), 1);
        let plan_text = result.rows()[0][0].to_string();
        assert!(
            plan_text.contains("TripleScan"),
            "Plan should contain TripleScan, got: {plan_text}"
        );
    }

    /// After the PREFIX lines, `EXPLAIN` shows the plan without running the
    /// query, and `EXPLAIN ANALYZE` and `PROFILE` run it and show the rows
    /// each operator produced.
    #[test]
    fn each_explain_form_runs_after_the_prefix_lines() {
        let db = GrafeoDB::new_in_memory();
        db.execute_sparql(
            r#"INSERT DATA { <http://example.org/alix> <http://xmlns.com/foaf/0.1/name> "Alix" .
                <http://example.org/gus> <http://xmlns.com/foaf/0.1/name> "Gus" }"#,
        )
        .unwrap();
        let body = "SELECT ?name WHERE { ?s foaf:name ?name }";
        let prefix = "PREFIX foaf: <http://xmlns.com/foaf/0.1/>";
        let plan = db
            .execute_sparql(&format!("{prefix} EXPLAIN {body}"))
            .unwrap();
        assert_eq!(plan.columns, vec!["plan"]);
        assert!(plan.rows()[0][0].to_string().contains("TripleScan"));
        for keyword in ["EXPLAIN ANALYZE", "PROFILE"] {
            let query = format!("{prefix} {keyword} {body}");
            let profile = db
                .execute_sparql(&query)
                .unwrap_or_else(|e| panic!("{query}: {e}"));
            assert_eq!(profile.columns, vec!["profile"], "{query}");
            let text = profile.rows()[0][0].to_string();
            assert!(
                text.contains("TripleScan") && text.contains("rows=2"),
                "{query}: {text}"
            );
        }
    }

    /// The operator lines of a profile: each line's operator name and its
    /// rows, indented by depth.
    fn profile_lines(text: &str) -> Vec<String> {
        text.lines()
            .filter(|line| line.contains("rows="))
            .map(|line| {
                let depth = line.len() - line.trim_start().len();
                let name = line.split_whitespace().next().unwrap_or("");
                let rows = line
                    .split_whitespace()
                    .find(|word| word.starts_with("rows="))
                    .unwrap_or("");
                format!("{}{name} {rows}", " ".repeat(depth))
            })
            .collect()
    }

    /// PROFILE of a pattern whose planner leaves operators out of the
    /// physical plan gives the profile, each operator's numbers on its own
    /// line: the pattern of a `FILTER EXISTS` or `NOT EXISTS` runs inside
    /// the join that filters (and is named in its detail). A `DELETE WHERE`
    /// reads its pattern, also of a triple without variables, so its
    /// profile shows the scan under the delete; `EXPLAIN` of one deletes
    /// nothing.
    #[test]
    fn profile_of_patterns_the_planner_folds_in() {
        let db = GrafeoDB::new_in_memory();
        db.execute_sparql(
            r#"PREFIX ex: <http://example.org/> INSERT DATA {
                ex:alix ex:name "Alix" . ex:alix ex:nick "Al" .
                ex:gus ex:name "Gus" . ex:vincent ex:name "Vincent" }"#,
        )
        .unwrap();
        let prefix = "PREFIX ex: <http://example.org/>";
        for (body, lines, detail) in [
            (
                "SELECT ?n WHERE { ?s ex:name ?n FILTER EXISTS { ?s ex:nick ?k } }",
                &[
                    "RdfProject rows=1",
                    "  HashJoin rows=1",
                    "    RdfTripleScan rows=3",
                ][..],
                "(EXISTS {",
            ),
            (
                "SELECT ?n WHERE { ?s ex:name ?n FILTER NOT EXISTS { ?s ex:nick ?k } }",
                &[
                    "RdfProject rows=2",
                    "  HashJoin rows=2",
                    "    RdfTripleScan rows=3",
                ][..],
                "(Not EXISTS {",
            ),
            (
                "SELECT ?n WHERE { ?s ex:name ?n \
                 OPTIONAL { ?s ex:nick ?k FILTER NOT EXISTS { ?s ex:age ?a } } }",
                &[
                    "RdfProject rows=3",
                    "  Project rows=3",
                    "    RdfTripleScan rows=3",
                    "    RdfTripleScan rows=1",
                ][..],
                "",
            ),
            (
                "ASK { FILTER NOT EXISTS { ?s ex:age ?a } }",
                &[
                    "RdfProject rows=1",
                    "  SimpleAggregate rows=1",
                    "    Limit rows=1",
                    "      HashJoin rows=1",
                    "        SingleRow rows=1",
                ][..],
                "(Not EXISTS {",
            ),
        ] {
            let query = format!("PROFILE {prefix} {body}");
            let profile = db
                .execute_sparql(&query)
                .unwrap_or_else(|e| panic!("{query}: {e}"));
            let text = profile.rows()[0][0].as_str().expect("the profile text");
            assert_eq!(profile_lines(text), lines, "{query}\n{text}");
            assert!(text.contains(detail), "{query}\n{text}");
        }

        let nick_left = || {
            db.execute_sparql(&format!("{prefix} ASK {{ ex:alix ex:nick ?k }}"))
                .unwrap()
                .rows()[0][0]
                .clone()
        };
        let plan = db
            .execute_sparql(&format!(
                "{prefix} EXPLAIN DELETE WHERE {{ ex:alix ex:nick \"Al\" }}"
            ))
            .unwrap();
        let text = plan.rows()[0][0].to_string();
        assert!(text.contains("TripleScan"), "{text}");
        assert_eq!(nick_left(), grafeo_common::types::Value::Bool(true));

        // The triple goes, after the scan found it.
        let query = format!("PROFILE {prefix} DELETE WHERE {{ ex:alix ex:nick \"Al\" }}");
        let profile = db
            .execute_sparql(&query)
            .unwrap_or_else(|e| panic!("{query}: {e}"));
        let text = profile.rows()[0][0].as_str().expect("the profile text");
        assert_eq!(
            profile_lines(text),
            ["RdfModify rows=0", "  RdfTripleScan rows=1"],
            "{text}"
        );
        assert_eq!(nick_left(), grafeo_common::types::Value::Bool(false));
    }
}
