//! Parser tests aligned with the W3C SPARQL 1.1 Query Language specification.
//!
//! Each section header references the corresponding spec section from
//! <https://www.w3.org/TR/sparql11-query/>.
//!
//! ```bash
//! cargo test -p grafeo-adapters --all-features --test sparql_w3c_parser_tests
//! ```

#[cfg(feature = "sparql")]
mod tests {
    use grafeo_adapters::query::sparql::{self, ast};

    // ====================================================================
    // 2 - Making Simple Queries (Informative)
    // ====================================================================

    #[test]
    fn sec2_simple_select_wildcard() {
        let query = "SELECT * WHERE { ?s ?p ?o }";
        let result = sparql::parse(query).unwrap();
        if let ast::QueryForm::Select(select) = &result.query_form {
            assert!(matches!(select.projection, ast::Projection::Wildcard));
        } else {
            panic!("Expected SELECT");
        }
    }

    #[test]
    fn sec2_select_specific_variables() {
        let query = "SELECT ?name ?mbox WHERE { ?x <http://xmlns.com/foaf/0.1/name> ?name . ?x <http://xmlns.com/foaf/0.1/mbox> ?mbox }";
        let result = sparql::parse(query).unwrap();
        if let ast::QueryForm::Select(select) = &result.query_form {
            if let ast::Projection::Variables(vars) = &select.projection {
                assert_eq!(vars.len(), 2);
            } else {
                panic!("Expected variable projection");
            }
        } else {
            panic!("Expected SELECT");
        }
    }

    // ====================================================================
    // 2.1 - RDF Term Constraints (Typed & Language-Tagged Literals)
    // ====================================================================

    #[test]
    fn sec2_1_typed_literal_integer() {
        let query = r#"SELECT ?s WHERE { ?s <http://ex.org/age> "30"^^<http://www.w3.org/2001/XMLSchema#integer> }"#;
        let result = sparql::parse(query);
        assert!(
            result.is_ok(),
            "Typed literal with integer datatype: {result:?}"
        );
    }

    #[test]
    fn sec2_1_language_tagged_literal() {
        let query = r#"SELECT ?s WHERE { ?s <http://ex.org/label> "chat"@fr }"#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "Language-tagged literal: {result:?}");
    }

    #[test]
    fn sec2_1_boolean_literal_true() {
        let query = "SELECT ?s WHERE { ?s <http://ex.org/active> true }";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "Boolean literal true: {result:?}");
    }

    #[test]
    fn sec2_1_boolean_literal_false() {
        let query = "SELECT ?s WHERE { ?s <http://ex.org/active> false }";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "Boolean literal false: {result:?}");
    }

    #[test]
    fn sec2_1_integer_literal() {
        let query = "SELECT ?s WHERE { ?s <http://ex.org/age> 30 }";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "Bare integer literal: {result:?}");
    }

    #[test]
    fn sec2_1_decimal_literal() {
        let query = "SELECT ?s WHERE { ?s <http://ex.org/price> 19.99 }";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "Decimal literal: {result:?}");
    }

    #[test]
    fn sec2_1_double_literal() {
        let query = "SELECT ?s WHERE { ?s <http://ex.org/val> 1.5e2 }";
        let result = sparql::parse(query);
        assert!(
            result.is_ok(),
            "Double (scientific notation) literal: {result:?}"
        );
    }

    // ====================================================================
    // 2.5 - BASE and PREFIX
    // ====================================================================

    #[test]
    fn sec2_5_base_declaration() {
        let query = r#"
            BASE <http://example.org/>
            SELECT ?s WHERE { ?s <name> ?o }
        "#;
        let result = sparql::parse(query).unwrap();
        assert!(result.base.is_some());
        assert_eq!(result.base.unwrap().as_str(), "http://example.org/");
    }

    #[test]
    fn sec2_5_base_with_prefix() {
        let query = r#"
            BASE <http://example.org/>
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            SELECT ?name WHERE { ?x foaf:name ?name }
        "#;
        let result = sparql::parse(query).unwrap();
        assert!(result.base.is_some());
        assert_eq!(result.prefixes.len(), 1);
    }

    #[test]
    fn sec2_5_default_prefix() {
        let query = r#"
            PREFIX : <http://example.org/>
            SELECT ?s WHERE { ?s :name ?o }
        "#;
        let result = sparql::parse(query).unwrap();
        assert_eq!(result.prefixes.len(), 1);
        assert_eq!(result.prefixes[0].prefix, "");
    }

    // ====================================================================
    // 4 - Blank Nodes
    // ====================================================================

    #[test]
    fn sec4_labeled_blank_node_subject() {
        let query = "SELECT ?p ?o WHERE { _:b1 ?p ?o }";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "Labeled blank node as subject: {result:?}");
    }

    #[test]
    fn sec4_labeled_blank_node_object() {
        let query = "SELECT ?s WHERE { ?s <http://ex.org/knows> _:someone }";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "Labeled blank node as object: {result:?}");
    }

    #[test]
    fn sec4_anonymous_blank_node() {
        let query = r#"SELECT ?name WHERE { [ <http://xmlns.com/foaf/0.1/name> ?name ] }"#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "Anonymous blank node []: {result:?}");
    }

    #[test]
    fn sec4_anonymous_blank_node_with_properties() {
        let query = r#"
            SELECT ?name ?age WHERE {
                [ <http://xmlns.com/foaf/0.1/name> ?name ;
                  <http://xmlns.com/foaf/0.1/age>  ?age ]
            }
        "#;
        let result = sparql::parse(query);
        assert!(
            result.is_ok(),
            "Anonymous blank node with property list: {result:?}"
        );
    }

    // ====================================================================
    // 5 - Graph Patterns
    // ====================================================================

    #[test]
    fn sec5_group_graph_pattern_empty() {
        let query = "SELECT ?s WHERE { }";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "Empty group graph pattern: {result:?}");
    }

    #[test]
    fn sec5_nested_group_graph_pattern() {
        let query = r#"
            SELECT ?s WHERE {
                { ?s <http://ex.org/p1> ?o1 }
                { ?s <http://ex.org/p2> ?o2 }
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "Nested groups: {result:?}");
    }

    // ====================================================================
    // 6 - OPTIONAL
    // ====================================================================

    #[test]
    fn sec6_nested_optional() {
        let query = r#"
            SELECT ?name ?email ?phone WHERE {
                ?x <http://xmlns.com/foaf/0.1/name> ?name .
                OPTIONAL {
                    ?x <http://xmlns.com/foaf/0.1/mbox> ?email .
                    OPTIONAL { ?x <http://xmlns.com/foaf/0.1/phone> ?phone }
                }
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "Nested OPTIONAL: {result:?}");
    }

    // ====================================================================
    // 8 - Negation (MINUS, NOT EXISTS, FILTER NOT EXISTS)
    // ====================================================================

    #[test]
    fn sec8_filter_exists() {
        let query = r#"
            SELECT ?name WHERE {
                ?s <http://ex.org/name> ?name .
                FILTER EXISTS { ?s <http://ex.org/email> ?e }
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "FILTER EXISTS: {result:?}");
    }

    #[test]
    fn sec8_filter_not_exists() {
        let query = r#"
            SELECT ?name WHERE {
                ?s <http://ex.org/name> ?name .
                FILTER NOT EXISTS { ?s <http://ex.org/email> ?e }
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "FILTER NOT EXISTS: {result:?}");
    }

    #[test]
    fn sec8_minus_pattern() {
        let query = r#"
            SELECT ?s ?p ?o WHERE {
                ?s ?p ?o .
                MINUS { ?s <http://ex.org/type> "internal" }
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "MINUS pattern: {result:?}");
    }

    // ====================================================================
    // 9 - Property Paths
    // ====================================================================

    #[test]
    fn sec9_path_alternative() {
        let query = r#"
            SELECT ?name WHERE {
                ?x <http://xmlns.com/foaf/0.1/name>|<http://xmlns.com/foaf/0.1/nick> ?name
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "Alternative path (|): {result:?}");
    }

    #[test]
    fn sec9_path_sequence() {
        let query = r#"
            SELECT ?friendName WHERE {
                ?x <http://xmlns.com/foaf/0.1/knows>/<http://xmlns.com/foaf/0.1/name> ?friendName
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "Sequence path (/): {result:?}");
    }

    #[test]
    fn sec9_path_inverse() {
        let query = r#"
            SELECT ?parent WHERE {
                ?child ^<http://ex.org/hasChild> ?parent
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "Inverse path (^): {result:?}");
    }

    #[test]
    fn sec9_path_zero_or_more() {
        let query = r#"
            SELECT ?ancestor WHERE {
                ?x <http://ex.org/parent>* ?ancestor
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "Zero-or-more path (*): {result:?}");
    }

    #[test]
    fn sec9_path_one_or_more() {
        let query = r#"
            SELECT ?ancestor WHERE {
                ?x <http://ex.org/parent>+ ?ancestor
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "One-or-more path (+): {result:?}");
    }

    #[test]
    fn sec9_path_zero_or_one() {
        let query = r#"
            SELECT ?name WHERE {
                ?x <http://xmlns.com/foaf/0.1/knows>? ?friend .
                ?friend <http://xmlns.com/foaf/0.1/name> ?name
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "Zero-or-one path (?): {result:?}");
    }

    #[test]
    fn sec9_path_negated_property_set() {
        let query = r#"
            SELECT ?s ?o WHERE {
                ?s !<http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?o
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "Negated property set (!iri): {result:?}");
    }

    #[test]
    fn sec9_path_rdf_type_shorthand() {
        let query = r#"
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            SELECT ?person WHERE { ?person a foaf:Person }
        "#;
        let result = sparql::parse(query).unwrap();
        // Verify the 'a' shorthand parses as RdfType property path
        if let ast::QueryForm::Select(select) = &result.query_form {
            // We just need to verify it parsed without error
            assert!(matches!(select.modifier, ast::SelectModifier::None));
        }
    }

    // ====================================================================
    // 10 - Assignment (BIND, VALUES)
    // ====================================================================

    #[test]
    fn sec10_bind_arithmetic() {
        let query = r#"
            SELECT ?name ?discounted WHERE {
                ?item <http://ex.org/name> ?name .
                ?item <http://ex.org/price> ?price .
                BIND(?price * 0.9 AS ?discounted)
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "BIND with arithmetic: {result:?}");
    }

    #[test]
    fn sec10_values_multiple_variables() {
        let query = r#"
            SELECT ?name ?age WHERE {
                VALUES (?person ?age) {
                    (<http://ex.org/alix> "30")
                    (<http://ex.org/gus>  "25")
                }
                ?person <http://xmlns.com/foaf/0.1/name> ?name
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "VALUES with multiple variables: {result:?}");
    }

    #[test]
    fn sec10_values_with_undef() {
        let query = r#"
            SELECT ?name WHERE {
                VALUES (?x ?y) {
                    (<http://ex.org/a> UNDEF)
                    (UNDEF <http://ex.org/b>)
                }
                ?x <http://ex.org/name> ?name
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "VALUES with UNDEF: {result:?}");
    }

    // ====================================================================
    // 10.1 - BIND scope: the target must not be in scope (18.2.1)
    // ====================================================================

    /// The W3C negative syntax tests `syntax-BINDscope6.rq` to `8.rq`, as
    /// the test suite writes them, with the variable the error names.
    const W3C_BIND_SCOPE_NEGATIVES: [(&str, &str, &str); 3] = [
        (
            "syntax-BINDscope6",
            "PREFIX : <http://www.example.org>
             SELECT *
             WHERE {
                :s :p ?o .
                :s :q ?o1 .
                BIND((1+?o) AS ?o1)
             }",
            "?o1",
        ),
        (
            "syntax-BINDscope7",
            "PREFIX : <http://www.example.org>
             SELECT *
             WHERE {
                {
                :s :p ?o .
                :s :q ?o1 .
                }
                BIND((1+?o) AS ?o1)
             }",
            "?o1",
        ),
        (
            "syntax-BINDscope8",
            "PREFIX : <http://www.example.org>
            SELECT *
              {
                {
                   { :s :p ?Y }
                   UNION
                   { :s :p ?Z }
                }
               BIND (1 AS ?Y)
             }",
            "?Y",
        ),
    ];

    /// The W3C positive syntax tests `syntax-BINDscope1.rq` to `5.rq`.
    const W3C_BIND_SCOPE_POSITIVES: [(&str, &str); 5] = [
        (
            "syntax-BINDscope1",
            "PREFIX : <http://www.example.org>
             SELECT *
             WHERE {
                :s :p ?o .
                BIND((1+?o) AS ?o1)
                :s :q ?o1
             }",
        ),
        (
            "syntax-BINDscope2",
            "PREFIX : <http://www.example.org>
             SELECT *
             WHERE {
                :s :p ?o .
                :s :q ?o1
                { BIND((1+?o) AS ?o1) }
             }",
        ),
        (
            "syntax-BINDscope3",
            "PREFIX : <http://www.example.org>
             SELECT *
             WHERE {
                {
                :s :p ?o .
                :s :q ?o1
                }
                { BIND((1+?o) AS ?o1) }
             }",
        ),
        (
            "syntax-BINDscope4",
            "PREFIX : <http://www.example.org>
            SELECT *
               {
                { BIND (1 AS ?Y) }
                 UNION
                { :s :p ?Y }
              }",
        ),
        (
            "syntax-BINDscope5",
            "PREFIX : <http://www.example.org>
            SELECT *
               {
                { :s :p ?Y }
                 UNION
                { BIND (1 AS ?Y) }
               }",
        ),
    ];

    /// Collects the cases of `cases` that `sparql::parse` accepts, or refuses
    /// without naming the variable.
    fn not_refused(cases: &[(&str, &str, &str)]) -> Vec<String> {
        cases
            .iter()
            .filter_map(|(name, query, variable)| match sparql::parse(query) {
                Ok(_) => Some(format!("{name}: accepted")),
                Err(error) if !error.to_string().contains(variable) => Some(format!(
                    "{name}: refused without naming {variable}: {error}"
                )),
                Err(_) => None,
            })
            .collect()
    }

    /// Collects the cases of `cases` that `sparql::parse` refuses.
    fn not_accepted(cases: &[(&str, &str)]) -> Vec<String> {
        cases
            .iter()
            .filter_map(|(name, query)| {
                sparql::parse(query)
                    .err()
                    .map(|error| format!("{name}: {error}"))
            })
            .collect()
    }

    #[test]
    fn sec10_1_w3c_bind_scope_negatives_are_refused() {
        let failures = not_refused(&W3C_BIND_SCOPE_NEGATIVES);
        assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    }

    #[test]
    fn sec10_1_w3c_bind_scope_positives_are_accepted() {
        let failures = not_accepted(&W3C_BIND_SCOPE_POSITIVES);
        assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    }

    /// Every form that puts the target in scope before the BIND, in a query
    /// and in an update's WHERE (#703).
    #[test]
    fn sec10_1_bind_to_a_variable_in_scope_is_refused() {
        let failures = not_refused(&[
            (
                "a triple pattern",
                "SELECT ?s ?o WHERE { ?s <urn:p> ?o BIND(<urn:z> AS ?o) }",
                "?o",
            ),
            (
                "VALUES",
                r#"SELECT ?o WHERE { VALUES ?o { "x" } BIND("y" AS ?o) }"#,
                "?o",
            ),
            (
                "an OPTIONAL that may bind nothing",
                r#"SELECT ?o WHERE { ?s <urn:p> ?v OPTIONAL { ?s <urn:q> ?o } BIND("y" AS ?o) }"#,
                "?o",
            ),
            (
                "an earlier BIND",
                "SELECT ?x WHERE { BIND(1 AS ?x) BIND(2 AS ?x) }",
                "?x",
            ),
            (
                "a sub-SELECT that projects it",
                "SELECT ?n WHERE { { SELECT ?n WHERE { ?s <urn:name> ?n } } BIND(1 AS ?n) }",
                "?n",
            ),
            (
                "a GRAPH pattern",
                "SELECT ?o WHERE { GRAPH <urn:g> { ?s <urn:p> ?o } BIND(1 AS ?o) }",
                "?o",
            ),
            (
                "an INSERT's WHERE",
                "INSERT { <urn:w> <urn:got> ?o } WHERE { ?s <urn:p> ?o BIND(<urn:z3> AS ?o) }",
                "?o",
            ),
            (
                "a DELETE's WHERE",
                "DELETE { ?s <urn:p> ?o } WHERE { ?s <urn:p> ?o BIND(<urn:z3> AS ?o) }",
                "?o",
            ),
        ]);
        assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    }

    /// Forms that put nothing in scope, or only in another group: the BIND
    /// stays legal.
    #[test]
    fn sec10_1_bind_of_a_variable_out_of_scope_is_accepted() {
        let failures = not_accepted(&[
            (
                "after a FILTER that mentions it",
                r#"SELECT ?f WHERE { ?s <urn:p> ?o FILTER(?f = "y") BIND("y" AS ?f) }"#,
            ),
            (
                "after a FILTER EXISTS whose pattern binds it",
                "SELECT ?x WHERE { ?s <urn:p> ?o FILTER EXISTS { ?s <urn:q> ?x } BIND(1 AS ?x) }",
            ),
            (
                "after a MINUS whose pattern binds it",
                r#"SELECT ?m WHERE { ?s <urn:p> ?o MINUS { ?s <urn:q> ?m } BIND("y" AS ?m) }"#,
            ),
            (
                "in a nested group, of a variable a later sibling binds",
                r#"SELECT ?s WHERE { { BIND("y" AS ?n) } ?s <urn:p> ?n }"#,
            ),
            (
                "of a variable a sub-SELECT uses but does not project",
                r#"SELECT ?w WHERE { { SELECT ?s WHERE { ?s <urn:p> ?w } } BIND("y" AS ?w) }"#,
            ),
            (
                "inside an OPTIONAL, of a variable of the outer group",
                "SELECT ?o WHERE { ?s <urn:p> ?o OPTIONAL { BIND(1 AS ?o) } }",
            ),
            (
                "before the pattern that binds it",
                "SELECT ?o WHERE { BIND(1 AS ?o) ?s <urn:p> ?o }",
            ),
        ]);
        assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    }

    // ====================================================================
    // 11 - Aggregates
    // ====================================================================

    #[test]
    fn sec11_count_star() {
        let query = "SELECT (COUNT(*) AS ?total) WHERE { ?s ?p ?o }";
        let result = sparql::parse(query).unwrap();
        if let ast::QueryForm::Select(select) = &result.query_form
            && let ast::Projection::Variables(vars) = &select.projection
        {
            assert_eq!(vars[0].alias.as_deref(), Some("total"));
        }
    }

    #[test]
    fn sec11_count_distinct() {
        let query = "SELECT (COUNT(DISTINCT ?type) AS ?types) WHERE { ?s a ?type }";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "COUNT(DISTINCT): {result:?}");
    }

    #[test]
    fn sec11_sum() {
        let query = r#"
            SELECT ?dept (SUM(?salary) AS ?totalSalary)
            WHERE { ?emp <http://ex.org/dept> ?dept . ?emp <http://ex.org/salary> ?salary }
            GROUP BY ?dept
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "SUM aggregate: {result:?}");
    }

    #[test]
    fn sec11_min_max() {
        let query = r#"
            SELECT (MIN(?age) AS ?youngest) (MAX(?age) AS ?oldest)
            WHERE { ?s <http://ex.org/age> ?age }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "MIN/MAX aggregates: {result:?}");
    }

    #[test]
    fn sec11_group_concat() {
        let query = r#"
            SELECT ?dept (GROUP_CONCAT(?name; SEPARATOR=", ") AS ?names)
            WHERE { ?emp <http://ex.org/dept> ?dept . ?emp <http://ex.org/name> ?name }
            GROUP BY ?dept
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "GROUP_CONCAT with separator: {result:?}");
    }

    #[test]
    fn sec11_sample() {
        let query = r#"
            SELECT ?type (SAMPLE(?name) AS ?example)
            WHERE { ?s a ?type . ?s <http://ex.org/name> ?name }
            GROUP BY ?type
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "SAMPLE aggregate: {result:?}");
    }

    #[test]
    fn sec11_having_clause() {
        let query = r#"
            SELECT ?type (COUNT(?s) AS ?cnt)
            WHERE { ?s a ?type }
            GROUP BY ?type
            HAVING (COUNT(?s) > 1)
        "#;
        let result = sparql::parse(query).unwrap();
        if let ast::QueryForm::Select(select) = &result.query_form {
            assert!(select.solution_modifiers.having.is_some());
        }
    }

    // ====================================================================
    // 12 - Subqueries
    // ====================================================================

    #[test]
    fn sec12_subquery_in_where() {
        let query = r#"
            SELECT ?name ?maxAge WHERE {
                ?person <http://xmlns.com/foaf/0.1/name> ?name .
                {
                    SELECT ?person (MAX(?a) AS ?maxAge) WHERE {
                        ?person <http://xmlns.com/foaf/0.1/age> ?a
                    }
                    GROUP BY ?person
                }
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "Subquery with GROUP BY: {result:?}");
    }

    // ====================================================================
    // 13 - RDF Dataset (FROM / FROM NAMED)
    // ====================================================================

    #[test]
    fn sec13_from_clause() {
        let query = r#"
            SELECT ?s ?p ?o
            FROM <http://example.org/default-graph>
            WHERE { ?s ?p ?o }
        "#;
        let result = sparql::parse(query).unwrap();
        if let ast::QueryForm::Select(select) = &result.query_form {
            let ds = select.dataset.as_ref().unwrap();
            assert_eq!(ds.default_graphs.len(), 1);
            assert_eq!(
                ds.default_graphs[0].as_str(),
                "http://example.org/default-graph"
            );
        }
    }

    #[test]
    fn sec13_from_named_clause() {
        let query = r#"
            SELECT ?g ?s ?p ?o
            FROM NAMED <http://example.org/graph1>
            FROM NAMED <http://example.org/graph2>
            WHERE { GRAPH ?g { ?s ?p ?o } }
        "#;
        let result = sparql::parse(query).unwrap();
        if let ast::QueryForm::Select(select) = &result.query_form {
            let ds = select.dataset.as_ref().unwrap();
            assert_eq!(ds.named_graphs.len(), 2);
        }
    }

    #[test]
    fn sec13_from_and_from_named_combined() {
        let query = r#"
            SELECT ?s ?p ?o
            FROM <http://example.org/default>
            FROM NAMED <http://example.org/named1>
            WHERE { ?s ?p ?o }
        "#;
        let result = sparql::parse(query).unwrap();
        if let ast::QueryForm::Select(select) = &result.query_form {
            let ds = select.dataset.as_ref().unwrap();
            assert_eq!(ds.default_graphs.len(), 1);
            assert_eq!(ds.named_graphs.len(), 1);
        }
    }

    // ====================================================================
    // 13.3 - Querying Named Graphs (GRAPH)
    // ====================================================================

    #[test]
    fn sec13_3_graph_with_iri() {
        let query = r#"
            SELECT ?s ?p ?o WHERE {
                GRAPH <http://example.org/mygraph> { ?s ?p ?o }
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "GRAPH with IRI: {result:?}");
    }

    #[test]
    fn sec13_3_graph_with_variable() {
        let query = r#"
            SELECT ?g ?s ?p ?o WHERE {
                GRAPH ?g { ?s ?p ?o }
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "GRAPH with variable: {result:?}");
    }

    // ====================================================================
    // 15 - Solution Modifiers
    // ====================================================================

    #[test]
    fn sec15_select_reduced() {
        let query = "SELECT REDUCED ?type WHERE { ?s a ?type }";
        let result = sparql::parse(query).unwrap();
        if let ast::QueryForm::Select(select) = &result.query_form {
            assert_eq!(select.modifier, ast::SelectModifier::Reduced);
        }
    }

    #[test]
    fn sec15_order_by_multiple_keys() {
        let query = r#"
            SELECT ?name ?age WHERE {
                ?s <http://ex.org/name> ?name .
                ?s <http://ex.org/age> ?age
            }
            ORDER BY DESC(?age) ?name
        "#;
        let result = sparql::parse(query).unwrap();
        if let ast::QueryForm::Select(select) = &result.query_form {
            let order = select.solution_modifiers.order_by.as_ref().unwrap();
            assert_eq!(order.len(), 2);
            assert_eq!(order[0].direction, ast::SortDirection::Descending);
            assert_eq!(order[1].direction, ast::SortDirection::Ascending);
        }
    }

    #[test]
    fn sec15_limit_and_offset_combined() {
        let query = "SELECT ?s WHERE { ?s ?p ?o } LIMIT 10 OFFSET 20";
        let result = sparql::parse(query).unwrap();
        if let ast::QueryForm::Select(select) = &result.query_form {
            assert_eq!(select.solution_modifiers.limit, Some(10));
            assert_eq!(select.solution_modifiers.offset, Some(20));
        }
    }

    // ====================================================================
    // 16 - Query Forms: CONSTRUCT, ASK, DESCRIBE
    // ====================================================================

    #[test]
    fn sec16_construct_with_bind() {
        let query = r#"
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            CONSTRUCT {
                ?person foaf:fullName ?full
            }
            WHERE {
                ?person foaf:firstName ?first .
                ?person foaf:lastName ?last .
                BIND(CONCAT(?first, " ", ?last) AS ?full)
            }
        "#;
        let result = sparql::parse(query).unwrap();
        assert!(matches!(result.query_form, ast::QueryForm::Construct(_)));
    }

    #[test]
    fn sec16_ask_with_filter() {
        let query = r#"
            ASK {
                ?s <http://ex.org/age> ?age .
                FILTER(?age > 18)
            }
        "#;
        let result = sparql::parse(query).unwrap();
        assert!(matches!(result.query_form, ast::QueryForm::Ask(_)));
    }

    #[test]
    fn sec16_describe_iri() {
        let query = "DESCRIBE <http://example.org/alix>";
        let result = sparql::parse(query).unwrap();
        if let ast::QueryForm::Describe(desc) = &result.query_form {
            assert_eq!(desc.resources.len(), 1);
            assert!(desc.where_clause.is_none());
        }
    }

    #[test]
    fn sec16_describe_multiple_resources() {
        let query = "DESCRIBE <http://ex.org/alix> <http://ex.org/gus>";
        let result = sparql::parse(query).unwrap();
        if let ast::QueryForm::Describe(desc) = &result.query_form {
            assert_eq!(desc.resources.len(), 2);
        }
    }

    // ====================================================================
    // 17.4 - Function Definitions (Built-in Functions)
    // ====================================================================

    // -- 17.4.1 Functional Forms --

    #[test]
    fn sec17_bound() {
        let query = r#"
            SELECT ?name WHERE {
                ?s <http://ex.org/name> ?name .
                OPTIONAL { ?s <http://ex.org/email> ?email }
                FILTER(BOUND(?email))
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "BOUND(): {result:?}");
    }

    #[test]
    fn sec17_if_conditional() {
        let query = r#"
            SELECT ?label WHERE {
                ?s <http://ex.org/val> ?v .
                BIND(IF(?v > 10, "high", "low") AS ?label)
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "IF(): {result:?}");
    }

    #[test]
    fn sec17_coalesce() {
        let query = r#"
            SELECT ?display WHERE {
                ?s <http://ex.org/name> ?name .
                OPTIONAL { ?s <http://ex.org/nick> ?nick }
                BIND(COALESCE(?nick, ?name) AS ?display)
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "COALESCE(): {result:?}");
    }

    #[test]
    fn sec17_in_operator() {
        let query = r#"
            SELECT ?s WHERE {
                ?s <http://ex.org/type> ?t .
                FILTER(?t IN ("Person", "Organization"))
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "IN operator: {result:?}");
    }

    #[test]
    fn sec17_not_in_operator() {
        let query = r#"
            SELECT ?s WHERE {
                ?s <http://ex.org/status> ?st .
                FILTER(?st NOT IN ("deleted", "archived"))
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "NOT IN operator: {result:?}");
    }

    // -- 17.4.2 RDF Term Functions --

    #[test]
    fn sec17_is_iri() {
        let query = "SELECT ?s WHERE { ?s ?p ?o FILTER(isIRI(?o)) }";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "isIRI(): {result:?}");
    }

    #[test]
    fn sec17_is_blank() {
        let query = "SELECT ?s WHERE { ?s ?p ?o FILTER(isBLANK(?o)) }";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "isBLANK(): {result:?}");
    }

    #[test]
    fn sec17_is_literal() {
        let query = "SELECT ?o WHERE { ?s ?p ?o FILTER(isLITERAL(?o)) }";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "isLITERAL(): {result:?}");
    }

    #[test]
    fn sec17_is_numeric() {
        let query = "SELECT ?o WHERE { ?s ?p ?o FILTER(isNUMERIC(?o)) }";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "isNUMERIC(): {result:?}");
    }

    #[test]
    fn sec17_str_function() {
        let query = r#"SELECT (STR(?val) AS ?s) WHERE { ?x <http://ex.org/p> ?val }"#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "STR(): {result:?}");
    }

    #[test]
    fn sec17_lang_function() {
        let query = r#"
            SELECT ?label WHERE {
                ?s <http://www.w3.org/2000/01/rdf-schema#label> ?label .
                FILTER(LANG(?label) = "en")
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "LANG(): {result:?}");
    }

    #[test]
    fn sec17_langmatches() {
        let query = r#"
            SELECT ?label WHERE {
                ?s <http://www.w3.org/2000/01/rdf-schema#label> ?label .
                FILTER(LANGMATCHES(LANG(?label), "en"))
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "LANGMATCHES(): {result:?}");
    }

    #[test]
    fn sec17_datatype_function() {
        let query = r#"
            SELECT ?val WHERE {
                ?s <http://ex.org/val> ?val .
                FILTER(DATATYPE(?val) = <http://www.w3.org/2001/XMLSchema#integer>)
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "DATATYPE(): {result:?}");
    }

    #[test]
    fn sec17_same_term() {
        let query = r#"
            SELECT ?s WHERE {
                ?s <http://ex.org/p> ?a .
                ?s <http://ex.org/q> ?b .
                FILTER(sameTerm(?a, ?b))
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "sameTerm(): {result:?}");
    }

    // -- 17.4.3 String Functions --

    #[test]
    fn sec17_substr() {
        let query =
            r#"SELECT (SUBSTR(?name, 1, 3) AS ?prefix) WHERE { ?s <http://ex.org/name> ?name }"#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "SUBSTR(): {result:?}");
    }

    #[test]
    fn sec17_strends() {
        let query = r#"
            SELECT ?name WHERE {
                ?s <http://ex.org/name> ?name .
                FILTER(STRENDS(?name, "son"))
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "STRENDS(): {result:?}");
    }

    #[test]
    fn sec17_strbefore() {
        let query = r#"SELECT (STRBEFORE("Alix Vega", " ") AS ?first) WHERE { }"#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "STRBEFORE(): {result:?}");
    }

    #[test]
    fn sec17_strafter() {
        let query = r#"SELECT (STRAFTER("Alix Vega", " ") AS ?last) WHERE { }"#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "STRAFTER(): {result:?}");
    }

    #[test]
    fn sec17_lcase() {
        let query = r#"SELECT (LCASE(?name) AS ?lower) WHERE { ?s <http://ex.org/name> ?name }"#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "LCASE(): {result:?}");
    }

    #[test]
    fn sec17_encode_for_uri() {
        let query = r#"SELECT (ENCODE_FOR_URI(?label) AS ?encoded) WHERE { ?s <http://ex.org/label> ?label }"#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "ENCODE_FOR_URI(): {result:?}");
    }

    #[test]
    fn sec17_replace() {
        let query = r#"
            SELECT (REPLACE(?name, "a", "e") AS ?changed) WHERE {
                ?s <http://ex.org/name> ?name
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "REPLACE(): {result:?}");
    }

    #[test]
    fn sec17_regex_with_flags() {
        let query = r#"
            SELECT ?name WHERE {
                ?s <http://ex.org/name> ?name .
                FILTER(REGEX(?name, "^alix$", "i"))
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "REGEX with flags: {result:?}");
    }

    #[test]
    fn sec17_concat_multiple_args() {
        let query = r#"
            SELECT (CONCAT(?first, " ", ?last) AS ?full) WHERE {
                ?s <http://ex.org/first> ?first .
                ?s <http://ex.org/last> ?last
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "CONCAT(): {result:?}");
    }

    #[test]
    fn sec17_strdt() {
        let query = r#"SELECT (STRDT("123", <http://www.w3.org/2001/XMLSchema#integer>) AS ?typed) WHERE { }"#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "STRDT(): {result:?}");
    }

    #[test]
    fn sec17_strlang() {
        let query = r#"SELECT (STRLANG("chat", "fr") AS ?tagged) WHERE { }"#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "STRLANG(): {result:?}");
    }

    // -- 17.4.4 Numeric Functions --

    #[test]
    fn sec17_ceil() {
        let query = r#"SELECT (CEIL(10.5) AS ?result) WHERE { }"#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "CEIL(): {result:?}");
    }

    #[test]
    fn sec17_rand() {
        let query = "SELECT (RAND() AS ?rnd) WHERE { }";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "RAND(): {result:?}");
    }

    // -- 17.4.5 Date/Time Functions --

    #[test]
    fn sec17_now() {
        let query = "SELECT (NOW() AS ?ts) WHERE { }";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "NOW(): {result:?}");
    }

    #[test]
    fn sec17_year_month_day() {
        let query = r#"
            SELECT (YEAR(?d) AS ?y) (MONTH(?d) AS ?m) (DAY(?d) AS ?dy) WHERE {
                ?s <http://ex.org/date> ?d
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "YEAR/MONTH/DAY: {result:?}");
    }

    #[test]
    fn sec17_hours_minutes_seconds() {
        let query = r#"
            SELECT (HOURS(?t) AS ?h) (MINUTES(?t) AS ?min) (SECONDS(?t) AS ?sec) WHERE {
                ?s <http://ex.org/time> ?t
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "HOURS/MINUTES/SECONDS: {result:?}");
    }

    #[test]
    fn sec17_timezone_tz() {
        let query = r#"
            SELECT (TIMEZONE(?d) AS ?tz1) (TZ(?d) AS ?tz2) WHERE {
                ?s <http://ex.org/date> ?d
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "TIMEZONE/TZ: {result:?}");
    }

    // -- 17.4.6 Hash Functions --

    #[test]
    fn sec17_md5() {
        let query = r#"SELECT (MD5("hello") AS ?hash) WHERE { }"#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "MD5(): {result:?}");
    }

    #[test]
    fn sec17_sha1() {
        let query = r#"SELECT (SHA1("hello") AS ?hash) WHERE { }"#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "SHA1(): {result:?}");
    }

    #[test]
    fn sec17_sha256() {
        let query = r#"SELECT (SHA256("hello") AS ?hash) WHERE { }"#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "SHA256(): {result:?}");
    }

    #[test]
    fn sec17_sha384() {
        let query = r#"SELECT (SHA384("hello") AS ?hash) WHERE { }"#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "SHA384(): {result:?}");
    }

    #[test]
    fn sec17_sha512() {
        let query = r#"SELECT (SHA512("hello") AS ?hash) WHERE { }"#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "SHA512(): {result:?}");
    }

    // -- 17.4.7 RDF Term Construction --

    #[test]
    fn sec17_iri_function() {
        let query = r#"SELECT (IRI("http://example.org/test") AS ?i) WHERE { }"#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "IRI(): {result:?}");
    }

    #[test]
    fn sec17_bnode_function() {
        let query = r#"SELECT (BNODE("label") AS ?b) WHERE { }"#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "BNODE(): {result:?}");
    }

    #[test]
    fn sec17_uuid() {
        let query = "SELECT (UUID() AS ?id) WHERE { }";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "UUID(): {result:?}");
    }

    #[test]
    fn sec17_struuid() {
        let query = "SELECT (STRUUID() AS ?id) WHERE { }";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "STRUUID(): {result:?}");
    }

    // ====================================================================
    // 17.6 - Operator Precedence & Complex Expressions
    // ====================================================================

    #[test]
    fn sec17_unary_not() {
        let query = "SELECT ?s WHERE { ?s <http://ex.org/active> ?a FILTER(!?a) }";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "Unary NOT: {result:?}");
    }

    #[test]
    fn sec17_unary_minus() {
        let query = "SELECT (-5 AS ?neg) WHERE { }";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "Unary minus: {result:?}");
    }

    #[test]
    fn sec17_complex_expression_precedence() {
        let query = r#"
            SELECT ?s WHERE {
                ?s <http://ex.org/val> ?v .
                FILTER(?v > 10 && ?v < 100 || ?v = 0 && !(?v = 50))
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "Complex precedence: {result:?}");
    }

    #[test]
    fn sec17_arithmetic_precedence() {
        let query = "SELECT (?a + ?b * ?c - ?d / ?e AS ?result) WHERE { ?s <http://ex.org/a> ?a . ?s <http://ex.org/b> ?b . ?s <http://ex.org/c> ?c . ?s <http://ex.org/d> ?d . ?s <http://ex.org/e> ?e }";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "Arithmetic precedence: {result:?}");
    }

    #[test]
    fn sec17_comparison_operators() {
        let query = r#"
            SELECT ?s WHERE {
                ?s <http://ex.org/val> ?v .
                FILTER(?v >= 10 && ?v <= 100 && ?v != 50)
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), ">=, <=, != operators: {result:?}");
    }

    // ====================================================================
    // 18 - SERVICE (Federated Query)
    // ====================================================================

    #[test]
    fn sec18_service() {
        let query = r#"
            SELECT ?s ?name WHERE {
                SERVICE <http://dbpedia.org/sparql> {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name
                }
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "SERVICE: {result:?}");
    }

    #[test]
    fn sec18_service_silent() {
        let query = r#"
            SELECT ?s ?name WHERE {
                SERVICE SILENT <http://dbpedia.org/sparql> {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name
                }
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "SERVICE SILENT: {result:?}");
    }

    // ====================================================================
    // SPARQL Update Operations (W3C SPARQL 1.1 Update)
    // ====================================================================

    #[test]
    fn update_delete_where() {
        let query = r#"
            DELETE WHERE {
                ?s <http://ex.org/deprecated> ?o
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "DELETE WHERE: {result:?}");
    }

    #[test]
    fn update_modify_delete_insert() {
        let query = r#"
            DELETE { ?s <http://ex.org/status> "draft" }
            INSERT { ?s <http://ex.org/status> "published" }
            WHERE  { ?s <http://ex.org/status> "draft" }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "DELETE/INSERT (Modify): {result:?}");
    }

    #[test]
    fn update_insert_where() {
        let query = r#"
            INSERT { ?s <http://ex.org/fullName> ?full }
            WHERE  {
                ?s <http://ex.org/first> ?f .
                ?s <http://ex.org/last> ?l .
                BIND(CONCAT(?f, " ", ?l) AS ?full)
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "INSERT WHERE: {result:?}");
    }

    #[test]
    fn update_with_graph() {
        let query = r#"
            WITH <http://example.org/mygraph>
            DELETE { ?s <http://ex.org/old> ?o }
            INSERT { ?s <http://ex.org/new> ?o }
            WHERE  { ?s <http://ex.org/old> ?o }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "WITH graph: {result:?}");
    }

    #[test]
    fn update_load() {
        let query = r#"LOAD <http://example.org/data.ttl>"#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "LOAD: {result:?}");
    }

    #[test]
    fn update_load_into_graph() {
        let query = r#"LOAD <http://example.org/data.ttl> INTO GRAPH <http://example.org/mygraph>"#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "LOAD INTO GRAPH: {result:?}");
    }

    #[test]
    fn update_load_silent() {
        let query = r#"LOAD SILENT <http://example.org/data.ttl>"#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "LOAD SILENT: {result:?}");
    }

    #[test]
    fn update_clear_default() {
        let query = "CLEAR DEFAULT";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "CLEAR DEFAULT: {result:?}");
    }

    #[test]
    fn update_clear_named_graph() {
        let query = "CLEAR GRAPH <http://example.org/mygraph>";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "CLEAR GRAPH: {result:?}");
    }

    #[test]
    fn update_clear_all() {
        let query = "CLEAR ALL";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "CLEAR ALL: {result:?}");
    }

    #[test]
    fn update_drop_graph() {
        let query = "DROP GRAPH <http://example.org/mygraph>";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "DROP GRAPH: {result:?}");
    }

    #[test]
    fn update_drop_silent() {
        let query = "DROP SILENT GRAPH <http://example.org/mygraph>";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "DROP SILENT: {result:?}");
    }

    #[test]
    fn update_create_graph() {
        let query = "CREATE GRAPH <http://example.org/newgraph>";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "CREATE GRAPH: {result:?}");
    }

    #[test]
    fn update_create_silent() {
        let query = "CREATE SILENT GRAPH <http://example.org/newgraph>";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "CREATE SILENT: {result:?}");
    }

    #[test]
    fn update_copy() {
        let query = "COPY <http://example.org/src> TO <http://example.org/dst>";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "COPY: {result:?}");
    }

    #[test]
    fn update_move() {
        let query = "MOVE <http://example.org/src> TO <http://example.org/dst>";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "MOVE: {result:?}");
    }

    #[test]
    fn update_add() {
        let query = "ADD <http://example.org/src> TO <http://example.org/dst>";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "ADD: {result:?}");
    }

    #[test]
    fn update_copy_default_to_named() {
        let query = "COPY DEFAULT TO <http://example.org/dst>";
        let result = sparql::parse(query);
        assert!(result.is_ok(), "COPY DEFAULT TO named: {result:?}");
    }

    // ====================================================================
    // Parser Error Cases
    // ====================================================================

    #[test]
    fn error_missing_where() {
        // SELECT without WHERE is technically allowed in SPARQL 1.1 (section 18.2.1)
        // but some parsers require it. Test what our parser does.
        let query = "SELECT ?s { ?s ?p ?o }";
        // This should still parse (WHERE is optional per spec)
        let result = sparql::parse(query);
        assert!(result.is_ok(), "SELECT without WHERE keyword: {result:?}");
    }

    #[test]
    fn error_unterminated_string() {
        let query = r#"SELECT ?s WHERE { ?s <http://ex.org/name> "unterminated }"#;
        let result = sparql::parse(query);
        assert!(result.is_err(), "Unterminated string should fail");
    }

    #[test]
    fn error_missing_closing_angle() {
        let query = "SELECT ?s WHERE { ?s <http://ex.org/name ?o }";
        let result = sparql::parse(query);
        assert!(result.is_err(), "Missing > in IRI should fail");
    }

    #[test]
    fn error_select_without_projection() {
        let query = "SELECT WHERE { ?s ?p ?o }";
        let result = sparql::parse(query);
        assert!(result.is_err(), "SELECT with no variables should fail");
    }

    // ====================================================================
    // INSERT DATA with Named Graphs
    // ====================================================================

    #[test]
    fn update_insert_data_into_named_graph() {
        let query = r#"
            INSERT DATA {
                GRAPH <http://example.org/mygraph> {
                    <http://ex.org/alix> <http://ex.org/name> "Alix"
                }
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "INSERT DATA into named graph: {result:?}");
    }

    #[test]
    fn update_delete_data_from_named_graph() {
        let query = r#"
            DELETE DATA {
                GRAPH <http://example.org/mygraph> {
                    <http://ex.org/alix> <http://ex.org/name> "Alix"
                }
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "DELETE DATA from named graph: {result:?}");
    }

    // ====================================================================
    // SELECT Expressions (Computed Columns)
    // ====================================================================

    #[test]
    fn select_expression_arithmetic() {
        let query = r#"
            SELECT ?name (?price * 1.1 AS ?priceWithTax) WHERE {
                ?item <http://ex.org/name> ?name .
                ?item <http://ex.org/price> ?price
            }
        "#;
        let result = sparql::parse(query).unwrap();
        if let ast::QueryForm::Select(select) = &result.query_form
            && let ast::Projection::Variables(vars) = &select.projection
        {
            assert_eq!(vars.len(), 2);
            assert_eq!(vars[1].alias.as_deref(), Some("priceWithTax"));
        }
    }

    #[test]
    fn select_expression_string_function() {
        let query = r#"
            SELECT ?name (STRLEN(?name) AS ?len) (UCASE(?name) AS ?upper) (LCASE(?name) AS ?lower) WHERE {
                ?s <http://ex.org/name> ?name
            }
        "#;
        let result = sparql::parse(query).unwrap();
        if let ast::QueryForm::Select(select) = &result.query_form
            && let ast::Projection::Variables(vars) = &select.projection
        {
            assert_eq!(vars.len(), 4);
        }
    }

    // ====================================================================
    // Complex Combinations
    // ====================================================================

    #[test]
    fn complex_query_all_modifiers() {
        let query = r#"
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX ex: <http://example.org/>
            SELECT DISTINCT ?name (COUNT(?friend) AS ?friendCount)
            WHERE {
                ?person a foaf:Person .
                ?person foaf:name ?name .
                OPTIONAL { ?person foaf:knows ?friend }
                FILTER(BOUND(?name))
            }
            GROUP BY ?name
            HAVING (COUNT(?friend) > 0)
            ORDER BY DESC(?friendCount) ?name
            LIMIT 10
            OFFSET 5
        "#;
        let result = sparql::parse(query).unwrap();
        assert_eq!(result.prefixes.len(), 2);
        if let ast::QueryForm::Select(select) = &result.query_form {
            assert_eq!(select.modifier, ast::SelectModifier::Distinct);
            assert!(select.solution_modifiers.group_by.is_some());
            assert!(select.solution_modifiers.having.is_some());
            assert!(select.solution_modifiers.order_by.is_some());
            assert_eq!(select.solution_modifiers.limit, Some(10));
            assert_eq!(select.solution_modifiers.offset, Some(5));
        }
    }

    #[test]
    fn complex_nested_optional_union_filter() {
        let query = r#"
            SELECT ?name ?email ?phone WHERE {
                ?s <http://ex.org/name> ?name .
                OPTIONAL {
                    { ?s <http://ex.org/email> ?email }
                    UNION
                    { ?s <http://ex.org/phone> ?phone }
                }
                FILTER(?name != "")
            }
        "#;
        let result = sparql::parse(query);
        assert!(
            result.is_ok(),
            "Nested OPTIONAL with UNION and FILTER: {result:?}"
        );
    }

    #[test]
    fn complex_multiple_subqueries() {
        let query = r#"
            SELECT ?name ?avgAge WHERE {
                ?person <http://ex.org/name> ?name .
                {
                    SELECT (AVG(?age) AS ?avgAge) WHERE {
                        ?p <http://ex.org/age> ?age
                    }
                }
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok(), "Multiple subqueries: {result:?}");
    }
}
