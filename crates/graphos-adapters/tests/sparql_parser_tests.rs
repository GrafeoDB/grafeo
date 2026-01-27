//! Integration tests for SPARQL parser.

#[cfg(feature = "sparql")]
mod tests {
    use graphos_adapters::query::sparql::{self, ast};

    #[test]
    fn test_parse_simple_select() {
        let query = "SELECT ?x WHERE { ?x ?y ?z }";
        let result = sparql::parse(query);
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert!(matches!(ast.query_form, ast::QueryForm::Select(_)));
    }

    #[test]
    fn test_parse_select_with_multiple_prefixes() {
        let query = r#"
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            SELECT ?name ?age
            WHERE {
                ?person rdf:type foaf:Person .
                ?person foaf:name ?name .
                OPTIONAL { ?person foaf:age ?age }
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.prefixes.len(), 3);
    }

    #[test]
    fn test_parse_complex_filter() {
        let query = r#"
            SELECT ?x ?y
            WHERE {
                ?x ?p ?y
                FILTER(?y > 10 && ?y < 100)
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_union_query() {
        let query = r#"
            SELECT ?name
            WHERE {
                { ?x <http://xmlns.com/foaf/0.1/name> ?name }
                UNION
                { ?x <http://xmlns.com/foaf/0.1/givenName> ?name }
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_aggregation() {
        let query = r#"
            SELECT ?category (COUNT(?item) AS ?count) (AVG(?price) AS ?avgPrice)
            WHERE {
                ?item <http://example.org/category> ?category .
                ?item <http://example.org/price> ?price
            }
            GROUP BY ?category
            HAVING (COUNT(?item) > 5)
            ORDER BY DESC(?count)
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok());
        if let ast::QueryForm::Select(select) = result.unwrap().query_form {
            assert!(select.solution_modifiers.group_by.is_some());
            assert!(select.solution_modifiers.having.is_some());
            assert!(select.solution_modifiers.order_by.is_some());
        } else {
            panic!("Expected SELECT query");
        }
    }

    #[test]
    fn test_parse_property_paths() {
        let query = r#"
            SELECT ?ancestor
            WHERE {
                ?x <http://example.org/parent>+ ?ancestor
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_ask_query() {
        let query = r#"
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            ASK {
                ?person foaf:name "Alice" .
                ?person foaf:knows ?friend
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok());
        assert!(matches!(result.unwrap().query_form, ast::QueryForm::Ask(_)));
    }

    #[test]
    fn test_parse_construct_query() {
        let query = r#"
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            CONSTRUCT {
                ?person foaf:fullName ?name
            }
            WHERE {
                ?person foaf:firstName ?first .
                ?person foaf:lastName ?last
                BIND(CONCAT(?first, " ", ?last) AS ?name)
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok());
        assert!(matches!(
            result.unwrap().query_form,
            ast::QueryForm::Construct(_)
        ));
    }

    #[test]
    fn test_parse_describe_query() {
        let query = r#"
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            DESCRIBE ?person
            WHERE {
                ?person foaf:name "Alice"
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok());
        assert!(matches!(
            result.unwrap().query_form,
            ast::QueryForm::Describe(_)
        ));
    }

    #[test]
    fn test_parse_subquery() {
        let query = r#"
            SELECT ?name ?maxAge
            WHERE {
                ?person <http://xmlns.com/foaf/0.1/name> ?name .
                {
                    SELECT (MAX(?age) AS ?maxAge)
                    WHERE {
                        ?p <http://xmlns.com/foaf/0.1/age> ?age
                    }
                }
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_values_clause() {
        let query = r#"
            SELECT ?name
            WHERE {
                VALUES ?person { <http://example.org/alice> <http://example.org/bob> }
                ?person <http://xmlns.com/foaf/0.1/name> ?name
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_minus() {
        let query = r#"
            SELECT ?name
            WHERE {
                ?person <http://xmlns.com/foaf/0.1/name> ?name
                MINUS {
                    ?person <http://xmlns.com/foaf/0.1/knows> <http://example.org/bob>
                }
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_bind() {
        let query = r#"
            SELECT ?name ?upperName
            WHERE {
                ?person <http://xmlns.com/foaf/0.1/name> ?name
                BIND(UCASE(?name) AS ?upperName)
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_error_unclosed_brace() {
        let query = "SELECT ?x WHERE { ?x ?y ?z";
        let result = sparql::parse(query);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_invalid_query_form() {
        let query = "INVALID ?x WHERE { ?x ?y ?z }";
        let result = sparql::parse(query);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_string_functions() {
        let query = r#"
            SELECT ?name (STRLEN(?name) AS ?len) (UCASE(?name) AS ?upper)
            WHERE {
                ?x <http://xmlns.com/foaf/0.1/name> ?name
                FILTER(CONTAINS(?name, "Alice"))
                FILTER(STRSTARTS(?name, "A"))
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_numeric_functions() {
        let query = r#"
            SELECT ?val (ABS(?val) AS ?absVal) (ROUND(?val) AS ?rounded)
            WHERE {
                ?x <http://example.org/value> ?val
                FILTER(?val >= FLOOR(?val))
            }
        "#;
        let result = sparql::parse(query);
        assert!(result.is_ok());
    }
}
