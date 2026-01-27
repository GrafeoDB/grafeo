//! Integration tests for SPARQL translator.

#[cfg(feature = "sparql")]
mod tests {
    use graphos_engine::query::sparql_translator::translate;

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
}
