//! The parameters of a SPARQL query are initial bindings: a parameter `x`
//! binds `?x` (and `$x`, the same variable) in the WHERE clause before it is
//! evaluated, as Jena and RDF4J bind them. A GraphQL query over RDF takes the
//! defaults of its variables.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test sparql_initial_bindings
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use std::collections::HashMap;
use std::sync::Arc;

use grafeo_common::types::Value;
use grafeo_common::types::rdf::Term;
use grafeo_engine::{Config, GrafeoDB, GraphModel};

const DATA: &str = r#"INSERT DATA {
    <http://example.org/alix> a <http://example.org/Person> ;
        <http://example.org/name> "Alix" ; <http://example.org/age> 30 ;
        <http://example.org/knows> <http://example.org/gus> .
    <http://example.org/gus> a <http://example.org/Person> ;
        <http://example.org/name> "Gus" ; <http://example.org/age> 25 .
    <http://example.org/mia> a <http://example.org/Person> ;
        <http://example.org/name> "Mia" ; <http://example.org/age> 41 .
}"#;

fn db() -> GrafeoDB {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    db.execute_sparql(DATA).unwrap();
    db
}

fn iri(text: &str) -> Value {
    Value::RdfTerm(Arc::new(Term::iri(text)))
}

fn params(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
    pairs
        .iter()
        .map(|(name, value)| ((*name).to_string(), value.clone()))
        .collect()
}

/// The text of each cell of each row, sorted.
fn texts(db: &GrafeoDB, query: &str, bound: &[(&str, Value)]) -> Vec<Vec<String>> {
    let result = db
        .session()
        .execute_sparql_with_params(query, params(bound))
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    let mut rows: Vec<Vec<String>> = result
        .rows()
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| match value {
                    Value::RdfTerm(term) => term.to_string(),
                    Value::Null => "UNBOUND".to_string(),
                    other => other.to_string(),
                })
                .collect()
        })
        .collect();
    rows.sort();
    rows
}

#[test]
fn a_parameter_binds_its_variable() {
    let db = db();
    let query = "SELECT ?name WHERE { ?person <http://example.org/name> ?name }";
    assert_eq!(
        texts(&db, query, &[("person", iri("http://example.org/gus"))]),
        [[r#""Gus""#]]
    );
    assert_eq!(
        texts(&db, query, &[]).len(),
        3,
        "no parameter binds nothing"
    );
    assert_eq!(
        texts(&db, query, &[("person", Value::Null)]).len(),
        3,
        "a NULL binds nothing"
    );
}

#[test]
fn a_name_may_start_with_a_question_mark_or_a_dollar_sign() {
    let db = db();
    let query = "SELECT ?name WHERE { $person <http://example.org/name> ?name }";
    for name in ["person", "?person", "$person"] {
        assert_eq!(
            texts(&db, query, &[(name, iri("http://example.org/mia"))]),
            [[r#""Mia""#]],
            "{name}"
        );
    }
}

/// The bound variable is in the solutions, and a native value is the
/// literal of its type.
#[test]
fn the_binding_is_part_of_each_solution() {
    let db = db();
    assert_eq!(
        texts(
            &db,
            "SELECT ?person ?name WHERE { ?person <http://example.org/name> ?name }",
            &[("person", iri("http://example.org/alix"))]
        ),
        [["<http://example.org/alix>", r#""Alix""#]]
    );
    assert_eq!(
        texts(
            &db,
            "SELECT ?name WHERE { ?p <http://example.org/age> ?age ; <http://example.org/name> ?name \
             FILTER(?age >= ?min) }",
            &[("min", Value::Int64(30))]
        ),
        [[r#""Alix""#], [r#""Mia""#]]
    );
    assert_eq!(
        texts(
            &db,
            "SELECT ?p WHERE { ?p <http://example.org/age> ?age }",
            &[("age", Value::Int64(25))]
        ),
        [["<http://example.org/gus>"]],
        "an integer binds the xsd:integer it is"
    );
}

#[test]
fn ask_construct_and_an_update_take_their_parameters() {
    let db = db();
    let session = db.session();
    let ask = |who: &str| {
        session
            .execute_sparql_with_params(
                "ASK { ?a <http://example.org/knows> ?b }",
                params(&[("a", iri(who))]),
            )
            .unwrap()
            .rows()[0][0]
            .clone()
    };
    assert_eq!(ask("http://example.org/alix"), Value::Bool(true));
    assert_eq!(ask("http://example.org/gus"), Value::Bool(false));

    let constructed = session
        .execute_sparql_with_params(
            "CONSTRUCT { ?p <http://example.org/label> ?name } WHERE { ?p <http://example.org/name> ?name }",
            params(&[("p", iri("http://example.org/gus"))]),
        )
        .unwrap();
    assert_eq!(constructed.row_count(), 1);

    session
        .execute_sparql_with_params(
            "DELETE { ?p <http://example.org/name> ?name } WHERE { ?p <http://example.org/name> ?name }",
            params(&[("p", iri("http://example.org/mia"))]),
        )
        .unwrap();
    assert_eq!(
        texts(
            &db,
            "SELECT ?name WHERE { ?p <http://example.org/name> ?name }",
            &[]
        ),
        [[r#""Alix""#], [r#""Gus""#]],
        "only the bound person lost the name"
    );
}

/// `execute_language` and the query processor bind them the same way.
#[test]
fn every_entry_point_binds_them() {
    use grafeo_engine::query::{QueryLanguage, QueryProcessor};

    let db = db();
    let query = "SELECT ?name WHERE { ?person <http://example.org/name> ?name }";
    let bound = params(&[("person", iri("http://example.org/gus"))]);
    let result = db
        .session()
        .execute_language(query, "sparql", Some(bound.clone()))
        .unwrap();
    assert_eq!(result.row_count(), 1, "execute_language");

    let processor = QueryProcessor::with_rdf(db.store().unwrap(), Arc::clone(db.rdf_store()));
    let result = processor
        .process(query, QueryLanguage::Sparql, Some(&bound))
        .unwrap();
    assert_eq!(result.row_count(), 1, "the query processor");
}

/// A GraphQL query over RDF takes the default of a variable it is not
/// given, and a given value over it.
#[cfg(feature = "graphql")]
#[test]
fn graphql_over_rdf_takes_its_defaults() {
    let db = db();
    let session = db.session();
    let query = r#"query($name: String = "Gus") { Person(name: $name) { age } }"#;
    let ages = |result: grafeo_engine::database::QueryResult| -> Vec<String> {
        result
            .rows()
            .iter()
            .map(|row| format!("{:?}", row.last().unwrap()))
            .collect()
    };
    let by_default = session.execute_graphql_rdf(query).unwrap();
    assert_eq!(ages(by_default), ["Int64(25)"], "the default");
    let given = session
        .execute_graphql_rdf_with_params(query, params(&[("name", Value::from("Mia"))]))
        .unwrap();
    assert_eq!(ages(given), ["Int64(41)"], "the given value");
    let defaults_only = session
        .execute_graphql_rdf_with_params(query, HashMap::new())
        .unwrap();
    assert_eq!(ages(defaults_only), ["Int64(25)"], "no value given");
}
