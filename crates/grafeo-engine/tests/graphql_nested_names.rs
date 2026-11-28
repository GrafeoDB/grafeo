//! GraphQL over RDF names a nested field's columns as GraphQL over a property graph does:
//! the same query shape over the same people gives the same columns and as many rows, at
//! two levels, with aliases and through a fragment.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main graphql_nested_names::
//! ```

#![cfg(all(feature = "graphql", feature = "triple-store"))]

use grafeo_engine::{GrafeoDB, RdfFormat};

/// Alix knows Gus, who knows Mia, as triples in the example.org vocabulary.
const PEOPLE: &str = "@prefix ex: <http://example.org/> .
ex:alix a ex:Person ; ex:name \"Alix\" ; ex:knows ex:gus .
ex:gus a ex:Person ; ex:name \"Gus\" ; ex:knows ex:mia .
ex:mia a ex:Person ; ex:name \"Mia\" .
";

/// The same people as a property graph.
const PEOPLE_LPG: &str = "INSERT (:Person {name: 'Alix'})-[:knows]->(:Person {name: 'Gus'})\
    -[:knows]->(:Person {name: 'Mia'})";

#[test]
fn nested_columns_are_named_over_rdf_as_over_a_property_graph() {
    let rdf = GrafeoDB::new_in_memory();
    rdf.load_rdf_text(PEOPLE, RdfFormat::Turtle, None).unwrap();
    let lpg = GrafeoDB::new_in_memory();
    lpg.session().execute(PEOPLE_LPG).unwrap();
    for (over_rdf, over_lpg) in [
        (
            "{ person { name knows { name knows { name } } } }",
            "{ Person { name knows { name knows { name } } } }",
        ),
        (
            "{ person { name pal: knows { n: name } } }",
            "{ Person { name pal: knows { n: name } } }",
        ),
        (
            "{ person { knows { ...F } } } fragment F on Person { name }",
            "{ Person { knows { ...F } } } fragment F on Person { name }",
        ),
    ] {
        let from_rdf = rdf
            .session()
            .execute_graphql_rdf(over_rdf)
            .unwrap_or_else(|error| panic!("`{over_rdf}`: {error}"));
        let from_lpg = lpg
            .session()
            .execute_graphql(over_lpg)
            .unwrap_or_else(|error| panic!("`{over_lpg}`: {error}"));
        assert_eq!(from_rdf.columns, from_lpg.columns, "`{over_rdf}`");
        assert_eq!(from_rdf.rows().len(), from_lpg.rows().len(), "`{over_rdf}`");
    }
}
