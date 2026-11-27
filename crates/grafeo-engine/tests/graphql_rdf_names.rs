//! GraphQL over RDF names types and fields by their local name in any namespace, and
//! `EXPLAIN` returns a plan in GraphQL over LPG and over RDF.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main graphql_rdf_names::
//! ```

#![cfg(all(feature = "graphql", feature = "triple-store"))]

use grafeo_common::types::Value;
use grafeo_engine::{GrafeoDB, RdfFormat};

/// People and a company in schema.org, with a link of the dataset's own vocabulary.
const PEOPLE: &str = r#"
@prefix schema: <https://schema.org/> .
@prefix sv: <https://docs.grafeo.dev/datasets/silicon-valley/vocab#> .
@prefix svr: <https://docs.grafeo.dev/datasets/silicon-valley/id#> .
svr:richard a schema:Person ; schema:name "Richard Hendricks" ; sv:worksAt svr:pied-piper .
svr:gilfoyle a schema:Person ; schema:name "Bertram Gilfoyle" ; sv:worksAt svr:pied-piper .
svr:pied-piper a schema:Organization ; schema:name "Pied Piper" .
"#;

fn db_with(turtle: &str) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.load_rdf_text(turtle, RdfFormat::Turtle, None).unwrap();
    db
}

fn strings(db: &GrafeoDB, query: &str) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = db
        .session()
        .execute_graphql_rdf(query)
        .unwrap_or_else(|error| panic!("`{query}`: {error}"))
        .rows()
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| match value {
                    Value::String(text) => text.to_string(),
                    other => format!("{other:?}"),
                })
                .collect()
        })
        .collect();
    rows.sort();
    rows
}

#[test]
fn a_type_and_its_fields_match_the_data_by_local_name() {
    let db = db_with(PEOPLE);
    assert_eq!(
        strings(&db, "{ person { name } }"),
        vec![
            vec!["Bertram Gilfoyle".to_string()],
            vec!["Richard Hendricks".to_string()]
        ]
    );
    assert_eq!(
        strings(&db, "{ Organization { name } }"),
        vec![vec!["Pied Piper".to_string()]]
    );
}

#[test]
fn a_nested_field_follows_a_link_in_another_namespace() {
    let db = db_with(PEOPLE);
    let rows = strings(&db, "{ person { name worksAt { company: name } } }");
    assert_eq!(rows.len(), 2, "{rows:?}");
    assert!(
        rows.iter()
            .all(|row| row.contains(&"Pied Piper".to_string())),
        "{rows:?}"
    );
}

#[test]
fn a_local_name_in_two_namespaces_is_an_error_naming_both() {
    let db = db_with(&format!(
        "{PEOPLE}\n@prefix foaf: <http://xmlns.com/foaf/0.1/> .\nsvr:richard foaf:name \"Richard\" .\n"
    ));
    let error = db
        .session()
        .execute_graphql_rdf("{ person { name } }")
        .unwrap_err()
        .to_string();
    assert!(error.contains("<https://schema.org/name>"), "{error}");
    assert!(
        error.contains("<http://xmlns.com/foaf/0.1/name>"),
        "{error}"
    );
    assert!(error.contains("SPARQL"), "{error}");
}

#[test]
fn a_name_the_data_does_not_hold_falls_back_to_example_org() {
    let db =
        db_with("@prefix ex: <http://example.org/> .\nex:mia a ex:Person ; ex:name \"Mia\" .\n");
    assert_eq!(
        strings(&db, "{ person { name } }"),
        vec![vec!["Mia".to_string()]]
    );
    // Neither name is in the data: no match, no row, no error.
    assert_eq!(
        strings(&db, "{ robot { serial } }"),
        Vec::<Vec<String>>::new()
    );
}

#[test]
fn a_name_follows_data_added_after_an_earlier_query() {
    let db = GrafeoDB::new_in_memory();
    // Nothing to match yet: the same query again after the data arrives finds it.
    assert_eq!(
        strings(&db, "{ organization { name } }"),
        Vec::<Vec<String>>::new()
    );
    db.load_rdf_text(PEOPLE, RdfFormat::Turtle, None).unwrap();
    assert_eq!(
        strings(&db, "{ organization { name } }"),
        vec![vec!["Pied Piper".to_string()]]
    );
}

fn plan_text(db: &GrafeoDB, language: &str, query: &str) -> String {
    let result = db
        .execute_language(query, language, None)
        .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"));
    assert_eq!(
        result.rows().len(),
        1,
        "{language} EXPLAIN gives one row: {:?}",
        result.rows()
    );
    match &result.rows()[0][0] {
        Value::String(text) => text.to_string(),
        other => panic!("{language} EXPLAIN `{query}` returned {other:?}"),
    }
}

#[test]
fn explain_returns_the_plan_in_graphql_over_lpg_and_over_rdf() {
    let db = db_with(PEOPLE);
    db.session()
        .execute("INSERT (:Person {name: 'Gus'})")
        .unwrap();
    let lpg = plan_text(&db, "graphql", "EXPLAIN { Person { name } }");
    assert!(lpg.contains("Person"), "{lpg}");
    let rdf = plan_text(&db, "graphql-rdf", "EXPLAIN { person { name } }");
    assert!(rdf.contains("https://schema.org/Person"), "{rdf}");
    assert!(rdf.contains("https://schema.org/name"), "{rdf}");
}

#[test]
fn example_org_keeps_its_answer_beside_another_namespace() {
    // 0.6.3 read `name` as http://example.org/name: data that also holds schema:name keeps that answer.
    let db = db_with(&format!(
        "{PEOPLE}\n@prefix ex: <http://example.org/> .\nsvr:richard ex:name \"Richard (ex)\" .\n"
    ));
    assert_eq!(
        strings(&db, "{ person { name } }"),
        vec![vec!["Richard (ex)".to_string()]]
    );
}

#[test]
fn a_class_whose_namespace_holds_no_predicate_falls_back() {
    // urn:robots:Robot is a class, but no predicate lives in urn:robots: (the documented limit).
    let db = db_with(
        "@prefix ex: <http://example.org/> .\n<urn:robots:r1> a <urn:robots:Robot> ; ex:serial \"R-1\" .\n",
    );
    assert_eq!(
        strings(&db, "{ robot { serial } }"),
        Vec::<Vec<String>>::new()
    );
}

#[test]
fn a_predicate_written_in_the_open_transaction_matches_after_commit() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute_sparql(
            "INSERT DATA { <urn:a> a <https://schema.org/Person> ; <https://schema.org/name> \"Ana\" }",
        )
        .unwrap();
    // Not committed: the field matches the committed predicates only, so it falls back.
    let plan = session
        .execute_graphql_rdf("EXPLAIN { person { name } }")
        .unwrap();
    let plan = match &plan.rows()[0][0] {
        Value::String(text) => text.to_string(),
        other => panic!("EXPLAIN returned {other:?}"),
    };
    assert!(plan.contains("http://example.org/name"), "{plan}");
    assert!(!plan.contains("https://schema.org/name"), "{plan}");
    session.commit().unwrap();
    assert_eq!(
        strings(&db, "{ person { name } }"),
        vec![vec!["Ana".to_string()]]
    );
}

#[test]
fn explain_analyze_returns_the_profile_in_graphql_over_lpg_and_over_rdf() {
    let db = db_with(PEOPLE);
    db.session()
        .execute("INSERT (:Person {name: 'Gus'})")
        .unwrap();
    for (language, query) in [
        ("graphql", "EXPLAIN ANALYZE { Person { name } }"),
        ("graphql-rdf", "EXPLAIN ANALYZE { person { name } }"),
    ] {
        let result = db
            .execute_language(query, language, None)
            .unwrap_or_else(|error| panic!("{language}: {error}"));
        assert_eq!(result.columns, ["profile"], "{language}");
        assert!(
            result.profile.is_some(),
            "{language}: the operator tree as data"
        );
    }
}

#[test]
fn a_predicate_deleted_after_the_snapshot_falls_back() {
    let db =
        db_with("<urn:a> a <https://schema.org/Person> ; <https://schema.org/name> \"Ana\" .\n");
    let mut session = db.session();
    session.begin_transaction().unwrap();
    // The transaction's snapshot holds the predicate.
    assert_eq!(
        session
            .execute_graphql_rdf("{ person { name } }")
            .unwrap()
            .rows()
            .len(),
        1
    );
    db.session()
        .execute_sparql("DELETE DATA { <urn:a> <https://schema.org/name> \"Ana\" }")
        .unwrap();
    // Committed predicates only: the deleted one no longer names the field.
    let plan = session
        .execute_graphql_rdf("EXPLAIN { person { name } }")
        .unwrap();
    let plan = match &plan.rows()[0][0] {
        Value::String(text) => text.to_string(),
        other => panic!("EXPLAIN returned {other:?}"),
    };
    assert!(plan.contains("http://example.org/name"), "{plan}");
    assert!(!plan.contains("https://schema.org/name"), "{plan}");
}

#[test]
fn a_nested_column_is_named_after_its_field_as_over_lpg() {
    let db = db_with(PEOPLE);
    for (query, columns) in [
        // The nested `name` repeats its parent's: two columns, no clash.
        (
            "{ person { name worksAt { name } } }",
            ["name", "worksAt_name"],
        ),
        (
            "{ person { name worksAt { company: name } } }",
            ["name", "worksAt_company"],
        ),
        (
            "{ person { name employer: worksAt { name } } }",
            ["name", "employer_name"],
        ),
    ] {
        let result = db
            .session()
            .execute_graphql_rdf(query)
            .unwrap_or_else(|error| panic!("`{query}`: {error}"));
        assert_eq!(result.columns, columns, "`{query}`");
        assert_eq!(result.rows().len(), 2, "`{query}`");
    }
    let mut rows = strings(&db, "{ person { name worksAt { name } } }");
    rows.sort();
    assert_eq!(
        rows,
        [
            ["Bertram Gilfoyle", "Pied Piper"]
                .map(String::from)
                .to_vec(),
            ["Richard Hendricks", "Pied Piper"]
                .map(String::from)
                .to_vec(),
        ]
    );
}
