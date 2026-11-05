//! The zero-length step of a property path, SPARQL 1.1 section 18.4: a term and a term are one empty
//! solution when they are the same term, in the graph or not; a variable and a term bind the variable
//! to the term; two variables bind each node of the active graph, the subjects and objects of its
//! triples, whatever their predicate. `?` and `*` both take the step and `+` does not.
//! `zero_length_paths.gtest` runs the same table on every binding.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main sparql_zero_length_paths::
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

const DATA: &str = "INSERT DATA { <urn:a> <urn:q> <urn:b> . <urn:b> <urn:q> <urn:c> . <urn:e> <urn:q> <urn:a> . <urn:d> <urn:p> 1 . GRAPH <urn:g> { <urn:z> <urn:q> <urn:a> . <urn:d> <urn:p> 2 . } }";

/// Name, query, rows, each its cells joined, sorted.
const CASES: &[(&str, &str, &[&str])] = &[
    (
        "one_var_at_object",
        "SELECT ?s WHERE { ?s <urn:q>? <urn:a> }",
        &["urn:a", "urn:e"],
    ),
    (
        "one_var_at_object_not_a_subject",
        "SELECT ?s WHERE { ?s <urn:q>? <urn:b> }",
        &["urn:a", "urn:b"],
    ),
    (
        "one_var_at_object_no_edge",
        "SELECT ?s WHERE { ?s <urn:q>? <urn:d> }",
        &["urn:d"],
    ),
    (
        "one_var_at_object_not_in_the_graph",
        "SELECT ?s WHERE { ?s <urn:q>? <urn:nothing> }",
        &["urn:nothing"],
    ),
    (
        "one_var_at_subject",
        "SELECT ?o WHERE { <urn:a> <urn:q>? ?o }",
        &["urn:a", "urn:b"],
    ),
    (
        "one_var_at_subject_not_in_the_graph",
        "SELECT ?o WHERE { <urn:nothing> <urn:q>? ?o }",
        &["urn:nothing"],
    ),
    (
        "two_terms_same",
        "ASK { <urn:a> <urn:q>? <urn:a> }",
        &["true"],
    ),
    (
        "two_terms_same_no_edge_at_all",
        "ASK { <urn:d> <urn:q>? <urn:d> }",
        &["true"],
    ),
    (
        "two_terms_same_not_in_the_graph",
        "ASK { <urn:nothing> <urn:q>? <urn:nothing> }",
        &["true"],
    ),
    (
        "two_terms_edge",
        "ASK { <urn:a> <urn:q>? <urn:b> }",
        &["true"],
    ),
    (
        "two_terms_neither",
        "ASK { <urn:a> <urn:q>? <urn:c> }",
        &["false"],
    ),
    (
        "two_vars",
        "SELECT ?s ?o WHERE { ?s <urn:q>? ?o }",
        &[
            "1|1",
            "urn:a|urn:a",
            "urn:a|urn:b",
            "urn:b|urn:b",
            "urn:b|urn:c",
            "urn:c|urn:c",
            "urn:d|urn:d",
            "urn:e|urn:a",
            "urn:e|urn:e",
        ],
    ),
    (
        "two_vars_same",
        "SELECT ?s WHERE { ?s <urn:q>? ?s }",
        &["1", "urn:a", "urn:b", "urn:c", "urn:d", "urn:e"],
    ),
    (
        "two_vars_in_a_graph",
        "SELECT ?s ?o WHERE { GRAPH <urn:g> { ?s <urn:q>? ?o } }",
        &[
            "2|2",
            "urn:a|urn:a",
            "urn:d|urn:d",
            "urn:z|urn:a",
            "urn:z|urn:z",
        ],
    ),
    (
        "one_var_in_a_graph",
        "SELECT ?s WHERE { GRAPH <urn:g> { ?s <urn:q>? <urn:a> } }",
        &["urn:a", "urn:z"],
    ),
    (
        "two_terms_in_a_graph",
        "ASK { GRAPH <urn:g> { <urn:b> <urn:q>? <urn:b> } }",
        &["true"],
    ),
    (
        "star_one_var_at_object",
        "SELECT ?s WHERE { ?s <urn:q>* <urn:a> }",
        &["urn:a", "urn:e"],
    ),
    (
        "star_one_var_at_subject",
        "SELECT ?o WHERE { <urn:a> <urn:q>* ?o }",
        &["urn:a", "urn:b", "urn:c"],
    ),
    (
        "star_two_terms_same",
        "ASK { <urn:d> <urn:q>* <urn:d> }",
        &["true"],
    ),
    (
        "star_two_terms_not_in_the_graph",
        "ASK { <urn:nothing> <urn:q>* <urn:nothing> }",
        &["true"],
    ),
    (
        "star_two_vars",
        "SELECT ?s ?o WHERE { ?s <urn:q>* ?o }",
        &[
            "1|1",
            "urn:a|urn:a",
            "urn:a|urn:b",
            "urn:a|urn:c",
            "urn:b|urn:b",
            "urn:b|urn:c",
            "urn:c|urn:c",
            "urn:d|urn:d",
            "urn:e|urn:a",
            "urn:e|urn:b",
            "urn:e|urn:c",
            "urn:e|urn:e",
        ],
    ),
    (
        "plus_two_terms_same",
        "ASK { <urn:a> <urn:q>+ <urn:a> }",
        &["false"],
    ),
    (
        "plus_two_vars",
        "SELECT ?s ?o WHERE { ?s <urn:q>+ ?o }",
        &[
            "urn:a|urn:b",
            "urn:a|urn:c",
            "urn:b|urn:c",
            "urn:e|urn:a",
            "urn:e|urn:b",
            "urn:e|urn:c",
        ],
    ),
];

fn cell(value: &Value) -> String {
    match value {
        Value::String(text) => text.as_str().to_string(),
        Value::Null => "null".to_string(),
        Value::Int64(number) => number.to_string(),
        Value::Bool(flag) => flag.to_string(),
        // A term by its plain value: an IRI without its brackets, a literal without its quotes and type.
        Value::RdfTerm(term) => {
            let text = term.to_ntriples();
            match text.strip_prefix('<') {
                Some(iri) => iri.trim_end_matches('>').to_string(),
                None => text.split('"').nth(1).unwrap_or("").to_string(),
            }
        }
        other => format!("{other:?}"),
    }
}

/// Every case runs, so one failure names all that fail with it.
#[test]
fn a_zero_length_step_answers_as_the_standard_says() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute_sparql(DATA).unwrap();
    let mut wrong = Vec::new();
    for (name, query, want) in CASES {
        let result = session
            .execute_sparql(query)
            .unwrap_or_else(|error| panic!("{query}: {error}"));
        let mut found: Vec<String> = result
            .rows()
            .iter()
            .map(|row| row.iter().map(cell).collect::<Vec<_>>().join("|"))
            .collect();
        found.sort();
        if found != *want {
            wrong.push(format!(
                "{name}: {query}\n  found {found:?}\n  want {want:?}"
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} wrong:\n{}",
        wrong.len(),
        CASES.len(),
        wrong.join("\n")
    );
}

/// A statement and its update: the path in an update's `WHERE` reads the same rows.
#[test]
fn an_update_reads_a_zero_length_step_as_a_query_does() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute_sparql(DATA).unwrap();
    session
        .execute_sparql("INSERT { <urn:r> <urn:saw> ?s } WHERE { ?s <urn:q>? <urn:d> }")
        .unwrap();
    let saw = session
        .execute_sparql("SELECT ?s WHERE { <urn:r> <urn:saw> ?s }")
        .unwrap();
    let names: Vec<String> = saw.rows().iter().map(|row| cell(&row[0])).collect();
    assert_eq!(names, ["urn:d"]);
}

/// A graph's nodes are its own: a node of another graph, or one with no triple left, is not one.
#[test]
fn two_variables_read_the_nodes_of_the_graph_at_hand() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute_sparql(DATA).unwrap();
    let nodes = |query: &str| -> Vec<String> {
        let result = session.execute_sparql(query).unwrap();
        let mut found: Vec<String> = result.rows().iter().map(|row| cell(&row[0])).collect();
        found.sort();
        found
    };
    let at_default = "SELECT ?s WHERE { ?s <urn:q>? ?s }";
    let at_named = "SELECT ?s WHERE { GRAPH <urn:g> { ?s <urn:q>? ?s } }";
    assert!(
        !nodes(at_default).contains(&"urn:z".to_string()),
        "urn:z is in the named graph"
    );
    assert!(
        !nodes(at_named).contains(&"urn:b".to_string()),
        "urn:b is in the default graph"
    );
    session
        .execute_sparql("DELETE DATA { <urn:d> <urn:p> 1 }")
        .unwrap();
    assert!(
        !nodes(at_default).contains(&"urn:d".to_string()),
        "no triple holds it now"
    );
    assert!(
        nodes(at_named).contains(&"urn:d".to_string()),
        "the named graph still does"
    );
}

const COMPLEX_DATA: &str = "INSERT DATA { <urn:a> <urn:q> <urn:b> . <urn:b> <urn:q> <urn:c> . <urn:e> <urn:q> <urn:a> . <urn:d> <urn:p> 1 . GRAPH <urn:g> { <urn:z> <urn:q> <urn:a> . <urn:a> <urn:q> <urn:d> } }";

/// `?` over a sequence, an alternative or an inverse, which 0.6.1 refused or read as one predicate.
const COMPLEX: &[(&str, &str, &[&str])] = &[
    (
        "sequence_from_a_term",
        "SELECT ?o WHERE { <urn:a> (<urn:q>/<urn:q>)? ?o }",
        &["urn:a", "urn:c"],
    ),
    (
        "sequence_to_a_term",
        "SELECT ?s WHERE { ?s (<urn:q>/<urn:q>)? <urn:c> }",
        &["urn:a", "urn:c"],
    ),
    (
        "sequence_to_a_term_with_no_path",
        "SELECT ?s WHERE { ?s (<urn:q>/<urn:q>)? <urn:d> }",
        &["urn:d"],
    ),
    (
        "sequence_two_terms_same",
        "ASK { <urn:d> (<urn:q>/<urn:q>)? <urn:d> }",
        &["true"],
    ),
    (
        "sequence_two_terms_a_path",
        "ASK { <urn:a> (<urn:q>/<urn:q>)? <urn:c> }",
        &["true"],
    ),
    (
        "sequence_two_terms_one_step_is_not_enough",
        "ASK { <urn:a> (<urn:q>/<urn:q>)? <urn:b> }",
        &["false"],
    ),
    (
        "sequence_two_variables",
        "SELECT ?s ?o WHERE { ?s (<urn:q>/<urn:q>)? ?o }",
        &[
            "1|1",
            "urn:a|urn:a",
            "urn:a|urn:c",
            "urn:b|urn:b",
            "urn:c|urn:c",
            "urn:d|urn:d",
            "urn:e|urn:b",
            "urn:e|urn:e",
        ],
    ),
    (
        "alternative_from_a_term",
        "SELECT ?o WHERE { <urn:d> (<urn:q>|<urn:p>)? ?o }",
        &["1", "urn:d"],
    ),
    (
        "alternative_to_a_term",
        "SELECT ?s WHERE { ?s (<urn:q>|<urn:p>)? <urn:a> }",
        &["urn:a", "urn:e"],
    ),
    (
        "alternative_two_terms_same",
        "ASK { <urn:c> (<urn:q>|<urn:p>)? <urn:c> }",
        &["true"],
    ),
    (
        "alternative_two_variables",
        "SELECT ?s ?o WHERE { ?s (<urn:q>|<urn:p>)? ?o }",
        &[
            "1|1",
            "urn:a|urn:a",
            "urn:a|urn:b",
            "urn:b|urn:b",
            "urn:b|urn:c",
            "urn:c|urn:c",
            "urn:d|1",
            "urn:d|urn:d",
            "urn:e|urn:a",
            "urn:e|urn:e",
        ],
    ),
    (
        "inverse_from_a_term",
        "SELECT ?o WHERE { <urn:a> ^<urn:q>? ?o }",
        &["urn:a", "urn:e"],
    ),
    (
        "inverse_to_a_term",
        "SELECT ?s WHERE { ?s ^<urn:q>? <urn:a> }",
        &["urn:a", "urn:b"],
    ),
    (
        "inverse_two_terms_same",
        "ASK { <urn:d> ^<urn:q>? <urn:d> }",
        &["true"],
    ),
    (
        "inverse_two_variables",
        "SELECT ?s ?o WHERE { ?s ^<urn:q>? ?o }",
        &[
            "1|1",
            "urn:a|urn:a",
            "urn:a|urn:e",
            "urn:b|urn:a",
            "urn:b|urn:b",
            "urn:c|urn:b",
            "urn:c|urn:c",
            "urn:d|urn:d",
            "urn:e|urn:e",
        ],
    ),
    (
        "sequence_in_a_named_graph",
        "SELECT ?o WHERE { GRAPH <urn:g> { <urn:z> (<urn:q>/<urn:q>)? ?o } }",
        &["urn:d", "urn:z"],
    ),
    (
        "sequence_in_each_graph",
        "SELECT ?g ?o WHERE { GRAPH ?g { <urn:z> (<urn:q>/<urn:q>)? ?o } }",
        &["urn:g|urn:d", "urn:g|urn:z"],
    ),
    (
        "alternative_in_a_named_graph",
        "SELECT ?o WHERE { GRAPH <urn:g> { <urn:a> (<urn:q>|<urn:p>)? ?o } }",
        &["urn:a", "urn:d"],
    ),
    (
        "inverse_in_a_named_graph",
        "SELECT ?o WHERE { GRAPH <urn:g> { <urn:a> ^<urn:q>? ?o } }",
        &["urn:a", "urn:z"],
    ),
    (
        "sequence_after_a_triple",
        "SELECT ?s ?o WHERE { ?s <urn:q> <urn:c> . ?s (<urn:q>/<urn:q>)? ?o }",
        &["urn:b|urn:b"],
    ),
    (
        "nested_in_a_sequence",
        "SELECT ?o WHERE { <urn:e> <urn:q>/(<urn:q>/<urn:q>)? ?o }",
        &["urn:a", "urn:c"],
    ),
];
#[test]
fn a_zero_length_step_of_a_path_that_is_not_one_predicate() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute_sparql(COMPLEX_DATA).unwrap();
    let mut wrong = Vec::new();
    for (name, query, want) in COMPLEX {
        let result = session.execute_sparql(query);
        let found = result.map(|result| {
            let mut rows: Vec<String> = result
                .rows()
                .iter()
                .map(|row| row.iter().map(cell).collect::<Vec<_>>().join("|"))
                .collect();
            rows.sort();
            rows
        });
        match found {
            Ok(rows) if rows == *want => {}
            other => wrong.push(format!(
                "{name}: {query}\n  found {other:?}\n  want {want:?}"
            )),
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} wrong:\n{}",
        wrong.len(),
        COMPLEX.len(),
        wrong.join("\n")
    );
}
