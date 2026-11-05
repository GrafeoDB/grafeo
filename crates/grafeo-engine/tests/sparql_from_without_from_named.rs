//! `FROM` with no `FROM NAMED`: the dataset has a default graph and no named graphs (SPARQL 1.1 13.2),
//! and a graph `FROM` lists is not named by that (13.2.1). `GRAPH` over a graph that is not in the
//! dataset is empty (18.6). `WITH` is not a dataset clause: it sets the default graph of an update's
//! `WHERE` and leaves the named graphs as they are (Update 3.1.3). `from_without_from_named.gtest` runs
//! the same tables on every binding.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main sparql_from_without_from_named::
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

const DATA: &str = "INSERT DATA { <urn:d> <urn:p> 0 . GRAPH <urn:g1> { <urn:a> <urn:p> 1 . <urn:a> <urn:q> <urn:b> . <urn:b> <urn:q> <urn:c> } GRAPH <urn:g2> { <urn:a> <urn:p> 2 . <urn:x> <urn:q> <urn:y> } }";

fn data() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.session().execute_sparql(DATA).unwrap();
    db
}

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

fn rows_of(session: &Session, query: &str) -> Vec<String> {
    let result = session
        .execute_sparql(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    let mut rows: Vec<String> = result
        .rows()
        .iter()
        .map(|row| row.iter().map(cell).collect::<Vec<_>>().join("|"))
        .collect();
    rows.sort();
    rows
}

fn sorted(rows: &[&str]) -> Vec<String> {
    let mut rows: Vec<String> = rows.iter().map(ToString::to_string).collect();
    rows.sort();
    rows
}

/// Every case of a table runs, so one failure names all that fail with it.
fn each(cases: &[(&str, &str, &[&str])]) {
    let db = data();
    let session = db.session();
    let mut wrong = Vec::new();
    for (name, query, want) in cases {
        let found = rows_of(&session, query);
        if found != sorted(want) {
            wrong.push(format!(
                "{name}: {query}\n  found {found:?}\n  want {want:?}"
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} wrong:\n{}",
        wrong.len(),
        cases.len(),
        wrong.join("\n")
    );
}

const FROM_ALONE_NAMES_NO_GRAPHS: &[(&str, &str, &[&str])] = &[
    (
        "graph_variable",
        "SELECT ?g ?s FROM <urn:g1> WHERE { GRAPH ?g { ?s ?p ?o } }",
        &[],
    ),
    (
        "graph_variable_star",
        "SELECT * FROM <urn:g1> WHERE { GRAPH ?g { ?s ?p ?o } }",
        &[],
    ),
    (
        "graph_the_same_iri",
        "SELECT ?s FROM <urn:g1> WHERE { GRAPH <urn:g1> { ?s ?p ?o } }",
        &[],
    ),
    (
        "graph_another_iri",
        "SELECT ?s FROM <urn:g1> WHERE { GRAPH <urn:g2> { ?s ?p ?o } }",
        &[],
    ),
    (
        "graph_variable_empty_group",
        "SELECT ?g FROM <urn:g1> WHERE { GRAPH ?g { } }",
        &[],
    ),
    (
        "graph_variable_bind",
        "SELECT ?g ?x FROM <urn:g1> WHERE { GRAPH ?g { BIND(1 AS ?x) } }",
        &[],
    ),
    (
        "graph_iri_bind",
        "SELECT ?x FROM <urn:g1> WHERE { GRAPH <urn:g1> { BIND(1 AS ?x) } }",
        &[],
    ),
    (
        "graph_iri_optional",
        "SELECT ?s FROM <urn:g1> WHERE { GRAPH <urn:g1> { OPTIONAL { ?s <urn:p> ?o } } }",
        &[],
    ),
    (
        "two_from",
        "SELECT ?g ?s FROM <urn:g1> FROM <urn:g2> WHERE { GRAPH ?g { ?s ?p ?o } }",
        &[],
    ),
    (
        "two_from_iri",
        "SELECT ?s FROM <urn:g1> FROM <urn:g2> WHERE { GRAPH <urn:g2> { ?s ?p ?o } }",
        &[],
    ),
    (
        "the_same_from_twice",
        "SELECT ?g ?s FROM <urn:g1> FROM <urn:g1> WHERE { GRAPH ?g { ?s ?p ?o } }",
        &[],
    ),
    (
        "from_of_no_graph",
        "SELECT ?g ?s FROM <urn:nograph> WHERE { GRAPH ?g { ?s ?p ?o } }",
        &[],
    ),
    (
        "from_of_no_graph_empty_group",
        "SELECT ?g FROM <urn:nograph> WHERE { GRAPH ?g { } }",
        &[],
    ),
    (
        "two_patterns",
        "SELECT ?a ?c FROM <urn:g1> WHERE { GRAPH ?g { ?a <urn:q> ?b . ?b <urn:q> ?c } }",
        &[],
    ),
    (
        "three_patterns",
        "SELECT ?a ?v FROM <urn:g1> WHERE { GRAPH ?g { ?a <urn:q> ?b . ?b <urn:q> ?c . ?a <urn:p> ?v } }",
        &[],
    ),
    (
        "two_patterns_iri",
        "SELECT ?a ?c FROM <urn:g1> WHERE { GRAPH <urn:g1> { ?a <urn:q> ?b . ?b <urn:q> ?c } }",
        &[],
    ),
    (
        "filter_inside",
        "SELECT ?s FROM <urn:g1> WHERE { GRAPH ?g { ?s <urn:p> ?o FILTER(?o = 1) } }",
        &[],
    ),
    (
        "filter_on_the_variable",
        "SELECT ?s FROM <urn:g1> WHERE { GRAPH ?g { ?s ?p ?o } FILTER(?g = <urn:g1>) }",
        &[],
    ),
    (
        "values_of_the_variable",
        "SELECT ?s FROM <urn:g1> WHERE { VALUES ?g { <urn:g1> } GRAPH ?g { ?s ?p ?o } }",
        &[],
    ),
    (
        "bind_of_the_variable",
        "SELECT ?s FROM <urn:g1> WHERE { BIND(<urn:g1> AS ?g) GRAPH ?g { ?s ?p ?o } }",
        &[],
    ),
    (
        "nested",
        "SELECT ?s FROM <urn:g1> WHERE { GRAPH ?g { GRAPH <urn:g1> { ?s ?p ?o } } }",
        &[],
    ),
    (
        "distinct",
        "SELECT DISTINCT ?g FROM <urn:g1> WHERE { GRAPH ?g { ?s ?p ?o } }",
        &[],
    ),
    (
        "order_and_limit",
        "SELECT ?g FROM <urn:g1> WHERE { GRAPH ?g { ?s ?p ?o } } ORDER BY ?g LIMIT 1",
        &[],
    ),
    (
        "count",
        "SELECT (COUNT(*) AS ?c) FROM <urn:g1> WHERE { GRAPH ?g { ?s ?p ?o } }",
        &["0"],
    ),
    (
        "count_iri",
        "SELECT (COUNT(*) AS ?c) FROM <urn:g1> WHERE { GRAPH <urn:g1> { ?s ?p ?o } }",
        &["0"],
    ),
    (
        "count_by_graph",
        "SELECT ?g (COUNT(*) AS ?c) FROM <urn:g1> WHERE { GRAPH ?g { ?s ?p ?o } } GROUP BY ?g",
        &[],
    ),
    (
        "sub_select_around",
        "SELECT ?g ?s FROM <urn:g1> WHERE { { SELECT ?g ?s WHERE { GRAPH ?g { ?s <urn:p> ?o } } } }",
        &[],
    ),
    (
        "sub_select_inside",
        "SELECT ?g ?c FROM <urn:g1> WHERE { GRAPH ?g { { SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o } } } }",
        &[],
    ),
    (
        "ask_variable",
        "ASK FROM <urn:g1> { GRAPH ?g { ?s ?p ?o } }",
        &["false"],
    ),
    (
        "ask_iri",
        "ASK FROM <urn:g1> { GRAPH <urn:g1> { ?s ?p ?o } }",
        &["false"],
    ),
    (
        "construct",
        "CONSTRUCT { ?s ?p ?o } FROM <urn:g1> WHERE { GRAPH ?g { ?s ?p ?o } }",
        &[],
    ),
    (
        "construct_iri",
        "CONSTRUCT { ?s ?p ?o } FROM <urn:g1> WHERE { GRAPH <urn:g1> { ?s ?p ?o } }",
        &[],
    ),
    (
        "describe",
        "DESCRIBE ?s FROM <urn:g1> WHERE { GRAPH ?g { ?s ?p ?o } }",
        &[],
    ),
];

/// A dataset with `FROM` and no `FROM NAMED` has no named graphs (13.2), and `FROM` does not make its
/// graph a named one (13.2.1), so `GRAPH` finds nothing, whatever its pattern.
#[test]
fn from_alone_names_no_graphs() {
    each(FROM_ALONE_NAMES_NO_GRAPHS);
}

const A_PATH_IN_A_GRAPH_FROM_DOES_NOT_NAME_IS_EMPTY: &[(&str, &str, &[&str])] = &[
    (
        "path_plus_in_variable",
        "SELECT ?g ?o FROM <urn:g1> WHERE { GRAPH ?g { <urn:a> <urn:q>+ ?o } }",
        &[],
    ),
    (
        "path_plus_in_the_same_iri",
        "SELECT ?o FROM <urn:g1> WHERE { GRAPH <urn:g1> { <urn:a> <urn:q>+ ?o } }",
        &[],
    ),
    (
        "path_plus_both_ends_free",
        "SELECT ?g ?s ?o FROM <urn:g1> WHERE { GRAPH ?g { ?s <urn:q>+ ?o } }",
        &[],
    ),
    (
        "path_plus_of_no_graph",
        "SELECT ?g ?o FROM <urn:nograph> WHERE { GRAPH ?g { <urn:a> <urn:q>+ ?o } }",
        &[],
    ),
    (
        "path_star_in_variable",
        "SELECT ?g ?o FROM <urn:g1> WHERE { GRAPH ?g { <urn:a> <urn:q>* ?o } }",
        &[],
    ),
    (
        "path_star_in_the_same_iri",
        "SELECT ?o FROM <urn:g1> WHERE { GRAPH <urn:g1> { <urn:a> <urn:q>* ?o } }",
        &[],
    ),
    (
        "path_star_both_ends_free",
        "SELECT ?g ?s ?o FROM <urn:g1> WHERE { GRAPH ?g { ?s <urn:q>* ?o } }",
        &[],
    ),
    (
        "path_star_of_no_graph",
        "SELECT ?g ?o FROM <urn:nograph> WHERE { GRAPH ?g { <urn:a> <urn:q>* ?o } }",
        &[],
    ),
    (
        "path_optional_in_variable",
        "SELECT ?g ?o FROM <urn:g1> WHERE { GRAPH ?g { <urn:a> <urn:q>? ?o } }",
        &[],
    ),
    (
        "path_optional_in_the_same_iri",
        "SELECT ?o FROM <urn:g1> WHERE { GRAPH <urn:g1> { <urn:a> <urn:q>? ?o } }",
        &[],
    ),
    (
        "path_optional_both_ends_free",
        "SELECT ?g ?s ?o FROM <urn:g1> WHERE { GRAPH ?g { ?s <urn:q>? ?o } }",
        &[],
    ),
    (
        "path_optional_of_no_graph",
        "SELECT ?g ?o FROM <urn:nograph> WHERE { GRAPH ?g { <urn:a> <urn:q>? ?o } }",
        &[],
    ),
    (
        "path_sequence_in_variable",
        "SELECT ?g ?o FROM <urn:g1> WHERE { GRAPH ?g { <urn:a> <urn:q>/<urn:q> ?o } }",
        &[],
    ),
    (
        "path_sequence_in_the_same_iri",
        "SELECT ?o FROM <urn:g1> WHERE { GRAPH <urn:g1> { <urn:a> <urn:q>/<urn:q> ?o } }",
        &[],
    ),
    (
        "path_sequence_both_ends_free",
        "SELECT ?g ?s ?o FROM <urn:g1> WHERE { GRAPH ?g { ?s <urn:q>/<urn:q> ?o } }",
        &[],
    ),
    (
        "path_sequence_of_no_graph",
        "SELECT ?g ?o FROM <urn:nograph> WHERE { GRAPH ?g { <urn:a> <urn:q>/<urn:q> ?o } }",
        &[],
    ),
    (
        "path_alternative_in_variable",
        "SELECT ?g ?o FROM <urn:g1> WHERE { GRAPH ?g { <urn:a> <urn:q>|<urn:p> ?o } }",
        &[],
    ),
    (
        "path_alternative_in_the_same_iri",
        "SELECT ?o FROM <urn:g1> WHERE { GRAPH <urn:g1> { <urn:a> <urn:q>|<urn:p> ?o } }",
        &[],
    ),
    (
        "path_alternative_both_ends_free",
        "SELECT ?g ?s ?o FROM <urn:g1> WHERE { GRAPH ?g { ?s <urn:q>|<urn:p> ?o } }",
        &[],
    ),
    (
        "path_alternative_of_no_graph",
        "SELECT ?g ?o FROM <urn:nograph> WHERE { GRAPH ?g { <urn:a> <urn:q>|<urn:p> ?o } }",
        &[],
    ),
    (
        "path_inverse_in_variable",
        "SELECT ?g ?o FROM <urn:g1> WHERE { GRAPH ?g { <urn:b> ^<urn:q> ?o } }",
        &[],
    ),
    (
        "path_inverse_in_the_same_iri",
        "SELECT ?o FROM <urn:g1> WHERE { GRAPH <urn:g1> { <urn:b> ^<urn:q> ?o } }",
        &[],
    ),
    (
        "path_inverse_both_ends_free",
        "SELECT ?g ?s ?o FROM <urn:g1> WHERE { GRAPH ?g { ?s ^<urn:q> ?o } }",
        &[],
    ),
    (
        "path_inverse_of_no_graph",
        "SELECT ?g ?o FROM <urn:nograph> WHERE { GRAPH ?g { <urn:b> ^<urn:q> ?o } }",
        &[],
    ),
    (
        "path_negated_in_variable",
        "SELECT ?g ?o FROM <urn:g1> WHERE { GRAPH ?g { <urn:a> !<urn:p> ?o } }",
        &[],
    ),
    (
        "path_negated_in_the_same_iri",
        "SELECT ?o FROM <urn:g1> WHERE { GRAPH <urn:g1> { <urn:a> !<urn:p> ?o } }",
        &[],
    ),
    (
        "path_negated_both_ends_free",
        "SELECT ?g ?s ?o FROM <urn:g1> WHERE { GRAPH ?g { ?s !<urn:p> ?o } }",
        &[],
    ),
    (
        "path_negated_of_no_graph",
        "SELECT ?g ?o FROM <urn:nograph> WHERE { GRAPH ?g { <urn:a> !<urn:p> ?o } }",
        &[],
    ),
];

/// A path inside `GRAPH` reads the same dataset as a triple pattern, its zero-length step too: no graph
/// to visit means no row, even for a term the path would match with itself.
#[test]
fn a_path_in_a_graph_from_does_not_name_is_empty() {
    each(A_PATH_IN_A_GRAPH_FROM_DOES_NOT_NAME_IS_EMPTY);
}

const FROM_STILL_MAKES_THE_DEFAULT_GRAPH: &[(&str, &str, &[&str])] = &[
    (
        "default_reads_the_graph",
        "SELECT ?o FROM <urn:g1> WHERE { <urn:a> <urn:p> ?o }",
        &["1"],
    ),
    (
        "default_reads_that_graph_alone",
        "SELECT ?s ?o FROM <urn:g1> WHERE { ?s <urn:p> ?o }",
        &["urn:a|1"],
    ),
    (
        "default_two_from",
        "SELECT ?o FROM <urn:g1> FROM <urn:g2> WHERE { <urn:a> <urn:p> ?o }",
        &["1", "2"],
    ),
    (
        "default_of_no_graph",
        "SELECT ?s FROM <urn:nograph> WHERE { ?s ?p ?o }",
        &[],
    ),
    (
        "ask_default",
        "ASK FROM <urn:g1> { <urn:a> <urn:p> 1 }",
        &["true"],
    ),
    (
        "ask_default_other_graph",
        "ASK FROM <urn:g1> { <urn:a> <urn:p> 2 }",
        &["false"],
    ),
    (
        "default_path_plus",
        "SELECT ?o FROM <urn:g1> WHERE { <urn:a> <urn:q>+ ?o }",
        &["urn:b", "urn:c"],
    ),
    (
        "default_path_star",
        "SELECT ?o FROM <urn:g1> WHERE { <urn:a> <urn:q>* ?o }",
        &["urn:a", "urn:b", "urn:c"],
    ),
    (
        "default_path_optional",
        "SELECT ?o FROM <urn:g1> WHERE { <urn:a> <urn:q>? ?o }",
        &["urn:a", "urn:b"],
    ),
    (
        "default_path_sequence",
        "SELECT ?o FROM <urn:g1> WHERE { <urn:a> <urn:q>/<urn:q> ?o }",
        &["urn:c"],
    ),
    (
        "default_path_alternative",
        "SELECT ?o FROM <urn:g1> WHERE { <urn:a> <urn:q>|<urn:p> ?o }",
        &["urn:b", "1"],
    ),
    (
        "default_path_inverse",
        "SELECT ?o FROM <urn:g1> WHERE { <urn:b> ^<urn:q> ?o }",
        &["urn:a"],
    ),
    (
        "default_path_negated",
        "SELECT ?o FROM <urn:g1> WHERE { <urn:a> !<urn:p> ?o }",
        &["urn:b"],
    ),
];

/// `FROM` keeps its meaning: the default graph is the merge of what it lists, and no other graph.
#[test]
fn from_still_makes_the_default_graph() {
    each(FROM_STILL_MAKES_THE_DEFAULT_GRAPH);
}

const FROM_BESIDE_OTHER_PATTERNS: &[(&str, &str, &[&str])] = &[
    (
        "joined",
        "SELECT ?o ?g FROM <urn:g1> WHERE { <urn:a> <urn:p> ?o . GRAPH ?g { <urn:a> <urn:p> ?x } }",
        &[],
    ),
    (
        "joined_iri",
        "SELECT ?o FROM <urn:g1> WHERE { <urn:a> <urn:p> ?o . GRAPH <urn:g1> { <urn:a> <urn:p> ?x } }",
        &[],
    ),
    (
        "optional",
        "SELECT ?o ?x FROM <urn:g1> WHERE { <urn:a> <urn:p> ?o OPTIONAL { GRAPH ?g { <urn:a> <urn:p> ?x } } }",
        &["1|null"],
    ),
    (
        "optional_iri",
        "SELECT ?o ?x FROM <urn:g1> WHERE { <urn:a> <urn:p> ?o OPTIONAL { GRAPH <urn:g1> { <urn:a> <urn:p> ?x } } }",
        &["1|null"],
    ),
    (
        "union",
        "SELECT ?o FROM <urn:g1> WHERE { { <urn:a> <urn:p> ?o } UNION { GRAPH ?g { <urn:a> <urn:p> ?o } } }",
        &["1"],
    ),
    (
        "minus",
        "SELECT ?o FROM <urn:g1> WHERE { <urn:a> <urn:p> ?o MINUS { GRAPH ?g { <urn:a> <urn:p> ?o } } }",
        &["1"],
    ),
    (
        "minus_iri",
        "SELECT ?o FROM <urn:g1> WHERE { <urn:a> <urn:p> ?o MINUS { GRAPH <urn:g1> { <urn:a> <urn:p> ?o } } }",
        &["1"],
    ),
    (
        "exists",
        "SELECT ?o FROM <urn:g1> WHERE { <urn:a> <urn:p> ?o FILTER EXISTS { GRAPH ?g { ?s ?p ?y } } }",
        &[],
    ),
    (
        "exists_iri",
        "SELECT ?o FROM <urn:g1> WHERE { <urn:a> <urn:p> ?o FILTER EXISTS { GRAPH <urn:g1> { ?s ?p ?y } } }",
        &[],
    ),
    (
        "not_exists",
        "SELECT ?o FROM <urn:g1> WHERE { <urn:a> <urn:p> ?o FILTER NOT EXISTS { GRAPH ?g { ?s ?p ?y } } }",
        &["1"],
    ),
    (
        "not_exists_iri",
        "SELECT ?o FROM <urn:g1> WHERE { <urn:a> <urn:p> ?o FILTER NOT EXISTS { GRAPH <urn:g1> { ?s ?p ?y } } }",
        &["1"],
    ),
    (
        "exists_default",
        "SELECT ?o FROM <urn:g1> WHERE { <urn:a> <urn:p> ?o FILTER EXISTS { <urn:b> <urn:q> ?y } }",
        &["1"],
    ),
];

/// A `GRAPH` with no graph to visit is empty inside whatever holds it, and what holds it keeps its own
/// meaning over the default graph.
#[test]
fn from_beside_other_patterns() {
    each(FROM_BESIDE_OTHER_PATTERNS);
}

const FROM_NAMED_STILL_CHOOSES: &[(&str, &str, &[&str])] = &[
    (
        "both_variable",
        "SELECT ?g ?s FROM <urn:g1> FROM NAMED <urn:g2> WHERE { GRAPH ?g { ?s ?p ?o } }",
        &["urn:g2|urn:a", "urn:g2|urn:x"],
    ),
    (
        "both_empty_group",
        "SELECT ?g FROM <urn:g1> FROM NAMED <urn:g2> WHERE { GRAPH ?g { } }",
        &["urn:g2"],
    ),
    (
        "both_listed_iri",
        "SELECT ?s FROM <urn:g1> FROM NAMED <urn:g2> WHERE { GRAPH <urn:g2> { ?s <urn:p> ?o } }",
        &["urn:a"],
    ),
    (
        "both_unlisted_iri",
        "SELECT ?s FROM <urn:g1> FROM NAMED <urn:g2> WHERE { GRAPH <urn:g1> { ?s <urn:p> ?o } }",
        &[],
    ),
    (
        "both_default",
        "SELECT ?o FROM <urn:g1> FROM NAMED <urn:g2> WHERE { <urn:a> <urn:p> ?o }",
        &["1"],
    ),
    (
        "both_join",
        "SELECT ?v ?g ?o FROM <urn:g1> FROM NAMED <urn:g2> WHERE { <urn:a> <urn:p> ?v . GRAPH ?g { <urn:a> <urn:p> ?o } }",
        &["1|urn:g2|2"],
    ),
    (
        "both_count",
        "SELECT (COUNT(*) AS ?c) FROM <urn:g1> FROM NAMED <urn:g2> WHERE { GRAPH ?g { ?s ?p ?o } }",
        &["2"],
    ),
    (
        "both_ask_unlisted",
        "ASK FROM <urn:g1> FROM NAMED <urn:g2> { GRAPH <urn:g1> { ?s ?p ?o } }",
        &["false"],
    ),
    (
        "the_same_graph_in_both",
        "SELECT ?g ?v ?o FROM <urn:g1> FROM NAMED <urn:g1> WHERE { <urn:a> <urn:p> ?v . GRAPH ?g { <urn:a> <urn:p> ?o } }",
        &["urn:g1|1|1"],
    ),
    (
        "the_same_graph_in_both_iri",
        "SELECT ?s FROM <urn:g1> FROM NAMED <urn:g1> WHERE { GRAPH <urn:g1> { ?s <urn:p> ?o } }",
        &["urn:a"],
    ),
    (
        "named_only_variable",
        "SELECT ?g ?s FROM NAMED <urn:g2> WHERE { GRAPH ?g { ?s <urn:p> ?o } }",
        &["urn:g2|urn:a"],
    ),
    (
        "named_only_default_is_empty",
        "SELECT ?s FROM NAMED <urn:g2> WHERE { ?s ?p ?o }",
        &[],
    ),
    (
        "named_only_default_path_is_empty",
        "SELECT ?o FROM NAMED <urn:g1> WHERE { <urn:a> <urn:q>+ ?o }",
        &[],
    ),
    (
        "listed_with_no_triples_beside_from",
        "SELECT ?g ?x FROM <urn:g1> FROM NAMED <urn:nograph> WHERE { GRAPH ?g { BIND(1 AS ?x) } }",
        &["urn:nograph|1"],
    ),
    (
        "both_path_plus_variable",
        "SELECT ?g ?o FROM <urn:g1> FROM NAMED <urn:g2> WHERE { GRAPH ?g { <urn:x> <urn:q>+ ?o } }",
        &["urn:g2|urn:y"],
    ),
    (
        "both_path_plus_listed",
        "SELECT ?o FROM <urn:g1> FROM NAMED <urn:g2> WHERE { GRAPH <urn:g2> { <urn:x> <urn:q>+ ?o } }",
        &["urn:y"],
    ),
    (
        "both_path_plus_unlisted",
        "SELECT ?o FROM <urn:g1> FROM NAMED <urn:g2> WHERE { GRAPH <urn:g1> { <urn:x> <urn:q>+ ?o } }",
        &[],
    ),
    (
        "named_only_path_plus",
        "SELECT ?g ?o FROM NAMED <urn:g2> WHERE { GRAPH ?g { <urn:x> <urn:q>+ ?o } }",
        &["urn:g2|urn:y"],
    ),
    (
        "both_path_star_variable",
        "SELECT ?g ?o FROM <urn:g1> FROM NAMED <urn:g2> WHERE { GRAPH ?g { <urn:x> <urn:q>* ?o } }",
        &["urn:g2|urn:x", "urn:g2|urn:y"],
    ),
    (
        "both_path_star_listed",
        "SELECT ?o FROM <urn:g1> FROM NAMED <urn:g2> WHERE { GRAPH <urn:g2> { <urn:x> <urn:q>* ?o } }",
        &["urn:x", "urn:y"],
    ),
    (
        "both_path_star_unlisted",
        "SELECT ?o FROM <urn:g1> FROM NAMED <urn:g2> WHERE { GRAPH <urn:g1> { <urn:x> <urn:q>* ?o } }",
        &[],
    ),
    (
        "named_only_path_star",
        "SELECT ?g ?o FROM NAMED <urn:g2> WHERE { GRAPH ?g { <urn:x> <urn:q>* ?o } }",
        &["urn:g2|urn:x", "urn:g2|urn:y"],
    ),
    (
        "both_path_optional_variable",
        "SELECT ?g ?o FROM <urn:g1> FROM NAMED <urn:g2> WHERE { GRAPH ?g { <urn:x> <urn:q>? ?o } }",
        &["urn:g2|urn:x", "urn:g2|urn:y"],
    ),
    (
        "both_path_optional_listed",
        "SELECT ?o FROM <urn:g1> FROM NAMED <urn:g2> WHERE { GRAPH <urn:g2> { <urn:x> <urn:q>? ?o } }",
        &["urn:x", "urn:y"],
    ),
    (
        "both_path_optional_unlisted",
        "SELECT ?o FROM <urn:g1> FROM NAMED <urn:g2> WHERE { GRAPH <urn:g1> { <urn:x> <urn:q>? ?o } }",
        &[],
    ),
    (
        "named_only_path_optional",
        "SELECT ?g ?o FROM NAMED <urn:g2> WHERE { GRAPH ?g { <urn:x> <urn:q>? ?o } }",
        &["urn:g2|urn:x", "urn:g2|urn:y"],
    ),
    (
        "both_path_sequence_variable",
        "SELECT ?g ?o FROM <urn:g1> FROM NAMED <urn:g2> WHERE { GRAPH ?g { <urn:x> <urn:q>/<urn:q> ?o } }",
        &[],
    ),
    (
        "both_path_sequence_listed",
        "SELECT ?o FROM <urn:g1> FROM NAMED <urn:g2> WHERE { GRAPH <urn:g2> { <urn:x> <urn:q>/<urn:q> ?o } }",
        &[],
    ),
    (
        "both_path_sequence_unlisted",
        "SELECT ?o FROM <urn:g1> FROM NAMED <urn:g2> WHERE { GRAPH <urn:g1> { <urn:x> <urn:q>/<urn:q> ?o } }",
        &[],
    ),
    (
        "named_only_path_sequence",
        "SELECT ?g ?o FROM NAMED <urn:g2> WHERE { GRAPH ?g { <urn:x> <urn:q>/<urn:q> ?o } }",
        &[],
    ),
    (
        "both_path_alternative_variable",
        "SELECT ?g ?o FROM <urn:g1> FROM NAMED <urn:g2> WHERE { GRAPH ?g { <urn:x> <urn:q>|<urn:p> ?o } }",
        &["urn:g2|urn:y"],
    ),
    (
        "both_path_alternative_listed",
        "SELECT ?o FROM <urn:g1> FROM NAMED <urn:g2> WHERE { GRAPH <urn:g2> { <urn:x> <urn:q>|<urn:p> ?o } }",
        &["urn:y"],
    ),
    (
        "both_path_alternative_unlisted",
        "SELECT ?o FROM <urn:g1> FROM NAMED <urn:g2> WHERE { GRAPH <urn:g1> { <urn:x> <urn:q>|<urn:p> ?o } }",
        &[],
    ),
    (
        "named_only_path_alternative",
        "SELECT ?g ?o FROM NAMED <urn:g2> WHERE { GRAPH ?g { <urn:x> <urn:q>|<urn:p> ?o } }",
        &["urn:g2|urn:y"],
    ),
    (
        "both_path_inverse_variable",
        "SELECT ?g ?o FROM <urn:g1> FROM NAMED <urn:g2> WHERE { GRAPH ?g { <urn:y> ^<urn:q> ?o } }",
        &["urn:g2|urn:x"],
    ),
    (
        "both_path_inverse_listed",
        "SELECT ?o FROM <urn:g1> FROM NAMED <urn:g2> WHERE { GRAPH <urn:g2> { <urn:y> ^<urn:q> ?o } }",
        &["urn:x"],
    ),
    (
        "both_path_inverse_unlisted",
        "SELECT ?o FROM <urn:g1> FROM NAMED <urn:g2> WHERE { GRAPH <urn:g1> { <urn:y> ^<urn:q> ?o } }",
        &[],
    ),
    (
        "named_only_path_inverse",
        "SELECT ?g ?o FROM NAMED <urn:g2> WHERE { GRAPH ?g { <urn:y> ^<urn:q> ?o } }",
        &["urn:g2|urn:x"],
    ),
    (
        "both_path_negated_variable",
        "SELECT ?g ?o FROM <urn:g1> FROM NAMED <urn:g2> WHERE { GRAPH ?g { <urn:x> !<urn:p> ?o } }",
        &["urn:g2|urn:y"],
    ),
    (
        "both_path_negated_listed",
        "SELECT ?o FROM <urn:g1> FROM NAMED <urn:g2> WHERE { GRAPH <urn:g2> { <urn:x> !<urn:p> ?o } }",
        &["urn:y"],
    ),
    (
        "both_path_negated_unlisted",
        "SELECT ?o FROM <urn:g1> FROM NAMED <urn:g2> WHERE { GRAPH <urn:g1> { <urn:x> !<urn:p> ?o } }",
        &[],
    ),
    (
        "named_only_path_negated",
        "SELECT ?g ?o FROM NAMED <urn:g2> WHERE { GRAPH ?g { <urn:x> !<urn:p> ?o } }",
        &["urn:g2|urn:y"],
    ),
];

/// `FROM NAMED` still lists the named graphs, beside `FROM` or alone, and a graph may be in both.
#[test]
fn from_named_still_chooses() {
    each(FROM_NAMED_STILL_CHOOSES);
}

const NO_DATASET_CLAUSE_SEES_EVERY_NAMED_GRAPH: &[(&str, &str, &[&str])] = &[
    (
        "none_variable",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s ?p ?o } }",
        &[
            "urn:g1|urn:a",
            "urn:g1|urn:a",
            "urn:g1|urn:b",
            "urn:g2|urn:a",
            "urn:g2|urn:x",
        ],
    ),
    (
        "none_empty_group",
        "SELECT ?g WHERE { GRAPH ?g { } }",
        &["urn:g1", "urn:g2"],
    ),
    (
        "none_iri",
        "SELECT ?o WHERE { GRAPH <urn:g1> { <urn:a> <urn:p> ?o } }",
        &["1"],
    ),
    (
        "none_default",
        "SELECT ?s WHERE { ?s <urn:p> ?o }",
        &["urn:d"],
    ),
    ("none_ask", "ASK { GRAPH <urn:g2> { ?s ?p ?o } }", &["true"]),
    (
        "none_path_plus",
        "SELECT ?g ?o WHERE { GRAPH ?g { <urn:a> <urn:q>+ ?o } }",
        &["urn:g1|urn:b", "urn:g1|urn:c"],
    ),
    (
        "none_path_star",
        "SELECT ?g ?o WHERE { GRAPH ?g { <urn:a> <urn:q>* ?o } }",
        &[
            "urn:g1|urn:a",
            "urn:g1|urn:b",
            "urn:g1|urn:c",
            "urn:g2|urn:a",
        ],
    ),
    (
        "none_path_optional",
        "SELECT ?g ?o WHERE { GRAPH ?g { <urn:a> <urn:q>? ?o } }",
        &["urn:g1|urn:a", "urn:g1|urn:b", "urn:g2|urn:a"],
    ),
    (
        "none_path_sequence",
        "SELECT ?g ?o WHERE { GRAPH ?g { <urn:a> <urn:q>/<urn:q> ?o } }",
        &["urn:g1|urn:c"],
    ),
    (
        "none_path_alternative",
        "SELECT ?g ?o WHERE { GRAPH ?g { <urn:a> <urn:q>|<urn:p> ?o } }",
        &["urn:g1|urn:b", "urn:g1|1", "urn:g2|2"],
    ),
    (
        "none_path_inverse",
        "SELECT ?g ?o WHERE { GRAPH ?g { <urn:b> ^<urn:q> ?o } }",
        &["urn:g1|urn:a"],
    ),
    (
        "none_path_negated",
        "SELECT ?g ?o WHERE { GRAPH ?g { <urn:a> !<urn:p> ?o } }",
        &["urn:g1|urn:b"],
    ),
];

/// With no dataset clause the dataset is the store's: this change must not reach it.
#[test]
fn no_dataset_clause_sees_every_named_graph() {
    each(NO_DATASET_CLAUSE_SEES_EVERY_NAMED_GRAPH);
}

/// A `CONSTRUCT` under `FROM` builds from the default graph it makes.
#[test]
fn construct_reads_the_default_graph_from_makes() {
    let db = data();
    let session = db.session();
    assert_eq!(
        rows_of(
            &session,
            "CONSTRUCT { ?s <urn:r> ?o } FROM <urn:g1> WHERE { ?s <urn:p> ?o }"
        )
        .len(),
        1,
        "construct_default"
    );
}

/// Update, then the query that shows what it wrote.
const UPDATES: &[(&str, &str, &str, &[&str])] = &[
    (
        "with_graph_variable",
        "WITH <urn:g1> INSERT { <urn:r> <urn:saw> ?g } WHERE { GRAPH ?g { ?s <urn:p> ?o } }",
        "SELECT ?o WHERE { GRAPH <urn:g1> { <urn:r> <urn:saw> ?o } }",
        &["urn:g1", "urn:g2"],
    ),
    (
        "with_graph_variable_empty_group",
        "WITH <urn:g1> INSERT { <urn:r> <urn:saw> ?g } WHERE { GRAPH ?g { } }",
        "SELECT ?o WHERE { GRAPH <urn:g1> { <urn:r> <urn:saw> ?o } }",
        &["urn:g1", "urn:g2"],
    ),
    (
        "with_another_graph",
        "WITH <urn:g1> INSERT { <urn:r> <urn:saw> ?o } WHERE { GRAPH <urn:g2> { <urn:x> <urn:q> ?o } }",
        "SELECT ?o WHERE { GRAPH <urn:g1> { <urn:r> <urn:saw> ?o } }",
        &["urn:y"],
    ),
    (
        "with_the_same_graph",
        "WITH <urn:g1> INSERT { <urn:r> <urn:saw> ?o } WHERE { GRAPH <urn:g1> { <urn:a> <urn:p> ?o } }",
        "SELECT ?o WHERE { GRAPH <urn:g1> { <urn:r> <urn:saw> ?o } }",
        &["1"],
    ),
    (
        "with_a_path_in_another_graph",
        "WITH <urn:g1> INSERT { <urn:r> <urn:saw> ?o } WHERE { GRAPH <urn:g2> { <urn:x> <urn:q>+ ?o } }",
        "SELECT ?o WHERE { GRAPH <urn:g1> { <urn:r> <urn:saw> ?o } }",
        &["urn:y"],
    ),
    (
        "with_a_path_in_graph_variable",
        "WITH <urn:g1> INSERT { <urn:r> <urn:saw> ?g } WHERE { GRAPH ?g { <urn:a> <urn:p>? ?o } }",
        "SELECT ?o WHERE { GRAPH <urn:g1> { <urn:r> <urn:saw> ?o } }",
        &["urn:g1", "urn:g2"],
    ),
    (
        "with_default_is_the_graph",
        "WITH <urn:g1> INSERT { <urn:r> <urn:saw> ?o } WHERE { <urn:a> <urn:p> ?o }",
        "SELECT ?o WHERE { GRAPH <urn:g1> { <urn:r> <urn:saw> ?o } }",
        &["1"],
    ),
    (
        "with_delete_from_a_named_match",
        "WITH <urn:g1> DELETE { <urn:a> <urn:p> ?o } WHERE { GRAPH <urn:g1> { <urn:a> <urn:p> ?o } }",
        "SELECT ?o WHERE { GRAPH <urn:g1> { <urn:a> <urn:p> ?o } }",
        &[],
    ),
];

/// `WITH` is not a dataset clause: `GRAPH` in its `WHERE` still reads the store's named graphs. An
/// empty list of named graphs under `WITH` must not read as "none".
#[test]
fn with_keeps_the_named_graphs_of_the_store() {
    let mut wrong = Vec::new();
    for (name, update, query, want) in UPDATES {
        let db = data();
        let session = db.session();
        session
            .execute_sparql(update)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let found = rows_of(&session, query);
        if found != sorted(want) {
            wrong.push(format!(
                "{name}: {update}\n  found {found:?}\n  want {want:?}"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// A graph the transaction wrote is in the store's dataset, not in one `FROM` makes, though `FROM` can
/// read it as the default graph.
#[test]
fn a_transactions_own_graph_follows_the_same_rule() {
    let db = data();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute_sparql("INSERT DATA { GRAPH <urn:new> { <urn:n> <urn:p> 9 } }")
        .unwrap();
    let cases: &[(&str, &[&str])] = &[
        (
            "SELECT ?g WHERE { GRAPH ?g { } }",
            &["urn:g1", "urn:g2", "urn:new"],
        ),
        (
            "SELECT ?o WHERE { GRAPH <urn:new> { ?s <urn:p> ?o } }",
            &["9"],
        ),
        ("SELECT ?g FROM <urn:g1> WHERE { GRAPH ?g { } }", &[]),
        (
            "SELECT ?g ?s FROM <urn:g1> WHERE { GRAPH ?g { ?s ?p ?o } }",
            &[],
        ),
        (
            "SELECT ?g ?s FROM <urn:new> WHERE { GRAPH ?g { ?s ?p ?o } }",
            &[],
        ),
        (
            "SELECT ?o FROM <urn:new> WHERE { GRAPH <urn:new> { ?s <urn:p> ?o } }",
            &[],
        ),
        ("SELECT ?o FROM <urn:new> WHERE { ?s <urn:p> ?o }", &["9"]),
        (
            "SELECT ?g ?s FROM <urn:g1> FROM NAMED <urn:new> WHERE { GRAPH ?g { ?s ?p ?o } }",
            &["urn:new|urn:n"],
        ),
    ];
    for (query, want) in cases {
        assert_eq!(rows_of(&session, query), sorted(want), "{query}");
    }
    session.rollback().unwrap();
    assert_eq!(
        rows_of(&session, "SELECT ?g FROM <urn:new> WHERE { GRAPH ?g { } }"),
        Vec::<String>::new()
    );
}

/// An empty graph made with `CREATE GRAPH` is a named graph of the store and of nothing `FROM` makes.
#[test]
fn an_empty_graph_of_the_store_is_not_visited_under_from() {
    let db = data();
    let session = db.session();
    session.execute_sparql("CREATE GRAPH <urn:empty>").unwrap();
    assert_eq!(
        rows_of(&session, "SELECT ?g WHERE { GRAPH ?g { } }"),
        sorted(&["urn:empty", "urn:g1", "urn:g2"])
    );
    assert_eq!(
        rows_of(&session, "SELECT ?g FROM <urn:g1> WHERE { GRAPH ?g { } }"),
        Vec::<String>::new()
    );
    assert_eq!(
        rows_of(
            &session,
            "SELECT ?g FROM <urn:empty> WHERE { GRAPH ?g { } }"
        ),
        Vec::<String>::new()
    );
}

/// The same pattern under each dataset clause, in turn and again: a plan made for one is not used for
/// another, and a repeat reads what the first run read.
#[test]
fn each_clause_keeps_its_own_answer_when_repeated() {
    let db = data();
    let session = db.session();
    let forms: &[(&str, &[&str])] = &[
        (
            "SELECT ?g WHERE { GRAPH ?g { ?s <urn:p> ?o } }",
            &["urn:g1", "urn:g2"],
        ),
        (
            "SELECT ?g FROM <urn:g1> WHERE { GRAPH ?g { ?s <urn:p> ?o } }",
            &[],
        ),
        (
            "SELECT ?g FROM NAMED <urn:g1> WHERE { GRAPH ?g { ?s <urn:p> ?o } }",
            &["urn:g1"],
        ),
        (
            "SELECT ?g FROM <urn:g1> FROM NAMED <urn:g2> WHERE { GRAPH ?g { ?s <urn:p> ?o } }",
            &["urn:g2"],
        ),
    ];
    for _ in 0..3 {
        for (query, want) in forms {
            assert_eq!(rows_of(&session, query), sorted(want), "{query}");
        }
    }
}
