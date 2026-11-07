//! `?` over one predicate reads one list and keeps no set of what it has seen, so these hold that
//! a term still comes once: a self-loop and the zero-length step both give it, in a graph, per row,
//! inside a transaction and over a merge of graphs. Two bound ends answer from one lookup, with a
//! term the graph holds or not. `one_step_paths.gtest` runs the same table on every binding.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main sparql_one_step_paths::
//! ```

use std::fmt::Write;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

const DATA: &str = "INSERT DATA { <urn:l> <urn:q> <urn:l> . <urn:l> <urn:q> <urn:m> . <urn:l> <urn:q> <urn:n> . <urn:x> <urn:q> <urn:l> . <urn:s> <urn:p> 1 . <urn:s> <urn:p> 2 . <urn:l> <urn:t> <urn:k> . <urn:m> <urn:t> <urn:k> . <urn:y> <urn:t> <urn:k> . GRAPH <urn:g> { <urn:l> <urn:q> <urn:l> . <urn:l> <urn:q> <urn:k> } GRAPH <urn:h> { <urn:l> <urn:q> <urn:k> . <urn:y> <urn:q> <urn:y> } }";

/// Name, query, rows, each its cells joined, sorted.
const CASES: &[(&str, &str, &[&str])] = &[
    (
        "loop_from_a_term",
        "SELECT ?o WHERE { <urn:l> <urn:q>? ?o }",
        &["urn:l", "urn:m", "urn:n"],
    ),
    (
        "loop_to_a_term",
        "SELECT ?s WHERE { ?s <urn:q>? <urn:l> }",
        &["urn:l", "urn:x"],
    ),
    (
        "loop_inverse_from_a_term",
        "SELECT ?o WHERE { <urn:l> ^<urn:q>? ?o }",
        &["urn:l", "urn:x"],
    ),
    (
        "loop_inverse_to_a_term",
        "SELECT ?s WHERE { ?s ^<urn:q>? <urn:l> }",
        &["urn:l", "urn:m", "urn:n"],
    ),
    (
        "loop_star_from_a_term",
        "SELECT ?o WHERE { <urn:l> <urn:q>* ?o }",
        &["urn:l", "urn:m", "urn:n"],
    ),
    (
        "loop_plus_from_a_term",
        "SELECT ?o WHERE { <urn:l> <urn:q>+ ?o }",
        &["urn:l", "urn:m", "urn:n"],
    ),
    (
        "no_loop_from_a_term",
        "SELECT ?o WHERE { <urn:x> <urn:q>? ?o }",
        &["urn:l", "urn:x"],
    ),
    (
        "no_edge_from_a_term",
        "SELECT ?o WHERE { <urn:m> <urn:q>? ?o }",
        &["urn:m"],
    ),
    (
        "literals_from_a_term",
        "SELECT ?o WHERE { <urn:s> <urn:p>? ?o }",
        &["1", "2", "urn:s"],
    ),
    (
        "no_predicate_from_a_term",
        "SELECT ?o WHERE { <urn:l> <urn:nopredicate>? ?o }",
        &["urn:l"],
    ),
    (
        "no_predicate_to_a_term",
        "SELECT ?s WHERE { ?s <urn:nopredicate>? <urn:l> }",
        &["urn:l"],
    ),
    (
        "not_in_the_graph_from_a_term",
        "SELECT ?o WHERE { <urn:none> <urn:q>? ?o }",
        &["urn:none"],
    ),
    (
        "not_in_the_graph_to_a_term",
        "SELECT ?s WHERE { ?s <urn:q>? <urn:none> }",
        &["urn:none"],
    ),
    (
        "loop_in_a_graph",
        "SELECT ?o WHERE { GRAPH <urn:g> { <urn:l> <urn:q>? ?o } }",
        &["urn:k", "urn:l"],
    ),
    (
        "each_graph",
        "SELECT ?g ?o WHERE { GRAPH ?g { <urn:l> <urn:q>? ?o } }",
        &["urn:g|urn:k", "urn:g|urn:l", "urn:h|urn:k", "urn:h|urn:l"],
    ),
    (
        "each_graph_to_a_term",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s <urn:q>? <urn:y> } }",
        &["urn:g|urn:y", "urn:h|urn:y"],
    ),
    (
        "a_merge_of_two_graphs",
        "SELECT ?o FROM <urn:g> FROM <urn:h> WHERE { <urn:l> <urn:q>? ?o }",
        &["urn:k", "urn:l"],
    ),
    (
        "a_merge_to_a_term",
        "SELECT ?s FROM <urn:g> FROM <urn:h> WHERE { ?s <urn:q>? <urn:k> }",
        &["urn:k", "urn:l"],
    ),
    (
        "a_join_with_a_loop",
        "SELECT ?s ?o WHERE { ?s <urn:t> <urn:k> . ?s <urn:q>? ?o }",
        &[
            "urn:l|urn:l",
            "urn:l|urn:m",
            "urn:l|urn:n",
            "urn:m|urn:m",
            "urn:y|urn:y",
        ],
    ),
    (
        "a_join_to_the_far_end",
        "SELECT ?s ?o WHERE { ?o <urn:t> <urn:k> . ?s <urn:q>? ?o }",
        &[
            "urn:l|urn:l",
            "urn:l|urn:m",
            "urn:m|urn:m",
            "urn:x|urn:l",
            "urn:y|urn:y",
        ],
    ),
    ("ask_loop", "ASK { <urn:l> <urn:q>? <urn:l> }", &["true"]),
    (
        "ask_same_no_loop",
        "ASK { <urn:m> <urn:q>? <urn:m> }",
        &["true"],
    ),
    ("ask_an_edge", "ASK { <urn:l> <urn:q>? <urn:m> }", &["true"]),
    (
        "ask_the_edge_the_other_way",
        "ASK { <urn:m> <urn:q>? <urn:l> }",
        &["false"],
    ),
    (
        "ask_inverse_edge",
        "ASK { <urn:m> ^<urn:q>? <urn:l> }",
        &["true"],
    ),
    (
        "ask_far_end_not_in_the_graph",
        "ASK { <urn:l> <urn:q>? <urn:none> }",
        &["false"],
    ),
    (
        "ask_near_end_not_in_the_graph",
        "ASK { <urn:none> <urn:q>? <urn:l> }",
        &["false"],
    ),
    (
        "ask_both_not_in_the_graph_same",
        "ASK { <urn:none> <urn:q>? <urn:none> }",
        &["true"],
    ),
    (
        "ask_both_not_in_the_graph_differ",
        "ASK { <urn:none> <urn:q>? <urn:none2> }",
        &["false"],
    ),
    (
        "ask_no_predicate_same",
        "ASK { <urn:l> <urn:nopredicate>? <urn:l> }",
        &["true"],
    ),
    (
        "ask_no_predicate_differ",
        "ASK { <urn:l> <urn:nopredicate>? <urn:m> }",
        &["false"],
    ),
    ("ask_a_literal", "ASK { <urn:s> <urn:p>? 1 }", &["true"]),
    (
        "ask_a_literal_not_held",
        "ASK { <urn:s> <urn:p>? 3 }",
        &["false"],
    ),
    (
        "ask_star_far_end_not_in_the_graph",
        "ASK { <urn:l> <urn:q>* <urn:none> }",
        &["false"],
    ),
    (
        "ask_star_near_end_not_in_the_graph",
        "ASK { <urn:none> <urn:q>* <urn:l> }",
        &["false"],
    ),
    (
        "ask_star_both_not_in_the_graph_same",
        "ASK { <urn:none> <urn:q>* <urn:none> }",
        &["true"],
    ),
    (
        "ask_plus_far_end_not_in_the_graph",
        "ASK { <urn:l> <urn:q>+ <urn:none> }",
        &["false"],
    ),
    (
        "ask_plus_near_end_not_in_the_graph",
        "ASK { <urn:none> <urn:q>+ <urn:l> }",
        &["false"],
    ),
    (
        "ask_plus_both_not_in_the_graph_same",
        "ASK { <urn:none> <urn:q>+ <urn:none> }",
        &["false"],
    ),
    (
        "star_to_a_term_not_in_the_graph",
        "SELECT ?x WHERE { ?x <urn:q>* <urn:none> }",
        &["urn:none"],
    ),
    (
        "star_from_a_term_not_in_the_graph",
        "SELECT ?x WHERE { <urn:none> <urn:q>* ?x }",
        &["urn:none"],
    ),
    (
        "plus_to_a_term_not_in_the_graph",
        "SELECT ?x WHERE { ?x <urn:q>+ <urn:none> }",
        &[],
    ),
    (
        "plus_from_a_term_not_in_the_graph",
        "SELECT ?x WHERE { <urn:none> <urn:q>+ ?x }",
        &[],
    ),
    (
        "ask_star_two_steps",
        "ASK { <urn:x> <urn:q>* <urn:m> }",
        &["true"],
    ),
    (
        "ask_star_the_other_way",
        "ASK { <urn:m> <urn:q>* <urn:x> }",
        &["false"],
    ),
    (
        "ask_plus_same_with_a_loop",
        "ASK { <urn:l> <urn:q>+ <urn:l> }",
        &["true"],
    ),
    (
        "ask_plus_same_no_loop",
        "ASK { <urn:m> <urn:q>+ <urn:m> }",
        &["false"],
    ),
    (
        "ask_in_a_graph_held",
        "ASK { GRAPH <urn:g> { <urn:l> <urn:q>? <urn:k> } }",
        &["true"],
    ),
    (
        "ask_in_a_graph_not_held",
        "ASK { GRAPH <urn:g> { <urn:l> <urn:q>? <urn:m> } }",
        &["false"],
    ),
    (
        "ask_in_a_graph_the_store_does_not_hold",
        "ASK { GRAPH <urn:none> { <urn:l> <urn:q>? <urn:l> } }",
        &["false"],
    ),
    (
        "select_two_bound_ends",
        "SELECT * WHERE { <urn:l> <urn:q>? <urn:l> }",
        &[""],
    ),
    (
        "select_two_bound_ends_none",
        "SELECT * WHERE { <urn:l> <urn:q>? <urn:none> }",
        &[],
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

fn rows(session: &Session, query: &str) -> Vec<String> {
    let mut found: Vec<String> = session
        .execute_sparql(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .iter()
        .map(|row| row.iter().map(cell).collect::<Vec<_>>().join("|"))
        .collect();
    found.sort();
    found
}

/// Every case runs, so one failure names all that fail with it.
#[test]
fn a_term_comes_once_and_two_bound_ends_answer() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute_sparql(DATA).unwrap();
    let mut wrong = Vec::new();
    for (name, query, want) in CASES {
        let found = rows(&session, query);
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

/// A triple a transaction inserts again, one it inserts and one it deletes, read inside it: the
/// list of the store and the transaction's own make one answer, and another session sees neither.
#[test]
fn a_transaction_reads_one_list_with_its_own_changes() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.execute_sparql(DATA).unwrap();
    session.begin_transaction().unwrap();
    session
        .execute_sparql("INSERT DATA { <urn:l> <urn:q> <urn:m> . <urn:l> <urn:q> <urn:z> }")
        .unwrap();
    session
        .execute_sparql("DELETE DATA { <urn:l> <urn:q> <urn:n> . <urn:l> <urn:q> <urn:l> }")
        .unwrap();
    let from_a_term = "SELECT ?o WHERE { <urn:l> <urn:q>? ?o }";
    assert_eq!(
        rows(&session, from_a_term),
        ["urn:l", "urn:m", "urn:z"],
        "own"
    );
    assert_eq!(
        rows(&session, "ASK { <urn:l> <urn:q>? <urn:z> }"),
        ["true"],
        "its own insert"
    );
    assert_eq!(
        rows(&session, "ASK { <urn:l> <urn:q>? <urn:n> }"),
        ["false"],
        "its own delete"
    );
    assert_eq!(
        rows(&session, "ASK { <urn:l> <urn:q>? <urn:l> }"),
        ["true"],
        "the zero-length step, with the loop deleted"
    );
    assert_eq!(
        rows(&db.session(), from_a_term),
        ["urn:l", "urn:m", "urn:n"],
        "another session"
    );
    session.rollback().unwrap();
    assert_eq!(
        rows(&session, from_a_term),
        ["urn:l", "urn:m", "urn:n"],
        "rolled back"
    );
}

/// A list that grows past a chunk gives each term once.
#[test]
fn a_long_list_gives_each_term_once() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let mut triples = String::from("<urn:l> <urn:q> <urn:l> . ");
    for at in 0..5_000 {
        write!(
            triples,
            "<urn:l> <urn:q> <urn:n{at}> . <urn:s{at}> <urn:q> <urn:l> . "
        )
        .unwrap();
    }
    session
        .execute_sparql(&format!("INSERT DATA {{ {triples} }}"))
        .unwrap();
    let once = |query: &str, wanted: usize| {
        let found = rows(&session, query);
        assert_eq!(found.len(), wanted, "{query}");
        let mut unique = found.clone();
        unique.dedup();
        assert_eq!(unique.len(), wanted, "{query}");
    };
    once("SELECT ?o WHERE { <urn:l> <urn:q>? ?o }", 5_001);
    once("SELECT ?s WHERE { ?s <urn:q>? <urn:l> }", 5_001);
    once("SELECT ?o WHERE { <urn:l> ^<urn:q>? ?o }", 5_001);
    once("SELECT ?o WHERE { <urn:n7> ^<urn:q>? ?o }", 2);
    // A reader in a transaction reads the path as it reads the plain pattern, at the same moment.
    let mut reader = db.session();
    reader.begin_transaction().unwrap();
    let same_as_the_pattern = |moment: &str| {
        let mut wanted = rows(&reader, "SELECT ?o WHERE { <urn:l> <urn:q> ?o }");
        wanted.push("urn:l".to_string());
        wanted.sort();
        wanted.dedup();
        assert_eq!(
            rows(&reader, "SELECT ?o WHERE { <urn:l> <urn:q>? ?o }"),
            wanted,
            "{moment}"
        );
    };
    same_as_the_pattern("before the writes");
    session
        .execute_sparql("DELETE DATA { <urn:l> <urn:q> <urn:n7> . <urn:l> <urn:q> <urn:l> }")
        .unwrap();
    session
        .execute_sparql("INSERT DATA { <urn:l> <urn:q> <urn:new> }")
        .unwrap();
    same_as_the_pattern("after the writes");
    reader.rollback().unwrap();
    once("SELECT ?o WHERE { <urn:l> <urn:q>? ?o }", 5_001);
}
