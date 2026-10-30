//! A SPARQL text the parser cannot read is a syntax error, and the process
//! lives. A token that starts no triple in a group (a quoted triple's `<<`,
//! a stray `)` or `,`) made the group parser push an empty pattern forever,
//! until an allocation of tens of gigabytes aborted the process: any app
//! that passes user SPARQL could be killed by a typo. Quoted triples are
//! SPARQL 1.2, not supported yet: their error says so.
//!
//! Each query runs in a child process, so an abort ends the child, which the
//! parent sees, instead of the test run.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test sparql_parser_never_aborts
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use std::process::Command;

use grafeo_common::testing::child_process;
use grafeo_engine::GrafeoDB;

const CHILD_VAR: &str = "GRAFEO_SPARQL_NEVER_ABORTS_CHILD";
/// Exit code of a child whose query failed with a syntax error, after which
/// the database still answered.
const SYNTAX_ERROR: i32 = 19;
/// Exit code of a child whose query was accepted.
const ACCEPTED: i32 = 23;
/// Exit code of a child whose query failed with another kind of error.
const OTHER_ERROR: i32 = 29;

/// The queries, by name: (name, query, words the error must hold).
const QUERIES: &[(&str, &str, &str)] = &[
    // Quoted triples, well formed: SPARQL 1.2.
    (
        "a quoted triple in SELECT",
        "SELECT ?since WHERE { << <http://ex.org/a> <http://ex.org/b> <http://ex.org/c> >> <http://ex.org/since> ?since }",
        "SPARQL 1.2",
    ),
    (
        "a quoted triple with prefixes",
        "PREFIX : <http://ex.org/> SELECT ?since WHERE { << :a :b :c >> :since ?since ; :strength ?s . }",
        "SPARQL 1.2",
    ),
    (
        "a quoted triple in ASK",
        "ASK { << <http://ex.org/a> <http://ex.org/b> <http://ex.org/c> >> ?p ?o }",
        "SPARQL 1.2",
    ),
    (
        "a quoted triple in CONSTRUCT's WHERE",
        "CONSTRUCT { ?s ?p ?o } WHERE { << ?s ?p ?o >> <http://ex.org/since> 2020 }",
        "SPARQL 1.2",
    ),
    (
        "a quoted triple in an OPTIONAL",
        "SELECT ?s WHERE { ?s ?p ?o OPTIONAL { << ?s ?p ?o >> <http://ex.org/since> ?y } }",
        "SPARQL 1.2",
    ),
    (
        "a nested quoted triple",
        "SELECT ?s WHERE { << << ?a ?b ?c >> ?d ?e >> ?p ?s }",
        "SPARQL 1.2",
    ),
    // Quoted triples, cut short.
    (
        "an unclosed quoted triple",
        "SELECT ?s WHERE { <<",
        "SPARQL 1.2",
    ),
    (
        "a quoted triple without its >>",
        "SELECT ?s WHERE { << <http://ex.org/a> <http://ex.org/b> <http://ex.org/c> ?s }",
        "SPARQL 1.2",
    ),
    (
        "a quoted triple and no closing brace",
        "SELECT ?s WHERE { << :a :b :c >>",
        "",
    ),
    // Tokens that start no triple in a group.
    (
        "a stray )",
        "SELECT ?s WHERE { ?s <http://ex.org/p> ?o ) }",
        "unexpected",
    ),
    ("a stray ]", "SELECT ?s WHERE { ] }", "unexpected"),
    ("a stray ,", "SELECT ?s WHERE { , ?s ?p ?o }", "unexpected"),
    ("a lone .", "SELECT ?s WHERE { . }", ""),
    (
        "a ; at the group start",
        "SELECT ?s WHERE { ; ?s ?p ?o }",
        "unexpected",
    ),
    (
        "an unclosed FILTER(",
        "SELECT ?s WHERE { ?s ?p ?o FILTER( }",
        "",
    ),
    (
        "a stray ) in a nested group",
        "SELECT ?s WHERE { { ) } }",
        "unexpected",
    ),
    (
        "a stray ) in a MINUS",
        "SELECT ?s WHERE { ?s ?p ?o MINUS { ) } }",
        "unexpected",
    ),
    // Updates.
    (
        "a quoted triple in INSERT DATA",
        "INSERT DATA { << <http://ex.org/a> <http://ex.org/b> <http://ex.org/c> >> <http://ex.org/since> 2020 }",
        "",
    ),
    (
        "a quoted triple in a DELETE WHERE",
        "DELETE WHERE { << ?s ?p ?o >> ?q ?r }",
        "",
    ),
    (
        "a stray ) in an INSERT's WHERE",
        "INSERT { ?s <http://ex.org/q> 1 } WHERE { ?s ?p ?o ) }",
        "unexpected",
    ),
];

/// The child: runs query `index` and exits with [`SYNTAX_ERROR`] when it fails
/// as a syntax error naming its words (printing the message) and the
/// database still answers. A no-op when run directly.
#[test]
fn sparql_never_aborts_child() {
    let Ok(index) = std::env::var(CHILD_VAR) else {
        return;
    };
    let (_, query, words) = QUERIES[index.parse::<usize>().unwrap()];
    let db = GrafeoDB::new_in_memory();
    match db.execute_sparql(query) {
        Ok(_) => {
            println!("ACCEPTED");
            std::process::exit(ACCEPTED);
        }
        Err(error) => {
            let message = error.to_string();
            println!("ERROR {message}");
            let syntax = message.contains("GRAFEO-Q001") && message.contains(words);
            // The database still answers.
            assert!(
                db.execute_sparql("SELECT ?s WHERE { ?s ?p ?o }").is_ok(),
                "the database no longer answers"
            );
            std::process::exit(if syntax { SYNTAX_ERROR } else { OTHER_ERROR });
        }
    }
}

/// Every query is a syntax error (naming SPARQL 1.2 for a quoted triple,
/// the token for a stray one) with the process alive.
#[test]
fn malformed_and_unsupported_sparql_is_a_syntax_error_never_an_abort() {
    let mut wrong = Vec::new();
    for (index, (name, _, words)) in QUERIES.iter().enumerate() {
        let output = child_process::output(
            Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "sparql_never_aborts_child", "--nocapture"])
                .env(CHILD_VAR, index.to_string()),
        )
        .unwrap();
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        match output.status.code() {
            Some(SYNTAX_ERROR) => {}
            Some(ACCEPTED) => wrong.push(format!("{name}: accepted")),
            Some(OTHER_ERROR) => wrong.push(format!(
                "{name}: not a syntax error naming {words:?}: {}",
                text.lines().find(|l| l.starts_with("ERROR")).unwrap_or("")
            )),
            other => wrong.push(format!(
                "{name}: the process ended ({other:?}): {}",
                text.lines().rev().take(2).collect::<Vec<_>>().join(" / ")
            )),
        }
    }
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}

/// Control: well-formed SPARQL 1.1 next to the same tokens parses and runs.
#[test]
fn well_formed_queries_with_those_tokens_run() {
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql(
        "INSERT DATA { <http://ex.org/alix> <http://ex.org/name> \"Alix\" ; \
         <http://ex.org/knows> <http://ex.org/gus> , <http://ex.org/vincent> . }",
    )
    .unwrap();
    for query in [
        "SELECT ?o WHERE { <http://ex.org/alix> <http://ex.org/knows> ?o . }",
        "SELECT ?o WHERE { <http://ex.org/alix> <http://ex.org/name> ?n ; <http://ex.org/knows> ?o }",
        "SELECT ?s WHERE { ?s ?p ?o FILTER(?p = <http://ex.org/name>) }",
        "SELECT ?s WHERE { ?s <http://ex.org/knows> ?o MINUS { ?s <http://ex.org/name> \"Gus\" } }",
        "SELECT ?x WHERE { BIND((1 + 2) AS ?x) }",
    ] {
        db.execute_sparql(query)
            .unwrap_or_else(|error| panic!("{query}: {error}"));
    }
}
