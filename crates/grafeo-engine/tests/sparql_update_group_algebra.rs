//! SPARQL updates whose WHERE is one of the group forms that read wrong
//! (#701, #702, #703): an update writes or deletes exactly what the query
//! form of its WHERE selects.
//!
//! - A FILTER inside OPTIONAL is the condition of the left join, read over
//!   the joined solution (SPARQL 1.1 Query 18.2.2.6 and 18.5).
//! - A group that opens with MINUS is one solution that binds nothing, and a
//!   group that opens with an OPTIONAL without a match is that solution too
//!   (18.2.2.6: every group starts from the empty pattern).
//! - A BIND to a variable already in scope is a syntax error (10.1, 18.2.1),
//!   so the update fails and changes nothing.
//! - `GRAPH ?g` ranges over the named graphs the transaction sees, its own
//!   uncommitted ones included (18.6).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main sparql_update_group_algebra::
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use grafeo_engine::GrafeoDB;

const DATA: &str = r#"INSERT DATA {
    <http://ex.org/alix> <http://ex.org/name> "Alix" .
    <http://ex.org/alix> <http://ex.org/city> "Amsterdam" .
    <http://ex.org/gus> <http://ex.org/name> "Gus" .
    <http://ex.org/vincent> <http://ex.org/name> "Vincent" .
}"#;

/// A database holding [`DATA`].
fn with_data() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql(DATA).unwrap();
    db
}

/// The triples of the default graph as `[s, p, o]` debug strings, sorted.
fn triples(db: &GrafeoDB) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = db
        .execute_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row.iter().map(|value| format!("{value:?}")).collect())
        .collect();
    rows.sort();
    rows
}

/// The triples `update` adds to and removes from [`DATA`], each as a
/// `[s, p, o]` row, or the update's error.
fn change(update: &str) -> Result<(Vec<Vec<String>>, Vec<Vec<String>>), String> {
    let db = with_data();
    let before = triples(&db);
    db.execute_sparql(update)
        .map_err(|error| error.to_string())?;
    let after = triples(&db);
    let added = after
        .iter()
        .filter(|t| !before.contains(t))
        .cloned()
        .collect();
    let removed = before
        .iter()
        .filter(|t| !after.contains(t))
        .cloned()
        .collect();
    Ok((added, removed))
}

/// The number of triples `update` adds and removes, or its error.
fn counts(update: &str) -> Result<(usize, usize), String> {
    change(update).map(|(added, removed)| (added.len(), removed.len()))
}

/// Runs each `(name, update, added, removed)` case and lists the ones whose
/// counts differ.
fn check(cases: &[(&str, &str, usize, usize)]) {
    let failures: Vec<String> = cases
        .iter()
        .filter_map(|(name, update, added, removed)| match counts(update) {
            Ok(got) if got == (*added, *removed) => None,
            Ok((got_added, got_removed)) => Some(format!(
                "{name}: added {got_added} and removed {got_removed}, \
                 want {added} and {removed}"
            )),
            Err(error) => Some(format!("{name}: failed: {error}")),
        })
        .collect();
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

// =========================================================================
// #701: FILTER inside OPTIONAL in an update's WHERE
// =========================================================================

/// The OPTIONAL extends no row: ?n is bound in the joined solution, so
/// `!BOUND(?n)` is false. INSERT writes no triple from ?c, DELETE removes
/// nothing.
#[test]
fn a_filter_on_an_outer_variable_in_an_update_extends_no_row() {
    let optional = "?s <http://ex.org/name> ?n \
                    OPTIONAL { ?s <http://ex.org/city> ?c FILTER(!BOUND(?n)) }";
    check(&[
        (
            "INSERT from the optional variable",
            &format!(
                "INSERT {{ <http://ex.org/log> <http://ex.org/saw> ?c }} WHERE {{ {optional} }}"
            ),
            0,
            0,
        ),
        (
            "DELETE of the optional triple",
            &format!("DELETE {{ ?s <http://ex.org/city> ?c }} WHERE {{ {optional} }}"),
            0,
            0,
        ),
    ]);
}

/// The condition compares an outer and an inner variable and holds in the
/// joined solution: the extension is written and deleted.
#[test]
fn a_comparison_of_an_outer_and_an_inner_variable_in_an_update_holds() {
    let optional = "?s <http://ex.org/name> ?n \
                    OPTIONAL { ?s <http://ex.org/city> ?c FILTER(?c > ?n) }";
    check(&[
        (
            "INSERT from the optional variable",
            &format!(
                "INSERT {{ <http://ex.org/log> <http://ex.org/saw> ?c }} WHERE {{ {optional} }}"
            ),
            1,
            0,
        ),
        (
            "DELETE of the optional triple",
            &format!("DELETE {{ ?s <http://ex.org/city> ?c }} WHERE {{ {optional} }}"),
            0,
            1,
        ),
    ]);
}

/// Control: a condition on the optional pattern's own variable.
#[test]
fn a_filter_on_an_inner_variable_in_an_update_works() {
    check(&[
        (
            "INSERT, the condition holds",
            r#"INSERT { <http://ex.org/log> <http://ex.org/saw> ?c } WHERE {
                ?s <http://ex.org/name> ?n
                OPTIONAL { ?s <http://ex.org/city> ?c FILTER(?c = "Amsterdam") } }"#,
            1,
            0,
        ),
        (
            "DELETE, the condition fails",
            r#"DELETE { ?s <http://ex.org/city> ?c } WHERE {
                ?s <http://ex.org/name> ?n
                OPTIONAL { ?s <http://ex.org/city> ?c FILTER(?c = "Berlin") } }"#,
            0,
            0,
        ),
    ]);
}

// =========================================================================
// #702: a WHERE that opens with MINUS or OPTIONAL
// =========================================================================

/// `{ MINUS { P } }` is one solution that binds nothing: a template with
/// variables yields no triple, one of constants yields its triple once
/// (SPARQL 1.1 Update 3.1.3).
#[test]
fn a_where_of_only_a_minus_is_one_empty_solution() {
    check(&[
        (
            "DELETE of the MINUS pattern's triples",
            "DELETE { ?s <http://ex.org/name> ?n } WHERE { MINUS { ?s <http://ex.org/name> ?n } }",
            0,
            0,
        ),
        (
            "INSERT of a triple of constants",
            r#"INSERT { <http://ex.org/log> <http://ex.org/note> "once" }
               WHERE { MINUS { ?s <http://ex.org/name> ?n } }"#,
            1,
            0,
        ),
        (
            "INSERT from the MINUS pattern's variables",
            "INSERT { ?s <http://ex.org/seen> ?n } WHERE { MINUS { ?s <http://ex.org/name> ?n } }",
            0,
            0,
        ),
    ]);
}

/// `{ OPTIONAL { P } }` without a match is one solution that binds nothing;
/// with matches it is P's solutions.
#[test]
fn a_where_that_opens_with_an_optional_is_a_left_join_on_the_empty_solution() {
    check(&[
        (
            "INSERT of constants, no match",
            r#"INSERT { <http://ex.org/log> <http://ex.org/note> "once" }
               WHERE { OPTIONAL { ?s <http://ex.org/none> ?o } }"#,
            1,
            0,
        ),
        (
            "INSERT after an OPTIONAL without a match and a pattern",
            r#"INSERT { ?x <http://ex.org/seen> true }
               WHERE { OPTIONAL { ?s <http://ex.org/none> ?o } ?x <http://ex.org/name> "Gus" }"#,
            1,
            0,
        ),
        (
            "INSERT per match",
            "INSERT { ?s <http://ex.org/seen> true } WHERE { OPTIONAL { ?s <http://ex.org/name> ?n } }",
            3,
            0,
        ),
    ]);
}

// =========================================================================
// #703: BIND to a variable in scope
// =========================================================================

/// An update whose WHERE binds a variable in scope fails with a syntax error
/// and changes nothing.
#[test]
fn an_update_whose_where_rebinds_a_variable_fails_and_changes_nothing() {
    let failures: Vec<String> = [
        (
            "INSERT",
            "INSERT { <http://ex.org/log> <http://ex.org/saw> ?n } WHERE { \
             ?s <http://ex.org/name> ?n BIND(\"Mia\" AS ?n) }",
        ),
        (
            "DELETE",
            "DELETE { ?s <http://ex.org/name> ?n } WHERE { \
             ?s <http://ex.org/city> ?n BIND(\"Alix\" AS ?n) }",
        ),
        (
            "DELETE and INSERT",
            "DELETE { ?s <http://ex.org/name> ?n } INSERT { ?s <http://ex.org/name> \"Mia\" } \
             WHERE { ?s <http://ex.org/name> ?n BIND(\"Gus\" AS ?n) }",
        ),
    ]
    .into_iter()
    .filter_map(|(name, update)| {
        let db = with_data();
        let before = triples(&db);
        match db.execute_sparql(update) {
            Ok(_) => Some(format!(
                "{name}: accepted, the triples went from {before:?} to {:?}",
                triples(&db)
            )),
            Err(error) => {
                let message = error.to_string();
                let changed = triples(&db) != before;
                (!message.contains("?n") || changed)
                    .then(|| format!("{name}: {message} (changed: {changed})"))
            }
        }
    })
    .collect();
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// Control: the same updates with a fresh BIND target run.
#[test]
fn an_update_whose_where_binds_a_fresh_variable_runs() {
    check(&[(
        "INSERT from a fresh BIND",
        "INSERT { ?s <http://ex.org/nick> ?k } WHERE { \
         ?s <http://ex.org/name> ?n BIND(CONCAT(?n, \"!\") AS ?k) }",
        3,
        0,
    )]);
}

// =========================================================================
// GRAPH ?g over the named graphs a transaction sees
// =========================================================================

/// The named graphs `GRAPH ?g { }` lists in `session`, one row each, as
/// their IRIs, sorted.
fn graphs(session: &grafeo_engine::Session) -> Vec<String> {
    let mut graphs: Vec<String> = session
        .execute_sparql("SELECT ?g WHERE { GRAPH ?g { } }")
        .unwrap()
        .rows()
        .iter()
        .map(|row| format!("{:?}", row[0]))
        .collect();
    graphs.sort();
    graphs
}

/// A graph a transaction created by `INSERT DATA` is a named graph of its
/// own reads before the commit, and of no other session's; a rollback
/// leaves it out, a commit gives it to everyone.
#[test]
fn a_graph_variable_lists_the_graphs_a_transaction_wrote() {
    const NEW_GRAPH: &str = r#"INSERT DATA {
        GRAPH <http://ex.org/trips> { <http://ex.org/alix> <http://ex.org/city> "Prague" . }
    }"#;
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql(
        r#"INSERT DATA {
            GRAPH <http://ex.org/people> { <http://ex.org/alix> <http://ex.org/name> "Alix" . }
        }"#,
    )
    .unwrap();
    let people = vec!["RdfTerm(<http://ex.org/people>)".to_string()];
    let both = vec![
        "RdfTerm(<http://ex.org/people>)".to_string(),
        "RdfTerm(<http://ex.org/trips>)".to_string(),
    ];
    let other = db.session();

    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer.execute_sparql(NEW_GRAPH).unwrap();
    assert_eq!(graphs(&writer), both, "the writer sees its new graph");
    assert_eq!(graphs(&other), people, "no other session sees it yet");
    writer.rollback().unwrap();
    assert_eq!(graphs(&writer), people, "a rollback leaves it out");

    writer.begin_transaction().unwrap();
    writer.execute_sparql(NEW_GRAPH).unwrap();
    assert_eq!(graphs(&writer), both);
    writer.commit().unwrap();
    assert_eq!(graphs(&other), both, "after the commit everyone sees it");
}

// =========================================================================
// DELETE WHERE: the pattern is the WHERE and the template
// =========================================================================

/// An IRI of `http://ex.org/` from its debug form, as its local name.
fn local_name(debug: &str) -> String {
    debug
        .trim_start_matches("RdfTerm(<http://ex.org/")
        .trim_end_matches(">)")
        .to_string()
}

/// A literal's text from its debug form, or an IRI's local name.
fn term_text(debug: &str) -> String {
    match debug
        .strip_prefix("RdfTerm(\"")
        .and_then(|rest| rest.strip_suffix("\")"))
    {
        Some(text) => text.to_string(),
        None => local_name(debug),
    }
}

/// Each triple `update` removes from [`DATA`], as `"s p o"` (local names and
/// literal text), sorted, so a wrong object binding shows. The update adds
/// nothing.
fn removed(update: &str) -> Vec<String> {
    let (added, removed) = change(update).unwrap_or_else(|error| panic!("{update}: {error}"));
    assert!(added.is_empty(), "{update} added {added:?}");
    let mut triples: Vec<String> = removed
        .iter()
        .map(|row| {
            format!(
                "{} {} {}",
                local_name(&row[0]),
                local_name(&row[1]),
                term_text(&row[2])
            )
        })
        .collect();
    triples.sort();
    triples
}

/// A `DELETE WHERE` whose pattern has no solution deletes nothing, its
/// triples without variables included (SPARQL 1.1 Update 3.1.3.3).
#[test]
fn a_delete_where_without_a_solution_deletes_nothing() {
    for update in [
        r#"DELETE WHERE { <http://ex.org/alix> <http://ex.org/name> "Alix" .
                          <http://ex.org/alix> <http://ex.org/age> ?a }"#,
        r#"DELETE WHERE { <http://ex.org/alix> <http://ex.org/name> "Alix" .
                          <http://ex.org/alix> <http://ex.org/city> "Paris" }"#,
        r#"DELETE WHERE { <http://ex.org/gus> <http://ex.org/name> ?n .
                          <http://ex.org/alix> <http://ex.org/city> "Paris" }"#,
    ] {
        assert_eq!(removed(update), Vec::<String>::new(), "{update}");
    }
}

/// A `DELETE WHERE` with a solution deletes its triples without variables
/// and the instantiated ones.
#[test]
fn a_delete_where_with_a_solution_deletes_every_triple_of_the_pattern() {
    let alix = ["alix city Amsterdam", "alix name Alix"];
    assert_eq!(
        removed(
            r#"DELETE WHERE { <http://ex.org/alix> <http://ex.org/name> "Alix" .
                              <http://ex.org/alix> <http://ex.org/city> ?c }"#
        ),
        alix
    );
    assert_eq!(
        removed(
            r#"DELETE WHERE { <http://ex.org/alix> <http://ex.org/name> "Alix" .
                              <http://ex.org/alix> <http://ex.org/city> "Amsterdam" }"#
        ),
        alix
    );
    // A join on ?s: both triples of its one solution go, the second one
    // too after the first is deleted.
    assert_eq!(
        removed("DELETE WHERE { ?s <http://ex.org/name> ?n . ?s <http://ex.org/city> ?c }"),
        alix
    );
}

/// With several solutions every instantiated triple goes, and a triple
/// without variables goes once.
#[test]
fn a_delete_where_with_several_solutions_deletes_each_triple() {
    assert_eq!(
        removed("DELETE WHERE { ?s <http://ex.org/name> ?n }"),
        ["alix name Alix", "gus name Gus", "vincent name Vincent"]
    );
    assert_eq!(
        removed(
            r#"DELETE WHERE { <http://ex.org/alix> <http://ex.org/city> "Amsterdam" .
                              ?s <http://ex.org/name> ?n }"#
        ),
        [
            "alix city Amsterdam",
            "alix name Alix",
            "gus name Gus",
            "vincent name Vincent"
        ]
    );
}

/// The triples of graph `<http://ex.org/{graph}>` as `"s p o"` (local names
/// and literal text), sorted.
fn graph_pairs(db: &GrafeoDB, graph: &str) -> Vec<String> {
    let mut pairs: Vec<String> = db
        .execute_sparql(&format!(
            "SELECT ?s ?p ?o WHERE {{ GRAPH <http://ex.org/{graph}> {{ ?s ?p ?o }} }}"
        ))
        .unwrap()
        .rows()
        .iter()
        .map(|row| {
            format!(
                "{} {} {}",
                local_name(&format!("{:?}", row[0])),
                local_name(&format!("{:?}", row[1])),
                term_text(&format!("{:?}", row[2]))
            )
        })
        .collect();
    pairs.sort();
    pairs
}

/// In a named graph the same holds: no solution deletes nothing, a
/// solution deletes the pattern's triples from that graph only.
#[test]
fn a_delete_where_in_a_named_graph_deletes_only_on_a_solution() {
    let db = with_data();
    db.execute_sparql(
        r#"INSERT DATA { GRAPH <http://ex.org/people> {
            <http://ex.org/alix> <http://ex.org/name> "Alix" .
            <http://ex.org/alix> <http://ex.org/city> "Amsterdam" .
            <http://ex.org/gus> <http://ex.org/name> "Gus" } }"#,
    )
    .unwrap();
    let default_before = triples(&db);

    db.execute_sparql(
        r#"DELETE WHERE { GRAPH <http://ex.org/people> {
            <http://ex.org/alix> <http://ex.org/name> "Alix" .
            <http://ex.org/alix> <http://ex.org/age> ?a } }"#,
    )
    .unwrap();
    assert_eq!(
        graph_pairs(&db, "people"),
        ["alix city Amsterdam", "alix name Alix", "gus name Gus"],
        "no solution deletes nothing"
    );

    db.execute_sparql(
        r#"DELETE WHERE { GRAPH <http://ex.org/people> {
            <http://ex.org/alix> <http://ex.org/name> "Alix" .
            <http://ex.org/alix> <http://ex.org/city> ?c } }"#,
    )
    .unwrap();
    assert_eq!(graph_pairs(&db, "people"), ["gus name Gus"]);
    assert_eq!(triples(&db), default_before, "the default graph stays");
}
