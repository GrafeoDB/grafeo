//! SPARQL alternative paths whose branch is not a plain predicate (0.6.1): an
//! inverse (`knows|^knows`), a sequence (`knows|^knows/knows`) or a repeated
//! one (`knows|^knows+`) was refused. Each branch now matches as the path on
//! its own would, and the alternative is the bag union of the branches
//! (SPARQL 1.1, 18.4), with only the subject and object as columns.
//!
//! The fixture: Vincent knows Alix, Alix knows Gus. Every expected row set is
//! worked out by hand from those two triples, not by running a query.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main sparql_alternative_inverse_paths::
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use grafeo_common::types::Value;
use grafeo_engine::database::QueryResult;
use grafeo_engine::{Config, GrafeoDB, GraphModel};

const PREFIX: &str = "PREFIX e: <http://example.org/>";

fn rdf_db(triples: &str) -> GrafeoDB {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    db.execute_sparql(&format!("{PREFIX} INSERT DATA {{ {triples} }}"))
        .expect("insert the fixture");
    db
}

/// Vincent knows Alix, Alix knows Gus, in the default graph.
fn knows_db() -> GrafeoDB {
    rdf_db("e:vincent e:knows e:alix . e:alix e:knows e:gus .")
}

fn local(value: &Value) -> String {
    use grafeo_common::types::rdf::Term;
    match value {
        Value::RdfTerm(term) => match term.as_ref() {
            Term::Iri(iri) => iri
                .as_str()
                .strip_prefix("http://example.org/")
                .unwrap_or_else(|| panic!("an IRI outside the fixture: {iri:?}"))
                .to_string(),
            other => panic!("expected an IRI, got {other:?}"),
        },
        other => panic!("expected an RDF term, got {other:?}"),
    }
}

fn run(db: &GrafeoDB, query: &str) -> QueryResult {
    db.execute_sparql(&format!("{PREFIX} {query}"))
        .unwrap_or_else(|error| panic!("`{query}`: {error}"))
}

/// The rows as local names, sorted: the exact multiset, duplicates kept.
fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = run(db, query)
        .rows()
        .iter()
        .map(|row| row.iter().map(local).collect())
        .collect();
    rows.sort();
    rows
}

fn expected(rows: &[&[&str]]) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = rows
        .iter()
        .map(|row| row.iter().map(|term| (*term).to_string()).collect())
        .collect();
    rows.sort();
    rows
}

#[test]
fn an_inverse_branch_from_a_bound_subject() {
    let db = knows_db();
    // knows from Alix: Gus; ^knows from Alix (who knows Alix): Vincent.
    assert_eq!(
        rows(&db, "SELECT ?o WHERE { e:alix (e:knows|^e:knows) ?o }"),
        expected(&[&["gus"], &["vincent"]])
    );
    // The inverse branch first gives the same rows.
    assert_eq!(
        rows(&db, "SELECT ?o WHERE { e:alix (^e:knows|e:knows) ?o }"),
        expected(&[&["gus"], &["vincent"]])
    );
}

#[test]
fn an_inverse_branch_towards_a_bound_object() {
    let db = knows_db();
    // knows to Alix: Vincent; ^knows to Alix (whom Alix knows): Gus.
    assert_eq!(
        rows(&db, "SELECT ?s WHERE { ?s (e:knows|^e:knows) e:alix }"),
        expected(&[&["gus"], &["vincent"]])
    );
}

#[test]
fn an_inverse_branch_with_both_ends_free() {
    let db = knows_db();
    assert_eq!(
        rows(&db, "SELECT ?s ?o WHERE { ?s (e:knows|^e:knows) ?o }"),
        expected(&[
            &["alix", "gus"],
            &["vincent", "alix"],
            &["gus", "alix"],
            &["alix", "vincent"],
        ])
    );
    // Both ends bound: a fact in either direction holds, an absent one does not.
    for (subject, object, holds) in [
        ("gus", "alix", true),
        ("alix", "gus", true),
        ("vincent", "gus", false),
    ] {
        let query = format!("ASK {{ e:{subject} (e:knows|^e:knows) e:{object} }}");
        assert_eq!(
            run(&db, &query).rows(),
            [vec![Value::Bool(holds)]],
            "{query}"
        );
    }
}

#[test]
fn a_sequence_branch_keeps_only_the_endpoints() {
    let db = knows_db();
    // knows: (Alix, Gus), (Vincent, Alix). ^knows/knows pairs two people
    // someone knows: Alix knows Gus gives (Gus, Gus), Vincent knows Alix
    // gives (Alix, Alix).
    let want = expected(&[
        &["alix", "gus"],
        &["vincent", "alix"],
        &["gus", "gus"],
        &["alix", "alix"],
    ]);
    assert_eq!(
        rows(
            &db,
            "SELECT ?s ?o WHERE { ?s (e:knows|^e:knows/e:knows) ?o }"
        ),
        want
    );
    // SELECT * shows the two variables of the pattern, not the hidden middle.
    let result = run(&db, "SELECT * WHERE { ?s (e:knows|^e:knows/e:knows) ?o }");
    let mut columns = result.columns.clone();
    columns.sort();
    assert_eq!(columns, ["o", "s"]);
    assert_eq!(
        rows(&db, "SELECT * WHERE { ?s (e:knows|^e:knows/e:knows) ?o }").len(),
        4
    );
}

#[test]
fn a_repeated_inverse_branch() {
    let db = knows_db();
    // knows: (Alix, Gus), (Vincent, Alix). ^knows+ walks back along knows:
    // from Gus to Alix and Vincent, from Alix to Vincent.
    assert_eq!(
        rows(&db, "SELECT ?s ?o WHERE { ?s (e:knows|^e:knows+) ?o }"),
        expected(&[
            &["alix", "gus"],
            &["vincent", "alix"],
            &["gus", "alix"],
            &["gus", "vincent"],
            &["alix", "vincent"],
        ])
    );
    assert_eq!(
        rows(&db, "SELECT ?o WHERE { e:gus (e:knows|^e:knows+) ?o }"),
        expected(&[&["alix"], &["vincent"]])
    );
    // A repeated forward branch beside an inverse one.
    assert_eq!(
        rows(&db, "SELECT ?o WHERE { e:vincent (e:knows+|^e:knows) ?o }"),
        expected(&[&["alix"], &["gus"]])
    );
}

#[test]
fn both_branches_finding_one_pair_give_it_twice() {
    // Mia and Butch know each other: the pair comes from both branches, and
    // an alternative is a bag union, so it is there twice; DISTINCT folds it.
    let db = rdf_db("e:mia e:knows e:butch . e:butch e:knows e:mia .");
    assert_eq!(
        rows(&db, "SELECT ?o WHERE { e:mia (e:knows|^e:knows) ?o }"),
        expected(&[&["butch"], &["butch"]])
    );
    assert_eq!(
        rows(
            &db,
            "SELECT DISTINCT ?o WHERE { e:mia (e:knows|^e:knows) ?o }"
        ),
        expected(&[&["butch"]])
    );
}

/// The fixture in the named graph `e:paris`, a decoy in the default graph and
/// one in `e:berlin`: an inverse alternative inside GRAPH sees only its graph.
fn named_db() -> GrafeoDB {
    rdf_db(
        "GRAPH e:paris { e:vincent e:knows e:alix . e:alix e:knows e:gus . } \
         e:alix e:knows e:jules . \
         GRAPH e:berlin { e:mia e:knows e:alix . }",
    )
}

#[test]
fn an_inverse_alternative_in_a_named_graph() {
    let db = named_db();
    assert_eq!(
        rows(
            &db,
            "SELECT ?o WHERE { GRAPH e:paris { e:alix (e:knows|^e:knows) ?o } }"
        ),
        expected(&[&["gus"], &["vincent"]])
    );
    assert_eq!(
        rows(
            &db,
            "SELECT ?s ?o WHERE { GRAPH e:paris { ?s (e:knows|^e:knows/e:knows) ?o } }"
        ),
        expected(&[
            &["alix", "gus"],
            &["vincent", "alix"],
            &["gus", "gus"],
            &["alix", "alix"],
        ])
    );
    // The default graph holds only the decoy: Alix knows Jules.
    assert_eq!(
        rows(&db, "SELECT ?o WHERE { e:alix (e:knows|^e:knows) ?o }"),
        expected(&[&["jules"]])
    );
}

#[test]
fn an_inverse_alternative_over_every_named_graph() {
    let db = named_db();
    // GRAPH ?g binds the graph of each row; no pair crosses two graphs.
    assert_eq!(
        rows(
            &db,
            "SELECT ?g ?o WHERE { GRAPH ?g { e:alix (e:knows|^e:knows) ?o } }"
        ),
        expected(&[&["paris", "gus"], &["paris", "vincent"], &["berlin", "mia"]])
    );
    assert_eq!(
        rows(
            &db,
            "SELECT ?g ?o WHERE { GRAPH ?g { e:gus (e:knows|^e:knows+) ?o } }"
        ),
        expected(&[&["paris", "alix"], &["paris", "vincent"]])
    );
}
