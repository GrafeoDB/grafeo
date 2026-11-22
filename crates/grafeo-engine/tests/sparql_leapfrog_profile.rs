//! EXPLAIN and PROFILE of a SPARQL query on the Ring's leapfrog join: the join reads its triple
//! patterns itself, so their lines are absorbed into its line, as a lookup's scan is. Before
//! this, both failed with an internal error ("PROFILE cannot match the operators that ran to the
//! plan"), while the query itself answered.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main sparql_leapfrog_profile::
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store", feature = "ring-index"))]

use std::fmt::Write;

use grafeo_engine::GrafeoDB;

/// 200 triangles of `:knows`, with the Ring built: a cyclic join of three patterns runs on the
/// Ring's leapfrog join, three matches per triangle.
fn triangles() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let mut data = String::from("PREFIX : <urn:> INSERT DATA {");
    for i in 0..200 {
        write!(
            data,
            " :a{i} :knows :b{i} . :b{i} :knows :c{i} . :c{i} :knows :a{i} ."
        )
        .unwrap();
    }
    data.push_str(" }");
    db.execute_sparql(&data).unwrap();
    db.rdf_store().rebuild_ring();
    db
}

const TRIANGLE: &str =
    "PREFIX : <urn:> SELECT ?x ?y ?z WHERE { ?x :knows ?y . ?y :knows ?z . ?z :knows ?x }";

/// The text of the one-row, one-column result of `query`.
fn text(db: &GrafeoDB, query: &str) -> String {
    let result = db
        .execute_sparql(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    format!("{:?}", result.rows())
}

#[test]
fn explain_and_profile_of_the_ring_leapfrog_answer() {
    let db = triangles();
    assert_eq!(db.execute_sparql(TRIANGLE).unwrap().rows().len(), 600);

    let explain = text(&db, &format!("EXPLAIN {TRIANGLE}"));
    assert!(explain.contains("RdfLeapfrog"), "{explain}");

    let profile = text(&db, &format!("PROFILE {TRIANGLE}"));
    let line = profile
        .split("\\n")
        .find(|line| line.contains("RdfLeapfrog"))
        .unwrap_or_else(|| panic!("a line for the leapfrog join: {profile}"));
    assert!(line.contains("rows=600"), "its rows: {profile}");
    assert!(
        !profile.contains("RdfTripleScan ("),
        "the scans are absorbed, no line of their own: {profile}"
    );
}
