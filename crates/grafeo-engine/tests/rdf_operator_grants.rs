//! Two RDF operators that keep what they read take grants under the query memory limit: the
//! leapfrog join over the Ring Index buffers its matches, and a per-row `EXISTS` keeps an answer
//! for each distinct row it was asked about. Past the limit each fails with `GRAFEO-S001` naming
//! the query memory limit; under a large one it answers.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main rdf_operator_grants::
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use std::fmt::Write;

use grafeo_common::utils::error::Error;
use grafeo_engine::{Config, GrafeoDB};

const KIB: usize = 1024;
const MIB: usize = 1024 * KIB;

/// `result` is the S001 refusal naming the query memory limit.
fn assert_refused<T: std::fmt::Debug>(result: Result<T, Error>) {
    let error = result.expect_err("refused at the query memory limit");
    let text = error.to_string();
    assert_eq!(error.error_code().as_str(), "GRAFEO-S001", "{text}");
    assert!(
        text.contains("query memory limit"),
        "names the bound: {text}"
    );
}

/// A database under a query memory limit of `limit`, holding `data` (a SPARQL update). No
/// query timeout: these tests are about memory, and a coverage build runs the leapfrog join past
/// the default 30 seconds.
fn holding(limit: usize, data: &str) -> GrafeoDB {
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_query_memory_limit(limit)
            .without_query_timeout(),
    )
    .unwrap();
    db.execute_sparql(data).unwrap();
    db
}

/// `query` runs through the operator named `route`, as `explain` (`EXPLAIN` or `PROFILE`) shows.
fn assert_route(db: &GrafeoDB, explain: &str, query: &str, route: &str) {
    let profile = db.execute_sparql(&format!("{explain} {query}")).unwrap();
    assert!(
        format!("{:?}", profile.rows()).contains(route),
        "{query} runs as {route}: {:?}",
        profile.rows()
    );
}

/// 500 triangles of `:knows`: a cyclic join of three patterns, three matches per triangle, past
/// a 16 KiB query memory limit and small enough to run in seconds under coverage.
#[cfg(feature = "ring-index")]
fn triangles() -> String {
    let mut data = String::from("PREFIX : <urn:> INSERT DATA {");
    for i in 0..500 {
        write!(
            data,
            " :a{i} :knows :b{i} . :b{i} :knows :c{i} . :c{i} :knows :a{i} ."
        )
        .unwrap();
    }
    data.push_str(" }");
    data
}

#[cfg(feature = "ring-index")]
#[test]
fn the_rdf_leapfrog_join_refuses_at_the_query_limit() {
    let query =
        "PREFIX : <urn:> SELECT ?x ?y ?z WHERE { ?x :knows ?y . ?y :knows ?z . ?z :knows ?x }";
    let data = triangles();
    let small = holding(16 * KIB, &data);
    small.rdf_store().rebuild_ring();
    let large = holding(512 * MIB, &data);
    large.rdf_store().rebuild_ring();
    assert_route(&large, "EXPLAIN", query, "RdfLeapfrog");
    assert_refused(small.execute_sparql(query));
    assert_eq!(large.execute_sparql(query).unwrap().rows().len(), 1_500);
}

#[test]
fn a_per_row_exists_refuses_at_the_query_limit() {
    let mut data = String::from("INSERT DATA {");
    for i in 0..20_000 {
        write!(data, " <urn:s{i}> <urn:p> {i} . <urn:s{i}> <urn:q> {i} .").unwrap();
    }
    data.push_str(" }");
    // The pattern reads the row's `?o` in a BIND, so it is tested per row, and each of the
    // 20,000 rows is a value of its own.
    let query =
        "SELECT ?s (EXISTS { ?s <urn:q> ?x BIND(?x + ?o AS ?y) } AS ?e) WHERE { ?s <urn:p> ?o }";
    let small = holding(128 * KIB, &data);
    let large = holding(512 * MIB, &data);
    assert_route(&large, "PROFILE", query, "ExistsPerRow");
    assert_refused(small.execute_sparql(query));
    assert_eq!(large.execute_sparql(query).unwrap().rows().len(), 20_000);
}
