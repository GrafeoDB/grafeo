//! The query timeout stops a long property path: `*` over a dense graph, and `?` run per row over a large
//! input, both well past the limit on any machine. The path checks its deadline on its first row and every
//! 1,024 rows, so the error comes soon after the limit, and the database answers after it.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main sparql_path_deadline::
//! ```

use std::time::{Duration, Instant};

use grafeo_engine::{Config, GrafeoDB};

const LIMIT: Duration = Duration::from_millis(300);
/// How late the error may come on a slow CI runner.
const SLACK: Duration = Duration::from_secs(8);

/// 3,000 nodes with ten `q` edges each, spread so that `*` reaches every node from every node, and as
/// many `t` triples for a large input.
fn dense() -> GrafeoDB {
    let db = GrafeoDB::with_config(Config::in_memory().with_query_timeout(LIMIT)).unwrap();
    let session = db.session();
    let mut triples = Vec::new();
    for node in 0..3_000_u64 {
        for step in 1..=10_u64 {
            let target = (node * 7 + step * 131) % 3_000;
            triples.push(format!("<urn:n{node}> <urn:q> <urn:n{target}> ."));
        }
        triples.push(format!("<urn:n{node}> <urn:t> <urn:k{}> .", node % 7));
    }
    for part in triples.chunks(2_000) {
        session
            .execute_sparql(&format!("INSERT DATA {{ {} }}", part.join(" ")))
            .unwrap();
    }
    db
}

#[test]
fn a_long_path_walk_times_out() {
    let db = dense();
    for query in [
        "SELECT (COUNT(*) AS ?c) WHERE { ?x <urn:q>* ?y }",
        "SELECT (COUNT(*) AS ?c) WHERE { ?x <urn:q>+ ?y }",
        "SELECT (COUNT(*) AS ?n) WHERE { ?a <urn:t> ?b . ?c <urn:t> ?d . ?x <urn:q>? ?y }",
        "SELECT (COUNT(*) AS ?c) WHERE { ?x <urn:t> ?b . ?x <urn:q>* ?y }",
    ] {
        let session = db.session();
        let started = Instant::now();
        let outcome = session.execute_sparql(query);
        let elapsed = started.elapsed();
        let error = match outcome {
            Ok(result) => panic!("{query}: finished in {elapsed:?} with {:?}", result.rows()),
            Err(error) => error.to_string(),
        };
        assert!(
            error.to_lowercase().contains("timeout") || error.to_lowercase().contains("timed out"),
            "{query}: {error}"
        );
        assert!(
            elapsed <= LIMIT + SLACK,
            "{query}: the error came after {elapsed:?}"
        );
        // The database answers after it.
        let count = session
            .execute_sparql("SELECT (COUNT(*) AS ?c) WHERE { <urn:n1> <urn:q> ?y }")
            .unwrap();
        assert_eq!(count.row_count(), 1, "{query}");
    }
}
