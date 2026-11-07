//! The query timeout stops a long property path: `*` over a dense graph, and `?` run per row over a large
//! input, both well past the limit on any machine. The path checks its deadline on its first row and every
//! 1,024 rows, so the error comes soon after the limit, and the database answers after it. Each case
//! runs in a child process with a deadline, so a path that ignores the limit fails its test rather than
//! hanging the test binary.
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

fn a_long_path_walk_times_out_here() {
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

/// 5,000 nodes with up to ten `q` edges each, every edge to a later node: no cycle, and from an early
/// node `+` reaches most of the graph. `<urn:island>` is held (a `t` triple) but no `q` edge reaches it.
fn acyclic() -> GrafeoDB {
    let db = GrafeoDB::with_config(Config::in_memory().with_query_timeout(LIMIT)).unwrap();
    let session = db.session();
    let mut triples = vec!["<urn:island> <urn:t> <urn:k0> .".to_string()];
    for node in 0..5_000_u64 {
        for step in 1..=10_u64 {
            let target = node + step * 3;
            if target < 5_000 {
                triples.push(format!("<urn:n{node}> <urn:q> <urn:n{target}> ."));
            }
        }
    }
    for part in triples.chunks(2_000) {
        session
            .execute_sparql(&format!("INSERT DATA {{ {} }}", part.join(" ")))
            .unwrap();
    }
    db
}

/// The deadline counts the work a path search does, not the rows it gives: a search that finds nothing
/// still stops at the limit (every 1,024 states it visits). `?x q+ ?x` over a graph without a cycle walks
/// every node's descendants and gives no row; no term is missing from the graph, so `ASK`'s no-read
/// shortcut does not apply. A search to a held term no edge reaches may be answered quickly by a plan
/// that walks back from it; it must still end by the limit, empty or with the timeout.
fn a_path_search_that_finds_nothing_stops_at_the_limit_here() {
    let db = acyclic();
    for (query, must_time_out) in [
        ("SELECT ?x WHERE { ?x <urn:q>+ ?x }", true),
        ("SELECT (COUNT(*) AS ?c) WHERE { ?x <urn:q>+ ?x }", true),
        ("SELECT ?x WHERE { ?x <urn:q>+ <urn:island> }", false),
        ("ASK { ?x <urn:q>+ <urn:island> }", false),
        (
            "SELECT ?x WHERE { <urn:n0> <urn:q>+ ?x . ?x <urn:q>+ <urn:island> }",
            false,
        ),
    ] {
        let session = db.session();
        let started = Instant::now();
        let outcome = session.execute_sparql(query);
        let elapsed = started.elapsed();
        match outcome {
            Err(error) => {
                let error = error.to_string().to_lowercase();
                assert!(
                    error.contains("timeout") || error.contains("timed out"),
                    "{query}: {error}"
                );
            }
            Ok(result) => {
                assert!(!must_time_out, "{query}: finished in {elapsed:?}");
                // Nothing to find; `ASK` answers false, a count 0.
                use grafeo_common::types::Value;
                let rows = result.rows().to_vec();
                assert!(
                    rows.is_empty()
                        || rows == vec![vec![Value::Bool(false)]]
                        || rows == vec![vec![Value::Int64(0)]],
                    "{query}: {rows:?}"
                );
            }
        }
        assert!(
            elapsed <= LIMIT + SLACK,
            "{query}: it ended after {elapsed:?}"
        );
    }
}

/// The variable naming the case the child runs.
const CHILD_VAR: &str = "GRAFEO_SPARQL_PATH_DEADLINE_CHILD";
/// How long a child may take: building its graph, then every query within the limit and its slack.
const CHILD_DEADLINE: Duration = Duration::from_secs(120);

/// The child: runs the case [`CHILD_VAR`] names. A no-op when run directly.
#[test]
fn path_deadline_child() {
    match std::env::var(CHILD_VAR).as_deref() {
        Ok("walk") => a_long_path_walk_times_out_here(),
        Ok("nothing") => a_path_search_that_finds_nothing_stops_at_the_limit_here(),
        _ => {}
    }
}

fn in_a_child(case: &str) {
    let output = grafeo_common::testing::child_process::output_with_deadline(
        grafeo_common::child_test!("path_deadline_child").env(CHILD_VAR, case),
        CHILD_DEADLINE,
    )
    .unwrap_or_else(|error| panic!("{case}: still running after {CHILD_DEADLINE:?}: {error}"));
    assert!(
        output.status.success(),
        "{case}:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn a_long_path_walk_times_out() {
    in_a_child("walk");
}

#[test]
fn a_path_search_that_finds_nothing_stops_at_the_limit() {
    in_a_child("nothing");
}
