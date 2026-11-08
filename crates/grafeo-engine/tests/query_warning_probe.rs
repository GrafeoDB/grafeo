//! The warning channel end to end, through the test probe (`testing-warning-probe`):
//! `Session::set_warning_probe(Some(stage))` raises `Warning::Probe` (01Z99, a notice) into each
//! statement's translate-time set, which the plan cache stores with the plan, or into the run-time
//! sink when the executor starts a plan, a cache hit included. A result shows one warning per
//! kind, whichever stages raised it; a new warning needs no change here.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main query_warning_probe::
//! ```

#![cfg(feature = "testing-warning-probe")]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_common::utils::error::Result;
use grafeo_common::utils::warning::WarningKind;
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::QueryResult;
use grafeo_engine::session::{ProbeStage, Session};

const PROBE: &str = "01Z99";

fn codes(result: &QueryResult) -> Vec<String> {
    result
        .warnings()
        .iter()
        .map(|w| w.code.as_str().to_string())
        .collect()
}

fn run(session: &Session, query: &str) -> QueryResult {
    session
        .execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
}

/// The probe's warning as a result shows it: its code and kind exactly, a message that says so.
fn assert_probe(result: &QueryResult) {
    let [warning] = result.warnings() else {
        panic!("one warning: {:?}", result.warnings());
    };
    assert_eq!(warning.code.as_str(), PROBE);
    assert_eq!(warning.kind, WarningKind::Notice);
    assert!(warning.message.contains("probe"), "{}", warning.message);
}

#[test]
fn a_new_session_raises_nothing() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    assert_eq!(
        codes(&run(&session, "RETURN 101 AS v")),
        Vec::<String>::new()
    );
}

#[test]
fn the_translate_probe_reaches_the_result() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.set_warning_probe(Some(ProbeStage::Translate));
    assert_probe(&run(&session, "RETURN 102 AS v"));
}

#[test]
fn the_run_probe_reaches_the_result() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.set_warning_probe(Some(ProbeStage::Run));
    assert_probe(&run(&session, "RETURN 103 AS v"));
}

/// The translate-time set lives in the cache entry: a hit after the probe is cleared still warns,
/// and a statement translated after it does not.
#[test]
fn a_translate_time_warning_comes_back_from_the_cache() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.set_warning_probe(Some(ProbeStage::Translate));
    assert_probe(&run(&session, "RETURN 104 AS v"));
    session.set_warning_probe(None);
    assert_probe(&run(&session, "RETURN 104 AS v"));
    assert_eq!(
        codes(&run(&session, "RETURN 105 AS v")),
        Vec::<String>::new(),
        "a new statement"
    );
}

/// The run-time sink is filled on each run, a cache hit included, and only while the probe is set.
#[test]
fn a_run_time_warning_is_raised_on_each_run() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    assert_eq!(
        codes(&run(&session, "RETURN 106 AS v")),
        Vec::<String>::new()
    );
    session.set_warning_probe(Some(ProbeStage::Run));
    assert_probe(&run(&session, "RETURN 106 AS v"));
    assert_probe(&run(&session, "RETURN 106 AS v"));
    session.set_warning_probe(None);
    assert_eq!(
        codes(&run(&session, "RETURN 106 AS v")),
        Vec::<String>::new()
    );
}

/// Both stages raising for one statement give one warning: the translate-time one from the cache
/// entry and the run-time one on the hit.
#[test]
fn both_stages_give_one_warning() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.set_warning_probe(Some(ProbeStage::Translate));
    assert_probe(&run(&session, "RETURN 107 AS v"));
    session.set_warning_probe(Some(ProbeStage::Run));
    assert_probe(&run(&session, "RETURN 107 AS v"));
}

/// The probe and a real warning: both, ordered by code.
#[test]
fn the_probe_and_a_deprecation_are_ordered_by_code() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.set_warning_probe(Some(ProbeStage::Run));
    let result = run(
        &session,
        "RETURN ZONED DATETIME '2024-01-01T10:00:00+01:00' AS v",
    );
    assert_eq!(codes(&result), ["01X01", PROBE]);
}

/// A statement that fails returns its error only, whatever was raised before.
#[test]
fn a_failing_statement_returns_its_error_only() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.set_warning_probe(Some(ProbeStage::Run));
    assert!(session.execute("RETURN 1 / 0 AS v").is_err());
    // The session goes on, and the next statement warns again.
    assert_probe(&run(&session, "RETURN 108 AS v"));
}

/// A probe set on one session leaves another one alone.
#[test]
fn the_probe_belongs_to_its_session() {
    let db = GrafeoDB::new_in_memory();
    let probed = db.session();
    let other = db.session();
    probed.set_warning_probe(Some(ProbeStage::Run));
    assert_probe(&run(&probed, "RETURN 109 AS v"));
    assert_eq!(codes(&run(&other, "RETURN 110 AS v")), Vec::<String>::new());
}

/// How a probe test runs a query on one entry path.
type Run = fn(&Session, &str) -> Result<QueryResult>;

fn params() -> HashMap<String, Value> {
    HashMap::from([("p".to_string(), Value::Int64(1))])
}

/// One entry path: its name, how it runs a query, and two queries no other path runs (one
/// per stage, so the translate stage never meets a cached plan).
fn entry(
    name: &'static str,
    run: Run,
    queries: [&'static str; 2],
) -> (&'static str, Run, [&'static str; 2]) {
    (name, run, queries)
}

/// Every entry path of a session in this build: each language's execute and its
/// `_with_params`, `execute_language`, `PROFILE`, and the RDF path.
fn entry_paths() -> Vec<(&'static str, Run, [&'static str; 2])> {
    let mut paths = vec![
        entry(
            "execute",
            |s, q| s.execute(q),
            ["RETURN 201 AS v", "RETURN 202 AS v"],
        ),
        entry(
            "execute_with_params",
            |s, q| s.execute_with_params(q, params()),
            ["RETURN $p + 203 AS v", "RETURN $p + 204 AS v"],
        ),
        entry(
            "execute_language",
            |s, q| s.execute_language(q, "gql", None),
            ["RETURN 205 AS v", "RETURN 206 AS v"],
        ),
        entry(
            "PROFILE",
            |s, q| s.execute(q),
            [
                "PROFILE MATCH (n:Person) RETURN n.i + 207 AS v",
                "PROFILE MATCH (n:Person) RETURN n.i + 208 AS v",
            ],
        ),
    ];
    #[cfg(feature = "cypher")]
    paths.extend([
        entry(
            "execute_cypher",
            |s, q| s.execute_cypher(q),
            ["RETURN 211 AS v", "RETURN 212 AS v"],
        ),
        entry(
            "execute_cypher_with_params",
            |s, q| s.execute_cypher_with_params(q, params()),
            ["RETURN $p + 213 AS v", "RETURN $p + 214 AS v"],
        ),
    ]);
    #[cfg(feature = "gremlin")]
    paths.extend([
        entry(
            "execute_gremlin",
            |s, q| s.execute_gremlin(q),
            ["g.V().has('i', 1).count()", "g.V().has('i', 2).count()"],
        ),
        entry(
            "execute_gremlin_with_params",
            |s, q| s.execute_gremlin_with_params(q, params()),
            [
                "g.V().has('i', $p).count()",
                "g.V().has('name', $p).count()",
            ],
        ),
    ]);
    #[cfg(feature = "graphql")]
    paths.extend([
        entry(
            "execute_graphql",
            |s, q| s.execute_graphql(q),
            ["{ person { name } }", "{ person { i } }"],
        ),
        entry(
            "execute_graphql_with_params",
            |s, q| s.execute_graphql_with_params(q, params()),
            ["{ person { name i } }", "{ person { i name } }"],
        ),
    ]);
    #[cfg(feature = "sql-pgq")]
    paths.extend([
        entry(
            "execute_sql",
            |s, q| s.execute_sql(q),
            [
                "SELECT * FROM GRAPH_TABLE (MATCH (n:Person) WHERE n.i = 1 COLUMNS (n.name AS name))",
                "SELECT * FROM GRAPH_TABLE (MATCH (n:Person) WHERE n.i = 2 COLUMNS (n.name AS name))",
            ],
        ),
        entry(
            "execute_sql_with_params",
            |s, q| s.execute_sql_with_params(q, params()),
            [
                "SELECT * FROM GRAPH_TABLE (MATCH (n:Person) WHERE n.i = $p COLUMNS (n.name AS name))",
                "SELECT * FROM GRAPH_TABLE (MATCH (n:Person) WHERE n.i <> $p COLUMNS (n.name AS name))",
            ],
        ),
    ]);
    #[cfg(all(feature = "sparql", feature = "triple-store"))]
    paths.extend([
        entry(
            "execute_sparql",
            |s, q| s.execute_sparql(q),
            [
                "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1",
                "SELECT ?s WHERE { ?s ?p ?o } LIMIT 2",
            ],
        ),
        entry(
            "execute_sparql_with_params",
            |s, q| s.execute_sparql_with_params(q, HashMap::new()),
            [
                "SELECT ?s WHERE { ?s ?p ?o } LIMIT 3",
                "SELECT ?s WHERE { ?s ?p ?o } LIMIT 4",
            ],
        ),
    ]);
    paths
}

/// Each entry path of the session carries each stage's probe to its result.
#[test]
fn every_entry_path_carries_both_stages() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Alix', i: 1})").unwrap();
    for (path, run, queries) in entry_paths() {
        for (stage, query) in [ProbeStage::Run, ProbeStage::Translate]
            .into_iter()
            .zip(queries)
        {
            let session = db.session();
            session.set_warning_probe(Some(stage));
            let result =
                run(&session, query).unwrap_or_else(|error| panic!("{path} `{query}`: {error}"));
            assert_eq!(codes(&result), [PROBE], "{path}, {stage:?}: `{query}`");
        }
    }
}

/// A stream carries each stage's probe: the translate stage's from the start, the run
/// stage's by the time it is exhausted.
#[test]
fn a_stream_carries_both_stages() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Alix', i: 1})").unwrap();
    let session = db.session();
    session.set_warning_probe(Some(ProbeStage::Translate));
    let stream = session
        .execute_streaming("MATCH (n:Person) RETURN n.i + 221 AS v")
        .unwrap();
    let probe: Vec<String> = stream
        .warnings()
        .iter()
        .map(|w| w.code.as_str().to_string())
        .collect();
    assert_eq!(probe, [PROBE], "translate stage, from the start");
    drop(stream);
    session.set_warning_probe(Some(ProbeStage::Run));
    let mut stream = session
        .execute_streaming("MATCH (n:Person) RETURN n.i + 222 AS v")
        .unwrap();
    while stream.next_chunk().unwrap().is_some() {}
    let probe: Vec<String> = stream
        .warnings()
        .iter()
        .map(|w| w.code.as_str().to_string())
        .collect();
    assert_eq!(probe, [PROBE], "run stage, once exhausted");
}
