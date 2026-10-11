//! SPARQL 1.1 Update graph operations change the graphs they name, and no
//! other: a `COPY` or `MOVE` of a graph onto itself does nothing (sections
//! 3.2.3 and 3.2.4), and neither does an `ADD` (which would add the triples
//! the graph holds), once the source check passed; `CLEAR NAMED` and `DROP
//! NAMED` reach every named graph and leave the default graph (sections 3.1.5
//! and 3.2.2), `DEFAULT` the default graph only, `ALL` every graph.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test sparql_graph_operations
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use grafeo_engine::config::GraphModel;
use grafeo_engine::{Config, GrafeoDB};

const PARIS: &str = "http://example.org/paris";
const BERLIN: &str = "http://example.org/berlin";

/// Alix knows Gus in the default graph; Vincent knows Mia in Paris, Jules
/// knows Butch in Berlin.
fn load(db: &GrafeoDB) {
    db.execute_sparql(&format!(
        "INSERT DATA {{ <http://example.org/alix> <http://example.org/knows> <http://example.org/gus> . \
         GRAPH <{PARIS}> {{ <http://example.org/vincent> <http://example.org/knows> <http://example.org/mia> }} \
         GRAPH <{BERLIN}> {{ <http://example.org/jules> <http://example.org/knows> <http://example.org/butch> }} }}"
    ))
    .unwrap();
}

fn loaded() -> GrafeoDB {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    load(&db);
    db
}

/// The triples of the default graph, and the named graphs with their triple
/// counts, sorted.
fn shape(db: &GrafeoDB) -> (usize, Vec<(String, usize)>) {
    let store = db.rdf_store();
    let mut named: Vec<(String, usize)> = store
        .graph_names()
        .into_iter()
        .map(|name| {
            let len = store.graph(&name).map_or(0, |graph| graph.len());
            (name, len)
        })
        .collect();
    named.sort();
    (store.len(), named)
}

fn named(graphs: &[(&str, usize)]) -> Vec<(String, usize)> {
    graphs
        .iter()
        .map(|(name, len)| ((*name).to_string(), *len))
        .collect()
}

/// A copy, move or add of a graph onto itself leaves every graph as it was:
/// a named graph or the default graph, named with or without `GRAPH`, with
/// or without `SILENT`, through a session and through `execute_sparql`.
#[test]
fn copy_move_and_add_onto_the_same_graph_do_nothing() {
    let before = (1, named(&[(BERLIN, 1), (PARIS, 1)]));
    for operation in ["COPY", "MOVE", "ADD"] {
        for (source, target) in [
            (format!("<{PARIS}>"), format!("<{PARIS}>")),
            (format!("GRAPH <{PARIS}>"), format!("<{PARIS}>")),
            ("DEFAULT".to_string(), "DEFAULT".to_string()),
        ] {
            for silent in ["", "SILENT "] {
                let update = format!("{operation} {silent}{source} TO {target}");
                for through_session in [true, false] {
                    let db = loaded();
                    assert_eq!(shape(&db), before);
                    let outcome = if through_session {
                        db.session().execute_sparql(&update)
                    } else {
                        db.execute_sparql(&update)
                    };
                    outcome.unwrap_or_else(|error| panic!("{update}: {error}"));
                    assert_eq!(shape(&db), before, "{update} (session: {through_session})");
                }
            }
        }
    }
}

/// Onto itself, a source that does not exist still fails without `SILENT`
/// (the source check comes first, as in Jena); with `SILENT` nothing happens,
/// and no graph is created.
#[test]
fn onto_itself_a_missing_source_fails_without_silent_and_creates_nothing() {
    let prague = "http://example.org/prague";
    for operation in ["COPY", "MOVE", "ADD"] {
        let db = loaded();
        let error = db
            .execute_sparql(&format!("{operation} <{prague}> TO <{prague}>"))
            .expect_err(operation);
        assert!(
            error
                .to_string()
                .contains(&format!("Source graph <{prague}> does not exist")),
            "{operation}: {error}"
        );
        db.execute_sparql(&format!("{operation} SILENT <{prague}> TO <{prague}>"))
            .unwrap();
        assert_eq!(
            shape(&db),
            (1, named(&[(BERLIN, 1), (PARIS, 1)])),
            "{operation} SILENT: no graph created"
        );
    }
}

/// `CLEAR` and `DROP` of `NAMED`, `DEFAULT` and `ALL` change exactly the
/// graphs those keywords name.
#[test]
fn clear_and_drop_reach_the_graphs_their_target_names() {
    let cases = [
        ("CLEAR NAMED", (1, Vec::new())),
        ("DROP NAMED", (1, Vec::new())),
        ("CLEAR DEFAULT", (0, named(&[(BERLIN, 1), (PARIS, 1)]))),
        ("DROP DEFAULT", (0, named(&[(BERLIN, 1), (PARIS, 1)]))),
        ("CLEAR ALL", (0, Vec::new())),
        ("DROP ALL", (0, Vec::new())),
        (
            "CLEAR GRAPH <http://example.org/paris>",
            (1, named(&[(BERLIN, 1), (PARIS, 0)])),
        ),
        (
            "DROP GRAPH <http://example.org/paris>",
            (1, named(&[(BERLIN, 1)])),
        ),
    ];
    for (update, expected) in cases {
        for silent in [false, true] {
            let update = if silent {
                update.replacen(' ', " SILENT ", 1)
            } else {
                update.to_string()
            };
            let db = loaded();
            db.execute_sparql(&update)
                .unwrap_or_else(|error| panic!("{update}: {error}"));
            assert_eq!(shape(&db), expected, "{update}");
        }
    }
}

/// A graph operation that does nothing logs nothing: onto itself, a copy, a
/// move or an add adds no more WAL records than an update that changes no
/// triple (both commit their empty transaction).
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn a_graph_operation_onto_itself_logs_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let db =
        GrafeoDB::with_config(Config::persistent(dir.path().join("amsterdam.grafeo"))).unwrap();
    load(&db);
    let records = |db: &GrafeoDB| db.wal_status().record_count;

    let before = records(&db);
    db.execute_sparql(
        "DELETE DATA { <http://example.org/hans> <http://example.org/knows> <http://example.org/mia> }",
    )
    .unwrap();
    let empty_commit = records(&db) - before;

    for update in [
        format!("COPY <{PARIS}> TO <{PARIS}>"),
        format!("MOVE <{PARIS}> TO <{PARIS}>"),
        format!("ADD <{PARIS}> TO <{PARIS}>"),
        "MOVE DEFAULT TO DEFAULT".to_string(),
    ] {
        let before = records(&db);
        db.execute_sparql(&update).unwrap();
        assert_eq!(
            records(&db) - before,
            empty_commit,
            "{update} logs what an update that changes nothing logs"
        );
    }
    assert_eq!(shape(&db), (1, named(&[(BERLIN, 1), (PARIS, 1)])));
    db.close().unwrap();
}
