//! RDF change events, live and durable (#679 patch 8, after #718).
//!
//! - **Live:** each triple a commit changes gives one event: `Create` for an insert, `Delete` for
//!   a delete, with the triple's terms (N-Triples), its graph, and the id the documented rule gives
//!   (the public helpers' id for the same terms). A write that changes nothing (an insert of a
//!   triple there, a delete of one absent) and a rolled-back transaction give none. A transaction
//!   reports its net changes: a triple inserted and deleted in it, or an existing one deleted and
//!   inserted again, gives none.
//! - **Graph operations** (`CLEAR`, `DROP`, `COPY`, `MOVE`, `ADD`) give no events on their own.
//!   Inside a transaction `COPY`, `MOVE` and `ADD` record each copied triple as an insert, which
//!   reports a `Create`; the clear of the target and the drop of a moved source report nothing.
//! - **Durable:** after a close and after a crash, the feed's events equal the live ones field for
//!   field, in the default graph and named graphs, beside LPG events of the same transactions.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main cdc_rdf_feed::
//! ```

#![cfg(all(
    feature = "cdc",
    feature = "sparql",
    feature = "triple-store",
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file"
))]

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use grafeo_common::testing::child_process;
use grafeo_common::types::EpochId;
use grafeo_engine::cdc::{CdcBatch, CdcLog, ChangeEvent, ChangeKind, EntityId};
use grafeo_engine::{Config, GrafeoDB};

const ALL: NonZeroUsize = NonZeroUsize::new(1_000_000).unwrap();

const PREFIXES: &str =
    "PREFIX ex: <http://ex.org/> PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> ";
const EX: &str = "http://ex.org/";

fn sparql(db: &GrafeoDB, update: &str) {
    db.execute_sparql(&format!("{PREFIXES}{update}"))
        .unwrap_or_else(|error| panic!("{update}: {error}"));
}

fn iri(local: &str) -> String {
    format!("<{EX}{local}>")
}

/// The statements whose changes the oracle lists, in order: inserts of every term kind but blank
/// nodes (whose labels the store may rename), a named graph, no-op writes, a delete, and a
/// `DELETE/INSERT WHERE`.
fn listed_statements(db: &GrafeoDB) {
    sparql(
        db,
        r#"INSERT DATA { ex:alix ex:knows ex:gus . ex:alix ex:name "Alix" .
            ex:gus ex:age "28"^^xsd:integer . ex:gus ex:motto "chat"@fr . }"#,
    );
    sparql(
        db,
        "INSERT DATA { GRAPH ex:g { ex:mia ex:knows ex:alix . } }",
    );
    sparql(db, "INSERT DATA { ex:alix ex:knows ex:gus . }");
    sparql(db, r#"DELETE DATA { ex:gus ex:age "28"^^xsd:integer . }"#);
    sparql(db, "DELETE DATA { ex:nobody ex:knows ex:gus . }");
    sparql(
        db,
        "DELETE { ?s ex:name ?o } INSERT { ?s ex:label ?o } WHERE { ?s ex:name ?o }",
    );
}

/// What the listed statements change, in commit order: (kind, subject, predicate, object, graph).
fn listed_changes() -> Vec<(ChangeKind, String, String, String, Option<String>)> {
    let create = |s: &str, p: &str, o: String, g: Option<&str>| {
        (
            ChangeKind::Create,
            iri(s),
            iri(p),
            o,
            g.map(|g| format!("{EX}{g}")),
        )
    };
    let delete = |s: &str, p: &str, o: String| (ChangeKind::Delete, iri(s), iri(p), o, None);
    vec![
        create("alix", "knows", iri("gus"), None),
        create("alix", "name", "\"Alix\"".to_string(), None),
        create(
            "gus",
            "age",
            "\"28\"^^<http://www.w3.org/2001/XMLSchema#integer>".to_string(),
            None,
        ),
        create("gus", "motto", "\"chat\"@fr".to_string(), None),
        create("mia", "knows", iri("alix"), Some("g")),
        delete(
            "gus",
            "age",
            "\"28\"^^<http://www.w3.org/2001/XMLSchema#integer>".to_string(),
        ),
        delete("alix", "name", "\"Alix\"".to_string()),
        create("alix", "label", "\"Alix\"".to_string(), None),
    ]
}

/// The id the public helper gives the triple: the documented rule.
fn helper_id(subject: &str, predicate: &str, object: &str, graph: Option<&str>) -> u64 {
    let log = CdcLog::new();
    log.record_triple_insert(subject, predicate, object, graph, EpochId::new(1));
    assert_eq!(log.dropped_lines(), 0, "{subject} {predicate} {object}");
    match log.changes_between(EpochId::new(0), EpochId::new(2))[..] {
        [
            ChangeEvent {
                entity_id: EntityId::Triple(id),
                ..
            },
        ] => id,
        ref other => panic!("one triple event: {other:?}"),
    }
}

fn triple_events(db: &GrafeoDB) -> Vec<ChangeEvent> {
    db.changes_between(EpochId::new(0), db.current_epoch())
        .unwrap()
        .into_iter()
        .filter(|event| event.entity_id.is_triple())
        .collect()
}

/// Each triple a listed statement changed gives one live event, with its kind, terms, graph and
/// the documented id; the no-op insert and delete give none.
#[test]
fn each_changed_triple_gives_one_event() {
    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
    listed_statements(&db);
    let events = triple_events(&db);
    let got: Vec<_> = events
        .iter()
        .map(|event| {
            (
                event.kind.clone(),
                event.triple_subject.clone().unwrap(),
                event.triple_predicate.clone().unwrap(),
                event.triple_object.clone().unwrap(),
                event.triple_graph.clone(),
            )
        })
        .collect();
    assert_eq!(got, listed_changes());
    for event in &events {
        let (Some(s), Some(p), Some(o)) = (
            &event.triple_subject,
            &event.triple_predicate,
            &event.triple_object,
        ) else {
            unreachable!()
        };
        assert_eq!(
            event.entity_id,
            EntityId::Triple(helper_id(s, p, o, event.triple_graph.as_deref())),
            "{s} {p} {o}"
        );
    }
}

/// A rolled-back transaction gives no events; a transaction with an LPG and an RDF write gives
/// both events, at its one epoch.
#[test]
fn transactions_give_their_net_changes() {
    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute_sparql(&format!(
            "{PREFIXES}INSERT DATA {{ ex:hans ex:knows ex:gus . }}"
        ))
        .unwrap();
    session.rollback().unwrap();
    assert_eq!(
        triple_events(&db).len(),
        0,
        "the rolled-back one gives none"
    );

    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Django'})")
        .unwrap();
    session
        .execute_sparql(&format!(
            "{PREFIXES}INSERT DATA {{ ex:django ex:knows ex:gus . }}"
        ))
        .unwrap();
    session.commit().unwrap();
    let all = db
        .changes_between(EpochId::new(0), db.current_epoch())
        .unwrap();
    let last: Vec<&ChangeEvent> = all
        .iter()
        .filter(|event| event.epoch == db.current_epoch())
        .collect();
    assert_eq!(
        last.len(),
        2,
        "one node and one triple at the last commit: {last:?}"
    );
    assert_eq!(
        last.iter()
            .filter(|event| event.entity_id.is_triple())
            .count(),
        1
    );
}

/// A transaction reports its net changes: a triple inserted and deleted in it, and an existing one
/// deleted and inserted again, were never a committed change, so neither gives an event.
#[test]
fn a_triple_written_and_unwritten_in_one_transaction_gives_no_events() {
    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
    sparql(&db, "INSERT DATA { ex:alix ex:knows ex:gus . }");
    let before = triple_events(&db).len();
    let mut session = db.session();
    for (first, second) in [
        (
            "INSERT DATA { ex:hans ex:knows ex:gus . }",
            "DELETE DATA { ex:hans ex:knows ex:gus . }",
        ),
        (
            "DELETE DATA { ex:alix ex:knows ex:gus . }",
            "INSERT DATA { ex:alix ex:knows ex:gus . }",
        ),
    ] {
        session.begin_transaction().unwrap();
        for update in [first, second] {
            session
                .execute_sparql(&format!("{PREFIXES}{update}"))
                .unwrap();
        }
        session.commit().unwrap();
        assert_eq!(triple_events(&db).len(), before, "{first} then {second}");
    }
}

/// Graph operations on their own give no events, whatever triples they clear, copy or move.
#[test]
fn graph_operations_on_their_own_give_no_events() {
    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
    listed_statements(&db);
    let before = triple_events(&db).len();
    for operation in [
        "COPY DEFAULT TO ex:h",
        "ADD ex:g TO ex:h",
        "MOVE ex:h TO ex:k",
        "CLEAR GRAPH ex:g",
        "DROP GRAPH ex:k",
        "CLEAR DEFAULT",
    ] {
        sparql(&db, operation);
        assert_eq!(triple_events(&db).len(), before, "{operation}");
    }
}

/// Inside a transaction `COPY`, `MOVE` and `ADD` record each copied triple as an insert, so each
/// reports a `Create` in its target graph; the drop of the moved source reports nothing, and
/// `CLEAR` and `DROP` report nothing either.
#[test]
fn copied_triples_inside_a_transaction_report_creates() {
    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
    listed_statements(&db);
    let mut session = db.session();
    let mut run = |operation: &str| {
        let before = triple_events(&db).len();
        session.begin_transaction().unwrap();
        session
            .execute_sparql(&format!("{PREFIXES}{operation}"))
            .unwrap_or_else(|error| panic!("{operation}: {error}"));
        session.commit().unwrap();
        triple_events(&db)[before..]
            .iter()
            .map(|event| (event.kind.clone(), event.triple_graph.clone()))
            .collect::<Vec<_>>()
    };
    let graph = |local: &str| Some(format!("{EX}{local}"));
    // After the listed statements the default graph holds three triples and ex:g one.
    assert_eq!(
        run("COPY DEFAULT TO ex:h"),
        vec![(ChangeKind::Create, graph("h")); 3],
        "COPY: one create per copied triple"
    );
    assert_eq!(
        run("ADD ex:g TO ex:k"),
        vec![(ChangeKind::Create, graph("k"))],
        "ADD: the one triple of ex:g"
    );
    assert_eq!(
        run("MOVE ex:k TO ex:m"),
        vec![(ChangeKind::Create, graph("m"))],
        "MOVE: a create in the target, nothing for the dropped source"
    );
    assert_eq!(run("CLEAR GRAPH ex:h"), vec![], "CLEAR reports nothing");
    assert_eq!(run("DROP GRAPH ex:m"), vec![], "DROP reports nothing");
}

// ---------------------------------------------------------------------------- durable

/// The whole workload: the listed statements, the transactions (one copying a graph), and graph
/// operations between.
fn workload(db: &GrafeoDB) {
    listed_statements(db);
    sparql(db, "COPY DEFAULT TO ex:h");
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Django'})")
        .unwrap();
    session
        .execute_sparql(&format!(
            "{PREFIXES}INSERT DATA {{ GRAPH ex:h {{ ex:django ex:knows ex:mia . }} }}"
        ))
        .unwrap();
    session
        .execute_sparql(&format!(
            "{PREFIXES}DELETE DATA {{ ex:alix ex:knows ex:gus . }}"
        ))
        .unwrap();
    session.commit().unwrap();
    session.begin_transaction().unwrap();
    session
        .execute_sparql(&format!("{PREFIXES}COPY ex:g TO ex:j"))
        .unwrap();
    session.commit().unwrap();
    session.begin_transaction().unwrap();
    session
        .execute_sparql(&format!(
            "{PREFIXES}INSERT DATA {{ ex:hans ex:knows ex:gus . }}"
        ))
        .unwrap();
    session.rollback().unwrap();
    sparql(db, "DROP GRAPH ex:g");
    sparql(db, r#"INSERT DATA { ex:beatrix ex:name "Beatrix"@en . }"#);
}

fn as_json(events: &[ChangeEvent]) -> Vec<serde_json::Value> {
    events
        .iter()
        .map(|event| serde_json::to_value(event).unwrap())
        .collect()
}

fn live(db: &GrafeoDB) -> Vec<serde_json::Value> {
    as_json(
        &db.changes_between(EpochId::new(0), db.current_epoch())
            .unwrap(),
    )
}

fn durable(batches: &[CdcBatch]) -> Vec<serde_json::Value> {
    let events: Vec<ChangeEvent> = batches
        .iter()
        .flat_map(|batch| batch.events.iter().cloned())
        .collect();
    as_json(&events)
}

fn with_workload(path: &Path) -> GrafeoDB {
    let db = GrafeoDB::open(path).unwrap();
    db.set_cdc(true).unwrap();
    workload(&db);
    db
}

/// After a close the feed's events equal the live ones, field for field.
#[test]
fn the_durable_rdf_events_equal_the_live_ones_after_a_close() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("triples.grafeo");
    let db = with_workload(&path);
    let live = live(&db);
    assert!(live.len() >= 12, "{} live events", live.len());
    db.close().unwrap();
    drop(db);
    let reopened = GrafeoDB::open(&path).unwrap();
    assert_eq!(durable(&reopened.read_cdc_feed(ALL).unwrap()), live);
}

const PATH_VAR: &str = "GRAFEO_CDC_RDF_FEED_PATH";
const LIVE_VAR: &str = "GRAFEO_CDC_RDF_FEED_LIVE";

fn sidecar_wal(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.wal", path.display()))
}

/// After a crash (a child exits with the database open) the feed's events equal the live ones the
/// child wrote down.
#[test]
fn the_durable_rdf_events_equal_the_live_ones_after_a_crash() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("triples.grafeo");
    let live_file = dir.path().join("live.json");
    let status = child_process::run(
        grafeo_common::child_test!("rdf_workload_child")
            .env(PATH_VAR, &path)
            .env(LIVE_VAR, &live_file),
    )
    .unwrap();
    assert!(status.success(), "the child failed");
    assert!(sidecar_wal(&path).exists(), "the crash left the WAL");
    let live: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(&live_file).unwrap()).unwrap();
    let reopened = GrafeoDB::open(&path).unwrap();
    assert_eq!(durable(&reopened.read_cdc_feed(ALL).unwrap()), live);
}

/// Child-process entry for the test above; a no-op when run directly.
#[test]
fn rdf_workload_child() {
    let Some(path) = std::env::var_os(PATH_VAR) else {
        return;
    };
    let db = with_workload(Path::new(&path));
    let live_file = std::env::var_os(LIVE_VAR).unwrap();
    std::fs::write(live_file, serde_json::to_vec(&live(&db)).unwrap()).unwrap();
    std::process::exit(0);
}
