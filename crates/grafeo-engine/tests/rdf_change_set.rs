//! RDF updates are recorded in the transaction's change set (#414): a
//! commit applies them, logs them and reports them to change data capture at
//! its own epoch; a rollback, a rollback to a savepoint and a failed
//! statement leave nothing of them, in memory, in the WAL and in the change
//! feed.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main rdf_change_set::
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use grafeo_engine::config::GraphModel;
use grafeo_engine::{Config, GrafeoDB};

fn rdf_config() -> Config {
    Config::in_memory().with_graph_model(GraphModel::Rdf)
}

fn rdf_db() -> GrafeoDB {
    GrafeoDB::with_config(rdf_config()).unwrap()
}

const ALIX_KNOWS_GUS: &str =
    "INSERT DATA { <http://example.org/alix> <http://example.org/knows> <http://example.org/gus> }";

/// An update that inserts into the default graph and then fails: its second
/// template names a graph through a variable, which an update cannot target.
const FAILING_UPDATE: &str = "INSERT { <http://example.org/vincent> <http://example.org/knows> \
     <http://example.org/mia> . GRAPH ?g { <http://example.org/jules> \
     <http://example.org/knows> <http://example.org/butch> } } WHERE { BIND(<http://example.org/paris> AS ?g) }";

/// The answer of `ask`, an ASK query, in `session`: its one Boolean.
fn asks(session: &grafeo_engine::session::Session, ask: &str) -> bool {
    let result = session.execute_sparql(ask).unwrap();
    match result.rows() {
        [row] => row[0] == grafeo_common::types::Value::Bool(true),
        rows => panic!("ASK answers with one row: {rows:?}"),
    }
}

/// The subjects of every triple the default graph holds, sorted.
fn subjects(db: &GrafeoDB) -> Vec<String> {
    let mut subjects: Vec<String> = db
        .rdf_store()
        .triples(&grafeo_core::graph::rdf::RdfView::latest(
            grafeo_core::graph::rdf::LatestUse::Test,
        ))
        .iter()
        .map(|triple| triple.subject().to_string())
        .collect();
    subjects.sort();
    subjects
}

#[test]
fn a_failed_sparql_update_outside_a_transaction_leaves_nothing() {
    let db = rdf_db();
    let session = db.session();
    session
        .execute_sparql(FAILING_UPDATE)
        .expect_err("the update targets a variable graph");
    assert_eq!(
        subjects(&db),
        Vec::<String>::new(),
        "the triple the update inserted before it failed is gone"
    );
}

#[test]
fn a_failed_sparql_update_in_a_transaction_leaves_nothing() {
    let db = rdf_db();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute_sparql(ALIX_KNOWS_GUS).unwrap();
    session
        .execute_sparql(FAILING_UPDATE)
        .expect_err("the update targets a variable graph");
    session.commit().unwrap();
    assert_eq!(
        subjects(&db),
        ["<http://example.org/alix>"],
        "the transaction keeps its earlier update and nothing of the failed one"
    );
}

#[test]
fn a_rolled_back_savepoint_drops_the_rdf_changes_after_it() {
    let db = rdf_db();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute_sparql(ALIX_KNOWS_GUS).unwrap();
    session.savepoint("sp").unwrap();
    session
        .execute_sparql(
            "INSERT DATA { <http://example.org/mia> <http://example.org/knows> <http://example.org/jules> }",
        )
        .unwrap();
    session.rollback_to_savepoint("sp").unwrap();
    let mia =
        "ASK { <http://example.org/mia> <http://example.org/knows> <http://example.org/jules> }";
    assert!(
        !asks(&session, mia),
        "the transaction no longer reads what it undid"
    );
    session.commit().unwrap();
    assert_eq!(subjects(&db), ["<http://example.org/alix>"]);
}

/// A transaction no longer reads a committed triple it deleted, in the
/// default graph or a named graph, by any form of read; others still read
/// it until the commit.
#[test]
fn a_transaction_does_not_read_a_committed_triple_it_deleted() {
    let db = rdf_db();
    db.session().execute_sparql(ALIX_KNOWS_GUS).unwrap();
    insert_into(&db.session(), PARIS, "jules");
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute_sparql(
            "DELETE DATA { <http://example.org/alix> <http://example.org/knows> <http://example.org/gus> }",
        )
        .unwrap();
    session
        .execute_sparql(&format!(
            "DELETE DATA {{ GRAPH <{PARIS}> {{ <http://example.org/jules>              <http://example.org/knows> <http://example.org/gus> }} }}"
        ))
        .unwrap();
    let rows = |session: &grafeo_engine::session::Session, query: &str| {
        session.execute_sparql(query).unwrap().row_count()
    };
    let reads = [
        "SELECT ?s WHERE { ?s ?p ?o }".to_string(),
        format!("SELECT ?s WHERE {{ GRAPH <{PARIS}> {{ ?s ?p ?o }} }}"),
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s ?p ?o } }".to_string(),
    ];
    for read in &reads {
        assert_eq!(rows(&session, read), 0, "in the transaction: {read}");
        assert_eq!(rows(&db.session(), read), 1, "another session: {read}");
    }
    session.commit().unwrap();
    for read in &reads {
        assert_eq!(rows(&db.session(), read), 0, "after the commit: {read}");
    }
}

/// A count in a transaction counts its own writes, as a scan reads them.
#[test]
fn a_transaction_counts_its_own_rdf_writes() {
    let db = rdf_db();
    db.session()
        .execute_sparql(
            "INSERT DATA { <http://example.org/mia> <http://example.org/knows> <http://example.org/jules> }",
        )
        .unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute_sparql(ALIX_KNOWS_GUS).unwrap();
    let count = |session: &grafeo_engine::session::Session, query: &str| {
        session.execute_sparql(query).unwrap().rows()[0][0].clone()
    };
    assert_eq!(
        count(&session, "SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }"),
        integer(2)
    );
    assert_eq!(
        count(
            &session,
            "SELECT (COUNT(*) AS ?n) WHERE { ?s <http://example.org/knows> ?o }"
        ),
        integer(2)
    );
    assert_eq!(
        count(&db.session(), "SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }"),
        integer(1),
        "another session counts the committed triple only"
    );
}

/// An `xsd:integer` as a SPARQL result holds it.
fn integer(n: i64) -> grafeo_common::types::Value {
    grafeo_common::types::Value::RdfTerm(std::sync::Arc::new(
        grafeo_common::types::rdf::Term::Literal(grafeo_common::types::rdf::Literal::integer(n)),
    ))
}

#[test]
fn a_transaction_reads_its_own_rdf_writes_and_others_do_not() {
    let db = rdf_db();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute_sparql(ALIX_KNOWS_GUS).unwrap();
    let ask =
        "ASK { <http://example.org/alix> <http://example.org/knows> <http://example.org/gus> }";
    assert!(asks(&session, ask), "the transaction reads its own insert");
    assert!(
        !asks(&db.session(), ask),
        "another session does not see it before the commit"
    );
    session
        .execute_sparql(
            "DELETE DATA { <http://example.org/alix> <http://example.org/knows> <http://example.org/gus> }",
        )
        .unwrap();
    assert!(!asks(&session, ask), "the transaction reads its own delete");
    session.commit().unwrap();
    assert_eq!(subjects(&db), Vec::<String>::new());
}

/// The triples of the named graph `graph` as `[subject]`, sorted.
fn subjects_in(db: &GrafeoDB, graph: &str) -> Vec<String> {
    let mut subjects: Vec<String> = db
        .rdf_store()
        .graph_latest(graph)
        .map(|graph| {
            graph
                .triples(&grafeo_core::graph::rdf::RdfView::latest(
                    grafeo_core::graph::rdf::LatestUse::Test,
                ))
                .iter()
                .map(|triple| triple.subject().to_string())
                .collect()
        })
        .unwrap_or_default();
    subjects.sort();
    subjects
}

const PARIS: &str = "http://example.org/paris";
const BERLIN: &str = "http://example.org/berlin";

/// Inserts a triple into the named graph `graph` in `session`.
fn insert_into(session: &grafeo_engine::session::Session, graph: &str, subject: &str) {
    session
        .execute_sparql(&format!(
            "INSERT DATA {{ GRAPH <{graph}> {{ <http://example.org/{subject}>              <http://example.org/knows> <http://example.org/gus> }} }}"
        ))
        .unwrap();
}

/// A graph operation and an open transaction's triple writes in the same graph do not conflict
/// (#718, item 8): outside a transaction the operation runs and commits at once, and the open
/// writer's triples apply at its later commit, after it. So the end state is the serial order of
/// the two commits: the operation, then the writer's insert of `alix` into Paris. A copy, an add
/// or a move reads its source as committed at its own statement, without the writer's triple.
#[test]
fn a_graph_operation_runs_beside_open_changes_in_the_graphs_it_changes() {
    let alix = "<http://example.org/alix>";
    let jules = "<http://example.org/jules>";
    let mia = "<http://example.org/mia>";
    for (operation, paris, berlin) in [
        (format!("CLEAR GRAPH <{PARIS}>"), vec![alix], vec![mia]),
        (format!("DROP GRAPH <{PARIS}>"), vec![alix], vec![mia]),
        ("CLEAR ALL".to_string(), vec![alix], vec![]),
        ("DROP ALL".to_string(), vec![alix], vec![]),
        (
            format!("COPY <{BERLIN}> TO <{PARIS}>"),
            vec![alix, mia],
            vec![mia],
        ),
        (
            format!("ADD <{BERLIN}> TO <{PARIS}>"),
            vec![alix, jules, mia],
            vec![mia],
        ),
        (
            format!("MOVE <{PARIS}> TO <{BERLIN}>"),
            vec![alix],
            vec![jules],
        ),
        (
            format!("COPY <{PARIS}> TO <{BERLIN}>"),
            vec![alix, jules],
            vec![jules],
        ),
    ] {
        let db = rdf_db();
        insert_into(&db.session(), BERLIN, "mia");
        insert_into(&db.session(), PARIS, "jules");
        let mut writer = db.session();
        writer.begin_transaction().unwrap();
        insert_into(&writer, PARIS, "alix");

        db.session()
            .execute_sparql(&operation)
            .unwrap_or_else(|error| panic!("{operation} beside an open insert: {error}"));
        writer
            .commit()
            .unwrap_or_else(|error| panic!("{operation}, then the writer's commit: {error}"));
        assert_eq!(subjects_in(&db, PARIS), paris, "{operation}: Paris");
        assert_eq!(subjects_in(&db, BERLIN), berlin, "{operation}: Berlin");
    }
}

/// In a transaction a graph operation is recorded for the commit: against another transaction's
/// open triple changes in its graph both commit (#718, item 8), in the serial order of their
/// commits. The clear commits first and takes Jules; the writer's Alix comes after it.
#[test]
fn a_queued_graph_operation_and_open_changes_both_commit() {
    let db = rdf_db();
    insert_into(&db.session(), PARIS, "jules");
    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    insert_into(&writer, PARIS, "alix");

    let mut other = db.session();
    other.begin_transaction().unwrap();
    other
        .execute_sparql(&format!("CLEAR GRAPH <{PARIS}>"))
        .unwrap();
    other
        .commit()
        .unwrap_or_else(|error| panic!("the clear commits: {error}"));
    writer
        .commit()
        .unwrap_or_else(|error| panic!("the insert commits after it: {error}"));
    assert_eq!(
        subjects_in(&db, PARIS),
        ["<http://example.org/alix>"],
        "cleared, then Alix"
    );
}

/// Against its own open changes it applies after them, in the order they ran.
#[test]
fn a_queued_graph_operation_applies_after_its_own_triples() {
    let db = rdf_db();
    insert_into(&db.session(), PARIS, "jules");
    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    insert_into(&writer, PARIS, "alix");
    writer
        .execute_sparql(&format!("CLEAR GRAPH <{PARIS}>"))
        .unwrap();
    assert_eq!(
        subjects_in(&db, PARIS),
        ["<http://example.org/jules>"],
        "another reader before the commit"
    );
    writer.commit().unwrap();
    assert_eq!(
        subjects_in(&db, PARIS),
        Vec::<String>::new(),
        "cleared after both"
    );
}

/// A graph operation queues for the commit in a transaction: another reader does
/// not see it, and a rollback undoes it with the transaction's triples (#727).
#[test]
fn a_graph_operation_in_a_transaction_goes_with_its_rollback() {
    let db = rdf_db();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute_sparql(&format!("CREATE GRAPH <{PARIS}>"))
        .unwrap();
    assert!(
        db.rdf_store().graph_latest(PARIS).is_none(),
        "another reader does not see the graph before the commit"
    );
    session.rollback().unwrap();
    assert!(
        db.rdf_store().graph_latest(PARIS).is_none(),
        "the rollback undoes it"
    );
}

/// A rolled-back insert leaves no graph behind.
#[test]
fn a_rolled_back_insert_leaves_no_graph() {
    let db = rdf_db();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    insert_into(&session, BERLIN, "vincent");
    session.rollback().unwrap();
    assert!(db.rdf_store().graph_latest(BERLIN).is_none());
}

/// `DROP ALL` empties the default graph and removes every named graph.
#[test]
fn drop_all_drops_every_graph() {
    let db = rdf_db();
    let session = db.session();
    session.execute_sparql(ALIX_KNOWS_GUS).unwrap();
    insert_into(&session, PARIS, "jules");
    session.execute_sparql("DROP ALL").unwrap();
    assert_eq!(subjects(&db), Vec::<String>::new());
    assert_eq!(
        db.rdf_store()
            .graph_names(&grafeo_core::graph::rdf::RdfView::latest(
                grafeo_core::graph::rdf::LatestUse::Test
            )),
        Vec::<String>::new()
    );
}

#[cfg(feature = "cdc")]
mod change_feed {
    use grafeo_common::types::EpochId;
    use grafeo_engine::GrafeoDB;
    use grafeo_engine::cdc::ChangeEvent;

    use super::{ALIX_KNOWS_GUS, rdf_config};

    fn rdf_db_with_cdc() -> GrafeoDB {
        GrafeoDB::with_config(rdf_config().with_cdc()).unwrap()
    }

    fn triple_events(db: &GrafeoDB) -> Vec<ChangeEvent> {
        db.changes_between(EpochId::new(0), db.current_epoch())
            .unwrap()
            .into_iter()
            .filter(|event| event.triple_subject.is_some())
            .collect()
    }

    #[test]
    fn a_rolled_back_rdf_update_emits_no_cdc_events() {
        let db = rdf_db_with_cdc();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session.execute_sparql(ALIX_KNOWS_GUS).unwrap();
        session.rollback().unwrap();
        assert!(triple_events(&db).is_empty(), "{:?}", triple_events(&db));
    }

    /// #414: a committed update starts an epoch of its own, and its events
    /// carry it, so a reader that has seen every epoch before it misses
    /// nothing.
    #[test]
    fn a_committed_rdf_update_advances_the_epoch_and_its_events_carry_it() {
        let db = rdf_db_with_cdc();
        let before = db.current_epoch();
        db.session().execute_sparql(ALIX_KNOWS_GUS).unwrap();
        let after = db.current_epoch();
        assert!(after > before, "{before:?} to {after:?}");
        let events = triple_events(&db);
        assert_eq!(events.len(), 1, "{events:?}");
        assert_eq!(events[0].epoch, after);
        assert!(
            db.changes_between(EpochId::new(0), before)
                .unwrap()
                .is_empty(),
            "nothing new in epochs a reader has seen"
        );
    }

    /// Two transactions insert the same triple: the second commit changes
    /// nothing, and the change feed reports the insert once.
    #[test]
    fn a_triple_two_transactions_insert_is_reported_once() {
        let db = rdf_db_with_cdc();
        let mut first = db.session();
        let mut second = db.session();
        first.begin_transaction().unwrap();
        second.begin_transaction().unwrap();
        first.execute_sparql(ALIX_KNOWS_GUS).unwrap();
        second.execute_sparql(ALIX_KNOWS_GUS).unwrap();
        first.commit().unwrap();
        second.commit().unwrap();
        let events = triple_events(&db);
        assert_eq!(events.len(), 1, "{events:?}");
        assert_eq!(db.rdf_store().len(), 1);
    }

    /// An insert of a triple the store holds already, and a delete of one it
    /// does not hold, change nothing and report nothing.
    #[test]
    fn rdf_updates_that_change_nothing_report_nothing() {
        let db = rdf_db_with_cdc();
        let session = db.session();
        session.execute_sparql(ALIX_KNOWS_GUS).unwrap();
        session.execute_sparql(ALIX_KNOWS_GUS).unwrap();
        session
            .execute_sparql(
                "DELETE DATA { <http://example.org/mia> <http://example.org/knows> <http://example.org/jules> }",
            )
            .unwrap();
        assert_eq!(triple_events(&db).len(), 1, "{:?}", triple_events(&db));
    }
}
