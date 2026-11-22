//! Triple ids in change events (#679) are the same in every process and release: the first 8
//! bytes of SHA-256 over the documented canonical encoding of the triple's terms and its graph's
//! storage key (golden values beside `triple_id` in the engine). The two public helpers,
//! `CdcLog::record_triple_insert` and `record_triple_delete`, parse their terms and use the same
//! id rule as a commit, and terms that do not parse record nothing (a warning names why). `"x"` and
//! `"x"^^xsd:string` are one RDF term and so one id. A new graph may not have an empty name, and
//! one an older release made (a 0.6.1 file) opens, reads and keeps its own ids, apart from the
//! default graph's by the graph's tag byte.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main cdc_triple_ids::
//! ```

#![cfg(all(
    feature = "cdc",
    feature = "sparql",
    feature = "triple-store",
    feature = "gql"
))]

use grafeo_common::testing::child_process;
use grafeo_common::types::EpochId;
use grafeo_engine::cdc::{CdcLog, ChangeEvent, EntityId};
use grafeo_engine::{Config, GrafeoDB};

const ALIX: &str = "<http://ex.org/alix>";
const KNOWS: &str = "<http://ex.org/knows>";
const GUS: &str = "<http://ex.org/gus>";
/// `<alix> <knows> <gus>` in the default graph, computed from the documented encoding.
const ALIX_KNOWS_GUS: u64 = 8_198_305_858_944_992_339;
/// `_:b1 <name> "Alix"` in the default graph.
const BLANK_NAMED_ALIX: u64 = 358_890_899_194_566_529;

fn ids(events: &[ChangeEvent]) -> Vec<u64> {
    events
        .iter()
        .map(|event| match event.entity_id {
            EntityId::Triple(id) => id,
            other => panic!("not a triple: {other:?}"),
        })
        .collect()
}

/// The id the insert helper gives `subject predicate object` in `graph`.
fn helper_id(subject: &str, predicate: &str, object: &str, graph: Option<&str>) -> u64 {
    let log = CdcLog::new();
    log.record_triple_insert(subject, predicate, object, graph, EpochId::new(1));
    let [id] = ids(&log.changes_between(EpochId::new(0), EpochId::new(2)))[..] else {
        panic!("one event for {subject} {predicate} {object}")
    };
    id
}

#[test]
fn a_helpers_id_is_the_golden_one() {
    assert_eq!(helper_id(ALIX, KNOWS, GUS, None), ALIX_KNOWS_GUS);
    assert_eq!(
        helper_id("_:b1", "<http://ex.org/name>", "\"Alix\"", None),
        BLANK_NAMED_ALIX
    );
    let log = CdcLog::new();
    log.record_triple_delete(ALIX, KNOWS, GUS, None, EpochId::new(1));
    assert_eq!(
        ids(&log.changes_between(EpochId::new(0), EpochId::new(2))),
        [ALIX_KNOWS_GUS],
        "insert and delete of one triple carry one id"
    );
}

/// RDF 1.1: a plain literal is an `xsd:string`, so the two spellings are one term and one id.
#[test]
fn a_plain_literal_and_its_xsd_string_are_one_id() {
    let plain = helper_id("_:b1", "<http://ex.org/name>", "\"Alix\"", None);
    let typed = helper_id(
        "_:b1",
        "<http://ex.org/name>",
        "\"Alix\"^^<http://www.w3.org/2001/XMLSchema#string>",
        None,
    );
    assert_eq!(plain, typed);
    assert_ne!(
        plain,
        helper_id("_:b1", "<http://ex.org/name>", "\"Alix\"@nl", None)
    );
}

/// A line that does not parse leaves the log exactly as it was: the events it held, and the clock,
/// so the next event's timestamp is the one it would have had.
#[test]
fn a_line_that_does_not_parse_leaves_the_log_as_it_was() {
    let log = CdcLog::new();
    log.record_triple_insert(ALIX, KNOWS, GUS, None, EpochId::new(1));
    let before = log.changes_between(EpochId::new(0), EpochId::new(9));
    let clock = log.clock().peek();
    log.record_triple_insert(ALIX, KNOWS, "\"unterminated", None, EpochId::new(2));
    log.record_triple_delete("", KNOWS, GUS, None, EpochId::new(2));
    assert_eq!(
        serde_json::to_value(log.changes_between(EpochId::new(0), EpochId::new(9))).unwrap(),
        serde_json::to_value(&before).unwrap(),
        "the events are as they were"
    );
    assert_eq!(log.clock().peek(), clock, "no timestamp was taken");
}

/// Terms that are not one N-Triples triple record nothing (the helper logs a warning naming why);
/// the helpers keep their signature, returning nothing, in 0.6.
#[test]
fn a_term_that_does_not_parse_records_nothing() {
    let log = CdcLog::new();
    for (subject, predicate, object) in [
        ("<http://ex.org/alix", KNOWS, GUS),
        (ALIX, "knows", GUS),
        (ALIX, KNOWS, "\"unterminated"),
        ("", KNOWS, GUS),
        (
            ALIX,
            KNOWS,
            "<http://ex.org/gus> .\n<http://ex.org/mia> <http://ex.org/knows> <http://ex.org/jules>",
        ),
    ] {
        let () = log.record_triple_insert(subject, predicate, object, None, EpochId::new(1));
        let () = log.record_triple_delete(subject, predicate, object, None, EpochId::new(1));
    }
    assert!(
        log.changes_between(EpochId::new(0), EpochId::new(2))
            .is_empty(),
        "terms that do not parse record no event"
    );
    assert_eq!(log.dropped_lines(), 10, "each refused call is counted");
}

/// A commit's event and the helper give one id for one triple, in the default graph and in a
/// named one, and the named graph's id differs from the default's.
#[test]
fn a_commit_and_a_helper_give_one_id() {
    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
    db.execute_sparql(&format!("INSERT DATA {{ {ALIX} {KNOWS} {GUS} }}"))
        .unwrap();
    db.execute_sparql(&format!(
        "INSERT DATA {{ GRAPH <http://ex.org/people> {{ {ALIX} {KNOWS} {GUS} }} }}"
    ))
    .unwrap();
    let events = db
        .changes_between(EpochId::new(0), db.current_epoch())
        .unwrap();
    let default = events
        .iter()
        .find(|e| e.graph.is_none())
        .expect("default graph event");
    let named = events
        .iter()
        .find(|e| e.graph.is_some())
        .expect("named graph event");
    assert_eq!(ids(std::slice::from_ref(default)), [ALIX_KNOWS_GUS]);
    let key = named.graph.as_deref().unwrap();
    assert_eq!(
        ids(std::slice::from_ref(named)),
        [helper_id(ALIX, KNOWS, GUS, Some(key))]
    );
    assert_ne!(ids(std::slice::from_ref(named)), [ALIX_KNOWS_GUS]);
}

/// Another process gives the same id: the child asserts the golden value.
#[test]
fn another_process_gives_the_same_id() {
    let mut child = grafeo_common::child_test!("triple_id_child");
    child.env("GRAFEO_TRIPLE_ID_CHILD", "1");
    let status = child_process::run(&mut child).unwrap();
    assert!(status.success(), "the child's id differs");
}

/// Child-process entry for [`another_process_gives_the_same_id`]; a no-op when run directly.
#[test]
fn triple_id_child() {
    if std::env::var("GRAFEO_TRIPLE_ID_CHILD").is_err() {
        return;
    }
    assert_eq!(helper_id(ALIX, KNOWS, GUS, None), ALIX_KNOWS_GUS);
}

/// A new graph may not have an empty name, in GQL or in SPARQL (an empty IRI with no base).
#[test]
fn a_new_graph_may_not_have_an_empty_name() {
    let db = GrafeoDB::new_in_memory();
    let gql = db.execute("CREATE GRAPH ``").unwrap_err().to_string();
    assert!(gql.contains(EMPTY_GQL_NAME), "GQL CREATE GRAPH ``: {gql}");
    for statement in [
        "CREATE GRAPH <>".to_string(),
        format!("INSERT DATA {{ GRAPH <> {{ {ALIX} {KNOWS} {GUS} }} }}"),
    ] {
        let error = db.execute_sparql(&statement).unwrap_err().to_string();
        assert!(error.contains(EMPTY_RDF_NAME), "{statement}: {error}");
    }
}

/// One error for an empty graph name at every entry point: GQL, SPARQL, SQL/PGQ and the direct
/// API give the same code and message.
#[test]
fn every_entry_point_gives_one_error_for_an_empty_name() {
    let db = GrafeoDB::new_in_memory();
    let gql = db.execute("CREATE GRAPH ``").unwrap_err();
    let mut others = vec![
        ("create_graph", db.create_graph("").unwrap_err()),
        (
            "GQL IF NOT EXISTS",
            db.execute("CREATE GRAPH IF NOT EXISTS ``").unwrap_err(),
        ),
        (
            "SPARQL CREATE",
            db.execute_sparql("CREATE GRAPH <>").unwrap_err(),
        ),
        (
            "SPARQL INSERT DATA",
            db.execute_sparql(&format!(
                "INSERT DATA {{ GRAPH <> {{ {ALIX} {KNOWS} {GUS} }} }}"
            ))
            .unwrap_err(),
        ),
    ];
    #[cfg(feature = "sql-pgq")]
    others.push((
        "SQL/PGQ",
        db.session()
            .execute_sql(r#"CREATE PROPERTY GRAPH "" NODE TABLES (Item (sku VARCHAR PRIMARY KEY))"#)
            .unwrap_err(),
    ));
    for (what, error) in others {
        assert_eq!(error.error_code(), gql.error_code(), "{what}: {error}");
        assert_eq!(error.to_string(), gql.to_string(), "{what}");
    }
    assert!(db.list_graphs().is_empty(), "no graph was created");
}

/// SPARQL's `SILENT` turns the refusal into success with no effect: `CREATE SILENT GRAPH <>`
/// creates nothing, and `COPY SILENT`, `MOVE SILENT` and `ADD SILENT` into `<>` write nothing.
#[test]
fn sparql_silent_into_an_unnamed_graph_does_nothing() {
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql(&format!("INSERT DATA {{ {ALIX} {KNOWS} {GUS} }}"))
        .unwrap();
    for statement in [
        "CREATE SILENT GRAPH <>",
        "COPY SILENT DEFAULT TO <>",
        "MOVE SILENT DEFAULT TO <>",
        "ADD SILENT DEFAULT TO <>",
    ] {
        db.execute_sparql(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
    let named = db
        .execute_sparql("SELECT ?g WHERE { GRAPH ?g { ?s ?p ?o } }")
        .unwrap();
    assert!(named.rows().is_empty(), "no graph was created or written");
    let default = db.execute_sparql("SELECT ?s WHERE { ?s ?p ?o }").unwrap();
    assert_eq!(default.rows().len(), 1, "the default graph is untouched");
}

/// The refusal of an empty GQL graph name, as GQL, SQL/PGQ and the direct API word it.
const EMPTY_GQL_NAME: &str = "A graph name must not be empty";
/// The refusal of a write that would create the named RDF graph `<>`: the same error.
const EMPTY_RDF_NAME: &str = EMPTY_GQL_NAME;

/// SQL/PGQ's `CREATE PROPERTY GRAPH` refuses an empty name as GQL does, and creates nothing.
#[cfg(feature = "sql-pgq")]
#[test]
fn sql_pgq_may_not_create_a_property_graph_with_an_empty_name() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let error = session
        .execute_sql(r#"CREATE PROPERTY GRAPH "" NODE TABLES (Item (sku VARCHAR PRIMARY KEY))"#)
        .unwrap_err();
    assert!(error.to_string().contains(EMPTY_GQL_NAME), "{error}");
    assert_eq!(
        error.error_code(),
        db.execute("CREATE GRAPH ``").unwrap_err().error_code()
    );
    assert!(db.list_graphs().is_empty(), "no graph was created");
}

/// A transaction's `COPY DEFAULT TO <>` is refused also when the default graph is empty (it
/// would create `<>` with nothing in it), and the commit leaves no such graph.
#[test]
fn a_transactions_copy_of_an_empty_default_graph_into_an_unnamed_graph_is_refused() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let error = session
        .execute_sparql("COPY DEFAULT TO <>")
        .unwrap_err()
        .to_string();
    assert!(error.contains(EMPTY_RDF_NAME), "{error}");
    session.commit().unwrap();
    let named = db
        .execute_sparql("SELECT ?g WHERE { GRAPH ?g { ?s ?p ?o } }")
        .unwrap();
    assert!(named.rows().is_empty(), "the commit left no graph `<>`");
}

/// A blank argument of a helper is refused, records nothing, though the three together would
/// parse as one triple once shifted (`<a> <b>`, `<c>`, ``).
#[test]
fn a_blank_helper_argument_records_nothing() {
    let log = CdcLog::new();
    for (subject, predicate, object) in [
        ("<http://ex.org/alix> <http://ex.org/knows>", GUS, ""),
        ("", "<http://ex.org/alix> <http://ex.org/knows>", GUS),
        (ALIX, " ", "<http://ex.org/knows> <http://ex.org/gus>"),
    ] {
        log.record_triple_insert(subject, predicate, object, None, EpochId::new(1));
        log.record_triple_delete(subject, predicate, object, None, EpochId::new(1));
    }
    assert!(
        log.changes_between(EpochId::new(0), EpochId::new(2))
            .is_empty(),
        "blank arguments record no event"
    );
    assert_eq!(log.dropped_lines(), 6, "each refused call is counted");
}

/// An escaped term: the helper's event equals a commit's, terms and id alike.
#[test]
fn an_escaped_term_gives_the_terms_and_the_id_of_a_commit() {
    const SAYS: &str = "<http://ex.org/says>";
    const QUOTED: &str = r#""say \"hallo\"\n""#;
    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
    db.execute_sparql(&format!("INSERT DATA {{ {ALIX} {SAYS} {QUOTED} }}"))
        .unwrap();
    let committed = db
        .changes_between(EpochId::new(0), db.current_epoch())
        .unwrap();
    let log = CdcLog::new();
    log.record_triple_insert(ALIX, SAYS, QUOTED, None, EpochId::new(1));
    let recorded = log.changes_between(EpochId::new(0), EpochId::new(2));
    let ([committed], [recorded]) = (&committed[..], &recorded[..]) else {
        panic!("one event each: {committed:?} and {recorded:?}")
    };
    assert_eq!(terms_and_id(recorded), terms_and_id(committed));
}

/// A commit into the legacy `<>` graph of a 0.6.1 file reports the graph `Some("")` and the id of
/// that graph, apart from the default graph's.
#[test]
fn a_commit_into_the_legacy_unnamed_graph_names_it() {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/released/0.6.1/empty-named-graph.grafeo");
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty-named-graph.grafeo");
    std::fs::copy(&fixture, &path).unwrap();
    // A triple the fixture's `<>` does not hold, so the insert is a write.
    const MIA: &str = "<http://ex.org/mia>";
    const JULES: &str = "<http://ex.org/jules>";
    let db = GrafeoDB::with_config(Config::persistent(&path).with_cdc()).unwrap();
    db.execute_sparql(&format!(
        "INSERT DATA {{ GRAPH <> {{ {MIA} {KNOWS} {JULES} }} }}"
    ))
    .unwrap();
    let events = db
        .changes_between(EpochId::new(0), db.current_epoch())
        .unwrap();
    let [event] = &events[..] else {
        panic!("one event: {events:?}")
    };
    assert_eq!(event.graph.as_deref(), Some(""));
    assert_eq!(
        ids(std::slice::from_ref(event)),
        [helper_id(MIA, KNOWS, JULES, Some(""))]
    );
    assert_ne!(
        ids(std::slice::from_ref(event)),
        [helper_id(MIA, KNOWS, JULES, None)]
    );
    db.close().unwrap();
}

/// The direct API refuses an empty graph name with the error GQL gives, and creates nothing.
#[test]
fn the_direct_api_may_not_create_a_graph_with_an_empty_name() {
    use grafeo_common::utils::error::Error;

    let db = GrafeoDB::new_in_memory();
    let refused = db.create_graph("").unwrap_err();
    let gql = db.execute("CREATE GRAPH ``").unwrap_err();
    let kind = |error: &Error| match error {
        Error::Query(query) => Some(query.kind),
        _ => None,
    };
    assert!(kind(&refused).is_some(), "{refused:?}");
    assert_eq!(kind(&refused), kind(&gql), "{refused} against {gql}");
    assert_eq!(refused.error_code(), gql.error_code());
    assert!(db.list_graphs().is_empty(), "no graph was created");
}

/// A 0.6.1 file holding an empty-named graph (made with the 0.6.1 wheel: two `:Person` nodes in
/// the GQL graph `` and one triple in the SPARQL graph `<>`, beside one in the default graph)
/// opens, and both graphs read; the triple in it has another id than the default graph's.
#[test]
fn a_released_file_with_an_empty_named_graph_opens_and_reads() {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/released/0.6.1/empty-named-graph.grafeo");
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty-named-graph.grafeo");
    std::fs::copy(&fixture, &path).unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    let people = db
        .execute("USE GRAPH `` MATCH (p:Person) RETURN p.name AS name")
        .unwrap();
    assert_eq!(people.rows().len(), 2, "the GQL graph `` reads");
    let triples = db
        .execute_sparql("SELECT ?s WHERE { GRAPH ?g { ?s ?p ?o } }")
        .unwrap();
    assert_eq!(triples.rows().len(), 1, "the SPARQL graph <> reads");
    assert_ne!(
        helper_id(ALIX, KNOWS, GUS, Some("")),
        ALIX_KNOWS_GUS,
        "its triple's id differs from the default graph's"
    );
    db.close().unwrap();
}

/// The terms an event carries and its id: what a consumer keys on.
fn terms_and_id(event: &ChangeEvent) -> (Option<&str>, Option<&str>, Option<&str>, u64) {
    (
        event.triple_subject.as_deref(),
        event.triple_predicate.as_deref(),
        event.triple_object.as_deref(),
        ids(std::slice::from_ref(event))[0],
    )
}

/// A helper given a term in another spelling of the same RDF term (an upper-case language tag,
/// an explicit `xsd:string`) records the event a commit of the normalised triple records: the
/// same id and the same terms, so a consumer keying on the terms sees one triple.
#[test]
fn a_helpers_event_carries_the_normalised_terms_of_a_commit() {
    const GREETS: &str = "<http://ex.org/greets>";
    for (spelled, normalised) in [
        ("\"hallo\"@NL", "\"hallo\"@nl"),
        (
            "\"Alix\"^^<http://www.w3.org/2001/XMLSchema#string>",
            "\"Alix\"",
        ),
    ] {
        let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
        db.execute_sparql(&format!("INSERT DATA {{ {ALIX} {GREETS} {normalised} }}"))
            .unwrap();
        let committed = db
            .changes_between(EpochId::new(0), db.current_epoch())
            .unwrap();
        let [committed] = &committed[..] else {
            panic!("one event for the commit of {normalised}: {committed:?}")
        };

        let log = CdcLog::new();
        log.record_triple_insert(ALIX, GREETS, spelled, None, EpochId::new(1));
        let recorded = log.changes_between(EpochId::new(0), EpochId::new(2));
        let [recorded] = &recorded[..] else {
            panic!("one event for the helper's {spelled}")
        };
        assert_eq!(
            terms_and_id(recorded),
            terms_and_id(committed),
            "{spelled} against a commit of {normalised}"
        );
    }
}

/// `COPY`, `MOVE` and `ADD` into `<>` would create a graph with an empty name: refused, and no
/// such graph appears.
#[test]
fn copy_move_and_add_into_an_unnamed_graph_are_refused() {
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql(&format!("INSERT DATA {{ {ALIX} {KNOWS} {GUS} }}"))
        .unwrap();
    for statement in [
        "COPY DEFAULT TO <>",
        "MOVE DEFAULT TO <>",
        "ADD DEFAULT TO <>",
    ] {
        let error = db.execute_sparql(statement).unwrap_err().to_string();
        assert!(error.contains(EMPTY_RDF_NAME), "{statement}: {error}");
    }
    let named = db
        .execute_sparql("SELECT ?g WHERE { GRAPH ?g { ?s ?p ?o } }")
        .unwrap();
    assert!(named.rows().is_empty(), "no graph was created");
    let default = db.execute_sparql("SELECT ?s WHERE { ?s ?p ?o }").unwrap();
    assert_eq!(default.rows().len(), 1, "the default graph is untouched");
}

/// The empty-named graphs a 0.6.1 file holds stay writable: a GQL insert into `` and a SPARQL
/// insert into `<>` succeed and read back after a reopen.
#[test]
fn a_released_empty_named_graph_stays_writable() {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/released/0.6.1/empty-named-graph.grafeo");
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty-named-graph.grafeo");
    std::fs::copy(&fixture, &path).unwrap();
    let mia = "<http://ex.org/mia> <http://ex.org/knows> <http://ex.org/jules>";

    let db = GrafeoDB::open(&path).unwrap();
    db.execute("USE GRAPH `` INSERT (:Person {name: 'Mia'})")
        .unwrap();
    db.execute_sparql(&format!("INSERT DATA {{ GRAPH <> {{ {mia} }} }}"))
        .unwrap();
    db.close().unwrap();
    drop(db);

    let db = GrafeoDB::open(&path).unwrap();
    let people = db
        .execute("USE GRAPH `` MATCH (p:Person) RETURN p.name AS name")
        .unwrap();
    assert_eq!(people.rows().len(), 3, "the GQL graph `` took the insert");
    let triples = db
        .execute_sparql("SELECT ?g WHERE { GRAPH ?g { <http://ex.org/mia> ?p ?o } }")
        .unwrap();
    assert_eq!(
        triples.rows().len(),
        1,
        "the SPARQL graph <> took the insert"
    );
    db.close().unwrap();
}
