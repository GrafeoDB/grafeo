//! A SPARQL read sees one snapshot: the triples and graphs committed when its transaction (or its
//! statement, outside one) began, plus the transaction's own writes, whatever commits while it
//! runs. Two transactions that write one triple, where one write is a delete, conflict (`T001`), as
//! do two graph operations on one graph; writes that change nothing record nothing (#718).
//!
//! The oracle, unless a test says otherwise: the answer a fresh session gives before the
//! concurrent commit (the snapshot's answer), and for own writes and conflicts the state of a
//! serial order, computed by running the same statements one by one on a database of its own.
//!
//! - every read form (`SELECT`, `ASK`, `CONSTRUCT`, `DESCRIBE`, `COUNT`, paths, `EXISTS`,
//!   `OPTIONAL`, `MINUS`, `GRAPH <g>`, `GRAPH ?g`, `FROM`, `FROM NAMED`, the graph list), in
//!   read-only and read-write transactions, read before the commit or first after it;
//! - graphs created, dropped and cleared after the snapshot (named, `DEFAULT`, `NAMED`, `ALL`),
//!   and loads that bypass a session (`load_rdf_text`, `batch_insert_rdf`);
//! - the change map: a triple untouched when a reader's map was built and deleted later, and a
//!   triple dropped with its graph, inserted again and deleted (conditions 2 and 3);
//! - one state per statement outside a transaction, LPG and SPARQL at one snapshot, and property
//!   paths whose nested lookups run while commits wait;
//! - own writes and own graph operations, in order, with a savepoint, read by another session
//!   before and after, and replayed from the log after a crash;
//! - the conflict rules, in both commit orders, on exact RDF terms;
//! - a checkpoint during an open reader, and `rdf_schema()`'s named graphs.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main sparql_snapshot_isolation::
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use std::fmt::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use grafeo_common::utils::error::{Error, ErrorCode};
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

const PREFIXES: &str =
    "PREFIX ex: <http://ex.org/> PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> ";

/// `text` with the `ex:` and `xsd:` prefixes declared.
fn q(text: &str) -> String {
    format!("{PREFIXES}{text}")
}

/// The seed, in the default graph: Alix and Gus with a value each, Alix knows Gus knows Mia (a
/// path). Graph g holds Vincent, graph h Shosanna.
const SEED: &str = r#"INSERT DATA {
    ex:alix ex:p "1" . ex:gus ex:p "2" .
    ex:alix ex:knows ex:gus . ex:gus ex:knows ex:mia .
    GRAPH ex:g { ex:vincent ex:p "3" . }
    GRAPH ex:h { ex:shosanna ex:p "7" . }
}"#;

fn seeded() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql(&q(SEED)).unwrap();
    db
}

/// Sorted rows of `query` (prefixes added), each as its debug text, or the error's text.
fn try_answer(session: &Session, query: &str) -> Result<Vec<String>, String> {
    let result = session
        .execute_sparql(&q(query))
        .map_err(|error| format!("{query}: {error}"))?;
    let mut rows: Vec<String> = result.rows().iter().map(|row| format!("{row:?}")).collect();
    rows.sort();
    Ok(rows)
}

/// [`try_answer`], panicking on an error.
fn answer(session: &Session, query: &str) -> Vec<String> {
    try_answer(session, query).unwrap_or_else(|error| panic!("{error}"))
}

/// Every triple of every graph, with its graph (unbound for the default graph).
const ALL: &str = "SELECT ?g ?s ?p ?o WHERE { { ?s ?p ?o } UNION { GRAPH ?g { ?s ?p ?o } } }";

/// The read forms. Each one's answer changes with [`change`], so a form that still reads the
/// latest state shows.
fn forms() -> Vec<(&'static str, &'static str)> {
    vec![
        ("select", "SELECT ?s ?o WHERE { ?s ex:p ?o }"),
        (
            "select in g",
            "SELECT ?s ?o WHERE { GRAPH ex:g { ?s ex:p ?o } }",
        ),
        (
            "join with a point lookup",
            "SELECT ?x ?o WHERE { ?s ex:p ?o . ?s ex:knows ?x }",
        ),
        ("ask", r#"ASK { ex:alix ex:p "1" }"#),
        ("ask in g", r#"ASK { GRAPH ex:g { ex:vincent ex:p "3" } }"#),
        (
            "count with a bound predicate",
            "SELECT (COUNT(*) AS ?c) WHERE { ?s ex:p ?o }",
        ),
        (
            "count of the default graph",
            "SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o }",
        ),
        (
            "count in g",
            "SELECT (COUNT(*) AS ?c) WHERE { GRAPH ex:g { ?s ?p ?o } }",
        ),
        ("construct", "CONSTRUCT { ?s ex:p ?o } WHERE { ?s ex:p ?o }"),
        (
            "construct in g",
            "CONSTRUCT { ?s ex:p ?o } WHERE { GRAPH ex:g { ?s ex:p ?o } }",
        ),
        ("describe", "DESCRIBE ex:alix"),
        ("path p+", "SELECT ?o WHERE { ex:alix ex:knows+ ?o }"),
        ("path p*", "SELECT ?o WHERE { ex:alix ex:knows* ?o }"),
        ("path ^p", "SELECT ?s WHERE { ex:mia ^ex:knows ?s }"),
        ("path p|q", "SELECT ?o WHERE { ex:gus (ex:knows|ex:p) ?o }"),
        (
            "exists",
            "SELECT ?s WHERE { ?s ex:p ?o FILTER EXISTS { ?s ex:knows ?x } }",
        ),
        (
            "not exists",
            "SELECT ?s WHERE { ?s ex:p ?o FILTER NOT EXISTS { ?s ex:knows ?x } }",
        ),
        (
            "optional",
            "SELECT ?s ?x WHERE { ?s ex:p ?o OPTIONAL { ?s ex:knows ?x } }",
        ),
        (
            "minus",
            "SELECT ?s WHERE { ?s ex:p ?o MINUS { ?s ex:knows ?x } }",
        ),
        (
            "a graph created after the snapshot",
            "SELECT ?s ?o WHERE { GRAPH ex:k { ?s ?p ?o } }",
        ),
        (
            "a graph dropped after the snapshot",
            "SELECT ?s ?o WHERE { GRAPH ex:h { ?s ?p ?o } }",
        ),
        ("from", "SELECT ?s ?o FROM ex:g WHERE { ?s ?p ?o }"),
        (
            "from named, listed",
            "SELECT ?g ?s FROM NAMED ex:g FROM NAMED ex:h FROM NAMED ex:k \
             WHERE { GRAPH ?g { ?s ?p ?o } }",
        ),
        (
            "graph list",
            "SELECT DISTINCT ?g WHERE { GRAPH ?g { ?s ?p ?o } }",
        ),
        ("everything", ALL),
    ]
}

/// One commit of another session that deletes Alix's "1", the step from Gus to Mia and Vincent's
/// "3" in g, and inserts Mia and Jules in the default graph and in g and a step from Gus to Butch.
fn change_triples(db: &GrafeoDB) {
    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer
        .execute_sparql(&q(
            r#"DELETE DATA { ex:alix ex:p "1" . ex:gus ex:knows ex:mia .
                GRAPH ex:g { ex:vincent ex:p "3" . } }"#,
        ))
        .unwrap();
    writer
        .execute_sparql(&q(
            r#"INSERT DATA { ex:mia ex:p "4" . ex:jules ex:p "5" . ex:gus ex:knows ex:butch .
                GRAPH ex:g { ex:mia ex:p "4" . ex:jules ex:p "5" . } }"#,
        ))
        .unwrap();
    writer.commit().unwrap();
}

/// Three commits of another session that change every form's answer: [`change_triples`], a drop
/// of graph h and the creation of graph k (an insert into it).
fn change(db: &GrafeoDB) {
    change_triples(db);
    db.execute_sparql(&q("DROP GRAPH ex:h")).unwrap();
    db.execute_sparql(&q(r#"INSERT DATA { GRAPH ex:k { ex:butch ex:p "6" . } }"#))
        .unwrap();
}

/// How a reader's transaction begins.
#[derive(Clone, Copy, Debug)]
enum Mode {
    ReadOnly,
    ReadWrite,
}

fn begin(session: &mut Session, mode: Mode) {
    match mode {
        Mode::ReadOnly => {
            session.execute("START TRANSACTION READ ONLY").unwrap();
        }
        Mode::ReadWrite => session.begin_transaction().unwrap(),
    }
}

/// A `GRAFEO-T001` conflict.
fn is_conflict(error: &Error) -> bool {
    error.error_code() == ErrorCode::TransactionConflict
}

/// A strict expected failure: `body` must panic (the behavior `spec` names is not there yet,
/// `issue` tracks it); if it passes, the test fails so the mark is removed.
fn xfail(spec: &str, issue: &str, body: impl FnOnce()) {
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)).is_err();
    assert!(
        failed,
        "XPASS: {spec} ({issue}) holds now; remove the xfail"
    );
}

// ------------------------------------------------------------------ snapshot reads (B)

/// The issue's reproduction, as written.
#[test]
fn the_issue_reproduction() {
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql(&q(
        r#"INSERT DATA { ex:alix ex:name "Alix" . ex:gus ex:name "Gus" . }"#,
    ))
    .unwrap();
    let names = "SELECT ?n WHERE { ?s ex:name ?n }";
    let mut a = db.session();
    a.begin_transaction().unwrap();
    let before = answer(&a, names);
    assert_eq!(before.len(), 2, "{before:?}");
    let mut b = db.session();
    b.begin_transaction().unwrap();
    b.execute_sparql(&q(r#"INSERT DATA { ex:vincent ex:name "Vincent" . }"#))
        .unwrap();
    b.execute_sparql(&q(r#"DELETE DATA { ex:gus ex:name "Gus" . }"#))
        .unwrap();
    b.commit().unwrap();
    assert_eq!(answer(&a, names), before, "Alix and Gus, as at the start");
}

/// Every form reads its transaction's snapshot after another session's commits, in a read-only
/// and a read-write transaction, whether it read before the commits or reads first after them;
/// after its own commit it reads the new state.
#[test]
fn every_read_form_reads_the_transactions_snapshot() {
    let mut wrong = Vec::new();
    for mode in [Mode::ReadOnly, Mode::ReadWrite] {
        for read_first in [true, false] {
            for (name, query) in forms() {
                let db = seeded();
                let snapshot = answer(&db.session(), query);
                let mut reader = db.session();
                begin(&mut reader, mode);
                if read_first {
                    let read = answer(&reader, query);
                    assert_eq!(
                        read, snapshot,
                        "{name}: a transaction reads what a session does"
                    );
                }
                change(&db);
                let during = answer(&reader, query);
                let when = if read_first {
                    "read again"
                } else {
                    "read first"
                };
                if during != snapshot {
                    wrong.push(format!(
                        "{mode:?} {name}, {when} after commits: {during:?}, the snapshot holds {snapshot:?}"
                    ));
                }
                reader.commit().unwrap();
                let after = answer(&reader, query);
                if after == snapshot {
                    wrong.push(format!(
                        "{mode:?} {name}: the commits change nothing this form reads: {after:?}"
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// Graph operations of another session while a reader is open: the reader keeps the graphs, their
/// contents and the graph list as they were.
#[test]
fn a_reader_keeps_the_graphs_another_session_drops_or_clears() {
    let mut wrong = Vec::new();
    for operation in [
        "DROP GRAPH ex:g",
        "CLEAR GRAPH ex:g",
        "CLEAR DEFAULT",
        "CLEAR NAMED",
        "CLEAR ALL",
        "DROP NAMED",
        "DROP ALL",
    ] {
        let db = seeded();
        let mut reader = db.session();
        reader.begin_transaction().unwrap();
        let read = |reader: &Session| -> Vec<Vec<String>> {
            forms()
                .iter()
                .map(|(_, query)| answer(reader, query))
                .collect()
        };
        let before = read(&reader);
        db.execute_sparql(&q(operation)).unwrap();
        if read(&reader) != before {
            wrong.push(format!("{operation}: the reader's graphs changed"));
        }
        reader.commit().unwrap();
        if read(&reader) == before {
            wrong.push(format!("{operation}: changed nothing a reader sees"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// Loads that bypass a session (a text load into the default graph and into a named one, a batch
/// insert) are invisible to a reader that began before them.
#[test]
fn loads_after_the_snapshot_are_invisible_to_a_reader() {
    use grafeo_core::graph::rdf::{Term, Triple};
    use grafeo_engine::RdfFormat;
    let db = seeded();
    let mut reader = db.session();
    reader.begin_transaction().unwrap();
    let before = answer(&reader, ALL);
    let turtle = r#"<http://ex.org/django> <http://ex.org/p> "8" ."#;
    db.load_rdf_text(turtle, RdfFormat::Turtle, None).unwrap();
    db.load_rdf_text(turtle, RdfFormat::Turtle, Some("http://ex.org/k"))
        .unwrap();
    db.batch_insert_rdf([Triple::new(
        Term::iri("http://ex.org/hans"),
        Term::iri("http://ex.org/p"),
        Term::literal("9"),
    )])
    .unwrap();
    assert_eq!(answer(&reader, ALL), before);
    reader.commit().unwrap();
    assert_eq!(answer(&reader, ALL).len(), before.len() + 3);
}

/// Condition 2 of the change map: a reader whose map was built by a read after one commit still
/// reads a triple that a later commit deletes, one that was untouched when the map was built.
#[test]
fn a_triple_deleted_after_the_readers_map_was_built_stays_visible() {
    let db = seeded();
    let mut reader = db.session();
    reader.begin_transaction().unwrap();
    let alix = r#"ASK { ex:alix ex:p "1" }"#;
    let gus = r#"ASK { ex:gus ex:p "2" }"#;
    let (alix_at_s, gus_at_s) = (answer(&reader, alix), answer(&reader, gus));
    db.execute_sparql(&q(r#"DELETE DATA { ex:gus ex:p "2" . }"#))
        .unwrap();
    // Builds the map: Gus's triple has a change after the snapshot, Alix's has none yet.
    assert_eq!(answer(&reader, gus), gus_at_s, "Gus at the snapshot");
    assert_eq!(
        answer(&reader, alix),
        alix_at_s,
        "Alix before the second commit"
    );
    db.execute_sparql(&q(r#"DELETE DATA { ex:alix ex:p "1" . }"#))
        .unwrap();
    assert_eq!(
        answer(&reader, alix),
        alix_at_s,
        "Alix after the second commit"
    );
    assert_eq!(
        answer(&reader, gus),
        gus_at_s,
        "Gus after the second commit"
    );
    assert_eq!(answer(&reader, ALL), answer(&seeded().session(), ALL));
}

/// Condition 3 of the change map: a triple of g present at the snapshot, dropped with g, inserted
/// again into the new g and then deleted, is still read as present, once, whether the reader read
/// before, between or after the commits.
#[test]
fn a_triple_dropped_with_its_graph_reinserted_and_deleted_stays_visible() {
    let in_g = "SELECT ?s ?o WHERE { GRAPH ex:g { ?s ?p ?o } }";
    let commits = [
        "DROP GRAPH ex:g",
        r#"INSERT DATA { GRAPH ex:g { ex:vincent ex:p "3" . ex:jules ex:p "5" . } }"#,
        r#"DELETE DATA { GRAPH ex:g { ex:vincent ex:p "3" . } }"#,
    ];
    for read_at in 0..=commits.len() {
        let db = seeded();
        let snapshot = answer(&db.session(), in_g);
        let mut reader = db.session();
        reader.begin_transaction().unwrap();
        for (step, commit) in commits.iter().enumerate() {
            if step == read_at {
                assert_eq!(answer(&reader, in_g), snapshot, "read before commit {step}");
            }
            db.execute_sparql(&q(commit)).unwrap();
        }
        assert_eq!(
            answer(&reader, in_g),
            snapshot,
            "first read before commit {read_at}"
        );
    }
}

/// A single query outside a transaction reads one state while other sessions commit: each commit
/// moves Alix's value from n to n + 1 and Vincent's in g likewise, so every read holds Alix, Gus
/// and Vincent once each.
#[test]
fn a_single_query_reads_one_state_while_commits_run() {
    let db = Arc::new(seeded());
    let stop = Arc::new(AtomicBool::new(false));
    let writer = {
        let (db, stop) = (Arc::clone(&db), Arc::clone(&stop));
        std::thread::spawn(move || {
            let mut session = db.session();
            for n in 1..400 {
                session.begin_transaction().unwrap();
                session
                    .execute_sparql(&q(&format!(
                        r#"DELETE DATA {{ ex:alix ex:p "{n}" . GRAPH ex:g {{ ex:vincent ex:p "{m}" . }} }}"#,
                        m = n + 2,
                    )))
                    .unwrap();
                session
                    .execute_sparql(&q(&format!(
                        r#"INSERT DATA {{ ex:alix ex:p "{a}" . GRAPH ex:g {{ ex:vincent ex:p "{b}" . }} }}"#,
                        a = n + 1,
                        b = n + 3,
                    )))
                    .unwrap();
                session.commit().unwrap();
            }
            stop.store(true, Ordering::SeqCst);
        })
    };
    let all = "SELECT ?s ?o WHERE { { ?s ex:p ?o } UNION { GRAPH ex:g { ?s ex:p ?o } } }";
    let mut reads = 0;
    let mut torn = Vec::new();
    let session = db.session();
    while !stop.load(Ordering::SeqCst) {
        let rows = answer(&session, all);
        reads += 1;
        if rows.len() != 3 && torn.len() < 5 {
            torn.push(rows);
        }
    }
    writer.join().unwrap();
    assert!(reads > 0);
    assert!(
        torn.is_empty(),
        "{} of {reads} reads saw a torn state: {torn:?}",
        torn.len()
    );
}

/// A transaction that reads the property graph and the RDF store sees one snapshot of both, in GQL
/// and in Cypher.
#[test]
fn lpg_and_sparql_reads_share_one_snapshot() {
    let db = seeded();
    db.execute("INSERT (:City {name: 'Amsterdam'})").unwrap();
    let mut reader = db.session();
    reader.begin_transaction().unwrap();
    let cities = |s: &Session| {
        let gql = s
            .execute("MATCH (c:City) RETURN count(c) AS n")
            .unwrap()
            .rows()[0][0]
            .clone();
        let cypher = s
            .execute_cypher("MATCH (c:City) RETURN count(c) AS n")
            .unwrap()
            .rows()[0][0]
            .clone();
        (gql, cypher)
    };
    let triples = |s: &Session| answer(s, ALL);
    let before = (cities(&reader), triples(&reader));
    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer.execute("INSERT (:City {name: 'Berlin'})").unwrap();
    writer
        .execute_sparql(&q(r#"INSERT DATA { ex:mia ex:p "4" . }"#))
        .unwrap();
    writer.commit().unwrap();
    assert_eq!((cities(&reader), triples(&reader)), before);
    reader.commit().unwrap();
    assert_ne!(cities(&reader), before.0);
    assert_ne!(triples(&reader), before.1);
}

/// Property paths over a long chain, whose lookups nest, while another session commits in a loop:
/// every path and every commit finishes within the deadline (a reader's nested lookup must not wait
/// behind a commit that waits for the reader), and each path reads its snapshot's chain.
#[test]
fn paths_finish_while_commits_wait_for_their_lookups() {
    const CHAIN: usize = 2_000;
    let db = Arc::new(GrafeoDB::new_in_memory());
    let mut steps = String::new();
    for i in 0..CHAIN {
        write!(steps, "ex:n{i} ex:next ex:n{} . ", i + 1).unwrap();
    }
    let chain = format!("INSERT DATA {{ {steps} }}");
    db.execute_sparql(&q(&chain)).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let (done_tx, done_rx) = mpsc::channel::<&'static str>();
    let committer = {
        let (db, stop, done) = (Arc::clone(&db), Arc::clone(&stop), done_tx.clone());
        std::thread::spawn(move || {
            let mut n = 0;
            while !stop.load(Ordering::SeqCst) {
                db.execute_sparql(&q(&format!(r#"INSERT DATA {{ ex:gus ex:p "{n}" . }}"#)))
                    .unwrap();
                db.execute_sparql(&q(&format!(r#"DELETE DATA {{ ex:gus ex:p "{n}" . }}"#)))
                    .unwrap();
                n += 1;
            }
            done.send("committer").unwrap();
        })
    };
    let reader = {
        let (db, stop, done) = (Arc::clone(&db), Arc::clone(&stop), done_tx);
        std::thread::spawn(move || {
            let session = db.session();
            let mut lengths = Vec::new();
            for _ in 0..20 {
                lengths.push(answer(&session, "SELECT ?o WHERE { ex:n0 ex:next+ ?o }").len());
            }
            stop.store(true, Ordering::SeqCst);
            done.send("reader").unwrap();
            lengths
        })
    };
    for _ in 0..2 {
        let finished = done_rx.recv_timeout(Duration::from_secs(120));
        assert!(
            finished.is_ok(),
            "a path or a commit did not finish within two minutes: a deadlock"
        );
    }
    committer.join().unwrap();
    let lengths = reader.join().unwrap();
    assert!(lengths.iter().all(|&n| n == CHAIN), "{lengths:?}");
}

// ------------------------------------------------------------------ own writes

/// A transaction's own writes over its snapshot follow the serial model: insert then delete leaves
/// nothing, delete then insert leaves the triple once, a triple inserted by the transaction and by
/// another session's commit is read once, and a savepoint rollback takes back what followed it.
#[test]
fn own_writes_follow_the_serial_model_over_the_snapshot() {
    let count =
        |s: &Session, triple: &str| answer(s, &format!("SELECT * WHERE {{ {triple} }}")).len();
    let jules = r#"ex:jules ex:p "5""#;
    let alix = r#"ex:alix ex:p "1""#;
    let mia = r#"ex:mia ex:p "4""#;
    let butch = r#"ex:butch ex:p "6""#;
    let db = seeded();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute_sparql(&q(&format!("INSERT DATA {{ {jules} . }}")))
        .unwrap();
    session
        .execute_sparql(&q(&format!("DELETE DATA {{ {jules} . }}")))
        .unwrap();
    assert_eq!(count(&session, jules), 0, "insert then delete");
    session
        .execute_sparql(&q(&format!("DELETE DATA {{ {alix} . }}")))
        .unwrap();
    session
        .execute_sparql(&q(&format!("INSERT DATA {{ {alix} . }}")))
        .unwrap();
    assert_eq!(count(&session, alix), 1, "delete then insert");
    session
        .execute_sparql(&q(&format!("INSERT DATA {{ {mia} . }}")))
        .unwrap();
    db.execute_sparql(&q(&format!("INSERT DATA {{ {mia} . }}")))
        .unwrap();
    assert_eq!(
        count(&session, mia),
        1,
        "inserted here and by another commit"
    );
    session.savepoint("before_butch").unwrap();
    session
        .execute_sparql(&q(&format!("INSERT DATA {{ {butch} . }}")))
        .unwrap();
    session.rollback_to_savepoint("before_butch").unwrap();
    assert_eq!(count(&session, butch), 0, "rolled back to the savepoint");
    session.commit().unwrap();
    let after = db.session();
    assert_eq!(
        (
            count(&after, jules),
            count(&after, alix),
            count(&after, mia),
            count(&after, butch)
        ),
        (0, 1, 1, 0)
    );
}

/// Runs `statements` one by one as auto-commit statements on a seeded database of its own: the
/// serial state the same statements in one transaction must reach.
fn serial(statements: &[&str]) -> Vec<String> {
    let db = seeded();
    for statement in statements {
        db.execute_sparql(&q(statement))
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
    answer(&db.session(), ALL)
}

#[test]
fn own_graph_operations_apply_in_order_and_are_seen() {
    // Comes off with #718 D2.
    xfail(
        "SPARQL Update in a transaction: its graph operations apply in order with its triple writes, and its later reads see them",
        "#718",
        own_graph_operations_apply_in_order_and_are_seen_holds,
    );
}

/// A transaction's own graph operations run in order with its triple writes, and its later reads
/// (the graph list and paths included) see them: inside the transaction and after its commit the
/// state is the serial one.
fn own_graph_operations_apply_in_order_and_are_seen_holds() {
    let cases: [&[&str]; 7] = [
        &[
            "DROP GRAPH ex:g",
            r#"INSERT DATA { GRAPH ex:g { ex:jules ex:p "5" . } }"#,
        ],
        &[
            "CLEAR GRAPH ex:g",
            r#"INSERT DATA { GRAPH ex:g { ex:jules ex:p "5" . } }"#,
        ],
        &[
            r#"INSERT DATA { GRAPH ex:g { ex:jules ex:p "5" . } }"#,
            "CLEAR GRAPH ex:g",
        ],
        &["CLEAR DEFAULT", r#"INSERT DATA { ex:jules ex:p "5" . }"#],
        &[
            "DROP ALL",
            r#"INSERT DATA { GRAPH ex:g { ex:jules ex:p "5" . } }"#,
        ],
        &[
            "CREATE GRAPH ex:k",
            r#"INSERT DATA { GRAPH ex:k { ex:jules ex:knows ex:butch . ex:butch ex:knows ex:mia . } }"#,
        ],
        &[
            "DROP GRAPH ex:h",
            "CLEAR NAMED",
            r#"INSERT DATA { GRAPH ex:h { ex:jules ex:p "5" . } }"#,
        ],
    ];
    let graphs = "SELECT DISTINCT ?g WHERE { GRAPH ?g { ?s ?p ?o } }";
    let path = "SELECT ?o WHERE { GRAPH ex:k { ex:jules ex:knows+ ?o } }";
    let mut wrong = Vec::new();
    for statements in cases {
        let name = statements.join("; ");
        let want = serial(statements);
        let db = seeded();
        let want_graphs = {
            let other = seeded();
            for statement in statements {
                other.execute_sparql(&q(statement)).unwrap();
            }
            (
                answer(&other.session(), graphs),
                answer(&other.session(), path),
            )
        };
        let mut session = db.session();
        session.begin_transaction().unwrap();
        let ran = statements
            .iter()
            .try_for_each(|statement| session.execute_sparql(&q(statement)).map(|_| ()));
        if let Err(error) = ran {
            wrong.push(format!("{name}: {error}"));
            continue;
        }
        let inside = try_answer(&session, ALL);
        if inside.as_ref() != Ok(&want) {
            wrong.push(format!(
                "{name}: inside the transaction, {inside:?}, not the serial state"
            ));
        }
        let seen = try_answer(&session, graphs)
            .and_then(|graphs| try_answer(&session, path).map(|path| (graphs, path)));
        if seen.as_ref() != Ok(&want_graphs) {
            wrong.push(format!(
                "{name}: the graph list or a path inside the transaction: {seen:?}"
            ));
        }
        if let Err(error) = session.commit() {
            wrong.push(format!("{name}: the commit: {error}"));
            continue;
        }
        if answer(&db.session(), ALL) != want {
            wrong.push(format!("{name}: after the commit, not the serial state"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// A savepoint rollback takes back a graph operation that followed it.
#[test]
fn a_savepoint_rollback_takes_back_a_graph_operation() {
    let insert = r#"INSERT DATA { GRAPH ex:g { ex:jules ex:p "5" . } }"#;
    let db = seeded();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute_sparql(&q(insert)).unwrap();
    session.savepoint("before_the_drop").unwrap();
    session.execute_sparql(&q("DROP GRAPH ex:g")).unwrap();
    session.rollback_to_savepoint("before_the_drop").unwrap();
    assert_eq!(answer(&session, ALL), serial(&[insert]));
    session.commit().unwrap();
    assert_eq!(answer(&db.session(), ALL), serial(&[insert]));
}

#[test]
fn a_drop_and_insert_commit_reads_old_before_and_new_after() {
    // Comes off with #718 D2.
    xfail(
        "snapshot isolation: a reader that began before a DROP GRAPH and INSERT commit reads the old graph",
        "#718",
        a_drop_and_insert_commit_reads_old_before_and_new_after_holds,
    );
}

/// `DROP GRAPH g; INSERT DATA { GRAPH g ... }` committed by another transaction: a reader that
/// began before reads the old g, one that begins after reads the insert only.
fn a_drop_and_insert_commit_reads_old_before_and_new_after_holds() {
    let statements = [
        "DROP GRAPH ex:g",
        r#"INSERT DATA { GRAPH ex:g { ex:jules ex:p "5" . } }"#,
    ];
    let db = seeded();
    let mut before = db.session();
    before.begin_transaction().unwrap();
    let at_start = answer(&before, ALL);
    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    for statement in statements {
        writer.execute_sparql(&q(statement)).unwrap();
    }
    writer.commit().unwrap();
    assert_eq!(
        answer(&before, ALL),
        at_start,
        "the reader that began before"
    );
    assert_eq!(
        answer(&db.session(), ALL),
        serial(&statements),
        "a reader after"
    );
}

// ------------------------------------------------------------------ own graph operations after a crash (D)

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
mod replay {
    use std::path::{Path, PathBuf};

    use grafeo_common::testing::child_process;
    use grafeo_engine::config::StorageFormat;
    use grafeo_engine::{Config, GrafeoDB};

    use super::{ALL, SEED, answer, q, serial, xfail};

    const SCENARIO_VAR: &str = "GRAFEO_SPARQL_SNAPSHOT_REPLAY_SCENARIO";
    const PATH_VAR: &str = "GRAFEO_SPARQL_SNAPSHOT_REPLAY_PATH";

    /// One transaction each: a graph operation between triple writes, in its WAL group.
    const SCENARIOS: [&[&str]; 3] = [
        &[
            "DROP GRAPH ex:g",
            r#"INSERT DATA { GRAPH ex:g { ex:jules ex:p "5" . } }"#,
        ],
        &[
            r#"INSERT DATA { GRAPH ex:g { ex:jules ex:p "5" . } }"#,
            "CLEAR GRAPH ex:g",
        ],
        &["CLEAR DEFAULT", r#"INSERT DATA { ex:jules ex:p "5" . }"#],
    ];

    fn open(path: &Path) -> GrafeoDB {
        GrafeoDB::with_config(Config::persistent(path).with_storage_format(StorageFormat::Auto))
            .unwrap()
    }

    fn sidecar_wal(path: &Path) -> PathBuf {
        let mut sidecar = path.as_os_str().to_owned();
        sidecar.push(".wal");
        PathBuf::from(sidecar)
    }

    #[test]
    fn a_transactions_graph_operations_replay_in_order_after_a_crash() {
        // Comes off with #718 D1.
        xfail(
            "WAL replay: a transaction's graph operations replay in order with its triple writes",
            "#718",
            a_transactions_graph_operations_replay_in_order_after_a_crash_holds,
        );
    }

    /// A child seeds a database file, commits one scenario's transaction and exits without
    /// closing; the reopened database replays the log to the serial state.
    fn a_transactions_graph_operations_replay_in_order_after_a_crash_holds() {
        let mut wrong = Vec::new();
        for (number, statements) in SCENARIOS.iter().enumerate() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("graphs.grafeo");
            let output = child_process::output_with_deadline(
                grafeo_common::child_test!("replay_child")
                    .env(SCENARIO_VAR, number.to_string())
                    .env(PATH_VAR, &path),
                std::time::Duration::from_secs(60),
            )
            .unwrap();
            if !output.status.success() {
                wrong.push(format!(
                    "{}: the child failed: {}",
                    statements.join("; "),
                    [&output.stdout, &output.stderr]
                        .iter()
                        .flat_map(|bytes| String::from_utf8_lossy(bytes)
                            .lines()
                            .map(str::to_string)
                            .collect::<Vec<_>>())
                        .find(|line| line.contains("Err") || line.contains("panicked"))
                        .unwrap_or_default()
                ));
                continue;
            }
            assert!(sidecar_wal(&path).exists(), "the crash left the WAL");
            let reopened = open(&path);
            if answer(&reopened.session(), ALL) != serial(statements) {
                wrong.push(statements.join("; "));
            }
        }
        assert!(
            wrong.is_empty(),
            "replayed out of order: {}",
            wrong.join("\n")
        );
    }

    /// Child-process entry for the test above; a no-op when run directly.
    #[test]
    fn replay_child() {
        let Ok(number) = std::env::var(SCENARIO_VAR) else {
            return;
        };
        let path = PathBuf::from(std::env::var_os(PATH_VAR).unwrap());
        let db = open(&path);
        db.execute_sparql(&q(SEED)).unwrap();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        for statement in SCENARIOS[number.parse::<usize>().unwrap()] {
            session.execute_sparql(&q(statement)).unwrap();
        }
        session.commit().unwrap();
        // Exit without closing: no checkpoint, the WAL stays.
        std::process::exit(0);
    }
}

// ------------------------------------------------------------------ conflicts (C, D)

/// The order two racing transactions commit in.
#[derive(Clone, Copy, Debug)]
enum Order {
    /// The second runs its statements while the first is still open, then both commit.
    FirstOpen,
    /// The first runs and commits before the second runs its statements.
    FirstCommitted,
}

/// How the second of two racing transactions ends.
#[derive(Debug, PartialEq)]
enum Outcome {
    /// It committed.
    Committed,
    /// It failed with a `T001` conflict, at a statement or at its commit.
    Conflict,
    /// Anything else, with what failed.
    Failed(String),
}

/// Two transactions begin at one snapshot, run `first` and `second` and commit in that order. A
/// second that failed at its commit has no open transaction left.
fn race(db: &GrafeoDB, order: Order, first: &[&str], second: &[&str]) -> Outcome {
    let run = |session: &Session, statements: &[&str]| -> grafeo_common::utils::error::Result<()> {
        statements
            .iter()
            .try_for_each(|statement| session.execute_sparql(&q(statement)).map(|_| ()))
    };
    let ended = |error: Error| {
        if is_conflict(&error) {
            Outcome::Conflict
        } else {
            Outcome::Failed(error.to_string())
        }
    };
    let mut a = db.session();
    let mut b = db.session();
    a.begin_transaction().unwrap();
    b.begin_transaction().unwrap();
    if let Err(error) = run(&a, first).and_then(|()| match order {
        Order::FirstOpen => Ok(()),
        Order::FirstCommitted => a.commit(),
    }) {
        return Outcome::Failed(format!("the first transaction: {error}"));
    }
    let second_ran = run(&b, second);
    if let Order::FirstOpen = order
        && let Err(error) = a.commit()
    {
        return Outcome::Failed(format!("the first transaction's commit: {error}"));
    }
    match second_ran {
        Ok(()) => match b.commit() {
            Ok(()) => Outcome::Committed,
            Err(error) => {
                assert!(
                    !b.in_transaction(),
                    "a failed commit leaves no open transaction"
                );
                ended(error)
            }
        },
        Err(error) => {
            if b.in_transaction() {
                b.rollback().unwrap();
            }
            ended(error)
        }
    }
}

/// One case of two racing transactions: its name, the first's and the second's statements, and
/// how the second ends.
type Case<'a> = (&'a str, Vec<&'a str>, Vec<&'a str>, Outcome);

/// Each case in both orders, on a fresh database: the ones that end otherwise.
fn wrong_outcomes(cases: &[Case<'_>]) -> Vec<String> {
    let mut wrong = Vec::new();
    for order in [Order::FirstOpen, Order::FirstCommitted] {
        for (name, first, second, want) in cases {
            let got = race(&seeded(), order, first, second);
            if got != *want {
                wrong.push(format!("{order:?} {name}: {got:?}, want {want:?}"));
            }
        }
    }
    wrong
}

/// The triple rules (C): a recorded delete conflicts with a committed write of the same triple;
/// inserts of one triple, a delete of a triple the snapshot does not hold and writes of different
/// triples (one subject, or one triple in two graphs) do not. Each case in both orders.
#[test]
fn triple_conflicts_follow_what_each_transaction_changed() {
    let delete_alix = r#"DELETE DATA { ex:alix ex:p "1" . }"#;
    let fresh = r#"INSERT DATA { ex:hans ex:p "new" . }"#;
    let cases: Vec<Case<'_>> = vec![
        (
            "two deletes of one triple",
            vec![delete_alix],
            vec![delete_alix],
            Outcome::Conflict,
        ),
        (
            "a replace twice",
            vec![delete_alix, r#"INSERT DATA { ex:alix ex:p "10" . }"#],
            vec![delete_alix, r#"INSERT DATA { ex:alix ex:p "11" . }"#],
            Outcome::Conflict,
        ),
        (
            "a delete against a committed delete and insert of the triple",
            vec![
                r#"DELETE DATA { ex:gus ex:p "2" . }"#,
                r#"INSERT DATA { ex:gus ex:p "2" . }"#,
            ],
            vec![r#"DELETE DATA { ex:gus ex:p "2" . }"#],
            Outcome::Conflict,
        ),
        (
            "a no-op insert against a delete of the triple",
            vec![delete_alix],
            vec![r#"INSERT DATA { ex:alix ex:p "1" . }"#],
            Outcome::Committed,
        ),
        (
            "two inserts of one triple",
            vec![r#"INSERT DATA { ex:mia ex:p "4" . }"#],
            vec![r#"INSERT DATA { ex:mia ex:p "4" . }"#],
            Outcome::Committed,
        ),
        (
            "a no-op delete against an insert of the triple",
            vec![r#"INSERT DATA { ex:jules ex:p "5" . }"#],
            vec![r#"DELETE DATA { ex:jules ex:p "5" . }"#],
            Outcome::Committed,
        ),
        (
            "a fresh triple inserted, against one inserted then deleted",
            vec![fresh],
            vec![fresh, r#"DELETE DATA { ex:hans ex:p "new" . }"#],
            Outcome::Conflict,
        ),
        (
            "different triples of one subject",
            vec![delete_alix],
            vec![r#"DELETE DATA { ex:alix ex:knows ex:gus . }"#],
            Outcome::Committed,
        ),
        (
            "one triple in two graphs",
            vec![delete_alix],
            vec![r#"INSERT DATA { GRAPH ex:g { ex:alix ex:p "1" . } }"#],
            Outcome::Committed,
        ),
    ];
    let wrong = wrong_outcomes(&cases);
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn graph_key_conflicts_follow_what_each_transaction_changed() {
    // Comes off with #718 D2.
    xfail(
        "snapshot isolation: a graph operation conflicts with a committed one of the same graph",
        "#718",
        graph_key_conflicts_follow_what_each_transaction_changed_holds,
    );
}

/// The graph-key rules (D): a graph operation conflicts with a committed one of the same graph;
/// operations on different graphs and a graph operation against a triple write in that graph do
/// not. Each case in both orders.
fn graph_key_conflicts_follow_what_each_transaction_changed_holds() {
    let in_g = r#"INSERT DATA { GRAPH ex:g { ex:jules ex:p "5" . } }"#;
    let in_g_too = r#"INSERT DATA { GRAPH ex:g { ex:butch ex:p "6" . } }"#;
    let delete_in_g = r#"DELETE DATA { GRAPH ex:g { ex:vincent ex:p "3" . } }"#;
    let cases: Vec<Case<'_>> = vec![
        (
            "two DROP g then an insert into g",
            vec!["DROP GRAPH ex:g", in_g],
            vec!["DROP GRAPH ex:g", in_g_too],
            Outcome::Conflict,
        ),
        (
            "two CLEAR g then an insert into g",
            vec!["CLEAR GRAPH ex:g", in_g],
            vec!["CLEAR GRAPH ex:g", in_g_too],
            Outcome::Conflict,
        ),
        (
            "two DROP g",
            vec!["DROP GRAPH ex:g"],
            vec!["DROP GRAPH ex:g"],
            Outcome::Conflict,
        ),
        (
            "CLEAR g against DROP g",
            vec!["DROP GRAPH ex:g"],
            vec!["CLEAR GRAPH ex:g"],
            Outcome::Conflict,
        ),
        (
            "two CREATE n",
            vec!["CREATE GRAPH ex:n"],
            vec!["CREATE SILENT GRAPH ex:n"],
            Outcome::Conflict,
        ),
        (
            "graph operations on different graphs",
            vec!["CLEAR GRAPH ex:g"],
            vec!["DROP GRAPH ex:h"],
            Outcome::Committed,
        ),
        (
            "CLEAR g against a triple delete in g",
            vec!["CLEAR GRAPH ex:g"],
            vec![delete_in_g],
            Outcome::Committed,
        ),
        (
            "a triple delete in g against CLEAR g",
            vec![delete_in_g],
            vec!["CLEAR GRAPH ex:g"],
            Outcome::Committed,
        ),
    ];
    let wrong = wrong_outcomes(&cases);
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// A transaction that begins while another's commit is written but not yet published reads below
/// that commit, alone or not: a delete of the triple the commit deletes conflicts, when it commits
/// inside that window (the other's writes are still in its change set) and when it commits after
/// the other is published (the other kept them for it). The conflict may come at the delete or at
/// the commit, as in `race`; the first commit stands either way.
#[cfg(feature = "testing-statement-injection")]
#[test]
fn a_transaction_begun_during_a_commit_conflicts_with_it() {
    use grafeo_common::testing::commit_hook::after_next_commit_written;
    use std::cell::RefCell;
    use std::rc::Rc;

    let delete = q(r#"DELETE DATA { ex:alix ex:p "1" . }"#);
    let alix = "SELECT ?o WHERE { ex:alix ex:p ?o }";
    // Inside the window, on another thread: a commit that wrongly passes the check would wait
    // for the first one's publish, which waits for this hook, so the receive times out.
    let db = Arc::new(seeded());
    let inside = Rc::new(RefCell::new(None));
    {
        let (db, inside, delete) = (Arc::clone(&db), Rc::clone(&inside), delete.clone());
        after_next_commit_written(move || {
            let (sent, outcome) = mpsc::channel();
            std::thread::spawn(move || {
                let mut late = db.session();
                late.begin_transaction().unwrap();
                let ended = late
                    .execute_sparql(&delete)
                    .map(|_| ())
                    .and_then(|()| late.commit());
                let _ = sent.send(ended.map_err(|error| is_conflict(&error)));
            });
            *inside.borrow_mut() = Some(outcome.recv_timeout(Duration::from_secs(10)));
        });
    }
    let mut first = db.session();
    first.begin_transaction().unwrap();
    first.execute_sparql(&delete).unwrap();
    first.commit().unwrap();
    assert_eq!(
        inside.borrow_mut().take(),
        Some(Ok(Err(true))),
        "a commit inside the window conflicts"
    );
    assert!(
        answer(&db.session(), alix).is_empty(),
        "the first commit stands"
    );

    // Begun inside the window, committed after the first one is published. The hook runs on the
    // committing thread, so a statement error is kept for after the commit, never unwrapped there.
    let db = Arc::new(seeded());
    let late = Rc::new(RefCell::new(None));
    {
        let (db, late, delete) = (Arc::clone(&db), Rc::clone(&late), delete.clone());
        after_next_commit_written(move || {
            let mut session = db.session();
            session.begin_transaction().unwrap();
            let ran = session.execute_sparql(&delete).map(|_| ());
            *late.borrow_mut() = Some((session, ran));
        });
    }
    let mut first = db.session();
    first.begin_transaction().unwrap();
    first.execute_sparql(&delete).unwrap();
    first.commit().unwrap();
    let (mut late, ran) = late.borrow_mut().take().expect("the hook ran");
    let error = ran
        .and_then(|()| late.commit())
        .expect_err("the delete lost to the published one");
    assert!(is_conflict(&error), "{error}");
    assert!(
        answer(&db.session(), alix).is_empty(),
        "the first commit stands"
    );
}

/// The writes a commit keeps for the transactions begun before it conflict with none begun after
/// it: with an older reader open (so the deleting commit keeps its writes), a transaction begun
/// after the delete is published puts the triple back and commits.
///
/// An unrelated commit lands while that transaction is open, so its commit runs the RDF conflict
/// check (it is skipped when nothing committed since the transaction began) and walks the
/// deleting commit's kept writes, whose epoch is the transaction's start: only a strict
/// `commit_epoch > start` passes them by.
#[test]
fn a_transaction_begun_after_a_commit_does_not_conflict_with_it() {
    let db = seeded();
    let alix = "SELECT ?o WHERE { ex:alix ex:p ?o }";
    let mut reader = db.session();
    reader.begin_transaction().unwrap();
    let mut first = db.session();
    first.begin_transaction().unwrap();
    first
        .execute_sparql(&q(r#"DELETE DATA { ex:alix ex:p "1" . }"#))
        .unwrap();
    first.commit().unwrap();

    let mut after = db.session();
    after.begin_transaction().unwrap();
    after
        .execute_sparql(&q(r#"INSERT DATA { ex:alix ex:p "1" . }"#))
        .unwrap();
    db.session()
        .execute_sparql(&q(r#"INSERT DATA { ex:butch ex:p "9" . }"#))
        .unwrap();
    after.commit().unwrap();
    assert_eq!(answer(&db.session(), alix).len(), 1, "the triple is back");
    assert_eq!(
        answer(&db.session(), "SELECT ?o WHERE { ex:butch ex:p ?o }").len(),
        1,
        "the unrelated commit stands"
    );
    reader.rollback().unwrap();
}

/// A no-op delete changes nothing and its transaction commits; the concurrent insert stays.
#[test]
fn a_no_op_delete_leaves_a_concurrent_insert() {
    let db = seeded();
    let jules = r#"ex:jules ex:p "5""#;
    assert_eq!(
        race(
            &db,
            Order::FirstOpen,
            &[&format!("INSERT DATA {{ {jules} . }}")],
            &[&format!("DELETE DATA {{ {jules} . }}")]
        ),
        Outcome::Committed
    );
    assert_eq!(
        answer(&db.session(), &format!("ASK {{ {jules} }}")),
        answer(&db.session(), "ASK { ex:gus ex:p \"2\" }")
    );
}

/// Conflicts key on the exact RDF term: two deletes of one term conflict, deletes of two terms
/// that differ only in kind, language, datatype or lexical form do not.
#[test]
fn conflicts_key_on_the_exact_rdf_term() {
    let terms = [
        ("plain 1", r#""1""#),
        ("integer 1", r#""1"^^xsd:integer"#),
        ("integer 01", r#""01"^^xsd:integer"#),
        ("Zoë in Dutch", r#""Zoë"@nl"#),
        ("Zoë in English", r#""Zoë"@en"#),
        ("plain Zoë", r#""Zoë""#),
        ("an IRI", "<http://ex.org/x>"),
        ("the IRI's text", r#""http://ex.org/x""#),
    ];
    let seed = |db: &GrafeoDB| {
        let objects: Vec<String> = terms
            .iter()
            .map(|(_, term)| format!("ex:harm ex:v {term} ."))
            .collect();
        db.execute_sparql(&q(&format!("INSERT DATA {{ {} }}", objects.join(" "))))
            .unwrap();
    };
    let delete = |term: &str| format!("DELETE DATA {{ ex:harm ex:v {term} . }}");
    let mut wrong = Vec::new();
    for (i, (name, term)) in terms.iter().enumerate() {
        let db = seeded();
        seed(&db);
        let same = race(&db, Order::FirstOpen, &[&delete(term)], &[&delete(term)]);
        if same != Outcome::Conflict {
            wrong.push(format!("{name} twice: {same:?}, want a conflict"));
        }
        if let Some((other_name, other)) = terms.get(i + 1) {
            let db = seeded();
            seed(&db);
            let two = race(&db, Order::FirstOpen, &[&delete(term)], &[&delete(other)]);
            if two != Outcome::Committed {
                wrong.push(format!(
                    "{name} and {other_name}: {two:?}, want both to commit"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

// ------------------------------------------------------------------ checkpoints and the schema (E)

/// A checkpoint while a reader is open leaves the reader's snapshot, and the checkpointed file
/// holds the new state.
#[cfg(feature = "grafeo-file")]
#[test]
fn a_checkpoint_during_a_reader_keeps_its_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cities.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    db.execute_sparql(&q(SEED)).unwrap();
    let mut reader = db.session();
    reader.begin_transaction().unwrap();
    let before: Vec<_> = forms()
        .iter()
        .map(|(_, query)| answer(&reader, query))
        .collect();
    change(&db);
    db.wal_checkpoint().unwrap();
    let during: Vec<_> = forms()
        .iter()
        .map(|(_, query)| answer(&reader, query))
        .collect();
    assert_eq!(during, before);
    reader.commit().unwrap();
    let after: Vec<_> = forms()
        .iter()
        .map(|(_, query)| answer(&db.session(), query))
        .collect();
    db.close().unwrap();
    let reopened = GrafeoDB::open(&path).unwrap();
    let read_back: Vec<_> = forms()
        .iter()
        .map(|(_, query)| answer(&reopened.session(), query))
        .collect();
    assert_eq!(read_back, after);
}

#[test]
fn rdf_schema_lists_the_named_graphs() {
    // Comes off with #718 E1.
    xfail(
        "rdf_schema(): it lists the named graphs",
        "#718",
        rdf_schema_lists_the_named_graphs_holds,
    );
}

/// `rdf_schema()` lists the named graphs.
fn rdf_schema_lists_the_named_graphs_holds() {
    let db = seeded();
    let grafeo_engine::admin::SchemaInfo::Rdf(info) = db.rdf_schema() else {
        panic!("rdf_schema() returns the RDF schema");
    };
    let mut graphs: Vec<String> = info
        .named_graphs
        .iter()
        .map(|name| name.trim_matches(|c| c == '<' || c == '>').to_string())
        .collect();
    graphs.sort();
    assert_eq!(graphs, ["http://ex.org/g", "http://ex.org/h"]);
}

// ------------------------------------------------------------------ the apply steps and the overlay (B1, B3)
//
// These call the seams the design adds: `commit_hook::during_next_rdf_apply` with its steps, and
// the overlay's counters on `GrafeoDB` under the test feature.

#[cfg(feature = "testing-statement-injection")]
mod apply_steps {
    use std::sync::Arc;
    use std::sync::mpsc;
    use std::time::Duration;

    use grafeo_common::testing::commit_hook::{RdfApplyStep, during_next_rdf_apply};

    use super::{ALL, answer, change, change_triples, forms, q, seeded};

    /// A commit held at each step of its apply under the dataset guard: a reader whose transaction
    /// began before reads its snapshot, and a statement started meanwhile outside a transaction
    /// reads the state before the commit or after it, never a mix. Readers started from the hook
    /// may wait for the guard, so they run on threads of their own and are joined after it.
    #[test]
    fn a_reader_reads_one_state_at_each_apply_step() {
        let after = {
            let other = seeded();
            change_triples(&other);
            answer(&other.session(), ALL)
        };
        for step in [
            RdfApplyStep::GuardTaken,
            RdfApplyStep::Halfway,
            RdfApplyStep::BeforePublish,
        ] {
            let db = Arc::new(seeded());
            let before = answer(&db.session(), ALL);
            let (ready_tx, ready_rx) = mpsc::channel::<()>();
            let (go_tx, go_rx) = mpsc::channel::<()>();
            let reader = {
                let db = Arc::clone(&db);
                std::thread::spawn(move || {
                    let mut session = db.session();
                    session.begin_transaction().unwrap();
                    let first = answer(&session, ALL);
                    ready_tx.send(()).unwrap();
                    go_rx.recv().unwrap();
                    let during = answer(&session, ALL);
                    session.commit().unwrap();
                    (first, during)
                })
            };
            ready_rx.recv().unwrap();
            let (single_tx, single_rx) = mpsc::channel::<Vec<String>>();
            let committer = {
                let db = Arc::clone(&db);
                std::thread::spawn(move || {
                    let in_hook = Arc::clone(&db);
                    during_next_rdf_apply(step, move || {
                        go_tx.send(()).unwrap();
                        std::thread::spawn(move || {
                            single_tx.send(answer(&in_hook.session(), ALL)).unwrap();
                        });
                        // Give both readers the time to reach the store while the commit holds.
                        std::thread::sleep(Duration::from_millis(100));
                    });
                    change_triples(&db);
                })
            };
            committer.join().unwrap();
            let single = single_rx
                .recv_timeout(Duration::from_secs(60))
                .expect("the statement started in the hook finishes after it");
            let (first, during) = reader.join().unwrap();
            assert_eq!(first, before, "{step:?}: the reader's first read");
            assert_eq!(during, before, "{step:?}: the reader during the apply");
            assert!(
                single == before || single == after,
                "{step:?}: a statement read neither the state before the commit nor after it: {single:?}"
            );
            assert_eq!(
                answer(&db.session(), ALL),
                after,
                "{step:?}: after the commit"
            );
        }
    }

    /// With no reader open, a finished commit leaves no overlay entry; an open reader pins the
    /// entry of a commit that changes something, and the next commit after the reader ends
    /// drops it. The commits around the reader are fresh inserts: a commit of no-ops publishes
    /// no entry (Q-1 a), so `change_triples` a second time would pin nothing.
    #[test]
    fn no_entry_is_kept_without_a_reader() {
        let db = seeded();
        let fresh = |name: &str| {
            db.execute_sparql(&q(&format!(r#"INSERT DATA {{ ex:{name} ex:p "7" }}"#)))
                .unwrap();
        };
        change(&db);
        assert_eq!(db.rdf_overlay_len(), 0);
        let mut reader = db.session();
        reader.begin_transaction().unwrap();
        fresh("harm");
        assert!(db.rdf_overlay_len() > 0, "an open reader pins the entry");
        reader.commit().unwrap();
        fresh("maxence");
        assert_eq!(
            db.rdf_overlay_len(),
            0,
            "the entry goes once the reader ends"
        );
    }

    /// A reader with no commit after its snapshot reads the store as is: it builds no change map.
    #[test]
    fn a_reader_without_later_commits_builds_no_map() {
        let db = seeded();
        let mut reader = db.session();
        reader.begin_transaction().unwrap();
        for (_, query) in forms() {
            answer(&reader, query);
        }
        assert_eq!(db.rdf_overlay_views_extended(), 0, "the fast path");
        change(&db);
        answer(&reader, ALL);
        assert!(
            db.rdf_overlay_views_extended() > 0,
            "a read after a commit builds the map, so the counter counts"
        );
        reader.commit().unwrap();
    }
}
