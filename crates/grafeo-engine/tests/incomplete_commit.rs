//! A commit that is not complete yet, or never completes.
//!
//! A commit stamps its versions with its epoch and publishes the epoch only
//! once its events and WAL records are written. Until then the database's
//! direct reads (`get_node`, `get_edge`, `node_count`, `edge_count`,
//! `iter_nodes`, `iter_edges`, `validate`, `current_epoch`, the history reads
//! and the change history) read at the published epoch, as queries do, and do
//! not see the commit.
//!
//! When the commit code panics in between, the commit never completes: its
//! stamped versions stay in the store at an epoch that is never published. No
//! transaction commits afterwards (a later epoch would publish them), and
//! nothing checkpoints, saves, restores, merges, makes a full backup of or
//! copies the store (an incremental backup copies only the WAL). `close()`
//! keeps the WAL and leaves the file at its last checkpoint. A reopen then
//! shows the database without the failed commit when the panic came before
//! its WAL records were written; after them, the WAL holds the whole commit,
//! and a reopen replays it whole.
//!
//! What these tests do not cover: without the `temporal` feature, property
//! values and labels have no versions and a deletion takes effect at once, so
//! a query or a direct read can see the property values and labels written,
//! and miss what was deleted, by a commit that is not complete, or never
//! completes (#412); the property assertions below need `temporal`.
//!
//! These tests panic inside a commit, or run reads from another thread inside
//! one, with the `testing-statement-injection` commit hook:
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test incomplete_commit
//! ```

#![cfg(all(
    feature = "testing-statement-injection",
    feature = "lpg",
    feature = "gql"
))]

use std::sync::{Arc, mpsc};

use grafeo_common::testing::commit_hook::{after_next_commit_logged, after_next_commit_stamped};
use grafeo_common::types::{EdgeId, EpochId, NodeId, Value};
use grafeo_engine::GrafeoDB;

#[path = "common/started.rs"]
mod started;

use started::Started;

/// The names of the people in `db`, sorted.
fn people(db: &GrafeoDB) -> Vec<Value> {
    db.execute("MATCH (p:Person) RETURN p.name AS name ORDER BY name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect()
}

/// Alix's city, as a query sees it.
fn city_of_alix(db: &GrafeoDB) -> Value {
    db.execute("MATCH (p:Person {name: 'Alix'}) RETURN p.city")
        .unwrap()
        .rows()[0][0]
        .clone()
}

/// In one transaction: moves Alix to Paris, creates Gus and an edge from
/// Alix to Gus; the commit panics once its versions are stamped. Returns Gus
/// and the edge.
fn fail_a_commit(db: &GrafeoDB, alix: NodeId) -> (NodeId, EdgeId) {
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .set_node_property(alix, "city", Value::from("Paris"))
        .unwrap();
    let gus = session
        .create_node_with_props(&["Person"], [("name", Value::from("Gus"))])
        .unwrap();
    let knows = session.create_edge(alix, gus, "KNOWS").unwrap();
    after_next_commit_stamped(|| panic!("injected: the commit stops after stamping"));
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| session.commit()));
    assert!(unwound.is_err(), "the commit panicked");
    (gus, knows)
}

/// What the direct reads of `db` see of Alix, Gus and their edge.
#[derive(Debug, PartialEq)]
struct Seen {
    nodes: usize,
    edges: usize,
    epoch: EpochId,
    gus: bool,
    knows: bool,
    /// Alix's city, with `temporal` (property values have versions).
    city: Option<Value>,
    /// The nodes and edges `iter_nodes` and `iter_edges` return.
    iterated: (usize, usize),
    /// Whether `validate` warns that there are nodes but no edges.
    no_edges_warning: bool,
}

fn seen(db: &GrafeoDB, alix: NodeId, gus: NodeId, knows: EdgeId) -> Seen {
    let city = if cfg!(feature = "temporal") {
        db.get_node(alix)
            .and_then(|node| node.get_property("city").cloned())
    } else {
        None
    };
    Seen {
        nodes: db.node_count(),
        edges: db.edge_count(),
        epoch: db.current_epoch(),
        gus: db.get_node(gus).is_some(),
        knows: db.get_edge(knows).is_some(),
        city,
        iterated: (db.iter_nodes().count(), db.iter_edges().count()),
        no_edges_warning: db
            .validate()
            .warnings
            .iter()
            .any(|warning| warning.code == "NO_EDGES"),
    }
}

#[test]
fn after_a_commit_that_does_not_complete_no_commit_publishes_part_of_it() {
    let db = GrafeoDB::new_in_memory();
    let alix = db
        .create_node_with_props(&["Person"], [("name", Value::from("Alix"))])
        .unwrap();
    let before = db.current_epoch();

    let (gus, knows) = fail_a_commit(&db, alix);
    assert_eq!(
        people(&db),
        [Value::from("Alix")],
        "the failed commit's write is not visible"
    );
    if cfg!(feature = "temporal") {
        assert_eq!(city_of_alix(&db), Value::Null, "nor its property value");
    }
    assert_eq!(
        seen(&db, alix, gus, knows),
        Seen {
            nodes: 1,
            edges: 0,
            epoch: before,
            gus: false,
            knows: false,
            city: None,
            iterated: (1, 0),
            no_edges_warning: true,
        },
        "the direct reads see the database as it was before the failed commit"
    );

    let error = db
        .execute("INSERT (:Person {name: 'Vincent'})")
        .expect_err("no commit succeeds after one that did not complete");
    assert!(
        error.to_string().contains("did not complete") && error.to_string().contains("reopen"),
        "the error says a commit did not complete and the database must be reopened: {error}"
    );
    assert!(
        db.create_node(&["Person"]).is_err(),
        "a direct write commits too, and fails the same way"
    );
    let mut explicit = db.session();
    explicit.begin_transaction().unwrap();
    explicit.execute("INSERT (:Person {name: 'Mia'})").unwrap();
    assert!(
        explicit.commit().is_err(),
        "an explicit transaction cannot commit either"
    );

    // The direct graph and index calls change the store outside any commit,
    // and a checkpoint could never write them: they fail the same way.
    let incomplete = |call: &str, error: Option<grafeo_common::utils::error::Error>| {
        let error = error.unwrap_or_else(|| panic!("{call} after a failed commit succeeded"));
        assert!(
            matches!(
                error,
                grafeo_common::utils::error::Error::Transaction(
                    grafeo_common::utils::error::TransactionError::IncompleteCommit
                )
            ),
            "{call}: {error:?}"
        );
    };
    incomplete("create_graph", db.create_graph("berlin").err());
    incomplete("drop_graph", db.drop_graph("berlin").err());
    incomplete(
        "create_property_index",
        db.create_property_index("name").err(),
    );
    incomplete("drop_property_index", db.drop_property_index("name").err());
    #[cfg(feature = "vector-index")]
    incomplete(
        "create_vector_index",
        db.create_vector_index("Person", "embedding", Some(3), None, None, None, None)
            .err(),
    );
    #[cfg(feature = "text-index")]
    incomplete(
        "create_text_index",
        db.create_text_index("Person", "name").err(),
    );
    // So do the imports and the RDF batch insert, which would stamp their
    // writes at the failed commit's epoch, or write what no checkpoint can
    // persist.
    incomplete(
        "import_tsv_str",
        db.import_tsv_str("3\t19\n19\t88\n", "KNOWS", true).err(),
    );
    #[cfg(feature = "triple-store")]
    incomplete(
        "batch_insert_rdf",
        db.batch_insert_rdf([grafeo_core::graph::rdf::Triple::new(
            grafeo_core::graph::rdf::Term::iri("http://ex.org/alix"),
            grafeo_core::graph::rdf::Term::iri("http://ex.org/city"),
            grafeo_core::graph::rdf::Term::literal("Amsterdam"),
        )])
        .err(),
    );
    assert!(db.list_graphs().is_empty(), "no graph was created");
    assert!(!db.has_property_index("name"), "no index was created");
    // A projection is only the session's state: it still works.
    db.execute("CREATE PROJECTION social LABELS (Person)")
        .unwrap_or_else(|error| panic!("CREATE PROJECTION after a failed commit: {error}"));

    let snapshot = {
        let other = GrafeoDB::new_in_memory();
        other.create_node(&["Person"]).unwrap();
        other.export_snapshot().unwrap()
    };
    let error = db
        .restore_snapshot(&snapshot)
        .expect_err("a restore after a failed commit could never be checkpointed");
    assert!(error.to_string().contains("did not complete"), "{error}");

    assert_eq!(
        people(&db),
        [Value::from("Alix")],
        "reads see what was published before the failed commit, never part of it"
    );
    assert_eq!(db.current_epoch(), before);
}

/// After a commit that did not complete, SPARQL updates and graph operations
/// fail with the incomplete-commit error (`GRAFEO-T008`), through a session
/// and through `GrafeoDB::execute_sparql`, and change nothing (#414).
#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn after_a_commit_that_does_not_complete_sparql_updates_fail_with_its_code() {
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql("CREATE GRAPH <http://ex.org/paris>")
        .unwrap();
    let alix = db
        .create_node_with_props(&["Person"], [("name", Value::from("Alix"))])
        .unwrap();
    fail_a_commit(&db, alix);

    let session = db.session();
    for update in [
        "INSERT DATA { <http://ex.org/gus> <http://ex.org/city> \"Berlin\" }",
        "CREATE GRAPH <http://ex.org/berlin>",
        "CLEAR ALL",
        "DROP GRAPH <http://ex.org/paris>",
        "COPY DEFAULT TO <http://ex.org/prague>",
    ] {
        for (through, outcome) in [
            ("a session", session.execute_sparql(update)),
            ("execute_sparql", db.execute_sparql(update)),
        ] {
            let error = outcome.expect_err(update);
            assert_eq!(
                error.error_code().as_str(),
                "GRAFEO-T008",
                "{update} through {through}: {error}"
            );
        }
    }
    assert!(db.rdf_store().is_empty(), "no triple was added");
    assert_eq!(
        db.rdf_store().graph_names(),
        ["http://ex.org/paris".to_string()],
        "no graph was created or dropped"
    );
}

/// After a commit that did not complete, an RDF batch insert refuses before
/// it pulls the caller's iterator (which may parse or compute the triples),
/// and an import before it opens its file or parses its data: a refused call
/// does no work, and a missing file or malformed data still gets the
/// incomplete-commit error.
#[test]
fn after_a_commit_that_does_not_complete_imports_refuse_before_reading_their_input() {
    use grafeo_common::utils::error::{Error, TransactionError};

    #[track_caller]
    fn incomplete<T: std::fmt::Debug>(what: &str, result: grafeo_common::utils::error::Result<T>) {
        match result {
            Err(Error::Transaction(TransactionError::IncompleteCommit)) => {}
            other => panic!("{what}: the incomplete-commit error, got {other:?}"),
        }
    }

    let db = GrafeoDB::new_in_memory();
    let alix = db
        .create_node_with_props(&["Person"], [("name", Value::from("Alix"))])
        .unwrap();
    fail_a_commit(&db, alix);
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("missing.tsv");
    assert!(!missing.exists(), "the input file does not exist");

    #[cfg(feature = "triple-store")]
    {
        use grafeo_core::graph::rdf::{Term, Triple};
        let pulled = std::cell::Cell::new(false);
        let triples = std::iter::once_with(|| {
            pulled.set(true);
            Triple::new(
                Term::iri("http://ex.org/gus"),
                Term::iri("http://ex.org/city"),
                Term::literal("Berlin"),
            )
        });
        incomplete("batch_insert_rdf", db.batch_insert_rdf(triples));
        assert!(
            !pulled.get(),
            "the refused batch insert pulled the caller's iterator"
        );
        incomplete(
            "import_tsv_rdf of a missing file",
            db.import_tsv_rdf(&missing, "http://ex.org/knows", "http://ex.org/"),
        );
        assert!(db.rdf_store().is_empty(), "no triple was added");
    }
    incomplete(
        "import_tsv of a missing file",
        db.import_tsv(&missing, "KNOWS", true),
    );
    incomplete(
        "import_mmio of a missing file",
        db.import_mmio(dir.path().join("missing.mtx"), "KNOWS"),
    );
    incomplete(
        "import_tsv_str of malformed data",
        db.import_tsv_str("Alix\tGus\n", "KNOWS", true),
    );
    assert_eq!(people(&db), [Value::from("Alix")], "nothing was imported");
}

/// A bulk import or an RDF batch insert, which hold commits off while they
/// write, by name.
type Import = (
    &'static str,
    fn(&GrafeoDB) -> grafeo_common::utils::error::Result<()>,
);

/// The bulk imports and the RDF batch insert.
fn imports() -> Vec<Import> {
    let imports: Vec<Import> = vec![("import_tsv_str", |db| {
        db.import_tsv_str("3\t19\n19\t88\n", "KNOWS", true)
            .map(drop)
    })];
    #[cfg(feature = "triple-store")]
    let imports = imports.into_iter().chain::<[Import; 2]>([
        ("import_tsv_rdf", |db| {
            let dir = tempfile::tempdir().unwrap();
            let tsv = dir.path().join("edges.tsv");
            std::fs::write(&tsv, "3\t19\n19\t88\n").unwrap();
            db.import_tsv_rdf(&tsv, "http://ex.org/knows", "http://ex.org/")
                .map(drop)
        }),
        ("batch_insert_rdf", |db| {
            db.batch_insert_rdf([grafeo_core::graph::rdf::Triple::new(
                grafeo_core::graph::rdf::Term::iri("http://ex.org/gus"),
                grafeo_core::graph::rdf::Term::iri("http://ex.org/city"),
                grafeo_core::graph::rdf::Term::literal("Berlin"),
            )])
            .map(drop)
        }),
    ]);
    imports.into_iter().collect()
}

/// An import that starts while a commit is in progress waits for it, and
/// when that commit then fails, the import refuses with the failed commit's
/// error and adds nothing: after a commit that did not complete no commit
/// may follow, and its own would publish the failed commit's stamped
/// versions with its epoch (before, it returned `Ok` for data that was lost
/// on reopen).
#[test]
fn an_import_waiting_for_a_commit_that_fails_refuses() {
    let mut failures = Vec::new();
    for (name, import) in imports() {
        let db = Arc::new(GrafeoDB::new_in_memory());
        let alix = db
            .create_node_with_props(&["Person"], [("name", Value::from("Alix"))])
            .unwrap();

        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .set_node_property(alix, "city", Value::from("Paris"))
            .unwrap();
        let importer = Arc::clone(&db);
        let (sender, during) = mpsc::channel();
        after_next_commit_stamped(move || {
            let started = Started::spawn(move || import(&importer));
            let finished = started.finishes_briefly();
            sender.send((started, finished)).unwrap();
            panic!("injected: the commit stops after stamping");
        });
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| session.commit()));
        assert!(unwound.is_err(), "{name}: the commit panicked");

        let (started, finished) = during.recv().expect("the commit ran the hook");
        if finished {
            failures.push(format!(
                "{name}: the import did not wait for the commit in progress"
            ));
        }
        match started.join() {
            Ok(()) => failures.push(format!(
                "{name}: the import returned Ok after the commit failed"
            )),
            Err(error) => {
                if !matches!(
                    error,
                    grafeo_common::utils::error::Error::Transaction(
                        grafeo_common::utils::error::TransactionError::IncompleteCommit
                    )
                ) {
                    failures.push(format!("{name}: another error: {error:?}"));
                }
            }
        }
        if db.node_count() != 1 || db.edge_count() != 0 {
            failures.push(format!(
                "{name}: the import added {} nodes and {} edges",
                db.node_count() - 1,
                db.edge_count()
            ));
        }
        #[cfg(feature = "triple-store")]
        if !db.rdf_store().is_empty() {
            failures.push(format!(
                "{name}: the import added {} triples",
                db.rdf_store().len()
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The change events of a commit are recorded before it is complete; the
/// change history returns them only once it is, and never those of a commit
/// that does not complete.
#[cfg(feature = "cdc")]
#[test]
fn change_events_of_a_commit_are_seen_only_once_it_is_complete() {
    let db = Arc::new(GrafeoDB::new_in_memory());
    db.set_cdc_enabled(true);
    let alix = db
        .create_node_with_props(&["Person"], [("name", Value::from("Alix"))])
        .unwrap();
    // Events of Alix, of Gus, and of every entity.
    let events = move |db: &GrafeoDB, gus: NodeId| {
        (
            db.history(alix).unwrap().len(),
            db.history(gus).unwrap().len(),
            db.changes_between(EpochId::new(0), EpochId::new(u64::MAX))
                .unwrap()
                .len(),
        )
    };

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .set_node_property(alix, "city", Value::from("Paris"))
        .unwrap();
    let gus = session
        .create_node_with_props(&["Person"], [("name", Value::from("Gus"))])
        .unwrap();
    let (sender, during) = mpsc::channel();
    let reader = Arc::clone(&db);
    after_next_commit_logged(move || {
        let seen = std::thread::spawn(move || events(&reader, gus))
            .join()
            .expect("the reads during the commit panicked");
        sender.send(seen).unwrap();
    });
    session.commit().unwrap();
    assert_eq!(
        during.recv().expect("the commit ran the hook"),
        (1, 0, 1),
        "during the commit only Alix's creation is in the change history"
    );
    assert_eq!(
        events(&db, gus),
        (2, 1, 3),
        "once the commit is complete its events are"
    );

    // A commit that fails once its events and WAL records are written.
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let vincent = session
        .create_node_with_props(&["Person"], [("name", Value::from("Vincent"))])
        .unwrap();
    after_next_commit_logged(|| panic!("injected: the commit stops after its WAL records"));
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| session.commit()));
    assert!(unwound.is_err(), "the commit panicked");
    assert_eq!(
        db.history(vincent).unwrap().len(),
        0,
        "the events of a commit that does not complete are never returned"
    );
}

/// A commit that panics once its WAL records are written leaves a complete
/// group in the WAL: it is not visible in the open database, and a reopen
/// replays all of it.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn a_commit_failing_after_its_wal_records_is_replayed_whole_on_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prague.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    let alix = db
        .create_node_with_props(&["Person"], [("name", Value::from("Alix"))])
        .unwrap();

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .set_node_property(alix, "city", Value::from("Paris"))
        .unwrap();
    session
        .create_node_with_props(&["Person"], [("name", Value::from("Gus"))])
        .unwrap();
    after_next_commit_logged(|| panic!("injected: the commit stops after its WAL records"));
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| session.commit()));
    assert!(unwound.is_err(), "the commit panicked");
    assert_eq!(people(&db), [Value::from("Alix")], "not published");
    drop(session);
    assert!(db.close().is_err(), "close reports the failed commit");
    drop(db);

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        people(&db),
        [Value::from("Alix"), Value::from("Gus")],
        "the reopen replays the commit's complete WAL group"
    );
    assert_eq!(city_of_alix(&db), Value::from("Paris"));
    db.close().unwrap();
}

/// Under memory pressure a compacted database merges its overlay into the
/// base, which has no versions: after a failed commit the merge would make
/// the commit's stamped part visible, so it does not run.
#[test]
fn a_memory_pressure_merge_never_folds_a_failed_commit_into_the_base() {
    let mut db = GrafeoDB::new_in_memory();
    let alix = db
        .create_node_with_props(&["Person"], [("name", Value::from("Alix"))])
        .unwrap();
    db.compact().unwrap();
    fail_a_commit(&db, alix);
    db.buffer_manager().spill_all();
    assert_eq!(
        people(&db),
        [Value::from("Alix")],
        "the merge did not fold the failed commit into the base"
    );
    assert_eq!(db.node_count(), 1);
}

/// Direct reads from another thread inside a commit, once its versions are
/// stamped, see the database without the commit; once the commit is
/// complete, they see it.
#[test]
fn direct_reads_see_a_commit_only_once_it_is_complete() {
    let db = Arc::new(GrafeoDB::new_in_memory());
    let alix = db
        .create_node_with_props(&["Person"], [("name", Value::from("Alix"))])
        .unwrap();
    let before = db.current_epoch();

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .set_node_property(alix, "city", Value::from("Paris"))
        .unwrap();
    let gus = session
        .create_node_with_props(&["Person"], [("name", Value::from("Gus"))])
        .unwrap();
    let knows = session.create_edge(alix, gus, "KNOWS").unwrap();

    let (sender, during) = mpsc::channel();
    let reader = Arc::clone(&db);
    after_next_commit_stamped(move || {
        // Another thread reads while the commit is stamped and not complete;
        // it never waits for the commit, so it is joined here.
        let seen = std::thread::spawn(move || seen(&reader, alix, gus, knows))
            .join()
            .expect("the reads during the commit panicked");
        sender.send(seen).unwrap();
    });
    session.commit().unwrap();

    assert_eq!(
        during.recv().expect("the commit ran the hook"),
        Seen {
            nodes: 1,
            edges: 0,
            epoch: before,
            gus: false,
            knows: false,
            city: None,
            iterated: (1, 0),
            no_edges_warning: true,
        },
        "during the commit the direct reads do not see it"
    );
    let after = seen(&db, alix, gus, knows);
    assert!(after.epoch > before, "the commit published its epoch");
    assert_eq!(
        after,
        Seen {
            nodes: 2,
            edges: 1,
            epoch: after.epoch,
            gus: true,
            knows: true,
            city: cfg!(feature = "temporal").then(|| Value::from("Paris")),
            iterated: (2, 1),
            no_edges_warning: false,
        },
        "once the commit is complete the direct reads see it"
    );
}

/// After a commit that did not complete, nothing persists or copies the
/// store, which holds the commit's stamped part: every checkpoint, save,
/// backup and copy fails with the error of the failed commit and leaves the
/// file as it was. `close()` fails the same way, keeps the WAL and releases
/// the file; a reopen shows the database without the failed commit, with
/// what the WAL holds, and commits again.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn a_failed_commit_is_never_checkpointed_saved_or_copied() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    let alix = {
        let db = GrafeoDB::open(&path).unwrap();
        let alix = db
            .create_node_with_props(&["Person"], [("name", Value::from("Alix"))])
            .unwrap();
        db.close().unwrap();
        alix
    };
    let checkpointed = std::fs::read(&path).unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    // Only in the WAL: a reopen must replay it.
    db.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
    fail_a_commit(&db, alix);
    let header = db.file_manager().unwrap().active_header();
    let assert_refused =
        |db: &GrafeoDB, operation: &str, outcome: grafeo_common::utils::error::Result<()>| {
            let error = outcome
                .err()
                .unwrap_or_else(|| panic!("{operation} succeeded after a failed commit"));
            assert!(
                error.to_string().contains("did not complete"),
                "{operation}: the error is the failed commit's: {error}"
            );
            assert_eq!(
                db.file_manager().unwrap().active_header(),
                header,
                "{operation}: the file has no new checkpoint"
            );
        };

    let copy = dir.path().join("copy.grafeo");
    let copy_without_extension = dir.path().join("copy");
    let backups = dir.path().join("backups");
    assert_refused(&db, "wal_checkpoint", db.wal_checkpoint());
    assert_refused(&db, "save to a .grafeo file", db.save(&copy));
    assert_refused(
        &db,
        "save to a path without the extension",
        db.save(&copy_without_extension),
    );
    assert_refused(&db, "to_memory", db.to_memory().map(drop));
    assert_refused(&db, "export_snapshot", db.export_snapshot().map(drop));
    assert_refused(&db, "backup_full", db.backup_full(&backups).map(drop));
    let db = {
        let mut db = db;
        let outcome = db.compact().map(drop);
        assert_refused(&db, "compact", outcome);
        db
    };
    for target in [&copy, &copy_without_extension] {
        assert!(
            !target.exists(),
            "nothing is written to {}",
            target.display()
        );
    }
    assert!(
        !backups.exists() || std::fs::read_dir(&backups).unwrap().next().is_none(),
        "no backup is written"
    );

    let error = db.close().expect_err("close reports the failed commit");
    assert!(error.to_string().contains("did not complete"), "{error}");
    drop(db);
    let wal = dir.path().join("amsterdam.grafeo.wal");
    assert!(
        wal.is_dir() && std::fs::read_dir(&wal).unwrap().next().is_some(),
        "close keeps the WAL, which holds Vincent"
    );
    assert!(
        std::fs::read(&path).unwrap() == checkpointed,
        "the file is byte for byte the last checkpoint"
    );

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        people(&db),
        [Value::from("Alix"), Value::from("Vincent")],
        "the reopened database has the WAL's commits and nothing of the failed one"
    );
    assert_eq!(city_of_alix(&db), Value::Null, "Alix never moved to Paris");
    db.execute("INSERT (:Person {name: 'Mia'})").unwrap();
    assert_eq!(
        people(&db),
        [
            Value::from("Alix"),
            Value::from("Mia"),
            Value::from("Vincent")
        ],
        "the reopened database commits again"
    );
    db.close().unwrap();
}

/// The periodic checkpoint timer writes the store every interval; after a
/// commit that did not complete, it writes nothing more.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn the_checkpoint_timer_stops_after_a_failed_commit() {
    use std::time::{Duration, Instant};

    use grafeo_common::testing::commit_hook::checkpoints_started;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("berlin.grafeo");
    let db = GrafeoDB::with_config(
        grafeo_engine::Config::persistent(&path)
            .with_checkpoint_interval(Duration::from_millis(200)),
    )
    .unwrap();
    let alix = db
        .create_node_with_props(&["Person"], [("name", Value::from("Alix"))])
        .unwrap();
    let header = || db.file_manager().unwrap().active_header();

    // The timer is running: a checkpoint replaces the active header.
    let first = header();
    let deadline = Instant::now() + Duration::from_secs(10);
    while header() == first {
        assert!(Instant::now() < deadline, "the timer never checkpointed");
        std::thread::sleep(Duration::from_millis(50));
    }

    fail_a_commit(&db, alix);
    // A checkpoint holds commits off, so none is running now: the header
    // stays as it is from the failure on.
    let after_failure = header();
    let attempts = checkpoints_started(&path);
    std::thread::sleep(Duration::from_millis(1000));
    assert_eq!(
        header(),
        after_failure,
        "no checkpoint in five intervals after the failed commit"
    );
    let since = checkpoints_started(&path) - attempts;
    assert!(
        since <= 1,
        "the timer stops after its first failed attempt, it made {since}"
    );
    assert!(db.close().is_err(), "close reports the failed commit");
}
