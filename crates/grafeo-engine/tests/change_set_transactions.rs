//! Every transaction records its writes in one change set, a direct call's
//! too: one savepoint position covers every graph it wrote, a direct call is
//! a private transaction that conflicts with open transactions (and they
//! with it), a rollback on a store without undo says which graph keeps its
//! writes, and the rollbacks the stores' undo logs got wrong come out right.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test change_set_transactions
//! ```

#![cfg(all(feature = "lpg", feature = "gql"))]

use std::sync::Arc;

use grafeo_common::types::{NodeId, Value};
use grafeo_common::utils::error::{Error, TransactionError};
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

/// The values a single-column query returns, as strings, sorted.
fn column(session: &Session, query: &str) -> Vec<String> {
    let mut values: Vec<String> = session
        .execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .iter()
        .map(|row| text(&row[0]))
        .collect();
    values.sort();
    values
}

/// A value as text: a string without quotes.
fn text(value: &Value) -> String {
    match value {
        Value::String(text) => text.to_string(),
        other => other.to_string(),
    }
}

/// What a new session sees in graph `graph` (`None` for the default graph).
fn seen_in(db: &GrafeoDB, graph: Option<&str>, query: &str) -> Vec<String> {
    let session = db.session();
    if let Some(graph) = graph {
        session.use_graph(graph);
    }
    column(&session, query)
}

/// Whether `error` is a write conflict.
fn is_conflict(error: &Error) -> bool {
    matches!(
        error,
        Error::Transaction(TransactionError::WriteConflict(_))
    )
}

/// A rollback to a savepoint undoes what came after it in every graph at
/// once, in a graph written before the savepoint as in one first written
/// after it, and keeps what came before it in each.
#[test]
fn one_savepoint_position_undoes_two_graphs() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE GRAPH trips").unwrap();
    db.execute("CREATE GRAPH stops").unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session.use_graph("trips");
    session
        .execute("INSERT (:City {name: 'Amsterdam'})")
        .unwrap();

    session.savepoint("before_berlin").unwrap();
    session.execute("INSERT (:City {name: 'Berlin'})").unwrap();
    session
        .execute("MATCH (c:City {name: 'Amsterdam'}) SET c.visits = 3")
        .unwrap();
    session.use_graph("stops");
    session.execute("INSERT (:Stop {name: 'Prague'})").unwrap();
    session.use_graph("default");
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    session
        .execute("MATCH (p:Person {name: 'Alix'}) SET p.city = 'Paris'")
        .unwrap();
    session.rollback_to_savepoint("before_berlin").unwrap();

    assert_eq!(
        column(
            &session,
            "MATCH (p:Person) RETURN p.name + '/' + coalesce(p.city, '-')"
        ),
        ["Alix/-"],
        "the transaction itself sees the savepoint's state"
    );
    session.commit().unwrap();

    assert_eq!(
        seen_in(
            &db,
            None,
            "MATCH (p:Person) RETURN p.name + '/' + coalesce(p.city, '-')"
        ),
        ["Alix/-"]
    );
    assert_eq!(
        seen_in(
            &db,
            Some("trips"),
            "MATCH (c:City) RETURN c.name + '/' + coalesce(toString(c.visits), '-')"
        ),
        ["Amsterdam/-"]
    );
    assert_eq!(
        seen_in(&db, Some("stops"), "MATCH (s) RETURN s.name"),
        Vec::<String>::new(),
        "a graph first written after the savepoint"
    );
}

/// A direct call that writes a node an open transaction wrote first fails
/// with a write conflict and leaves nothing; the transaction goes on and
/// commits. A direct call on another node runs beside it.
#[test]
fn a_direct_call_conflicts_with_an_open_transaction_that_wrote_first() {
    let db = GrafeoDB::new_in_memory();
    let alix = db
        .create_node_with_props(&["Person"], [("name", Value::from("Alix"))])
        .unwrap();
    let gus = db
        .create_node_with_props(&["Person"], [("name", Value::from("Gus"))])
        .unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (p:Person {name: 'Alix'}) SET p.city = 'Amsterdam'")
        .unwrap();

    let error = db
        .set_node_property(alix, "city", Value::from("Berlin"))
        .unwrap_err();
    assert!(is_conflict(&error), "{error}");
    let error = db.delete_node(alix).unwrap_err();
    assert!(is_conflict(&error), "a delete too: {error}");
    db.set_node_property(gus, "city", Value::from("Paris"))
        .expect("another node is free");

    session.commit().unwrap();
    let city = |id: NodeId| {
        db.get_node(id)
            .and_then(|node| node.properties.get(&"city".into()).cloned())
    };
    assert_eq!(city(alix), Some(Value::from("Amsterdam")));
    assert_eq!(city(gus), Some(Value::from("Paris")));
}

/// A transaction that writes a node a direct call wrote and committed after
/// the transaction began fails at its commit, and its write is undone: the
/// direct call's value stays.
#[test]
fn a_transaction_conflicts_with_a_direct_call_committed_after_its_start() {
    let db = GrafeoDB::new_in_memory();
    let alix = db
        .create_node_with_props(&["Person"], [("name", Value::from("Alix"))])
        .unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    db.set_node_property(alix, "city", Value::from("Berlin"))
        .expect("the transaction has not written the node");
    session
        .execute("MATCH (p:Person {name: 'Alix'}) SET p.city = 'Paris'")
        .unwrap();

    let error = session.commit().unwrap_err();
    assert!(is_conflict(&error), "{error}");
    assert!(!session.in_transaction());
    assert_eq!(
        seen_in(&db, None, "MATCH (p:Person) RETURN p.city"),
        ["Berlin"]
    );
}

/// Direct calls from eight threads run beside two open transactions: every
/// call commits on its own (none waits for the transactions to end), and
/// the transactions commit too.
#[test]
fn direct_calls_from_eight_threads_beside_open_transactions() {
    const CALLS: i64 = 88;
    let db = Arc::new(GrafeoDB::new_in_memory());
    let mut mia = db.session();
    mia.begin_transaction().unwrap();
    mia.execute("INSERT (:Open {name: 'Mia'})").unwrap();
    let mut jules = db.session();
    jules.begin_transaction().unwrap();
    jules.execute("INSERT (:Open {name: 'Jules'})").unwrap();
    let epoch = db.current_epoch();

    std::thread::scope(|scope| {
        for thread in 0..8_i64 {
            let db = Arc::clone(&db);
            scope.spawn(move || {
                for call in 0..CALLS {
                    let id = db
                        .create_node_with_props(
                            &["Direct"],
                            [
                                ("thread", Value::Int64(thread)),
                                ("call", Value::Int64(call)),
                            ],
                        )
                        .unwrap();
                    db.set_node_property(id, "city", Value::from("Prague"))
                        .unwrap();
                }
            });
        }
        // The open transactions go on meanwhile.
        mia.execute("INSERT (:Open {name: 'Vincent'})").unwrap();
    });
    assert_eq!(
        db.current_epoch().as_u64() - epoch.as_u64(),
        8 * 2 * CALLS as u64,
        "every call committed at an epoch of its own"
    );
    assert_eq!(
        seen_in(&db, None, "MATCH (n:Open) RETURN n.name"),
        Vec::<String>::new(),
        "the open transactions' writes stay theirs"
    );
    mia.commit().unwrap();
    jules.commit().unwrap();

    assert_eq!(
        seen_in(
            &db,
            None,
            "MATCH (n:Direct {city: 'Prague'}) RETURN count(n)"
        ),
        [(8 * CALLS).to_string()]
    );
    assert_eq!(
        seen_in(&db, None, "MATCH (n:Open) RETURN n.name"),
        ["Jules", "Mia", "Vincent"]
    );
}

/// A rollback to a savepoint undoes what the transaction created and then
/// deleted after it: the node comes back with its labels and values, and
/// commits with them (the undo log restored it without its labels).
#[test]
fn a_savepoint_rollback_restores_a_deleted_node_it_created_whole() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person:Traveller {name: 'Mia', city: 'Barcelona'})")
        .unwrap();
    session.savepoint("before_delete").unwrap();
    session
        .execute("MATCH (p:Person {name: 'Mia'}) DELETE p")
        .unwrap();
    session.rollback_to_savepoint("before_delete").unwrap();
    let query = "MATCH (p:Person) RETURN p.name + ':' + p.city + ':' + \
                 reduce(s = '', l IN labels(p) | s + l)";
    let expected = |labels: &str| vec![format!("Mia:Barcelona:{labels}")];
    let labels = column(&session, "MATCH (p:Traveller) RETURN p.name");
    assert_eq!(labels, ["Mia"], "the label index holds the node again");
    let seen = column(&session, query);
    assert!(
        seen == expected("PersonTraveller") || seen == expected("TravellerPerson"),
        "{seen:?}"
    );
    session.commit().unwrap();
    let seen = seen_in(&db, None, query);
    assert!(
        seen == expected("PersonTraveller") || seen == expected("TravellerPerson"),
        "{seen:?}"
    );
}

/// With `temporal`, a node deleted keeps its labels in its history: a read
/// at an epoch before the delete finds it with them (the undo log path
/// dropped the node's whole label history).
#[cfg(feature = "temporal")]
#[test]
fn a_read_before_a_delete_sees_the_deleted_nodes_labels() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person:Boxer {name: 'Butch'})")
        .unwrap();
    let before = db.current_epoch();
    // A later commit: a delete marks the node deleted at the epoch it
    // reads at (inbox: deletes keep the start epoch).
    db.execute("INSERT (:City {name: 'Barcelona'})").unwrap();
    db.execute("MATCH (p:Person {name: 'Butch'}) DELETE p")
        .unwrap();
    // Not through the label index, which holds the present.
    let query =
        "MATCH (p) WHERE p.name = 'Butch' RETURN reduce(s = '', l IN labels(p) | s + ':' + l)";
    let result = db.execute_at_epoch(query, before).unwrap();
    let labels: Vec<String> = result.rows().iter().map(|row| text(&row[0])).collect();
    assert!(
        labels == [":Person:Boxer"] || labels == [":Boxer:Person"],
        "{labels:?}"
    );
    assert_eq!(db.execute(query).unwrap().row_count(), 0, "deleted now");
}

/// With `temporal`, a rolled-back value and a rolled-back label leave the
/// text index as they found it (the undo log path left both in the index).
#[cfg(all(feature = "temporal", feature = "text-index"))]
#[test]
fn a_rollback_leaves_the_text_index_as_committed() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Doc {body: 'canals of Amsterdam'})")
        .unwrap();
    db.execute("INSERT (:Note {body: 'bridges of Prague'})")
        .unwrap();
    db.create_text_index("Doc", "body").unwrap();
    let found = |word: &str| db.text_search("Doc", "body", word, 3).unwrap().len();
    assert_eq!((found("canals"), found("bridges")), (1, 0));

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (d:Doc) SET d.body = 'trams of Berlin'")
        .unwrap();
    session.execute("MATCH (n:Note) SET n:Doc").unwrap();
    session.rollback().unwrap();

    assert_eq!(found("canals"), 1, "the committed value is indexed again");
    assert_eq!(found("trams"), 0, "the rolled-back value is not");
    assert_eq!(found("bridges"), 0, "the rolled-back label took nothing in");
}

/// On a database built on an external store, which has no undo, a rollback
/// undoes what it can, aborts the transaction and fails naming the graph
/// whose store keeps the writes; it does not poison the database. A
/// rollback to a savepoint after such writes is refused before it changes
/// anything.
#[test]
fn a_rollback_names_the_external_store_that_keeps_its_writes() {
    use grafeo_core::graph::GraphStoreMut;
    use grafeo_core::graph::lpg::LpgStore;

    let store = Arc::new(LpgStore::new().unwrap());
    let db = GrafeoDB::with_store(
        Arc::clone(&store) as Arc<dyn GraphStoreMut>,
        grafeo_engine::Config::in_memory(),
    )
    .unwrap();
    let kept = |error: &Error| {
        let message = error.to_string();
        message.contains("graph 'default'") && message.contains("keeps")
    };

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.savepoint("start").unwrap();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    let refused = session.rollback_to_savepoint("start").unwrap_err();
    assert!(kept(&refused), "{refused}");
    assert_eq!(
        column(&session, "MATCH (p:Person) RETURN p.name"),
        ["Alix"],
        "a refused rollback to a savepoint changes nothing"
    );
    assert!(
        session.rollback_to_savepoint("start").is_err(),
        "the savepoint stays, and is refused again"
    );

    let error = session.rollback().unwrap_err();
    assert!(kept(&error), "{error}");
    assert!(!session.in_transaction(), "the transaction is aborted");
    assert_eq!(store.all_node_ids().len(), 1, "the store keeps the write");

    // Not poisoned: another transaction commits.
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    session.commit().unwrap();

    // A statement that fails after a write says the store keeps it, with
    // its own error first: in a transaction, and on its own.
    session.execute("CREATE NODE TYPE Doc (id INT64)").unwrap();
    let partial = "UNWIND [3, 'x'] AS v INSERT (:Doc {id: v})";
    session.begin_transaction().unwrap();
    let error = session.execute(partial).unwrap_err().to_string();
    assert!(
        error.contains("not all undone") && error.contains("graph 'default'"),
        "{error}"
    );
    // The statement reported what the store keeps: nothing is left to undo.
    session.rollback().unwrap();
    let error = session.execute(partial).unwrap_err().to_string();
    assert!(error.contains("not all undone"), "{error}");
}

/// Direct calls that returned survive a crash (the process exits without
/// `close()`, so nothing is checkpointed and the reopen replays the WAL):
/// each wrote its changes to the WAL as one group when it committed. A
/// batch whose second row failed left nothing to replay.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
mod crash {
    use std::path::{Path, PathBuf};

    use grafeo_common::testing::child_process;
    use grafeo_common::types::{PropertyKey, Value};
    use grafeo_engine::config::StorageFormat;
    use grafeo_engine::{Config, GrafeoDB};

    const PATH_VAR: &str = "GRAFEO_DIRECT_CALLS_CRASH_PATH";

    fn config(path: &Path) -> Config {
        Config::persistent(path).with_storage_format(StorageFormat::Auto)
    }

    /// The direct calls, every kind once, and a batch that fails.
    fn calls(db: &GrafeoDB) {
        let alix = db
            .create_node_with_props(&["Person", "Traveller"], [("name", Value::from("Alix"))])
            .unwrap();
        let gus = db
            .create_node_with_props(&["Person"], [("name", Value::from("Gus"))])
            .unwrap();
        let knows = db
            .create_edge_with_props(alix, gus, "KNOWS", [("since", Value::Int64(3))])
            .unwrap();
        db.set_node_property(alix, "city", Value::from("Amsterdam"))
            .unwrap();
        db.set_edge_property(knows, "since", Value::Int64(19))
            .unwrap();
        assert!(db.remove_node_label(alix, "Traveller").unwrap());
        assert!(db.add_node_label(gus, "Traveller").unwrap());
        let vincent = db
            .create_node_with_props(&["Person"], [("name", Value::from("Vincent"))])
            .unwrap();
        assert!(db.delete_node(vincent).unwrap());
        db.batch_create_nodes_with_props(
            "City",
            vec![
                [(PropertyKey::new("name"), Value::from("Berlin"))].into(),
                [(PropertyKey::new("name"), Value::from("Paris"))].into(),
            ],
        )
        .unwrap();
        db.execute("CREATE NODE TYPE Stop (name STRING NOT NULL)")
            .unwrap();
        db.batch_create_nodes_with_props(
            "Stop",
            vec![
                [(PropertyKey::new("name"), Value::from("Prague"))].into(),
                [(PropertyKey::new("name"), Value::Int64(88))].into(),
            ],
        )
        .unwrap_err();
    }

    /// Everything a database holds, as sorted rows of text.
    fn state(db: &GrafeoDB) -> Vec<String> {
        let mut rows: Vec<String> = [
            "MATCH (n) RETURN n.name + ' ' + coalesce(n.city, '-') + ' ' + \
             reduce(s = '', l IN labels(n) | s + ':' + l)",
            "MATCH (a)-[r]->(b) RETURN a.name + '-' + type(r) + '-' + b.name + ' ' + \
             toString(r.since)",
        ]
        .into_iter()
        .flat_map(|query| db.execute(query).unwrap().rows().to_vec())
        .map(|row| row[0].to_string())
        .collect();
        rows.sort();
        rows
    }

    /// Child-process entry: runs the calls on a new database and exits
    /// without `close()`; a no-op when run directly.
    #[test]
    fn crash_child() {
        let Some(path) = std::env::var_os(PATH_VAR) else {
            return;
        };
        let db = GrafeoDB::with_config(config(&PathBuf::from(path))).unwrap();
        calls(&db);
        // Crash: no close(), no checkpoint, no destructors.
        std::process::exit(0);
    }

    #[test]
    fn acknowledged_direct_calls_survive_a_crash() {
        let expected = {
            let db = GrafeoDB::new_in_memory();
            calls(&db);
            state(&db)
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("barcelona.grafeo");
        let status = child_process::run(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "crash::crash_child", "--nocapture"])
                .env(PATH_VAR, &path),
        )
        .unwrap();
        assert!(status.success(), "the child process failed");
        let mut sidecar = path.as_os_str().to_owned();
        sidecar.push(".wal");
        assert!(
            PathBuf::from(sidecar).exists(),
            "the crash left the WAL for the reopen to replay"
        );

        let db = GrafeoDB::with_config(config(&path)).unwrap();
        assert_eq!(state(&db), expected);
        db.close().unwrap();
    }
}

/// Two `QueryProcessor`s with transaction contexts on one store write
/// through the store's versioned methods (no change set: the caller commits)
/// and claim what they write through the transaction manager: the second
/// writer of a node fails with a write conflict and changes nothing.
#[test]
fn query_processors_in_transactions_claim_what_they_write() {
    use grafeo_core::graph::lpg::LpgStore;
    use grafeo_engine::query::{QueryLanguage, QueryProcessor};
    use grafeo_engine::transaction::TransactionManager;

    let store = Arc::new(LpgStore::new().unwrap());
    let alix = store.create_node_with_props(&["Person"], [("name", Value::from("Alix"))]);
    let manager = Arc::new(TransactionManager::new());
    let (first, second) = (manager.begin(), manager.begin());
    let epoch = manager.current_epoch();
    let processor = |transaction| {
        QueryProcessor::for_lpg_with_transaction(Arc::clone(&store), Arc::clone(&manager))
            .with_transaction_context(epoch, transaction)
    };

    processor(first)
        .process(
            "MATCH (p:Person) SET p.city = 'Berlin'",
            QueryLanguage::Gql,
            None,
        )
        .unwrap();
    let error = processor(second)
        .process(
            "MATCH (p:Person) SET p.city = 'Paris'",
            QueryLanguage::Gql,
            None,
        )
        .unwrap_err();
    assert!(is_conflict(&error), "{error}");
    assert_eq!(
        store.get_node_property(alix, &"city".into()),
        Some(Value::from("Berlin")),
        "the refused write changed nothing"
    );
}
