//! After `close()` every call that can fail fails with `GRAFEO-T007` (#712): a statement, a
//! session's statement, a stream's pull and a view's read, whether made before or after the
//! close, in a database with a file and in one without; a read of a spilled value too. A second
//! `close()` answers. An explicit transaction open across `close()` fails at its commit and
//! leaves nothing (in a database with a file, `checkpoint_committed_state` pins the same).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main close_refuses_every_call::
//! ```

#![cfg(all(feature = "gql", feature = "wal", feature = "grafeo-file"))]

use std::path::Path;

use grafeo_common::utils::error::Result;
use grafeo_engine::config::HistoryRetention;
use grafeo_engine::{Config, GrafeoDB};

const READ: &str = "MATCH (p:Person) RETURN p.name ORDER BY p.name";
/// A read a stream can pull (no ORDER BY).
const STREAMED: &str = "MATCH (p:Person) RETURN p.name";

/// `result` failed with the closed-database error.
fn assert_closed<T: std::fmt::Debug>(result: Result<T>, what: &str) {
    match result {
        Err(error) => assert_eq!(
            error.error_code().as_str(),
            "GRAFEO-T007",
            "{what}: {error}"
        ),
        Ok(value) => panic!("{what}: answered {value:?} after close()"),
    }
}

/// Alix and Gus, with history kept so a view of an earlier epoch can open.
fn people(path: Option<&Path>) -> GrafeoDB {
    let config = match path {
        Some(path) => Config::persistent(path),
        None => Config::in_memory(),
    }
    .with_history_retention(HistoryRetention::Forever);
    let db = GrafeoDB::with_config(config).unwrap();
    db.execute("INSERT (:Person {name: 'Alix'}), (:Person {name: 'Gus'})")
        .unwrap();
    db
}

/// A statement, a session's statement (the session made before or after the close), a
/// stream's pull (opened before, never read), a view's read and a point read all fail with
/// `T007` after `close()`, in a database with a file and in one without.
#[test]
fn every_call_after_close_fails_with_the_closed_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("closed.grafeo");
    for (name, path) in [("file", Some(path.as_path())), ("in memory", None)] {
        let db = people(path);
        let id = db
            .execute("MATCH (p:Person {name: 'Alix'}) RETURN id(p)")
            .unwrap()
            .rows()[0][0]
            .as_int64()
            .unwrap();
        let before = db.session();
        let streaming = db.session();
        let mut stream = streaming.execute_streaming_owned(STREAMED).unwrap();
        let view = db.as_of(db.current_epoch()).unwrap();

        db.close().unwrap();

        assert_closed(db.execute(READ), &format!("{name}: a read"));
        assert_closed(
            db.execute("INSERT (:Person {name: 'Vincent'})"),
            &format!("{name}: a write"),
        );
        assert_closed(
            before.execute(READ),
            &format!("{name}: a session from before"),
        );
        assert_closed(
            db.session().execute(READ),
            &format!("{name}: a session from after"),
        );
        assert_closed(stream.next_chunk(), &format!("{name}: a stream's pull"));
        assert_closed(view.execute(READ), &format!("{name}: a view's read"));
        assert_closed(
            db.get_node(grafeo_common::types::NodeId::new(
                u64::try_from(id).unwrap(),
            )),
            &format!("{name}: a point read"),
        );
        db.close().unwrap();
    }
}

/// The closed-database error says what to do: open the database again.
#[test]
fn the_closed_error_says_to_open_the_database_again() {
    let db = people(None);
    db.close().unwrap();
    let error = db.execute(READ).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("the database is closed: open it again"),
        "{error}"
    );
}

/// An explicit transaction open across `close()` of a database without a file fails at its
/// commit, and nothing of it is read afterwards (nor anything at all: the database is closed).
#[test]
fn a_transaction_open_across_close_fails_at_its_commit() {
    let db = people(None);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Mia'})").unwrap();

    db.close().unwrap();

    assert_closed(session.commit(), "a commit after close()");
    assert_closed(session.execute(READ), "a read after the failed commit");
}

/// The counts, which can fail, fail with `T007` after `close()`; the getters that cannot fail
/// answer empty values (the user's choice for 0.6.x: they return `Result` from 0.7.0), and
/// `memory_usage()` reports nothing held.
#[test]
fn getters_answer_nothing_after_close() {
    let db = people(None);
    db.execute("CREATE GRAPH trips").unwrap();
    db.execute("CREATE INDEX on_name FOR (p:Person) ON (p.name)")
        .unwrap();
    assert!(db.memory_usage().total_bytes > 0, "the positive control");
    assert!(
        db.list_graphs().contains(&"trips".to_string()),
        "the positive control: the graph is listed"
    );
    assert!(
        !db.list_indexes().is_empty(),
        "the positive control: the index is listed"
    );

    db.close().unwrap();

    assert_closed(db.node_count(), "node_count()");
    assert_closed(db.edge_count(), "edge_count()");
    assert_eq!(db.list_graphs(), Vec::<String>::new(), "list_graphs()");
    assert!(db.list_indexes().is_empty(), "list_indexes()");
    assert_eq!(db.memory_usage().total_bytes, 0, "memory_usage()");
}

/// A direct call of the database: what it does, and the call.
type DirectCall = (&'static str, fn(&mut GrafeoDB, &Path) -> Result<()>);

/// Every direct call of the database that writes, reads a value or does I/O, made after
/// `close()`, without a session of the caller's: the calls a session-based check would miss.
fn direct_calls() -> Vec<DirectCall> {
    use std::collections::HashMap;

    use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};

    let mut calls: Vec<DirectCall> = vec![
        ("create_node", |db, _| db.create_node(&["Person"]).map(drop)),
        ("create_node_with_props", |db, _| {
            db.create_node_with_props(&["Person"], [("name", Value::from("Mia"))])
                .map(drop)
        }),
        ("set_node_property", |db, _| {
            db.set_node_property(NodeId::new(0), "age", Value::Int64(3))
        }),
        ("remove_node_property", |db, _| {
            db.remove_node_property(NodeId::new(0), "age").map(drop)
        }),
        ("add_node_label", |db, _| {
            db.add_node_label(NodeId::new(0), "Traveller").map(drop)
        }),
        ("remove_node_label", |db, _| {
            db.remove_node_label(NodeId::new(0), "Person").map(drop)
        }),
        ("delete_node", |db, _| {
            db.delete_node(NodeId::new(0)).map(drop)
        }),
        ("detach_delete_node", |db, _| {
            db.detach_delete_node(NodeId::new(1)).map(drop)
        }),
        ("create_edge", |db, _| {
            db.create_edge(NodeId::new(0), NodeId::new(1), "KNOWS")
                .map(drop)
        }),
        ("set_edge_property", |db, _| {
            db.set_edge_property(EdgeId::new(0), "since", Value::Int64(1988))
        }),
        ("delete_edge", |db, _| {
            db.delete_edge(EdgeId::new(0)).map(drop)
        }),
        ("get_node", |db, _| db.get_node(NodeId::new(0)).map(drop)),
        ("upsert_nodes", |db, _| {
            db.upsert_nodes(
                &["Person"],
                "name",
                vec![HashMap::from([(
                    PropertyKey::new("name"),
                    Value::from("Mia"),
                )])],
                false,
            )
            .map(drop)
        }),
        ("create_graph", |db, _| db.create_graph("trips").map(drop)),
        ("drop_graph", |db, _| db.drop_graph("trips").map(drop)),
        ("create_property_index", |db, _| {
            db.create_property_index("name")
        }),
        ("drop_property_index", |db, _| {
            db.drop_property_index("name").map(drop)
        }),
        ("compact", |db, _| db.compact().map(drop)),
        ("wal_checkpoint", |db, _| db.wal_checkpoint()),
        ("save", |db, dir| db.save(dir.join("saved.grafeo"))),
        ("backup_full", |db, dir| {
            db.backup_full(&dir.join("backup")).map(drop)
        }),
        ("export_snapshot", |db, _| db.export_snapshot().map(drop)),
        ("restore_snapshot", |db, _| db.restore_snapshot(&[])),
        ("to_memory", |db, _| db.to_memory().map(drop)),
        ("as_of, made after close()", |db, _| {
            db.as_of(db.current_epoch()).map(drop)
        }),
        (
            "begin_transaction on a session made after close()",
            |db, _| db.session().begin_transaction(),
        ),
        (
            "execute_streaming_owned on a session made after close()",
            |db, _| db.session().execute_streaming_owned(STREAMED).map(drop),
        ),
    ];
    #[cfg(feature = "text-index")]
    calls.push(("create_text_index", |db, _| {
        db.create_text_index("Person", "name")
    }));
    #[cfg(feature = "text-index")]
    calls.push(("drop_text_index", |db, _| {
        db.drop_text_index("Person", "name").map(drop)
    }));
    #[cfg(feature = "vector-index")]
    calls.push(("create_vector_index", |db, _| {
        db.create_vector_index("Person", "embedding", Some(3), None, None, None, None)
    }));
    #[cfg(feature = "vector-index")]
    calls.push(("drop_vector_index", |db, _| {
        db.drop_vector_index("Person", "embedding").map(drop)
    }));
    calls
}

/// Every direct call fails with `T007` after `close()`, with a file and in memory: each call on
/// a database of its own, so no call's failure hides the next.
#[test]
fn every_direct_call_after_close_fails_with_the_closed_error() {
    let mut wrong = Vec::new();
    for (call, run) in direct_calls() {
        for with_file in [true, false] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("direct.grafeo");
            let mut db = people(with_file.then_some(path.as_path()));
            db.execute(
                "MATCH (a:Person {name: 'Alix'}), (g:Person {name: 'Gus'}) INSERT (a)-[:KNOWS]->(g)",
            )
            .unwrap();
            db.close().unwrap();
            let name = if with_file { "file" } else { "in memory" };
            match run(&mut db, dir.path()) {
                Err(error) if error.error_code().as_str() == "GRAFEO-T007" => {}
                Err(error) => wrong.push(format!("{name}: {call}: failed with {error}")),
                Ok(()) => wrong.push(format!("{name}: {call}: answered after close()")),
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{}",
        wrong.join(
            "
"
        )
    );
}

/// `close()` racing statements on several sessions, with a file and in memory: every statement
/// answers (before the close) or fails with `T007` (after), never with another error, and the
/// close returns, so taking each session's slot waits for its statement without a deadlock.
/// With a file, the reopen holds exactly the commits the sessions saw acknowledged.
#[test]
fn close_racing_statements_on_several_sessions() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("racing.grafeo");
    for (name, path) in [("file", Some(path.as_path())), ("in memory", None)] {
        let db = Arc::new(people(path));
        let ran = Arc::new(AtomicUsize::new(0));
        let workers: Vec<_> = (0..4)
            .map(|worker| {
                let db = Arc::clone(&db);
                let ran = Arc::clone(&ran);
                std::thread::spawn(move || {
                    let mut session = db.session();
                    let mut acknowledged = 0usize;
                    for turn in 0..1_000_000_u64 {
                        let result = if turn % 3 == 0 {
                            session
                                .begin_transaction()
                                .and_then(|()| {
                                    session
                                        .execute(&format!(
                                            "INSERT (:Person {{name: 'w{worker}-{turn}'}})"
                                        ))
                                        .map(drop)
                                })
                                .and_then(|()| session.commit())
                        } else {
                            session.execute(STREAMED).map(drop)
                        };
                        match result {
                            Ok(()) => {
                                if turn % 3 == 0 {
                                    acknowledged += 1;
                                }
                                ran.fetch_add(1, Ordering::Relaxed);
                            }
                            Err(error) if error.error_code().as_str() == "GRAFEO-T007" => {
                                return (None, acknowledged);
                            }
                            Err(error) => {
                                return (Some(format!("worker {worker}: {error}")), acknowledged);
                            }
                        }
                    }
                    (
                        Some(format!("worker {worker}: never saw the close")),
                        acknowledged,
                    )
                })
            })
            .collect();
        // Until the statements are well under way; a worker that stops early ends the wait.
        while ran.load(Ordering::Relaxed) < 200 && !workers.iter().all(|w| w.is_finished()) {
            std::thread::yield_now();
        }
        db.close().unwrap();
        let mut wrong = Vec::new();
        let mut acknowledged = 0usize;
        for worker in workers {
            let (error, commits) = worker.join().unwrap();
            wrong.extend(error);
            acknowledged += commits;
        }
        assert!(wrong.is_empty(), "{name}: {wrong:?}");
        if let Some(path) = path {
            drop(db);
            let reopened = GrafeoDB::open(path).unwrap();
            let people = reopened.execute(READ).unwrap().rows().len();
            assert_eq!(
                people,
                2 + acknowledged,
                "{name}: the fixture and every acknowledged commit, nothing else"
            );
            reopened.close().unwrap();
        }
    }
}

/// A stream a session lends (`Session::execute_streaming`) held with its session across
/// `close()`: its next pull fails with `T007`, and so does the session.
#[test]
fn a_borrowed_stream_held_across_close_fails_its_next_pull() {
    let db = people(None);
    let session = db.session();
    let mut stream = session.execute_streaming(STREAMED).unwrap();
    db.close().unwrap();
    assert_closed(stream.next_chunk(), "a pull of the borrowed stream");
    assert_closed(session.execute(READ), "a statement on its session");
}

/// A `close()` that fails before it released the database (its final checkpoint, whose new WAL
/// segment cannot start) leaves the data as it was: reads answer, while commits, writes and
/// schema changes fail with `T007`, and a session it handed out stays revoked. A later
/// `close()` persists, as the reopen shows (#712).
#[test]
#[expect(
    deprecated,
    reason = "wal() reaches the log's end, which names the segment the checkpoint starts"
)]
fn a_close_whose_checkpoint_fails_keeps_the_reads_and_refuses_the_writes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("failed.grafeo");
    // No background flusher, whose syncs would move the log's end.
    let db = GrafeoDB::with_config(
        Config::persistent(&path)
            .with_wal_durability(grafeo_engine::config::DurabilityMode::NoSync),
    )
    .unwrap();
    db.execute("INSERT (:Person {name: 'Alix'}), (:Person {name: 'Gus'})")
        .unwrap();
    let session = db.session();

    // The segment the final checkpoint starts, at the end of the log, is a directory.
    db.wal().unwrap().sync().unwrap();
    let end = db.wal().unwrap().end_lsn();
    let mut sidecar = path.as_os_str().to_owned();
    sidecar.push(".wal");
    let blocker =
        std::path::PathBuf::from(sidecar).join(grafeo_storage::wal::segment_file_name(end));
    std::fs::create_dir(&blocker).unwrap();
    assert!(db.close().is_err(), "the final checkpoint fails");

    assert!(!db.is_closed(), "the failed close() put the database back");
    assert_eq!(db.execute(READ).unwrap().rows().len(), 2, "reads answer");
    assert_closed(
        db.execute("INSERT (:Person {name: 'Mia'})"),
        "a write after the failed close()",
    );
    assert_closed(
        db.execute("CREATE GRAPH berlin"),
        "a schema change after the failed close()",
    );
    assert_closed(session.execute(READ), "the session it had handed out");

    std::fs::remove_dir(&blocker).unwrap();
    db.close().expect("the later close() persists");
    assert!(db.is_closed());
    drop(db);
    let reopened = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        reopened.execute(READ).unwrap().rows().to_vec(),
        vec![
            vec![grafeo_common::types::Value::from("Alix")],
            vec![grafeo_common::types::Value::from("Gus")]
        ],
        "the later close() wrote both, and nothing of the refused write"
    );
    reopened.close().unwrap();
}

/// Dropping the database without `close()` persists, releases the file and takes no slot: a
/// session still held reads on (the stores it holds go when it drops), a write through it fails
/// with `T007` and leaves nothing in the file, and a second open works while it is held.
#[test]
fn a_session_held_past_the_drop_reads_on_and_leaves_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dropped.grafeo");
    let db = people(Some(&path));
    let session = db.session();
    drop(db);

    let names = session.execute(READ).unwrap();
    assert_eq!(names.rows().len(), 2, "the held session reads on");
    assert_closed(
        session.execute("INSERT (:Person {name: 'Mia'})"),
        "a write after the drop",
    );
    let reopened = GrafeoDB::open(&path).expect("the drop released the file");
    let names: Vec<_> = reopened.execute(READ).unwrap().rows().to_vec();
    assert_eq!(names.len(), 2, "nothing of the refused write: {names:?}");
    assert_eq!(
        session.execute(READ).unwrap().rows().len(),
        2,
        "the held session still reads what it had"
    );
    drop(session);
    reopened.close().unwrap();
}

/// Every public method of `GrafeoDB` and `Session` reaches the open state through the checked
/// accessors, so none answers from a closed database: `live()?`, `live_arc()?`,
/// `refuse_if_closed()`, a session's `inner()?` or `inner_mut()?` (each `T007` once closed), or
/// `live_if()` / a session's `inner()` for the methods without a `Result`, which answer an
/// empty value. The rest answer from the facade's own copies, listed here by name, or call one
/// that does. The rows above check a sample of each kind at run time; this check makes a new
/// method that reaches past the accessors fail (#712).
#[test]
fn every_public_method_reaches_the_database_through_the_checked_accessors() {
    /// Tokens that reach the open state through a check.
    const CHECKED: [&str; 11] = [
        "live()?",
        "live_arc()?",
        "live_if()",
        "refuse_if_closed()",
        "inner()?",
        "inner_mut()?",
        "self.inner()",
        "self.inner_mut()",
        "check_exists()?",
        "with_one_call_session(",
        "PreparedCommit::new(self)",
    ];
    /// The methods that answer from the facade's own copies (configuration, how it was
    /// opened, what the open found, whether it is closed), and `close()` itself.
    const COPIES: [&str; 10] = [
        "GrafeoDB::close",
        "GrafeoDB::is_closed",
        "GrafeoDB::config",
        "GrafeoDB::open_warnings",
        "GrafeoDB::is_read_only",
        "GrafeoDB::is_persistent",
        "GrafeoDB::path",
        "GrafeoDB::graph_model",
        "GrafeoDB::memory_limit",
        "Session::identity",
    ];

    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut methods: Vec<(String, String)> = Vec::new();
    let mut files = Vec::new();
    for dir in ["database", "session"] {
        let mut stack = vec![src.join(dir)];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|ext| ext == "rs") {
                    files.push(path);
                }
            }
        }
    }
    for path in files {
        let text = std::fs::read_to_string(&path).unwrap();
        let mut owner: Option<&str> = None;
        let mut current: Option<(String, String)> = None;
        for line in text.lines() {
            // A blank line stays inside its block.
            if line.is_empty() {
                continue;
            }
            if !line.starts_with(' ') {
                if let Some(method) = current.take() {
                    methods.push(method);
                }
                owner = ["GrafeoDB", "Session"].into_iter().find(|name| {
                    line == format!("impl {name} {{")
                        || line == format!("impl super::{name} {{")
                        || line == format!("impl super::super::{name} {{")
                        || line == format!("impl crate::session::{name} {{")
                });
                continue;
            }
            let Some(owner) = owner else { continue };
            let signature = line
                .strip_prefix("    pub fn ")
                .or_else(|| line.strip_prefix("    pub async fn "));
            if let Some(signature) = signature {
                if let Some(method) = current.take() {
                    methods.push(method);
                }
                let name: String = signature
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                // The signature goes with the body: whether it takes `self` is read once the
                // signature, which may span lines, is whole.
                current = Some((
                    format!("{owner}::{name}"),
                    format!(
                        "{signature}
"
                    ),
                ));
            } else if let Some((_, body)) = current.as_mut() {
                body.push_str(line);
                body.push('\n');
                if line == "    }" {
                    methods.push(current.take().unwrap());
                }
            }
        }
        if let Some(method) = current.take() {
            methods.push(method);
        }
    }
    // Only methods: an associated function without `self` opens a database.
    methods.retain(|(_, body)| {
        body.split('{')
            .next()
            .is_some_and(|sig| sig.contains("self"))
    });
    assert!(
        methods.len() > 250,
        "the scan found {} methods: it no longer reads the facade",
        methods.len()
    );

    // A method passes when it reaches the state through a check, answers from a copy, or calls
    // a method of the same owner that passes.
    let mut passing: std::collections::BTreeSet<String> = methods
        .iter()
        .filter(|(name, body)| {
            COPIES.contains(&name.as_str()) || CHECKED.iter().any(|token| body.contains(token))
        })
        .map(|(name, _)| name.clone())
        .collect();
    loop {
        let before = passing.len();
        for (name, body) in &methods {
            let owner = name.split("::").next().unwrap();
            if !passing.contains(name)
                && passing.iter().any(|other| {
                    other.starts_with(owner)
                        && body.contains(&format!("self.{}(", other.split("::").nth(1).unwrap()))
                })
            {
                passing.insert(name.clone());
            }
        }
        if passing.len() == before {
            break;
        }
    }
    let unchecked: Vec<&str> = methods
        .iter()
        .map(|(name, _)| name.as_str())
        .filter(|name| !passing.contains(*name))
        .collect();
    assert!(
        unchecked.is_empty(),
        "public methods that reach the database without a check: {unchecked:?}"
    );

    // A method that returns a `Result` fails with `T007`: its body propagates the check (`?`)
    // or calls a method that does, never answering an empty value instead.
    let answers_empty: Vec<&str> = methods
        .iter()
        .filter(|(name, body)| {
            let (signature, rest) = body.split_once('{').unwrap_or((body.as_str(), ""));
            signature.contains("Result<")
                && !rest.contains('?')
                && !COPIES.contains(&name.as_str())
                && !rest.contains("PreparedCommit::new(self)")
                && !methods.iter().any(|(other, _)| {
                    other != name
                        && other.split("::").next() == name.split("::").next()
                        && rest.contains(&format!("self.{}(", other.split("::").nth(1).unwrap()))
                })
        })
        .map(|(name, _)| name.as_str())
        .collect();
    assert!(
        answers_empty.is_empty(),
        "methods returning a Result that cannot fail with T007: {answers_empty:?}"
    );
}
