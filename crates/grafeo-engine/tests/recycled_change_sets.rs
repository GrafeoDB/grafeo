//! A transaction's change set is reused by the next one, between direct
//! calls and session transactions alike, once its transaction committed or
//! rolled back. A reused set must carry nothing over: not a change, a
//! before-image or a graph slot of the transaction before. Each step below
//! reports exactly its own change events, and the database a crash leaves
//! reopens (from the WAL) to exactly the state memory held.

#![cfg(all(
    feature = "gql",
    feature = "cdc",
    feature = "wal",
    feature = "testing-statement-injection"
))]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use grafeo_common::testing::child_process;
use grafeo_common::testing::commit_hook::after_next_data_write;
use grafeo_common::testing::statement_failure::InjectedFailure;
use grafeo_common::types::{EpochId, NodeId, Value};
use grafeo_engine::cdc::ChangeKind;
use grafeo_engine::{Config, GrafeoDB};

const CHILD_VAR: &str = "GRAFEO_RECYCLED_CHANGE_SETS_CHILD";
const PATH_VAR: &str = "GRAFEO_RECYCLED_CHANGE_SETS_PATH";
const EXITED: i32 = 19;

fn config(path: &Path) -> Config {
    Config::persistent(path).with_cdc()
}

/// Fails the next direct or session write on this thread after its
/// `writes`-th data write.
fn fail_after(writes: usize) {
    after_next_data_write(move || {
        if writes > 1 {
            fail_after(writes - 1);
            Ok(())
        } else {
            Err(InjectedFailure)
        }
    });
}

/// Every node and edge of the default graph and of `archive`, one line
/// each, in ID order: what memory holds, or what a reopen rebuilt.
fn dump(db: &GrafeoDB) -> Vec<String> {
    let mut lines = Vec::new();
    for graph in ["DEFAULT", "archive"] {
        let session = db.session();
        session.execute(&format!("USE GRAPH {graph}")).unwrap();
        for query in [
            "MATCH (n) RETURN id(n), labels(n), properties(n) ORDER BY id(n)",
            "MATCH (a)-[r]->(b) RETURN id(r), type(r), id(a), id(b), properties(r) ORDER BY id(r)",
        ] {
            for row in session.execute(query).unwrap().rows() {
                lines.push(format!("{graph}: {row:?}"));
            }
        }
    }
    lines
}

/// The change events published after `since`, as (graph, kind, entity),
/// sorted; and the epoch they end at.
fn events_since(db: &GrafeoDB, since: EpochId) -> (Vec<String>, EpochId) {
    let now = db.current_epoch();
    let mut events: Vec<String> = db
        .changes_between(EpochId::new(since.as_u64() + 1), now)
        .unwrap()
        .into_iter()
        .map(|event| {
            let kind = match event.kind {
                ChangeKind::Create => "create",
                ChangeKind::Update => "update",
                ChangeKind::Delete => "delete",
                _ => "other",
            };
            format!(
                "{} {kind} {:?}",
                event.graph.as_deref().unwrap_or("default"),
                event.entity_id
            )
        })
        .collect();
    events.sort();
    (events, now)
}

/// A node with two edges to a second node, in `graph` (the default graph
/// when `None`), made by direct calls; the first node's ID.
fn hub_and_spoke(db: &GrafeoDB, graph: Option<&str>) -> NodeId {
    let handle = graph.map(|name| db.graph(name).unwrap());
    let create = |label: &str| match &handle {
        Some(handle) => handle.create_node(&[label]).unwrap(),
        None => db.create_node(&[label]).unwrap(),
    };
    let spoke = create("Spoke");
    let hub = create("Hub");
    for _ in 0..2 {
        match &handle {
            Some(handle) => handle.create_edge(spoke, hub, "LINK").unwrap(),
            None => db.create_edge(spoke, hub, "LINK").unwrap(),
        };
    }
    spoke
}

fn assignment(age: i64, city: &str) -> Value {
    Value::Map(Arc::new(BTreeMap::from([
        ("age".into(), Value::Int64(age)),
        ("city".into(), Value::from(city)),
    ])))
}

/// Alternates direct calls (committed, rolled back after a failure) and
/// session transactions (committed, rolled back), in the default graph and
/// in `archive`, and checks after each that the change events are exactly
/// its own: a set reused from the step before adds nothing.
fn alternate(db: &GrafeoDB) {
    assert!(db.create_graph("archive").unwrap());
    let person = db
        .create_node_with_props(
            &["Person"],
            [("age", Value::Int64(5)), ("city", Value::from("Amsterdam"))],
        )
        .unwrap();
    let archived = hub_and_spoke(db, Some("archive"));
    let local = hub_and_spoke(db, None);
    let mut epoch = db.current_epoch();
    let mut step = |what: &str, expected: &[&str]| {
        let (events, now) = events_since(db, epoch);
        let mut expected: Vec<String> = expected.iter().map(ToString::to_string).collect();
        expected.sort();
        let kinds: Vec<String> = events
            .iter()
            .map(|event| event.rsplit_once(' ').unwrap().0.to_string())
            .collect();
        assert_eq!(kinds, expected, "{what}: {events:?}");
        epoch = now;
    };

    // A private transaction in a named graph, committed.
    assert!(
        db.graph("archive")
            .unwrap()
            .detach_delete_node(archived)
            .unwrap()
    );
    step(
        "detach in archive",
        &["archive delete", "archive delete", "archive delete"],
    );

    // A private transaction in the default graph, rolled back.
    fail_after(2);
    db.detach_delete_node(local).unwrap_err();
    step("a failed detach", &[]);

    // A session transaction after both: only its node.
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Session {n: 1})").unwrap();
    session.commit().unwrap();
    step("a session commit", &["default create"]);

    // A session transaction rolled back, then a private one committed.
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Session {n: 2})").unwrap();
    session.rollback().unwrap();
    step("a session rollback", &[]);
    db.set_node_property(person, "*", assignment(9, "Berlin"))
        .unwrap();
    let (events, now) = events_since(db, epoch);
    assert!(
        !events.is_empty()
            && events
                .iter()
                .all(|event| event.starts_with("default update")),
        "a map assignment: {events:?}"
    );
    epoch = now;

    // A failed map assignment, then a session in the named graph.
    fail_after(1);
    db.set_node_property(person, "*", assignment(11, "Paris"))
        .unwrap_err();
    let (events, now) = events_since(db, epoch);
    assert!(events.is_empty(), "a failed map assignment: {events:?}");
    epoch = now;
    session.execute("USE GRAPH archive").unwrap();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Session {n: 3})").unwrap();
    session.commit().unwrap();
    let (events, now) = events_since(db, epoch);
    assert_eq!(events.len(), 1, "a session commit in archive: {events:?}");
    assert!(events[0].starts_with("archive create"), "{events:?}");
    epoch = now;

    // The failed detach's node is whole, and a detach now works.
    assert!(db.detach_delete_node(local).unwrap());
    let (events, _) = events_since(db, epoch);
    assert_eq!(events.len(), 3, "the detach after all of it: {events:?}");
    let found = db.get_node(person).unwrap().unwrap();
    assert_eq!(found.get_property("age"), Some(&Value::Int64(9)));
    assert_eq!(found.get_property("city"), Some(&Value::from("Berlin")));
}

/// One session transaction: an insert, committed when `i` is even, else
/// rolled back. A failed one is rolled back.
fn session_step(
    session: &mut grafeo_engine::Session,
    i: i64,
) -> grafeo_common::utils::error::Result<()> {
    session.begin_transaction()?;
    let outcome = session
        .execute(&format!("INSERT (:Session {{n: {i}}})"))
        .and_then(|_| {
            if i % 2 == 0 {
                session.commit()
            } else {
                session.rollback()
            }
        });
    if outcome.is_err() {
        let _ = session.rollback();
    }
    outcome
}

/// Direct calls (some failing) and session transactions (some rolled back)
/// on four threads at once, in both graphs.
fn concurrently(db: &Arc<GrafeoDB>) {
    assert!(db.create_graph("archive").unwrap());
    let mut workers = Vec::new();
    {
        let db = Arc::clone(db);
        workers.push(std::thread::spawn(move || {
            for i in 0..150 {
                let spoke = hub_and_spoke(&db, None);
                if i % 3 == 0 {
                    fail_after(2);
                    db.detach_delete_node(spoke).unwrap_err();
                } else {
                    assert!(db.detach_delete_node(spoke).unwrap());
                }
            }
        }));
    }
    {
        let db = Arc::clone(db);
        workers.push(std::thread::spawn(move || {
            let node = db.create_node(&["Person"]).unwrap();
            for i in 0..150_i64 {
                if i % 4 == 0 {
                    fail_after(1);
                    db.set_node_property(node, "*", assignment(-i, "Prague"))
                        .unwrap_err();
                } else {
                    db.set_node_property(node, "*", assignment(i, "Barcelona"))
                        .unwrap();
                }
            }
        }));
    }
    for graph in ["DEFAULT", "archive"] {
        let db = Arc::clone(db);
        workers.push(std::thread::spawn(move || {
            let mut session = db.session();
            session.execute(&format!("USE GRAPH {graph}")).unwrap();
            for i in 0..150_i64 {
                for _ in 0..50 {
                    match session_step(&mut session, i) {
                        Ok(()) => break,
                        Err(err)
                            if err.error_code()
                                == grafeo_common::utils::error::ErrorCode::TransactionConflict => {}
                        Err(err) => panic!("{graph} session {i}: {err}"),
                    }
                }
            }
        }));
    }
    for worker in workers {
        worker.join().expect("a worker failed");
    }
    // Every committed session insert, and only those.
    for graph in ["DEFAULT", "archive"] {
        let session = db.session();
        session.execute(&format!("USE GRAPH {graph}")).unwrap();
        let count = session
            .execute("MATCH (s:Session) RETURN count(s)")
            .unwrap()
            .scalar::<i64>()
            .unwrap();
        assert_eq!(count, 75, "{graph}: the committed session inserts");
    }
}

/// The children: run the scenario on a persistent database, write what
/// memory holds, exit without `close()`. A no-op when run directly.
#[test]
fn recycled_child() {
    let (Ok(which), Some(path)) = (std::env::var(CHILD_VAR), std::env::var_os(PATH_VAR)) else {
        return;
    };
    let path = PathBuf::from(path);
    let db = Arc::new(GrafeoDB::with_config(config(&path)).unwrap());
    match which.as_str() {
        "alternate" => alternate(&db),
        "concurrent" => concurrently(&db),
        other => panic!("unknown child {other}"),
    }
    std::fs::write(path.with_extension("dump"), dump(&db).join("\n")).unwrap();
    std::fs::write(
        path.with_extension("epoch"),
        db.current_epoch().as_u64().to_string(),
    )
    .unwrap();
    std::process::exit(EXITED);
}

/// Runs the child `which`, then reopens its database: the same nodes,
/// edges and properties as memory held, at the same epoch.
fn crash_and_compare(which: &str) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph.grafeo");
    let output = child_process::output(
        Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "recycled_child", "--nocapture"])
            .env(CHILD_VAR, which)
            .env(PATH_VAR, &path),
    )
    .unwrap();
    assert_eq!(
        output.status.code(),
        Some(EXITED),
        "the child {which} exited early:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let memory = std::fs::read_to_string(path.with_extension("dump")).unwrap();
    let epoch: u64 = std::fs::read_to_string(path.with_extension("epoch"))
        .unwrap()
        .parse()
        .unwrap();
    let db = GrafeoDB::with_config(config(&path)).unwrap();
    let reopened = dump(&db).join("\n");
    assert_eq!(reopened, memory, "{which}: the reopen matches memory");
    assert_eq!(db.current_epoch().as_u64(), epoch, "{which}");
    db.close().unwrap();
}

#[test]
fn a_reused_set_carries_nothing_into_the_next_transaction() {
    crash_and_compare("alternate");
}

#[test]
fn direct_calls_and_sessions_at_once_reopen_as_memory_held() {
    crash_and_compare("concurrent");
}
