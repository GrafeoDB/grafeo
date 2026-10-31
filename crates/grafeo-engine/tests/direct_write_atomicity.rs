//! A direct call must undo completed writes if a later step fails or panics.

#![cfg(all(feature = "gql", feature = "testing-statement-injection"))]

use grafeo_common::testing::commit_hook::after_next_data_write;
use grafeo_common::testing::statement_failure::InjectedFailure;
use grafeo_common::types::{EdgeId, NodeId, Value};
use grafeo_core::graph::Direction;
use grafeo_engine::GrafeoDB;
use std::panic::{AssertUnwindSafe, catch_unwind};

struct Graph {
    center: NodeId,
    other: NodeId,
    edges: Vec<EdgeId>,
}

fn graph(db: &GrafeoDB) -> Graph {
    let center = db
        .create_node_with_props(&["Center"], [("name", Value::from("center"))])
        .unwrap();
    let other = db
        .create_node_with_props(&["Other"], [("name", Value::from("other"))])
        .unwrap();
    let edges = [
        (center, other),
        (center, other),
        (other, center),
        (center, center),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, (src, dst))| {
        db.create_edge_with_props(
            src,
            dst,
            "LINK",
            [("weight", Value::Int64(i64::try_from(i).unwrap() + 1))],
        )
        .unwrap()
    })
    .collect();
    Graph {
        center,
        other,
        edges,
    }
}

fn unchanged(db: &GrafeoDB, graph: &Graph) {
    let mut actual = db
        .get_node_edges(graph.center, Direction::Both, &[])
        .unwrap();
    actual.sort_by_key(|(_, edge)| edge.as_u64());
    let mut expected = vec![
        (graph.other, graph.edges[0]),
        (graph.other, graph.edges[1]),
        (graph.other, graph.edges[2]),
        (graph.center, graph.edges[3]),
    ];
    expected.sort_by_key(|(_, edge)| edge.as_u64());
    assert_eq!(
        actual, expected,
        "all parallel edges and the self-loop remain"
    );
    assert!(
        db.get_node(graph.center)
            .unwrap()
            .unwrap()
            .has_label("Center")
    );
    assert_eq!(db.node_count().unwrap(), 2);
    assert_eq!(db.edge_count().unwrap(), 4);
    for (index, id) in graph.edges.iter().enumerate() {
        let edge = db.get_edge(*id).unwrap().expect("original edge remains");
        assert_eq!(
            edge.get_property("weight"),
            Some(&Value::Int64(i64::try_from(index).unwrap() + 1))
        );
    }
}

fn fail_after(writes: usize, panic: bool) {
    after_next_data_write(move || {
        if writes > 1 {
            fail_after(writes - 1, panic);
            Ok(())
        } else if panic {
            panic!("injected data-write panic");
        } else {
            Err(InjectedFailure)
        }
    });
}

#[test]
fn direct_detach_error_restores_every_completed_step() {
    for after in 1..=5 {
        let db = GrafeoDB::new_in_memory();
        let graph = graph(&db);
        let before = db.current_epoch();
        fail_after(after, false);
        let error = db.detach_delete_node(graph.center).unwrap_err();
        assert!(
            error.to_string().contains("injected data-write failure"),
            "{error}"
        );
        unchanged(&db, &graph);
        assert_eq!(
            db.current_epoch(),
            before,
            "a failed call publishes nothing"
        );
        assert!(db.detach_delete_node(graph.center).unwrap());
        assert_eq!(db.node_count().unwrap(), 1);
        assert_eq!(db.edge_count().unwrap(), 0);
    }
}

#[test]
fn direct_detach_panic_restores_every_completed_step() {
    for after in 1..=5 {
        let db = GrafeoDB::new_in_memory();
        let graph = graph(&db);
        let before = db.current_epoch();
        fail_after(after, true);
        let panic =
            catch_unwind(AssertUnwindSafe(|| db.detach_delete_node(graph.center))).unwrap_err();
        assert_eq!(
            panic.downcast_ref::<&str>(),
            Some(&"injected data-write panic")
        );
        unchanged(&db, &graph);
        assert_eq!(
            db.current_epoch(),
            before,
            "an unwound call publishes nothing"
        );
        assert!(db.detach_delete_node(graph.center).unwrap());
        assert_eq!(db.node_count().unwrap(), 1);
        assert_eq!(db.edge_count().unwrap(), 0);
    }
}

/// Runs `call` with a failure injected after its `after`-th write: an error,
/// or a panic, which is caught. Panics when the call does not fail there.
fn fails_after<T: std::fmt::Debug>(
    after: usize,
    panic: bool,
    call: impl FnOnce() -> grafeo_common::utils::error::Result<T>,
    what: &str,
) {
    fail_after(after, panic);
    if panic {
        let payload = catch_unwind(AssertUnwindSafe(call))
            .expect_err(&format!("{what}: the injected panic after write {after}"));
        assert_eq!(
            payload.downcast_ref::<&str>(),
            Some(&"injected data-write panic"),
            "{what}"
        );
    } else {
        let error = call().expect_err(&format!("{what}: the injected error after write {after}"));
        assert!(
            error.to_string().contains("injected data-write failure"),
            "{what}: {error}"
        );
    }
}

/// Five single-write calls on the `:Base` node `base`, each failing after
/// its one write.
fn single_calls_fail_after_their_write(db: &GrafeoDB, base: NodeId, panic: bool) {
    fails_after(
        1,
        panic,
        || db.create_node_with_props(&["Failed"], [("value", Value::Int64(7))]),
        "create_node_with_props",
    );
    fails_after(
        1,
        panic,
        || db.set_node_property(base, "age", Value::Int64(31)),
        "set_node_property",
    );
    fails_after(
        1,
        panic,
        || db.add_node_label(base, "Tagged"),
        "add_node_label",
    );
    fails_after(
        1,
        panic,
        || db.create_edge(base, base, "SELF"),
        "create_edge",
    );
    fails_after(
        1,
        panic,
        || db.remove_node_property(base, "name"),
        "remove_node_property",
    );
}

/// What [`single_calls_fail_after_their_write`] leaves: each call's one
/// write, committed.
fn single_calls_committed(db: &GrafeoDB, base: NodeId) {
    assert_eq!(
        db.execute("MATCH (n:Failed) RETURN n.value")
            .unwrap()
            .rows(),
        &[vec![Value::Int64(7)]],
        "the created node, with its property"
    );
    let node = db.get_node(base).unwrap().unwrap();
    assert_eq!(node.get_property("age"), Some(&Value::Int64(31)));
    assert!(node.has_label("Base") && node.has_label("Tagged"));
    assert_eq!(node.get_property("name"), None);
    assert_eq!(
        db.get_node_edges(base, Direction::Outgoing, &[])
            .unwrap()
            .len(),
        1,
        "the self-loop"
    );
}

/// A call of one write commits at once: once that write is applied and
/// recorded it is committed, also when the call fails after it (only a
/// failure to record it can leave memory and the log apart, and that poisons
/// the database). The call reports the failure; memory, the published epoch
/// (one per call) and the next call agree, and no ID is reused.
#[test]
fn a_single_call_that_fails_after_its_write_has_committed_it() {
    for panic in [false, true] {
        let db = GrafeoDB::new_in_memory();
        let base = db
            .create_node_with_props(&["Base"], [("name", Value::from("Alix"))])
            .unwrap();
        let before = db.current_epoch();
        single_calls_fail_after_their_write(&db, base, panic);
        single_calls_committed(&db, base);
        assert_eq!(
            db.current_epoch().as_u64(),
            before.as_u64() + 5,
            "each failed call published its write: panic={panic}"
        );
        let failed = db
            .execute("MATCH (n:Failed) RETURN id(n)")
            .unwrap()
            .scalar::<i64>()
            .unwrap();
        let next = db.create_node(&["After"]).unwrap();
        assert!(
            next.as_u64() > u64::try_from(failed).unwrap(),
            "no ID is reused"
        );
        assert_eq!(db.node_count().unwrap(), 3);
    }
}

#[test]
fn failed_immediate_property_map_restores_values_and_index() {
    use std::collections::BTreeMap;
    use std::sync::Arc;
    let db = GrafeoDB::new_in_memory();
    let id = db
        .create_node_with_props(
            &["Person"],
            [("age", Value::Int64(5)), ("name", Value::from("Alix"))],
        )
        .unwrap();
    db.create_property_index("age").unwrap();
    let before = db.current_epoch();
    for after in 1..=2 {
        fail_after(after, false);
        let map = Value::Map(Arc::new(BTreeMap::from([
            ("age".into(), Value::Int64(9)),
            ("name".into(), Value::from("Changed")),
        ])));
        assert!(db.set_node_property(id, "*", map).is_err());
        let node = db.get_node(id).unwrap().unwrap();
        assert_eq!(node.get_property("age"), Some(&Value::Int64(5)));
        assert_eq!(node.get_property("name"), Some(&Value::from("Alix")));
        assert_eq!(
            db.find_nodes_by_property("age", &Value::Int64(5)).unwrap(),
            [id]
        );
        let changed = db.find_nodes_by_property("age", &Value::Int64(9)).unwrap();
        assert!(changed.is_empty(), "{changed:?}");
        assert_eq!(db.current_epoch(), before);
    }
    db.set_node_property(id, "age", Value::Int64(6)).unwrap();
    assert_eq!(
        db.find_nodes_by_property("age", &Value::Int64(6)).unwrap(),
        [id]
    );
}

/// The node type of [`typed_label_node`]: a FLOAT64 score.
const TYPED: &str = "CREATE NODE TYPE Typed (score FLOAT64)";

/// A `:Plain` node with an integer score, which a `Typed` label converts
/// to a float: the label and the conversion are two writes.
fn typed_label_node(db: &GrafeoDB, tag: &str) -> NodeId {
    db.create_node_with_props(
        &["Plain"],
        [("tag", Value::from(tag)), ("score", Value::Int64(5))],
    )
    .unwrap()
}

/// Asserts that `node` of [`typed_label_node`] has no `Typed` label and its
/// integer score.
fn typed_label_unchanged(db: &GrafeoDB, node: NodeId, what: &str) {
    let node = db.get_node(node).unwrap().unwrap();
    assert!(node.has_label("Plain"), "{what}");
    assert!(!node.has_label("Typed"), "{what}: the label is taken back");
    assert_eq!(
        node.get_property("score"),
        Some(&Value::Int64(5)),
        "{what}: the converted value is taken back"
    );
}

/// Asserts that `node` of [`typed_label_node`] has the `Typed` label and
/// the converted score.
fn typed_label_added(db: &GrafeoDB, node: NodeId, what: &str) {
    let node = db.get_node(node).unwrap().unwrap();
    assert!(node.has_label("Typed"), "{what}");
    assert_eq!(
        node.get_property("score"),
        Some(&Value::Float64(5.0)),
        "{what}"
    );
}

/// With node types in the catalog, adding a label may convert the node's
/// values: one write for the label and one per value, so the call is a
/// private transaction, all or nothing. A failure after either write leaves
/// the node, the label counts, the published epoch and change data capture
/// as they were, and the next call works.
#[test]
fn a_failed_label_change_that_converts_values_restores_the_node() {
    use grafeo_engine::{Config, HistoryRetention};

    for retention in [HistoryRetention::Off, HistoryRetention::Forever] {
        for panic in [false, true] {
            for after in 1..=2 {
                let what = format!("after write {after}, panic={panic}, retention={retention:?}");
                let db =
                    GrafeoDB::with_config(Config::in_memory().with_history_retention(retention))
                        .unwrap();
                db.execute(TYPED).unwrap();
                #[cfg(feature = "cdc")]
                db.set_cdc_enabled(true);
                let node = typed_label_node(&db, "Alix");
                let before = db.current_epoch();
                #[cfg(feature = "cdc")]
                let cdc_before = cdc_events(&db);

                fails_after(after, panic, || db.add_node_label(node, "Typed"), &what);
                typed_label_unchanged(&db, node, &what);
                assert_eq!(db.current_epoch(), before, "{what}: nothing published");
                assert_eq!(
                    db.execute("MATCH (n:Typed) RETURN count(n)")
                        .unwrap()
                        .rows(),
                    &[vec![Value::Int64(0)]],
                    "{what}"
                );
                #[cfg(feature = "cdc")]
                assert_eq!(cdc_events(&db), cdc_before, "{what}: no change events");

                assert!(db.add_node_label(node, "Typed").unwrap(), "{what}");
                typed_label_added(&db, node, &what);
            }
        }
    }
}

/// Every change event of `db`, as JSON.
#[cfg(feature = "cdc")]
fn cdc_events(db: &GrafeoDB) -> serde_json::Value {
    use grafeo_common::types::EpochId;
    serde_json::to_value(
        db.changes_between(EpochId::new(0), EpochId::new(u64::MAX))
            .unwrap(),
    )
    .unwrap()
}

#[cfg(feature = "wal")]
#[test]
fn failed_direct_detach_reopens_without_an_intervening_write() {
    use grafeo_engine::{Config, HistoryRetention};

    #[cfg(feature = "cdc")]
    use grafeo_common::types::EpochId;

    for retention in [HistoryRetention::Off, HistoryRetention::Forever] {
        for panic in [false, true] {
            for after in 1..=5 {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("graph.grafeo");
                let db = GrafeoDB::with_config(
                    Config::persistent(&path).with_history_retention(retention),
                )
                .unwrap();
                #[cfg(feature = "cdc")]
                db.set_cdc_enabled(true);
                let graph = graph(&db);
                let before = db.current_epoch();
                #[cfg(feature = "cdc")]
                let cdc_before = {
                    let events = db
                        .changes_between(EpochId::new(0), EpochId::new(u64::MAX))
                        .unwrap();
                    assert_eq!(events.len(), 6, "both nodes and all four edges emit events");
                    serde_json::to_value(events).unwrap()
                };

                fail_after(after, panic);
                if panic {
                    let payload =
                        catch_unwind(AssertUnwindSafe(|| db.detach_delete_node(graph.center)))
                            .unwrap_err();
                    assert_eq!(
                        payload.downcast_ref::<&str>(),
                        Some(&"injected data-write panic"),
                        "the requested panic fires after step {after}"
                    );
                } else {
                    let error = db.detach_delete_node(graph.center).unwrap_err();
                    assert!(
                        error.to_string().contains("injected data-write failure"),
                        "the requested error fires after step {after}: {error}"
                    );
                }
                unchanged(&db, &graph);
                assert_eq!(
                    db.current_epoch(),
                    before,
                    "failed step {after}, panic={panic}, retention={retention:?}"
                );
                #[cfg(feature = "cdc")]
                assert_eq!(
                    serde_json::to_value(
                        db.changes_between(EpochId::new(0), EpochId::new(u64::MAX))
                            .unwrap()
                    )
                    .unwrap(),
                    cdc_before,
                    "a failed call exposes no CDC events: step {after}, panic={panic}, retention={retention:?}"
                );

                // Close checkpoints immediately: no successful write can advance
                // the published epoch and conceal a failed call's store epoch.
                db.close().unwrap();
                drop(db);
                let reopened = GrafeoDB::open(&path).unwrap();
                unchanged(&reopened, &graph);
                assert_eq!(reopened.config().history_retention(), retention);
                assert_eq!(
                    reopened.current_epoch(),
                    before,
                    "reopen retains the published epoch: step {after}, panic={panic}, retention={retention:?}"
                );
                assert!(reopened.detach_delete_node(graph.center).unwrap());
                assert_eq!(reopened.node_count().unwrap(), 1);
                assert_eq!(reopened.edge_count().unwrap(), 0);
                reopened.close().unwrap();
            }
        }
    }
}

#[cfg(feature = "wal")]
#[test]
fn successful_direct_graph_reopens_at_its_published_epoch() {
    use grafeo_engine::{Config, HistoryRetention};

    for retention in [HistoryRetention::Off, HistoryRetention::Forever] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("graph.grafeo");
        let db = GrafeoDB::with_config(Config::persistent(&path).with_history_retention(retention))
            .unwrap();
        let graph = graph(&db);
        let before = db.current_epoch();
        assert_eq!(before.as_u64(), 6, "the six direct calls committed");
        unchanged(&db, &graph);

        db.close().unwrap();
        drop(db);
        let reopened = GrafeoDB::open(&path).unwrap();
        unchanged(&reopened, &graph);
        assert_eq!(reopened.config().history_retention(), retention);
        assert_eq!(
            reopened.current_epoch(),
            before,
            "a successful graph reopens at its published epoch: retention={retention:?}"
        );
        reopened.close().unwrap();
    }
}

/// The same calls with a real crash: a child process makes them on a
/// database with a WAL and exits without `close()` (which would checkpoint
/// and hide the replay), then the reopen replays the WAL. A routed call that
/// failed comes back as none of it, one that succeeded as all of it, and a
/// single call that failed after its write as that write; the published
/// epoch is the one the child ended at.
#[cfg(feature = "wal")]
mod crash {
    use super::*;
    use grafeo_common::testing::child_process;
    use grafeo_engine::{Config, HistoryRetention};
    use std::collections::{BTreeMap, HashMap};
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::sync::Arc;

    /// Which child, and with which history retention: `<scenario>:<off|forever>`.
    const CHILD_VAR: &str = "GRAFEO_DIRECT_WRITE_ATOMICITY_CHILD";
    const PATH_VAR: &str = "GRAFEO_DIRECT_WRITE_ATOMICITY_PATH";
    /// Exit code of a child that reached its end.
    const EXITED: i32 = 19;

    const RETENTIONS: [(&str, HistoryRetention); 2] = [
        ("off", HistoryRetention::Off),
        ("forever", HistoryRetention::Forever),
    ];

    fn retention(name: &str) -> HistoryRetention {
        match RETENTIONS.iter().find(|(known, _)| *known == name) {
            Some((_, retention)) => *retention,
            None => panic!("unknown retention {name}"),
        }
    }

    /// The database a child crashes on and the parent reopens.
    fn open(path: &Path, retention: HistoryRetention) -> GrafeoDB {
        GrafeoDB::with_config(Config::persistent(path).with_history_retention(retention)).unwrap()
    }

    /// The sidecar WAL of the database file at `path`, which a process that
    /// ends without `close()` leaves behind.
    fn sidecar_wal(path: &Path) -> PathBuf {
        let mut sidecar = path.as_os_str().to_owned();
        sidecar.push(".wal");
        PathBuf::from(sidecar)
    }

    /// The IDs and epochs a child reports, one `<key> <numbers...>` line each.
    fn report_path(path: &Path) -> PathBuf {
        path.with_extension("report")
    }

    /// Runs the child `which` on a new database, then reopens it (the WAL a
    /// crash leaves is there) and returns it with what the child reported.
    fn crash_and_reopen(which: &str) -> (tempfile::TempDir, PathBuf, Vec<(String, Vec<u64>)>) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("graph.grafeo");
        let output = child_process::output(
            Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "crash::direct_write_child", "--nocapture"])
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
        assert!(
            sidecar_wal(&path).exists(),
            "{which}: a crash leaves the WAL behind"
        );
        let report = std::fs::read_to_string(report_path(&path))
            .unwrap()
            .lines()
            .map(|line| {
                let mut words = line.split_whitespace();
                let key = words.next().unwrap().to_string();
                (key, words.map(|word| word.parse().unwrap()).collect())
            })
            .collect();
        (dir, path, report)
    }

    /// The one entry `key` of a report.
    fn entry<'a>(report: &'a [(String, Vec<u64>)], key: &str) -> &'a [u64] {
        let mut found = report.iter().filter(|(known, _)| known == key);
        let (_, numbers) = found.next().unwrap_or_else(|| panic!("no {key}"));
        assert!(found.next().is_none(), "one {key}");
        numbers
    }

    fn graph_line(graph: &Graph) -> String {
        [graph.center.as_u64(), graph.other.as_u64()]
            .into_iter()
            .chain(graph.edges.iter().map(|edge| edge.as_u64()))
            .fold(String::from("detach"), |line, id| {
                line + " " + &id.to_string()
            })
    }

    fn graph_of(numbers: &[u64]) -> Graph {
        Graph {
            center: NodeId::new(numbers[0]),
            other: NodeId::new(numbers[1]),
            edges: numbers[2..].iter().map(|id| EdgeId::new(*id)).collect(),
        }
    }

    /// Like `unchanged`, for one of several graphs in the database.
    fn graph_unchanged(db: &GrafeoDB, graph: &Graph, what: &str) {
        let mut actual = db
            .get_node_edges(graph.center, Direction::Both, &[])
            .unwrap();
        actual.sort_by_key(|(_, edge)| edge.as_u64());
        actual.dedup();
        let mut expected = vec![
            (graph.other, graph.edges[0]),
            (graph.other, graph.edges[1]),
            (graph.other, graph.edges[2]),
            (graph.center, graph.edges[3]),
        ];
        expected.sort_by_key(|(_, edge)| edge.as_u64());
        assert_eq!(actual, expected, "{what}: every edge remains");
        assert!(
            db.get_node(graph.center)
                .unwrap()
                .unwrap_or_else(|| panic!("{what}: the node remains"))
                .has_label("Center")
        );
        for (index, id) in graph.edges.iter().enumerate() {
            let edge = db.get_edge(*id).unwrap().expect("original edge remains");
            assert_eq!(
                edge.get_property("weight"),
                Some(&Value::Int64(i64::try_from(index).unwrap() + 1)),
                "{what}"
            );
        }
    }

    /// A `:Person` node with an indexed age and a city, which [`assignment`]
    /// sets both of: two writes.
    fn map_node(db: &GrafeoDB) -> NodeId {
        db.create_node_with_props(
            &["Person"],
            [
                ("tag", Value::from("map")),
                ("age", Value::Int64(5)),
                ("city", Value::from("Amsterdam")),
            ],
        )
        .unwrap()
    }

    fn assignment() -> Value {
        Value::Map(Arc::new(BTreeMap::from([
            ("tag".into(), Value::from("map")),
            ("age".into(), Value::Int64(9)),
            ("city".into(), Value::from("Berlin")),
        ])))
    }

    fn map_node_holds(db: &GrafeoDB, node: NodeId, age: i64, city: &str, what: &str) {
        let found = db.get_node(node).unwrap().unwrap();
        assert_eq!(
            found.get_property("age"),
            Some(&Value::Int64(age)),
            "{what}"
        );
        assert_eq!(
            found.get_property("city"),
            Some(&Value::from(city)),
            "{what}"
        );
        for (indexed, expected) in [(5, age == 5), (9, age == 9)] {
            let ids = db
                .find_nodes_by_property("age", &Value::Int64(indexed))
                .unwrap();
            assert_eq!(
                ids.contains(&node),
                expected,
                "{what}: the age index finds the node under {indexed}"
            );
        }
    }

    /// Makes a `:Base` node, then a single call whose change cannot be
    /// recorded (the one way a single call fails after its write in real
    /// code): the call fails and poisons the database. Its write is applied
    /// but not in the WAL, so its epoch is never published: reads do not see
    /// it, and every commit and checkpoint is refused until a reopen.
    fn poisoned(db: &GrafeoDB) -> NodeId {
        let base = db
            .create_node_with_props(&["Base"], [("name", Value::from("Alix"))])
            .unwrap();
        let before = db.current_epoch();
        grafeo_common::testing::commit_hook::fail_next_change_record();
        db.create_node_with_props(&["Lost"], [("value", Value::Int64(7))])
            .expect_err("a change that cannot be recorded fails the call");
        assert_eq!(db.current_epoch(), before, "nothing is published");
        assert_eq!(
            db.execute("MATCH (n:Lost) RETURN count(n)").unwrap().rows(),
            &[vec![Value::Int64(0)]],
            "a read does not see the unlogged write"
        );
        for (what, result) in [
            ("a direct call", db.create_node(&["After"]).map(|_| ())),
            ("a query", db.execute("INSERT (:After)").map(|_| ())),
            ("a checkpoint", db.wal_checkpoint()),
        ] {
            let err = result.expect_err(&format!("{what} after the poison"));
            assert_eq!(
                err.error_code(),
                grafeo_common::utils::error::ErrorCode::IncompleteCommit,
                "{what}: {err}"
            );
        }
        base
    }

    /// After a crash the reopen matches the log, which does not have the
    /// write that poisoned the database, and takes writes again.
    #[test]
    fn a_change_that_cannot_be_recorded_poisons_and_the_reopen_drops_it() {
        for (kept, retention) in RETENTIONS {
            let which = format!("poison:{kept}");
            let (_dir, path, report) = crash_and_reopen(&which);
            let db = open(&path, retention);
            assert!(
                db.get_node(NodeId::new(entry(&report, "base")[0]))
                    .unwrap()
                    .is_some(),
                "{which}: the write before the poison"
            );
            assert_eq!(
                db.execute("MATCH (n:Lost) RETURN count(n)").unwrap().rows(),
                &[vec![Value::Int64(0)]],
                "{which}: the unlogged write is gone"
            );
            assert_eq!(db.current_epoch().as_u64(), entry(&report, "epoch")[0]);
            db.create_node(&["After"]).expect("the reopen takes writes");
            db.close().unwrap();
        }
    }

    /// The same without a crash: closing the poisoned database writes no
    /// checkpoint of the unlogged write, so the reopen drops it too.
    #[test]
    fn closing_a_poisoned_database_keeps_the_unlogged_write_out() {
        for (_, retention) in RETENTIONS {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("graph.grafeo");
            let db = open(&path, retention);
            let base = poisoned(&db);
            let _ = db.close();
            drop(db);
            let db = open(&path, retention);
            assert!(db.get_node(base).unwrap().is_some(), "{retention:?}");
            assert_eq!(
                db.execute("MATCH (n:Lost) RETURN count(n)").unwrap().rows(),
                &[vec![Value::Int64(0)]],
                "{retention:?}: the unlogged write is gone"
            );
            db.create_node(&["After"]).expect("the reopen takes writes");
            db.close().unwrap();
        }
    }

    /// The child processes: each makes its calls, reports what the parent
    /// checks and exits without `close()`. A no-op when run directly.
    #[test]
    fn direct_write_child() {
        let (Ok(which), Some(path)) = (std::env::var(CHILD_VAR), std::env::var_os(PATH_VAR)) else {
            return;
        };
        let path = PathBuf::from(path);
        let (scenario, kept) = which.split_once(':').unwrap();
        let db = open(&path, retention(kept));
        let mut report = Vec::new();
        match scenario {
            "single-error" | "single-panic" => {
                let base = db
                    .create_node_with_props(&["Base"], [("name", Value::from("Alix"))])
                    .unwrap();
                single_calls_fail_after_their_write(&db, base, scenario == "single-panic");
                single_calls_committed(&db, base);
                report.push(format!("base {}", base.as_u64()));
            }
            "routed-fail" => {
                db.execute(TYPED).unwrap();
                db.create_property_index("age").unwrap();
                for panic in [false, true] {
                    for after in 1..=5 {
                        let graph = graph(&db);
                        fails_after(
                            after,
                            panic,
                            || db.detach_delete_node(graph.center),
                            &format!("detach after write {after}, panic={panic}"),
                        );
                        report.push(graph_line(&graph));
                    }
                    for after in 1..=2 {
                        let node = map_node(&db);
                        fails_after(
                            after,
                            panic,
                            || db.set_node_property(node, "*", assignment()),
                            &format!("map assignment after write {after}, panic={panic}"),
                        );
                        report.push(format!("map {}", node.as_u64()));
                        let node = typed_label_node(&db, "label");
                        fails_after(
                            after,
                            panic,
                            || db.add_node_label(node, "Typed"),
                            &format!("label after write {after}, panic={panic}"),
                        );
                        report.push(format!("label {}", node.as_u64()));
                    }
                }
            }
            "routed-ok" => {
                db.execute(TYPED).unwrap();
                db.create_property_index("age").unwrap();
                let graph = graph(&db);
                assert!(db.detach_delete_node(graph.center).unwrap());
                report.push(graph_line(&graph));
                let node = map_node(&db);
                db.set_node_property(node, "*", assignment()).unwrap();
                report.push(format!("map {}", node.as_u64()));
                let node = typed_label_node(&db, "label");
                assert!(db.add_node_label(node, "Typed").unwrap());
                report.push(format!("label {}", node.as_u64()));
            }
            // Raw writes to the root store only, a checkpoint, then a commit
            // of the engine. The checkpoint adopts the epoch the raw writes
            // stamped, so the commit takes one above it: before, it took one
            // below, and read the raw node without its name.
            "raw-root" => {
                let store = db.store().unwrap();
                let alix = store.create_node(&["Person"]);
                store.set_node_property(alix, "name", Value::from("Alix"));
                let stamped = store.current_epoch();
                db.wal_checkpoint().unwrap();
                let checkpointed = db.current_epoch();
                assert_eq!(checkpointed, stamped, "the checkpoint adopts the raw epoch");
                let gus = db
                    .create_node_with_props(&["Person"], [("name", Value::from("Gus"))])
                    .unwrap();
                assert!(db.current_epoch() > checkpointed);
                assert_eq!(
                    db.execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
                        .unwrap()
                        .rows(),
                    [[Value::from("Alix")], [Value::from("Gus")]],
                    "the commit reads the raw writes"
                );
                report.push(format!("raw {}", alix.as_u64()));
                report.push(format!("engine {}", gus.as_u64()));
                report.push(format!("checkpointed {}", checkpointed.as_u64()));
            }
            // A single call whose change cannot be recorded: the database is
            // poisoned, so the child ends here, without the call after it.
            "poison" => {
                let base = poisoned(&db);
                report.push(format!("base {}", base.as_u64()));
                report.push(format!("epoch {}", db.current_epoch().as_u64()));
                std::fs::write(report_path(&path), report.join("\n")).unwrap();
                std::process::exit(EXITED);
            }
            other => panic!("unknown child {other}"),
        }
        // The call after the failed ones works.
        let after = db.create_node(&["After"]).unwrap();
        report.push(format!("after {}", after.as_u64()));
        report.push(format!("epoch {}", db.current_epoch().as_u64()));
        std::fs::write(report_path(&path), report.join("\n")).unwrap();
        // Crash: no close(), no checkpoint, no destructors.
        std::process::exit(EXITED);
    }

    /// The parts of a reopened database every child leaves: the node of the
    /// call after the failed ones, and the published epoch.
    fn after_and_epoch(db: &GrafeoDB, report: &[(String, Vec<u64>)], what: &str) {
        let after = NodeId::new(entry(report, "after")[0]);
        assert!(
            db.get_node(after)
                .unwrap()
                .unwrap_or_else(|| panic!("{what}: the call after the failed ones"))
                .has_label("After")
        );
        assert_eq!(
            db.current_epoch().as_u64(),
            entry(report, "epoch")[0],
            "{what}: the reopen publishes the epoch the child ended at"
        );
    }

    #[test]
    fn single_calls_that_failed_after_their_write_come_back_with_it() {
        for scenario in ["single-error", "single-panic"] {
            for (kept, retention) in RETENTIONS {
                let which = format!("{scenario}:{kept}");
                let (_dir, path, report) = crash_and_reopen(&which);
                let db = open(&path, retention);
                single_calls_committed(&db, NodeId::new(entry(&report, "base")[0]));
                after_and_epoch(&db, &report, &which);
                db.close().unwrap();
            }
        }
    }

    #[test]
    fn routed_calls_that_failed_come_back_as_none_of_it() {
        for (kept, retention) in RETENTIONS {
            let which = format!("routed-fail:{kept}");
            let (_dir, path, report) = crash_and_reopen(&which);
            let db = open(&path, retention);
            let mut counts = HashMap::new();
            for (key, numbers) in &report {
                *counts.entry(key.as_str()).or_insert(0) += 1;
                match key.as_str() {
                    "detach" => graph_unchanged(&db, &graph_of(numbers), &which),
                    "map" => map_node_holds(&db, NodeId::new(numbers[0]), 5, "Amsterdam", &which),
                    "label" => typed_label_unchanged(&db, NodeId::new(numbers[0]), &which),
                    _ => {}
                }
            }
            assert_eq!(
                (counts["detach"], counts["map"], counts["label"]),
                (10, 4, 4),
                "{which}: every failed call is checked"
            );
            after_and_epoch(&db, &report, &which);
            db.close().unwrap();
        }
    }

    #[test]
    fn routed_calls_that_succeeded_come_back_as_all_of_it() {
        for (kept, retention) in RETENTIONS {
            let which = format!("routed-ok:{kept}");
            let (_dir, path, report) = crash_and_reopen(&which);
            let db = open(&path, retention);
            let graph = graph_of(entry(&report, "detach"));
            assert!(db.get_node(graph.center).unwrap().is_none(), "{which}");
            assert!(db.get_node(graph.other).unwrap().is_some(), "{which}");
            for edge in &graph.edges {
                assert!(db.get_edge(*edge).unwrap().is_none(), "{which}: {edge:?}");
            }
            assert!(
                db.get_node_edges(graph.other, Direction::Both, &[])
                    .unwrap()
                    .is_empty(),
                "{which}"
            );
            map_node_holds(
                &db,
                NodeId::new(entry(&report, "map")[0]),
                9,
                "Berlin",
                &which,
            );
            typed_label_added(&db, NodeId::new(entry(&report, "label")[0]), &which);
            after_and_epoch(&db, &report, &which);
            db.close().unwrap();
        }
    }

    /// Raw writes to the root store only (no named graph), a checkpoint and
    /// a later commit: the reopen after a crash replays the commit, so both
    /// the raw node and the committed one are there, at the epoch the child
    /// ended at; a second reopen, after a clean close, too.
    #[test]
    fn a_commit_after_a_checkpoint_of_raw_root_writes_survives_a_crash() {
        for (kept, retention) in RETENTIONS {
            let which = format!("raw-root:{kept}");
            let (_dir, path, report) = crash_and_reopen(&which);
            for reopen in ["after the crash", "after a clean close"] {
                let db = open(&path, retention);
                for (key, name) in [("raw", "Alix"), ("engine", "Gus")] {
                    let node = db
                        .get_node(NodeId::new(entry(&report, key)[0]))
                        .unwrap()
                        .unwrap_or_else(|| panic!("{which}, {reopen}: the {key} node"));
                    assert_eq!(
                        node.get_property("name"),
                        Some(&Value::from(name)),
                        "{which}, {reopen}"
                    );
                }
                assert!(
                    db.current_epoch().as_u64() > entry(&report, "checkpointed")[0],
                    "{which}, {reopen}"
                );
                after_and_epoch(&db, &report, &format!("{which}, {reopen}"));
                assert_eq!(db.node_count().unwrap(), 3, "{which}, {reopen}");
                db.close().unwrap();
            }
        }
    }
}
