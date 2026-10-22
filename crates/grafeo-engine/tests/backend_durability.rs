//! A database on a backend of one's own (`GrafeoDB::with_store`) with a path
//! logs its commits to its own WAL, `<path>.wal`, and gets them back after a
//! crash: a backend that keeps nothing across a restart replays the whole
//! log at every open; a durable one replays only the commits after its
//! durable epoch, and a checkpoint drops the log it holds, starting the new
//! log with the catalog. A crash at any point of that checkpoint recovers
//! the same state. Each crash is a child process that exits without
//! `close()` (a panic hook ends it, so no `Drop` runs), and the parent
//! checks the files it left before it opens them again.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test backend_durability
//! ```

#![cfg(all(feature = "wal", feature = "gql", feature = "testing-crash-injection"))]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use grafeo_common::testing::child_process;
use grafeo_common::types::Value;
use grafeo_core::graph::GraphBackend;
use grafeo_core::graph::rowgroup::RowGroupStore;
use grafeo_engine::{Config, GrafeoDB};

const STEP_VAR: &str = "GRAFEO_BACKEND_DURABILITY_STEP";
const PATH_VAR: &str = "GRAFEO_BACKEND_DURABILITY_PATH";
const CRASH_VAR: &str = "GRAFEO_BACKEND_DURABILITY_CRASH_AT";
/// The exit code of a child whose checkpoint ran to the end: no crash point
/// was left to fire.
const COMPLETED: i32 = 3;

fn sidecar(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".wal");
    PathBuf::from(name)
}

/// The file a [`durable::Durable`] backend keeps beside the database path.
fn backend_file(path: &Path) -> PathBuf {
    path.with_extension("rows")
}

fn segments(path: &Path) -> usize {
    grafeo_storage::wal::list_wal_directory(&sidecar(path))
        .unwrap()
        .segments
        .len()
}

fn on_memory(path: &Path) -> GrafeoDB {
    GrafeoDB::with_store(
        Arc::new(RowGroupStore::new()) as Arc<dyn GraphBackend>,
        Config::persistent(path),
    )
    .unwrap()
}

fn on_durable(path: &Path) -> GrafeoDB {
    GrafeoDB::with_store(
        durable::Durable::open(&backend_file(path)) as Arc<dyn GraphBackend>,
        Config::persistent(path),
    )
    .unwrap()
}

/// The people's names, with their city, sorted.
fn people(db: &GrafeoDB) -> Vec<String> {
    let mut names: Vec<String> = db
        .execute("MATCH (p:Person) RETURN p.name + ' ' + coalesce(p.city, '-') AS row")
        .unwrap()
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(text) => text.to_string(),
            other => panic!("a row of text: {other:?}"),
        })
        .collect();
    names.sort();
    names
}

fn knows(db: &GrafeoDB) -> Vec<String> {
    db.execute("MATCH (a)-[:KNOWS]->(b) RETURN a.name + '-' + b.name AS pair")
        .unwrap()
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(text) => text.to_string(),
            other => panic!("a row of text: {other:?}"),
        })
        .collect()
}

/// The schema the children create: a node type whose name is required.
fn schema_holds(db: &GrafeoDB) -> bool {
    db.execute("INSERT (:Person {city: 'Paris'})").is_err()
}

/// Runs this binary's `test` as a child process for `step`, at `path`.
fn child(test: &str, step: &str, path: &Path, crash_at: Option<u64>) -> i32 {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", test, "--nocapture"])
        .env(STEP_VAR, step)
        .env(PATH_VAR, path);
    if let Some(point) = crash_at {
        command.env(CRASH_VAR, point.to_string());
    }
    child_process::run(&mut command)
        .unwrap()
        .code()
        .expect("the child exits with a code")
}

/// Ends the process at a panic (the injected crash) without unwinding into
/// the database's `Drop`, which would close it.
fn exit_at_a_panic() {
    std::panic::set_hook(Box::new(|info| {
        eprintln!("crashed: {info}");
        std::process::exit(0);
    }));
}

/// Child-process entry for the tests below; a no-op when run directly.
#[test]
fn backend_child() {
    let (Some(step), Some(path)) = (std::env::var_os(STEP_VAR), std::env::var_os(PATH_VAR)) else {
        return;
    };
    let path = PathBuf::from(path);
    exit_at_a_panic();
    match step.to_str().unwrap() {
        // Commits on a backend that keeps nothing, a rolled-back transaction
        // and a schema; no close.
        "memory" => {
            let db = on_memory(&path);
            db.execute("CREATE NODE TYPE Person (name STRING NOT NULL, city STRING)")
                .unwrap();
            db.execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
                .unwrap();
            let mut session = db.session();
            session.begin_transaction().unwrap();
            session
                .execute("INSERT (:Person {name: 'Vincent'})")
                .unwrap();
            session.rollback().unwrap();
            session.begin_transaction().unwrap();
            session
                .execute("MATCH (p:Person {name: 'Gus'}) SET p.city = 'Berlin'")
                .unwrap();
            session.commit().unwrap();
            std::process::exit(0);
        }
        // On a durable backend: Alix before a checkpoint, Gus after it.
        "durable" => {
            let db = on_durable(&path);
            db.execute("CREATE NODE TYPE Person (name STRING NOT NULL, city STRING)")
                .unwrap();
            db.execute("INSERT (:Person {name: 'Alix', city: 'Amsterdam'})")
                .unwrap();
            db.wal_checkpoint().unwrap();
            db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
            std::process::exit(0);
        }
        // The same, with a crash at the `crash_at`th point of the checkpoint.
        "crash" => {
            let point: u64 = std::env::var(CRASH_VAR).unwrap().parse().unwrap();
            let db = on_durable(&path);
            db.execute("CREATE NODE TYPE Person (name STRING NOT NULL, city STRING)")
                .unwrap();
            db.execute("INSERT (:Person {name: 'Alix', city: 'Amsterdam'})")
                .unwrap();
            let checkpointed = grafeo_common::testing::crash::with_crash_at(
                point,
                std::panic::AssertUnwindSafe(|| db.wal_checkpoint()),
            );
            // A crash ended the process in the hook: this is the checkpoint
            // that ran to the end.
            assert!(matches!(
                checkpointed,
                grafeo_common::testing::crash::CrashResult::Completed(Ok(()))
            ));
            std::process::exit(COMPLETED);
        }
        other => panic!("unknown step {other}"),
    }
}

/// A backend that keeps nothing across a restart gets every commit back
/// from the log at the next open, the rolled-back transaction left out and
/// the schema kept; the log is kept whole across a close, and the next open
/// has the later commits too.
#[test]
fn a_backend_that_keeps_nothing_gets_its_commits_back_from_the_log() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prague");
    assert_eq!(child("backend_child", "memory", &path, None), 0);
    assert!(sidecar(&path).is_dir(), "the crash left the log");
    assert!(
        !path.exists(),
        "a backend of its own writes no database file"
    );

    let db = on_memory(&path);
    assert_eq!(people(&db), ["Alix -", "Gus Berlin"]);
    assert_eq!(knows(&db), ["Alix-Gus"]);
    assert!(schema_holds(&db), "the node type came back from the log");
    db.execute("INSERT (:Person {name: 'Jules'})").unwrap();
    db.close().unwrap();
    drop(db);

    assert!(sidecar(&path).is_dir(), "a log that is all the data stays");
    let db = on_memory(&path);
    assert_eq!(people(&db), ["Alix -", "Gus Berlin", "Jules -"]);
    assert!(schema_holds(&db));
}

/// A durable backend holds what its checkpoint flushed, and the next open
/// replays only the commits after that: Alix is not created twice. The
/// checkpoint dropped the older log, which now starts with the catalog.
#[test]
fn a_durable_backend_replays_only_the_commits_after_its_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vienna");
    assert_eq!(child("backend_child", "durable", &path, None), 0);
    assert!(
        backend_file(&path).is_file(),
        "the checkpoint flushed the backend"
    );
    assert_eq!(
        segments(&path),
        1,
        "the checkpoint dropped the segments before the catalog group"
    );

    let backend = durable::Durable::open(&backend_file(&path));
    assert_eq!(
        backend.node_count(),
        1,
        "the backend holds Alix, flushed at the checkpoint"
    );
    let db = GrafeoDB::with_store(backend, Config::persistent(&path)).unwrap();
    assert_eq!(people(&db), ["Alix Amsterdam", "Gus -"]);
    assert!(schema_holds(&db), "the catalog group kept the node type");
}

/// A crash at any point of a checkpoint on a durable backend (before or
/// after the flush, between writing the catalog group and dropping the old
/// segments, after dropping them) recovers exactly the committed state, with
/// the schema; a crash after the catalog group and before the old segments
/// go leaves both, and replaying both changes nothing.
#[test]
fn a_crash_anywhere_in_a_checkpoint_recovers_the_same_state() {
    let mut crashed = 0;
    for point in 1.. {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lisbon");
        let code = child("backend_child", "crash", &path, Some(point));
        if code == COMPLETED {
            break;
        }
        assert_eq!(code, 0, "the child crashed at point {point}");
        crashed += 1;
        assert!(sidecar(&path).is_dir(), "point {point}: the log is there");
        let db = on_durable(&path);
        assert_eq!(people(&db), ["Alix Amsterdam"], "point {point}");
        assert!(schema_holds(&db), "point {point}: the schema");
        db.close().unwrap();
        drop(db);
        let db = on_durable(&path);
        assert_eq!(people(&db), ["Alix Amsterdam"], "point {point}: reopened");
        assert!(point < 64, "the checkpoint has fewer crash points");
    }
    assert!(crashed >= 2, "the sweep crashed inside the checkpoint");
}

/// The catalog group a checkpoint writes holds the catalog as it is, not
/// its history: a schema created and dropped many times leaves a log no
/// larger than one with the schema written once.
#[test]
fn the_catalog_group_is_bounded_by_the_catalog() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("oslo");
    let db = on_durable(&path);
    db.execute("CREATE NODE TYPE Person (name STRING NOT NULL)")
        .unwrap();
    for round in 0..200 {
        db.execute(&format!("CREATE NODE TYPE Trip{round} (km INT64)"))
            .unwrap();
        db.execute(&format!("DROP NODE TYPE Trip{round}")).unwrap();
    }
    let before = log_bytes(&path);
    db.wal_checkpoint().unwrap();
    let after = log_bytes(&path);
    assert!(
        after * 20 < before,
        "the new log holds the live catalog: {after} bytes, the old one {before}"
    );
    db.close().unwrap();
    drop(db);
    let db = on_durable(&path);
    assert!(schema_holds(&db), "Person is in the catalog group");
    db.execute("CREATE NODE TYPE Trip0 (km INT64)")
        .expect("a dropped type stays dropped");
}

fn log_bytes(path: &Path) -> u64 {
    grafeo_storage::wal::list_wal_directory(&sidecar(path))
        .unwrap()
        .segments
        .iter()
        .map(|(_, segment)| std::fs::metadata(segment).unwrap().len())
        .sum()
}

/// A second process cannot open the log another one holds.
#[test]
fn a_second_open_of_the_same_log_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("madrid");
    let first = on_memory(&path);
    let second = GrafeoDB::with_store(
        Arc::new(RowGroupStore::new()) as Arc<dyn GraphBackend>,
        Config::persistent(&path),
    );
    let error = second.err().expect("the log is held").to_string();
    assert!(error.contains("locked"), "{error}");
    first.close().unwrap();
    drop(first);
    let again = on_memory(&path);
    again.close().unwrap();
}

/// A backend that keeps its rows in a file of its own beside the database
/// path: a flush writes the committed state at the epoch (to a temporary
/// file, then renamed), and an open loads it, so it holds every commit up
/// to its last flush durably.
mod durable {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use arcstr::ArcStr;
    use grafeo_common::change::{Change, ChangeMark, DataOp};
    use grafeo_common::storage::{LogRecord, read_log_records};
    use grafeo_common::types::{EdgeId, EpochId, NodeId, ReadView, TransactionId};
    use grafeo_common::utils::error::Result;
    use grafeo_core::graph::apply::{Applied, ApplyError, BackendTransaction, Writer};
    use grafeo_core::graph::lpg::{Edge, Node};
    use grafeo_core::graph::rowgroup::RowGroupStore;
    use grafeo_core::graph::{Direction, GraphBackend, GraphStore};
    use grafeo_core::statistics::Statistics;

    pub struct Durable {
        rows: RowGroupStore,
        file: PathBuf,
        durable: AtomicU64,
    }

    impl Durable {
        /// The backend whose rows are in `file`: what its last flush wrote.
        pub fn open(file: &Path) -> Arc<Self> {
            let rows = RowGroupStore::new();
            let mut durable = 0;
            if let Ok(bytes) = std::fs::read(file) {
                let (head, records) = bytes.split_at(8);
                durable = u64::from_le_bytes(head.try_into().unwrap());
                let epoch = EpochId::new(durable);
                read_log_records(records, &mut |record| {
                    let LogRecord::Data { op, .. } = record else {
                        panic!("the file holds data records");
                    };
                    rows.apply(&op, Writer::Replay { epoch }).unwrap();
                    Ok(())
                })
                .unwrap();
            }
            Arc::new(Self {
                rows,
                file: file.to_path_buf(),
                durable: AtomicU64::new(durable),
            })
        }

        /// The nodes it holds, at its last commit.
        pub fn node_count(&self) -> usize {
            self.rows.node_count(ReadView::latest()).unwrap()
        }
    }

    impl GraphStore for Durable {
        fn node(&self, id: NodeId, view: ReadView) -> Result<Option<Node>> {
            self.rows.node(id, view)
        }
        fn edge(&self, id: EdgeId, view: ReadView) -> Result<Option<Edge>> {
            self.rows.edge(id, view)
        }
        fn edges(
            &self,
            node: NodeId,
            direction: Direction,
            types: &[ArcStr],
            view: ReadView,
        ) -> Result<Vec<(NodeId, EdgeId)>> {
            self.rows.edges(node, direction, types, view)
        }
        fn has_backward_adjacency(&self) -> bool {
            self.rows.has_backward_adjacency()
        }
        fn node_ids(&self, view: ReadView) -> Result<Vec<NodeId>> {
            self.rows.node_ids(view)
        }
        fn all_labels(&self) -> Result<Vec<String>> {
            self.rows.all_labels()
        }
        fn all_edge_types(&self) -> Result<Vec<String>> {
            self.rows.all_edge_types()
        }
        fn all_property_keys(&self) -> Result<Vec<String>> {
            self.rows.all_property_keys()
        }
        fn statistics(&self) -> Arc<Statistics> {
            self.rows.statistics()
        }
        fn committed_epoch(&self) -> EpochId {
            self.rows.committed_epoch()
        }
    }

    impl GraphBackend for Durable {
        fn reserve_node_ids(
            &self,
            count: u64,
        ) -> std::result::Result<std::ops::Range<u64>, ApplyError> {
            self.rows.reserve_node_ids(count)
        }
        fn reserve_edge_ids(
            &self,
            count: u64,
        ) -> std::result::Result<std::ops::Range<u64>, ApplyError> {
            self.rows.reserve_edge_ids(count)
        }
        fn begin(
            &self,
            transaction: TransactionId,
            snapshot: EpochId,
        ) -> std::result::Result<BackendTransaction, ApplyError> {
            self.rows.begin(transaction, snapshot)
        }
        fn apply(&self, op: &DataOp, writer: Writer) -> std::result::Result<Applied, ApplyError> {
            self.rows.apply(op, writer)
        }
        fn commit(
            &self,
            transaction: BackendTransaction,
            changes: &mut dyn Iterator<Item = &Change>,
            epoch: EpochId,
        ) -> std::result::Result<(), ApplyError> {
            self.rows.commit(transaction, changes, epoch)
        }
        fn rollback(
            &self,
            transaction: BackendTransaction,
            changes: &mut dyn DoubleEndedIterator<Item = &Change>,
        ) -> std::result::Result<(), ApplyError> {
            self.rows.rollback(transaction, changes)
        }
        fn rollback_to(
            &self,
            transaction: BackendTransaction,
            savepoint: ChangeMark,
            changes_since: &mut dyn DoubleEndedIterator<Item = &Change>,
        ) -> std::result::Result<(), ApplyError> {
            self.rows.rollback_to(transaction, savepoint, changes_since)
        }
        fn collect(&self, horizon: EpochId) {
            GraphBackend::collect(&self.rows, horizon);
        }
        fn durable_epoch(&self) -> EpochId {
            EpochId::new(self.durable.load(Ordering::Acquire))
        }
        /// Writes every node and edge a reader at `epoch` sees, as create
        /// records, to a new file, then puts it in place of the old one.
        fn flush(&self, epoch: EpochId) -> std::result::Result<(), ApplyError> {
            let refused = |error: &dyn std::fmt::Display| ApplyError::Refused(error.to_string());
            let view = ReadView::committed(epoch);
            let mut bytes = epoch.as_u64().to_le_bytes().to_vec();
            let mut push = |op: DataOp| {
                LogRecord::Data { graph: None, op }
                    .encode_framed(&mut bytes)
                    .map_err(|error| refused(&error))
            };
            let nodes = self.rows.node_ids(view).map_err(|error| refused(&error))?;
            let mut edges = Vec::new();
            for &id in &nodes {
                let node = self.rows.node(id, view).map_err(|error| refused(&error))?;
                let Some(node) = node else { continue };
                let mut properties: Vec<_> = node
                    .properties
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect();
                properties.sort_by(|a, b| a.0.cmp(&b.0));
                push(DataOp::CreateNode {
                    id,
                    labels: node.labels.iter().cloned().collect(),
                    properties,
                })?;
                for (_, edge) in self
                    .rows
                    .edges(id, Direction::Outgoing, &[], view)
                    .map_err(|error| refused(&error))?
                {
                    edges.push(edge);
                }
            }
            for id in edges {
                let edge = self.rows.edge(id, view).map_err(|error| refused(&error))?;
                let Some(edge) = edge else { continue };
                let mut properties: Vec<_> = edge
                    .properties
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect();
                properties.sort_by(|a, b| a.0.cmp(&b.0));
                push(DataOp::CreateEdge {
                    id,
                    src: edge.src,
                    dst: edge.dst,
                    edge_type: edge.edge_type.clone(),
                    properties,
                })?;
            }
            let temporary = self.file.with_extension("rows.new");
            std::fs::write(&temporary, &bytes).map_err(|error| refused(&error))?;
            std::fs::OpenOptions::new()
                .write(true)
                .open(&temporary)
                .and_then(|file| file.sync_all())
                .map_err(|error| refused(&error))?;
            std::fs::rename(&temporary, &self.file).map_err(|error| refused(&error))?;
            self.durable.store(epoch.as_u64(), Ordering::Release);
            Ok(())
        }
    }
}
