//! An example of a store of one's own, written against the facade's store
//! contract (`grafeo::store`) only: `ExampleBackend` keeps its rows in a
//! `RowGroupStore` and, when durable, in a file of its own that each flush
//! rewrites. Both variants pass the published conformance suite, and a
//! database on each rolls back, reopens and gets every commit back after a
//! crash: the durable one from its file plus the commits its log holds after
//! the file's epoch, the in-memory one from the whole log.
//!
//! ```bash
//! cargo test -p grafeo --all-features --test example_backend
//! ```

use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use grafeo::store::{
    Applied, ApplyError, ArcStr, BackendTransaction, Change, ChangeMark, DataOp, Direction, Edge,
    EdgeId, EpochId, GraphBackend, GraphStore, Node, NodeId, ReadView, Result, RowGroupStore,
    Statistics, TransactionId, Writer,
};
// The file's format is the example's own choice; it reuses the encoding of
// the write-ahead log's records.
use grafeo_common::storage::{LogRecord, read_log_records};

/// A store of one's own: the rows in a [`RowGroupStore`], which answers the
/// reads and runs the transactions, and with a file, what a reader at the
/// last flushed epoch sees, written there by [`flush`](GraphBackend::flush).
pub struct ExampleBackend {
    rows: RowGroupStore,
    /// The file the flushes write, `None` for a backend that keeps nothing
    /// across a restart.
    file: Option<PathBuf>,
    /// The epoch of the last flush: every commit up to it is in the file.
    durable: AtomicU64,
}

impl ExampleBackend {
    /// A backend that keeps nothing across a restart: its durable epoch is
    /// always [`EpochId::INITIAL`], so a database on it replays its whole
    /// log at every open.
    pub fn in_memory() -> Self {
        Self {
            rows: RowGroupStore::new(),
            file: None,
            durable: AtomicU64::new(EpochId::INITIAL.as_u64()),
        }
    }

    /// The durable backend of `file`, with what its last flush wrote (none
    /// for a file that does not exist yet).
    ///
    /// # Panics
    ///
    /// When the file cannot be read or holds something else.
    pub fn open(file: &Path) -> Self {
        let rows = RowGroupStore::new();
        let mut durable = EpochId::INITIAL.as_u64();
        if let Ok(bytes) = std::fs::read(file) {
            let (head, records) = bytes.split_at(8);
            durable = u64::from_le_bytes(head.try_into().expect("the epoch"));
            let epoch = EpochId::new(durable);
            read_log_records(records, &mut |record| {
                let LogRecord::Data { op, .. } = record else {
                    panic!("the file holds rows only");
                };
                rows.apply(&op, Writer::Replay { epoch })
                    .expect("a row of the file");
                Ok(())
            })
            .expect("the file's rows");
        }
        Self {
            rows,
            file: Some(file.to_path_buf()),
            durable: AtomicU64::new(durable),
        }
    }

    /// The rows a reader at `epoch` sees, as create records after the epoch.
    fn image(&self, epoch: EpochId) -> std::result::Result<Vec<u8>, ApplyError> {
        let refused = |error: &dyn std::fmt::Display| ApplyError::Refused(error.to_string());
        let view = ReadView::committed(epoch);
        let mut bytes = epoch.as_u64().to_le_bytes().to_vec();
        let mut push = |op: DataOp| {
            LogRecord::Data { graph: None, op }
                .encode_framed(&mut bytes)
                .map_err(|error| refused(&error))
        };
        let mut edges = Vec::new();
        for id in self.rows.node_ids(view).map_err(|error| refused(&error))? {
            let Some(node) = self.rows.node(id, view).map_err(|error| refused(&error))? else {
                continue;
            };
            push(DataOp::CreateNode {
                id,
                labels: node.labels.iter().cloned().collect(),
                properties: node.properties.into_iter().collect(),
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
            let Some(edge) = self.rows.edge(id, view).map_err(|error| refused(&error))? else {
                continue;
            };
            push(DataOp::CreateEdge {
                id,
                src: edge.src,
                dst: edge.dst,
                edge_type: edge.edge_type.clone(),
                properties: edge.properties.into_iter().collect(),
            })?;
        }
        Ok(bytes)
    }
}

/// The reads: the rows' own.
impl GraphStore for ExampleBackend {
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

/// The transactions run in the rows; the durability is the file's.
impl GraphBackend for ExampleBackend {
    fn reserve_node_ids(&self, count: u64) -> std::result::Result<Range<u64>, ApplyError> {
        self.rows.reserve_node_ids(count)
    }
    fn reserve_edge_ids(&self, count: u64) -> std::result::Result<Range<u64>, ApplyError> {
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
    /// Writes the rows a reader at `epoch` sees to a new file, synced, and
    /// puts it in place of the old one; only then is `epoch` durable. A
    /// backend without a file has nothing to write and stays at
    /// [`EpochId::INITIAL`].
    fn flush(&self, epoch: EpochId) -> std::result::Result<(), ApplyError> {
        let Some(file) = &self.file else {
            return Ok(());
        };
        let refused = |error: std::io::Error| ApplyError::Refused(error.to_string());
        let bytes = self.image(epoch)?;
        let temporary = file.with_extension("rows.new");
        std::fs::write(&temporary, &bytes).map_err(refused)?;
        std::fs::OpenOptions::new()
            .write(true)
            .open(&temporary)
            .and_then(|written| written.sync_all())
            .map_err(refused)?;
        std::fs::rename(&temporary, file).map_err(refused)?;
        self.durable.store(epoch.as_u64(), Ordering::Release);
        Ok(())
    }
}

/// Both variants against the published conformance suite.
#[cfg(feature = "conformance")]
mod conformance {
    use grafeo::store::{KnownGap, conformance_suite};

    use super::ExampleBackend;

    /// The cases the example fails: none, as its rows answer every one.
    const GAPS: &[KnownGap] = &[];

    mod in_memory {
        use super::{ExampleBackend, GAPS, conformance_suite};

        conformance_suite!(ExampleBackend, ExampleBackend::in_memory, GAPS);
    }

    mod durable {
        use std::sync::atomic::{AtomicU64, Ordering};

        use super::{ExampleBackend, GAPS, conformance_suite};

        /// A durable backend of a file of its own, which the cases never
        /// flush, so it is never written.
        fn store() -> ExampleBackend {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let file = std::env::temp_dir().join(format!(
                "grafeo-example-backend-{}-{}.rows",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            ExampleBackend::open(&file)
        }

        conformance_suite!(ExampleBackend, store, GAPS);
    }
}

/// A database on the example backend: rollbacks, reopens and crashes.
#[cfg(all(feature = "wal", feature = "gql"))]
mod database {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use grafeo::store::{GraphBackend, GraphStore, ReadView};
    use grafeo::{Config, GrafeoDB, Value};
    use grafeo_common::testing::child_process;

    use super::ExampleBackend;

    /// Tells [`crash_child`] which step to run and where.
    const STEP_VAR: &str = "GRAFEO_EXAMPLE_BACKEND_STEP";
    const PATH_VAR: &str = "GRAFEO_EXAMPLE_BACKEND_PATH";

    /// A new directory for one test's files.
    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "grafeo-example-backend-{name}-{}",
            std::process::id()
        ));
        // A run that failed leaves its files behind.
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The file the durable backend of the database at `path` keeps.
    fn rows_file(path: &Path) -> PathBuf {
        path.with_extension("rows")
    }

    /// The database's write-ahead log, which a crash leaves behind.
    fn log_dir(path: &Path) -> PathBuf {
        let mut name = path.as_os_str().to_owned();
        name.push(".wal");
        PathBuf::from(name)
    }

    fn on_durable(path: &Path) -> GrafeoDB {
        let backend = Arc::new(ExampleBackend::open(&rows_file(path)));
        GrafeoDB::with_store(backend as Arc<dyn GraphBackend>, Config::persistent(path)).unwrap()
    }

    fn on_memory(path: &Path) -> GrafeoDB {
        let backend = Arc::new(ExampleBackend::in_memory());
        GrafeoDB::with_store(backend as Arc<dyn GraphBackend>, Config::persistent(path)).unwrap()
    }

    /// The people's names with their city, sorted.
    fn people(db: &GrafeoDB) -> Vec<(Value, Value)> {
        db.execute("MATCH (p:Person) RETURN p.name, p.city ORDER BY p.name")
            .unwrap()
            .rows()
            .iter()
            .map(|row| (row[0].clone(), row[1].clone()))
            .collect()
    }

    fn person(name: &str, city: Option<&str>) -> (Value, Value) {
        (Value::from(name), city.map_or(Value::Null, Value::from))
    }

    /// A rolled-back transaction leaves the backend as the last commit left
    /// it: its node, its value and its edge are gone, for every reader.
    #[test]
    fn a_rollback_leaves_the_backend_as_the_last_commit_left_it() {
        let backend = Arc::new(ExampleBackend::in_memory());
        let db = GrafeoDB::with_store(
            Arc::clone(&backend) as Arc<dyn GraphBackend>,
            Config::in_memory(),
        )
        .unwrap();
        db.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute("MATCH (a:Person {name: 'Alix'}) SET a.city = 'Paris' INSERT (a)-[:KNOWS]->(:Person {name: 'Gus'})")
            .unwrap();
        assert_eq!(
            session
                .execute("MATCH (p:Person) RETURN count(p)")
                .unwrap()
                .rows(),
            [[Value::Int64(2)]],
            "the transaction sees its own writes"
        );
        session.rollback().unwrap();

        assert_eq!(people(&db), [person("Alix", None)]);
        let view = ReadView::latest();
        assert_eq!(backend.node_ids(view).unwrap().len(), 1);
        assert_eq!(
            db.execute("MATCH ()-[k:KNOWS]->() RETURN count(k)")
                .unwrap()
                .rows(),
            [[Value::Int64(0)]]
        );
    }

    /// A close flushes the durable backend: its file holds every commit and
    /// the reopen reads it, with the ids the commits took never given again.
    #[test]
    fn a_reopen_reads_what_the_close_flushed() {
        let dir = test_dir("reopen");
        let path = dir.join("trips");
        let db = on_durable(&path);
        db.execute(
            "INSERT (:Person {name: 'Alix', city: 'Amsterdam'})-[:KNOWS]->(:Person {name: 'Gus'})",
        )
        .unwrap();
        db.close().unwrap();
        drop(db);
        let flushed = ExampleBackend::open(&rows_file(&path));
        assert_eq!(flushed.node_ids(ReadView::latest()).unwrap().len(), 2);

        let db = on_durable(&path);
        assert_eq!(
            people(&db),
            [person("Alix", Some("Amsterdam")), person("Gus", None)]
        );
        db.execute("INSERT (:Person {name: 'Mia'})").unwrap();
        assert_eq!(
            db.execute("MATCH (p:Person) RETURN count(DISTINCT id(p))")
                .unwrap()
                .rows(),
            [[Value::Int64(3)]],
            "the new node has an id of its own"
        );
        db.close().unwrap();
        drop(db);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Child-process entry for the crash tests; a no-op when run directly.
    /// Commits, and exits while the database is open: no `close()`, so no
    /// flush, and no destructors.
    #[test]
    fn crash_child() {
        let (Some(step), Some(path)) = (std::env::var_os(STEP_VAR), std::env::var_os(PATH_VAR))
        else {
            return;
        };
        let path = PathBuf::from(path);
        match step.to_str().unwrap() {
            // Alix before a checkpoint, which flushes her to the file; Gus
            // after it, in the log only.
            "durable" => {
                let db = on_durable(&path);
                db.execute("INSERT (:Person {name: 'Alix', city: 'Amsterdam'})")
                    .unwrap();
                db.wal_checkpoint().unwrap();
                db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
                // A crash: the database is never dropped, which would close it.
                std::process::exit(0);
            }
            // Two commits and a rolled-back transaction, in the log only.
            "memory" => {
                let db = on_memory(&path);
                db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
                let mut session = db.session();
                session.begin_transaction().unwrap();
                session
                    .execute("INSERT (:Person {name: 'Vincent'})")
                    .unwrap();
                session.rollback().unwrap();
                db.execute("MATCH (a:Person {name: 'Alix'}) SET a.city = 'Prague'")
                    .unwrap();
                std::process::exit(0);
            }
            other => panic!("unknown step {other}"),
        }
    }

    fn crash(step: &str, path: &Path) {
        let status = child_process::run(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "database::crash_child", "--nocapture"])
                .env(STEP_VAR, step)
                .env(PATH_VAR, path),
        )
        .unwrap();
        assert!(status.success(), "the child process failed");
    }

    /// After a crash the durable backend's file holds what the checkpoint
    /// flushed, and the open replays the commit after it from the log.
    #[test]
    fn a_durable_backend_gets_the_commits_after_its_flush_back_from_the_log() {
        let dir = test_dir("durable-crash");
        let path = dir.join("trips");
        crash("durable", &path);

        assert!(log_dir(&path).exists(), "the crash leaves the log");
        let flushed = ExampleBackend::open(&rows_file(&path));
        assert_eq!(
            flushed.node_ids(ReadView::latest()).unwrap().len(),
            1,
            "the file holds what the checkpoint flushed: Alix"
        );
        drop(flushed);

        let db = on_durable(&path);
        assert_eq!(
            people(&db),
            [person("Alix", Some("Amsterdam")), person("Gus", None)]
        );
        db.close().unwrap();
        drop(db);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// After a crash a backend that keeps nothing gets every commit back
    /// from the whole log, without the rolled-back transaction.
    #[test]
    fn a_backend_that_keeps_nothing_gets_every_commit_back_from_the_log() {
        let dir = test_dir("memory-crash");
        let path = dir.join("trips");
        crash("memory", &path);

        assert!(log_dir(&path).exists(), "the crash leaves the log");
        let db = on_memory(&path);
        assert_eq!(people(&db), [person("Alix", Some("Prague"))]);
        db.close().unwrap();
        drop(db);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
