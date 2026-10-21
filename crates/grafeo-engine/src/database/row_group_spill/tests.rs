//! Spilled columns of the row-group store in their files: what the files
//! hold and when they go, the consumers the buffer manager asks, and a
//! database that checkpoints, crashes and reopens with spill files present.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use grafeo_common::types::{EpochId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::error::Error;
use grafeo_core::execution::operators::{GraphWriter, WriteTarget};
use grafeo_core::graph::apply::ChangeTarget;
use grafeo_core::graph::rowgroup::RowGroupStore;
use grafeo_core::graph::{GraphStore, GraphStoreMut};

use super::ColumnSpillFiles;
use crate::catalog::Catalog;
use crate::database::row_group_sections::{continue_row_group_epochs, load_row_group_sections};
use crate::database::sections::CheckpointSources;
use crate::transaction::{TransactionChanges, TransactionManager};

/// A database on a row-group root, as `GrafeoDB` holds its parts.
struct Database {
    root: Arc<RowGroupStore>,
    catalog: Arc<Catalog>,
    transaction_manager: Arc<TransactionManager>,
    #[cfg(feature = "triple-store")]
    rdf_store: Arc<grafeo_core::graph::rdf::RdfStore>,
}

/// An open transaction: its id, its changes and its writer.
struct Open {
    transaction: TransactionId,
    changes: Arc<TransactionChanges>,
    writer: GraphWriter,
}

impl Database {
    fn new() -> Self {
        Self {
            root: Arc::new(RowGroupStore::new()),
            catalog: Arc::new(Catalog::new()),
            transaction_manager: Arc::new(TransactionManager::new()),
            #[cfg(feature = "triple-store")]
            rdf_store: Arc::new(grafeo_core::graph::rdf::RdfStore::new()),
        }
    }

    fn store(&self, graph: Option<&str>) -> Arc<RowGroupStore> {
        match graph {
            None => Arc::clone(&self.root),
            Some(name) => self.root.graph(name).expect("a graph the test created"),
        }
    }

    fn begin(&self, graph: Option<&str>) -> Open {
        let (transaction, changes) = self.transaction_manager.begin_private().unwrap();
        let store = self.store(graph);
        let recording = changes
            .recording(
                &self.transaction_manager,
                graph,
                WriteTarget::Store(Arc::clone(&store) as Arc<dyn ChangeTarget>),
            )
            .unwrap();
        let writer = GraphWriter::new(store as Arc<dyn GraphStoreMut>).with_recording(recording);
        Open {
            transaction,
            changes,
            writer,
        }
    }

    fn commit(&self, open: Open) -> EpochId {
        let Open {
            transaction,
            changes,
            writer,
        } = open;
        drop(writer);
        let commit = self.transaction_manager.start_commit(transaction).unwrap();
        let epoch = commit.epoch();
        commit
            .written()
            .complete_with(|| {
                changes
                    .stamp(epoch)
                    .map_err(|error| Error::Internal(error.to_string()))?;
                self.root.sync_epoch(epoch);
                Ok(())
            })
            .unwrap();
        epoch
    }

    fn write(&self, graph: Option<&str>, write: impl FnOnce(&GraphWriter)) -> EpochId {
        let open = self.begin(graph);
        write(&open.writer);
        self.commit(open)
    }

    #[cfg(feature = "grafeo-file")]
    fn sources(&self) -> CheckpointSources {
        CheckpointSources {
            store: Some(Arc::clone(&self.root)),
            catalog: Arc::clone(&self.catalog),
            transaction_manager: Arc::clone(&self.transaction_manager),
            #[cfg(feature = "triple-store")]
            rdf_store: Arc::clone(&self.rdf_store),
            wal: None,
            wal_records_at_checkpoint: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }
}

fn embedding() -> PropertyKey {
    PropertyKey::new("embedding")
}

/// The embedding of the item with rank `rank`.
fn embedding_of(rank: u16) -> Value {
    Value::Vector(vec![f32::from(rank), 1.0].into())
}

/// Writes `count` items to the graph `graph`: each a name, a rank and an
/// embedding from its rank (two vectors of rows of a row group, so two
/// chunks), in one transaction.
fn items(database: &Database, graph: Option<&str>, count: u16) -> Vec<NodeId> {
    let mut ids = Vec::new();
    database.write(graph, |writer| {
        let rows = (0..count)
            .map(|rank| {
                vec![
                    (
                        "name".to_string(),
                        Value::from(
                            ["Alix", "Gus", "Vincent", "Jules", "Mia"][usize::from(rank % 5)],
                        ),
                    ),
                    ("rank".to_string(), Value::Int64(i64::from(rank))),
                    ("embedding".to_string(), embedding_of(rank)),
                ]
            })
            .collect();
        ids = writer.create_nodes(&["Item".to_string()], rows).unwrap();
    });
    ids
}

/// Every item reads its embedding.
fn assert_embeddings(store: &RowGroupStore, ids: &[NodeId], what: &str) {
    for (rank, id) in (0u16..).zip(ids) {
        assert_eq!(
            store.get_node_property(*id, &embedding()),
            Some(embedding_of(rank)),
            "{what}: item {rank}"
        );
    }
}

/// The files in `directory`, none when it is missing.
fn files_in(directory: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries.flatten().map(|entry| entry.path()).collect();
    files.sort();
    files
}

/// A spilled column's file holds its chunks while a chunk reads it: a
/// write thaws the column and removes the file, so does a reload, and so
/// does dropping the store.
#[test]
fn a_spill_file_lives_as_long_as_a_chunk_reads_it() {
    let dir = tempfile::tempdir().unwrap();
    let files = ColumnSpillFiles::in_dir(dir.path().join("cache"));
    let database = Database::new();
    let ids = items(&database, None, 3_000);
    let before = database.root.memory();

    let report = database
        .root
        .spill_node_property_column(&embedding(), &files)
        .unwrap();
    assert_eq!(report.spilled_groups, 1, "{report:?}");
    let spilled = files_in(files.directory());
    assert_eq!(spilled.len(), 1, "{spilled:?}");
    assert_eq!(
        usize::try_from(std::fs::metadata(&spilled[0]).unwrap().len()).unwrap(),
        report.file_bytes
    );
    let after = database.root.memory();
    assert_eq!(after.node_spilled_bytes, report.file_bytes);
    assert!(
        before.total_bytes() - after.total_bytes() >= 3_000 * 8,
        "the vectors left the heap: {before:?} {after:?}"
    );
    assert_embeddings(&database.root, &ids, "spilled");
    assert_eq!(database.root.memory(), after, "the reads decoded nothing");

    // A write thaws the column and lets go of its file; the collect drops
    // the value it kept for older readers, so the column spills again.
    let written = database.write(None, |writer| {
        writer
            .set_node_properties(
                ids[19],
                &[("embedding".to_string(), embedding_of(19))],
                false,
            )
            .unwrap();
    });
    assert!(files_in(files.directory()).is_empty(), "thawed");
    assert_eq!(
        database.root.spilled_node_property_columns(),
        Vec::<PropertyKey>::new()
    );
    database.root.collect(written);

    // A reload lets go of it too.
    database
        .root
        .spill_node_property_column(&embedding(), &files)
        .unwrap();
    assert_eq!(files_in(files.directory()).len(), 1);
    assert!(
        database
            .root
            .reload_node_property_column(&embedding())
            .unwrap()
    );
    assert!(files_in(files.directory()).is_empty(), "reloaded");
    assert_embeddings(&database.root, &ids, "reloaded");

    // And dropping the store.
    database
        .root
        .spill_node_property_column(&embedding(), &files)
        .unwrap();
    assert_eq!(files_in(files.directory()).len(), 1);
    drop(database);
    assert!(files_in(files.directory()).is_empty(), "dropped");
}

/// A spill file that cannot be written leaves nothing behind and the
/// column in memory.
#[test]
fn a_failed_spill_file_leaves_nothing_behind() {
    let dir = tempfile::tempdir().unwrap();
    // A file where the directory should be.
    let blocked = dir.path().join("cache");
    std::fs::write(&blocked, b"Butch's").unwrap();
    let files = ColumnSpillFiles::in_dir(&blocked);
    let database = Database::new();
    let ids = items(&database, None, 100);
    let before = database.root.memory();
    assert!(matches!(
        database
            .root
            .spill_node_property_column(&embedding(), &files),
        Err(Error::Io(_))
    ));
    assert_eq!(database.root.memory(), before);
    assert_eq!(files_in(dir.path()), [blocked]);
    assert_embeddings(&database.root, &ids, "kept in memory");
}

/// The consumers the registration adds: the store's section, which reports
/// the store's heap and does not spill, and its indexes'; without a spill
/// directory nothing spills.
#[test]
fn the_registered_consumers_report_the_store_and_its_indexes() {
    use grafeo_common::memory::StorageTier;
    use grafeo_common::memory::buffer::BufferManager;

    let database = Database::new();
    items(&database, None, 100);
    #[cfg(feature = "text-index")]
    database.root.add_text_index(
        "Item",
        "name",
        Arc::new(parking_lot::RwLock::new(
            super::super::index::text_index_from_data(
                &*database.root,
                "Item",
                "name",
                &grafeo_core::index::text::TextIndexOptions::new(),
            ),
        )),
    );
    let manager = BufferManager::with_budget(1 << 30);
    super::register_row_group_consumers(&manager, &database.root, None, None);
    let tiers = manager.snapshot_consumer_tiers();
    let tier = |name: &str| {
        tiers
            .iter()
            .find(|(consumer, _)| consumer == name)
            .map(|(_, tier)| *tier)
    };
    assert_eq!(
        tier("section:LpgStore"),
        Some(StorageTier::InMemory),
        "{tiers:?}"
    );
    #[cfg(feature = "vector-index")]
    assert_eq!(
        tier("section:VectorStore"),
        Some(StorageTier::Uninitialized),
        "no vector index"
    );
    #[cfg(feature = "text-index")]
    assert_eq!(tier("section:TextIndex"), Some(StorageTier::InMemory));
    assert_eq!(manager.spill_all(), 0);
    assert_eq!(
        database.root.spilled_node_property_columns(),
        Vec::<PropertyKey>::new()
    );
}

#[cfg(feature = "vector-index")]
mod consumers {
    use std::path::PathBuf;
    use std::sync::Arc;

    use grafeo_common::memory::StorageTier;
    use grafeo_common::memory::buffer::{BufferManager, MemoryConsumer, MemoryRegion};
    use grafeo_common::types::PropertyKey;
    use grafeo_core::graph::GraphStoreSearch;
    use grafeo_core::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind};

    use super::super::{ColumnSpillFiles, RowGroupVectorConsumer, register_row_group_consumers};
    use super::{Database, assert_embeddings, embedding, files_in, items};

    /// A root and its named graph trips, each with items and a vector index
    /// on their embeddings.
    fn indexed() -> (Database, Vec<grafeo_common::types::NodeId>) {
        let database = Database::new();
        let ids = items(&database, None, 3_000);
        assert!(database.root.create_graph("trips").unwrap());
        items(&database, Some("trips"), 300);
        for graph in [None, Some("trips")] {
            database.store(graph).add_vector_index(
                "Item",
                "embedding",
                Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
                    2,
                    DistanceMetric::Euclidean,
                )))),
            );
        }
        (database, ids)
    }

    /// The nearest items to Vincent's query in a graph, through its index.
    fn nearest(database: &Database, graph: Option<&str>) -> Vec<(u64, f64)> {
        database
            .store(graph)
            .vector_search(
                Some("Item"),
                "embedding",
                &[150.25, 1.0],
                3,
                DistanceMetric::Euclidean,
            )
            .into_iter()
            .map(|(id, distance)| (id.as_u64(), distance))
            .collect()
    }

    /// The vector consumer spills the embedding column of every graph's
    /// vector index, search reads them from the files without bringing them
    /// back, and its reload brings them back and removes the files.
    #[test]
    fn the_vector_consumer_spills_every_graphs_embeddings_and_reloads_them() {
        let dir = tempfile::tempdir().unwrap();
        let (database, ids) = indexed();
        let consumer = RowGroupVectorConsumer::new(
            &database.root,
            Some(ColumnSpillFiles::in_dir(dir.path().join("cache"))),
        );
        let searched = [nearest(&database, None), nearest(&database, Some("trips"))];
        assert_eq!(consumer.current_tier(), StorageTier::InMemory);
        assert!(consumer.memory_usage() > 0, "the indexes' heap");
        let heap = database.root.memory().total_bytes();

        let freed = consumer.spill(usize::MAX).unwrap();
        assert!(freed >= 3_300 * 8, "{freed} bytes freed");
        assert_eq!(consumer.current_tier(), StorageTier::OnDisk);
        assert_eq!(
            files_in(&dir.path().join("cache")).len(),
            2,
            "a file per graph"
        );
        for graph in [None, Some("trips")] {
            assert_eq!(
                database.store(graph).spilled_node_property_columns(),
                [embedding()]
            );
        }
        let spilled = database.root.memory();
        assert!(heap - spilled.total_bytes() >= 3_000 * 8);
        assert_eq!(
            [nearest(&database, None), nearest(&database, Some("trips"))],
            searched
        );
        assert_eq!(
            database.root.memory(),
            spilled,
            "the searches read the files"
        );
        assert_eq!(consumer.spill(usize::MAX).unwrap(), 0, "spilled already");

        consumer.reload().unwrap();
        assert_eq!(consumer.current_tier(), StorageTier::InMemory);
        assert_eq!(files_in(&dir.path().join("cache")), Vec::<PathBuf>::new());
        assert_embeddings(&database.root, &ids, "reloaded");
        assert_eq!(
            [nearest(&database, None), nearest(&database, Some("trips"))],
            searched
        );
    }

    /// Under memory pressure the buffer manager asks the consumers to spill,
    /// and the row-group store's embeddings go to their files; its reload
    /// brings them back once the pressure is gone.
    #[test]
    fn memory_pressure_spills_the_embeddings_and_a_reload_brings_them_back() {
        let dir = tempfile::tempdir().unwrap();
        let (database, ids) = indexed();
        let manager = BufferManager::with_budget(1 << 20);
        register_row_group_consumers(
            &manager,
            &database.root,
            Some(std::sync::Arc::new(
                crate::database::spill_directory::SpillDirectory::at(dir.path().join("cache")),
            )),
            None,
        );
        let names: Vec<String> = manager
            .snapshot_consumer_tiers()
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        assert!(names.contains(&"section:LpgStore".to_string()), "{names:?}");
        assert!(names.contains(&"section:VectorStore".to_string()));

        let grant = manager
            .try_allocate(900_000, MemoryRegion::ExecutionBuffers)
            .expect("below the hard limit");
        assert_eq!(
            database.root.spilled_node_property_columns(),
            [embedding()],
            "the pressure spilled the root's embeddings"
        );
        assert_eq!(files_in(&dir.path().join("cache")).len(), 2);
        drop(grant);

        assert!(manager.reload_eligible(1.0) >= 1);
        assert_eq!(
            database.root.spilled_node_property_columns(),
            Vec::<PropertyKey>::new()
        );
        assert_eq!(files_in(&dir.path().join("cache")), Vec::<PathBuf>::new());
        assert_embeddings(&database.root, &ids, "reloaded");
    }

    /// Without a spill directory the consumer does not spill.
    #[test]
    fn without_a_directory_nothing_spills() {
        let (database, _) = indexed();
        let consumer = RowGroupVectorConsumer::new(&database.root, None);
        assert!(!consumer.can_spill());
        assert!(consumer.spill(usize::MAX).is_err());
        assert_eq!(
            database.root.spilled_node_property_columns(),
            Vec::<PropertyKey>::new()
        );
    }
}

#[cfg(feature = "grafeo-file")]
mod file {
    use std::path::{Path, PathBuf};

    use grafeo_common::types::{EpochId, NodeId, PropertyKey, Value};
    use grafeo_common::utils::error::Result;
    use grafeo_storage::file::GrafeoFileManager;

    use super::super::ColumnSpillFiles;
    use super::{
        Database, assert_embeddings, continue_row_group_epochs, embedding, files_in, items,
        load_row_group_sections,
    };
    use crate::database::spill_directory::{SpillLayout, remove_stale_cache};

    /// A database on a row-group root with its file.
    pub(super) struct FileDatabase {
        pub(super) database: Database,
        pub(super) fm: GrafeoFileManager,
    }

    impl FileDatabase {
        pub(super) fn create(path: &Path) -> Self {
            Self {
                database: Database::new(),
                fm: GrafeoFileManager::create(path, None).unwrap(),
            }
        }

        /// Opens the file at `path` as a read-write open does: what a
        /// previous open or a crash left in its spill cache goes first,
        /// then the active image loads.
        pub(super) fn open(path: &Path) -> Self {
            if let Some(stale) = SpillLayout::for_open(None, Some(path), false).stale_cache {
                remove_stale_cache(&stale);
            }
            let fm = GrafeoFileManager::open(path, None).unwrap();
            let database = fm
                .read_image(|image| {
                    let database = Database::new();
                    let history = load_row_group_sections(
                        image,
                        Some(path),
                        &database.root,
                        &database.catalog,
                        #[cfg(feature = "triple-store")]
                        &database.rdf_store,
                    )?;
                    assert_eq!(history, None, "no history is kept");
                    Ok(database)
                })
                .unwrap();
            database
                .root
                .sync_epoch(EpochId::new(fm.active_header().epoch));
            continue_row_group_epochs(&database.root, &database.transaction_manager);
            Self { database, fm }
        }

        pub(super) fn checkpoint(&self) -> Result<()> {
            crate::database::flush::flush(&self.fm, &self.database.sources()).map(drop)
        }
    }

    /// The spill cache of the read-write open of `path`.
    pub(super) fn cache(path: &Path) -> PathBuf {
        SpillLayout::for_open(None, Some(path), false)
            .stale_cache
            .expect("a read-write file database has a spill cache")
    }

    /// Leaves a copy of every spill file in `cache` under another name, as
    /// a crash leaves its files: no process maps them, and the store that
    /// wrote them is gone.
    pub(super) fn leave_copies(cache: &Path) -> Vec<PathBuf> {
        files_in(cache)
            .into_iter()
            .map(|file| {
                let left = file.with_extension("left");
                std::fs::copy(&file, &left).unwrap();
                left
            })
            .collect()
    }

    /// The embedding of `id`, as the reopened database reads it.
    pub(super) fn embedding_at(database: &Database, id: NodeId) -> Option<Value> {
        use grafeo_core::graph::GraphStore;
        database.root.get_node_property(id, &embedding())
    }

    /// A checkpoint of a spilled column writes its values (the section
    /// writer reads its chunks from the file) and leaves it spilled; the
    /// database closes, a crash's copies of its spill files are left in the
    /// cache, and the reopen removes them and reads every value from the
    /// database file, its columns cold in memory, none spilled.
    #[test]
    fn a_checkpointed_spilled_column_reopens_from_the_database_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("items.grafeo");
        let created = FileDatabase::create(&path);
        let ids = items(&created.database, None, 3_000);
        let files = ColumnSpillFiles::in_dir(cache(&path));
        created
            .database
            .root
            .spill_node_property_column(&embedding(), &files)
            .unwrap();
        let spilled = created.database.root.memory();
        created.checkpoint().unwrap();
        assert_eq!(
            created.database.root.memory(),
            spilled,
            "the checkpoint read the file and left the column spilled"
        );
        let left = leave_copies(&cache(&path));
        assert_eq!(left.len(), 1);
        created.fm.close().unwrap();
        drop(created);
        assert_eq!(files_in(&cache(&path)), left, "the store's own file went");

        let opened = FileDatabase::open(&path);
        assert!(files_in(&cache(&path)).is_empty(), "the open removed them");
        assert_embeddings(&opened.database.root, &ids, "reopened");
        assert_eq!(
            opened.database.root.spilled_node_property_columns(),
            Vec::<PropertyKey>::new()
        );
        assert!(opened.database.root.memory().cold_columns > 0);
        assert_eq!(
            embedding_at(&opened.database, ids[2_999]),
            Some(super::embedding_of(2_999))
        );
    }
}

/// The crash points of a checkpoint without a WAL, as the row-group
/// sections' crash test lists them; the new header is on disk from the
/// fifth on.
#[cfg(all(feature = "grafeo-file", feature = "testing-crash-injection"))]
const CHECKPOINT_POINTS: [&str; 7] = [
    "flush:before_serialize",
    "flush:after_rotate",
    "checkpoint:after_chunks",
    "checkpoint:after_data_sync",
    "checkpoint:after_header",
    "checkpoint:before_trim",
    "flush:after_write",
];

/// A valid checkpoint, a spill of the embeddings, a write, then a crash at
/// each point of the next checkpoint, which reads the spilled chunks: the
/// reopen removes the spill files the crash left and opens the last
/// checkpoint whose header reached the file, every embedding read from the
/// database file. The database is built outside the crash, and nothing
/// closes it: its spill files are left as a crash leaves them.
#[cfg(all(feature = "grafeo-file", feature = "testing-crash-injection"))]
#[test]
fn a_crash_while_a_column_is_spilled_reopens_from_the_last_checkpoint() {
    use std::panic::AssertUnwindSafe;

    use grafeo_common::testing::crash::{CrashResult, with_crash_at};

    use self::file::{FileDatabase, cache, embedding_at, leave_copies};

    for point in 1..=CHECKPOINT_POINTS.len() + 1 {
        let name = CHECKPOINT_POINTS.get(point - 1).unwrap_or(&"none");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("items.grafeo");
        let database = FileDatabase::create(&path);
        let ids = items(&database.database, None, 3_000);
        database.checkpoint().unwrap();
        let first_iteration = database.fm.active_header().iteration;
        let files = ColumnSpillFiles::in_dir(cache(&path));
        let report = database
            .database
            .root
            .spill_node_property_column(&embedding(), &files)
            .unwrap();
        assert_eq!(report.spilled_groups, 1, "point {point} ({name})");
        // A write that leaves the embeddings spilled.
        let mut mia = NodeId::new(0);
        database.database.write(None, |writer| {
            mia = writer
                .create_node(
                    &["Item".to_string()],
                    vec![("name".to_string(), Value::from("Mia"))],
                )
                .unwrap();
        });
        assert_eq!(
            database.database.root.spilled_node_property_columns(),
            [embedding()],
            "point {point} ({name})"
        );

        let crashed = matches!(
            with_crash_at(
                u64::try_from(point).unwrap(),
                AssertUnwindSafe(|| database.checkpoint())
            ),
            CrashResult::Crashed
        );
        assert_eq!(
            crashed,
            point <= CHECKPOINT_POINTS.len(),
            "point {point} ({name})"
        );
        // What a crash leaves: the spill files, which no process maps.
        let left = leave_copies(&cache(&path));
        assert!(!left.is_empty(), "point {point} ({name})");
        drop(database);

        let reopened = FileDatabase::open(&path);
        assert!(
            files_in(&cache(&path)).is_empty(),
            "point {point} ({name}): the open removes what the crash left"
        );
        let header_written = point >= 5;
        assert_eq!(
            reopened.fm.active_header().iteration,
            first_iteration + u64::from(header_written),
            "point {point} ({name}): the active header"
        );
        assert_embeddings(
            &reopened.database.root,
            &ids,
            &format!("point {point} ({name})"),
        );
        assert_eq!(
            reopened
                .database
                .root
                .get_node_property(mia, &PropertyKey::new("name")),
            header_written.then(|| Value::from("Mia")),
            "point {point} ({name}): the write after the spill"
        );
        assert_eq!(
            embedding_at(&reopened.database, ids[2_999]),
            Some(embedding_of(2_999))
        );
    }
}
