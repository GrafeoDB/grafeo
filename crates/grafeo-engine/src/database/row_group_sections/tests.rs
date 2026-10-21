//! Databases on a row-group root, checkpointed and opened through the
//! engine's sections, flush and file (#412 S8): what a reopen holds, a
//! checkpoint during an open write, and a crash during a later checkpoint.

use std::collections::BTreeMap;
use std::sync::Arc;

use grafeo_common::storage::{ImageSource, MemoryImage};
use grafeo_common::types::{EpochId, NodeId, TransactionId, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_core::execution::operators::{GraphWriter, WriteTarget};
use grafeo_core::graph::apply::ChangeTarget;
use grafeo_core::graph::rowgroup::RowGroupStore;
use grafeo_core::graph::{Direction, GraphStore, GraphStoreMut};

use super::super::sections::CheckpointSources;
use super::{
    continue_row_group_epochs, load_row_group_sections, restore_row_group_history, row_group_root,
};
use crate::catalog::Catalog;
use crate::config::Config;
use crate::transaction::{TransactionChanges, TransactionManager};

/// A database on a row-group root, as `GrafeoDB` holds its parts: the root
/// store, the catalog and the transaction manager.
struct Database {
    root: Arc<RowGroupStore>,
    catalog: Arc<Catalog>,
    transaction_manager: Arc<TransactionManager>,
    #[cfg(feature = "triple-store")]
    rdf_store: Arc<grafeo_core::graph::rdf::RdfStore>,
    /// The WAL commits write their groups to, and checkpoints start a new
    /// segment of, when the test gives one.
    #[cfg(feature = "wal")]
    wal: Option<Arc<grafeo_storage::wal::Wal>>,
}

/// A transaction that is open: its id, its changes and its writer.
struct Open {
    transaction: TransactionId,
    changes: Arc<TransactionChanges>,
    writer: GraphWriter,
}

impl Database {
    fn new() -> Self {
        Self::with_config(&Config::in_memory())
    }

    /// A database whose root is the one `config` builds (see
    /// [`row_group_root`]).
    fn with_config(config: &Config) -> Self {
        Self {
            root: row_group_root(config),
            catalog: Arc::new(Catalog::new()),
            transaction_manager: Arc::new(TransactionManager::new()),
            #[cfg(feature = "triple-store")]
            rdf_store: Arc::new(grafeo_core::graph::rdf::RdfStore::new()),
            #[cfg(feature = "wal")]
            wal: None,
        }
    }

    /// What a checkpoint of this database writes from.
    fn sources(&self) -> CheckpointSources {
        CheckpointSources {
            store: Some(Arc::clone(&self.root)),
            catalog: Arc::clone(&self.catalog),
            transaction_manager: Arc::clone(&self.transaction_manager),
            #[cfg(feature = "triple-store")]
            rdf_store: Arc::clone(&self.rdf_store),
            #[cfg(feature = "wal")]
            wal: self.wal.clone(),
            #[cfg(feature = "wal")]
            wal_records_at_checkpoint: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// The store of the graph `graph` (`None` for the default graph).
    fn store(&self, graph: Option<&str>) -> Arc<RowGroupStore> {
        match graph {
            None => Arc::clone(&self.root),
            Some(name) => self.root.graph(name).expect("a graph the test created"),
        }
    }

    /// Begins a transaction that writes the graph `graph`, as a private
    /// transaction of the direct API does.
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

    /// Commits `open` as the engine commits: an epoch, then the stamp in the
    /// commit's turn, with the root store following the database's epoch.
    fn commit(&self, open: Open) -> EpochId {
        let Open {
            transaction,
            changes,
            writer,
        } = open;
        drop(writer);
        let commit = self.transaction_manager.start_commit(transaction).unwrap();
        let epoch = commit.epoch();
        // With a WAL, the commit's group first, as a database commits.
        #[cfg(feature = "wal")]
        let commit = match &self.wal {
            Some(wal) => {
                let (commit, logged) = super::super::wal_log::write_commit(
                    wal,
                    commit,
                    transaction,
                    &changes,
                    grafeo_common::storage::log_record::Origin::Transaction,
                )
                .unwrap();
                if let Some(logged) = logged {
                    super::super::wal_log::sync_commit(wal, logged).unwrap();
                }
                commit
            }
            None => commit,
        };
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

    /// Runs `write` as one transaction on the graph `graph`, and commits it.
    fn write(&self, graph: Option<&str>, write: impl FnOnce(&GraphWriter)) -> EpochId {
        let open = self.begin(graph);
        write(&open.writer);
        self.commit(open)
    }

    /// Every section a checkpoint writes, as an image in memory, built under
    /// a commit hold as a checkpoint builds them; and the epoch it holds.
    fn image(&self) -> (MemoryImage, EpochId) {
        let mut image = MemoryImage::new();
        let commits = self.transaction_manager.hold_commits().unwrap();
        let sources = self.sources();
        for section in sources.sections(&commits) {
            image
                .begin_section(section.section_type(), section.version())
                .unwrap();
            section.write_to(&mut image).unwrap();
        }
        (image, EpochId::new(sources.epoch()))
    }

    /// A database loaded from `image` (of the database file `file`), with
    /// its epochs continued as an open continues them.
    fn load(image: &dyn ImageSource, file: Option<&std::path::Path>) -> Result<Self> {
        Self::load_with(image, file, &Config::in_memory())
    }

    /// A database with `config`'s root (see [`row_group_root`]) loaded from
    /// `image`, as [`load`](Self::load) loads one.
    fn load_with(
        image: &dyn ImageSource,
        file: Option<&std::path::Path>,
        config: &Config,
    ) -> Result<Self> {
        let database = Self::with_config(config);
        let history = load_row_group_sections(
            image,
            file,
            &database.root,
            &database.catalog,
            #[cfg(feature = "triple-store")]
            &database.rdf_store,
        )?;
        restore_row_group_history(&database.root, &database.transaction_manager, history);
        continue_row_group_epochs(&database.root, &database.transaction_manager);
        Ok(database)
    }
}

/// Label names as the writer takes them.
fn names(labels: &[&str]) -> Vec<String> {
    labels.iter().map(ToString::to_string).collect()
}

/// Values as the writer takes them.
fn values(pairs: &[(&str, Value)]) -> Vec<(String, Value)> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_string(), value.clone()))
        .collect()
}

/// A node as a committed reader sees it: its labels and values, sorted.
type NodeView = (Vec<String>, Vec<(String, Value)>);

/// An edge as a committed reader sees it: its ends, type and values.
type EdgeView = (u64, u64, String, Vec<(String, Value)>);

/// What a committed reader at the store's epoch sees of a graph, by id.
fn committed(store: &RowGroupStore) -> (BTreeMap<u64, NodeView>, BTreeMap<u64, EdgeView>) {
    committed_of(store)
}

/// What a reader now sees of a graph of either store, by id.
fn committed_of(store: &dyn GraphStore) -> (BTreeMap<u64, NodeView>, BTreeMap<u64, EdgeView>) {
    let sorted = |map: &grafeo_common::types::PropertyMap| {
        let mut pairs: Vec<(String, Value)> = map
            .iter()
            .map(|(key, value)| (key.as_str().to_string(), value.clone()))
            .collect();
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        pairs
    };
    let mut nodes = BTreeMap::new();
    let mut edges = BTreeMap::new();
    for id in store.node_ids() {
        let node = store.get_node(id).expect("a node it lists");
        let mut labels: Vec<String> = node.labels.iter().map(ToString::to_string).collect();
        labels.sort();
        nodes.insert(id.as_u64(), (labels, sorted(&node.properties)));
        for (_, edge) in store.edges_from(id, Direction::Outgoing) {
            if let Some(edge) = store.get_edge(edge) {
                edges.insert(
                    edge.id.as_u64(),
                    (
                        edge.src.as_u64(),
                        edge.dst.as_u64(),
                        edge.edge_type.to_string(),
                        sorted(&edge.properties),
                    ),
                );
            }
        }
    }
    (nodes, edges)
}

/// The committed state of the root and of trips.
type Images = [(BTreeMap<u64, NodeView>, BTreeMap<u64, EdgeView>); 2];

fn images(database: &Database) -> Images {
    [
        committed(&database.root),
        committed(&database.store(Some("trips"))),
    ]
}

/// The nodes of [`travels`].
#[derive(Clone, Copy)]
struct Travellers {
    alix: NodeId,
    gus: NodeId,
    /// In trips: a document with an embedding.
    guide: NodeId,
    /// In trips.
    paris: NodeId,
}

/// Alix and Gus, who knows her, in the default graph, with an index on the
/// names and a text index on their bios; in the named graph trips, Vincent
/// visited Paris and wrote a guide, with a vector index on its embedding.
fn travels(database: &Database) -> Travellers {
    let mut people = (NodeId::new(0), NodeId::new(0));
    database.write(None, |writer| {
        let alix = writer
            .create_node(
                &names(&["Person"]),
                values(&[
                    ("name", Value::from("Alix")),
                    ("bio", Value::from("Alix lives in Amsterdam")),
                ]),
            )
            .unwrap();
        let gus = writer
            .create_node(
                &names(&["Person", "Employee"]),
                values(&[
                    ("name", Value::from("Gus")),
                    ("bio", Value::from("Gus cycles through Paris")),
                ]),
            )
            .unwrap();
        writer
            .create_edge(gus, alix, "KNOWS", values(&[("since", Value::Int64(2019))]))
            .unwrap();
        people = (alix, gus);
    });
    assert!(database.root.create_graph("trips").unwrap());
    let mut trip = (NodeId::new(0), NodeId::new(0));
    database.write(Some("trips"), |writer| {
        let vincent = writer
            .create_node(
                &names(&["Person"]),
                values(&[("name", Value::from("Vincent"))]),
            )
            .unwrap();
        let paris = writer
            .create_node(&names(&["City"]), values(&[("name", Value::from("Paris"))]))
            .unwrap();
        let guide = writer
            .create_node(
                &names(&["Doc"]),
                values(&[("embedding", Value::Vector(vec![1.0, 0.0].into()))]),
            )
            .unwrap();
        writer
            .create_edge(vincent, paris, "VISITED", Vec::new())
            .unwrap();
        trip = (guide, paris);
    });

    database.root.create_property_index("name");
    #[cfg(feature = "text-index")]
    database.root.add_text_index(
        "Person",
        "bio",
        Arc::new(parking_lot::RwLock::new(
            super::super::index::text_index_from_data(
                &*database.root,
                "Person",
                "bio",
                &grafeo_core::index::text::TextIndexOptions::new(),
            ),
        )),
    );
    #[cfg(feature = "vector-index")]
    {
        let trips = database.store(Some("trips"));
        let index = super::super::index::vector_index_from_data(
            &*trips,
            "Doc",
            "embedding",
            Some(2),
            Some("cosine"),
            None,
            None,
            None,
        )
        .unwrap();
        trips.add_vector_index("Doc", "embedding", Arc::new(index));
    }
    Travellers {
        alix: people.0,
        gus: people.1,
        guide: trip.0,
        paris: trip.1,
    }
}

/// What [`open_writes`] created: Mia in the default graph, a second guide
/// in trips.
struct OpenWrites {
    root: Open,
    trips: Open,
    mia: NodeId,
    second_guide: NodeId,
}

/// Open transactions in the default graph and in trips that change values,
/// labels and edges, create nodes and delete an edge, each of which an
/// index would show: Alix moves to Berlin, Gus travels, Mia joins, the
/// guide's embedding turns, Paris is no City.
fn open_writes(database: &Database, travellers: Travellers) -> OpenWrites {
    let root = database.begin(None);
    root.writer
        .set_node_properties(
            travellers.alix,
            &values(&[("bio", Value::from("Alix moved to Berlin"))]),
            false,
        )
        .unwrap();
    root.writer
        .add_labels(travellers.gus, &names(&["Traveller"]))
        .unwrap();
    let mia = root
        .writer
        .create_node(
            &names(&["Person"]),
            values(&[
                ("name", Value::from("Mia")),
                ("bio", Value::from("Mia works in Berlin")),
            ]),
        )
        .unwrap();
    for (_, edge) in database
        .root
        .edges_from(travellers.gus, Direction::Outgoing)
    {
        root.writer.delete_edge(edge).unwrap();
    }

    let trips = database.begin(Some("trips"));
    trips
        .writer
        .set_node_properties(
            travellers.guide,
            &values(&[("embedding", Value::Vector(vec![0.0, 1.0].into()))]),
            false,
        )
        .unwrap();
    trips
        .writer
        .remove_labels(travellers.paris, &names(&["City"]))
        .unwrap();
    let second_guide = trips
        .writer
        .create_node(
            &names(&["Doc"]),
            values(&[("embedding", Value::Vector(vec![0.6, 0.8].into()))]),
        )
        .unwrap();
    OpenWrites {
        root,
        trips,
        mia,
        second_guide,
    }
}

/// Checks that `loaded` holds `expected` and the indexes of [`travels`],
/// built from what it holds: each name through the name index, the bios of
/// Berlin through the text index, the documents of trips in the vector
/// index, and nothing else in them.
fn assert_holds(loaded: &Database, expected: &Images, what: &str) {
    assert_eq!(&images(loaded), expected, "{what}: the data");
    assert_eq!(
        loaded.root.graph_names(),
        ["trips"],
        "{what}: the named graphs"
    );
    let root = &loaded.root;
    assert!(root.has_property_index("name"), "{what}: the name index");
    for (_, values) in expected[0].0.values() {
        let name = values
            .iter()
            .find(|(key, _)| key == "name")
            .map(|(_, value)| value.clone())
            .unwrap();
        assert_eq!(
            root.find_nodes_by_property("name", &name).len(),
            1,
            "{what}: {name:?} through the index"
        );
    }
    #[cfg(feature = "text-index")]
    {
        let bios = |query: &str| -> Vec<NodeId> {
            let mut found: Vec<NodeId> = grafeo_core::graph::GraphStoreSearch::text_search(
                &**root, "Person", "bio", query, 10,
            )
            .into_iter()
            .map(|(id, _)| id)
            .collect();
            found.sort_unstable();
            found
        };
        let berlin: Vec<NodeId> = expected[0]
            .0
            .iter()
            .filter(|(_, (_, values))| {
                values.iter().any(|(key, value)| {
                    key == "bio" && value.as_str().is_some_and(|bio| bio.contains("Berlin"))
                })
            })
            .map(|(id, _)| NodeId::new(*id))
            .collect();
        assert_eq!(bios("Berlin"), berlin, "{what}: the bios of Berlin");
        assert!(
            root.get_text_index("Person", "bio").is_some(),
            "{what}: the text index"
        );
    }
    #[cfg(feature = "vector-index")]
    {
        let trips = loaded.store(Some("trips"));
        let index = trips.get_vector_index("Doc", "embedding").unwrap();
        let docs: Vec<NodeId> = expected[1]
            .0
            .iter()
            .filter(|(_, (labels, _))| labels.iter().any(|label| label == "Doc"))
            .map(|(id, _)| NodeId::new(*id))
            .collect();
        assert_eq!(index.len(), docs.len(), "{what}: the vector index of trips");
        assert!(
            docs.iter().all(|doc| index.contains(*doc)),
            "{what}: the documents of trips"
        );
    }
}

/// A copy through the sections, taken while transactions are open in the
/// default graph and in a named graph, holds each graph's committed state:
/// no value, label, node or deletion of the open transactions, also not in
/// the indexes the load builds; it continues at the epoch it holds, and a
/// commit after the load lands above it.
#[test]
fn a_copy_during_open_writes_holds_the_committed_state_of_every_graph() {
    let database = Database::new();
    let travellers = travels(&database);
    let before = images(&database);
    let open = open_writes(&database, travellers);

    let (image, epoch) = database.image();
    let loaded = Database::load(&image, None).unwrap();
    assert_holds(&loaded, &before, "the copy");
    assert!(
        loaded.root.get_node(open.mia).is_none()
            && loaded
                .root
                .find_nodes_by_property("name", &Value::from("Mia"))
                .is_empty(),
        "Mia was never committed"
    );
    assert!(
        loaded
            .store(Some("trips"))
            .get_node(open.second_guide)
            .is_none(),
        "nor the second guide"
    );
    assert_eq!(loaded.transaction_manager.current_epoch(), epoch);
    assert_eq!(loaded.root.current_epoch(), epoch);
    assert_eq!(loaded.store(Some("trips")).current_epoch(), epoch);

    let jules = loaded.write(None, |writer| {
        writer
            .create_node(
                &names(&["Person"]),
                values(&[("name", Value::from("Jules"))]),
            )
            .unwrap();
    });
    assert_eq!(jules, EpochId::new(epoch.as_u64() + 1), "the next commit");
    assert_eq!(
        loaded
            .root
            .find_nodes_by_property("name", &Value::from("Jules"))
            .len(),
        1,
        "the index the load built takes the next commit"
    );

    // The open transactions commit in the source: the next copy holds them.
    database.commit(open.root);
    database.commit(open.trips);
    let after = images(&database);
    let (image, _) = database.image();
    let again = Database::load(&image, None).unwrap();
    assert_eq!(images(&again), after, "after the commits");
    assert_eq!(
        again
            .root
            .find_nodes_by_property("name", &Value::from("Mia"))
            .len(),
        1
    );
}

#[cfg(feature = "grafeo-file")]
mod file {
    use grafeo_storage::file::GrafeoFileManager;

    use super::*;

    /// A database on a row-group root with its file.
    pub(super) struct FileDatabase {
        pub(super) database: Database,
        pub(super) fm: GrafeoFileManager,
    }

    impl FileDatabase {
        /// A new database file at `path`.
        pub(super) fn create(path: &std::path::Path) -> Self {
            Self::create_with(path, &Config::persistent(path))
        }

        /// A new database file at `path` created with `config`, which it
        /// stores (#551) as `GrafeoDB` stores it when it creates the file.
        pub(super) fn create_with(path: &std::path::Path, config: &Config) -> Self {
            let database = Database::with_config(config);
            database.catalog.set_stored_config(Some(
                super::super::super::stored_config::record_of(config, path),
            ));
            Self {
                database,
                fm: GrafeoFileManager::create(path, None).unwrap(),
            }
        }

        /// Opens the database file at `path`: loads its active image and
        /// continues at the epoch of its last checkpoint.
        pub(super) fn open(path: &std::path::Path) -> Self {
            Self::open_with(path, Config::persistent(path)).unwrap()
        }

        /// Opens the database file at `path` with `config`, to which the
        /// configuration the file stores applies before the root is built,
        /// as `GrafeoDB` opens a file: the stored values of what `config`
        /// does not set, a refusal of a fixed one it sets otherwise.
        pub(super) fn open_with(path: &std::path::Path, config: Config) -> Result<Self> {
            let stored_config = super::super::super::stored_config::read(
                path,
                &super::super::super::encryption::DatabaseKeys::from_config(&config),
            )?;
            let config = match stored_config {
                Some(stored) => super::super::super::stored_config::applied(config, &stored, path)?,
                None => config,
            };
            let fm = GrafeoFileManager::open(path, None).unwrap();
            let database = fm
                .read_image(|image| Database::load_with(image, Some(path), &config))
                .unwrap();
            database
                .root
                .sync_epoch(EpochId::new(fm.active_header().epoch));
            continue_row_group_epochs(&database.root, &database.transaction_manager);
            Ok(Self { database, fm })
        }

        /// A checkpoint, through the flush every checkpoint of a database
        /// takes.
        pub(super) fn checkpoint(&self) -> Result<()> {
            super::super::super::flush::flush(&self.fm, &self.database.sources()).map(drop)
        }

        /// A checkpoint, then the file closed.
        pub(super) fn close(self) {
            self.checkpoint().unwrap();
            self.fm.close().unwrap();
        }
    }

    /// A database on row groups opens what it checkpointed and closed: every
    /// graph, its indexes built from the rows, the epoch of its last
    /// checkpoint, which commits after the open continue; and so again
    /// after a second close.
    #[test]
    fn a_reopened_database_holds_what_it_closed_with() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("travels.grafeo");
        let created = FileDatabase::create(&path);
        travels(&created.database);
        created.checkpoint().unwrap();
        created.database.write(None, |writer| {
            writer
                .create_node(
                    &names(&["Person"]),
                    values(&[
                        ("name", Value::from("Jules")),
                        ("bio", Value::from("Jules drives to Berlin")),
                    ]),
                )
                .unwrap();
        });
        let closed = images(&created.database);
        let epoch = created.database.transaction_manager.current_epoch();
        created.close();

        let opened = FileDatabase::open(&path);
        assert_holds(&opened.database, &closed, "the reopen");
        assert_eq!(opened.fm.active_header().epoch, epoch.as_u64());
        assert_eq!(opened.database.transaction_manager.current_epoch(), epoch);
        let next = opened.database.write(Some("trips"), |writer| {
            writer
                .create_node(
                    &names(&["City"]),
                    values(&[("name", Value::from("Prague"))]),
                )
                .unwrap();
        });
        assert!(next > epoch, "a commit after the open continues the epochs");
        let closed = images(&opened.database);
        opened.close();

        let again = FileDatabase::open(&path);
        assert_holds(&again.database, &closed, "the second reopen");
        assert_eq!(again.database.transaction_manager.current_epoch(), next);
    }

    /// A checkpoint taken while transactions are open in the default graph
    /// and in trips holds what was committed: a database that ends right
    /// after it, its transactions never committed, opens without any of
    /// their writes, in its data or in its indexes.
    #[test]
    fn a_checkpoint_during_open_writes_reopens_without_them() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("travels.grafeo");
        let created = FileDatabase::create(&path);
        let travellers = travels(&created.database);
        let committed_state = images(&created.database);
        let open = open_writes(&created.database, travellers);
        created.checkpoint().unwrap();
        // The process ends: the transactions are never committed.
        drop(open);
        drop(created);

        let opened = FileDatabase::open(&path);
        assert_holds(&opened.database, &committed_state, "after the checkpoint");
        assert_eq!(
            opened
                .database
                .root
                .find_nodes_by_property("name", &Value::from("Mia")),
            Vec::<NodeId>::new(),
            "Mia was never committed"
        );
    }
}

/// The crash points of a checkpoint without a WAL, in the order the flush
/// and the file manager reach them; the new header is on disk from the
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

/// A valid checkpoint, then a crash at each point of the next one: the
/// file opens with the last checkpoint whose header reached it, the first
/// one up to the header write and the second from it on, each with every
/// graph and the indexes built from its rows, and never with a write of a
/// transaction that was open. The database is built outside the crash, and
/// nothing closes it after: the reopen reads the files the crash left.
#[cfg(all(feature = "grafeo-file", feature = "testing-crash-injection"))]
#[test]
fn a_crash_during_the_next_checkpoint_keeps_the_last_valid_one() {
    crash_during_the_next_checkpoint(|path| Config::persistent(path));
}

/// [`a_crash_during_the_next_checkpoint_keeps_the_last_valid_one`] for a
/// database created without backward edges (10b): every reopen after a
/// crash keeps none, in the root and in trips.
#[cfg(all(feature = "grafeo-file", feature = "testing-crash-injection"))]
#[test]
fn a_crash_without_backward_edges_keeps_the_last_valid_checkpoint() {
    crash_during_the_next_checkpoint(|path| Config::persistent(path).without_backward_edges());
}

/// A valid checkpoint of a database created with `config`, then a crash at
/// each point of the next one (see
/// [`a_crash_during_the_next_checkpoint_keeps_the_last_valid_one`]); each
/// reopen keeps the backward edges the database was created with.
#[cfg(all(feature = "grafeo-file", feature = "testing-crash-injection"))]
fn crash_during_the_next_checkpoint(config: impl Fn(&std::path::Path) -> Config) {
    use std::panic::AssertUnwindSafe;

    use grafeo_common::testing::crash::{CrashResult, with_crash_at};

    use self::file::FileDatabase;

    for point in 1..=CHECKPOINT_POINTS.len() + 1 {
        let name = CHECKPOINT_POINTS.get(point - 1).unwrap_or(&"none");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("travels.grafeo");
        let config = config(&path);
        let backward_edges = config.backward_edges();
        let database = FileDatabase::create_with(&path, &config);
        travels(&database.database);
        database.checkpoint().unwrap();
        let first = images(&database.database);
        let first_iteration = database.fm.active_header().iteration;

        database.database.write(None, |writer| {
            writer
                .create_node(
                    &names(&["Person"]),
                    values(&[
                        ("name", Value::from("Butch")),
                        ("bio", Value::from("Butch boxes in Berlin")),
                    ]),
                )
                .unwrap();
        });
        let open = database.database.begin(Some("trips"));
        open.writer
            .create_node(
                &names(&["City"]),
                values(&[("name", Value::from("Barcelona"))]),
            )
            .unwrap();
        let second = images(&database.database);

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
            "point {point} ({name}): a checkpoint has {} crash points",
            CHECKPOINT_POINTS.len()
        );
        // What a crash leaves: nothing closes the database or commits.
        drop(open);
        drop(database);

        let reopened = FileDatabase::open(&path);
        for store in [
            Arc::clone(&reopened.database.root),
            reopened.database.store(Some("trips")),
        ] {
            assert_eq!(
                store.keeps_backward_edges(),
                backward_edges,
                "point {point} ({name}): the backward edges it was created with"
            );
        }
        let header_written = point >= 5;
        let (expected, iteration) = if header_written {
            (&second, first_iteration + 1)
        } else {
            (&first, first_iteration)
        };
        assert_eq!(
            reopened.fm.active_header().iteration,
            iteration,
            "point {point} ({name}): the active header"
        );
        assert_holds(
            &reopened.database,
            expected,
            &format!("point {point} ({name})"),
        );
        assert!(
            !images(&reopened.database)[1]
                .0
                .values()
                .any(|(_, values)| values
                    .iter()
                    .any(|(_, value)| *value == Value::from("Barcelona"))),
            "point {point} ({name}): the open write is in no checkpoint"
        );
    }
}

// The paths of earlier formats, snapshots and copies on a row-group root:
#[cfg(all(feature = "grafeo-file", feature = "wal"))]
mod legacy;

// History and time travel on a row-group root (H2c): every tier, the
// retention window, a crash and a reopen.
#[cfg(all(feature = "grafeo-file", feature = "wal"))]
mod history;

// A row-group root without backward edges (10b): incoming reads, detach
// deletes, checkpoints and reopens.
mod backward_edges;
