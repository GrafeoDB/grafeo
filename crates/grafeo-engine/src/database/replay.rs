//! Replay of the write-ahead log's groups (WAL v2) when a database opens.
//!
//! The image of the last checkpoint holds every commit up to its
//! `checkpoint_lsn`; the log holds the groups after it. Replay scans the log
//! from there and applies each complete group, in log order, at the epoch its
//! [`GroupBegin`] names, through the functions the live writes use: a data
//! record through [`ChangeTarget::apply`] as a [`Writer::Replay`] (strict: a
//! record that does not fit the store fails the open), a triple through the
//! RDF store's own insert and remove, a standalone op through
//! [`standalone::apply`] or [`standalone::apply_rdf`]. A group the scan finds
//! incomplete (no LAST frame) was never committed and is not applied.
//!
//! While the database keeps history, each group's time is recorded in the
//! history's time index (the time its commit recorded live), and the store,
//! told it keeps history, keeps what the group's writes replace, stamped at
//! its epoch: a crash keeps the history the log holds (H2c).
//!
//! [`GroupBegin`]: grafeo_common::storage::log_record::GroupBegin
//! [`ChangeTarget::apply`]: grafeo_core::graph::apply::ChangeTarget::apply
//! [`Writer::Replay`]: grafeo_core::graph::apply::Writer::Replay

use std::path::Path;
use std::sync::Arc;

use grafeo_common::change::{DataModel, DataOp, StandaloneOp};
use grafeo_common::storage::{LogRecord, read_log_records};
use grafeo_common::types::EpochId;
use grafeo_common::utils::error::{Error, Result, StorageError};
use grafeo_core::graph::apply::Writer;
#[cfg(feature = "triple-store")]
use grafeo_core::graph::rdf::RdfStore;
use grafeo_core::graph::rowgroup::RowGroupStore;
use grafeo_storage::wal::{ScanEnd, ScanOptions, WalScan};

use super::catalog_section::GraphIndexes;
use super::root_store::RootStore;
use super::standalone::{self, Applying};
use crate::catalog::Catalog;

/// What a replay applied, and where the log ends.
#[derive(Debug)]
pub(crate) struct Replayed {
    /// Where the scan ended: the writer continues at its `end_lsn` once a
    /// read-write open cut what follows (see [`ScanEnd::cut_torn_tail`]).
    pub(crate) end: ScanEnd,
    /// How many groups were applied.
    pub(crate) groups: u64,
}

/// The stores and the catalog a replay applies groups to.
pub(crate) struct Targets<'a, S: RootStore = RowGroupStore> {
    /// The root LPG store, which holds the named graphs.
    pub(crate) store: &'a Arc<S>,
    /// The RDF store.
    #[cfg(feature = "triple-store")]
    pub(crate) rdf: &'a RdfStore,
    /// The catalog.
    pub(crate) catalog: &'a Catalog,
    /// The vector and text indexes the replayed index records put, built
    /// once the data is loaded.
    pub(crate) unbuilt: &'a mut Vec<GraphIndexes>,
    /// The transaction manager whose history records each group's time;
    /// `None` records nothing.
    pub(crate) history: Option<&'a crate::transaction::TransactionManager>,
}

/// Replays the groups of the WAL in `dir` that follow `options.from_lsn`
/// into `targets`, and returns where the log ends. Nothing is written: the
/// caller cuts a torn tail for a read-write open.
///
/// # Errors
///
/// [`StorageError::RecoveryFailed`] naming the WAL and the group for a group
/// that does not start with its `GroupBegin`, a record that does not decode,
/// and a record that does not apply (the log does not fit the image); the
/// scanner's errors (damage the scan does not end at, a gap, a foreign or
/// newer segment, a wrong key) as I/O or storage errors.
pub(crate) fn replay<S: RootStore>(
    dir: &Path,
    options: ScanOptions,
    targets: &mut Targets<'_, S>,
) -> Result<Replayed> {
    let mut scan = WalScan::open(dir, options).map_err(Error::from)?;
    let mut groups = 0;
    while let Some(mut group) = scan.next_group().map_err(Error::from)? {
        let start = group.start_lsn();
        let failed = |reason: String| {
            Error::Storage(StorageError::RecoveryFailed(format!(
                "cannot replay the group at log position {start} of the WAL {}: {reason}",
                dir.display()
            )))
        };
        let mut records = Vec::new();
        while let Some(payload) = group.next_payload().map_err(Error::from)? {
            read_log_records(payload, &mut |record| {
                records.push(record);
                Ok(())
            })
            .map_err(|error| failed(error.to_string()))?;
        }
        let mut records = records.into_iter();
        let Some(LogRecord::GroupBegin(begin)) = records.next() else {
            return Err(failed("it does not start with its GroupBegin".to_string()));
        };
        for record in records {
            apply_record(record, begin.epoch, targets)
                .map_err(|error| failed(error.to_string()))?;
        }
        targets.store.sync_epoch(begin.epoch);
        if let Some(manager) = targets.history {
            manager.record_commit_time(begin.epoch, begin.timestamp_ms);
        }
        groups += 1;
    }
    let end = scan.finish().map_err(Error::from)?;
    Ok(Replayed { end, groups })
}

/// Applies one record of a group at `epoch`.
fn apply_record<S: RootStore>(
    record: LogRecord,
    epoch: EpochId,
    targets: &mut Targets<'_, S>,
) -> Result<()> {
    match record {
        LogRecord::GroupBegin(_) => Err(Error::Internal(
            "a second GroupBegin inside a group".to_string(),
        )),
        LogRecord::Data { graph, op } if op.model() == DataModel::Rdf => {
            apply_triple(graph.as_deref(), op, targets)
        }
        LogRecord::Data { graph, op } => {
            let store = match graph.as_deref() {
                None => Arc::clone(targets.store),
                Some(name) => targets.store.graph(name).ok_or_else(|| {
                    Error::Internal(format!("a write to the graph {name}, which does not exist"))
                })?,
            };
            store
                .apply(&op, Writer::Replay { epoch })
                .map(|_| ())
                .map_err(|error| Error::Internal(error.to_string()))
        }
        LogRecord::Standalone(op) => apply_standalone(&op, targets),
    }
}

/// Applies a logged triple insert or delete to the RDF store.
#[cfg(feature = "triple-store")]
fn apply_triple<S: RootStore>(
    graph: Option<&str>,
    op: DataOp,
    targets: &Targets<'_, S>,
) -> Result<()> {
    use grafeo_core::graph::rdf::Triple;

    match op {
        DataOp::InsertTriple { triple } => {
            targets.rdf.insert_into(graph, Triple::from(&*triple));
            Ok(())
        }
        DataOp::DeleteTriple { triple } => {
            targets.rdf.remove_from(graph, &Triple::from(&*triple));
            Ok(())
        }
        other => Err(Error::Internal(format!("{other:?} is no RDF op"))),
    }
}

/// A build without the triple store cannot replay a triple: the next
/// checkpoint would write the file without it.
#[cfg(not(feature = "triple-store"))]
fn apply_triple<S: RootStore>(
    _graph: Option<&str>,
    _op: DataOp,
    _targets: &Targets<'_, S>,
) -> Result<()> {
    Err(Error::Internal(
        "the log holds RDF triples, which this build (without the triple store) cannot keep"
            .to_string(),
    ))
}

/// Applies a logged standalone op as the statement that logged it did.
fn apply_standalone<S: RootStore>(op: &StandaloneOp, targets: &mut Targets<'_, S>) -> Result<()> {
    if matches!(op, StandaloneOp::RdfGraph(_)) {
        #[cfg(feature = "triple-store")]
        {
            return standalone::apply_rdf(
                op,
                targets.rdf,
                &Applying::Replay {
                    unbuilt: targets.unbuilt,
                },
            );
        }
        #[cfg(not(feature = "triple-store"))]
        {
            return Err(Error::Internal(
                "the log holds an RDF graph operation, which this build cannot apply".to_string(),
            ));
        }
    }
    standalone::apply(
        op,
        None,
        targets.store,
        targets.catalog,
        &mut Applying::Replay {
            unbuilt: targets.unbuilt,
        },
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use grafeo_common::change::{
        Before, ChangeSet, DataModel, DataOp, GraphRef, PendingVersion, StandaloneOp,
    };
    use grafeo_common::storage::log_record::Origin;
    use grafeo_common::types::{ArcStr, EpochId, NodeId, PropertyKey, TransactionId, Value};
    use grafeo_common::utils::error::{Error, StorageError};
    use grafeo_core::graph::GraphStore as _;
    use grafeo_core::graph::rowgroup::RowGroupStore;
    use grafeo_storage::wal::{DurabilityMode, ScanOptions, TailKind, Wal, WalOptions};

    use super::{Replayed, Targets, replay};
    use crate::catalog::Catalog;
    use crate::database::wal_log::{write_group, write_standalone};
    use crate::transaction::StandaloneChange;

    const DATABASE: u128 = 0x6772_6166_656f_0000_0000_0000_0000_0007;

    fn wal(dir: &std::path::Path, start_lsn: u64) -> Wal {
        Wal::open(
            dir,
            WalOptions {
                durability: DurabilityMode::NoSync,
                ..WalOptions::new(DATABASE, start_lsn)
            },
        )
        .unwrap()
    }

    /// A set of LPG writes in `graph` (`None` for the default graph).
    fn set_of(graph: Option<&str>, ops: Vec<(DataOp, Before)>) -> ChangeSet {
        let mut set = ChangeSet::new();
        let slot = set
            .slot(GraphRef {
                model: DataModel::Lpg,
                key: graph.map(ArcStr::from),
            })
            .unwrap();
        for (op, before) in ops {
            set.push(slot, op, before, PendingVersion::Created).unwrap();
        }
        set
    }

    fn create(id: u64, name: &str) -> (DataOp, Before) {
        (
            DataOp::CreateNode {
                id: NodeId::new(id),
                labels: [ArcStr::from("Person")].into_iter().collect(),
                properties: vec![(PropertyKey::new("name"), Value::from(name))],
            },
            Before::Absent,
        )
    }

    fn set_city(id: u64, city: &str) -> (DataOp, Before) {
        (
            DataOp::SetNodeProperty {
                id: NodeId::new(id),
                key: PropertyKey::new("city"),
                value: Value::from(city),
            },
            Before::Value(None),
        )
    }

    fn commit(wal: &Wal, transaction: u64, set: &ChangeSet, epoch: u64) {
        let mut group = wal.begin_group(TransactionId::new(transaction)).unwrap();
        write_group(&mut group, set, EpochId::new(epoch), Origin::Transaction).unwrap();
        group.finish().unwrap();
    }

    fn replay_into(
        dir: &std::path::Path,
        from_lsn: u64,
        store: &Arc<RowGroupStore>,
        catalog: &Catalog,
    ) -> grafeo_common::utils::error::Result<Replayed> {
        #[cfg(feature = "triple-store")]
        let rdf = grafeo_core::graph::rdf::RdfStore::new();
        let mut unbuilt = Vec::new();
        replay(
            dir,
            ScanOptions::new(DATABASE, from_lsn),
            &mut Targets {
                store,
                #[cfg(feature = "triple-store")]
                rdf: &rdf,
                catalog,
                unbuilt: &mut unbuilt,
                history: None,
            },
        )
    }

    fn name_and_city(store: &RowGroupStore, id: u64) -> (Option<Value>, Option<Value>) {
        (
            store.get_node_property(NodeId::new(id), &PropertyKey::new("name")),
            store.get_node_property(NodeId::new(id), &PropertyKey::new("city")),
        )
    }

    /// Every complete group is applied, in log order, at its epoch: the store
    /// ends at the last group's epoch with every write of every group.
    #[test]
    fn replay_applies_every_group_at_its_epoch() {
        let dir = tempfile::tempdir().unwrap();
        {
            let wal = wal(dir.path(), 0);
            commit(
                &wal,
                2,
                &set_of(None, vec![create(3, "Alix"), create(19, "Gus")]),
                1,
            );
            commit(&wal, 3, &set_of(None, vec![set_city(3, "Amsterdam")]), 2);
        }
        let store = Arc::new(RowGroupStore::new());
        let replayed = replay_into(dir.path(), 0, &store, &Catalog::new()).unwrap();

        assert_eq!(replayed.groups, 2);
        assert!(replayed.end.is_clean());
        assert_eq!(store.current_epoch(), EpochId::new(2));
        assert_eq!(
            name_and_city(&store, 3),
            (Some(Value::from("Alix")), Some(Value::from("Amsterdam")))
        );
        assert_eq!(name_and_city(&store, 19), (Some(Value::from("Gus")), None));
    }

    /// Replay starts at the checkpoint's log position: the groups before it
    /// are in the image already, and applying one again would fail (a create
    /// at an id in use), so only the later ones are applied.
    #[test]
    fn replay_after_a_checkpoint_applies_only_later_groups() {
        let dir = tempfile::tempdir().unwrap();
        let checkpoint;
        {
            let wal = wal(dir.path(), 0);
            commit(&wal, 2, &set_of(None, vec![create(3, "Alix")]), 1);
            checkpoint = wal.end_lsn();
            commit(&wal, 3, &set_of(None, vec![set_city(3, "Berlin")]), 2);
        }
        // The image of the checkpoint: Alix, without a city.
        let store = Arc::new(RowGroupStore::new());
        replay_into(dir.path(), 0, &store, &Catalog::new()).unwrap();
        let image = Arc::new(RowGroupStore::new());
        {
            let first_only = tempfile::tempdir().unwrap();
            let wal = wal(first_only.path(), 0);
            commit(&wal, 2, &set_of(None, vec![create(3, "Alix")]), 1);
            drop(wal);
            replay_into(first_only.path(), 0, &image, &Catalog::new()).unwrap();
        }

        let replayed = replay_into(dir.path(), checkpoint, &image, &Catalog::new()).unwrap();
        assert_eq!(replayed.groups, 1, "only the group after the checkpoint");
        assert_eq!(
            name_and_city(&image, 3),
            (Some(Value::from("Alix")), Some(Value::from("Berlin")))
        );
        assert_eq!(name_and_city(&image, 3), name_and_city(&store, 3));
    }

    /// A group that does not fit the image fails the open, naming the WAL and
    /// the group's log position: replay is strict, and never skips a record.
    #[test]
    fn a_group_contradicting_the_image_fails_the_open() {
        let dir = tempfile::tempdir().unwrap();
        {
            let wal = wal(dir.path(), 0);
            // A property of a node no group created.
            commit(&wal, 2, &set_of(None, vec![set_city(88, "Paris")]), 1);
        }
        let store = Arc::new(RowGroupStore::new());
        let error = replay_into(dir.path(), 0, &store, &Catalog::new()).unwrap_err();
        let Error::Storage(StorageError::RecoveryFailed(message)) = &error else {
            panic!("not a recovery failure: {error}");
        };
        assert!(
            message.contains("log position 0")
                && message.contains(&dir.path().display().to_string()),
            "{message}"
        );
    }

    /// A group cut off by a crash (no LAST frame) was never committed: it is
    /// not applied, and the scan reports it as a torn tail to cut.
    #[test]
    fn a_torn_group_is_not_applied() {
        let dir = tempfile::tempdir().unwrap();
        let committed_end;
        {
            let wal = wal(dir.path(), 0);
            commit(&wal, 2, &set_of(None, vec![create(3, "Alix")]), 1);
            committed_end = wal.end_lsn();
            commit(&wal, 3, &set_of(None, vec![set_city(3, "Prague")]), 2);
        }
        // Cut the last bytes of the second group, as a crash in its LAST
        // frame's write would.
        let segment = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|path| path.extension().is_some_and(|extension| extension == "log"))
            .unwrap();
        let length = std::fs::metadata(&segment).unwrap().len();
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&segment)
            .unwrap();
        file.set_len(length - 3).unwrap();
        drop(file);

        let store = Arc::new(RowGroupStore::new());
        let replayed = replay_into(dir.path(), 0, &store, &Catalog::new()).unwrap();
        assert_eq!(replayed.groups, 1);
        assert_eq!(replayed.end.end_lsn, committed_end);
        assert!(matches!(
            replayed.end.tail.as_ref().map(|tail| tail.kind),
            Some(TailKind::Torn)
        ));
        assert_eq!(name_and_city(&store, 3), (Some(Value::from("Alix")), None));
    }

    /// A standalone group (a graph created) applies as the statement did, so
    /// a later commit's writes in that graph replay into it.
    #[test]
    fn a_standalone_group_and_writes_in_its_graph_replay() {
        let dir = tempfile::tempdir().unwrap();
        {
            let wal = wal(dir.path(), 0);
            let mut change = StandaloneChange::new();
            change.push(StandaloneOp::CreateGraph {
                name: "trips".to_string(),
            });
            let mut group = wal.begin_group(TransactionId::SYSTEM).unwrap();
            write_standalone(&mut group, &change, EpochId::new(0), Origin::Schema).unwrap();
            group.finish().unwrap();
            commit(
                &wal,
                2,
                &set_of(Some("trips"), vec![create(3, "Vincent")]),
                1,
            );
        }
        let store = Arc::new(RowGroupStore::new());
        let replayed = replay_into(dir.path(), 0, &store, &Catalog::new()).unwrap();
        assert_eq!(replayed.groups, 2);
        let trips = store.graph("trips").expect("the graph was created");
        assert_eq!(
            name_and_city(&trips, 3),
            (Some(Value::from("Vincent")), None)
        );
        assert_eq!(
            name_and_city(&store, 3),
            (None, None),
            "not in the default graph"
        );
    }

    /// A triple of a named RDF graph replays into that graph.
    #[cfg(feature = "triple-store")]
    #[test]
    fn a_triple_replays_into_its_graph() {
        use grafeo_common::storage::log_record::{TermRecord, TripleRecord};
        use grafeo_core::graph::rdf::{RdfStore, Triple};

        let dir = tempfile::tempdir().unwrap();
        let record = TripleRecord {
            subject: TermRecord::Iri("http://example.org/mia".to_string()),
            predicate: TermRecord::Iri("http://example.org/livesIn".to_string()),
            object: TermRecord::Iri("http://example.org/barcelona".to_string()),
        };
        {
            let wal = wal(dir.path(), 0);
            let mut set = ChangeSet::new();
            let slot = set
                .slot(GraphRef {
                    model: DataModel::Rdf,
                    key: Some(ArcStr::from("http://example.org/cities")),
                })
                .unwrap();
            set.push(
                slot,
                DataOp::InsertTriple {
                    triple: Box::new(record.clone()),
                },
                Before::Absent,
                PendingVersion::Created,
            )
            .unwrap();
            commit(&wal, 2, &set, 1);
        }
        let store = Arc::new(RowGroupStore::new());
        let rdf = RdfStore::new();
        let mut unbuilt = Vec::new();
        replay(
            dir.path(),
            ScanOptions::new(DATABASE, 0),
            &mut Targets {
                store: &store,
                rdf: &rdf,
                catalog: &Catalog::new(),
                unbuilt: &mut unbuilt,
                history: None,
            },
        )
        .unwrap();
        let cities = rdf
            .graph("http://example.org/cities")
            .expect("the graph exists");
        assert!(cities.contains(&Triple::from(&record)));
        assert_eq!(rdf.len(), 0, "not in the default graph");
    }
}
