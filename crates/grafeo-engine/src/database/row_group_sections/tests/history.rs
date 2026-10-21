//! History and time travel on a row-group root, through the engine's
//! commits, checkpoints, open and WAL replay (H2c): every tier a version
//! can be in (the write tail in memory, a column a load left cold, the file
//! after a checkpoint and a reopen), the retention window and the floor, a
//! crash at every point of a checkpoint, and a database that keeps no
//! history, which writes and opens as before.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use grafeo_common::storage::SectionType;
use grafeo_common::types::{EpochId, NodeId, ReadView, Timestamp, Value};
use grafeo_core::graph::rowgroup::RowGroupStore;
use grafeo_core::graph::{Direction, GraphStore};
use grafeo_storage::file::GrafeoFileManager;
use grafeo_storage::wal::{DurabilityMode, ScanOptions, Wal, WalOptions};

use super::file::FileDatabase;
use super::{
    Database, EdgeView, NodeView, continue_row_group_epochs, load_row_group_sections, names,
    restore_row_group_history, travels, values,
};
use crate::config::HistoryRetention;

/// What a committed reader at `epoch` sees of a graph: every node and edge
/// the store holds in any version, as the reader sees it.
type State = (BTreeMap<u64, NodeView>, BTreeMap<u64, EdgeView>);

fn state_at(store: &RowGroupStore, epoch: EpochId) -> State {
    let view = ReadView::committed(epoch);
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
    for id in store.all_node_ids() {
        let Some(node) = store.get_node_at(id, view) else {
            continue;
        };
        let mut labels: Vec<String> = node.labels.iter().map(ToString::to_string).collect();
        labels.sort();
        nodes.insert(id.as_u64(), (labels, sorted(&node.properties)));
        for (_, edge) in store.edges_from_at(id, Direction::Outgoing, view) {
            if let Some(edge) = store.get_edge_at(edge, view) {
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

/// The place of `epoch` in what [`states`] returns.
fn index(epoch: EpochId) -> usize {
    usize::try_from(epoch.as_u64()).unwrap()
}

/// What every reader from epoch 0 to `last` sees of the root and of trips.
fn states(database: &Database, last: EpochId) -> Vec<[State; 2]> {
    let trips = database.store(Some("trips"));
    (0..=last.as_u64())
        .map(EpochId::new)
        .map(|epoch| [state_at(&database.root, epoch), state_at(&trips, epoch)])
        .collect()
}

/// A database that keeps history, as an open with `retention` makes it.
fn keeping(database: &Database, retention: HistoryRetention) {
    database
        .transaction_manager
        .set_history_retention(retention);
    database
        .root
        .set_keeps_history(retention != HistoryRetention::Off);
}

/// After [`travels`]: Alix moves to Berlin and Gus stops knowing her, then
/// Gus is no Employee any more and Vincent's Paris is gone from trips, each
/// commit with the collect a database runs after it. Returns the epochs.
fn moves(database: &Database, alix: NodeId, gus: NodeId, paris: NodeId) -> [EpochId; 2] {
    let collect = |database: &Database| {
        database
            .root
            .gc_versions_after_write(database.transaction_manager.collect_horizon());
    };
    let moved = database.write(None, |writer| {
        writer
            .set_node_properties(
                alix,
                &values(&[("bio", Value::from("Alix lives in Berlin"))]),
                false,
            )
            .unwrap();
        let knows = writer
            .store()
            .edges_from(gus, Direction::Outgoing)
            .first()
            .map(|(_, edge)| *edge)
            .unwrap();
        writer.delete_edge(knows).unwrap();
    });
    collect(database);
    let gone = database.write(Some("trips"), |writer| {
        writer.delete_node(paris, true).unwrap();
    });
    let promoted = database.write(None, |writer| {
        writer.remove_labels(gus, &names(&["Employee"])).unwrap();
    });
    collect(database);
    assert!(gone < promoted);
    [moved, promoted]
}

/// Opens the database file at `path` as an open does, keeping history as
/// `retention` says: its image, the history it holds, the WAL groups after
/// its checkpoint replayed onto the root, then the epochs continued.
fn reopen(path: &Path, retention: HistoryRetention) -> FileDatabase {
    let fm = GrafeoFileManager::open(path, None).unwrap();
    let database = Database::new();
    database.transaction_manager.open_history(retention);
    let history = fm
        .read_image(|image| {
            load_row_group_sections(
                image,
                Some(path),
                &database.root,
                &database.catalog,
                #[cfg(feature = "triple-store")]
                &database.rdf_store,
            )
        })
        .unwrap();
    database
        .root
        .sync_epoch(EpochId::new(fm.active_header().epoch));
    restore_row_group_history(&database.root, &database.transaction_manager, history);
    if fm.has_sidecar_wal() {
        super::super::super::replay::replay(
            &fm.sidecar_wal_path(),
            ScanOptions::new(fm.database_id(), fm.active_header().checkpoint_lsn),
            &mut super::super::super::replay::Targets {
                store: &database.root,
                #[cfg(feature = "triple-store")]
                rdf: &database.rdf_store,
                catalog: &database.catalog,
                unbuilt: &mut Vec::new(),
                history: Some(&database.transaction_manager),
            },
        )
        .unwrap();
    }
    continue_row_group_epochs(&database.root, &database.transaction_manager);
    FileDatabase { database, fm }
}

/// A new database file at `path` whose commits write a WAL, keeping
/// history as `retention` says.
fn created_with_wal(path: &Path, retention: HistoryRetention) -> FileDatabase {
    let mut created = FileDatabase::create(path);
    let wal_dir = created.fm.sidecar_wal_path();
    std::fs::create_dir_all(&wal_dir).unwrap();
    created.database.wal = Some(Arc::new(
        Wal::open(
            &wal_dir,
            WalOptions {
                durability: DurabilityMode::Sync,
                ..WalOptions::new(created.fm.database_id(), 0)
            },
        )
        .unwrap(),
    ));
    keeping(&created.database, retention);
    created
}

/// Every tier: the readers at each epoch read the same in the write tail
/// (in memory, nothing collected), after a checkpoint and a reopen (the
/// file's `History` chunks; columns a load left cold until a write thaws
/// them), and after writes and a second reopen; a read at a time finds the
/// same epoch before and after; history goes on across the reopen.
#[test]
fn every_tier_reads_what_each_epoch_committed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("travels.grafeo");
    let created = created_with_wal(&path, HistoryRetention::Forever);
    let travellers = travels(&created.database);
    let [moved, promoted] = moves(
        &created.database,
        travellers.alix,
        travellers.gus,
        travellers.paris,
    );
    let tail = states(&created.database, promoted);
    assert_ne!(tail[index(moved) - 1], tail[index(moved)]);
    let bio = |state: &[State; 2]| -> Option<Value> {
        state[0]
            .0
            .get(&travellers.alix.as_u64())
            .and_then(|(_, values)| {
                values
                    .iter()
                    .find(|(key, _)| key == "bio")
                    .map(|(_, value)| value.clone())
            })
    };
    assert_eq!(
        bio(&tail[index(moved) - 1]),
        Some(Value::from("Alix lives in Amsterdam")),
        "the tail keeps the bio before the move"
    );
    let times = created
        .database
        .transaction_manager
        .history_image()
        .unwrap();
    created.checkpoint().unwrap();
    assert!(
        created
            .fm
            .read_image(|image| Ok(image.section_source(SectionType::TimeIndex).is_some()))
            .unwrap(),
        "the checkpoint holds the time index"
    );
    drop(created);

    let opened = reopen(&path, HistoryRetention::Forever);
    assert_eq!(states(&opened.database, promoted), tail, "after the reopen");
    assert_eq!(
        opened
            .database
            .transaction_manager
            .history_image()
            .unwrap()
            .times,
        times.times,
        "the commit times"
    );
    assert_eq!(
        opened.database.transaction_manager.history_floor(),
        EpochId::new(0)
    );
    for (epoch, millis) in &times.times {
        let at = Timestamp::from_millis(i64::try_from(*millis).unwrap());
        let found = opened.database.transaction_manager.epoch_at(at).unwrap();
        assert!(
            found.as_u64() >= *epoch,
            "a time finds its commit or a later one at the same millisecond"
        );
    }
    // A write thaws the cold column Alix's bio is in; the tail of the
    // reopened database keeps the loaded bio for the readers before it.
    let rewritten = opened.database.write(None, |writer| {
        writer
            .set_node_properties(
                travellers.alix,
                &values(&[("bio", Value::from("Alix lives in Paris"))]),
                false,
            )
            .unwrap();
    });
    let after = states(&opened.database, rewritten);
    assert_eq!(after[..tail.len()], tail[..], "every earlier reader");
    assert_eq!(
        bio(&after[index(rewritten)]),
        Some(Value::from("Alix lives in Paris"))
    );
    opened.close();
    let again = reopen(&path, HistoryRetention::Forever);
    assert_eq!(
        states(&again.database, rewritten),
        after,
        "the second reopen"
    );
}

/// The retention window: a reader inside it reads its epoch's state, the
/// collect horizon is the window's start (or an older reader's epoch), a
/// reader at exactly the horizon still reads what it read, and one before
/// it is refused with "not retained", never answered.
#[test]
fn the_window_keeps_its_readers_and_refuses_the_ones_before_it() {
    let database = Database::new();
    keeping(&database, HistoryRetention::Forever);
    let travellers = travels(&database);
    let [moved, promoted] = moves(&database, travellers.alix, travellers.gus, travellers.paris);
    let all = states(&database, promoted);
    // Forever: nothing goes, every epoch from 0 is read.
    assert_eq!(
        database.transaction_manager.history_floor(),
        EpochId::new(0)
    );
    database
        .root
        .gc_versions(database.transaction_manager.collect_horizon());
    assert_eq!(states(&database, promoted), all);

    // The window shrinks to the newest commit: the horizon moves to the
    // published epoch, which the readers at it and after keep.
    database
        .transaction_manager
        .set_history_retention(HistoryRetention::Duration(std::time::Duration::ZERO));
    let horizon = database.transaction_manager.collect_horizon();
    assert_eq!(horizon, promoted, "the window starts at the last commit");
    database.root.gc_versions(horizon);
    assert_eq!(states(&database, promoted).last(), all.last());
    assert_eq!(database.transaction_manager.history_floor(), promoted);
    assert!(
        database.transaction_manager.pin_read_at(promoted).is_ok(),
        "a read at the horizon"
    );
    let before = database
        .transaction_manager
        .pin_read_at(moved)
        .unwrap_err()
        .to_string();
    assert!(
        before.contains(&format!(
            "history before epoch {} is not retained",
            promoted.as_u64()
        )),
        "{before}"
    );
    assert_eq!(
        database.root.history_floor(),
        promoted,
        "the store's floor follows its collect"
    );
}

/// A database that keeps no history writes no `TIME_INDEX` section and no
/// `History` chunk, collects at the read horizon, and opens as before:
/// readers before the reopen's epoch are refused.
#[test]
fn without_history_nothing_is_kept_or_written() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("travels.grafeo");
    let created = created_with_wal(&path, HistoryRetention::Off);
    let travellers = travels(&created.database);
    let [_, promoted] = moves(
        &created.database,
        travellers.alix,
        travellers.gus,
        travellers.paris,
    );
    assert_eq!(created.database.transaction_manager.history_image(), None);
    assert_eq!(
        created.database.transaction_manager.collect_horizon(),
        created.database.transaction_manager.read_horizon()
    );
    created.checkpoint().unwrap();
    let (time_index, history_chunks) = created
        .fm
        .read_image(|image| {
            let lpg = image.section_source(SectionType::LpgStore).unwrap();
            let history = lpg
                .chunks()
                .iter()
                .filter(|chunk| chunk.kind == grafeo_common::storage::ChunkKind::History)
                .count();
            Ok((
                image.section_source(SectionType::TimeIndex).is_some(),
                history,
            ))
        })
        .unwrap();
    assert_eq!((time_index, history_chunks), (false, 0));
    drop(created);
    let opened = reopen(&path, HistoryRetention::Off);
    assert_eq!(
        opened.database.transaction_manager.history_floor(),
        promoted
    );
    assert!(
        opened
            .database
            .transaction_manager
            .pin_read_at(EpochId::new(1))
            .is_err()
    );
}

/// A crash at every point of a checkpoint after commits the WAL logs: the
/// reopen (the image whose header reached the file, and the WAL groups
/// after its checkpoint replayed, each keeping what it replaced at its
/// epoch) reads at every epoch what the database read before the crash,
/// and its time index holds the times the commits recorded. The database
/// is built outside the crash and never closed after it.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn a_crash_at_every_checkpoint_point_keeps_the_history() {
    use std::panic::AssertUnwindSafe;

    use grafeo_common::testing::crash::{CrashResult, with_crash_at};

    /// The crash points of a checkpoint with a WAL, at least: one past the
    /// last stops the loop.
    const MOST_POINTS: u64 = 16;

    let mut completed = false;
    for point in 1..=MOST_POINTS {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("travels.grafeo");
        let database = created_with_wal(&path, HistoryRetention::Forever);
        let travellers = travels(&database.database);
        database.checkpoint().unwrap();
        // After the first checkpoint, in the WAL only.
        let [_, promoted] = moves(
            &database.database,
            travellers.alix,
            travellers.gus,
            travellers.paris,
        );
        let before = states(&database.database, promoted);
        let times = database
            .database
            .transaction_manager
            .history_image()
            .unwrap()
            .times;

        let crashed = matches!(
            with_crash_at(point, AssertUnwindSafe(|| database.checkpoint())),
            CrashResult::Crashed
        );
        // What a crash leaves: nothing closes the database.
        drop(database);

        let reopened = reopen(&path, HistoryRetention::Forever);
        assert_eq!(
            states(&reopened.database, promoted),
            before,
            "point {point}: every epoch's readers"
        );
        assert_eq!(
            reopened
                .database
                .transaction_manager
                .history_image()
                .unwrap()
                .times,
            times,
            "point {point}: the commit times"
        );
        if !crashed {
            completed = true;
            break;
        }
    }
    assert!(
        completed,
        "a checkpoint has fewer than {MOST_POINTS} crash points"
    );
}

// The closed timeline below is adapted from `temporal_persistence_fidelity.rs`
// by temporaryfix (github.com/temporaryfix/grafeo @ 427daf50, offered in
// discussion #606): persistence must keep the timeline, not only the
// current image, so it ends with a deleted edge and node.

/// The epochs of a closed timeline: a subject created, updated, relabelled,
/// its edge deleted, then the subject deleted.
#[derive(Debug, Clone, Copy)]
struct Timeline {
    subject: NodeId,
    peer: NodeId,
    edge: grafeo_common::types::EdgeId,
    before_create: EpochId,
    created: EpochId,
    updated: EpochId,
    label_removed: EpochId,
    edge_deleted: EpochId,
    node_deleted: EpochId,
}

/// Builds a closed timeline in the graph `graph` (`None`: the default).
fn closed_timeline(database: &Database, graph: Option<&str>) -> Timeline {
    let before_create = database.transaction_manager.current_epoch();
    let mut ids = (
        NodeId::new(0),
        NodeId::new(0),
        grafeo_common::types::EdgeId::new(0),
    );
    let created = database.write(graph, |writer| {
        let subject = writer
            .create_node(
                &names(&["Entity", "Transient"]),
                values(&[("state", Value::from("born"))]),
            )
            .unwrap();
        let peer = writer
            .create_node(
                &names(&["Anchor"]),
                values(&[("name", Value::from("persistent-peer"))]),
            )
            .unwrap();
        let edge = writer
            .create_edge(
                subject,
                peer,
                "LINKS",
                values(&[("weight", Value::Int64(1))]),
            )
            .unwrap();
        ids = (subject, peer, edge);
    });
    let (subject, peer, edge) = ids;
    let updated = database.write(graph, |writer| {
        writer
            .set_node_properties(
                subject,
                &values(&[("state", Value::from("updated"))]),
                false,
            )
            .unwrap();
        writer
            .set_edge_properties(edge, &values(&[("weight", Value::Int64(2))]), false)
            .unwrap();
        assert_eq!(writer.add_labels(subject, &names(&["Tracked"])).unwrap(), 1);
    });
    let label_removed = database.write(graph, |writer| {
        assert_eq!(
            writer
                .remove_labels(subject, &names(&["Transient"]))
                .unwrap(),
            1
        );
    });
    let edge_deleted = database.write(graph, |writer| {
        assert!(writer.delete_edge(edge).unwrap());
    });
    let node_deleted = database.write(graph, |writer| {
        assert!(writer.delete_node(subject, false).unwrap());
    });
    Timeline {
        subject,
        peer,
        edge,
        before_create,
        created,
        updated,
        label_removed,
        edge_deleted,
        node_deleted,
    }
}

/// Checks the timeline in `store` at each of its epochs.
fn assert_timeline(store: &RowGroupStore, timeline: Timeline, what: &str) {
    let node = |epoch: EpochId| store.get_node_at(timeline.subject, ReadView::committed(epoch));
    let state = |epoch: EpochId| {
        store.get_node_property_at_epoch(
            timeline.subject,
            &grafeo_common::types::PropertyKey::new("state"),
            epoch,
        )
    };
    let weight = |epoch: EpochId| {
        store
            .get_edge_at(timeline.edge, ReadView::committed(epoch))
            .and_then(|edge| edge.get_property("weight").cloned())
    };
    let labels = |epoch: EpochId| {
        let mut labels: Vec<String> = node(epoch)
            .unwrap_or_else(|| panic!("{what}: the subject at {epoch:?}"))
            .labels
            .iter()
            .map(ToString::to_string)
            .collect();
        labels.sort();
        labels
    };
    assert!(
        node(timeline.before_create).is_none(),
        "{what}: before the create"
    );
    assert!(weight(timeline.before_create).is_none(), "{what}");
    assert_eq!(state(timeline.created), Some(Value::from("born")), "{what}");
    assert_eq!(labels(timeline.created), ["Entity", "Transient"], "{what}");
    assert_eq!(weight(timeline.created), Some(Value::Int64(1)), "{what}");
    assert_eq!(
        state(timeline.updated),
        Some(Value::from("updated")),
        "{what}"
    );
    assert_eq!(
        labels(timeline.updated),
        ["Entity", "Tracked", "Transient"],
        "{what}"
    );
    assert_eq!(weight(timeline.updated), Some(Value::Int64(2)), "{what}");
    assert_eq!(
        labels(timeline.label_removed),
        ["Entity", "Tracked"],
        "{what}"
    );
    assert_eq!(
        weight(timeline.label_removed),
        Some(Value::Int64(2)),
        "{what}"
    );
    assert!(
        weight(timeline.edge_deleted).is_none(),
        "{what}: the edge deleted"
    );
    assert_eq!(
        state(timeline.edge_deleted),
        Some(Value::from("updated")),
        "{what}"
    );
    assert!(
        node(timeline.node_deleted).is_none(),
        "{what}: the subject deleted"
    );
    assert!(
        store
            .get_node_at(timeline.peer, ReadView::committed(timeline.node_deleted))
            .is_some(),
        "{what}: the peer stays"
    );
}

/// A closed timeline, in the default graph and in a named graph, reads the
/// same at each of its epochs in memory, after a checkpoint and a reopen,
/// and after a second one.
#[test]
fn a_closed_timeline_survives_persistence() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("timeline.grafeo");
    let created = created_with_wal(&path, HistoryRetention::Forever);
    assert!(created.database.root.create_graph("trips").unwrap());
    let default = closed_timeline(&created.database, None);
    let named = closed_timeline(&created.database, Some("trips"));
    assert_timeline(&created.database.root, default, "in memory");
    assert_timeline(
        &created.database.store(Some("trips")),
        named,
        "trips in memory",
    );
    created.close();

    let opened = reopen(&path, HistoryRetention::Forever);
    assert_timeline(&opened.database.root, default, "after the reopen");
    assert_timeline(
        &opened.database.store(Some("trips")),
        named,
        "trips after the reopen",
    );
    opened.close();
    let again = reopen(&path, HistoryRetention::Forever);
    assert_timeline(&again.database.root, default, "after the second reopen");
    assert_timeline(
        &again.database.store(Some("trips")),
        named,
        "trips, second reopen",
    );
}
