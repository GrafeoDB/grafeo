//! Incremental checkpoints on the row-group store: what a checkpoint
//! reuses of the image it follows, what it writes again, and that the image
//! it writes reads back as the store's committed state at its epoch.

use std::sync::Arc;

use bytes::Bytes;
use grafeo_common::change::DataOp;
use grafeo_common::storage::{
    ChunkCaps, ChunkKind, ChunkMeta, ChunkNamespace, ImageId, SectionSink, SectionSource,
};
use grafeo_common::types::{ArcStr, EdgeId, EpochId, NodeId, PropertyKey, Value};
use grafeo_common::utils::error::Result;

use super::CheckpointSnapshot;
use crate::graph::apply::{GraphBackend, Writer};
use crate::graph::conformance::{
    Reader, Rng, Tx, committed, image, labels, next_epoch, people, properties, random_write,
};
use crate::graph::rowgroup::RowGroupStore;
use crate::graph::rowgroup::section::{LPG_SECTION_VERSION, write_lpg_snapshot};

/// An image's LPG section in memory.
#[derive(Default, Clone)]
struct Written {
    metas: Vec<ChunkMeta>,
    bytes: Vec<Bytes>,
}

impl SectionSource for Written {
    fn chunks(&self) -> &[ChunkMeta] {
        &self.metas
    }

    fn fetch(&self, index: usize) -> Result<Bytes> {
        Ok(self.bytes[index].clone())
    }

    fn stored_length(&self, index: usize) -> Result<u64> {
        Ok(self.bytes[index].len() as u64)
    }

    fn section_version(&self) -> u8 {
        LPG_SECTION_VERSION
    }
}

/// A sink that writes a new image following `previous`, as the file's
/// checkpoint writer does: a reused chunk is the previous image's, as it is.
struct Following<'p> {
    previous: Option<(ImageId, &'p Written)>,
    image: Written,
    /// The chunks written, the metadata chunk included.
    written: Vec<ChunkMeta>,
    reused: usize,
}

impl SectionSink for Following<'_> {
    fn write_chunk(&mut self, meta: ChunkMeta, bytes: &[u8]) -> Result<()> {
        self.image.metas.push(meta);
        self.image.bytes.push(Bytes::copy_from_slice(bytes));
        self.written.push(meta);
        Ok(())
    }

    fn previous(&self) -> Option<(ImageId, &dyn SectionSource)> {
        self.previous
            .map(|(image, previous)| (image, previous as &dyn SectionSource))
    }

    fn reuse_chunk(&mut self, index: usize) -> Result<()> {
        let (_, previous) = self.previous.expect("a sink that follows an image");
        self.image.metas.push(previous.metas[index]);
        self.image.bytes.push(previous.bytes[index].clone());
        self.reused += 1;
        Ok(())
    }
}

fn image_id(iteration: u64) -> ImageId {
    ImageId {
        database_id: 88,
        iteration,
    }
}

/// Checkpoints `root` at its epoch into a new image following `previous`,
/// which becomes image `iteration`.
fn checkpoint<'p>(
    root: &Arc<RowGroupStore>,
    previous: Option<(ImageId, &'p Written)>,
    iteration: u64,
) -> Following<'p> {
    checkpoint_with(root, previous, iteration, ChunkCaps::DEFAULT, None)
}

/// [`checkpoint`] with chunks of `caps`, and the history from
/// `history_from`.
fn checkpoint_with<'p>(
    root: &Arc<RowGroupStore>,
    previous: Option<(ImageId, &'p Written)>,
    iteration: u64,
    caps: ChunkCaps,
    history_from: Option<EpochId>,
) -> Following<'p> {
    let snapshot = CheckpointSnapshot::take(root, root.current_epoch(), history_from);
    let mut sink = Following {
        previous,
        image: Written::default(),
        written: Vec::new(),
        reused: 0,
    };
    write_lpg_snapshot(&snapshot, caps, &mut sink).unwrap();
    snapshot.written_as(image_id(iteration));
    sink
}

/// A root loaded from `written`.
fn loaded(written: &Written) -> RowGroupStore {
    let store = RowGroupStore::new();
    store.read_section(written).unwrap();
    store
}

/// The data chunks written (the metadata chunk left out).
fn data(sink: &Following<'_>) -> Vec<ChunkMeta> {
    sink.written
        .iter()
        .filter(|meta| meta.kind != ChunkKind::Meta)
        .copied()
        .collect()
}

/// Writes at once at the next epoch.
fn immediately(store: &RowGroupStore, op: &DataOp) {
    let epoch = EpochId::new(store.current_epoch().as_u64() + 1);
    store
        .apply(
            op,
            Writer::Immediate {
                epoch,
                before_images: false,
            },
        )
        .unwrap();
}

/// The people, and Vincent and Jules in the second row group of the node
/// table (from id 65,536), Vincent knowing Alix.
fn people_in_two_groups() -> (Arc<RowGroupStore>, NodeId, NodeId) {
    let root = Arc::new(RowGroupStore::new());
    let people = people(&*root).unwrap();
    let vincent = NodeId::new(65_536);
    let jules = NodeId::new(65_537);
    for (id, name) in [(vincent, "Vincent"), (jules, "Jules")] {
        immediately(
            &root,
            &DataOp::CreateNode {
                id,
                labels: labels(&["Person"]),
                properties: properties(&[("name", Value::from(name)), ("age", Value::Int64(40))]),
            },
        );
    }
    let edge = root.reserve_edge_ids(1).unwrap().start;
    immediately(
        &root,
        &DataOp::CreateEdge {
            id: EdgeId::new(edge),
            src: vincent,
            dst: people.alix,
            edge_type: ArcStr::from("KNOWS"),
            properties: properties(&[]),
        },
    );
    (root, vincent, jules)
}

/// A checkpoint of a store none of whose rows changed since the image it
/// follows writes its metadata chunk alone, and the image reads back.
#[test]
fn an_unchanged_store_is_its_previous_image() {
    let (root, _, _) = people_in_two_groups();
    let first = checkpoint(&root, None, 1);
    assert_eq!(first.reused, 0, "the first image follows none");
    let second = checkpoint(&root, Some((image_id(1), &first.image)), 2);
    assert_eq!(
        data(&second),
        Vec::<ChunkMeta>::new(),
        "no data chunk written"
    );
    assert_eq!(second.reused, first.image.metas.len() - 1);
    assert_eq!(committed(&loaded(&second.image)), committed(&*root));
}

/// One value set in the second node row group writes that column's chunk
/// of that group alone; the first group, the edges and the other columns
/// are reused.
#[test]
fn a_value_set_writes_its_column_of_its_row_group_alone() {
    let (root, vincent, _) = people_in_two_groups();
    let first = checkpoint(&root, None, 1);
    immediately(
        &root,
        &DataOp::SetNodeProperty {
            id: vincent,
            key: PropertyKey::new("age"),
            value: Value::Int64(41),
        },
    );
    let second = checkpoint(&root, Some((image_id(1), &first.image)), 2);
    let written = data(&second);
    assert_eq!(written.len(), 1, "one chunk: {written:?}");
    assert_eq!(
        (written[0].namespace, written[0].row_start),
        (ChunkNamespace::NodeProperties, 65_536),
        "the age column of the second group"
    );
    assert_eq!(committed(&loaded(&second.image)), committed(&*root));
}

/// A label added in the second group writes that label's chunk; a node
/// created there writes the group whole; the first group is reused each
/// time.
#[test]
fn labels_and_creates_write_their_row_group_only() {
    let (root, vincent, _) = people_in_two_groups();
    let first = checkpoint(&root, None, 1);
    immediately(
        &root,
        &DataOp::AddNodeLabel {
            id: vincent,
            label: ArcStr::from("Boxer"),
        },
    );
    let second = checkpoint(&root, Some((image_id(1), &first.image)), 2);
    let written = data(&second);
    assert_eq!(
        written
            .iter()
            .map(|meta| (meta.namespace, meta.row_start))
            .collect::<Vec<_>>(),
        [(ChunkNamespace::NodeLabels, 65_536)],
        "the new label's chunk alone"
    );
    assert_eq!(committed(&loaded(&second.image)), committed(&*root));

    immediately(
        &root,
        &DataOp::CreateNode {
            id: NodeId::new(65_538),
            labels: labels(&["Person"]),
            properties: properties(&[("name", Value::from("Mia"))]),
        },
    );
    let third = checkpoint(&root, Some((image_id(2), &second.image)), 3);
    let written = data(&third);
    assert!(
        written
            .iter()
            .all(|meta| meta.row_start >= 65_536 && meta.namespace != ChunkNamespace::EdgeStructure),
        "only the second node group: {written:?}"
    );
    assert!(
        written
            .iter()
            .any(|meta| meta.namespace == ChunkNamespace::NodeStructure),
        "its rows changed, so it is written whole"
    );
    assert_eq!(committed(&loaded(&third.image)), committed(&*root));
}

/// An edge created between the groups writes the edge group, Gus's
/// outgoing adjacency in the first node group and Jules's incoming one in
/// the second, and none of the nodes' columns.
#[test]
fn an_edge_writes_its_group_and_both_ends_adjacency() {
    let (root, _, jules) = people_in_two_groups();
    let first = checkpoint(&root, None, 1);
    let gus = NodeId::new(1);
    let edge = root.reserve_edge_ids(1).unwrap().start;
    immediately(
        &root,
        &DataOp::CreateEdge {
            id: EdgeId::new(edge),
            src: gus,
            dst: jules,
            edge_type: ArcStr::from("KNOWS"),
            properties: properties(&[("since", Value::Int64(1994))]),
        },
    );
    let second = checkpoint(&root, Some((image_id(1), &first.image)), 2);
    let mut kinds: Vec<(ChunkNamespace, u64)> = data(&second)
        .iter()
        .map(|meta| (meta.namespace, meta.row_start))
        .collect();
    kinds.sort_by_key(|(namespace, row)| (namespace.to_byte(), *row));
    assert!(
        !kinds.iter().any(|(namespace, _)| matches!(
            namespace,
            ChunkNamespace::NodeStructure | ChunkNamespace::NodeProperties
        )),
        "no node column is written: {kinds:?}"
    );
    assert!(kinds.contains(&(ChunkNamespace::OutgoingAdjacency, 0)));
    assert!(kinds.contains(&(ChunkNamespace::IncomingAdjacency, 65_537)));
    assert!(
        kinds
            .iter()
            .any(|(namespace, _)| *namespace == ChunkNamespace::EdgeStructure)
    );
    assert_eq!(committed(&loaded(&second.image)), committed(&*root));
}

/// A checkpoint writes the state at its epoch: a commit after the snapshot
/// is not in the image, and the next checkpoint writes what it changed.
#[test]
fn a_commit_after_the_snapshot_waits_for_the_next_checkpoint() {
    let (root, vincent, _) = people_in_two_groups();
    let first = checkpoint(&root, None, 1);
    let at = root.current_epoch();
    let snapshot = CheckpointSnapshot::take(&root, at, None);
    let mut tx = Tx::begin(&*root, 30);
    tx.set_node(vincent, "age", Value::Int64(52)).unwrap();
    tx.delete_node(NodeId::new(65_537)).unwrap();
    tx.commit().unwrap();
    let mut sink = Following {
        previous: Some((image_id(1), &first.image)),
        image: Written::default(),
        written: Vec::new(),
        reused: 0,
    };
    write_lpg_snapshot(&snapshot, ChunkCaps::DEFAULT, &mut sink).unwrap();
    snapshot.written_as(image_id(2));
    assert_eq!(data(&sink), Vec::<ChunkMeta>::new(), "nothing changed by E");
    assert_eq!(
        committed(&loaded(&sink.image)),
        image(&*root, Reader::At(at)),
        "the image holds the state at its epoch"
    );
    let third = checkpoint(&root, Some((image_id(2), &sink.image)), 3);
    assert!(!data(&third).is_empty(), "the commit is written now");
    assert_eq!(committed(&loaded(&third.image)), committed(&*root));
}

/// A transaction's writes mark at its commit: one applied before the
/// snapshot and committed after it is written by the next checkpoint.
#[test]
fn a_transaction_committed_after_the_snapshot_is_marked_for_the_next() {
    let (root, vincent, _) = people_in_two_groups();
    let first = checkpoint(&root, None, 1);
    let mut tx = Tx::begin(&*root, 31);
    tx.set_node(vincent, "age", Value::Int64(60)).unwrap();
    let second = checkpoint(&root, Some((image_id(1), &first.image)), 2);
    assert_eq!(data(&second), Vec::<ChunkMeta>::new(), "nothing committed");
    tx.commit().unwrap();
    let third = checkpoint(&root, Some((image_id(2), &second.image)), 3);
    assert_eq!(data(&third).len(), 1, "the age chunk of the second group");
    assert_eq!(committed(&loaded(&third.image)), committed(&*root));
}

/// A rolled-back write marks nothing, and its image is the previous one.
#[test]
fn a_rollback_writes_nothing() {
    let (root, vincent, _) = people_in_two_groups();
    let first = checkpoint(&root, None, 1);
    let mut tx = Tx::begin(&*root, 32);
    tx.set_node(vincent, "age", Value::Int64(60)).unwrap();
    tx.roll_back().unwrap();
    let second = checkpoint(&root, Some((image_id(1), &first.image)), 2);
    assert_eq!(data(&second), Vec::<ChunkMeta>::new());
    assert_eq!(committed(&loaded(&second.image)), committed(&*root));
}

/// Chunks are reused only from the image the store's chunks are in: a sink
/// following another image, or a store whose checkpoint failed, writes
/// every chunk.
#[test]
fn another_image_or_a_failed_checkpoint_writes_everything() {
    let (root, _, _) = people_in_two_groups();
    let first = checkpoint(&root, None, 1);
    let every = first.image.metas.len();
    let other = checkpoint(&root, Some((image_id(9), &first.image)), 2);
    assert_eq!((other.reused, other.written.len()), (0, every));

    let snapshot = CheckpointSnapshot::take(&root, root.current_epoch(), None);
    snapshot.failed();
    let after_failure = checkpoint(&root, Some((image_id(2), &other.image)), 3);
    assert_eq!(
        (after_failure.reused, after_failure.written.len()),
        (0, every)
    );
    assert_eq!(committed(&loaded(&after_failure.image)), committed(&*root));
}

/// A store loaded from an image and given it as its baseline writes only
/// what changed since, and a value removed and a node deleted read back.
#[test]
fn a_loaded_store_follows_the_image_it_was_read_from() {
    let (root, vincent, jules) = people_in_two_groups();
    let first = checkpoint(&root, None, 1);
    let reloaded = Arc::new(loaded(&first.image));
    reloaded.set_checkpoint_baseline(image_id(1));
    immediately(
        &reloaded,
        &DataOp::RemoveNodeProperty {
            id: vincent,
            key: PropertyKey::new("age"),
        },
    );
    let second = checkpoint(&reloaded, Some((image_id(1), &first.image)), 2);
    assert_eq!(data(&second).len(), 1);
    assert_eq!(committed(&loaded(&second.image)), committed(&*reloaded));

    immediately(&reloaded, &DataOp::DeleteNode { id: jules });
    let third = checkpoint(&reloaded, Some((image_id(2), &second.image)), 3);
    assert!(
        data(&third).iter().all(|meta| meta.row_start >= 65_536),
        "the second group alone"
    );
    assert_eq!(committed(&loaded(&third.image)), committed(&*reloaded));
}

/// A seeded workload, checkpointed after every transaction into an image
/// that follows the one before, reads back as the committed state at
/// every cap, and the checkpoints reuse chunks and write columns alone.
#[test]
fn seeded_writes_between_incremental_checkpoints_read_back_at_every_cap() {
    let (mut reused, mut partial) = (0, 0);
    for caps in [
        ChunkCaps::DEFAULT,
        ChunkCaps {
            max_rows: 3,
            max_bytes: 4_096,
        },
        ChunkCaps {
            max_rows: 7,
            max_bytes: 160,
        },
    ] {
        for seed in 1..=10_u64 {
            let root = Arc::new(RowGroupStore::new());
            people(&*root).unwrap();
            let mut rng = Rng::new(seed);
            let mut current = checkpoint_with(&root, None, 1, caps, None).image;
            for transaction in 0..14_u64 {
                let mut tx = Tx::begin(&*root, 100 + transaction);
                for _ in 0..=rng.below(5) {
                    random_write(&mut tx, &mut rng).unwrap();
                }
                if rng.below(4) == 0 {
                    tx.roll_back().unwrap();
                } else {
                    tx.commit().unwrap();
                }
                let iteration = transaction + 1;
                let next = checkpoint_with(
                    &root,
                    Some((image_id(iteration), &current)),
                    iteration + 1,
                    caps,
                    None,
                );
                let what = format!("seed {seed}, caps {caps:?}, transaction {transaction}");
                assert_eq!(committed(&loaded(&next.image)), committed(&*root), "{what}");
                reused += usize::from(next.reused > 0);
                partial += usize::from(
                    next.reused > 0
                        && next.written.iter().any(|meta| {
                            meta.namespace == ChunkNamespace::NodeProperties
                                && !next.written.iter().any(|other| {
                                    other.namespace == ChunkNamespace::NodeStructure
                                        && other.row_start == meta.row_start
                                })
                        }),
                );
                current = next.image;
            }
        }
    }
    assert!(reused >= 200, "{reused} checkpoints reused chunks");
    assert!(partial >= 20, "{partial} checkpoints wrote a column alone");
}

/// With history kept, incremental checkpoints of a seeded workload read
/// back alike at every epoch of their window, also where a row group was
/// reused from an image written with an earlier window.
#[test]
fn seeded_history_checkpoints_read_alike_at_every_epoch_of_the_window() {
    for seed in 1..=8_u64 {
        let root = Arc::new(RowGroupStore::new());
        root.set_keeps_history(true);
        people(&*root).unwrap();
        let from = root.current_epoch();
        let mut rng = Rng::new(seed);
        let caps = if seed % 2 == 0 {
            ChunkCaps {
                max_rows: 4,
                max_bytes: 256,
            }
        } else {
            ChunkCaps::DEFAULT
        };
        let mut current = checkpoint_with(&root, None, 1, caps, Some(from)).image;
        for transaction in 0..16_u64 {
            let mut tx = Tx::begin(&*root, 100 + transaction);
            for _ in 0..=rng.below(3) {
                random_write(&mut tx, &mut rng).unwrap();
            }
            tx.commit().unwrap();
            if rng.below(3) == 0 {
                let seen = committed(&*root);
                if let Some(id) = rng.pick(&seen.nodes.keys().copied().collect::<Vec<_>>()) {
                    root.apply(
                        &DataOp::SetNodeProperty {
                            id: NodeId::new(*id),
                            key: PropertyKey::new("city"),
                            value: Value::from(["Amsterdam", "Berlin", "Paris"][rng.below(3)]),
                        },
                        Writer::Immediate {
                            epoch: next_epoch(&*root),
                            before_images: false,
                        },
                    )
                    .unwrap();
                }
            }
            let iteration = transaction + 1;
            current = checkpoint_with(
                &root,
                Some((image_id(iteration), &current)),
                iteration + 1,
                caps,
                Some(from),
            )
            .image;
        }
        let loaded = loaded(&current);
        for at in from.as_u64()..=root.current_epoch().as_u64() {
            let reader = Reader::At(EpochId::new(at));
            assert_eq!(
                image(&loaded, reader),
                image(&*root, reader),
                "seed {seed}, epoch {at}"
            );
        }
    }
}

/// With caps whose row count does not divide the store's row groups, a file
/// row group can lie across two of them (also when the first holds no row):
/// it is written once, whole, and reads back, before and after a change.
#[test]
fn a_file_row_group_across_two_store_row_groups_is_written_once() {
    let caps = ChunkCaps {
        max_rows: 7,
        max_bytes: 4_096,
    };
    for ids in [65_530..65_545_u64, 65_536..65_540] {
        let root = Arc::new(RowGroupStore::new());
        for id in ids.clone() {
            immediately(
                &root,
                &DataOp::CreateNode {
                    id: NodeId::new(id),
                    labels: labels(&["Person"]),
                    properties: properties(&[("age", Value::Int64(i64::try_from(id).unwrap()))]),
                },
            );
        }
        let first = checkpoint_with(&root, None, 1, caps, None);
        assert_eq!(
            committed(&loaded(&first.image)),
            committed(&*root),
            "{ids:?}"
        );
        immediately(
            &root,
            &DataOp::SetNodeProperty {
                id: NodeId::new(ids.start + 1),
                key: PropertyKey::new("age"),
                value: Value::Int64(19),
            },
        );
        let second = checkpoint_with(&root, Some((image_id(1), &first.image)), 2, caps, None);
        assert_eq!(
            committed(&loaded(&second.image)),
            committed(&*root),
            "{ids:?}"
        );
        // The second set lies in one file row group, across the two store
        // row groups: nothing of it is reused.
        assert_eq!(
            second.reused > 0,
            ids.start == 65_530,
            "{ids:?}: the unchanged groups are reused"
        );
    }
}
