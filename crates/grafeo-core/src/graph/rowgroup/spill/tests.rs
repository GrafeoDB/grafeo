//! Column spill: what a spilled column reads, what thaws it, what stays in
//! memory, and when its files go.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use bytes::Bytes;
use grafeo_common::change::DataOp;
use grafeo_common::storage::{ChunkCaps, ChunkMeta, SectionSink, SectionSource};
use grafeo_common::types::{NodeId, PropertyKey, ReadView, Value};
use grafeo_common::utils::error::{Error, Result};

use super::{ColumnSpill, ColumnSpillReport};
use crate::graph::apply::{ChangeTarget, Writer};
use crate::graph::conformance::{
    Reader, Tx, apply_all, committed, image, labels, next_epoch, properties,
};
use crate::graph::rowgroup::section::{
    GraphToWrite, LPG_SECTION_VERSION, read_lpg_section, write_lpg_section,
};
use crate::graph::rowgroup::{ROWS_PER_GROUP, RowGroupStore};
use crate::graph::traits::GraphStore;

/// A spill file in memory: the bytes it was given, which count as one live
/// file until the last slice of them is dropped.
struct Mapped {
    bytes: Vec<u8>,
    live: Arc<AtomicUsize>,
}

impl AsRef<[u8]> for Mapped {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Spill files in memory, counted: the files written and those still live.
#[derive(Default)]
struct Files {
    written: AtomicUsize,
    live: Arc<AtomicUsize>,
    /// Fail every write from the next one on.
    fail: std::sync::atomic::AtomicBool,
}

impl Files {
    fn live(&self) -> usize {
        self.live.load(Ordering::SeqCst)
    }

    fn written(&self) -> usize {
        self.written.load(Ordering::SeqCst)
    }
}

impl ColumnSpill for Files {
    fn write(&self, bytes: &[u8]) -> std::io::Result<Bytes> {
        if self.fail.load(Ordering::SeqCst) {
            return Err(std::io::Error::other("the disk of Vincent is full"));
        }
        self.written.fetch_add(1, Ordering::SeqCst);
        self.live.fetch_add(1, Ordering::SeqCst);
        Ok(Bytes::from_owner(Mapped {
            bytes: bytes.to_vec(),
            live: Arc::clone(&self.live),
        }))
    }
}

fn embedding() -> PropertyKey {
    PropertyKey::new("embedding")
}

fn vector(components: &[f32]) -> Value {
    Value::Vector(components.into())
}

/// The embedding of node `id`: its row and its row group.
fn embedding_of(id: u64) -> Value {
    let part = |part: u64| f32::from(u16::try_from(part).expect("a test id below 2^32"));
    vector(&[part(id % ROWS_PER_GROUP), part(id / ROWS_PER_GROUP)])
}

/// Items in two row groups: ids `0..4_100` (three vectors of rows of the
/// first group), and two in the second; each with a name, a rank and an
/// embedding, every seventh without an embedding.
fn items(store: &RowGroupStore) -> Vec<NodeId> {
    let reserved = store.reserve_node_ids(ROWS_PER_GROUP + 4).unwrap();
    assert_eq!(reserved.start, 0);
    let ids: Vec<u64> = (0..4_100)
        .chain([ROWS_PER_GROUP + 1, ROWS_PER_GROUP + 3])
        .collect();
    let ops: Vec<DataOp> = ids
        .iter()
        .map(|&id| {
            let name = ["Alix", "Gus", "Vincent", "Jules", "Mia"][(id % 5) as usize];
            let mut values = vec![
                ("name", Value::from(name)),
                ("rank", Value::Int64(i64::try_from(id).unwrap())),
            ];
            if id % 7 != 3 {
                values.push(("embedding", embedding_of(id)));
            }
            DataOp::CreateNode {
                id: NodeId::new(id),
                labels: labels(&["Item"]),
                properties: properties(&values),
            }
        })
        .collect();
    apply_all(
        store,
        &ops,
        Writer::Replay {
            epoch: next_epoch(store),
        },
    )
    .unwrap();
    ids.into_iter().map(NodeId::new).collect()
}

/// Spills the embeddings into `files`.
fn spill(store: &RowGroupStore, files: &Files) -> ColumnSpillReport {
    store
        .spill_node_property_column(&embedding(), files)
        .unwrap()
}

/// Every item's embedding reads as written, and an item without one reads
/// none.
fn assert_embeddings(store: &RowGroupStore, ids: &[NodeId], what: &str) {
    for id in ids {
        let expected = (id.as_u64() % 7 != 3).then(|| embedding_of(id.as_u64()));
        assert_eq!(
            store.get_node_property(*id, &embedding()),
            expected,
            "{what}: node {id:?}"
        );
    }
}

/// A spilled column reads every row from its files and stays spilled: one
/// file per row group, its chunks off the heap, the store's figures moved
/// from the decoded columns to the spilled bytes.
#[test]
fn a_spilled_column_reads_every_row_from_its_files_and_stays_spilled() {
    let store = RowGroupStore::new();
    let ids = items(&store);
    let before = store.memory();
    let image_before = committed(&store);
    let files = Files::default();

    let report = spill(&store, &files);
    assert_eq!(report.spilled_groups, 2, "{report:?}");
    assert_eq!((report.kept_with_chains, report.kept_unsupported), (0, 0));
    assert_eq!(
        (files.written(), files.live()),
        (2, 2),
        "a file per row group"
    );
    assert_eq!(store.spilled_node_property_columns(), [embedding()]);

    let after = store.memory();
    assert_eq!(after.spilled_columns, 2);
    assert_eq!(after.node_spilled_bytes, report.file_bytes);
    assert!(
        after.total_bytes() < before.total_bytes(),
        "{} heap bytes before, {} after",
        before.total_bytes(),
        after.total_bytes()
    );
    assert!(
        report.freed_bytes > 4_000 * 8,
        "the vectors left the heap: {report:?}"
    );

    assert_embeddings(&store, &ids, "spilled");
    let image_after = committed(&store);
    assert_eq!(image_after.nodes, image_before.nodes);
    assert_eq!(
        store.memory(),
        after,
        "the reads left the column spilled and decoded nothing"
    );

    let again = spill(&store, &files);
    assert_eq!(again.spilled_groups, 0, "spilled already");
    assert_eq!(files.written(), 2);
}

/// A string column stays in memory: its chunks would have no point view,
/// so its first read would decode it back; and a key no node has spills
/// nothing.
#[test]
fn a_column_without_point_views_stays_in_memory() {
    let store = RowGroupStore::new();
    items(&store);
    let files = Files::default();
    let report = store
        .spill_node_property_column(&PropertyKey::new("name"), &files)
        .unwrap();
    assert_eq!(report.kept_unsupported, 2, "{report:?}");
    assert_eq!((report.spilled_groups, files.written()), (0, 0));
    let none = store
        .spill_node_property_column(&PropertyKey::new("Butch"), &files)
        .unwrap();
    assert_eq!(none, ColumnSpillReport::default());
    assert_eq!(
        store.spilled_node_property_columns(),
        Vec::<PropertyKey>::new()
    );

    let ranks = store
        .spill_node_property_column(&PropertyKey::new("rank"), &files)
        .unwrap();
    assert_eq!(ranks.spilled_groups, 2, "integers have point views");
    assert_eq!(
        store.get_node_property(NodeId::new(4_099), &PropertyKey::new("rank")),
        Some(Value::Int64(4_099))
    );
}

/// A write to a spilled column thaws its row group's column alone (Q9),
/// which lets go of its file; the other row group stays spilled. A reload
/// brings the rest back and lets go of every file.
#[test]
fn a_write_thaws_its_row_group_and_a_reload_the_rest() {
    let store = RowGroupStore::new();
    let ids = items(&store);
    let files = Files::default();
    spill(&store, &files);

    let jules = NodeId::new(ROWS_PER_GROUP + 1);
    let mut tx = Tx::begin(&store, 10);
    tx.set_node(jules, "embedding", vector(&[3.0, 19.0]))
        .unwrap();
    assert_eq!(files.live(), 1, "the second row group's file went");
    assert_eq!(store.memory().spilled_columns, 1);
    let written = tx.commit().unwrap();
    assert_eq!(
        store.get_node_property(jules, &embedding()),
        Some(vector(&[3.0, 19.0]))
    );
    store.collect(written);

    assert!(store.reload_node_property_column(&embedding()).unwrap());
    assert_eq!(files.live(), 0);
    assert_eq!(
        store.spilled_node_property_columns(),
        Vec::<PropertyKey>::new()
    );
    assert_eq!(store.memory().node_spilled_bytes, 0);
    assert!(!store.reload_node_property_column(&embedding()).unwrap());
    let others: Vec<NodeId> = ids.into_iter().filter(|id| *id != jules).collect();
    assert_embeddings(&store, &others, "reloaded");
}

/// A row group's column with update chains stays in memory, so a reader
/// before the write still reads the value the chain kept; the collect that
/// drops the chain lets the next spill take it.
#[test]
fn a_column_with_update_chains_stays_in_memory_until_collect() {
    let store = RowGroupStore::new();
    items(&store);
    let before = store.current_epoch();
    let mia = NodeId::new(4);
    let mut tx = Tx::begin(&store, 10);
    tx.set_node(mia, "embedding", vector(&[88.0, 0.5])).unwrap();
    let files = Files::default();
    let report = spill(&store, &files);
    assert_eq!(
        (report.spilled_groups, report.kept_with_chains),
        (1, 1),
        "{report:?}"
    );
    let own = || Reader::Transaction(tx.id, tx.snapshot);
    let at_before = || Reader::At(before);
    let read = |reader: Reader| {
        image(&store, reader).nodes[&mia.as_u64()]
            .1
            .iter()
            .find(|(key, _)| key == "embedding")
            .map(|(_, value)| value.clone())
    };
    assert_eq!(read(own()), Some(vector(&[88.0, 0.5])), "the writer's");
    assert_eq!(read(at_before()), Some(embedding_of(4)), "the kept one");

    let committed_at = tx.commit().unwrap();
    assert_eq!(
        spill(&store, &files).kept_with_chains,
        1,
        "the chain lives until collect"
    );
    assert_eq!(read(at_before()), Some(embedding_of(4)));
    store.collect(committed_at);
    let report = spill(&store, &files);
    assert_eq!(report.spilled_groups, 1, "{report:?}");
    assert_eq!(store.memory().spilled_columns, 2);
    assert_eq!(
        store.get_node_property(mia, &embedding()),
        Some(vector(&[88.0, 0.5]))
    );
}

/// A failed file write leaves the column as it was; dropping the store
/// lets go of its files.
#[test]
fn a_failed_write_changes_nothing_and_a_dropped_store_lets_go_of_its_files() {
    let store = RowGroupStore::new();
    let ids = items(&store);
    let files = Files::default();
    files.fail.store(true, Ordering::SeqCst);
    let before = store.memory();
    match store.spill_node_property_column(&embedding(), &files) {
        Err(Error::Io(_)) => {}
        other => panic!("expected the write's error, got {other:?}"),
    }
    assert_eq!(store.memory(), before);
    assert_embeddings(&store, &ids, "after the failed write");

    files.fail.store(false, Ordering::SeqCst);
    spill(&store, &files);
    assert_eq!(files.live(), 2);
    drop(store);
    assert_eq!(files.live(), 0, "the store's files went with it");
}

/// A row collected after its delete leaves the spilled columns it held no
/// value in spilled, and thaws the one it held a value in.
#[test]
fn a_collected_row_thaws_only_the_columns_it_held_a_value_in() {
    let store = RowGroupStore::new();
    items(&store);
    let files = Files::default();
    spill(&store, &files);
    let ranks = store
        .spill_node_property_column(&PropertyKey::new("rank"), &files)
        .unwrap();
    assert_eq!(ranks.spilled_groups, 2);
    // Node 3 has no embedding (3 % 7 == 3), but a rank.
    let mut tx = Tx::begin(&store, 10);
    tx.delete_node(NodeId::new(3)).unwrap();
    let deleted = tx.commit().unwrap();
    store.collect(deleted);
    assert_eq!(
        store.spilled_node_property_columns(),
        [embedding(), PropertyKey::new("rank")]
    );
    assert_eq!(
        store.memory().spilled_columns,
        3,
        "the first row group's rank column thawed"
    );
}

/// A section written from a spilled column holds its values (the writer
/// reads its chunks); a column the load left cold spills its chunks as they
/// are, and reads the same.
#[test]
fn a_checkpoint_reads_a_spilled_column_and_a_loaded_column_spills() {
    let store = RowGroupStore::new();
    let ids = items(&store);
    let files = Files::default();
    spill(&store, &files);
    let spilled = store.memory();
    let section = write(&store);
    assert_eq!(store.memory(), spilled, "the writer left it spilled");

    let loaded = read(&section);
    assert_embeddings(&loaded, &ids, "loaded");
    assert_eq!(committed(&loaded).nodes, committed(&store).nodes);
    let cold = loaded.memory();
    assert!(
        cold.cold_columns > 0 && cold.node_cold_bytes > 0,
        "{cold:?}"
    );
    let report = spill(&loaded, &files);
    assert_eq!(report.spilled_groups, 2, "{report:?}");
    let after = loaded.memory();
    assert_eq!(after.node_spilled_bytes, report.file_bytes);
    assert!(
        cold.node_cold_bytes - after.node_cold_bytes >= report.file_bytes,
        "the loaded chunks left the heap as they were: {cold:?} {after:?}"
    );
    assert_eq!(
        after.cold_columns, cold.cold_columns,
        "the names and ranks stay cold in memory"
    );
    assert_embeddings(&loaded, &ids, "loaded, then spilled");
    assert_eq!(loaded.memory(), after);
}

/// A section in memory.
#[derive(Default)]
struct Chunks {
    metas: Vec<ChunkMeta>,
    bytes: Vec<Bytes>,
}

impl SectionSink for Chunks {
    fn write_chunk(&mut self, meta: ChunkMeta, bytes: &[u8]) -> Result<()> {
        self.metas.push(meta);
        self.bytes.push(Bytes::copy_from_slice(bytes));
        Ok(())
    }
}

impl SectionSource for Chunks {
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

fn write(store: &RowGroupStore) -> Chunks {
    let mut chunks = Chunks::default();
    let graphs = [GraphToWrite {
        id: 0,
        name: "",
        store,
    }];
    write_lpg_section(
        &graphs,
        1,
        store.current_epoch(),
        ChunkCaps::DEFAULT,
        &mut chunks,
    )
    .unwrap();
    chunks
}

fn read(chunks: &Chunks) -> RowGroupStore {
    let (_, mut graphs) = read_lpg_section(chunks).unwrap();
    graphs.remove(0).store
}

#[cfg(feature = "vector-index")]
mod search {
    use std::sync::Arc;

    use grafeo_common::types::ReadView;

    use super::{Files, embedding, items, spill};
    use crate::graph::rowgroup::RowGroupStore;
    use crate::graph::traits::GraphStoreSearch;
    use crate::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind};

    /// Vector search over a spilled embedding column, through the index and
    /// by a scan, finds what it found before the spill, and reads every
    /// vector from the files: the column stays spilled.
    #[test]
    fn vector_search_reads_a_spilled_column_without_thawing_it() {
        let store = RowGroupStore::new();
        items(&store);
        store.add_vector_index(
            "Item",
            "embedding",
            Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
                2,
                DistanceMetric::Euclidean,
            )))),
        );
        let query = [2_050.25, 0.0];
        let searches = |store: &RowGroupStore| {
            let view = ReadView::committed(store.current_epoch());
            (
                store.vector_search_at(
                    Some("Item"),
                    "embedding",
                    &query,
                    5,
                    DistanceMetric::Euclidean,
                    view,
                ),
                store.vector_search_at(
                    None,
                    "embedding",
                    &query,
                    5,
                    DistanceMetric::Euclidean,
                    view,
                ),
                store.vector_search_with_threshold_at(
                    Some("Item"),
                    "embedding",
                    &query,
                    2.5,
                    DistanceMetric::Euclidean,
                    view,
                ),
            )
        };
        let before = searches(&store);
        assert_eq!(
            before
                .0
                .iter()
                .map(|(id, _)| id.as_u64())
                .collect::<Vec<_>>(),
            [2_050, 2_051, 2_049, 2_052, 2_048],
            "{before:?}"
        );
        let files = Files::default();
        spill(&store, &files);
        let spilled = store.memory();
        assert_eq!(spilled.spilled_columns, 2);
        assert_eq!(searches(&store), before);
        assert_eq!(
            store.memory(),
            spilled,
            "the searches read the files, nothing was decoded"
        );
        assert_eq!(store.spilled_node_property_columns(), [embedding()]);
    }
}

/// The spilled tier keeps history: in a store that keeps history, a write
/// of a spilled column thaws its row group's column and keeps the spilled
/// value for a reader before the write, a spill while that version lives
/// keeps the column in memory, and once a collect drops it the column
/// spills again and every reader at or after the horizon reads alike.
#[test]
fn a_spilled_column_keeps_its_value_for_readers_before_a_write() {
    let store = RowGroupStore::new();
    store.set_keeps_history(true);
    let ids = items(&store);
    store.collect(store.current_epoch());
    let files = Files::default();
    assert_eq!(spill(&store, &files).spilled_groups, 2);
    let spilled_at = store.current_epoch();
    let first = ids[1];
    let epoch = next_epoch(&store);
    apply_all(
        &store,
        &[DataOp::SetNodeProperty {
            id: first,
            key: embedding(),
            value: vector(&[88.0, 88.0]),
        }],
        Writer::Immediate {
            epoch,
            before_images: false,
        },
    )
    .unwrap();
    let at = |at| store.node_property_at(first, "embedding", ReadView::committed(at));
    assert_eq!(
        at(spilled_at),
        Some(embedding_of(first.as_u64())),
        "the spilled value"
    );
    assert_eq!(at(epoch), Some(vector(&[88.0, 88.0])));
    let report = spill(&store, &files);
    assert_eq!(
        (report.spilled_groups, report.kept_with_chains),
        (0, 1),
        "the thawed group stays in memory while its version lives: {report:?}"
    );
    assert_eq!(at(spilled_at), Some(embedding_of(first.as_u64())));

    store.collect(epoch);
    assert_eq!(spill(&store, &files).spilled_groups, 1, "it spills again");
    assert_eq!(at(epoch), Some(vector(&[88.0, 88.0])));
    assert_eq!(
        image(&store, Reader::At(epoch)).nodes,
        committed(&store).nodes
    );
}
