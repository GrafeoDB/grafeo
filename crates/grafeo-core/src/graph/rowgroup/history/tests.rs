//! History on the row-group store: replay with and without it, the floor a
//! collect leaves, the history calls, and history through a checkpoint and
//! a load (the tier on disk).

use bytes::Bytes;
use grafeo_common::change::DataOp;
use grafeo_common::storage::{ChunkCaps, ChunkKind, ChunkMeta, SectionSink, SectionSource};
use grafeo_common::types::{ArcStr, EpochId, PropertyKey, ReadView, Value};
use grafeo_common::utils::error::{Error, Result};

use crate::graph::apply::{ChangeTarget, Writer};
use crate::graph::conformance::{
    Reader, Rng, Tx, apply_all, committed, image, next_epoch, people, random_write,
};
use crate::graph::rowgroup::RowGroupStore;
use crate::graph::rowgroup::section::{
    GraphToWrite, LPG_SECTION_VERSION, read_lpg_section, write_lpg_section,
    write_lpg_section_with_history,
};
use crate::graph::traits::{GraphStore, GraphStoreSearch};

/// A section in memory, open to tampering.
#[derive(Default, Clone)]
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

/// The default graph of `store`, to write.
fn graph(store: &RowGroupStore) -> [GraphToWrite<'_>; 1] {
    [GraphToWrite {
        id: 0,
        name: "",
        store,
    }]
}

/// `store` written at its epoch with its history from `from`.
fn written(store: &RowGroupStore, from: Option<EpochId>, caps: ChunkCaps) -> Chunks {
    let mut chunks = Chunks::default();
    write_lpg_section_with_history(
        &graph(store),
        1,
        store.current_epoch(),
        from,
        caps,
        &mut chunks,
    )
    .unwrap();
    chunks
}

/// The default graph a section holds.
fn loaded(chunks: &Chunks) -> RowGroupStore {
    let (_, mut graphs) = read_lpg_section(chunks).unwrap();
    graphs.remove(0).store
}

/// What the history chunks of a section are: kind, namespace byte and
/// column.
fn history_chunks(chunks: &Chunks) -> Vec<(u8, u32)> {
    chunks
        .metas
        .iter()
        .filter(|meta| meta.kind == ChunkKind::History)
        .map(|meta| (meta.namespace.to_byte(), meta.column_id))
        .collect()
}

/// A store that keeps history, with the people, then Gus's age and labels
/// and Mia written by commits: returns the store and the epochs of its
/// commits (the people's first).
fn gus_ages() -> (
    RowGroupStore,
    crate::graph::conformance::People,
    Vec<EpochId>,
) {
    let store = RowGroupStore::new();
    store.set_keeps_history(true);
    let people = people(&store).unwrap();
    let mut epochs = vec![store.current_epoch()];
    let mut tx = Tx::begin(&store, 10);
    tx.set_node(people.gus, "age", Value::Int64(40)).unwrap();
    tx.add_label(people.gus, "Traveller").unwrap();
    epochs.push(tx.commit().unwrap());
    let mut tx = Tx::begin(&store, 11);
    tx.remove_node_value(people.gus, "age").unwrap();
    tx.set_edge(people.alix_gus, "since", Value::Int64(2020))
        .unwrap();
    epochs.push(tx.commit().unwrap());
    let mut tx = Tx::begin(&store, 12);
    tx.set_node(people.gus, "age", Value::Int64(41)).unwrap();
    tx.delete_edge(people.gus_mia).unwrap();
    tx.delete_node(people.mia).unwrap();
    epochs.push(tx.commit().unwrap());
    (store, people, epochs)
}

/// Replay keeps what a value or label write replaces only in a store that
/// keeps history: there a reader before the write's epoch reads the old
/// value and labels, and the vector is listed; without, the write keeps
/// nothing and lists nothing, as before.
#[test]
fn replay_keeps_what_it_replaces_only_in_a_store_that_keeps_history() {
    for keeps in [true, false] {
        let store = RowGroupStore::new();
        store.set_keeps_history(keeps);
        let people = people(&store).unwrap();
        store.collect(store.current_epoch());
        let before = store.current_epoch();
        let ops = [
            DataOp::SetNodeProperty {
                id: people.gus,
                key: PropertyKey::new("age"),
                value: Value::Int64(40),
            },
            DataOp::AddNodeLabel {
                id: people.gus,
                label: ArcStr::from("Traveller"),
            },
            DataOp::SetEdgeProperty {
                id: people.gus_mia,
                key: PropertyKey::new("since"),
                value: Value::Int64(2024),
            },
        ];
        let epoch = next_epoch(&store);
        apply_all(&store, &ops, Writer::Replay { epoch }).unwrap();
        let at = |epoch: EpochId| {
            let view = ReadView::committed(epoch);
            let labels = store.get_node_at(people.gus, view).unwrap().labels.len();
            (
                store.node_property_at(people.gus, "age", view),
                labels,
                store.edge_property_at(people.gus_mia, "since", view),
            )
        };
        assert_eq!(
            at(epoch),
            (Some(Value::Int64(40)), 3, Some(Value::Int64(2024))),
            "keeps {keeps}: the replayed writes"
        );
        if keeps {
            assert_eq!(
                at(before),
                (Some(Value::Int64(3)), 2, Some(Value::Int64(319))),
                "a reader before the group reads what it replaced"
            );
            assert_eq!(store.update_nodes(), 3, "the age, the labels, the since");
        } else {
            assert_eq!(
                at(before),
                at(epoch),
                "without history replay keeps nothing for earlier readers"
            );
            assert_eq!(store.update_nodes(), 0);
            assert!(store.listed_vectors().is_empty(), "and lists nothing");
        }
    }
}

/// A named graph created in a root that keeps history keeps it too, and
/// turning history off reaches the named graphs.
#[test]
fn named_graphs_keep_history_as_their_root_does() {
    let root = RowGroupStore::new();
    root.set_keeps_history(true);
    let trips = root.graph_or_create("trips").unwrap();
    assert!(trips.keeps_history());
    root.set_keeps_history(false);
    assert!(!trips.keeps_history());
    assert!(!root.graph_or_create("visits").unwrap().keeps_history());
}

/// A collect at a horizon keeps what every reader at the horizon or later
/// reads and moves the floor there; a reader before it may read what a
/// later commit wrote, which is why the engine refuses such a read.
#[test]
fn a_collect_at_the_horizon_keeps_every_later_read_and_moves_the_floor() {
    let (store, people, epochs) = gus_ages();
    let ages = |store: &RowGroupStore| -> Vec<Option<Value>> {
        epochs
            .iter()
            .map(|epoch| store.node_property_at(people.gus, "age", ReadView::committed(*epoch)))
            .collect()
    };
    let before = ages(&store);
    assert_eq!(
        before,
        [
            Some(Value::Int64(3)),
            Some(Value::Int64(40)),
            None,
            Some(Value::Int64(41))
        ]
    );
    assert_eq!(store.history_floor(), EpochId::new(0));
    let horizon = epochs[2];
    store.collect(horizon);
    assert_eq!(store.history_floor(), horizon, "the floor is the horizon");
    let after = ages(&store);
    assert_eq!(
        after[2..],
        before[2..],
        "every reader at the horizon or later"
    );
    assert_ne!(
        after[1], before[1],
        "a reader before the horizon no longer reads what it read"
    );
    assert!(
        store
            .get_node_at(people.mia, ReadView::committed(horizon))
            .is_some(),
        "Mia, deleted after the horizon, is still there for a reader at it"
    );
    store.collect(epochs[1]);
    assert_eq!(store.history_floor(), horizon, "the floor never goes back");
}

/// The history calls list each commit that wrote a property, with what it
/// wrote and a removal as null, oldest first; a node's versions, newest
/// first, each with its values and labels; and, past a collect, history
/// from the floor only.
#[test]
fn the_history_calls_list_each_commit_with_what_it_wrote() {
    let (store, people, epochs) = gus_ages();
    let [created, labelled, removed, again] = epochs[..] else {
        unreachable!("four commits")
    };
    assert_eq!(
        store.node_property_history_for_key(people.gus, "age"),
        [
            (created, Value::Int64(3)),
            (labelled, Value::Int64(40)),
            (removed, Value::Null),
            (again, Value::Int64(41)),
        ]
    );
    let all = store.node_property_history(people.gus);
    let keys: Vec<&str> = all.iter().map(|(key, _)| key.as_str()).collect();
    assert_eq!(keys, ["age", "name"]);
    assert_eq!(all[1].1, [(created, Value::from("Gus"))]);
    assert_eq!(
        store.edge_property_history(people.alix_gus),
        [(
            PropertyKey::new("since"),
            vec![(created, Value::Int64(1988)), (removed, Value::Int64(2020))]
        )]
    );
    assert_eq!(
        store.get_node_property_at_epoch(people.gus, &PropertyKey::new("age"), labelled),
        Some(Value::Int64(40))
    );
    assert_eq!(
        store.get_node_property_at_epoch(people.gus, &PropertyKey::new("age"), removed),
        None
    );

    let versions = store.get_node_history(people.gus);
    let spans: Vec<(EpochId, Option<EpochId>)> = versions
        .iter()
        .map(|(start, end, _)| (*start, *end))
        .collect();
    assert_eq!(
        spans,
        [
            (again, None),
            (removed, Some(again)),
            (labelled, Some(removed)),
            (created, Some(labelled)),
        ],
        "newest first, each ended by the next"
    );
    assert_eq!(
        versions[3].2.labels.len(),
        2,
        "Person and Employee at first"
    );
    assert_eq!(
        versions[2].2.labels.len(),
        3,
        "and a Traveller from then on"
    );
    assert_eq!(
        versions[1].2.properties.get(&PropertyKey::new("age")),
        None,
        "the age removed"
    );
    let mia = store.get_node_history(people.mia);
    assert_eq!(
        mia.iter()
            .map(|(start, end, _)| (*start, *end))
            .collect::<Vec<_>>(),
        [(created, Some(again))],
        "Mia's one version, ended by her delete"
    );
    assert_eq!(
        store
            .get_edge_history(people.alix_gus)
            .iter()
            .map(|(start, end, _)| (*start, *end))
            .collect::<Vec<_>>(),
        [(removed, None), (created, Some(removed))]
    );

    store.collect(removed);
    assert_eq!(
        store.node_property_history_for_key(people.gus, "age"),
        [(again, Value::Int64(41))],
        "from the floor: the age was removed there"
    );
    assert_eq!(
        store.node_property_history_for_key(people.alix, "age"),
        [(removed, Value::Int64(19))],
        "a value older than the floor starts at it"
    );
}

/// Without history, or with history from the section's own epoch, the
/// section is the one written before, chunk for chunk: no `History` chunk.
#[test]
fn without_history_the_section_is_written_as_before() {
    let (store, _, epochs) = gus_ages();
    let caps = ChunkCaps::current();
    let mut plain = Chunks::default();
    write_lpg_section(&graph(&store), 1, store.current_epoch(), caps, &mut plain).unwrap();
    for from in [None, Some(store.current_epoch())] {
        let chunks = written(&store, from, caps);
        assert_eq!(chunks.metas, plain.metas, "from {from:?}");
        assert_eq!(chunks.bytes, plain.bytes, "from {from:?}");
    }
    assert_eq!(history_chunks(&plain), []);
    let with = written(&store, Some(epochs[0]), caps);
    assert_eq!(
        history_chunks(&with),
        [
            (16, 0), // the node versions: Mia
            (16, 1), // Gus's labels
            (17, 1), // Gus's age (key 1)
            (32, 0), // the deleted edge's versions
            (33, 3), // Alix's since
        ]
    );
}

/// A checkpoint with history from H holds what every reader from H to the
/// checkpoint's epoch reads: random commits and rollbacks of every kind of
/// write, rows spread over vectors, caps that cut row groups into several
/// file row groups and chunks; the loaded store reads alike at every epoch
/// of the window, through every access path, and a collect at the epoch
/// leaves both stores in one state.
#[test]
fn a_checkpoint_with_history_reads_alike_at_every_epoch_of_its_window() {
    // How many seeds wrote history chunks, and how many reads in a window
    // saw another state than the one at its epoch: the workload has
    // history to keep.
    let (mut with_history, mut earlier_states) = (0, 0);
    for seed in 1..=24_u64 {
        let store = RowGroupStore::new();
        store.set_keeps_history(true);
        people(&store).unwrap();
        let mut rng = Rng::new(seed);
        let first = store.current_epoch();
        for transaction in 0..30_u64 {
            let mut tx = Tx::begin(&store, 100 + transaction);
            for _ in 0..=rng.below(4) {
                random_write(&mut tx, &mut rng).unwrap();
            }
            if rng.below(6) == 0 {
                tx.roll_back().unwrap();
            } else {
                tx.commit().unwrap();
            }
            if rng.below(4) == 0 {
                // A direct write, stamped at once at an epoch of its own.
                let seen = committed(&store);
                if let Some(id) = rng.pick(&seen.nodes.keys().copied().collect::<Vec<_>>()) {
                    apply_all(
                        &store,
                        &[DataOp::SetNodeProperty {
                            id: grafeo_common::types::NodeId::new(*id),
                            key: PropertyKey::new("city"),
                            value: Value::from(["Amsterdam", "Berlin", "Paris"][rng.below(3)]),
                        }],
                        Writer::Immediate {
                            epoch: next_epoch(&store),
                            before_images: false,
                        },
                    )
                    .unwrap();
                }
            }
        }
        let epoch = store.current_epoch();
        let span = usize::try_from(epoch.as_u64() - first.as_u64()).unwrap();
        let from = EpochId::new(first.as_u64() + rng.below(span + 1) as u64);
        store.collect(from);
        let caps = if seed % 3 == 0 {
            ChunkCaps {
                max_rows: 4,
                max_bytes: 256,
            }
        } else {
            ChunkCaps::current()
        };
        let chunks = written(&store, Some(from), caps);
        if !history_chunks(&chunks).is_empty() {
            with_history += 1;
        }
        let loaded = loaded(&chunks);
        let now = committed(&store);
        for at in from.as_u64()..=epoch.as_u64() {
            let reader = Reader::At(EpochId::new(at));
            let seen = image(&store, reader);
            assert!(seen.problems.is_empty(), "seed {seed}: {:?}", seen.problems);
            assert_eq!(image(&loaded, reader), seen, "seed {seed}, epoch {at}");
            if seen != now {
                earlier_states += 1;
            }
        }
        assert_eq!(loaded.current_epoch(), epoch, "seed {seed}");
        assert_eq!(
            (loaded.node_count(), loaded.edge_count()),
            (store.node_count(), store.edge_count()),
            "seed {seed}: the counts at the epoch"
        );
        store.collect(epoch);
        loaded.collect(epoch);
        assert_eq!(committed(&loaded), committed(&store), "seed {seed}");
        assert_eq!(
            (loaded.update_nodes(), loaded.stamped_vectors()),
            (0, 0),
            "seed {seed}"
        );
    }
    assert!(with_history >= 20, "{with_history} seeds wrote history");
    assert!(
        earlier_states >= 100,
        "{earlier_states} earlier states read"
    );
}

/// The history a load restores is listed for the collect: a collect at the
/// section's epoch leaves the store with the memory of one loaded without
/// history.
#[test]
fn a_loaded_history_collects_to_the_memory_of_a_load_without_it() {
    let (store, _, epochs) = gus_ages();
    let caps = ChunkCaps::current();
    let with = loaded(&written(&store, Some(epochs[0]), caps));
    let without = loaded(&written(&store, None, caps));
    assert!(with.update_nodes() > 0 && !with.listed_vectors().is_empty());
    assert_eq!(without.update_nodes(), 0);
    with.collect(with.current_epoch());
    without.collect(without.current_epoch());
    assert_eq!(with.listed_vectors(), without.listed_vectors());
    assert_eq!(with.update_nodes(), 0);
    assert_eq!(committed(&with), committed(&without));
    assert_eq!(with.version_bytes(), without.version_bytes());
}

/// A column a load left cold (the tier compressed in memory) keeps its
/// loaded value for a reader before a later write: the write thaws it and
/// keeps the value, and a checkpoint with history after it holds both.
#[test]
fn a_cold_column_keeps_its_loaded_value_for_readers_before_a_write() {
    let source = RowGroupStore::new();
    let people = people(&source).unwrap();
    let store = loaded(&written(&source, None, ChunkCaps::current()));
    store.set_keeps_history(true);
    assert!(
        store.memory().cold_columns > 0,
        "the load left the columns cold"
    );
    let loaded_at = store.current_epoch();
    let epoch = next_epoch(&store);
    store
        .apply(
            &DataOp::SetNodeProperty {
                id: people.alix,
                key: PropertyKey::new("city"),
                value: Value::from("Berlin"),
            },
            Writer::Immediate {
                epoch,
                before_images: false,
            },
        )
        .unwrap();
    let city = |store: &RowGroupStore, at: EpochId| {
        store.node_property_at(people.alix, "city", ReadView::committed(at))
    };
    assert_eq!(city(&store, loaded_at), Some(Value::from("Amsterdam")));
    assert_eq!(city(&store, epoch), Some(Value::from("Berlin")));
    let reloaded = loaded(&written(&store, Some(loaded_at), ChunkCaps::current()));
    assert_eq!(city(&reloaded, loaded_at), Some(Value::from("Amsterdam")));
    assert_eq!(city(&reloaded, epoch), Some(Value::from("Berlin")));
}

/// Tampers with the one `History` chunk of kind `namespace`/`column` in a
/// section: its cell for `row` becomes `cell`.
fn tampered(chunks: &Chunks, namespace: u8, column: u32, cell: Value) -> Chunks {
    use crate::codec::column_chunk::encode_column_chunk;

    let mut tampered = chunks.clone();
    let index = tampered
        .metas
        .iter()
        .position(|meta| {
            meta.kind == ChunkKind::History
                && meta.namespace.to_byte() == namespace
                && meta.column_id == column
        })
        .expect("such a history chunk");
    let meta = tampered.metas[index];
    let (codec, bytes) = encode_column_chunk(meta.row_count, &[(0, cell)], None).unwrap();
    tampered.metas[index].codec = codec.to_byte();
    tampered.bytes[index] = Bytes::from(bytes);
    tampered
}

/// The reader refuses history no writer writes, naming the chunk: a delete
/// not after its create, an epoch past the section's, epochs that go back,
/// a label not in the dictionary, and a history chunk where no column has
/// one.
#[test]
fn the_reader_refuses_history_no_writer_writes() {
    let (store, _, epochs) = gus_ages();
    let chunks = written(&store, Some(epochs[0]), ChunkCaps::current());
    let epoch = i64::try_from(store.current_epoch().as_u64()).unwrap();
    let pair = |a: Value, b: Value| Value::List(vec![a, b].into());
    let cases = [
        (
            tampered(&chunks, 16, 0, pair(Value::Int64(3), Value::Int64(3))),
            "not after its create",
        ),
        (
            tampered(&chunks, 16, 0, pair(Value::Int64(epoch + 1), Value::Null)),
            "past the section's epoch",
        ),
        (
            tampered(
                &chunks,
                17,
                1,
                Value::List(
                    vec![
                        pair(Value::Int64(3), Value::Int64(40)),
                        pair(Value::Int64(2), Value::Int64(41)),
                    ]
                    .into(),
                ),
            ),
            "not ascending",
        ),
        (
            tampered(
                &chunks,
                16,
                1,
                Value::List(
                    vec![pair(
                        Value::Int64(2),
                        Value::List(vec![Value::Int64(88)].into()),
                    )]
                    .into(),
                ),
            ),
            "no label of the graph",
        ),
    ];
    for (chunks, what) in cases {
        match read_lpg_section(&chunks) {
            Err(Error::Corruption(corruption)) => {
                let text = corruption.to_string();
                assert!(text.contains(what), "{text:?} names {what:?}");
                assert!(text.contains("History"), "{text:?} names the chunk");
            }
            other => panic!("{what}: {other:?}"),
        }
    }
    let mut misplaced = chunks.clone();
    let index = misplaced
        .metas
        .iter()
        .position(|meta| meta.kind == ChunkKind::History && meta.column_id == 1)
        .unwrap();
    misplaced.metas[index].column_id = 7;
    assert!(matches!(
        read_lpg_section(&misplaced),
        Err(Error::Corruption(_))
    ));
}

/// An edge a commit in the window deleted is read until its delete after a
/// load, and stays in its source's adjacency for those readers.
#[test]
fn an_edge_deleted_in_the_window_is_read_until_its_delete_after_a_load() {
    let (store, people, epochs) = gus_ages();
    let loaded = loaded(&written(&store, Some(epochs[0]), ChunkCaps::current()));
    let edge = |at: EpochId| loaded.get_edge_at(people.gus_mia, ReadView::committed(at));
    assert_eq!(
        edge(epochs[2]).map(|edge| edge.dst),
        Some(people.mia),
        "before the delete"
    );
    assert!(edge(epochs[3]).is_none(), "from the delete on");
    let outgoing = |at: EpochId| {
        loaded.edges_from_at(
            people.gus,
            crate::graph::Direction::Outgoing,
            ReadView::committed(at),
        )
    };
    assert_eq!(outgoing(epochs[2]), [(people.mia, people.gus_mia)]);
    assert_eq!(outgoing(epochs[3]), []);
}

/// Documents of Alix (on graphs, near the first axis) and Gus (on storage),
/// created by one commit, then Alix's deleted and Gus's rewritten onto
/// graphs near the first axis by the next: the store keeps history and is
/// collected at the first commit, as a retention window holds it. Returns
/// the store, the two documents and the two epochs.
#[cfg(any(feature = "text-index", feature = "vector-index"))]
fn rewritten_docs(
    index: impl Fn(&RowGroupStore),
) -> (
    RowGroupStore,
    [grafeo_common::types::NodeId; 2],
    [EpochId; 2],
) {
    let store = RowGroupStore::new();
    store.set_keeps_history(true);
    index(&store);
    let mut tx = Tx::begin(&store, 10);
    let alix = tx
        .create_node(
            &["Doc"],
            &[
                ("body", Value::from("graph databases in Amsterdam")),
                ("embedding", Value::Vector(vec![1.0, 0.0, 0.0].into())),
            ],
        )
        .unwrap();
    let gus = tx
        .create_node(
            &["Doc"],
            &[
                ("body", Value::from("rust storage engines in Berlin")),
                ("embedding", Value::Vector(vec![0.0, 1.0, 0.0].into())),
            ],
        )
        .unwrap();
    let created = tx.commit().unwrap();
    let mut tx = Tx::begin(&store, 11);
    tx.delete_node(alix).unwrap();
    tx.set_node(gus, "body", Value::from("graph storage in Paris"))
        .unwrap();
    tx.set_node(gus, "embedding", Value::Vector(vec![0.9, 0.1, 0.0].into()))
        .unwrap();
    let rewritten = tx.commit().unwrap();
    store.collect(created);
    (store, [alix, gus], [created, rewritten])
}

/// A text search at an epoch the window keeps finds what existed then: a
/// deleted document until the horizon passes its delete, a rewritten one
/// by its words then; after a load of the history, through an index built
/// from the rows of now as the engine builds it, alike.
#[cfg(feature = "text-index")]
#[test]
fn a_text_search_in_the_window_finds_what_existed_then() {
    use std::sync::Arc;

    use parking_lot::RwLock;

    use crate::index::text::{BM25Config, InvertedIndex};

    let index = || Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
    let (store, [alix, gus], [created, rewritten]) =
        rewritten_docs(|store| store.add_text_index("Doc", "body", index()));
    let found = |store: &RowGroupStore, at: EpochId| -> Vec<grafeo_common::types::NodeId> {
        let mut found: Vec<_> = store
            .text_search_at("Doc", "body", "graph", 10, ReadView::committed(at))
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        found.sort_unstable();
        found
    };
    assert_eq!(found(&store, created), [alix], "Alix's words before");
    assert_eq!(found(&store, rewritten), [gus], "Gus's after");

    let loaded = loaded(&written(&store, Some(created), ChunkCaps::current()));
    let built = index();
    {
        let mut built = built.write();
        for id in loaded.nodes_by_label("Doc") {
            if let Some(Value::String(text)) =
                loaded.get_node_property(id, &PropertyKey::new("body"))
            {
                built.insert(id, &text);
            }
        }
    }
    loaded.add_text_index("Doc", "body", built);
    assert_eq!(found(&loaded, created), [alix], "after the load");
    assert_eq!(found(&loaded, rewritten), [gus]);

    store.collect(rewritten);
    assert_eq!(found(&store, rewritten), [gus], "a reader at the horizon");
}

/// A vector search at an epoch the window keeps finds the vectors of then:
/// Alix's, deleted since, nearest to the first axis before, Gus's moved
/// there after; after a load of the history alike.
#[cfg(feature = "vector-index")]
#[test]
fn a_vector_search_in_the_window_finds_the_vectors_of_then() {
    use std::sync::Arc;

    use crate::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind};

    let index = || {
        Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
            3,
            DistanceMetric::Euclidean,
        ))))
    };
    let (store, [alix, gus], [created, rewritten]) =
        rewritten_docs(|store| store.add_vector_index("Doc", "embedding", index()));
    let nearest = |store: &RowGroupStore, at: EpochId| {
        store
            .vector_search_at(
                Some("Doc"),
                "embedding",
                &[1.0, 0.0, 0.0],
                1,
                DistanceMetric::Euclidean,
                ReadView::committed(at),
            )
            .first()
            .map(|(id, _)| *id)
    };
    assert_eq!(nearest(&store, created), Some(alix));
    assert_eq!(nearest(&store, rewritten), Some(gus));
    let loaded = loaded(&written(&store, Some(created), ChunkCaps::current()));
    loaded.add_vector_index("Doc", "embedding", index());
    assert_eq!(nearest(&loaded, created), Some(alix), "after the load");
    assert_eq!(nearest(&loaded, rewritten), Some(gus));
}

/// A property index lookup at an epoch the window keeps finds the nodes by
/// their value then: the index is a superset of every value a reader at
/// the horizon or later reads, rechecked at the view; after a load, an
/// index built from the rows finds them alike.
#[test]
fn a_property_lookup_in_the_window_finds_the_values_of_then() {
    let (store, people, epochs) = gus_ages();
    store.create_property_index("age");
    let found = |store: &RowGroupStore, age: i64, at: EpochId| {
        store.find_nodes_by_property_at("age", &Value::Int64(age), ReadView::committed(at))
    };
    for store in [
        &store,
        &loaded(&written(&store, Some(epochs[0]), ChunkCaps::current())),
    ] {
        store.create_property_index("age");
        assert_eq!(found(store, 3, epochs[0]), [people.gus], "Gus was 3");
        assert_eq!(found(store, 40, epochs[1]), [people.gus]);
        assert_eq!(found(store, 40, epochs[2]), [], "his age removed");
        assert_eq!(found(store, 3, epochs[3]), []);
        assert_eq!(found(store, 41, epochs[3]), [people.gus]);
    }
}
