//! The saved vector index format (#711): round trips, equal bytes, the
//! pieces' cut and every refusal.

use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use parking_lot::RwLock;

use grafeo_common::storage::ChunkCaps;
use grafeo_common::storage::catalog_record::{
    DistanceMetricRecord, IndexKindRecord, QuantizationRecord,
};
use grafeo_common::storage::section::{ChunkMeta, SectionSink, SectionSource};
use grafeo_common::types::NodeId;
use grafeo_common::utils::error::{Error, Result};

use super::*;
use crate::index::vector::{
    DistanceMetric, HnswConfig, HnswIndex, QuantizationType, QuantizedHnswIndex, VectorAccessor,
    VectorIndexKind,
};

const DIMENSIONS: usize = 8;

/// The vectors the indexes read, kept as a store keeps them.
#[derive(Default, Clone)]
struct Store(Arc<RwLock<HashMap<NodeId, Arc<[f32]>>>>);

impl VectorAccessor for Store {
    fn get_vector(&self, id: NodeId) -> Option<Arc<[f32]>> {
        self.0.read().get(&id).cloned()
    }
}

impl Store {
    fn insert(&self, index: &VectorIndexKind, id: NodeId, vector: &[f32]) {
        self.0.write().insert(id, vector.into());
        index.insert(id, vector, self);
    }

    fn remove(&self, index: &VectorIndexKind, id: NodeId) {
        index.remove(id, self);
        self.0.write().remove(&id);
    }
}

/// The chunks of one section, written and read in memory.
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
        self.bytes
            .get(index)
            .cloned()
            .ok_or_else(|| Error::Internal(format!("no chunk {index}")))
    }

    fn stored_length(&self, index: usize) -> Result<u64> {
        self.fetch(index).map(|bytes| bytes.len() as u64)
    }

    fn section_version(&self) -> u8 {
        VECTOR_INDEX_SECTION_VERSION
    }
}

impl Chunks {
    /// The position of the first chunk `matches` says yes to.
    fn position(&self, matches: impl Fn(&ChunkMeta) -> bool) -> usize {
        self.metas
            .iter()
            .position(matches)
            .expect("a matching chunk")
    }
}

/// A vector of `DIMENSIONS` values drawn from `seed`.
fn vector_of(seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(19);
    (0..DIMENSIONS)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) % 1000) as f32 / 100.0 - 5.0
        })
        .collect()
}

/// The ids of the build: 300 nodes over three row groups.
fn ids() -> Vec<NodeId> {
    (0..300u64)
        .map(|i| NodeId::new((i % 3) * ROWS_PER_GROUP + i * 7 + 1))
        .collect()
}

/// What an index is built as.
#[derive(Debug, Clone, Copy)]
enum Build {
    Plain,
    Quantized {
        quantization: QuantizationType,
        threshold: usize,
    },
}

/// Every form the round trips cover: plain, each quantization fitted, and
/// scalar and product before their fit.
fn every_build() -> Vec<Build> {
    let quantized = |quantization, threshold| Build::Quantized {
        quantization,
        threshold,
    };
    vec![
        Build::Plain,
        quantized(QuantizationType::None, 10),
        quantized(QuantizationType::Scalar, 10),
        quantized(QuantizationType::Scalar, 310),
        quantized(QuantizationType::Binary, 10),
        quantized(QuantizationType::Product { num_subvectors: 4 }, 10),
        quantized(QuantizationType::Product { num_subvectors: 4 }, 310),
    ]
}

/// An empty index of `build`.
fn empty(build: Build) -> VectorIndexKind {
    let config = HnswConfig::new(DIMENSIONS, DistanceMetric::Euclidean).with_m(6);
    match build {
        Build::Plain => VectorIndexKind::Hnsw(HnswIndex::new(config)),
        Build::Quantized {
            quantization,
            threshold,
        } => VectorIndexKind::Quantized(
            QuantizedHnswIndex::new(config, quantization).with_training_threshold(threshold),
        ),
    }
}

/// An index of `build` over `ids()`: every 25th vector an exact duplicate
/// of the one before (an alias), then a few removals, among them a
/// representative with an alias (its alias takes its place).
fn built(build: Build) -> (VectorIndexKind, Store) {
    let index = empty(build);
    let store = Store::default();
    let ids = ids();
    for (i, &id) in ids.iter().enumerate() {
        let seed = if i % 25 == 24 { i as u64 - 1 } else { i as u64 };
        store.insert(&index, id, &vector_of(seed));
    }
    for i in [3, 23, 150, 151] {
        store.remove(&index, ids[i]);
    }
    (index, store)
}

/// `index` written as column 5 of graph 2 (quantizer stream 6), with
/// `listed`, at `max_bytes` per piece, and its metadata last.
fn written(index: &VectorIndexKind, listed: &[NodeId], max_bytes: u32) -> (Chunks, SavedIndexMeta) {
    let mut chunks = Chunks::default();
    let quantizer_column = index
        .quantization_type()
        .filter(|quantization| *quantization != QuantizationType::None)
        .map(|_| 6);
    let caps = ChunkCaps {
        max_rows: 65_536,
        max_bytes,
    };
    let written =
        write_vector_index(&mut chunks, 2, 5, quantizer_column, index, listed, caps).unwrap();
    let meta = SavedIndexMeta {
        graph_id: 2,
        column: 5,
        quantizer_column,
        definition: SavedDefinition::Index(IndexKindRecord::Vector {
            label: "Doc".to_string(),
            property: "emb".to_string(),
            dimensions: u32::try_from(DIMENSIONS).expect("a few dimensions"),
            metric: DistanceMetricRecord::Euclidean,
            m: 6,
            ef_construction: 200,
            quantization: QuantizationRecord::None,
        }),
        members: written.members,
        entry: written.entry,
    };
    let section = SavedMeta {
        layout: SAVED_META_LAYOUT,
        epoch: 19,
        max_bytes,
        next_column: 7,
        indexes: vec![meta.clone()],
    };
    chunks
        .write_chunk(ChunkMeta::meta(), &encode_meta(&section).unwrap())
        .unwrap();
    (chunks, meta)
}

/// `chunks` read into a fresh index of `build`, with its listed nodes.
fn restored(build: Build, chunks: &Chunks) -> Result<(VectorIndexKind, Vec<NodeId>)> {
    let meta = read_meta(chunks)?;
    let index = empty(build);
    let listed = read_vector_index(chunks, &meta.indexes[0], &index)?;
    Ok((index, listed))
}

/// What an index holds, compared whole: its members' saved forms and codes,
/// its quantizer, entry point and length.
fn content(index: &VectorIndexKind) -> String {
    let hnsw = index.hnsw();
    let members: Vec<_> = hnsw
        .member_ids()
        .into_iter()
        .map(|id| {
            let code = index
                .as_quantized()
                .and_then(|quantized| quantized.code_of(id));
            (id, hnsw.saved_form(id), code)
        })
        .collect();
    let quantizer = index
        .as_quantized()
        .map(QuantizedHnswIndex::saved_quantizer);
    format!(
        "{members:?} {quantizer:?} {:?} {}",
        hnsw.entry_and_top(),
        index.len()
    )
}

/// The searches compared: `k` 5 and every member, for a few queries.
fn searches(index: &VectorIndexKind, store: &Store) -> Vec<Vec<(NodeId, f32)>> {
    let mut found = Vec::new();
    for seed in [3, 19, 88, 1000] {
        for k in [5, index.len()] {
            found.push(index.search(&vector_of(seed), k, store));
        }
    }
    found
}

#[test]
fn every_build_round_trips_through_its_pieces_and_streams() {
    for build in every_build() {
        let (index, store) = built(build);
        let (chunks, _) = written(&index, &[], 512);
        let (back, listed) = restored(build, &chunks).unwrap();
        assert!(listed.is_empty(), "{build:?}");
        assert_eq!(content(&back), content(&index), "{build:?}");
        assert_eq!(
            searches(&back, &store),
            searches(&index, &store),
            "{build:?}"
        );
        assert_eq!(
            back.counted_bytes(),
            back.heap_memory_bytes(),
            "{build:?}: the restore counts what it holds"
        );
    }
}

/// The aliases come back: the removed representative's alias still stands
/// in its place, and a search returns an alias at its representative's
/// distance.
#[test]
fn aliases_survive_a_restore() {
    let (index, store) = built(Build::Plain);
    let aliases = |index: &VectorIndexKind| -> Vec<NodeId> {
        let hnsw = index.hnsw();
        hnsw.member_ids()
            .into_iter()
            .filter(|&id| matches!(hnsw.saved_form(id), Some(SavedNode::Alias { .. })))
            .collect()
    };
    assert!(!aliases(&index).is_empty(), "the build holds aliases");
    let (chunks, _) = written(&index, &[], 1 << 20);
    let (back, _) = restored(Build::Plain, &chunks).unwrap();
    assert_eq!(aliases(&back), aliases(&index));
    let alias = aliases(&index)[0];
    let query = store.get_vector(alias).unwrap();
    let found = back.search(&query, 2, &store);
    assert!(
        found
            .iter()
            .any(|&(id, distance)| id == alias && distance == 0.0)
    );
}

/// Inserts after a restore give what they give without it: the levels are
/// drawn as before, and a quantizer not fitted at the save fits on the same
/// samples in the same order.
#[test]
fn a_restored_index_takes_inserts_as_the_original_does() {
    for build in every_build() {
        let (index, store) = built(build);
        let (chunks, _) = written(&index, &[], 4096);
        let (back, _) = restored(build, &chunks).unwrap();
        for i in 0..20u64 {
            let id = NodeId::new(4 * ROWS_PER_GROUP + i);
            store.insert(&index, id, &vector_of(500 + i));
            back.insert(id, &vector_of(500 + i), &store);
        }
        assert_eq!(content(&back), content(&index), "{build:?}");
    }
}

/// The same content writes the same bytes: a restored index writes what
/// the original wrote.
#[test]
fn equal_content_writes_equal_bytes() {
    for build in every_build() {
        let (index, _) = built(build);
        let (first, _) = written(&index, &[NodeId::new(8)], 700);
        let (back, _) = restored(build, &first).unwrap();
        let (second, _) = written(&back, &[NodeId::new(8)], 700);
        assert_eq!(first.metas, second.metas, "{build:?}");
        assert_eq!(first.bytes, second.bytes, "{build:?}");
    }
}

/// Pieces lie within one row group, ascend, do not overlap and keep to the
/// cap, but for a node that passes it alone.
#[test]
fn pieces_keep_to_their_row_group_and_their_cap() {
    let (index, _) = built(Build::Quantized {
        quantization: QuantizationType::Scalar,
        threshold: 10,
    });
    for max_bytes in [1, 120, 512, 1 << 20] {
        let (chunks, _) = written(&index, &[], max_bytes);
        let pieces: Vec<(&ChunkMeta, &Bytes)> = chunks
            .metas
            .iter()
            .zip(&chunks.bytes)
            .filter(|(meta, _)| meta.kind == ChunkKind::Column)
            .collect();
        let mut end = 0;
        let mut groups = std::collections::BTreeSet::new();
        for (meta, bytes) in &pieces {
            let last = meta.row_start + u64::from(meta.row_count) - 1;
            assert_eq!(meta.row_start / ROWS_PER_GROUP, last / ROWS_PER_GROUP);
            assert!(meta.row_start >= end, "ascending and apart");
            end = last + 1;
            groups.insert(meta.row_start / ROWS_PER_GROUP);
            let nodes = u32::from_le_bytes(bytes[..4].try_into().unwrap());
            assert!(
                bytes.len() <= max_bytes as usize || nodes == 1,
                "{max_bytes}"
            );
        }
        assert_eq!(groups.len(), 3, "{max_bytes}: the three row groups");
        if max_bytes == 1 {
            assert_eq!(
                pieces.len(),
                index.len(),
                "a piece per node at a 1-byte cap"
            );
        }
        if max_bytes == 1 << 20 {
            assert_eq!(pieces.len(), 3, "a piece per row group at the default cap");
        }
    }
}

/// A listed node's links need not hold: the open derives it again. The
/// list comes back as written.
#[test]
fn links_to_listed_nodes_are_accepted_and_the_list_comes_back() {
    let (index, _) = built(Build::Plain);
    let hnsw = index.hnsw();
    let members = hnsw.member_ids();
    let (gone, keeper) = (members[10], members[11]);
    // A neighbor that is no node: refused, unless it is listed.
    let Some(SavedNode::Graph { hash, mut levels }) = hnsw.saved_form(keeper) else {
        panic!("a graph node");
    };
    levels[0].push(NodeId::new(9_999_999));
    let edited = empty(Build::Plain);
    let (entry, top) = hnsw.entry_and_top();
    edited.hnsw().begin_restore(entry, top, 0);
    for id in members {
        let node = if id == keeper {
            SavedNode::Graph {
                hash,
                levels: levels.clone(),
            }
        } else {
            hnsw.saved_form(id).unwrap()
        };
        edited.hnsw().restore_saved(id, node);
    }
    let (chunks, _) = written(&edited, &[gone], 1 << 20);
    let refused = restored(Build::Plain, &chunks)
        .map(drop)
        .unwrap_err()
        .to_string();
    assert!(refused.contains("9999999"), "{refused}");
    let (chunks, _) = written(&edited, &[gone, NodeId::new(9_999_999)], 1 << 20);
    let (_, listed) = restored(Build::Plain, &chunks).unwrap();
    assert_eq!(listed, vec![gone, NodeId::new(9_999_999)]);
}

/// An entry point that is listed and no longer the top node gives way to
/// the graph node with the most levels; one that is not listed is refused.
#[test]
fn a_listed_entry_point_is_picked_again() {
    let (index, _) = built(Build::Plain);
    let hnsw = index.hnsw();
    let (Some(_), top) = hnsw.entry_and_top() else {
        panic!("an entry point");
    };
    assert!(top > 0, "the build has levels above the first");
    let expected = hnsw
        .member_ids()
        .into_iter()
        .find(|&id| matches!(hnsw.saved_form(id), Some(SavedNode::Graph { levels, .. }) if levels.len() == top + 1));
    let lower = hnsw
        .member_ids()
        .into_iter()
        .find(|&id| matches!(hnsw.saved_form(id), Some(SavedNode::Graph { levels, .. }) if levels.len() == 1))
        .unwrap();
    let mut chunks = written(&index, &[], 1 << 20).0;
    let mut meta = read_meta(&chunks).unwrap();
    meta.indexes[0].entry = Some((lower.as_u64(), u32::try_from(top).expect("a few levels")));
    let last = chunks.metas.len() - 1;
    chunks.bytes[last] = Bytes::from(encode_meta(&meta).unwrap());
    let refused = restored(Build::Plain, &chunks)
        .map(drop)
        .unwrap_err()
        .to_string();
    assert!(refused.contains("entry point"), "{refused}");

    let mut listed_chunks = written(&index, &[lower], 1 << 20).0;
    let last = listed_chunks.metas.len() - 1;
    listed_chunks.bytes[last] = Bytes::from(encode_meta(&meta).unwrap());
    let (back, _) = restored(Build::Plain, &listed_chunks).unwrap();
    assert_eq!(
        back.hnsw().entry_and_top(),
        (expected, top),
        "the top node with the lowest id"
    );
}

/// Pieces that break the format's rules, each refused as a corruption.
#[test]
fn a_piece_that_breaks_the_rules_is_refused() {
    let entry = |offset: u32, tag: u8, rest: &[u8]| -> Vec<u8> {
        let mut out = offset.to_le_bytes().to_vec();
        out.push(tag);
        out.extend_from_slice(rest);
        out
    };
    let graph = |neighbors: &[u64]| -> Vec<u8> {
        let mut out = 7u64.to_le_bytes().to_vec();
        out.push(1);
        out.extend_from_slice(
            &u32::try_from(neighbors.len())
                .expect("a few neighbors")
                .to_le_bytes(),
        );
        for neighbor in neighbors {
            out.extend_from_slice(&neighbor.to_le_bytes());
        }
        out
    };
    let piece = |nodes: u32, entries: &[Vec<u8>]| -> Vec<u8> {
        let mut out = nodes.to_le_bytes().to_vec();
        for entry in entries {
            out.extend_from_slice(entry);
        }
        out
    };
    let good = piece(
        2,
        &[entry(0, 0, &graph(&[101])), entry(1, 0, &graph(&[100]))],
    );
    assert_eq!(decode_piece(&good, 100, 2, None).unwrap().len(), 2);
    let cases: Vec<(&str, Vec<u8>, Option<usize>)> = vec![
        (
            "offsets not increasing",
            piece(
                2,
                &[entry(1, 0, &graph(&[100])), entry(1, 0, &graph(&[100]))],
            ),
            None,
        ),
        (
            "an offset past the rows",
            piece(1, &[entry(2, 0, &graph(&[100]))]),
            None,
        ),
        (
            "a graph node without levels",
            piece(
                1,
                &[entry(0, 0, &{
                    let mut out = 7u64.to_le_bytes().to_vec();
                    out.push(0);
                    out
                })],
            ),
            None,
        ),
        (
            "a node that lists itself",
            piece(1, &[entry(0, 0, &graph(&[100]))]),
            None,
        ),
        (
            "an alias of itself",
            piece(1, &[entry(0, 1, &100u64.to_le_bytes())]),
            None,
        ),
        (
            "an unknown tag",
            piece(1, &[entry(0, 2, &101u64.to_le_bytes())]),
            None,
        ),
        (
            "bytes after the nodes",
            [good.clone(), vec![0]].concat(),
            None,
        ),
        (
            "a piece that ends early",
            good[..good.len() - 3].to_vec(),
            None,
        ),
        (
            "more neighbors than the piece holds",
            piece(
                1,
                &[entry(0, 0, &{
                    let mut out = 7u64.to_le_bytes().to_vec();
                    out.push(1);
                    out.extend_from_slice(&u32::MAX.to_le_bytes());
                    out
                })],
            ),
            None,
        ),
        (
            "a code of another length",
            piece(
                1,
                &[[
                    entry(0, 1, &101u64.to_le_bytes()),
                    3u32.to_le_bytes().to_vec(),
                    vec![1, 2, 3],
                ]
                .concat()],
            ),
            Some(4),
        ),
        ("a node count past the bytes", piece(u32::MAX, &[]), None),
    ];
    for (case, bytes, code_length) in cases {
        let error = decode_piece(&bytes, 100, 2, code_length).unwrap_err();
        assert!(matches!(error, Error::Corruption(_)), "{case}: {error}");
    }
}

/// Sections whose pieces, members, aliases or quantizer break the rules.
#[test]
fn an_index_that_breaks_the_rules_is_refused() {
    let (index, _) = built(Build::Quantized {
        quantization: QuantizationType::Scalar,
        threshold: 10,
    });
    let (chunks, _) = written(&index, &[], 512);
    let first_piece = chunks.position(|meta| meta.kind == ChunkKind::Column);
    let refuse = |case: &str, chunks: &Chunks, build: Build, says: &str| {
        let error = restored(build, chunks).map(drop).unwrap_err();
        assert!(
            matches!(error, Error::Corruption(_)) && error.to_string().contains(says),
            "{case}: {error}"
        );
    };
    let scalar = Build::Quantized {
        quantization: QuantizationType::Scalar,
        threshold: 10,
    };

    let mut across = chunks.clone();
    across.metas[first_piece].row_start = ROWS_PER_GROUP - 1;
    across.metas[first_piece].row_count = 2;
    refuse(
        "a piece across two row groups",
        &across,
        scalar,
        "row group",
    );

    let mut overlapping = chunks.clone();
    let second = first_piece + 1;
    overlapping.metas[second].row_start = overlapping.metas[first_piece].row_start;
    refuse("overlapping pieces", &overlapping, scalar, "row group");

    let mut fewer = chunks.clone();
    fewer.metas.remove(second);
    fewer.bytes.remove(second);
    refuse(
        "a member count that does not add up",
        &fewer,
        scalar,
        "members",
    );

    refuse(
        "a quantizer of another kind",
        &chunks,
        Build::Quantized {
            quantization: QuantizationType::Binary,
            threshold: 10,
        },
        "kind",
    );
    refuse(
        "a quantizer stream for a plain index",
        &chunks,
        Build::Plain,
        "quantizer",
    );

    let mut no_entry = chunks.clone();
    let mut meta = read_meta(&no_entry).unwrap();
    meta.indexes[0].entry = None;
    let last = no_entry.metas.len() - 1;
    no_entry.bytes[last] = Bytes::from(encode_meta(&meta).unwrap());
    refuse(
        "graph nodes without an entry point",
        &no_entry,
        scalar,
        "no entry point",
    );

    // A stray alias: an alias written without its representative.
    let (plain, _) = built(Build::Plain);
    let hnsw = plain.hnsw();
    let (alias, representative) = hnsw
        .member_ids()
        .into_iter()
        .find_map(|id| match hnsw.saved_form(id) {
            Some(SavedNode::Alias { representative }) => Some((id, representative)),
            _ => None,
        })
        .unwrap();
    let stray = empty(Build::Plain);
    let (entry, top) = hnsw.entry_and_top();
    stray.hnsw().begin_restore(entry, top, 0);
    for id in hnsw.member_ids() {
        if id != representative {
            stray.hnsw().restore_saved(id, hnsw.saved_form(id).unwrap());
        }
    }
    let listed: Vec<NodeId> = hnsw
        .member_ids()
        .into_iter()
        .filter(|&id| id == representative || entry == Some(id))
        .collect();
    let (chunks, _) = written(&stray, &listed, 1 << 20);
    refuse(
        "a stray alias",
        &chunks,
        Build::Plain,
        &format!("alias {}", alias.as_u64()),
    );
}

#[test]
fn the_list_stream_round_trips_and_refuses_what_breaks_it() {
    for listed in [
        vec![],
        vec![NodeId::new(0)],
        vec![NodeId::new(3), NodeId::new(19), NodeId::new(u64::MAX)],
    ] {
        assert_eq!(decode_list(&encode_list(&listed)).unwrap(), listed);
    }
    let twice = [2u64.to_le_bytes().to_vec(), vec![5, 0]].concat();
    let trailing = [encode_list(&[NodeId::new(3)]), vec![1]].concat();
    let short = [3u64.to_le_bytes().to_vec(), vec![5, 1]].concat();
    for (case, bytes) in [
        ("twice", twice),
        ("trailing", trailing),
        ("short", short),
        ("no count", vec![1, 2]),
    ] {
        assert!(
            matches!(decode_list(&bytes), Err(Error::Corruption(_))),
            "{case}"
        );
    }
}

#[test]
fn the_quantizer_stream_round_trips_and_refuses_what_breaks_it() {
    for build in every_build().into_iter().skip(2) {
        let (index, _) = built(build);
        let kind = quantizer_kind(&index).unwrap();
        let saved = index.as_quantized().unwrap().saved_quantizer();
        assert_eq!(
            decode_quantizer(&encode_quantizer(kind, &saved).unwrap(), kind).unwrap(),
            saved,
            "{build:?}"
        );
        let bytes = encode_quantizer(kind, &saved).unwrap();
        assert!(
            decode_quantizer(&bytes, if kind == 1 { 3 } else { 1 }).is_err(),
            "{build:?}: another kind"
        );
        assert!(
            decode_quantizer(&bytes[..bytes.len() - 1], kind).is_err() || bytes.len() == 2,
            "{build:?}: short"
        );
        assert!(
            decode_quantizer(&[bytes.clone(), vec![0]].concat(), kind).is_err(),
            "{build:?}: trailing"
        );
    }
    let mut claims = vec![3, 1];
    for count in [u32::MAX, u32::MAX, u32::MAX] {
        claims.extend_from_slice(&count.to_le_bytes());
    }
    assert!(
        matches!(decode_quantizer(&claims, 3), Err(Error::Corruption(_))),
        "values past the stream"
    );
    let mut parts = vec![3, 1];
    for count in [2u32, 257, 1] {
        parts.extend_from_slice(&count.to_le_bytes());
    }
    parts.extend(std::iter::repeat_n(0, 2 * 257 * 4));
    assert!(
        matches!(decode_quantizer(&parts, 3), Err(Error::Corruption(_))),
        "257 centroids"
    );
}

#[test]
fn the_metadata_round_trips_and_refuses_what_breaks_it() {
    let (index, _) = built(Build::Plain);
    let (chunks, meta) = written(&index, &[], 1 << 20);
    let read = read_meta(&chunks).unwrap();
    assert_eq!(read.indexes, vec![meta.clone()]);
    assert_eq!((read.epoch, read.next_column), (19, 7));

    let with_meta = |edit: &dyn Fn(&mut SavedMeta)| -> Chunks {
        let mut edited = chunks.clone();
        let mut changed = read.clone();
        edit(&mut changed);
        let last = edited.metas.len() - 1;
        edited.bytes[last] = Bytes::from(encode_meta(&changed).unwrap());
        edited
    };
    let second = SavedIndexMeta {
        column: 3,
        ..meta.clone()
    };
    let cases: Vec<(&str, Chunks)> = vec![
        ("another layout", with_meta(&|meta| meta.layout = 2)),
        (
            "out of order",
            with_meta(&|meta| meta.indexes.push(second.clone())),
        ),
        (
            "a column given twice",
            with_meta(&|meta| {
                meta.indexes[0].quantizer_column = Some(5);
            }),
        ),
        (
            "a column at the next column",
            with_meta(&|meta| meta.next_column = 5),
        ),
        ("not last", {
            let mut moved = chunks.clone();
            let last = moved.metas.len() - 1;
            moved.metas.swap(0, last);
            moved.bytes.swap(0, last);
            moved
        }),
        ("a chunk of no index", {
            let mut foreign = chunks.clone();
            foreign.metas.insert(0, ChunkMeta::column(2, 4, 0, 1, 0));
            foreign.bytes.insert(0, Bytes::from_static(&[0, 0, 0, 0]));
            foreign
        }),
        ("empty", Chunks::default()),
        ("bytes after the metadata", {
            let mut longer = chunks.clone();
            let last = longer.metas.len() - 1;
            longer.bytes[last] = Bytes::from([longer.bytes[last].to_vec(), vec![0]].concat());
            longer
        }),
    ];
    for (case, chunks) in cases {
        assert!(
            matches!(read_meta(&chunks), Err(Error::Corruption(_))),
            "{case}"
        );
    }
}

/// A metadata chunk in the shape of the unreleased first draft, which
/// recorded each index's level draws after its entry point, is refused, for
/// one index and for two, whatever the draws.
#[test]
fn metadata_with_level_draws_is_refused() {
    #[derive(serde::Serialize)]
    struct DraftIndex {
        graph_id: u32,
        column: u32,
        quantizer_column: Option<u32>,
        definition: SavedDefinition,
        members: u64,
        entry: Option<(u64, u32)>,
        level_draws: u64,
    }
    #[derive(serde::Serialize)]
    struct Draft {
        layout: u8,
        epoch: u64,
        max_bytes: u32,
        next_column: u32,
        indexes: Vec<DraftIndex>,
    }
    let (index, _) = built(Build::Plain);
    let (chunks, _) = written(&index, &[], 1 << 20);
    let read = read_meta(&chunks).unwrap();
    let draft = |meta: &SavedIndexMeta, level_draws: u64| DraftIndex {
        graph_id: meta.graph_id,
        column: meta.column,
        quantizer_column: meta.quantizer_column,
        definition: meta.definition.clone(),
        members: meta.members,
        entry: meta.entry,
        level_draws,
    };
    let second = SavedIndexMeta {
        column: 3,
        ..read.indexes[0].clone()
    };
    for (indexes, level_draws) in [
        (vec![read.indexes[0].clone()], 0),
        (vec![read.indexes[0].clone()], 300),
        (vec![read.indexes[0].clone()], u64::MAX),
        (vec![second.clone(), read.indexes[0].clone()], 300),
    ] {
        let bytes = bincode::serde::encode_to_vec(
            Draft {
                layout: read.layout,
                epoch: read.epoch,
                max_bytes: read.max_bytes,
                next_column: read.next_column,
                indexes: indexes
                    .iter()
                    .map(|meta| draft(meta, level_draws))
                    .collect(),
            },
            bincode::config::standard(),
        )
        .unwrap();
        let mut drafted = chunks.clone();
        let last = drafted.metas.len() - 1;
        drafted.bytes[last] = Bytes::from(bytes);
        let error = read_meta(&drafted).unwrap_err();
        assert!(
            matches!(error, Error::Corruption(_)),
            "{} indexes, {level_draws} draws: {error}",
            indexes.len()
        );
    }
}
