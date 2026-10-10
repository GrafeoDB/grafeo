//! Vector Store section serializer for the `.grafeo` container format.
//!
//! Serializes HNSW topology (neighbor graphs) for all vector indexes.
//! Embeddings are not stored here: they live in LPG node properties and
//! are accessed via `VectorAccessor` during search.
//!
//! Persisting the topology eliminates the O(N log N) HNSW rebuild on
//! database open. For 1M vectors this saves 30-60 seconds of startup time.
//!
//! ## Section version 3: streams (written by checkpoints)
//!
//! [`Section::write_to`] writes a metadata chunk, then one byte stream per
//! HNSW index ([`ChunkKind::Stream`] pieces of graph 0, cut at the byte
//! cap), so a checkpoint holds one piece and one node's bytes at a time (and
//! references to an in-memory topology's nodes, sorted by id), never a whole
//! topology's bytes:
//!
//! - The metadata chunk is the bincode of `VectorMeta`: the layout byte
//!   (1), the byte cap the pieces were cut with, and the indexes in key order,
//!   each with its dimensions, metric byte (0 cosine, 1 Euclidean, 2 dot
//!   product, 3 Manhattan), `m` and `ef_construction`.
//! - Stream `i` holds the topology of index `i` of the metadata:
//!   `[has_entry_point u8][entry_point u64][max_level u32][node_count u64]`,
//!   then `node_count` records `[id u64][level_count u32]` followed, for each
//!   level, by `[neighbor_count u32]` and that many `[neighbor u64]`, all
//!   little-endian, ids strictly increasing. There is an entry point
//!   exactly when there are nodes; the entry point is a node with exactly
//!   `max_level + 1` levels, and every node has 1 to `max_level + 1` levels.
//!   Every neighbor listed at a level is a node of the stream with that
//!   level, and no node lists itself.
//!
//! Quantized indexes are not written: a load builds them from the data,
//! because the section holds no quantized codes.
//!
//! [`Section::read_from`] restores each index it was given from the stream
//! with its key, node by node, and skips the streams of indexes it was not
//! given. A stream that ends early, runs on past its last node or breaks
//! the rules above is refused, naming the index, which is then left empty.
//!
//! ## 0.5.x formats (one raw chunk)
//!
//! A 0.5.x file holds the section as one raw chunk, which `read_from` hands
//! to [`Section::deserialize`]. It reads both formats 0.5.x wrote:
//!
//! - **v2 paged:** packed envelope (`GVST` magic, envelope version 2, index
//!   directory, per-index meta, per-index `GTOP` paged topology). Reads
//!   parse the directory and feed each topology blob into
//!   [`super::paged_topology::deserialize_topology`].
//!   [`Section::serialize`] still writes this envelope; no checkpoint or
//!   spill calls it (tests and the builders of 0.5.x fixtures do).
//! - **v1 bincode (legacy):** detected by the absence of the `GVST` magic
//!   at offset 0.
//!
//! The next checkpoint after reading either writes version 3. The 0.5.x
//! readers do not check the stream rules of version 3: a 0.5.x topology
//! that breaks them (a top level its entry point does not reach, say) loads
//! as it is, and once its first 0.6 checkpoint has written it as version 3,
//! the next open refuses that stream with a warning and builds the index
//! from the data, once: the checkpoint after that writes a topology that
//! keeps the rules.

use std::io::{self, Read, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bytes::Bytes;
use serde::{Deserialize, Serialize};

use grafeo_common::storage::section::{
    ChunkKind, ChunkMeta, Section, SectionSink, SectionSource, SectionType, check_version,
    legacy_bytes,
};
use grafeo_common::storage::{ChunkCaps, ChunkStreamReader, ChunkStreamWriter, stream_error};
use grafeo_common::types::NodeId;
use grafeo_common::utils::error::{Error, Result};

use super::paged_topology::{deserialize_topology, serialize_topology};
use super::{DistanceMetric, HnswConfig, TopologyVisitor, VectorIndexKind};

/// The section version this build writes: 1 was bincode, 2 the paged
/// envelope of 0.5.x, 3 the streams. A source of version 3 is read as
/// streams, a single raw chunk of any version as 0.5.x bytes.
pub(crate) const VECTOR_SECTION_VERSION: u8 = 3;

/// The version byte inside the 0.5.x `GVST` envelope, which
/// [`Section::serialize`] writes (for tests and 0.5.x fixtures) and
/// [`Section::deserialize`] reads.
const ENVELOPE_VERSION: u8 = 2;

/// The layout byte of the metadata chunk this build writes and reads.
const META_LAYOUT: u8 = 1;

/// The most bytes a metadata chunk's decoding may claim, so a corrupt length
/// prefix fails the decode instead of requesting a huge allocation. Far
/// above the metadata of any real set of indexes (about 40 bytes plus the
/// key per index).
const META_DECODE_LIMIT: usize = 1 << 24;

/// The most list entries a read allocates for up front: a count read from
/// a stream is not trusted with an allocation, the lists grow as entries
/// arrive.
const PREALLOCATE_AT_MOST: usize = 64;

/// An empty list for `count` entries read from a stream, with room for at
/// most [`PREALLOCATE_AT_MOST`] of them up front.
fn list_with_room_for<T>(count: usize) -> Vec<T> {
    Vec::with_capacity(count.min(PREALLOCATE_AT_MOST))
}

/// What the metadata chunk of version 3 holds.
#[derive(Serialize, Deserialize)]
struct VectorMeta {
    /// [`META_LAYOUT`].
    layout: u8,
    /// The byte cap the streams were cut with (readers accept any).
    max_bytes: u32,
    /// The indexes in strictly increasing key order; index `i` is stream `i`.
    indexes: Vec<VectorIndexMeta>,
}

/// One index in the metadata chunk.
#[derive(Serialize, Deserialize)]
struct VectorIndexMeta {
    /// "label:property".
    key: String,
    dimensions: u64,
    /// 0 cosine, 1 Euclidean, 2 dot product, 3 Manhattan.
    metric: u8,
    m: u64,
    ef_construction: u64,
}

/// First 4 bytes of the v2 envelope; absent in v1 bincode output.
const V2_MAGIC: &[u8; 4] = b"GVST";

/// v2 envelope header size (magic + version + reserved + num_indexes).
const V2_HEADER_SIZE: usize = 16;

/// v2 directory entry size (meta_offset + meta_len + topology_offset + topology_len).
const V2_DIR_ENTRY_SIZE: usize = 32;

// ── v1 (legacy) snapshot types ─────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct VectorStoreSnapshotV1 {
    version: u8,
    indexes: Vec<IndexSnapshotV1>,
}

#[derive(Serialize, Deserialize)]
struct IndexSnapshotV1 {
    /// Index key: "label:property"
    key: String,
    /// HNSW configuration
    dimensions: usize,
    metric: DistanceMetric,
    m: usize,
    ef_construction: usize,
    /// Topology
    entry_point: Option<NodeId>,
    max_level: usize,
    /// Node neighbors: Vec<(NodeId, Vec<Vec<NodeId>>)>
    nodes: Vec<(NodeId, Vec<Vec<NodeId>>)>,
}

// ── v2 (0.5.x) per-index metadata ──────────────────────────────────

#[derive(Serialize, Deserialize)]
struct IndexMetaV2 {
    /// Index key: "label:property"
    key: String,
    dimensions: usize,
    metric: DistanceMetric,
    m: usize,
    ef_construction: usize,
}

// ── Section implementation ──────────────────────────────────────────

/// Vector Store section for the `.grafeo` container.
///
/// Wraps a collection of `(key, Arc<VectorIndexKind>)` pairs and serializes
/// their HNSW topologies for persistence.
pub struct VectorStoreSection {
    /// Vector indexes: (key, index) pairs from LpgStore::vector_index_entries()
    indexes: Vec<(String, Arc<VectorIndexKind>)>,
    /// The caps [`Section::write_to`] cuts the streams with, taken when the
    /// section is built.
    caps: ChunkCaps,
    dirty: AtomicBool,
}

impl VectorStoreSection {
    /// Create a new Vector Store section from the current indexes, with this
    /// thread's chunk caps ([`ChunkCaps::current`]).
    pub fn new(indexes: Vec<(String, Arc<VectorIndexKind>)>) -> Self {
        Self {
            indexes,
            caps: ChunkCaps::current(),
            dirty: AtomicBool::new(false),
        }
    }

    /// Mark this section as dirty.
    pub fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }
}

/// Serializes all in-memory indexes to the v2 paged envelope.
///
/// Layout: 16-byte header (`GVST` magic, version, num_indexes) + index
/// directory (32 bytes/entry: meta_offset, meta_len, topology_offset,
/// topology_len) + bincode'd metadata blobs + per-index `GTOP` paged
/// topology blobs.
fn serialize_v2(indexes: &[(String, Arc<VectorIndexKind>)]) -> Result<Vec<u8>> {
    let bincode_config = bincode::config::standard();

    // Build per-index meta blobs and topology blobs upfront so we can
    // compute absolute offsets.
    let mut meta_blobs: Vec<Vec<u8>> = Vec::with_capacity(indexes.len());
    let mut topology_blobs: Vec<Vec<u8>> = Vec::with_capacity(indexes.len());

    for (key, index) in indexes {
        let config = index.config();
        let meta = IndexMetaV2 {
            key: key.clone(),
            dimensions: config.dimensions,
            metric: config.metric,
            m: config.m,
            ef_construction: config.ef_construction,
        };
        let meta_bytes = bincode::serde::encode_to_vec(&meta, bincode_config).map_err(|e| {
            Error::Internal(format!("Vector Store v2 meta serialization failed: {e}"))
        })?;
        meta_blobs.push(meta_bytes);

        let (entry_point, max_level, nodes) = index.snapshot_topology();
        let topology_bytes = serialize_topology(entry_point, max_level, &nodes);
        topology_blobs.push(topology_bytes);
    }

    let n = indexes.len();
    let header_size = V2_HEADER_SIZE;
    let dir_size = n * V2_DIR_ENTRY_SIZE;
    let body_start = header_size + dir_size;

    // Compute absolute offsets for each meta + topology blob.
    let mut meta_offsets: Vec<u64> = Vec::with_capacity(n);
    let mut topology_offsets: Vec<u64> = Vec::with_capacity(n);
    let mut cursor = body_start;
    for blob in &meta_blobs {
        meta_offsets.push(cursor as u64);
        cursor += blob.len();
    }
    for blob in &topology_blobs {
        topology_offsets.push(cursor as u64);
        cursor += blob.len();
    }

    let mut buf = Vec::with_capacity(cursor);

    // Header
    buf.extend_from_slice(V2_MAGIC);
    buf.push(ENVELOPE_VERSION);
    buf.extend_from_slice(&[0u8; 3]);
    buf.extend_from_slice(&(n as u64).to_le_bytes());
    debug_assert_eq!(buf.len(), V2_HEADER_SIZE);

    // Directory
    for i in 0..n {
        buf.extend_from_slice(&meta_offsets[i].to_le_bytes());
        buf.extend_from_slice(&(meta_blobs[i].len() as u64).to_le_bytes());
        buf.extend_from_slice(&topology_offsets[i].to_le_bytes());
        buf.extend_from_slice(&(topology_blobs[i].len() as u64).to_le_bytes());
    }
    debug_assert_eq!(buf.len(), header_size + dir_size);

    // Body: meta blobs first, then topology blobs (matches the offsets
    // computed above).
    for blob in &meta_blobs {
        buf.extend_from_slice(blob);
    }
    for blob in &topology_blobs {
        buf.extend_from_slice(blob);
    }

    Ok(buf)
}

/// Restores indexes from a v2 paged envelope.
fn deserialize_v2(data: &[u8], indexes: &mut [(String, Arc<VectorIndexKind>)]) -> Result<()> {
    let bincode_config = bincode::config::standard();

    if data.len() < V2_HEADER_SIZE {
        return Err(Error::corruption("Vector Store v2 header truncated"));
    }
    if &data[0..4] != V2_MAGIC {
        return Err(Error::corruption("Vector Store v2 bad magic"));
    }
    let version = data[4];
    if version != ENVELOPE_VERSION {
        return Err(Error::Serialization(format!(
            "Vector Store v2 unsupported version: {version}"
        )));
    }
    let n_u64 = u64::from_le_bytes(
        data[8..16]
            .try_into()
            .expect("slice length 8 fits u64 array"),
    );
    let n = usize::try_from(n_u64).map_err(|_| Error::corruption("v2 n_indexes overflow"))?;

    let dir_size = n
        .checked_mul(V2_DIR_ENTRY_SIZE)
        .ok_or_else(|| Error::corruption("v2 directory size overflow"))?;
    let body_start = V2_HEADER_SIZE
        .checked_add(dir_size)
        .ok_or_else(|| Error::corruption("v2 directory size overflow"))?;
    if data.len() < body_start {
        return Err(Error::corruption(format!(
            "Vector Store v2 directory truncated: expected {body_start} bytes, got {}",
            data.len()
        )));
    }

    for i in 0..n {
        let dir_off = V2_HEADER_SIZE + i * V2_DIR_ENTRY_SIZE;
        let meta_off = u64::from_le_bytes(
            data[dir_off..dir_off + 8]
                .try_into()
                .expect("slice length 8 fits u64 array"),
        );
        let meta_len = u64::from_le_bytes(
            data[dir_off + 8..dir_off + 16]
                .try_into()
                .expect("slice length 8 fits u64 array"),
        );
        let topology_off = u64::from_le_bytes(
            data[dir_off + 16..dir_off + 24]
                .try_into()
                .expect("slice length 8 fits u64 array"),
        );
        let topology_len = u64::from_le_bytes(
            data[dir_off + 24..dir_off + 32]
                .try_into()
                .expect("slice length 8 fits u64 array"),
        );

        let meta_off_usize =
            usize::try_from(meta_off).map_err(|_| Error::corruption("v2 meta_off overflow"))?;
        let meta_len_usize =
            usize::try_from(meta_len).map_err(|_| Error::corruption("v2 meta_len overflow"))?;
        let topology_off_usize = usize::try_from(topology_off)
            .map_err(|_| Error::corruption("v2 topology_off overflow"))?;
        let topology_len_usize = usize::try_from(topology_len)
            .map_err(|_| Error::corruption("v2 topology_len overflow"))?;

        let meta_end = meta_off_usize
            .checked_add(meta_len_usize)
            .ok_or_else(|| Error::corruption("v2 meta range overflow"))?;
        let topology_end = topology_off_usize
            .checked_add(topology_len_usize)
            .ok_or_else(|| Error::corruption("v2 topology range overflow"))?;
        if meta_end > data.len() || topology_end > data.len() {
            return Err(Error::corruption(format!(
                "Vector Store v2 directory entry {i} out of range"
            )));
        }

        let meta_bytes = &data[meta_off_usize..meta_end];
        let (meta, _): (IndexMetaV2, _) =
            bincode::serde::decode_from_slice(meta_bytes, bincode_config).map_err(|e| {
                Error::corruption(format!("Vector Store v2 meta deserialization failed: {e}"))
            })?;

        // Find the matching index by key. v2 doesn't require ordering;
        // the section receives indexes in any order, so we look up by key.
        if let Some((_, index)) = indexes.iter().find(|(k, _)| *k == meta.key) {
            // Copy the topology bytes into a Bytes so the paged decoder
            // can hold them. Phase 7c will Bytes::from_owner the section
            // mmap directly and slice without copying.
            let topology_bytes = Bytes::copy_from_slice(&data[topology_off_usize..topology_end]);
            let (entry_point, max_level, nodes) =
                deserialize_topology(topology_bytes).map_err(|e| {
                    Error::corruption(format!(
                        "Vector Store v2 topology decode failed for key '{}': {e}",
                        meta.key
                    ))
                })?;
            index.restore_topology(entry_point, max_level, nodes);
        }
    }

    Ok(())
}

/// Restores indexes from a v1 bincode envelope (legacy fallback).
fn deserialize_v1(data: &[u8], indexes: &mut [(String, Arc<VectorIndexKind>)]) -> Result<()> {
    let config = bincode::config::standard();
    let (snapshot, _): (VectorStoreSnapshotV1, _) = bincode::serde::decode_from_slice(data, config)
        .map_err(|e| Error::corruption(format!("Vector Store v1 deserialization failed: {e}")))?;

    for idx_snap in snapshot.indexes {
        if let Some((_, index)) = indexes.iter().find(|(k, _)| *k == idx_snap.key) {
            index.restore_topology(idx_snap.entry_point, idx_snap.max_level, idx_snap.nodes);
        }
    }
    Ok(())
}

// ── Version 3: streams ──────────────────────────────────────────────

/// The metric byte of the metadata chunk.
fn metric_byte(metric: DistanceMetric) -> u8 {
    // No wildcard: a new metric must be given a byte here.
    match metric {
        DistanceMetric::Cosine => 0,
        DistanceMetric::Euclidean => 1,
        DistanceMetric::DotProduct => 2,
        DistanceMetric::Manhattan => 3,
    }
}

/// The metric of a metadata chunk's metric byte.
fn metric_of(byte: u8, key: &str) -> Result<DistanceMetric> {
    match byte {
        0 => Ok(DistanceMetric::Cosine),
        1 => Ok(DistanceMetric::Euclidean),
        2 => Ok(DistanceMetric::DotProduct),
        3 => Ok(DistanceMetric::Manhattan),
        other => Err(Error::corruption(format!(
            "section VectorStore: vector index '{key}' has metric {other}, which is none of 0 \
             (cosine), 1 (Euclidean), 2 (dot product) and 3 (Manhattan)"
        ))),
    }
}

/// `error` with the index whose stream it met named after its message, its
/// variant kept.
fn in_index(error: Error, key: &str, stream: usize) -> Error {
    let place = format!("in the topology of vector index '{key}' (stream {stream})");
    match error {
        Error::Serialization(message) => Error::Serialization(format!("{message}, {place}")),
        Error::Corruption(mut corruption) => {
            corruption.what = format!("{}, {place}", corruption.what);
            Error::Corruption(corruption)
        }
        Error::Internal(message) => Error::Internal(format!("{message}, {place}")),
        Error::InvalidValue(message) => Error::InvalidValue(format!("{message}, {place}")),
        Error::Io(inner) => Error::Io(io::Error::new(inner.kind(), format!("{inner}, {place}"))),
        other => other,
    }
}

/// The metadata chunk of `indexes`, which come in key order.
fn encode_meta(indexes: &[&(String, Arc<VectorIndexKind>)], caps: ChunkCaps) -> Result<Vec<u8>> {
    let meta = VectorMeta {
        layout: META_LAYOUT,
        max_bytes: caps.max_bytes,
        indexes: indexes
            .iter()
            .map(|(key, index)| {
                let config = index.config();
                VectorIndexMeta {
                    key: key.clone(),
                    dimensions: config.dimensions as u64,
                    metric: metric_byte(config.metric),
                    m: config.m as u64,
                    ef_construction: config.ef_construction as u64,
                }
            })
            .collect(),
    };
    bincode::serde::encode_to_vec(&meta, bincode::config::standard()).map_err(|error| {
        Error::Serialization(format!(
            "section VectorStore: the metadata chunk cannot be encoded: {error}"
        ))
    })
}

/// Writes a topology into its stream as [`TopologyVisitor`] hands it over,
/// one node's bytes at a time.
struct TopologyWriter<'s> {
    stream: ChunkStreamWriter<'s>,
    /// The bytes of the header or of one node, reused from node to node.
    scratch: Vec<u8>,
    /// Whether a write into `stream` failed: the stream then keeps the sink's
    /// own error, which `finish` returns.
    write_failed: bool,
}

impl TopologyWriter<'_> {
    /// Writes `scratch` into the stream.
    fn write_scratch(&mut self) -> Result<()> {
        self.stream.write_all(&self.scratch).map_err(|error| {
            self.write_failed = true;
            stream_error(SectionType::VectorStore, error)
        })
    }
}

/// `count` as the u32 the stream stores, or an error naming `what`.
fn count_u32(count: usize, what: &str) -> Result<u32> {
    u32::try_from(count).map_err(|_| {
        Error::Serialization(format!(
            "section VectorStore: {what} {count} does not fit the 32 bits the stream stores"
        ))
    })
}

impl TopologyVisitor for TopologyWriter<'_> {
    fn header(
        &mut self,
        entry_point: Option<NodeId>,
        max_level: usize,
        node_count: usize,
    ) -> Result<()> {
        let max_level = count_u32(max_level, "a top level of")?;
        self.scratch.clear();
        self.scratch.push(u8::from(entry_point.is_some()));
        self.scratch
            .extend_from_slice(&entry_point.map_or(0u64, |id| id.as_u64()).to_le_bytes());
        self.scratch.extend_from_slice(&max_level.to_le_bytes());
        self.scratch
            .extend_from_slice(&(node_count as u64).to_le_bytes());
        self.write_scratch()
    }

    fn node(&mut self, id: NodeId, layers: &[Vec<NodeId>]) -> Result<()> {
        self.scratch.clear();
        self.scratch.extend_from_slice(&id.as_u64().to_le_bytes());
        self.scratch
            .extend_from_slice(&count_u32(layers.len(), "a level count of")?.to_le_bytes());
        for layer in layers {
            self.scratch
                .extend_from_slice(&count_u32(layer.len(), "a neighbor count of")?.to_le_bytes());
            for neighbor in layer {
                self.scratch
                    .extend_from_slice(&neighbor.as_u64().to_le_bytes());
            }
        }
        self.write_scratch()
    }
}

/// Writes the topology of `index` as stream `stream` of graph 0.
fn write_topology(
    sink: &mut dyn SectionSink,
    stream: u32,
    caps: ChunkCaps,
    index: &VectorIndexKind,
) -> Result<()> {
    let mut writer = TopologyWriter {
        stream: ChunkStreamWriter::new(sink, 0, stream, caps),
        scratch: Vec::new(),
        write_failed: false,
    };
    match index.visit_topology(&mut writer) {
        Ok(()) => writer.stream.finish().map(drop),
        // The stream keeps the sink's error: return it, not its text.
        Err(error) if writer.write_failed => Err(writer.stream.finish().err().unwrap_or(error)),
        Err(error) => Err(error),
    }
}

/// The metadata chunk of a version 3 source, after checking that it comes
/// first, that its keys increase and its metric bytes are known, and that
/// every other chunk is a piece of a stream it lists.
fn read_meta(source: &dyn SectionSource) -> Result<VectorMeta> {
    let chunks = source.chunks();
    match chunks.first() {
        Some(first) if *first == ChunkMeta::meta() => {}
        Some(first) if first.kind == ChunkKind::Meta => {
            return Err(Error::corruption(format!(
                "section VectorStore: the metadata chunk has codec {}, graph {}, column {}, \
                 first row {} and rows {}, where all are 0",
                first.codec, first.graph_id, first.column_id, first.row_start, first.row_count
            )));
        }
        Some(first) => {
            return Err(Error::corruption(format!(
                "section VectorStore: the first chunk is of kind {:?}, where the metadata chunk \
                 comes first",
                first.kind
            )));
        }
        None => {
            return Err(Error::corruption(
                "section VectorStore: the section holds no chunk, not even its metadata chunk"
                    .to_string(),
            ));
        }
    }
    let bytes = source.fetch(0)?;
    let (meta, read): (VectorMeta, usize) = bincode::serde::decode_from_slice(
        &bytes,
        bincode::config::standard().with_limit::<META_DECODE_LIMIT>(),
    )
    .map_err(|error| {
        Error::corruption(format!(
            "section VectorStore: the metadata chunk does not decode: {error}"
        ))
    })?;
    if read != bytes.len() {
        return Err(Error::corruption(format!(
            "section VectorStore: the metadata chunk holds {} bytes after its {read} bytes of \
             metadata",
            bytes.len() - read
        )));
    }
    if meta.layout != META_LAYOUT {
        return Err(Error::corruption(format!(
            "section VectorStore: the metadata chunk has layout {}, this build reads layout \
             {META_LAYOUT}",
            meta.layout
        )));
    }
    for pair in meta.indexes.windows(2) {
        if pair[0].key >= pair[1].key {
            return Err(Error::corruption(format!(
                "section VectorStore: the metadata lists vector index '{}' after '{}', but keys \
                 are strictly increasing",
                pair[1].key, pair[0].key
            )));
        }
    }
    for index in &meta.indexes {
        metric_of(index.metric, &index.key)?;
    }
    for chunk in &chunks[1..] {
        let listed =
            usize::try_from(chunk.column_id).is_ok_and(|stream| stream < meta.indexes.len());
        if chunk.kind != ChunkKind::Stream || chunk.graph_id != 0 || !listed {
            return Err(Error::corruption(format!(
                "section VectorStore: a chunk of kind {:?} for graph {}, stream {}, where only \
                 pieces of the {} streams of graph 0 (one per index) follow the metadata chunk",
                chunk.kind,
                chunk.graph_id,
                chunk.column_id,
                meta.indexes.len()
            )));
        }
    }
    Ok(meta)
}

/// Refuses a topology for another number of dimensions or another metric
/// than `config`'s: its neighbors were chosen by other distances.
fn check_config(meta: &VectorIndexMeta, config: &HnswConfig) -> Result<()> {
    let metric = metric_of(meta.metric, &meta.key)?;
    if meta.dimensions != config.dimensions as u64 || metric != config.metric {
        return Err(Error::corruption(format!(
            "section VectorStore: vector index '{}' was written with {} dimensions and metric \
             {metric:?}, the index to restore has {} dimensions and metric {:?}",
            meta.key, meta.dimensions, config.dimensions, config.metric
        )));
    }
    Ok(())
}

/// Reads one topology stream, fixed-size fields at a time.
struct TopologyReader<'s> {
    stream: ChunkStreamReader<'s>,
}

impl TopologyReader<'_> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut buffer = [0u8; N];
        self.stream
            .read_exact(&mut buffer)
            .map_err(|error| stream_error(SectionType::VectorStore, error))?;
        Ok(buffer)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }

    /// `count` (read from the stream) as a `usize`.
    fn usize(count: u64, what: &str) -> Result<usize> {
        usize::try_from(count).map_err(|_| {
            Error::Serialization(format!(
                "section VectorStore: {what} {count} does not fit this platform"
            ))
        })
    }

    /// Refuses a stream that holds bytes after its last node.
    fn expect_end(&mut self) -> Result<()> {
        let mut probe = [0u8; 1];
        match self.stream.read(&mut probe) {
            Ok(0) => Ok(()),
            Ok(_) => Err(Error::corruption(
                "section VectorStore: the stream holds bytes after its last node",
            )),
            Err(error) => Err(stream_error(SectionType::VectorStore, error)),
        }
    }

    /// Reads the topology into `index`, node by node, checking what the
    /// writer keeps: an entry point exactly when there are nodes, the entry
    /// point a node with exactly `max_level + 1` levels, and every node with
    /// 1 to `max_level + 1` levels (each check O(1) per node). Once every node
    /// is in, every neighbor must be a node with a list at the level it is
    /// listed at, and no node may list itself (one lookup per neighbor).
    fn restore(&mut self, index: &VectorIndexKind) -> Result<()> {
        let flag = self.bytes::<1>()?[0];
        let entry_value = self.u64()?;
        let top_level = self.u32()?;
        let node_count = self.u64()?;
        let entry_point = match flag {
            0 if node_count > 0 => {
                return Err(Error::corruption(format!(
                    "section VectorStore: the stream has no entry point but {node_count} \
                     nodes, where every topology with nodes has one"
                )));
            }
            0 if entry_value != 0 => {
                return Err(Error::corruption(format!(
                    "section VectorStore: the stream has no entry point but entry point value \
                     {entry_value}, where the writer writes 0"
                )));
            }
            0 => None,
            1 if node_count == 0 => {
                return Err(Error::corruption(format!(
                    "section VectorStore: the stream has entry point {entry_value} but no nodes"
                )));
            }
            1 => Some(NodeId::new(entry_value)),
            flag => {
                return Err(Error::corruption(format!(
                    "section VectorStore: the stream has entry point flag {flag}, which is \
                     neither 0 nor 1"
                )));
            }
        };
        // Every node has 1 to `levels` levels, the entry point exactly `levels`.
        let levels = u64::from(top_level) + 1;
        let max_level = Self::usize(u64::from(top_level), "a top level of")?;
        let node_count = Self::usize(node_count, "a node count of")?;
        index.begin_restore(entry_point, max_level, node_count);
        let mut previous: Option<NodeId> = None;
        let mut entry_levels: Option<u32> = None;
        for _ in 0..node_count {
            let id = NodeId::new(self.u64()?);
            if let Some(previous) = previous
                && id <= previous
            {
                return Err(Error::corruption(format!(
                    "section VectorStore: node {} follows node {}, but node ids are strictly \
                     increasing",
                    id.as_u64(),
                    previous.as_u64()
                )));
            }
            let level_count = self.u32()?;
            if level_count == 0 || u64::from(level_count) > levels {
                return Err(Error::corruption(format!(
                    "section VectorStore: node {} has {level_count} levels, where every node \
                     has 1 to {levels} (the top level {top_level} plus one)",
                    id.as_u64()
                )));
            }
            if Some(id) == entry_point {
                entry_levels = Some(level_count);
            }
            let level_count = Self::usize(u64::from(level_count), "a level count of")?;
            let mut layers = list_with_room_for(level_count);
            for _ in 0..level_count {
                let neighbor_count = Self::usize(u64::from(self.u32()?), "a neighbor count of")?;
                let mut neighbors = list_with_room_for(neighbor_count);
                for _ in 0..neighbor_count {
                    neighbors.push(NodeId::new(self.u64()?));
                }
                layers.push(neighbors);
            }
            index.restore_node(id, layers);
            previous = Some(id);
        }
        if let Some(entry_point) = entry_point {
            match entry_levels {
                None => {
                    return Err(Error::corruption(format!(
                        "section VectorStore: entry point {} is not a node of the stream",
                        entry_point.as_u64()
                    )));
                }
                Some(count) if u64::from(count) != levels => {
                    return Err(Error::corruption(format!(
                        "section VectorStore: entry point {} has {count} levels, where the top \
                         level {top_level} needs {levels}",
                        entry_point.as_u64()
                    )));
                }
                Some(_) => {}
            }
        }
        self.expect_end()?;
        check_links(index)
    }
}

/// Refuses a restored topology with a neighbor reference that breaks the
/// rules the index keeps (see [`VectorIndexKind::first_broken_link`]): a
/// search would meet a dead end there. Checked once every node is in, in
/// place: no memory beyond what the index holds.
fn check_links(index: &VectorIndexKind) -> Result<()> {
    let Some(link) = index.first_broken_link() else {
        return Ok(());
    };
    let (node, level, neighbor) = (link.node.as_u64(), link.level, link.neighbor.as_u64());
    let what = if neighbor == node {
        format!("node {node} lists itself at level {level}, but no node is its own neighbor")
    } else {
        match link.neighbor_levels {
            None => format!(
                "node {node} lists node {neighbor} at level {level}, but node {neighbor} is \
                 not a node of the stream"
            ),
            Some(levels) => format!(
                "node {node} lists node {neighbor} at level {level}, but node {neighbor} has \
                 {levels} levels"
            ),
        }
    };
    Err(Error::corruption(format!("section VectorStore: {what}")))
}

/// Restores `index` from stream `stream`; on an error, leaves it empty.
fn read_topology(source: &dyn SectionSource, stream: u32, index: &VectorIndexKind) -> Result<()> {
    let mut reader = TopologyReader {
        stream: ChunkStreamReader::new(source, 0, stream),
    };
    let restored = reader.restore(index);
    if restored.is_err() {
        // No half topology stays behind for a search to start from.
        index.begin_restore(None, 0, 0);
    }
    restored
}

impl Section for VectorStoreSection {
    fn section_type(&self) -> SectionType {
        SectionType::VectorStore
    }

    fn version(&self) -> u8 {
        VECTOR_SECTION_VERSION
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        serialize_v2(&self.indexes)
    }

    /// Writes the metadata chunk, then each HNSW index's topology as its own
    /// stream, in key order. Quantized indexes are left out: a load builds
    /// them from the data.
    ///
    /// Each index's read locks (nodes, entry point and level) are held while
    /// its topology is written to `sink` (see
    /// [`VectorIndexKind::visit_topology`]): inserts into that index wait
    /// until its stream is written, and so do searches that arrive after a
    /// waiting insert (`parking_lot`'s locks are fair). Memory: one piece and
    /// one node's bytes, and for a heap-backed index a sorted reference to
    /// each of its nodes (16 bytes per node, O(node count)).
    ///
    /// # Errors
    ///
    /// Returns the sink's error, or [`Error::Serialization`] when a count
    /// does not fit the 32 bits the stream stores it in.
    fn write_to(&self, sink: &mut dyn SectionSink) -> Result<()> {
        let mut indexes: Vec<&(String, Arc<VectorIndexKind>)> = self
            .indexes
            .iter()
            .filter(|(_, index)| matches!(**index, VectorIndexKind::Hnsw(_)))
            .collect();
        indexes.sort_by(|(left, _), (right, _)| left.cmp(right));
        sink.write_chunk(ChunkMeta::meta(), &encode_meta(&indexes, self.caps)?)?;
        for (position, (key, index)) in indexes.into_iter().enumerate() {
            let stream = u32::try_from(position).map_err(|_| {
                Error::Serialization(format!(
                    "section VectorStore: {position} vector indexes do not fit the 32-bit stream \
                     numbers"
                ))
            })?;
            write_topology(sink, stream, self.caps, index)
                .map_err(|error| in_index(error, key, position))?;
        }
        Ok(())
    }

    /// Restores each index this section was given from the stream with its
    /// key; a single raw chunk (0.5.x bytes) goes to
    /// [`deserialize`](Section::deserialize).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Serialization`] for a section of another version, and
    /// [`Error::Corruption`] for a metadata chunk or a chunk sequence this
    /// writer does not produce, an
    /// index of other dimensions or another metric, or a stream that does not
    /// hold its topology exactly (the index named, and left empty); any error
    /// from fetching a chunk.
    fn read_from(&mut self, source: &dyn SectionSource) -> Result<()> {
        let legacy = legacy_bytes(source).map_err(|error| match error {
            Error::Serialization(_) | Error::Corruption(_) => error.wrapped("section VectorStore"),
            other => other,
        })?;
        if let Some(bytes) = legacy {
            return self.deserialize(&bytes);
        }
        check_version(SectionType::VectorStore, source, VECTOR_SECTION_VERSION)?;
        let meta = read_meta(source)?;
        for (position, index_meta) in meta.indexes.iter().enumerate() {
            let Some((_, index)) = self.indexes.iter().find(|(key, _)| *key == index_meta.key)
            else {
                continue;
            };
            check_config(index_meta, index.config())?;
            let stream = u32::try_from(position).map_err(|_| {
                Error::corruption(format!(
                    "section VectorStore: the metadata lists {position} indexes, more than \
                     32-bit stream numbers reach"
                ))
            })?;
            read_topology(source, stream, index)
                .map_err(|error| in_index(error, &index_meta.key, position))?;
        }
        Ok(())
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        // Phase 7b: detect v2 packed vs v1 bincode by magic bytes.
        if data.len() >= 4 && &data[0..4] == V2_MAGIC {
            deserialize_v2(data, &mut self.indexes)
        } else {
            // v1 fallback: bincode-encoded VectorStoreSnapshotV1.
            // Existing files keep loading; the next checkpoint writes
            // them as version 3.
            deserialize_v1(data, &mut self.indexes)
        }
    }

    fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    fn mark_clean(&self) {
        self.dirty.store(false, Ordering::Release);
    }

    fn memory_usage(&self) -> usize {
        self.indexes
            .iter()
            .map(|(_, idx)| idx.heap_memory_bytes())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::vector::{HnswConfig, HnswIndex};

    fn make_test_index() -> (String, Arc<VectorIndexKind>) {
        let config = HnswConfig::new(4, DistanceMetric::Cosine);
        let index = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(config)));

        // Manually set up a small topology via snapshot/restore
        let nodes = vec![
            (NodeId::new(1), vec![vec![NodeId::new(2), NodeId::new(3)]]),
            (NodeId::new(2), vec![vec![NodeId::new(1), NodeId::new(3)]]),
            (NodeId::new(3), vec![vec![NodeId::new(1), NodeId::new(2)]]),
        ];
        index.restore_topology(Some(NodeId::new(1)), 0, nodes);

        ("Item:embedding".to_string(), index)
    }

    fn make_v1_snapshot_bytes(key: &str) -> Vec<u8> {
        // Encode a v1 bincode snapshot directly so we can prove the
        // legacy fallback path on real bytes.
        let snapshot = VectorStoreSnapshotV1 {
            version: 1,
            indexes: vec![IndexSnapshotV1 {
                key: key.to_string(),
                dimensions: 4,
                metric: DistanceMetric::Cosine,
                m: 16,
                ef_construction: 200,
                entry_point: Some(NodeId::new(1)),
                max_level: 0,
                nodes: vec![
                    (NodeId::new(1), vec![vec![NodeId::new(2), NodeId::new(3)]]),
                    (NodeId::new(2), vec![vec![NodeId::new(1), NodeId::new(3)]]),
                    (NodeId::new(3), vec![vec![NodeId::new(1), NodeId::new(2)]]),
                ],
            }],
        };
        bincode::serde::encode_to_vec(&snapshot, bincode::config::standard())
            .expect("v1 bincode encode")
    }

    #[test]
    fn vector_section_round_trip() {
        let (key, index) = make_test_index();
        let section = VectorStoreSection::new(vec![(key.clone(), Arc::clone(&index))]);

        let bytes = section.serialize().expect("serialize should succeed");
        assert!(!bytes.is_empty(), "bytes is empty");

        // Create a fresh index with same config to restore into
        let config = index.config().clone();
        let fresh_index = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(config)));
        let mut section2 = VectorStoreSection::new(vec![(key, fresh_index.clone())]);
        section2
            .deserialize(&bytes)
            .expect("deserialize should succeed");

        assert_eq!(fresh_index.len(), 3);
        let (ep, ml, nodes) = fresh_index.snapshot_topology();
        assert_eq!(ep, Some(NodeId::new(1)));
        assert_eq!(ml, 0);
        assert_eq!(nodes.len(), 3);
    }

    #[test]
    fn vector_section_empty() {
        let section = VectorStoreSection::new(vec![]);
        let bytes = section.serialize().expect("serialize should succeed");

        let mut section2 = VectorStoreSection::new(vec![]);
        section2
            .deserialize(&bytes)
            .expect("deserialize should succeed");
    }

    #[test]
    fn vector_section_type() {
        let section = VectorStoreSection::new(vec![]);
        assert_eq!(section.section_type(), SectionType::VectorStore);
        // 1 was bincode, 2 the paged envelope of 0.5.x, 3 the streams.
        assert_eq!(section.version(), 3);
    }

    #[test]
    fn vector_section_dirty_tracking() {
        let section = VectorStoreSection::new(vec![]);
        assert!(!section.is_dirty());
        section.mark_dirty();
        assert!(section.is_dirty());
        section.mark_clean();
        assert!(!section.is_dirty());
    }

    // ── Phase 7b: format detection + v1 → v2 migration ───────────────

    /// New writes produce a v2 buffer (starts with `GVST` magic).
    #[test]
    fn alix_section_serialize_writes_v2_magic() {
        let (key, index) = make_test_index();
        let section = VectorStoreSection::new(vec![(key, Arc::clone(&index))]);
        let bytes = section.serialize().expect("serialize should succeed");
        assert!(bytes.len() > 4);
        assert_eq!(&bytes[0..4], V2_MAGIC, "new writes must use v2 magic");
    }

    /// v1 bincode-encoded buffers still deserialize correctly. The
    /// check uses a directly-constructed v1 snapshot, guaranteeing the
    /// migration path works for files written by older Grafeo versions.
    #[test]
    fn gus_section_v1_bincode_buffer_still_loads() {
        let v1_bytes = make_v1_snapshot_bytes("Item:embedding");
        // Sanity: v1 bytes do NOT start with GVST.
        assert_ne!(
            &v1_bytes[0..4],
            V2_MAGIC,
            "v1 bincode must not have GVST magic"
        );

        let config = HnswConfig::new(4, DistanceMetric::Cosine);
        let fresh = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(config)));
        let mut section =
            VectorStoreSection::new(vec![("Item:embedding".to_string(), Arc::clone(&fresh))]);
        section
            .deserialize(&v1_bytes)
            .expect("v1 fallback path must load");

        assert_eq!(fresh.len(), 3);
        let (ep, ml, nodes) = fresh.snapshot_topology();
        assert_eq!(ep, Some(NodeId::new(1)));
        assert_eq!(ml, 0);
        assert_eq!(nodes.len(), 3);
    }

    /// After a v1 read + a re-serialize, the new buffer is v2.
    /// Demonstrates the on-checkpoint migration.
    #[test]
    fn vincent_section_v1_then_reserialize_yields_v2() {
        let v1_bytes = make_v1_snapshot_bytes("Item:embedding");
        let config = HnswConfig::new(4, DistanceMetric::Cosine);
        let fresh = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(config)));
        let mut section =
            VectorStoreSection::new(vec![("Item:embedding".to_string(), Arc::clone(&fresh))]);
        section.deserialize(&v1_bytes).expect("v1 load");

        // Re-serialize: now in v2.
        let v2_bytes = section.serialize().expect("v2 serialize");
        assert_eq!(&v2_bytes[0..4], V2_MAGIC, "post-migration write is v2");

        // And v2 round-trips cleanly.
        let config2 = HnswConfig::new(4, DistanceMetric::Cosine);
        let restored = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(config2)));
        let mut section2 =
            VectorStoreSection::new(vec![("Item:embedding".to_string(), Arc::clone(&restored))]);
        section2.deserialize(&v2_bytes).expect("v2 load");
        assert_eq!(restored.len(), 3);
    }

    /// v2 with multiple indexes round-trips by key, including indexes
    /// with different shapes.
    #[test]
    fn jules_section_v2_multiple_indexes_round_trip() {
        let cfg_a = HnswConfig::new(4, DistanceMetric::Cosine);
        let idx_a = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(cfg_a)));
        idx_a.restore_topology(
            Some(NodeId::new(10)),
            1,
            vec![
                (NodeId::new(10), vec![vec![NodeId::new(20)], vec![]]),
                (NodeId::new(20), vec![vec![NodeId::new(10)]]),
            ],
        );

        let cfg_b = HnswConfig::new(8, DistanceMetric::Euclidean);
        let idx_b = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(cfg_b)));
        idx_b.restore_topology(
            Some(NodeId::new(100)),
            0,
            vec![(NodeId::new(100), vec![vec![]])],
        );

        let section = VectorStoreSection::new(vec![
            ("Doc:embedding".to_string(), Arc::clone(&idx_a)),
            ("User:embedding".to_string(), Arc::clone(&idx_b)),
        ]);
        let bytes = section.serialize().expect("v2 serialize");

        // Restore into fresh indexes and verify topology counts and
        // entry points match.
        let restored_a = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
            4,
            DistanceMetric::Cosine,
        ))));
        let restored_b = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
            8,
            DistanceMetric::Euclidean,
        ))));
        let mut section2 = VectorStoreSection::new(vec![
            ("Doc:embedding".to_string(), Arc::clone(&restored_a)),
            ("User:embedding".to_string(), Arc::clone(&restored_b)),
        ]);
        section2.deserialize(&bytes).expect("v2 load");

        assert_eq!(restored_a.len(), 2);
        assert_eq!(restored_b.len(), 1);
        let (ep_a, _, _) = restored_a.snapshot_topology();
        let (ep_b, _, _) = restored_b.snapshot_topology();
        assert_eq!(ep_a, Some(NodeId::new(10)));
        assert_eq!(ep_b, Some(NodeId::new(100)));
    }

    /// Truncated v2 envelope is rejected without panicking.
    #[test]
    fn shosanna_section_truncated_v2_rejected() {
        let (key, index) = make_test_index();
        let section = VectorStoreSection::new(vec![(key.clone(), Arc::clone(&index))]);
        let bytes = section.serialize().expect("v2 serialize");

        // Truncate to less than the v2 header.
        let truncated = &bytes[..8];
        let fresh = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(
            index.config().clone(),
        )));
        let mut section2 = VectorStoreSection::new(vec![(key, fresh)]);
        let err = section2
            .deserialize(truncated)
            .expect_err("must reject truncated v2");
        match err {
            Error::Corruption(_) => {}
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    // ── Version 3: each topology in a stream ─────────────────────────

    use std::collections::HashMap;
    use std::io;

    use grafeo_common::storage::{
        ChunkCaps, ChunkKind, ChunkMeta, ImageSource, MemoryImage, SectionSink, SectionSource,
    };
    use grafeo_common::testing::chunk_caps::with_chunk_caps;

    use crate::index::vector::{QuantizationType, QuantizedHnswIndex};

    /// Pieces of at most 512 bytes, so every topology below takes several.
    const TINY: ChunkCaps = ChunkCaps {
        max_rows: 3,
        max_bytes: 512,
    };

    /// A section's chunks as written, for tests to change before reading
    /// them back.
    #[derive(Clone, Debug, PartialEq)]
    struct Written {
        version: u8,
        metas: Vec<ChunkMeta>,
        chunks: Vec<Bytes>,
    }

    impl Written {
        fn of(section: &VectorStoreSection) -> Self {
            let mut written = Self {
                version: section.version(),
                metas: Vec::new(),
                chunks: Vec::new(),
            };
            section.write_to(&mut written).unwrap();
            written
        }

        /// Positions of the pieces of `stream` of graph 0, in order.
        fn pieces(&self, stream: u32) -> Vec<usize> {
            self.metas
                .iter()
                .enumerate()
                .filter(|(_, meta)| {
                    meta.kind == ChunkKind::Stream && meta.graph_id == 0 && meta.column_id == stream
                })
                .map(|(position, _)| position)
                .collect()
        }

        fn remove(&mut self, position: usize) {
            self.metas.remove(position);
            self.chunks.remove(position);
        }

        fn push(&mut self, meta: ChunkMeta, bytes: &[u8]) {
            self.metas.push(meta);
            self.chunks.push(Bytes::copy_from_slice(bytes));
        }

        /// A copy whose metadata chunk is `meta` changed by `change`.
        fn with_meta(&self, change: impl FnOnce(&mut VectorMeta)) -> Self {
            let mut meta = decode_meta(&self.chunks[0]);
            change(&mut meta);
            let mut copy = self.clone();
            copy.chunks[0] = Bytes::from(
                bincode::serde::encode_to_vec(&meta, bincode::config::standard()).unwrap(),
            );
            copy
        }
    }

    impl SectionSink for Written {
        fn write_chunk(&mut self, meta: ChunkMeta, bytes: &[u8]) -> Result<()> {
            self.push(meta, bytes);
            Ok(())
        }
    }

    impl SectionSource for Written {
        fn chunks(&self) -> &[ChunkMeta] {
            &self.metas
        }

        fn fetch(&self, index: usize) -> Result<Bytes> {
            self.chunks
                .get(index)
                .cloned()
                .ok_or_else(|| Error::Internal(format!("no chunk {index}")))
        }

        fn stored_length(&self, index: usize) -> Result<u64> {
            self.chunks
                .get(index)
                .map(|bytes| bytes.len() as u64)
                .ok_or_else(|| Error::Internal(format!("no chunk {index}")))
        }

        fn section_version(&self) -> u8 {
            self.version
        }
    }

    fn decode_meta(bytes: &[u8]) -> VectorMeta {
        bincode::serde::decode_from_slice(bytes, bincode::config::standard())
            .unwrap()
            .0
    }

    /// The vector of node `i`: numbers built from 3, 19 and 88.
    fn vector_of(i: u64) -> Vec<f32> {
        vec![
            (i * 3 % 19) as f32,
            (i * 19 % 88) as f32,
            (i % 88) as f32 / 3.0,
        ]
    }

    /// "Doc:emb": a plain index of 200 vectors of 3 dimensions.
    fn doc_index() -> Arc<VectorIndexKind> {
        let index = HnswIndex::with_seed(HnswConfig::new(3, DistanceMetric::Euclidean), 3);
        let vectors: HashMap<NodeId, Arc<[f32]>> = (1..=200u64)
            .map(|i| (NodeId::new(i), vector_of(i).into()))
            .collect();
        let accessor = |id: NodeId| -> Option<Arc<[f32]>> { vectors.get(&id).cloned() };
        for i in 1..=200u64 {
            index.insert(NodeId::new(i), &vector_of(i), &accessor);
        }
        Arc::new(VectorIndexKind::Hnsw(index))
    }

    /// "Note:emb": a scalar-quantized index of 50 vectors.
    fn note_index() -> Arc<VectorIndexKind> {
        let index = QuantizedHnswIndex::with_seed(
            HnswConfig::new(3, DistanceMetric::Cosine),
            QuantizationType::Scalar,
            19,
        );
        for i in 1..=50u64 {
            index.insert(NodeId::new(i * 88), &vector_of(i));
        }
        Arc::new(VectorIndexKind::Quantized(index))
    }

    /// "Person:emb": a plain index of 88 vectors, cosine.
    fn person_index() -> Arc<VectorIndexKind> {
        let index = HnswIndex::with_seed(HnswConfig::new(3, DistanceMetric::Cosine), 88);
        let vectors: HashMap<NodeId, Arc<[f32]>> = (1..=88u64)
            .map(|i| (NodeId::new(i * 3), vector_of(i + 19).into()))
            .collect();
        let accessor = |id: NodeId| -> Option<Arc<[f32]>> { vectors.get(&id).cloned() };
        for i in 1..=88u64 {
            index.insert(NodeId::new(i * 3), &vector_of(i + 19), &accessor);
        }
        Arc::new(VectorIndexKind::Hnsw(index))
    }

    /// "Note:emb" (quantized), "Doc:emb" and "Person:emb": the section writes
    /// the two plain ones, in key order, as streams 0 and 1.
    fn test_indexes() -> Vec<(String, Arc<VectorIndexKind>)> {
        vec![
            ("Note:emb".to_string(), note_index()),
            ("Doc:emb".to_string(), doc_index()),
            ("Person:emb".to_string(), person_index()),
        ]
    }

    /// An empty "Doc:emb" with the config of `doc_index`.
    fn doc_shell() -> Vec<(String, Arc<VectorIndexKind>)> {
        vec![(
            "Doc:emb".to_string(),
            Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
                3,
                DistanceMetric::Euclidean,
            )))),
        )]
    }

    /// The section of "Doc:emb" alone, with `bytes` as its stream.
    fn doc_section_with_stream(bytes: &[u8]) -> Written {
        let mut written = Written::of(&VectorStoreSection::new(doc_shell()));
        for position in written.pieces(0).into_iter().rev() {
            written.remove(position);
        }
        written.push(ChunkMeta::stream_piece(0, 0, 0), bytes);
        written
    }

    /// A stream header: the entry point flag and value, the top level and
    /// the node count.
    fn stream_header(flag: u8, entry: u64, max_level: u32, node_count: u64) -> Vec<u8> {
        let mut bytes = vec![flag];
        bytes.extend_from_slice(&entry.to_le_bytes());
        bytes.extend_from_slice(&max_level.to_le_bytes());
        bytes.extend_from_slice(&node_count.to_le_bytes());
        bytes
    }

    /// The section of "Doc:emb" with a stream built by hand: the header,
    /// then each node's id and lists.
    fn hand_built(flag: u8, entry: u64, max_level: u32, nodes: &[(u64, Vec<Vec<u64>>)]) -> Written {
        let mut bytes = stream_header(flag, entry, max_level, nodes.len() as u64);
        for (id, layers) in nodes {
            bytes.extend_from_slice(&id.to_le_bytes());
            bytes.extend_from_slice(&u32::try_from(layers.len()).unwrap().to_le_bytes());
            for layer in layers {
                bytes.extend_from_slice(&u32::try_from(layer.len()).unwrap().to_le_bytes());
                for neighbor in layer {
                    bytes.extend_from_slice(&neighbor.to_le_bytes());
                }
            }
        }
        doc_section_with_stream(&bytes)
    }

    /// Empty indexes with the keys and configs of `indexes`, as a load
    /// builds them before reading the section.
    fn shells(indexes: &[(String, Arc<VectorIndexKind>)]) -> Vec<(String, Arc<VectorIndexKind>)> {
        indexes
            .iter()
            .map(|(key, index)| {
                let config = index.config().clone();
                let shell = match index.quantization_type() {
                    None => VectorIndexKind::Hnsw(HnswIndex::new(config)),
                    Some(quantization) => {
                        VectorIndexKind::Quantized(QuantizedHnswIndex::new(config, quantization))
                    }
                };
                (key.clone(), Arc::new(shell))
            })
            .collect()
    }

    /// Reads `source` into `indexes`.
    fn read_into(
        indexes: &[(String, Arc<VectorIndexKind>)],
        source: &dyn SectionSource,
    ) -> Result<()> {
        VectorStoreSection::new(indexes.to_vec()).read_from(source)
    }

    /// Two HNSW topologies written in pieces of at most 512 bytes and read
    /// back into fresh shells: equal to the originals. The quantized index is
    /// not written (a load builds it from the data): its shell stays empty.
    #[test]
    fn hnsw_topologies_round_trip_through_streams() {
        let indexes = test_indexes();
        let mut image = MemoryImage::new();
        with_chunk_caps(TINY, || {
            let section = VectorStoreSection::new(indexes.clone());
            image
                .begin_section(SectionType::VectorStore, section.version())
                .unwrap();
            section.write_to(&mut image).unwrap();
        });
        let source = image.section_source(SectionType::VectorStore).unwrap();
        assert_eq!(source.section_version(), 3);
        let chunks = source.chunks();
        assert_eq!(
            chunks[0],
            ChunkMeta::meta(),
            "the metadata chunk comes first"
        );
        let meta = decode_meta(&source.fetch(0).unwrap());
        assert_eq!(meta.layout, 1);
        assert_eq!(meta.max_bytes, 512, "the metadata records the byte cap");
        let keys: Vec<&str> = meta
            .indexes
            .iter()
            .map(|index| index.key.as_str())
            .collect();
        assert_eq!(
            keys,
            ["Doc:emb", "Person:emb"],
            "the plain indexes in key order, without the quantized one"
        );
        for stream in 0..2u32 {
            let pieces: Vec<usize> = (1..chunks.len())
                .filter(|&position| chunks[position].column_id == stream)
                .collect();
            assert!(pieces.len() > 1, "stream {stream} is cut into pieces");
            for position in pieces {
                assert_eq!(chunks[position].kind, ChunkKind::Stream);
                assert_eq!(chunks[position].graph_id, 0);
                assert!(source.fetch(position).unwrap().len() <= 512);
            }
        }

        let restored = shells(&indexes);
        read_into(&restored, &*source).unwrap();
        for ((key, original), (_, copy)) in indexes[1..].iter().zip(&restored[1..]) {
            assert!(!original.is_empty(), "{key} has nodes");
            assert_eq!(
                copy.snapshot_topology(),
                original.snapshot_topology(),
                "{key}"
            );
        }
        assert!(
            !indexes[0].1.is_empty() && restored[0].1.is_empty(),
            "Note:emb, quantized, is neither written nor restored"
        );
    }

    /// An index without nodes writes only its stream header and reads back
    /// empty, also into an index that held nodes before.
    #[test]
    fn an_index_without_nodes_round_trips() {
        // The config of `make_test_index`, which the read goes into.
        let empty = vec![(
            "Doc:emb".to_string(),
            Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
                4,
                DistanceMetric::Cosine,
            )))),
        )];
        let written = Written::of(&VectorStoreSection::new(empty.clone()));
        assert_eq!(written.pieces(0).len(), 1);
        assert_eq!(written.chunks[1].len(), 21, "the stream header alone");

        let (_, filled) = make_test_index();
        let target = vec![("Doc:emb".to_string(), filled)];
        read_into(&target, &written).unwrap();
        assert_eq!(target[0].1.snapshot_topology(), (None, 0, Vec::new()));
    }

    /// A stream that ends early, or holds a byte after its last node, is
    /// refused naming its index, which is left empty.
    #[test]
    fn a_truncated_or_overlong_stream_is_refused() {
        let indexes = test_indexes();
        let written = with_chunk_caps(TINY, || {
            Written::of(&VectorStoreSection::new(indexes.clone()))
        });
        let last = *written.pieces(0).last().unwrap();
        let mut truncated = written.clone();
        truncated.remove(last);
        let mut overlong = written.clone();
        let mut longer = overlong.chunks[last].to_vec();
        longer.push(3);
        overlong.chunks[last] = Bytes::from(longer);

        for (case, source) in [("truncated", &truncated), ("overlong", &overlong)] {
            let restored = shells(&indexes);
            let error = read_into(&restored, source).unwrap_err();
            assert!(matches!(error, Error::Corruption(_)), "{case}: {error:?}");
            let message = error.to_string();
            assert!(
                message.contains("VectorStore") && message.contains("'Doc:emb'"),
                "{case}: the error names the section and the index: {message}"
            );
            assert!(
                restored[1].1.is_empty(),
                "{case}: the index whose stream failed is left empty"
            );
        }
    }

    /// 0.5.x bytes, the version 2 envelope `serialize` writes (and the
    /// version 1 bincode before it), arrive as one raw chunk of version 0 and
    /// still restore the topology.
    #[test]
    fn a_0_5_vector_section_still_loads() {
        let indexes = test_indexes();
        let envelope = VectorStoreSection::new(indexes.clone())
            .serialize()
            .unwrap();
        assert_eq!(
            &envelope[0..5],
            b"GVST\x02",
            "the 0.5.x envelope, version 2"
        );
        let image = MemoryImage::from_raw(vec![(SectionType::VectorStore, envelope)]).unwrap();
        let restored = shells(&indexes);
        read_into(
            &restored,
            &*image.section_source(SectionType::VectorStore).unwrap(),
        )
        .unwrap();
        for ((key, original), (_, copy)) in indexes.iter().zip(&restored) {
            assert_eq!(
                copy.snapshot_topology(),
                original.snapshot_topology(),
                "{key}"
            );
        }

        let v1 = MemoryImage::from_raw(vec![(
            SectionType::VectorStore,
            make_v1_snapshot_bytes("Item:embedding"),
        )])
        .unwrap();
        let (_, item) = make_test_index();
        let fresh = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(item.config().clone())));
        read_into(
            &[("Item:embedding".to_string(), Arc::clone(&fresh))],
            &*v1.section_source(SectionType::VectorStore).unwrap(),
        )
        .unwrap();
        assert_eq!(fresh.snapshot_topology(), item.snapshot_topology());
    }

    /// The bytes depend on the topologies alone: the same section written
    /// twice, and copies of its indexes restored into fresh maps (another
    /// hash order) and given in the other order, write the same chunks.
    #[test]
    fn the_same_topologies_write_the_same_bytes() {
        let indexes = test_indexes();
        let mut copies = shells(&indexes);
        for ((_, original), (_, copy)) in indexes.iter().zip(&copies) {
            let (entry_point, max_level, nodes) = original.snapshot_topology();
            copy.restore_topology(entry_point, max_level, nodes);
        }
        copies.reverse();
        with_chunk_caps(TINY, || {
            let first = Written::of(&VectorStoreSection::new(indexes.clone()));
            let second = Written::of(&VectorStoreSection::new(indexes.clone()));
            let copied = Written::of(&VectorStoreSection::new(copies));
            assert_eq!(first, second, "the same section written twice");
            assert_eq!(first, copied, "copies of the indexes in another order");
        });
    }

    /// An index of the metadata the section was not given is skipped without
    /// reading its stream; an index given but not in the metadata stays as
    /// it is.
    #[test]
    fn indexes_are_matched_by_key() {
        let indexes = test_indexes();
        let mut written = with_chunk_caps(TINY, || {
            Written::of(&VectorStoreSection::new(indexes.clone()))
        });
        // Without the stream of "Person:emb", which a read would refuse.
        for position in written.pieces(1).into_iter().rev() {
            written.remove(position);
        }

        let doc = shells(&indexes[1..2]);
        let item = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
            3,
            DistanceMetric::Euclidean,
        ))));
        let given = vec![doc[0].clone(), ("Item:emb".to_string(), Arc::clone(&item))];
        read_into(&given, &written).unwrap();
        assert_eq!(
            given[0].1.snapshot_topology(),
            indexes[1].1.snapshot_topology(),
            "Doc:emb is restored"
        );
        assert!(item.is_empty(), "Item:emb, not in the section, stays empty");
    }

    /// A topology is restored only into an index of its dimensions and
    /// metric: neighbors chosen by other distances would answer searches
    /// wrongly. The error names the index (a load then builds it from the
    /// data).
    #[test]
    fn a_topology_of_other_dimensions_or_a_metric_is_refused() {
        let indexes = test_indexes();
        let written = Written::of(&VectorStoreSection::new(indexes[1..2].to_vec()));
        for (config, needle) in [
            (HnswConfig::new(4, DistanceMetric::Euclidean), "dimensions"),
            (HnswConfig::new(3, DistanceMetric::Cosine), "metric"),
        ] {
            let shell = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(config)));
            let error =
                read_into(&[("Doc:emb".to_string(), Arc::clone(&shell))], &written).unwrap_err();
            assert!(matches!(error, Error::Corruption(_)), "{error:?}");
            let message = error.to_string();
            assert!(
                message.contains("'Doc:emb'") && message.contains(needle),
                "{needle}: {message}"
            );
            assert!(shell.is_empty(), "{needle}: nothing is restored");
        }
    }

    /// Chunk sequences this writer never produces are refused, naming what
    /// is wrong.
    #[test]
    fn crafted_vector_sections_are_refused() {
        let indexes = test_indexes();
        let written = with_chunk_caps(TINY, || {
            Written::of(&VectorStoreSection::new(indexes.clone()))
        });
        let mut trailing = written.clone();
        let mut meta_bytes = trailing.chunks[0].to_vec();
        meta_bytes.push(88);
        trailing.chunks[0] = Bytes::from(meta_bytes);
        let mut stream_first = written.clone();
        stream_first.metas.swap(0, 1);
        stream_first.chunks.swap(0, 1);
        let mut column = written.clone();
        column.push(ChunkMeta::column(0, 0, 0, 1, 0), b"Mia");
        let mut unlisted = written.clone();
        unlisted.push(ChunkMeta::stream_piece(0, 2, 0), b"Jules");
        let mut other_graph = written.clone();
        other_graph.push(ChunkMeta::stream_piece(3, 0, 0), b"Jules");
        let mut meta_codec = written.clone();
        meta_codec.metas[0].codec = 19;
        let mut no_raw = written.clone();
        no_raw.version = 2;

        let cases = [
            ("another version", no_raw, "version 2"),
            ("a stream piece first", stream_first, "metadata chunk"),
            ("a metadata chunk with a codec", meta_codec, "codec 19"),
            (
                "another layout",
                written.with_meta(|meta| meta.layout = 2),
                "layout 2",
            ),
            (
                "an unknown metric",
                written.with_meta(|meta| meta.indexes[0].metric = 4),
                "metric 4",
            ),
            (
                "keys out of order",
                written.with_meta(|meta| meta.indexes.swap(0, 1)),
                "increasing",
            ),
            (
                "a key twice",
                written.with_meta(|meta| meta.indexes[1].key = "Doc:emb".to_string()),
                "increasing",
            ),
            ("a byte after the metadata", trailing, "after"),
            ("a column chunk", column, "Column"),
            ("a piece of a stream not listed", unlisted, "stream 2"),
            ("a piece of another graph", other_graph, "graph 3"),
        ];
        for (case, source, needle) in cases {
            let error = read_into(&shells(&indexes), &source).unwrap_err();
            // Another version is another build's section, not damage.
            if case == "another version" {
                assert!(
                    matches!(error, Error::Serialization(_)),
                    "{case}: {error:?}"
                );
            } else {
                assert!(matches!(error, Error::Corruption(_)), "{case}: {error:?}");
            }
            let message = error.to_string();
            assert!(
                message.contains("VectorStore") && message.contains(needle),
                "{case}: {message}"
            );
        }

        // Every metric byte is checked, also that of an index not given.
        let error = read_into(
            &shells(&indexes[1..2]),
            &written.with_meta(|meta| meta.indexes[1].metric = 88),
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("'Person:emb' has metric 88"),
            "{error}"
        );
    }

    /// Lengths in the metadata chunk are not trusted with an allocation: a
    /// key or an index list that claims 2^40 entries is refused as metadata
    /// that does not decode.
    #[test]
    fn huge_metadata_lengths_are_refused_without_allocating() {
        let written = Written::of(&VectorStoreSection::new(doc_shell()));
        // Layout 1, max_bytes 512 (varint 251 and a u16), then a list of one
        // index whose key claims 2^40 bytes (varint 253 and a u64).
        let mut key_claim = vec![1u8, 251, 0, 2, 1, 253];
        key_claim.extend_from_slice(&(1u64 << 40).to_le_bytes());
        // The same, with a list that claims 2^40 indexes.
        let mut list_claim = vec![1u8, 251, 0, 2, 253];
        list_claim.extend_from_slice(&(1u64 << 40).to_le_bytes());
        for (case, bytes) in [("key", key_claim), ("list", list_claim)] {
            let mut claimed = written.clone();
            claimed.chunks[0] = Bytes::from(bytes);
            let error = read_into(&doc_shell(), &claimed).unwrap_err();
            assert!(
                matches!(&error, Error::Corruption(corruption) if corruption.what.contains("does not decode")),
                "{case}: {error:?}"
            );
        }
    }

    /// Streams this writer never produces are refused, leaving the index
    /// empty: node ids that do not increase, an entry point flag other than 0
    /// or 1, and headers that contradict their nodes. The writer keeps an
    /// entry point exactly when there are nodes, the entry point a node with
    /// exactly top level + 1 levels, and every node with 1 to top level + 1
    /// levels.
    #[test]
    fn crafted_streams_are_refused() {
        let flat = |ids: &[u64]| -> Vec<(u64, Vec<Vec<u64>>)> {
            ids.iter().map(|&id| (id, vec![Vec::new()])).collect()
        };
        let pair =
            |left: u64, right: u64| vec![(left, vec![vec![right]]), (right, vec![vec![left]])];

        let valid = hand_built(
            1,
            19,
            1,
            &[
                (3, vec![vec![19]]),
                (19, vec![vec![3, 88], vec![]]),
                (88, vec![vec![19]]),
            ],
        );
        let shell = doc_shell();
        read_into(&shell, &valid).unwrap();
        let id = NodeId::new;
        assert_eq!(
            shell[0].1.snapshot_topology(),
            (
                Some(id(19)),
                1,
                vec![
                    (id(3), vec![vec![id(19)]]),
                    (id(19), vec![vec![id(3), id(88)], vec![]]),
                    (id(88), vec![vec![id(19)]]),
                ]
            ),
            "a stream built by hand reads"
        );

        for (case, source, needle) in [
            (
                "a repeated id",
                hand_built(1, 3, 0, &flat(&[3, 19, 19])),
                "increasing",
            ),
            (
                "a smaller id",
                hand_built(1, 3, 0, &flat(&[3, 88, 19])),
                "increasing",
            ),
            (
                "an entry point flag of 2",
                hand_built(2, 3, 0, &flat(&[3])),
                "flag 2",
            ),
            (
                "no entry point but nodes",
                hand_built(0, 0, 0, &pair(3, 19)),
                "no entry point but 2 nodes",
            ),
            (
                "no entry point but an entry point value",
                hand_built(0, 3, 0, &[]),
                "entry point value 3",
            ),
            (
                "an entry point but no nodes",
                hand_built(1, 3, 2, &[]),
                "entry point 3 but no nodes",
            ),
            (
                "an entry point that is not a node",
                hand_built(1, 7, 0, &pair(3, 19)),
                "entry point 7 is not a node",
            ),
            (
                "a top level above the entry point's levels",
                hand_built(1, 3, 5, &pair(3, 19)),
                "entry point 3 has 1 levels",
            ),
            (
                "a top level of u32::MAX",
                hand_built(1, 3, u32::MAX, &flat(&[3])),
                "entry point 3 has 1 levels",
            ),
            (
                "a node without levels",
                hand_built(1, 3, 0, &[(3, vec![vec![]]), (19, vec![])]),
                "node 19 has 0 levels",
            ),
            (
                "a node above the top level",
                hand_built(
                    1,
                    3,
                    0,
                    &[(3, vec![vec![19]]), (19, vec![vec![3], vec![3]])],
                ),
                "node 19 has 2 levels",
            ),
            (
                "a neighbor that is not a node",
                hand_built(1, 3, 0, &[(3, vec![vec![19]]), (19, vec![vec![3, 7]])]),
                "node 19 lists node 7 at level 0, but node 7 is not a node of the stream",
            ),
            (
                "a neighbor listed at a level it does not have",
                hand_built(
                    1,
                    19,
                    1,
                    &[(3, vec![vec![19]]), (19, vec![vec![3], vec![3]])],
                ),
                "node 19 lists node 3 at level 1, but node 3 has 1 levels",
            ),
            (
                "a node listing itself",
                hand_built(1, 3, 0, &[(3, vec![vec![19, 3]]), (19, vec![vec![3]])]),
                "node 3 lists itself at level 0",
            ),
        ] {
            let target = doc_shell();
            let error = read_into(&target, &source).unwrap_err();
            assert!(matches!(error, Error::Corruption(_)), "{case}: {error:?}");
            let message = error.to_string();
            assert!(
                message.contains("'Doc:emb'") && message.contains(needle),
                "{case}: {message}"
            );
            assert_eq!(
                target[0].1.snapshot_topology(),
                (None, 0, Vec::new()),
                "{case}: the index is left empty"
            );
        }
    }

    /// A list read from a stream reserves room for at most 64 entries up
    /// front, whatever count the stream claims, and for every entry of a
    /// smaller count.
    #[test]
    fn a_list_read_from_a_stream_reserves_room_for_at_most_64_entries() {
        let claimed: Vec<NodeId> = list_with_room_for(u32::MAX as usize);
        assert!(
            claimed.capacity() <= 64,
            "a claimed count of u32::MAX reserves {} entries",
            claimed.capacity()
        );
        let levels: Vec<Vec<NodeId>> = list_with_room_for(u32::MAX as usize);
        assert!(levels.capacity() <= 64, "{}", levels.capacity());
        let small: Vec<NodeId> = list_with_room_for(19);
        assert!(small.capacity() >= 19, "{}", small.capacity());
    }

    /// Counts read from a stream are not trusted with an allocation: a
    /// neighbor count, or a level count under a top level that allows it, of
    /// u32::MAX in a stream that ends right after it is refused as a stream
    /// that ends early.
    #[test]
    fn huge_counts_are_refused_without_allocating() {
        for (case, max_level, counts) in [
            ("neighbor count", 0, vec![1u32, u32::MAX]),
            ("level count", u32::MAX, vec![u32::MAX]),
        ] {
            let mut bytes = stream_header(1, 3, max_level, 1);
            bytes.extend_from_slice(&3u64.to_le_bytes());
            for count in counts {
                bytes.extend_from_slice(&count.to_le_bytes());
            }
            let error = read_into(&doc_shell(), &doc_section_with_stream(&bytes)).unwrap_err();
            assert!(
                matches!(&error, Error::Corruption(corruption) if corruption.what.contains("ends before")),
                "{case}: {error:?}"
            );
        }
    }

    /// A sink that fails a chunk fails the write with its own error, kind
    /// included, not with the text of it; a chunk that cannot be fetched
    /// fails the read with its own kind and names the index.
    #[test]
    fn sink_and_fetch_errors_keep_their_kind() {
        /// Accepts `accept` chunks, then fails with an I/O error.
        struct FullDisk {
            accept: usize,
        }

        impl SectionSink for FullDisk {
            fn write_chunk(&mut self, _meta: ChunkMeta, _bytes: &[u8]) -> Result<()> {
                if self.accept == 0 {
                    return Err(Error::Io(io::Error::other("Gus: the disk is full")));
                }
                self.accept -= 1;
                Ok(())
            }
        }

        /// Serves `written`, failing every fetch of a stream piece.
        struct Unreadable<'a>(&'a Written);

        impl SectionSource for Unreadable<'_> {
            fn chunks(&self) -> &[ChunkMeta] {
                self.0.chunks()
            }

            fn fetch(&self, index: usize) -> Result<Bytes> {
                if index == 0 {
                    return self.0.fetch(0);
                }
                Err(Error::Io(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("chunk {index} fails its checksum"),
                )))
            }

            fn stored_length(&self, index: usize) -> Result<u64> {
                self.0.stored_length(index)
            }

            fn section_version(&self) -> u8 {
                self.0.section_version()
            }
        }

        let indexes = test_indexes();
        with_chunk_caps(TINY, || {
            let section = VectorStoreSection::new(indexes.clone());
            for accept in [0, 1, 3] {
                let error = section.write_to(&mut FullDisk { accept }).unwrap_err();
                assert!(
                    matches!(&error, Error::Io(inner) if inner.to_string().contains("Gus")),
                    "a failure after {accept} chunks: {error:?}"
                );
            }

            let written = Written::of(&section);
            let error = read_into(&shells(&indexes), &Unreadable(&written)).unwrap_err();
            assert!(
                matches!(&error, Error::Io(inner) if inner.kind() == io::ErrorKind::InvalidData),
                "{error:?}"
            );
            let message = error.to_string();
            assert!(
                message.contains("fails its checksum") && message.contains("'Doc:emb'"),
                "{message}"
            );
        });
    }
}
