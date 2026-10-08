//! [`Section`](grafeo_common::storage::section::Section) implementation for [`CompactStore`].
//!
//! The section (version 5, since 0.6) is a metadata chunk (`CompactMeta`)
//! and stream 0, which holds the version 4 encoding: a header, the node
//! tables, the relationship tables, the id maps and a CRC32 of all of it.
//! Writing holds one piece of the stream at a time, except that each column
//! and each adjacency is encoded whole before it is written. The reader reads
//! the stream into one buffer, which becomes the store's column storage, as a
//! mapped spill file does. A 0.5.x file holds the version 1, 2 or 3 encoding
//! as one raw chunk, which still loads.

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bytes::Bytes;
use grafeo_common::storage::section::{Section, SectionType};
use grafeo_common::storage::{
    ChunkCaps, ChunkKind, ChunkMeta, ChunkStreamWriter, SectionSink, SectionSource, check_version,
    legacy_bytes, read_stream, stream_error,
};
use grafeo_common::types::{EdgeId, NodeId, PropertyKey};
use grafeo_common::utils::error::Error;
use grafeo_common::utils::hash::FxHashMap;
use parking_lot::RwLock;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use super::column::{ColumnCodec, CompactColumn};
use super::csr::CsrAdjacency;
use super::node_table::NodeTable;
use super::rel_table::RelTable;
use super::schema::{ColumnDef, ColumnType, EdgeSchema, TableSchema};
use super::zone_map::ZoneMap;
use super::{CompactStore, label_set_key, labels_of_key, labels_of_unescaped_key};
use crate::codec::limits::{checked_u16, checked_u32};

/// Magic bytes identifying a CompactStore encoding.
const MAGIC: [u8; 4] = *b"GCST";

/// The encoding this build writes: v3 plus, after each column body, which
/// rows have a value (#542). Phase 2c bumped 2 to 3 to embed per-block zone
/// maps in the column index for skip pruning. Spill files hold it as it is,
/// and stream 0 of a version 5 section holds it.
const FORMAT_VERSION: u8 = 4;

/// The section version since 0.6: a metadata chunk, then stream 0 holding
/// the [`FORMAT_VERSION`] encoding. Only directory entries carry it; the
/// encoding in the stream keeps its own version byte.
const FORMAT_VERSION_CHUNKED: u8 = 5;

/// v3 layout: per-block index with per-block stats, every row has a value.
/// Retained as a read-only compat path; files written by 0.5.42 to 0.5.44
/// carry this byte (their missing properties already hold empty values).
const FORMAT_VERSION_V3: u8 = 3;

/// v2 (Phase 2b) layout: per-block index + bodies, no per-block stats.
/// Retained as a read-only compat path for one release.
const FORMAT_VERSION_V2: u8 = 2;

/// v1 layout: flat columns, no blocks. Retained as a read-only compat
/// path for one release. Files written by 0.5.41 and earlier carry
/// this byte; 0.5.42+ writers always emit [`FORMAT_VERSION`].
const FORMAT_VERSION_V1: u8 = 1;

/// The encodings a raw chunk (0.5.x bytes) or a spill file may hold, and
/// [`CompactStoreSection::write_with_version`] writes.
const READABLE_ENCODINGS: [u8; 4] = [
    FORMAT_VERSION_V1,
    FORMAT_VERSION_V2,
    FORMAT_VERSION_V3,
    FORMAT_VERSION,
];

/// The stream holding the encoding.
const COMPACT_STREAM: u32 = 0;

/// The layout byte every metadata chunk of the compact module's sections
/// starts with.
pub(super) const META_LAYOUT: u8 = 1;

/// Bytes of id map entries gathered before they go to the stream.
const ID_MAP_PIECE: usize = 64 * 1024;

/// The metadata chunk of a version 5 section.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct CompactMeta {
    /// [`META_LAYOUT`].
    layout: u8,
    /// The byte cap the stream was cut with; readers accept any.
    max_bytes: u32,
}

/// Wraps a [`CompactStore`] as a container [`Section`].
pub struct CompactStoreSection {
    store: RwLock<Option<Arc<CompactStore>>>,
    dirty: AtomicBool,
    /// The caps the stream is cut with: [`ChunkCaps::current`] when the
    /// section was built.
    caps: ChunkCaps,
}

impl CompactStoreSection {
    /// Creates a new section wrapping an existing store.
    #[must_use]
    pub fn new(store: Arc<CompactStore>) -> Self {
        Self {
            store: RwLock::new(Some(store)),
            dirty: AtomicBool::new(false),
            caps: ChunkCaps::current(),
        }
    }

    /// Creates an empty section (for deserialization).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            store: RwLock::new(None),
            dirty: AtomicBool::new(false),
            caps: ChunkCaps::current(),
        }
    }

    /// Marks this section as dirty.
    pub fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }

    /// Returns a reference to the inner store, if any.
    #[must_use]
    pub fn store(&self) -> Option<Arc<CompactStore>> {
        self.store.read().clone()
    }

    /// Deserializes from a refcounted [`Bytes`] buffer (Phase 3c).
    ///
    /// This is the zero-copy entry point: when `data` wraps a mmap
    /// region (via [`bytes::Bytes::from_owner`]), column codec storage
    /// is constructed via `data.slice(range)` rather than copying. The
    /// trait [`Section::deserialize`] entry point still works on
    /// `&[u8]` and incurs one heap copy (a single `Bytes::copy_from_slice`
    /// at the boundary).
    ///
    /// # Errors
    ///
    /// Same error semantics as [`Section::deserialize`].
    pub fn deserialize_from_bytes(
        &mut self,
        data: bytes::Bytes,
    ) -> grafeo_common::utils::error::Result<()> {
        let store = deserialize_compact_store(&data, &READABLE_ENCODINGS)
            .map_err(|e| Error::Internal(format!("CompactStore deserialization failed: {e}")))?;
        *self.store.write() = Some(Arc::new(store));
        Ok(())
    }

    /// Reads the store of a section from `data`, an encoding of one of the
    /// versions `accepted`; its column storage slices `data`. An encoding
    /// that does not decode is corrupt section data.
    fn load_encoding(
        &mut self,
        data: Bytes,
        accepted: &[u8],
    ) -> grafeo_common::utils::error::Result<()> {
        let store = deserialize_compact_store(&data, accepted)
            .map_err(|e| Error::Serialization(format!("section CompactStore: {e}")))?;
        *self.store.write() = Some(Arc::new(store));
        Ok(())
    }

    /// Serializes at the requested format version (see
    /// [`write_with_version`](Self::write_with_version)).
    ///
    /// [`Section::serialize`] writes [`FORMAT_VERSION`]. This entry point is
    /// kept (test-only outside this crate) so the compat readers can be
    /// exercised without keeping any externally committed fixtures.
    pub(crate) fn serialize_with_version(
        &self,
        version: u8,
    ) -> grafeo_common::utils::error::Result<Vec<u8>> {
        let mut bytes = Vec::with_capacity(self.memory_usage());
        self.write_with_version(version, &mut bytes)?;
        Ok(bytes)
    }

    /// The encoding of `version` written piece by piece to `out`: each
    /// column and each adjacency is encoded whole and then written, the id
    /// maps in pieces of at most 64 KiB, and last the CRC32 of every byte
    /// before it.
    ///
    /// The bytes depend on what the store holds, not on the order its hash
    /// maps iterate in: columns are written in key order, and id map entries
    /// in table order and then row order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Internal`] when the section holds no store, `version`
    /// is not an encoding the reader reads, or the id maps of the store do
    /// not agree with their reverse; [`Error::Serialization`] when a name, a
    /// length or a column does not fit the encoding; [`Error::Io`] when `out`
    /// fails.
    pub(crate) fn write_with_version(
        &self,
        version: u8,
        out: &mut dyn io::Write,
    ) -> grafeo_common::utils::error::Result<()> {
        if !READABLE_ENCODINGS.contains(&version) {
            return Err(Error::Internal(format!(
                "CompactStore has no encoding of version {version}"
            )));
        }
        let guard = self.store.read();
        let store = guard
            .as_ref()
            .ok_or_else(|| Error::Internal("no CompactStore to serialize".into()))?;
        let mut out = CrcWriter::new(out);
        // The piece being encoded: at most one column or adjacency, or
        // 64 KiB of id map entries, besides a few names and lengths.
        let mut buf = Vec::new();

        // Header.
        buf.extend_from_slice(&MAGIC);
        buf.push(version);
        let flags: u8 = u8::from(store.preserves_ids());
        buf.push(flags);

        // Node tables, each named by its key; the 0.5.x encodings by the
        // labels joined without escapes, as 0.5.x wrote them.
        write_len(&mut buf, store.node_tables_by_id.len())?;
        for (nt, labels) in store.node_tables_by_id.iter().zip(&store.table_labels) {
            if version >= FORMAT_VERSION {
                write_str(&mut buf, nt.label())?;
            } else {
                let joined: Vec<&str> = labels.iter().map(arcstr::ArcStr::as_str).collect();
                write_str(&mut buf, &joined.join("|"))?;
            }
            write_len(&mut buf, nt.len())?;
            let columns = nt.columns();
            let zone_maps = nt.zone_maps();
            write_len(&mut buf, columns.len())?;
            for (key, codec) in in_key_order(columns) {
                write_str(&mut buf, key.as_str())?;
                // Zone map for this column.
                if let Some(zm) = zone_maps.get(key) {
                    buf.push(1);
                    write_zone_map(&mut buf, zm)?;
                } else {
                    buf.push(0);
                }
                write_column(
                    codec,
                    &mut buf,
                    version,
                    nt.block_zone_maps().get(key).map(Vec::as_slice),
                )?;
                out.drain(&mut buf)?;
            }
        }

        // Relationship tables.
        write_len(&mut buf, store.rel_tables_by_id.len())?;
        for rt in &store.rel_tables_by_id {
            write_str(&mut buf, rt.edge_type().as_str())?;
            write_u16(&mut buf, rt.src_table_id());
            write_u16(&mut buf, rt.dst_table_id());
            rt.fwd().write_to(&mut buf)?;
            out.drain(&mut buf)?;
            if let Some(bwd) = rt.bwd() {
                buf.push(1);
                bwd.write_to(&mut buf)?;
                out.drain(&mut buf)?;
            } else {
                buf.push(0);
            }
            let properties = rt.properties();
            write_len(&mut buf, properties.len())?;
            for (key, codec) in in_key_order(properties) {
                write_str(&mut buf, key.as_str())?;
                // Edge property columns don't keep per-block zone maps; v3+
                // computes them during write, from the column when some rows
                // have no value (the codec alone would count their empty
                // values).
                let block_stats = codec
                    .present()
                    .map(|_| super::zone_map::compute_block_zone_maps(codec));
                write_column(codec, &mut buf, version, block_stats.as_deref())?;
                out.drain(&mut buf)?;
            }
        }

        // ID maps.
        if store.preserves_ids() {
            let node_rows: Vec<usize> =
                store.node_tables_by_id.iter().map(NodeTable::len).collect();
            write_id_map(
                &mut out,
                &mut buf,
                store.node_id_map.as_ref(),
                store.node_offset_to_id.as_deref(),
                &node_rows,
            )?;
            let edge_rows: Vec<usize> = store
                .rel_tables_by_id
                .iter()
                .map(RelTable::num_edges)
                .collect();
            write_id_map(
                &mut out,
                &mut buf,
                store.edge_id_map.as_ref(),
                store.edge_offset_to_id.as_deref(),
                &edge_rows,
            )?;
        }
        out.drain(&mut buf)?;
        out.finish()?;
        Ok(())
    }
}

/// Passes the pieces of an encoding on to `out`, keeping the CRC32 of every
/// byte, which the encoding ends with.
struct CrcWriter<'w> {
    out: &'w mut dyn io::Write,
    crc: crc32fast::Hasher,
}

impl<'w> CrcWriter<'w> {
    fn new(out: &'w mut dyn io::Write) -> Self {
        Self {
            out,
            crc: crc32fast::Hasher::new(),
        }
    }

    /// Writes `pending` and empties it, keeping its capacity for the next
    /// piece.
    fn drain(&mut self, pending: &mut Vec<u8>) -> io::Result<()> {
        self.crc.update(pending);
        self.out.write_all(pending)?;
        pending.clear();
        Ok(())
    }

    /// Writes the CRC32 of every byte drained before.
    fn finish(self) -> io::Result<()> {
        self.out.write_all(&self.crc.finalize().to_le_bytes())
    }
}

/// The entries of `columns` in key order.
fn in_key_order(
    columns: &FxHashMap<PropertyKey, CompactColumn>,
) -> Vec<(&PropertyKey, &CompactColumn)> {
    let mut sorted: Vec<_> = columns.iter().collect();
    sorted.sort_unstable_by(|left, right| left.0.cmp(right.0));
    sorted
}

/// What an id map needs of its ids: node ids and edge ids.
trait MapId: Copy + Eq + std::hash::Hash {
    /// The id that marks a reverse slot without an entry; never a real id.
    const INVALID: Self;
    /// What the ids name, for messages.
    const WHAT: &'static str;

    fn from_raw(raw: u64) -> Self;

    fn raw(self) -> u64;
}

impl MapId for NodeId {
    const INVALID: Self = NodeId::INVALID;
    const WHAT: &'static str = "node";

    fn from_raw(raw: u64) -> Self {
        NodeId::new(raw)
    }

    fn raw(self) -> u64 {
        self.as_u64()
    }
}

impl MapId for EdgeId {
    const INVALID: Self = EdgeId::INVALID;
    const WHAT: &'static str = "edge";

    fn from_raw(raw: u64) -> Self {
        EdgeId::new(raw)
    }

    fn raw(self) -> u64 {
        self.as_u64()
    }
}

/// Writes an id map, when the store has one: its length, then `(id, table,
/// row)` per entry, in table order and then row order, as `rows` (the map's
/// reverse, built with it) lists them. Walking `rows` gives an order that
/// depends on the store alone without sorting a copy of the map; every entry
/// `rows` lists is checked against the map first, so an entry missing from
/// `rows` cannot be dropped unnoticed. As the reader requires, every row of
/// the tables (`table_rows` per table) has exactly one entry.
fn write_id_map<Id: MapId>(
    out: &mut CrcWriter<'_>,
    buf: &mut Vec<u8>,
    map: Option<&FxHashMap<Id, (u16, u64)>>,
    rows: Option<&[Vec<Id>]>,
    table_rows: &[usize],
) -> grafeo_common::utils::error::Result<()> {
    let Some(map) = map else {
        return Ok(());
    };
    let what = Id::WHAT;
    let disagree = |detail: String| {
        Error::Internal(format!(
            "the {what} id map of the compacted base and its reverse disagree: {detail}"
        ))
    };
    let rows = rows.ok_or_else(|| disagree("there is no reverse".to_string()))?;
    let table_id = |table: usize| {
        u16::try_from(table).map_err(|_| disagree(format!("the reverse has table {table}")))
    };
    let entries = || {
        rows.iter().enumerate().flat_map(move |(table, ids)| {
            ids.iter()
                .enumerate()
                .filter(move |(_, id)| **id != Id::INVALID)
                .map(move |(row, &id)| (table, row as u64, id))
        })
    };
    let mut listed = 0usize;
    for (table, row, id) in entries() {
        if table_rows
            .get(table)
            .is_none_or(|&count| row >= count as u64)
        {
            return Err(disagree(format!(
                "id {} is at table {table}, row {row}, which the tables do not hold",
                id.raw()
            )));
        }
        let table = table_id(table)?;
        if map.get(&id) != Some(&(table, row)) {
            return Err(disagree(format!(
                "id {} is at table {table}, row {row} in the reverse, at {:?} in the map",
                id.raw(),
                map.get(&id)
            )));
        }
        listed += 1;
    }
    if listed != map.len() {
        return Err(disagree(format!(
            "the map holds {} ids, the reverse {listed}",
            map.len()
        )));
    }
    let rows: usize = table_rows.iter().sum();
    if listed != rows {
        return Err(disagree(format!(
            "they hold {listed} ids for the {rows} rows of the tables"
        )));
    }
    write_len(buf, listed)?;
    for (table, row, id) in entries() {
        write_u64(buf, id.raw());
        write_u16(buf, table_id(table)?);
        write_u64(buf, row);
        if buf.len() >= ID_MAP_PIECE {
            out.drain(buf)?;
        }
    }
    Ok(())
}

/// Writes a single column codec body using the layout matching the
/// section's format version.
///
/// - v1 = flat columns (legacy)
/// - v2 = per-block index + concatenated bodies, no stats
/// - v3 = v2 layout + inline per-block zone map per index entry
///
/// `block_stats_hint` is consulted only at v3; when `None` or with a
/// mismatched length, [`ColumnCodec::write_to_v3`] computes the stats
/// from the column itself.
fn write_codec(
    codec: &ColumnCodec,
    buf: &mut Vec<u8>,
    version: u8,
    block_stats_hint: Option<&[ZoneMap]>,
) -> grafeo_common::utils::error::Result<()> {
    match version {
        FORMAT_VERSION_V1 => codec.write_to(buf),
        FORMAT_VERSION_V2 => codec.write_to_v2(buf),
        _ => codec.write_to_v3(buf, block_stats_hint),
    }
}

/// Writes a column: its codec body (see [`write_codec`]) and, from v4, which
/// rows have a value.
fn write_column(
    column: &CompactColumn,
    buf: &mut Vec<u8>,
    version: u8,
    block_stats_hint: Option<&[ZoneMap]>,
) -> grafeo_common::utils::error::Result<()> {
    write_codec(column.codec(), buf, version, block_stats_hint)?;
    if version >= FORMAT_VERSION {
        column.write_present(buf)?;
    }
    Ok(())
}

// ── Chunks of the stream layout ────────────────────────────────────

/// Writes `meta` as the metadata chunk of `section_type`: bincode in the
/// standard configuration.
///
/// # Errors
///
/// Returns [`Error::Serialization`] when `meta` does not encode, or the
/// sink's error.
pub(super) fn write_meta<T: Serialize>(
    sink: &mut dyn SectionSink,
    section_type: SectionType,
    meta: &T,
) -> grafeo_common::utils::error::Result<()> {
    let bytes = bincode::serde::encode_to_vec(meta, bincode::config::standard()).map_err(|e| {
        Error::Serialization(format!(
            "section {section_type:?}: the metadata chunk does not encode: {e}"
        ))
    })?;
    sink.write_chunk(ChunkMeta::meta(), &bytes)
}

/// The metadata chunk of a section of the stream layout, after checking the
/// chunks around it: the first chunk is the metadata chunk, with layout
/// [`META_LAYOUT`] and no bytes after its fields, and every other chunk is a
/// piece of one of the first `streams` streams of graph 0.
///
/// # Errors
///
/// Returns [`Error::Serialization`] naming the section and the chunk for any
/// other sequence or metadata, or the error of fetching the metadata chunk.
pub(super) fn read_meta<T: DeserializeOwned>(
    source: &dyn SectionSource,
    section_type: SectionType,
    streams: u32,
) -> grafeo_common::utils::error::Result<T> {
    let refuse = |what: String| Error::Serialization(format!("section {section_type:?}: {what}"));
    let chunks = source.chunks();
    match chunks.first() {
        Some(first) if *first == ChunkMeta::meta() => {}
        Some(first) => {
            return Err(refuse(format!(
                "the first chunk is {}, where the metadata chunk belongs",
                describe_chunk(first)
            )));
        }
        None => return Err(refuse("there is no metadata chunk".to_string())),
    }
    for (index, chunk) in chunks.iter().enumerate().skip(1) {
        if chunk.kind != ChunkKind::Stream || chunk.graph_id != 0 || chunk.column_id >= streams {
            return Err(refuse(format!(
                "chunk {index} is {}; after its metadata chunk the section holds pieces of its \
                 {streams} streams of graph 0 only",
                describe_chunk(chunk)
            )));
        }
    }
    let bytes = source.fetch(0)?;
    match bytes.first() {
        Some(&META_LAYOUT) => {}
        Some(layout) => {
            return Err(refuse(format!(
                "the metadata chunk has layout {layout}, this build reads layout {META_LAYOUT}"
            )));
        }
        None => return Err(refuse("the metadata chunk is empty".to_string())),
    }
    let (meta, read) = bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
        .map_err(|e| refuse(format!("the metadata chunk does not decode: {e}")))?;
    if read != bytes.len() {
        return Err(refuse(format!(
            "the metadata chunk holds {} bytes after its fields",
            bytes.len() - read
        )));
    }
    Ok(meta)
}

/// A chunk as an error message names it.
fn describe_chunk(chunk: &ChunkMeta) -> String {
    if chunk.kind == ChunkKind::Stream {
        format!(
            "a piece of stream {} of graph {}",
            chunk.column_id, chunk.graph_id
        )
    } else {
        format!(
            "a {:?} chunk of graph {}, column {}, first row {}",
            chunk.kind, chunk.graph_id, chunk.column_id, chunk.row_start
        )
    }
}

/// `error` with `section_type` in front of its message when it is a
/// [`Error::Serialization`] (corrupt section data); any other error as it is.
pub(super) fn in_section(section_type: SectionType, error: Error) -> Error {
    match error {
        Error::Serialization(message) => {
            Error::Serialization(format!("section {section_type:?}: {message}"))
        }
        other => other,
    }
}

/// The error of a write to `stream` that failed with `error`: the sink's own
/// error, which the stream writer keeps with its variant (the `io::Error` it
/// returns carries only the text), or else `error` with `section_type` named.
///
/// A stream writer keeps every error it returns, so `finish` writes nothing
/// here.
pub(super) fn stream_write_error(
    stream: ChunkStreamWriter<'_>,
    section_type: SectionType,
    error: io::Error,
) -> Error {
    stream
        .finish()
        .err()
        .unwrap_or_else(|| stream_error(section_type, error))
}

impl Section for CompactStoreSection {
    fn section_type(&self) -> SectionType {
        SectionType::CompactStore
    }

    fn version(&self) -> u8 {
        FORMAT_VERSION_CHUNKED
    }

    /// The version 4 encoding, as one buffer: what spill files and the
    /// stream of [`write_to`](Section::write_to) hold.
    fn serialize(&self) -> grafeo_common::utils::error::Result<Vec<u8>> {
        self.serialize_with_version(FORMAT_VERSION)
    }

    fn deserialize(&mut self, data: &[u8]) -> grafeo_common::utils::error::Result<()> {
        // Heap-copy entry point (Section trait). Phase 3c adds
        // [`deserialize_from_bytes`](Self::deserialize_from_bytes) which
        // skips the copy on the mmap path.
        let owned = bytes::Bytes::copy_from_slice(data);
        self.deserialize_from_bytes(owned)
    }

    /// The metadata chunk, then the version 4 encoding as stream 0, cut at
    /// the section's byte cap.
    fn write_to(&self, sink: &mut dyn SectionSink) -> grafeo_common::utils::error::Result<()> {
        let meta = CompactMeta {
            layout: META_LAYOUT,
            max_bytes: self.caps.max_bytes,
        };
        write_meta(sink, SectionType::CompactStore, &meta)?;
        let mut stream = ChunkStreamWriter::new(sink, 0, COMPACT_STREAM, self.caps);
        match self.write_with_version(FORMAT_VERSION, &mut stream) {
            Ok(()) => stream.finish().map(drop),
            // Every I/O error comes from the stream.
            Err(Error::Io(error)) => {
                Err(stream_write_error(stream, SectionType::CompactStore, error))
            }
            Err(error) => Err(error),
        }
    }

    /// A single raw chunk (0.5.x bytes) goes to the 0.5.x reader. Otherwise
    /// the section is version 5: its metadata chunk and stream 0, read into
    /// one buffer, which must hold the version 4 encoding.
    fn read_from(&mut self, source: &dyn SectionSource) -> grafeo_common::utils::error::Result<()> {
        if let Some(bytes) =
            legacy_bytes(source).map_err(|e| in_section(SectionType::CompactStore, e))?
        {
            return self.load_encoding(bytes, &READABLE_ENCODINGS);
        }
        check_version(SectionType::CompactStore, source, FORMAT_VERSION_CHUNKED)?;
        let _meta: CompactMeta = read_meta(source, SectionType::CompactStore, 1)?;
        let encoding = read_stream(source, 0, COMPACT_STREAM)
            .map_err(|e| in_section(SectionType::CompactStore, e))?;
        self.load_encoding(encoding, &[FORMAT_VERSION])
    }

    fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    fn mark_clean(&self) {
        self.dirty.store(false, Ordering::Release);
    }

    fn memory_usage(&self) -> usize {
        self.store.read().as_ref().map_or(0, |s| s.memory_bytes())
    }
}

// ── Deserialization ────────────────────────────────────────────────

/// Reads a single column codec body, dispatching by section version.
///
/// - v1 → [`ColumnCodec::read_from`] (flat layout, no per-block stats)
/// - v2 → [`ColumnCodec::read_from_v2`] (block index, no stats)
/// - v3 → [`ColumnCodec::read_from_v3`] (block index + per-block stats)
///
/// Returns the codec and an `Option<Vec<ZoneMap>>` carrying per-block
/// stats when the v3 path was taken.
fn read_codec(
    data: &Bytes,
    pos: &mut usize,
    version: u8,
) -> Result<(ColumnCodec, Option<Vec<ZoneMap>>), String> {
    match version {
        FORMAT_VERSION_V1 => ColumnCodec::read_from(data, pos)
            .map(|c| (c, None))
            .map_err(|e| e.to_string()),
        FORMAT_VERSION_V2 => ColumnCodec::read_from_v2(data, pos)
            .map(|c| (c, None))
            .map_err(|e| e.to_string()),
        FORMAT_VERSION_V3 | FORMAT_VERSION => ColumnCodec::read_from_v3(data, pos)
            .map(|(c, stats)| (c, Some(stats)))
            .map_err(|e| e.to_string()),
        _ => Err(format!("unsupported CompactStore version {version}")),
    }
}

/// Reads a column: its codec body (see [`read_codec`]) and, from v4, which
/// rows have a value. Before v4 every row has one.
fn read_column(
    data: &Bytes,
    pos: &mut usize,
    version: u8,
) -> Result<(CompactColumn, Option<Vec<ZoneMap>>), String> {
    let (codec, block_stats) = read_codec(data, pos, version)?;
    let present = if version >= FORMAT_VERSION {
        CompactColumn::read_present(data, pos).map_err(str::to_string)?
    } else {
        None
    };
    let column = match present {
        Some(present) if present.len() == codec.len() => {
            CompactColumn::with_present(codec, present)
        }
        Some(present) => {
            return Err(format!(
                "column presence has {} rows, the column {}",
                present.len(),
                codec.len()
            ));
        }
        None => CompactColumn::new(codec),
    };
    Ok((column, block_stats))
}

/// Reads a store from `data_bytes`, an encoding of one of the versions
/// `accepted`.
fn deserialize_compact_store(
    data_bytes: &bytes::Bytes,
    accepted: &[u8],
) -> Result<CompactStore, String> {
    let data: &[u8] = data_bytes.as_ref();
    if data.len() < 10 {
        return Err("data too short for CompactStore section".into());
    }

    // Verify CRC32.
    let payload = &data[..data.len() - 4];
    let stored_crc = u32::from_le_bytes([
        data[data.len() - 4],
        data[data.len() - 3],
        data[data.len() - 2],
        data[data.len() - 1],
    ]);
    let computed_crc = crc32fast::hash(payload);
    if stored_crc != computed_crc {
        return Err(format!(
            "CRC32 mismatch: stored {stored_crc:#010X}, computed {computed_crc:#010X}"
        ));
    }

    let mut pos = 0;

    // Header.
    if data[pos..pos + 4] != MAGIC {
        return Err("bad magic".into());
    }
    pos += 4;
    let version = data[pos];
    pos += 1;
    if !accepted.contains(&version) {
        return Err(format!(
            "unsupported CompactStore section version {version} (supported here: {accepted:?})"
        ));
    }
    let flags = data[pos];
    pos += 1;
    let preserves_ids = flags & 0x01 != 0;

    // Node tables. Every count read below is untrusted: it is checked
    // against the bytes left before anything is allocated for it.
    let num_node_tables = read_u32(data, &mut pos)? as usize;
    check_table_count(num_node_tables, "node tables")?;
    fits(
        num_node_tables,
        NODE_TABLE_MIN_BYTES,
        data,
        pos,
        "node tables",
    )?;
    let mut node_tables = Vec::with_capacity(num_node_tables);
    let mut table_labels: Vec<Vec<arcstr::ArcStr>> = Vec::with_capacity(num_node_tables);
    // Each table is named by its key: the labels of its nodes, escaped since
    // the version 4 encoding (see `labels_of_key`).
    let labels_of = if version >= FORMAT_VERSION {
        labels_of_key
    } else {
        labels_of_unescaped_key
    };

    for table_idx in 0..num_node_tables {
        let table_id = u16::try_from(table_idx).map_err(|_| "node table id overflow")?;
        let labels = labels_of(&read_string(data, &mut pos)?);
        let row_count = read_u32(data, &mut pos)? as usize;
        let num_cols = read_u32(data, &mut pos)? as usize;
        fits(num_cols, COLUMN_MIN_BYTES, data, pos, "columns")?;

        let mut columns: FxHashMap<PropertyKey, CompactColumn> = FxHashMap::default();
        let mut zone_maps: FxHashMap<PropertyKey, ZoneMap> = FxHashMap::default();
        let mut block_zone_maps: FxHashMap<PropertyKey, Vec<ZoneMap>> = FxHashMap::default();
        let mut col_defs = Vec::with_capacity(num_cols);

        for _ in 0..num_cols {
            let key_str = read_string(data, &mut pos)?;
            let key = PropertyKey::new(&key_str);

            let has_zm = *data.get(pos).ok_or("truncated zone map flag")?;
            pos += 1;
            if has_zm == 1 {
                let zm = read_zone_map(data, &mut pos)?;
                zone_maps.insert(key.clone(), zm);
            }

            let (column, maybe_block_stats) =
                read_column(data_bytes, &mut pos, version).map_err(|e| format!("codec: {e}"))?;
            if let Some(stats) = maybe_block_stats {
                block_zone_maps.insert(key.clone(), stats);
            }
            let col_type = infer_column_type_from_codec(column.codec());
            col_defs.push(ColumnDef::new(&key_str, col_type));
            columns.insert(key, column);
        }

        // The key as this build writes it, whichever build wrote the file.
        let schema = TableSchema::new(label_set_key(&labels), table_id, col_defs);
        let table = NodeTable::from_columns_with_block_stats(
            schema,
            columns,
            zone_maps,
            block_zone_maps,
            row_count,
        );
        node_tables.push(table);
        table_labels.push(labels);
    }
    let node_rows: Vec<usize> = node_tables.iter().map(NodeTable::len).collect();

    // Relationship tables.
    let num_rel_tables = read_u32(data, &mut pos)? as usize;
    check_table_count(num_rel_tables, "relationship tables")?;
    fits(
        num_rel_tables,
        REL_TABLE_MIN_BYTES,
        data,
        pos,
        "relationship tables",
    )?;
    let mut rel_tables = Vec::with_capacity(num_rel_tables);
    let mut edge_type_to_rel_id: FxHashMap<arcstr::ArcStr, Vec<u16>> = FxHashMap::default();
    let mut rel_table_id_to_type: Vec<arcstr::ArcStr> = Vec::with_capacity(num_rel_tables);

    for rel_idx in 0..num_rel_tables {
        let rel_table_id = u16::try_from(rel_idx).map_err(|_| "relationship table id overflow")?;
        let edge_type = read_string(data, &mut pos)?;
        let edge_type = arcstr::ArcStr::from(edge_type.as_str());
        let src_tid = read_u16(data, &mut pos)?;
        let dst_tid = read_u16(data, &mut pos)?;
        let (Some(&src_rows), Some(&dst_rows)) = (
            node_rows.get(usize::from(src_tid)),
            node_rows.get(usize::from(dst_tid)),
        ) else {
            return Err(format!(
                "relationship table {rel_idx} ({edge_type}) joins tables {src_tid} and \
                 {dst_tid} of {}",
                node_rows.len()
            ));
        };

        let fwd = CsrAdjacency::read_from(data, &mut pos).map_err(|e| format!("fwd CSR: {e}"))?;
        check_adjacency(&fwd, src_rows, dst_rows, rel_idx, "forward")?;

        let has_bwd = *data.get(pos).ok_or("truncated bwd flag")?;
        pos += 1;
        let bwd = if has_bwd == 1 {
            let bwd =
                CsrAdjacency::read_from(data, &mut pos).map_err(|e| format!("bwd CSR: {e}"))?;
            check_adjacency(&bwd, dst_rows, src_rows, rel_idx, "backward")?;
            if bwd.num_edges() != fwd.num_edges() {
                return Err(format!(
                    "relationship table {rel_idx} has {} edges backward and {} forward",
                    bwd.num_edges(),
                    fwd.num_edges()
                ));
            }
            if let Some(&position) = bwd
                .edge_data()
                .and_then(|positions| positions.iter().find(|&&p| p as usize >= fwd.num_edges()))
            {
                return Err(format!(
                    "the backward adjacency of relationship table {rel_idx} maps to forward \
                     edge {position} of {}",
                    fwd.num_edges()
                ));
            }
            Some(bwd)
        } else {
            None
        };

        let num_props = read_u32(data, &mut pos)? as usize;
        fits(
            num_props,
            EDGE_COLUMN_MIN_BYTES,
            data,
            pos,
            "edge property columns",
        )?;
        let mut properties: FxHashMap<PropertyKey, CompactColumn> = FxHashMap::default();
        let mut prop_defs = Vec::with_capacity(num_props);
        for _ in 0..num_props {
            let key_str = read_string(data, &mut pos)?;
            let key = PropertyKey::new(&key_str);
            let (column, _block_stats) = read_column(data_bytes, &mut pos, version)
                .map_err(|e| format!("edge codec: {e}"))?;
            let col_type = infer_column_type_from_codec(column.codec());
            prop_defs.push(ColumnDef::new(&key_str, col_type));
            properties.insert(key, column);
        }

        let schema = EdgeSchema::new(
            edge_type.as_str(),
            rel_table_id,
            node_tables[usize::from(src_tid)].label(),
            node_tables[usize::from(dst_tid)].label(),
            prop_defs,
        );

        let table = RelTable::new(schema, fwd, bwd, properties, src_tid, dst_tid);
        edge_type_to_rel_id
            .entry(edge_type.clone())
            .or_default()
            .push(rel_table_id);
        rel_table_id_to_type.push(edge_type);
        rel_tables.push(table);
    }

    let edge_rows: Vec<usize> = rel_tables.iter().map(RelTable::num_edges).collect();

    // The store computes the statistics.
    let mut store = CompactStore::new(
        node_tables,
        table_labels,
        rel_tables,
        edge_type_to_rel_id,
        rel_table_id_to_type,
    );

    // ID maps.
    if preserves_ids {
        let (node_id_map, node_offset_to_id) = read_id_map(data, &mut pos, &node_rows)?;
        let (edge_id_map, edge_offset_to_id) = read_id_map(data, &mut pos, &edge_rows)?;
        store.set_id_maps(
            node_id_map,
            edge_id_map,
            node_offset_to_id,
            edge_offset_to_id,
        );
    }

    Ok(store)
}

/// The fewest bytes a node table takes: its label's length, its row count
/// and its column count.
const NODE_TABLE_MIN_BYTES: usize = 2 + 4 + 4;

/// The fewest bytes a node property column takes: its key's length, the zone
/// map flag and the codec's discriminant.
const COLUMN_MIN_BYTES: usize = 2 + 1 + 1;

/// The fewest bytes a relationship table takes: its edge type's length, its
/// two table ids, an empty adjacency (two counts and the edge data flag), the
/// backward flag and the property count.
const REL_TABLE_MIN_BYTES: usize = 2 + 2 + 2 + 9 + 1 + 4;

/// The fewest bytes an edge property column takes: its key's length and the
/// codec's discriminant.
const EDGE_COLUMN_MIN_BYTES: usize = 2 + 1;

/// The bytes of an id map entry: the id, the table and the row.
const ID_MAP_ENTRY_BYTES: usize = 8 + 2 + 8;

/// Refuses more tables than table ids can name.
fn check_table_count(count: usize, what: &str) -> Result<(), String> {
    let most = usize::from(super::id::MAX_TABLE_ID) + 1;
    if count > most {
        return Err(format!(
            "{count} {what}, more than the {most} a compacted base holds"
        ));
    }
    Ok(())
}

/// Refuses `count` items of at least `each` bytes when the bytes left after
/// `pos` cannot hold them, so that nothing is allocated from a count no file
/// could hold.
fn fits(count: usize, each: usize, data: &[u8], pos: usize, what: &str) -> Result<(), String> {
    let left = data.len().saturating_sub(pos);
    if count.checked_mul(each).is_none_or(|needed| needed > left) {
        return Err(format!("{count} {what} do not fit the {left} bytes left"));
    }
    Ok(())
}

/// Refuses an adjacency of relationship table `table` that does not fit the
/// tables it joins: one node per row of the table its edges leave from
/// (`from_rows`), and every target a row of the table they reach (`to_rows`).
fn check_adjacency(
    csr: &CsrAdjacency,
    from_rows: usize,
    to_rows: usize,
    table: usize,
    direction: &str,
) -> Result<(), String> {
    if csr.num_nodes() != from_rows {
        return Err(format!(
            "the {direction} adjacency of relationship table {table} has {} nodes, the table \
             its edges leave from {from_rows} rows",
            csr.num_nodes()
        ));
    }
    if let Some(&target) = csr
        .targets()
        .iter()
        .find(|&&target| target as usize >= to_rows)
    {
        return Err(format!(
            "the {direction} adjacency of relationship table {table} names row {target} of a \
             table of {to_rows} rows"
        ));
    }
    Ok(())
}

/// Reads an id map: its length, then `(id, table, row)` per entry. Every row
/// of every table has exactly one entry (what the writer writes), so the
/// length must be the rows of all tables, each table and row must exist, and
/// no row or id may come twice. The reverse is allocated from the rows, which
/// the length (checked against the bytes left) bounds.
fn read_id_map<Id: MapId>(
    data: &[u8],
    pos: &mut usize,
    rows: &[usize],
) -> Result<(FxHashMap<Id, (u16, u64)>, Vec<Vec<Id>>), String> {
    let what = Id::WHAT;
    let len = read_u32(data, pos)? as usize;
    fits(
        len,
        ID_MAP_ENTRY_BYTES,
        data,
        *pos,
        &format!("{what} id map entries"),
    )?;
    let total = rows
        .iter()
        .try_fold(0usize, |total, &rows| total.checked_add(rows))
        .ok_or_else(|| format!("the {what} tables hold more rows than this platform counts"))?;
    if len != total {
        return Err(format!(
            "the {what} id map holds {len} entries for {total} rows"
        ));
    }
    let mut map = FxHashMap::with_capacity_and_hasher(len, Default::default());
    let mut reverse: Vec<Vec<Id>> = rows.iter().map(|&count| vec![Id::INVALID; count]).collect();
    for _ in 0..len {
        let entry = read_u64(data, pos)?;
        let table = read_u16(data, pos)?;
        let row = read_u64(data, pos)?;
        let entry_id = Id::from_raw(entry);
        if entry_id == Id::INVALID {
            return Err(format!("the {what} id map holds the invalid id {entry}"));
        }
        let tables = reverse.len();
        let slots = reverse.get_mut(usize::from(table)).ok_or_else(|| {
            format!("the {what} id map puts id {entry} in table {table} of {tables}")
        })?;
        let table_rows = slots.len();
        let slot = usize::try_from(row)
            .ok()
            .and_then(|row| slots.get_mut(row))
            .ok_or_else(|| {
                format!(
                    "the {what} id map puts id {entry} at row {row} of table {table}, which has \
                     {table_rows} rows"
                )
            })?;
        if *slot != Id::INVALID {
            return Err(format!(
                "the {what} id map puts ids {} and {entry} both at table {table}, row {row}",
                slot.raw()
            ));
        }
        *slot = entry_id;
        if map.insert(entry_id, (table, row)).is_some() {
            return Err(format!("the {what} id map lists id {entry} twice"));
        }
    }
    Ok((map, reverse))
}

// ── Write helpers ──────────────────────────────────────────────────

fn write_u16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_len(buf: &mut Vec<u8>, v: usize) -> grafeo_common::utils::error::Result<()> {
    let n = checked_u32(v, "compact store length")?;
    buf.extend_from_slice(&n.to_le_bytes());
    Ok(())
}

/// Labels, edge types and property keys are stored with a `u16` length, so a
/// name over 64 KiB cannot be written.
fn write_str(buf: &mut Vec<u8>, s: &str) -> grafeo_common::utils::error::Result<()> {
    let bytes = s.as_bytes();
    write_u16(buf, checked_u16(bytes.len(), "compact store name length")?);
    buf.extend_from_slice(bytes);
    Ok(())
}

fn write_zone_map(buf: &mut Vec<u8>, zm: &ZoneMap) -> grafeo_common::utils::error::Result<()> {
    write_len(buf, zm.null_count)?;
    write_len(buf, zm.row_count)?;
    // Encode min/max as (tag, value) pairs.
    write_optional_value(buf, &zm.min)?;
    write_optional_value(buf, &zm.max)?;
    Ok(())
}

fn write_optional_value(
    buf: &mut Vec<u8>,
    v: &Option<grafeo_common::types::Value>,
) -> grafeo_common::utils::error::Result<()> {
    match v {
        None => buf.push(0),
        Some(grafeo_common::types::Value::Int64(n)) => {
            buf.push(1);
            // Store as raw i64 bytes to avoid sign-loss lint.
            buf.extend_from_slice(&n.to_le_bytes());
        }
        Some(grafeo_common::types::Value::Bool(b)) => {
            buf.push(2);
            buf.push(u8::from(*b));
        }
        Some(grafeo_common::types::Value::String(s)) => {
            buf.push(3);
            write_str(buf, s.as_str())?;
        }
        Some(_) => {
            // Unsupported type for zone map: write as absent.
            buf.push(0);
        }
    }
    Ok(())
}

// ── Read helpers ───────────────────────────────────────────────────

fn read_u16(data: &[u8], pos: &mut usize) -> Result<u16, String> {
    if *pos + 2 > data.len() {
        return Err("truncated u16".into());
    }
    let v = u16::from_le_bytes([data[*pos], data[*pos + 1]]);
    *pos += 2;
    Ok(v)
}

fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32, String> {
    if *pos + 4 > data.len() {
        return Err("truncated u32".into());
    }
    let v = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
    Ok(v)
}

fn read_u64(data: &[u8], pos: &mut usize) -> Result<u64, String> {
    if *pos + 8 > data.len() {
        return Err("truncated u64".into());
    }
    let v = u64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    Ok(v)
}

fn read_string(data: &[u8], pos: &mut usize) -> Result<String, String> {
    let slen = read_u16(data, pos)? as usize;
    if *pos + slen > data.len() {
        return Err("truncated string".into());
    }
    let s =
        std::str::from_utf8(&data[*pos..*pos + slen]).map_err(|_| "invalid UTF-8".to_string())?;
    *pos += slen;
    Ok(s.to_string())
}

fn read_zone_map(data: &[u8], pos: &mut usize) -> Result<ZoneMap, String> {
    let null_count = read_u32(data, pos)? as usize;
    let row_count = read_u32(data, pos)? as usize;
    let min = read_optional_value(data, pos)?;
    let max = read_optional_value(data, pos)?;
    Ok(ZoneMap {
        min,
        max,
        null_count,
        row_count,
    })
}

fn read_optional_value(
    data: &[u8],
    pos: &mut usize,
) -> Result<Option<grafeo_common::types::Value>, String> {
    let tag = *data.get(*pos).ok_or("truncated value tag")?;
    *pos += 1;
    match tag {
        0 => Ok(None),
        1 => {
            // Read raw i64 bytes (written via i64::to_le_bytes).
            if *pos + 8 > data.len() {
                return Err("truncated i64 value".into());
            }
            let v = i64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
            *pos += 8;
            Ok(Some(grafeo_common::types::Value::Int64(v)))
        }
        2 => {
            let b = *data.get(*pos).ok_or("truncated bool")?;
            *pos += 1;
            Ok(Some(grafeo_common::types::Value::Bool(b != 0)))
        }
        3 => {
            let s = read_string(data, pos)?;
            Ok(Some(grafeo_common::types::Value::String(
                arcstr::ArcStr::from(s.as_str()),
            )))
        }
        _ => Err(format!("unknown value tag {tag}")),
    }
}

fn infer_column_type_from_codec(codec: &ColumnCodec) -> ColumnType {
    match codec {
        ColumnCodec::BitPacked(bp) => ColumnType::UInt {
            bits: bp.bits_per_value(),
        },
        ColumnCodec::Dict(_) => ColumnType::DictString,
        ColumnCodec::Bitmap(_) => ColumnType::Bool,
        ColumnCodec::Int8Vector { dimensions, .. } => ColumnType::Int8Vector {
            dimensions: *dimensions,
        },
        ColumnCodec::Float64(_) => ColumnType::Float64,
        ColumnCodec::Float32Vector { dimensions, .. } => ColumnType::Float32Vector {
            dimensions: *dimensions,
        },
        ColumnCodec::RawI64(_) => ColumnType::Int64,
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::compact::from_graph_store_preserving_ids;
    use crate::graph::lpg::LpgStore;
    use crate::graph::traits::GraphStore;
    use grafeo_common::storage::{
        ChunkCaps, ChunkMeta, ImageSource, MemoryImage, SectionSink, read_stream,
    };
    use grafeo_common::testing::chunk_caps::with_chunk_caps;
    use grafeo_common::types::Value;
    use grafeo_common::utils::error::Error;

    #[test]
    fn test_round_trip_empty() {
        let store = LpgStore::new().unwrap();
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let section = CompactStoreSection::new(Arc::new(compact));

        let bytes = section.serialize().unwrap();
        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&bytes).unwrap();

        let restored = section2.store().unwrap();
        assert_eq!(restored.node_count(), 0);
        assert_eq!(restored.edge_count(), 0);
    }

    /// Names are stored with a u16 length: one over 64 KiB is a serialization
    /// error (it used to panic), and names at the limit still round-trip.
    #[test]
    fn test_name_over_u16_limit_is_an_error_not_a_panic() {
        let too_long = "x".repeat(usize::from(u16::MAX) + 1);
        for (what, store) in [
            ("label", {
                let store = LpgStore::new().unwrap();
                store.create_node(&[too_long.as_str()]);
                store
            }),
            ("property key", {
                let store = LpgStore::new().unwrap();
                let n = store.create_node(&["Person"]);
                store.set_node_property(n, &too_long, Value::Int64(1));
                store
            }),
            ("string zone map bound", {
                let store = LpgStore::new().unwrap();
                let n = store.create_node(&["Person"]);
                store.set_node_property(n, "name", Value::from(too_long.as_str()));
                store
            }),
        ] {
            let compact = from_graph_store_preserving_ids(&store).unwrap();
            let err = CompactStoreSection::new(Arc::new(compact))
                .serialize()
                .expect_err(what);
            assert!(
                err.to_string().contains("compact store name length: 65536"),
                "{what}: {err}"
            );
        }

        let at_limit = "x".repeat(usize::from(u16::MAX));
        let store = LpgStore::new().unwrap();
        store.create_node(&[at_limit.as_str()]);
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let bytes = CompactStoreSection::new(Arc::new(compact))
            .serialize()
            .unwrap();
        let mut restored = CompactStoreSection::empty();
        restored.deserialize(&bytes).unwrap();
        assert_eq!(restored.store().unwrap().node_count(), 1);
    }

    #[test]
    fn test_round_trip_nodes_and_edges() {
        let store = LpgStore::new().unwrap();
        let alix = store.create_node(&["Person"]);
        store.set_node_property(alix, "name", Value::from("Alix"));
        store.set_node_property(alix, "age", Value::Int64(30));

        let gus = store.create_node(&["Person"]);
        store.set_node_property(gus, "name", Value::from("Gus"));
        store.set_node_property(gus, "age", Value::Int64(25));

        let amsterdam = store.create_node(&["City"]);
        store.set_node_property(amsterdam, "name", Value::from("Amsterdam"));

        store.create_edge(alix, amsterdam, "LIVES_IN");
        store.create_edge(gus, amsterdam, "LIVES_IN");

        let compact = from_graph_store_preserving_ids(&store).unwrap();
        assert!(compact.preserves_ids());

        let section = CompactStoreSection::new(Arc::new(compact));
        let bytes = section.serialize().unwrap();

        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&bytes).unwrap();
        let restored = section2.store().unwrap();

        assert!(restored.preserves_ids());
        assert_eq!(restored.node_count(), 3);
        assert_eq!(restored.edge_count(), 2);

        // Verify original IDs survive.
        let alix_node = restored.get_node(alix).expect("Alix by original ID");
        assert_eq!(
            alix_node.properties.get(&PropertyKey::new("name")),
            Some(&Value::String(arcstr::ArcStr::from("Alix")))
        );
        assert_eq!(
            alix_node.properties.get(&PropertyKey::new("age")),
            Some(&Value::Int64(30))
        );

        // Verify edge traversal.
        let neighbors = restored.neighbors(alix, crate::graph::Direction::Outgoing);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0], amsterdam);
    }

    #[test]
    fn test_round_trip_without_id_preservation() {
        use crate::graph::compact::from_graph_store;

        let lpg = LpgStore::new().unwrap();
        let a = lpg.create_node(&["Node"]);
        lpg.set_node_property(a, "val", Value::Int64(42));
        let b = lpg.create_node(&["Node"]);
        lpg.set_node_property(b, "val", Value::Int64(99));
        lpg.create_edge(a, b, "LINK");

        let compact = from_graph_store(&lpg).unwrap();
        assert!(!compact.preserves_ids());

        let section = CompactStoreSection::new(Arc::new(compact));
        let bytes = section.serialize().unwrap();

        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&bytes).unwrap();
        let restored = section2.store().unwrap();

        assert!(!restored.preserves_ids());
        assert_eq!(restored.node_count(), 2);
        assert_eq!(restored.edge_count(), 1);
    }

    #[test]
    fn test_crc_integrity() {
        let store = LpgStore::new().unwrap();
        store.create_node(&["Test"]);
        let compact = from_graph_store_preserving_ids(&store).unwrap();

        let section = CompactStoreSection::new(Arc::new(compact));
        let mut bytes = section.serialize().unwrap();

        // Corrupt a byte in the middle.
        if bytes.len() > 10 {
            bytes[10] ^= 0xFF;
        }

        let mut section2 = CompactStoreSection::empty();
        assert!(section2.deserialize(&bytes).is_err());
    }

    #[test]
    fn test_section_type_and_version() {
        let section = CompactStoreSection::empty();
        assert_eq!(section.section_type(), SectionType::CompactStore);
        assert_eq!(section.version(), FORMAT_VERSION_CHUNKED);
        assert!(!section.is_dirty());
        assert_eq!(section.memory_usage(), 0);
    }

    #[test]
    fn test_dirty_tracking() {
        let section = CompactStoreSection::empty();
        assert!(!section.is_dirty());
        section.mark_dirty();
        assert!(section.is_dirty());
        section.mark_clean();
        assert!(!section.is_dirty());
    }

    /// Phase 2b: confirm the v1 (flat-column) on-disk format still
    /// round-trips through the v2-aware deserializer, exercising the
    /// compat path users on 0.5.41 and earlier rely on for one release.
    #[test]
    fn nelson_v1_section_reads_through_v2_aware_deserializer() {
        let store = LpgStore::new().unwrap();
        let alix = store.create_node(&["Person"]);
        store.set_node_property(alix, "name", Value::from("Alix"));
        store.set_node_property(alix, "age", Value::Int64(30));

        let gus = store.create_node(&["Person"]);
        store.set_node_property(gus, "name", Value::from("Gus"));
        store.set_node_property(gus, "age", Value::Int64(25));

        store.create_edge(alix, gus, "KNOWS");

        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let section = CompactStoreSection::new(Arc::new(compact));

        // Force v1 layout (flat columns, version byte = 1).
        let v1_bytes = section.serialize_with_version(FORMAT_VERSION_V1).unwrap();
        // First byte after MAGIC must be the v1 marker.
        assert_eq!(
            v1_bytes[4], FORMAT_VERSION_V1,
            "expected v1 marker in version byte"
        );

        // The v2-aware deserializer must handle both versions.
        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&v1_bytes).unwrap();
        let restored = section2.store().unwrap();

        assert_eq!(restored.node_count(), 2);
        assert_eq!(restored.edge_count(), 1);
        assert_eq!(
            restored.get_node_property(alix, &PropertyKey::new("name")),
            Some(Value::String(arcstr::ArcStr::from("Alix")))
        );
        assert_eq!(
            restored.get_node_property(alix, &PropertyKey::new("age")),
            Some(Value::Int64(30))
        );
    }

    // ── Phase 2c: per-block zone maps ────────────────────────────────

    /// The builder must populate per-block zone maps for every column,
    /// one ZoneMap per block. `1024` rows per block (DEFAULT_BLOCK_ROWS).
    #[test]
    fn alix_builder_populates_per_block_zone_maps() {
        let store = LpgStore::new().unwrap();
        // 3000 nodes → 3 blocks (1024 + 1024 + 952).
        for i in 0i64..3000 {
            let n = store.create_node(&["Person"]);
            store.set_node_property(n, "age", Value::Int64(i));
        }
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let table = &compact.node_tables_by_id[0];
        let block_zms = table
            .block_zone_maps_for(&PropertyKey::new("age"))
            .expect("per-block stats present");
        assert_eq!(block_zms.len(), 3, "3000 rows should produce 3 blocks");
        assert_eq!(block_zms[0].row_count, 1024);
        assert_eq!(block_zms[1].row_count, 1024);
        assert_eq!(block_zms[2].row_count, 952);
        assert_eq!(block_zms[0].min, Some(Value::Int64(0)));
        assert_eq!(block_zms[0].max, Some(Value::Int64(1023)));
        assert_eq!(block_zms[1].min, Some(Value::Int64(1024)));
        assert_eq!(block_zms[1].max, Some(Value::Int64(2047)));
        assert_eq!(block_zms[2].min, Some(Value::Int64(2048)));
        assert_eq!(block_zms[2].max, Some(Value::Int64(2999)));
    }

    /// v3 round-trip preserves per-block zone maps verbatim.
    #[test]
    fn gus_v3_round_trip_preserves_block_zone_maps() {
        let store = LpgStore::new().unwrap();
        for i in 0i64..2500 {
            let n = store.create_node(&["Item"]);
            store.set_node_property(n, "score", Value::Int64(i));
        }
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let original = &compact.node_tables_by_id[0];
        let original_zms = original
            .block_zone_maps_for(&PropertyKey::new("score"))
            .expect("original block stats")
            .to_vec();

        let section = CompactStoreSection::new(Arc::new(compact));
        let bytes = section.serialize().unwrap();
        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&bytes).unwrap();
        let restored = section2.store().unwrap();
        let restored_table = &restored.node_tables_by_id[0];
        let restored_zms = restored_table
            .block_zone_maps_for(&PropertyKey::new("score"))
            .expect("restored block stats");

        assert_eq!(restored_zms.len(), original_zms.len());
        for (i, (orig, rest)) in original_zms.iter().zip(restored_zms.iter()).enumerate() {
            assert_eq!(orig.row_count, rest.row_count, "row_count mismatch at {i}");
            assert_eq!(
                orig.null_count, rest.null_count,
                "null_count mismatch at {i}"
            );
            assert_eq!(orig.min, rest.min, "min mismatch at {i}");
            assert_eq!(orig.max, rest.max, "max mismatch at {i}");
        }
    }

    /// v2 sections (Phase 2b) carry no per-block zone maps; the v3 reader
    /// must accept them and leave `block_zone_maps_for` returning `None`.
    #[test]
    fn vincent_v2_section_round_trip_leaves_block_zone_maps_empty() {
        let store = LpgStore::new().unwrap();
        for i in 0i64..1500 {
            let n = store.create_node(&["Item"]);
            store.set_node_property(n, "score", Value::Int64(i));
        }
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let section = CompactStoreSection::new(Arc::new(compact));
        let v2_bytes = section.serialize_with_version(FORMAT_VERSION_V2).unwrap();
        assert_eq!(v2_bytes[4], FORMAT_VERSION_V2);

        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&v2_bytes).unwrap();
        let restored = section2.store().unwrap();
        let table = &restored.node_tables_by_id[0];
        assert!(
            table
                .block_zone_maps_for(&PropertyKey::new("score"))
                .is_none(),
            "v2 stream must not populate block_zone_maps"
        );
        // But the column data still survives.
        assert_eq!(table.len(), 1500);
    }

    /// v1 sections likewise carry no per-block stats.
    #[test]
    fn jules_v1_section_round_trip_leaves_block_zone_maps_empty() {
        let store = LpgStore::new().unwrap();
        for i in 0i64..1500 {
            let n = store.create_node(&["Item"]);
            store.set_node_property(n, "score", Value::Int64(i));
        }
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let section = CompactStoreSection::new(Arc::new(compact));
        let v1_bytes = section.serialize_with_version(FORMAT_VERSION_V1).unwrap();
        assert_eq!(v1_bytes[4], FORMAT_VERSION_V1);

        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&v1_bytes).unwrap();
        let restored = section2.store().unwrap();
        let table = &restored.node_tables_by_id[0];
        assert!(
            table
                .block_zone_maps_for(&PropertyKey::new("score"))
                .is_none(),
            "v1 stream must not populate block_zone_maps"
        );
        assert_eq!(table.len(), 1500);
    }

    /// String columns also get per-block min/max.
    #[test]
    fn mia_block_zone_maps_for_string_column() {
        let store = LpgStore::new().unwrap();
        // Use enough nodes to force >= 2 blocks.
        for i in 0u32..1100 {
            let n = store.create_node(&["Tag"]);
            store.set_node_property(n, "name", Value::from(format!("tag_{i:04}")));
        }
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let table = &compact.node_tables_by_id[0];
        let block_zms = table
            .block_zone_maps_for(&PropertyKey::new("name"))
            .expect("string column block stats");
        assert_eq!(block_zms.len(), 2);
        assert_eq!(
            block_zms[0].min,
            Some(Value::String(arcstr::ArcStr::from("tag_0000")))
        );
        assert_eq!(
            block_zms[0].max,
            Some(Value::String(arcstr::ArcStr::from("tag_1023")))
        );
        assert_eq!(
            block_zms[1].min,
            Some(Value::String(arcstr::ArcStr::from("tag_1024")))
        );
        assert_eq!(
            block_zms[1].max,
            Some(Value::String(arcstr::ArcStr::from("tag_1099")))
        );
    }

    /// Phase 2b: an unsupported version byte must produce a clean error,
    /// not panic or silently misread the section.
    #[test]
    fn rita_unknown_version_returns_clear_error() {
        let store = LpgStore::new().unwrap();
        let _ = store.create_node(&["Item"]);
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let section = CompactStoreSection::new(Arc::new(compact));
        let mut bytes = section.serialize().unwrap();
        // Strip CRC, flip version byte to a future v9, recompute CRC.
        let crc_pos = bytes.len() - 4;
        bytes[4] = 9;
        let crc = crc32fast::hash(&bytes[..crc_pos]);
        bytes[crc_pos..].copy_from_slice(&crc.to_le_bytes());

        let mut section2 = CompactStoreSection::empty();
        let err = section2
            .deserialize(&bytes)
            .expect_err("expected version error");
        let msg = err.to_string();
        assert!(
            msg.contains("unsupported CompactStore section version"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn test_round_trip_bool_column() {
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["Item"]);
        store.set_node_property(a, "active", Value::Bool(true));
        let b = store.create_node(&["Item"]);
        store.set_node_property(b, "active", Value::Bool(false));

        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let section = CompactStoreSection::new(Arc::new(compact));
        let bytes = section.serialize().unwrap();

        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&bytes).unwrap();
        let restored = section2.store().unwrap();

        assert_eq!(
            restored.get_node_property(a, &PropertyKey::new("active")),
            Some(Value::Bool(true))
        );
        assert_eq!(
            restored.get_node_property(b, &PropertyKey::new("active")),
            Some(Value::Bool(false))
        );
    }

    #[test]
    fn test_round_trip_edge_properties() {
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let e = store.create_edge(a, b, "LINK");
        store.set_edge_property(e, "weight", Value::Int64(5));

        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let section = CompactStoreSection::new(Arc::new(compact));
        let bytes = section.serialize().unwrap();

        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&bytes).unwrap();
        let restored = section2.store().unwrap();

        // Find the edge via traversal.
        let edges = restored.edges_from(a, crate::graph::Direction::Outgoing);
        assert_eq!(edges.len(), 1);
        let edge = restored.get_edge(edges[0].1).unwrap();
        assert_eq!(
            edge.properties.get(&PropertyKey::new("weight")),
            Some(&Value::Int64(5))
        );
    }

    /// A store where one `:P` node and one `:R` edge lack properties that
    /// the others have. Returns the store and the ids of the sparse node and
    /// edge.
    fn store_with_missing_properties() -> (LpgStore, NodeId, EdgeId) {
        let store = LpgStore::new().unwrap();
        let full = store.create_node(&["P"]);
        let sparse = store.create_node(&["P"]);
        store.set_node_property(full, "n", Value::Int64(1));
        store.set_node_property(full, "u", Value::Int64(19));
        store.set_node_property(full, "s", Value::from("Amsterdam"));
        store.set_node_property(full, "b", Value::Bool(true));
        store.set_node_property(sparse, "n", Value::Int64(2));
        let full_edge = store.create_edge(full, sparse, "R");
        store.set_edge_property(full_edge, "n", Value::Int64(1));
        store.set_edge_property(full_edge, "w", Value::Int64(3));
        let sparse_edge = store.create_edge(sparse, full, "R");
        store.set_edge_property(sparse_edge, "n", Value::Int64(2));
        (store, sparse, sparse_edge)
    }

    /// The current format keeps which rows have a value: after a round trip
    /// the sparse node and edge still have no value for what they lack.
    #[test]
    fn missing_values_survive_a_round_trip() {
        let (store, sparse, sparse_edge) = store_with_missing_properties();
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let bytes = CompactStoreSection::new(Arc::new(compact))
            .serialize()
            .unwrap();
        let mut section = CompactStoreSection::empty();
        section.deserialize(&bytes).unwrap();
        let restored = section.store().unwrap();

        for key in ["u", "s", "b"] {
            assert_eq!(
                restored.get_node_property(sparse, &PropertyKey::new(key)),
                None,
                "{key}"
            );
        }
        assert_eq!(
            restored.get_node_property(sparse, &PropertyKey::new("n")),
            Some(Value::Int64(2))
        );
        assert_eq!(
            restored.get_edge_property(sparse_edge, &PropertyKey::new("w")),
            None
        );
    }

    /// A v3 section (0.5.44 and older) still loads. It records no missing
    /// values, so a missing property reads the empty value stored for it.
    #[test]
    fn a_v3_section_still_loads() {
        let (store, sparse, _) = store_with_missing_properties();
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let bytes = CompactStoreSection::new(Arc::new(compact))
            .serialize_with_version(FORMAT_VERSION_V3)
            .unwrap();
        let mut section = CompactStoreSection::empty();
        section.deserialize(&bytes).unwrap();
        let restored = section.store().unwrap();

        assert_eq!(
            restored.get_node_property(sparse, &PropertyKey::new("n")),
            Some(Value::Int64(2))
        );
        assert_eq!(
            restored.get_node_property(sparse, &PropertyKey::new("u")),
            Some(Value::Int64(0))
        );
    }

    // ── Streams: section version 5 ──────────────────────────────────

    /// Two node tables and a rel table, with sparse columns: four `:City`
    /// nodes (a name each, a population for every other one) and 57
    /// `:Person` nodes (a name each, an age for every third, a score for
    /// every fourth, an `active` flag for every fifth), each `LIVES_IN` a
    /// city, every other edge with a `since` year and every third with a
    /// distance in `km`. Returns the store and its node ids (the cities
    /// first) and edge ids, in creation order.
    fn store_for_streams() -> (LpgStore, Vec<NodeId>, Vec<EdgeId>) {
        let store = LpgStore::new().unwrap();
        let people = ["Alix", "Gus", "Vincent", "Jules", "Mia"];
        let mut nodes = Vec::new();
        for (city, population) in [
            ("Amsterdam", Some(880_000)),
            ("Berlin", None),
            ("Paris", Some(1_988_000)),
            ("Prague", None),
        ] {
            let id = store.create_node(&["City"]);
            store.set_node_property(id, "name", Value::from(city));
            if let Some(population) = population {
                store.set_node_property(id, "population", Value::Int64(population));
            }
            nodes.push(id);
        }
        let mut edges = Vec::new();
        for (i, n) in (0..57usize).zip(0i64..) {
            let id = store.create_node(&["Person"]);
            store.set_node_property(id, "name", Value::from(format!("{} {n}", people[i % 5])));
            if i % 3 == 0 {
                store.set_node_property(id, "age", Value::Int64(19 + n));
            }
            if i % 4 == 0 {
                let score = f64::from(u8::try_from(i).unwrap()) * 0.88;
                store.set_node_property(id, "score", Value::Float64(score));
            }
            if i % 5 == 0 {
                store.set_node_property(id, "active", Value::Bool(i % 2 == 0));
            }
            let edge = store.create_edge(id, nodes[i % 4], "LIVES_IN");
            if i % 2 == 0 {
                store.set_edge_property(edge, "since", Value::Int64(1988 + n));
            }
            if i % 3 == 0 {
                store.set_edge_property(edge, "km", Value::Int64(88 * n));
            }
            nodes.push(id);
            edges.push(edge);
        }
        (store, nodes, edges)
    }

    /// Where two byte strings first differ (a length counts), without
    /// printing kilobytes when they do.
    fn first_difference(left: &[u8], right: &[u8]) -> Option<usize> {
        left.iter()
            .zip(right)
            .position(|(left, right)| left != right)
            .or_else(|| (left.len() != right.len()).then(|| left.len().min(right.len())))
    }

    /// The encoding depends on what the store holds, not on the hash maps
    /// that hold it: a store read back writes the bytes it was read from,
    /// although its maps iterate in another order (their seeds are random).
    /// Golden fixtures and incremental checkpoints rest on this.
    #[test]
    fn a_compact_store_writes_the_same_bytes_after_a_round_trip() {
        let (store, _, _) = store_for_streams();
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let first = CompactStoreSection::new(Arc::new(compact))
            .serialize()
            .unwrap();
        let mut bytes = first.clone();
        for round in 1..=3 {
            let mut section = CompactStoreSection::empty();
            section.deserialize(&bytes).unwrap();
            bytes = CompactStoreSection::new(section.store().unwrap())
                .serialize()
                .unwrap();
            assert_eq!(
                first_difference(&first, &bytes),
                None,
                "round trip {round} wrote other bytes"
            );
        }
    }

    /// Caps small enough that the stream of a small store spans several
    /// pieces.
    const SMALL_CAPS: ChunkCaps = ChunkCaps {
        max_rows: 3,
        max_bytes: 512,
    };

    /// The section of the store `store` compacts to, built with `caps` as
    /// this thread's caps.
    fn section_of(store: &LpgStore, caps: ChunkCaps) -> CompactStoreSection {
        let compact = Arc::new(from_graph_store_preserving_ids(store).unwrap());
        with_chunk_caps(caps, || CompactStoreSection::new(compact))
    }

    /// The image holding what `section` writes.
    fn image_of(section: &CompactStoreSection) -> MemoryImage {
        MemoryImage::from_sections(&[section as &dyn Section]).unwrap()
    }

    /// The CompactStore section of `image`, read into an empty section.
    fn load(image: &MemoryImage) -> grafeo_common::utils::error::Result<CompactStoreSection> {
        let mut section = CompactStoreSection::empty();
        section.read_from(&*image.section_source(SectionType::CompactStore).unwrap())?;
        Ok(section)
    }

    /// A section of version `version` holding `chunks`, as a crafted file
    /// would.
    fn crafted(version: u8, chunks: &[(ChunkMeta, Vec<u8>)]) -> MemoryImage {
        let mut image = MemoryImage::new();
        image
            .begin_section(SectionType::CompactStore, version)
            .unwrap();
        for (meta, bytes) in chunks {
            image.write_chunk(*meta, bytes).unwrap();
        }
        image
    }

    /// The bytes of a metadata chunk with `layout`.
    fn meta_chunk(layout: u8) -> (ChunkMeta, Vec<u8>) {
        let meta = CompactMeta {
            layout,
            max_bytes: 512,
        };
        let bytes = bincode::serde::encode_to_vec(meta, bincode::config::standard()).unwrap();
        (ChunkMeta::meta(), bytes)
    }

    /// A node as its labels and properties, each sorted, to compare across
    /// stores.
    type DescribedNode = (Vec<String>, Vec<(String, Value)>);

    fn described_node(store: &CompactStore, id: NodeId) -> Option<DescribedNode> {
        store.get_node(id).map(|node| {
            let mut labels: Vec<String> = node.labels.iter().map(ToString::to_string).collect();
            labels.sort();
            (labels, sorted_properties(node.properties.iter()))
        })
    }

    /// An edge as its endpoints, type and sorted properties.
    type DescribedEdge = (NodeId, NodeId, String, Vec<(String, Value)>);

    fn described_edge(store: &CompactStore, id: EdgeId) -> Option<DescribedEdge> {
        store.get_edge(id).map(|edge| {
            (
                edge.src,
                edge.dst,
                edge.edge_type.to_string(),
                sorted_properties(edge.properties.iter()),
            )
        })
    }

    fn sorted_properties<'a>(
        properties: impl Iterator<Item = (&'a PropertyKey, &'a Value)>,
    ) -> Vec<(String, Value)> {
        let mut sorted: Vec<(String, Value)> = properties
            .map(|(key, value)| (key.as_str().to_string(), value.clone()))
            .collect();
        sorted.sort_by(|left, right| left.0.cmp(&right.0));
        sorted
    }

    /// The stream holds the version 4 encoding byte for byte: what
    /// `serialize_with_version(4)` returns and `serialize` still writes (for
    /// spill files), behind a metadata chunk with the caps.
    #[test]
    fn the_streamed_compact_encoding_equals_the_serialized_one() {
        let (store, _, _) = store_for_streams();
        let section = section_of(&store, SMALL_CAPS);
        let image = image_of(&section);
        let source = image.section_source(SectionType::CompactStore).unwrap();
        assert_eq!(source.section_version(), FORMAT_VERSION_CHUNKED);
        let chunks = source.chunks();
        assert_eq!(chunks[0], ChunkMeta::meta(), "the metadata chunk first");
        assert!(
            chunks[1..]
                .iter()
                .all(|meta| *meta == ChunkMeta::stream_piece(0, 0, meta.row_start)),
            "every other chunk is a piece of stream 0: {chunks:?}"
        );
        assert!(
            chunks.len() > 4,
            "at 512 bytes the stream spans several pieces: {} chunks",
            chunks.len()
        );
        let meta: CompactMeta = bincode::serde::decode_from_slice(
            &source.fetch(0).unwrap(),
            bincode::config::standard(),
        )
        .unwrap()
        .0;
        assert_eq!((meta.layout, meta.max_bytes), (1, 512));

        let streamed = read_stream(&*source, 0, 0).unwrap();
        let serialized = section.serialize_with_version(FORMAT_VERSION).unwrap();
        assert_eq!(serialized[4], FORMAT_VERSION);
        assert_eq!(
            first_difference(&streamed, &serialized),
            None,
            "the stream is the version 4 encoding"
        );
        assert_eq!(
            first_difference(&section.serialize().unwrap(), &serialized),
            None,
            "serialize writes the version 4 encoding"
        );
        let error = section
            .serialize_with_version(FORMAT_VERSION_CHUNKED)
            .unwrap_err();
        assert!(
            matches!(&error, Error::Internal(message) if message.contains("version 5")),
            "the section version is no encoding: {error:?}"
        );
    }

    /// An id map and its reverse are built together. Should they ever
    /// disagree, the writer refuses to write rather than write the entries of
    /// one and lose those only the other has.
    #[test]
    fn id_maps_that_disagree_with_their_reverse_are_not_written() {
        let (store, nodes, edges) = store_for_streams();
        assert_eq!(nodes.len(), 61, "the counts below");
        let written = |change: &dyn Fn(&mut CompactStore)| {
            let mut compact = from_graph_store_preserving_ids(&store).unwrap();
            change(&mut compact);
            CompactStoreSection::new(Arc::new(compact)).serialize()
        };
        written(&|_| {}).unwrap();
        let cases: [(&str, &dyn Fn(&mut CompactStore), &str); 6] = [
            (
                "a node the reverse lacks",
                &|compact| {
                    let rows = compact.node_offset_to_id.as_mut().unwrap();
                    *rows[0].last_mut().unwrap() = NodeId::INVALID;
                },
                "the map holds 61 ids, the reverse 60",
            ),
            (
                "two nodes swapped in the reverse",
                &|compact| compact.node_offset_to_id.as_mut().unwrap()[1].swap(0, 1),
                "in the reverse",
            ),
            (
                "an edge the map lacks",
                &|compact| {
                    compact.edge_id_map.as_mut().unwrap().remove(&edges[3]);
                },
                &format!("id {} is at table 0", edges[3].as_u64()),
            ),
            (
                "a node map without its reverse",
                &|compact| compact.node_offset_to_id = None,
                "there is no reverse",
            ),
            (
                "a row without an id",
                &|compact| {
                    let rows = compact.node_offset_to_id.as_mut().unwrap();
                    let id = std::mem::replace(&mut rows[0][0], NodeId::INVALID);
                    compact.node_id_map.as_mut().unwrap().remove(&id);
                },
                "60 ids for the 61 rows",
            ),
            (
                "an id past the rows of its table",
                &|compact| {
                    let rows = compact.node_offset_to_id.as_mut().unwrap();
                    let row = rows[0].len() as u64;
                    rows[0].push(NodeId::new(10_000));
                    let map = compact.node_id_map.as_mut().unwrap();
                    map.insert(NodeId::new(10_000), (0, row));
                },
                "which the tables do not hold",
            ),
        ];
        for (case, change, expected) in cases {
            let error = written(change).unwrap_err();
            assert!(
                matches!(&error, Error::Internal(message)
                    if message.contains("disagree") && message.contains(expected)),
                "{case}: {error:?}"
            );
        }
    }

    #[test]
    fn a_compact_store_round_trips_through_streams() {
        let (store, nodes, edges) = store_for_streams();
        let original = Arc::new(from_graph_store_preserving_ids(&store).unwrap());
        assert_eq!(original.node_tables_by_id.len(), 2, "two node tables");
        assert_eq!(original.rel_tables_by_id.len(), 1, "one rel table");
        let section = with_chunk_caps(SMALL_CAPS, || {
            CompactStoreSection::new(Arc::clone(&original))
        });
        let image = image_of(&section);
        let chunks = image
            .section_source(SectionType::CompactStore)
            .unwrap()
            .chunks()
            .len();
        assert!(
            chunks > 4,
            "the stream spans several pieces: {chunks} chunks"
        );

        let restored = load(&image).unwrap().store().unwrap();
        assert!(restored.preserves_ids());
        assert_eq!(
            (restored.node_count(), restored.edge_count()),
            (nodes.len(), edges.len())
        );
        for &id in &nodes {
            let node = described_node(&restored, id);
            assert!(node.is_some(), "node {id:?} by its original id");
            assert_eq!(node, described_node(&original, id), "node {id:?}");
        }
        for &id in &edges {
            let edge = described_edge(&restored, id);
            assert!(edge.is_some(), "edge {id:?} by its original id");
            assert_eq!(edge, described_edge(&original, id), "edge {id:?}");
        }
        // The first person has an age and lives in Amsterdam; the second has
        // no age, and still has none.
        let (first, second) = (nodes[4], nodes[5]);
        assert_eq!(
            restored.get_node_property(first, &PropertyKey::new("age")),
            Some(Value::Int64(19))
        );
        assert_eq!(
            restored.get_node_property(second, &PropertyKey::new("age")),
            None,
            "a missing value stays missing"
        );
        assert_eq!(
            restored.neighbors(first, crate::graph::Direction::Outgoing),
            vec![nodes[0]]
        );
    }

    /// Writing through streams, reading back and writing again gives the
    /// same chunks: two checkpoints of one compacted base agree.
    #[test]
    fn two_checkpoints_of_one_compact_store_write_the_same_chunks() {
        let (store, _, _) = store_for_streams();
        let first = image_of(&section_of(&store, SMALL_CAPS));
        let reopened = load(&first).unwrap().store().unwrap();
        let second = image_of(&with_chunk_caps(SMALL_CAPS, || {
            CompactStoreSection::new(reopened)
        }));
        let chunks = |image: &MemoryImage| {
            let source = image.section_source(SectionType::CompactStore).unwrap();
            (0..source.chunks().len())
                .map(|index| (source.chunks()[index], source.fetch(index).unwrap()))
                .collect::<Vec<_>>()
        };
        let (first, second) = (chunks(&first), chunks(&second));
        assert_eq!(first.len(), second.len(), "chunk count");
        for (index, ((left_meta, left), (right_meta, right))) in
            first.iter().zip(&second).enumerate()
        {
            assert_eq!(left_meta, right_meta, "chunk {index}");
            assert_eq!(first_difference(left, right), None, "chunk {index}");
        }
    }

    /// A 0.5.x file holds the section as one raw chunk of the version 3
    /// encoding, which `read_from` hands to the 0.5.x reader.
    #[test]
    fn a_0_5_compact_section_still_loads() {
        let (store, sparse, sparse_edge) = store_with_missing_properties();
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let bytes = CompactStoreSection::new(Arc::new(compact))
            .serialize_with_version(FORMAT_VERSION_V3)
            .unwrap();
        let image = MemoryImage::from_raw(vec![(SectionType::CompactStore, bytes)]).unwrap();
        let restored = load(&image).unwrap().store().unwrap();

        assert_eq!((restored.node_count(), restored.edge_count()), (2, 2));
        assert_eq!(
            restored.get_node_property(sparse, &PropertyKey::new("n")),
            Some(Value::Int64(2))
        );
        // Version 3 records no missing values: the empty value stored reads.
        assert_eq!(
            restored.get_node_property(sparse, &PropertyKey::new("u")),
            Some(Value::Int64(0))
        );
        assert_eq!(
            restored.get_edge_property(sparse_edge, &PropertyKey::new("n")),
            Some(Value::Int64(2))
        );
    }

    /// Chunk sequences no writer of this release produces are refused,
    /// naming the section and what is wrong.
    #[test]
    fn crafted_compact_chunk_sequences_are_refused() {
        let (store, _, _) = store_with_missing_properties();
        let section = section_of(&store, ChunkCaps::DEFAULT);
        let encoding = section.serialize_with_version(FORMAT_VERSION).unwrap();
        let version_3 = section.serialize_with_version(FORMAT_VERSION_V3).unwrap();
        let piece =
            |stream: u32, bytes: &[u8]| (ChunkMeta::stream_piece(0, stream, 0), bytes.to_vec());
        let mut long_meta = meta_chunk(1);
        long_meta.1.push(88);

        // What the writer writes loads: each case below changes one thing.
        load(&crafted(5, &[meta_chunk(1), piece(0, &encoding)])).unwrap();

        for (case, image, expected) in [
            (
                "an older section version",
                crafted(4, &[meta_chunk(1), piece(0, &encoding)]),
                "version 4",
            ),
            (
                "a newer metadata layout",
                crafted(5, &[meta_chunk(2), piece(0, &encoding)]),
                "layout 2",
            ),
            (
                "bytes after the metadata",
                crafted(5, &[long_meta, piece(0, &encoding)]),
                "1 bytes after its fields",
            ),
            (
                "the stream before the metadata chunk",
                crafted(5, &[piece(0, &encoding), meta_chunk(1)]),
                "the first chunk is a piece of stream 0",
            ),
            (
                "a second stream",
                crafted(5, &[meta_chunk(1), piece(0, &encoding), piece(1, b"Gus")]),
                "stream 1",
            ),
            (
                "a column chunk",
                crafted(
                    5,
                    &[
                        meta_chunk(1),
                        piece(0, &encoding),
                        (ChunkMeta::column(0, 0, 0, 1, 0), b"Mia".to_vec()),
                    ],
                ),
                "Column",
            ),
            (
                "the version 3 encoding in the stream",
                crafted(5, &[meta_chunk(1), piece(0, &version_3)]),
                "version 3",
            ),
            ("no stream", crafted(5, &[meta_chunk(1)]), "too short"),
        ] {
            let error = load(&image).map(drop).unwrap_err();
            assert!(
                matches!(&error, Error::Serialization(message)
                    if message.starts_with("section CompactStore") && message.contains(expected)),
                "{case}: {error:?}"
            );
        }
    }

    /// Accepts `left` chunks, then refuses every chunk as a full disk would.
    struct FullDisk {
        left: usize,
    }

    impl SectionSink for FullDisk {
        fn write_chunk(
            &mut self,
            _meta: ChunkMeta,
            _bytes: &[u8],
        ) -> grafeo_common::utils::error::Result<()> {
            if self.left == 0 {
                return Err(Error::Io(std::io::Error::new(
                    std::io::ErrorKind::StorageFull,
                    "no space left in Prague",
                )));
            }
            self.left -= 1;
            Ok(())
        }
    }

    /// A sink error in the middle of the stream comes back as it was: a full
    /// disk stays an I/O error of its kind.
    #[test]
    fn a_sink_error_in_the_stream_keeps_its_variant() {
        let (store, _, _) = store_for_streams();
        let section = section_of(&store, SMALL_CAPS);
        // The metadata chunk and the first piece.
        let error = section.write_to(&mut FullDisk { left: 2 }).unwrap_err();
        assert!(
            matches!(&error, Error::Io(inner)
                if inner.kind() == std::io::ErrorKind::StorageFull
                    && inner.to_string().contains("Prague")),
            "{error:?}"
        );
    }

    // ── Review round: determinism of compact(), crafted encodings ───

    /// Forty nodes over four labels and sixty edges over four edge types,
    /// each with a property: enough tables that a build in hash-map order
    /// almost never repeats its order. Returns the store and its highest node
    /// and edge ids.
    fn travels() -> (LpgStore, u64, u64) {
        let store = LpgStore::new().unwrap();
        let mut nodes = Vec::new();
        for (label, n) in ["Person", "City", "Museum", "Station"]
            .iter()
            .cycle()
            .take(40)
            .zip(0i64..)
        {
            let id = store.create_node(&[label]);
            store.set_node_property(id, "n", Value::Int64(n * 3));
            nodes.push(id);
        }
        let mut edges = Vec::new();
        for ((kind, i), n) in ["KNOWS", "LIVES_IN", "VISITED", "NEAR"]
            .iter()
            .cycle()
            .zip(0..60usize)
            .zip(0i64..)
        {
            let edge = store.create_edge(nodes[i % 40], nodes[(i * 7 + 3) % 40], kind);
            store.set_edge_property(edge, "km", Value::Int64(n * 19));
            edges.push(edge);
        }
        let max_node = nodes.iter().map(NodeId::as_u64).max().unwrap();
        let max_edge = edges.iter().map(EdgeId::as_u64).max().unwrap();
        (store, max_node, max_edge)
    }

    /// `compact()` builds its tables in an order that depends on the data
    /// alone: two compactions of the same data, and two merges of the same
    /// overlay into the same base, write the same bytes. Golden fixtures and
    /// incremental checkpoints rest on this.
    #[test]
    fn compactions_and_merges_of_the_same_data_write_the_same_bytes() {
        let compacted = || {
            let (store, _, _) = travels();
            let compact = from_graph_store_preserving_ids(&store).unwrap();
            CompactStoreSection::new(Arc::new(compact))
                .serialize()
                .unwrap()
        };
        let first = compacted();
        for round in 1..=3 {
            assert_eq!(
                first_difference(&first, &compacted()),
                None,
                "compaction {round} wrote other bytes"
            );
        }

        let merged = || {
            use crate::graph::compact::layered::LayeredStore;
            use crate::graph::traits::GraphStoreMut;
            let (store, max_node, max_edge) = travels();
            let base = from_graph_store_preserving_ids(&store).unwrap();
            let layered = LayeredStore::new(base, max_node, max_edge).unwrap();
            let prague = layered.create_node(&["City"]);
            layered.set_node_property(prague, "n", Value::Int64(88));
            let vincent = layered.create_node(&["Person"]);
            let edge = layered.create_edge(vincent, prague, "LIVES_IN");
            layered.set_edge_property(edge, "km", Value::Int64(3));
            layered.create_edge(NodeId::new(3), vincent, "KNOWS");
            layered.merge_overlay_in_place().unwrap();
            CompactStoreSection::new(layered.base_store_arc())
                .serialize()
                .unwrap()
        };
        let first = merged();
        for round in 1..=3 {
            assert_eq!(
                first_difference(&first, &merged()),
                None,
                "merge {round} wrote other bytes"
            );
        }
    }

    /// An `io::Write` that keeps the size of the largest write it was given.
    #[derive(Default)]
    struct LargestWrite {
        largest: usize,
        writes: usize,
        written: usize,
    }

    impl std::io::Write for LargestWrite {
        fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
            self.largest = self.largest.max(data.len());
            self.writes += 1;
            self.written += data.len();
            Ok(data.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// The id maps go to the stream in pieces of at most 64 KiB (and one
    /// entry), not as one buffer: 4,000 nodes hold 72,000 bytes of node id
    /// map entries.
    #[test]
    fn id_map_entries_are_written_in_pieces_of_at_most_64_kib() {
        let store = LpgStore::new().unwrap();
        for n in 0..4_000i64 {
            let id = store.create_node(&["Person"]);
            store.set_node_property(id, "n", Value::Int64(n % 88));
        }
        let section =
            CompactStoreSection::new(Arc::new(from_graph_store_preserving_ids(&store).unwrap()));
        let mut out = LargestWrite::default();
        section
            .write_with_version(FORMAT_VERSION, &mut out)
            .unwrap();
        assert!(out.written > 72_000, "{} bytes", out.written);
        assert!(
            out.largest <= ID_MAP_PIECE + 18,
            "a write of {} bytes",
            out.largest
        );
        assert!(out.writes >= 4, "{} writes", out.writes);
    }

    /// A version 4 encoding written field by field, for inputs no writer
    /// produces.
    struct Fields(Vec<u8>);

    impl Fields {
        /// Magic, version 4 and the flag of preserved ids.
        fn header() -> Self {
            let mut bytes = MAGIC.to_vec();
            bytes.push(FORMAT_VERSION);
            bytes.push(1);
            Self(bytes)
        }

        fn byte(mut self, value: u8) -> Self {
            self.0.push(value);
            self
        }

        fn u16(mut self, value: u16) -> Self {
            self.0.extend_from_slice(&value.to_le_bytes());
            self
        }

        fn u32(mut self, value: u32) -> Self {
            self.0.extend_from_slice(&value.to_le_bytes());
            self
        }

        fn u64(mut self, value: u64) -> Self {
            self.0.extend_from_slice(&value.to_le_bytes());
            self
        }

        fn name(mut self, name: &str) -> Self {
            self = self.u16(u16::try_from(name.len()).unwrap());
            self.0.extend_from_slice(name.as_bytes());
            self
        }

        /// A node table without columns.
        fn node_table(self, label: &str, rows: u32) -> Self {
            self.name(label).u32(rows).u32(0)
        }

        /// An adjacency: its offsets, its targets and, for a backward one, the
        /// forward position of each edge.
        fn csr(mut self, offsets: &[u32], targets: &[u32], edge_data: Option<&[u32]>) -> Self {
            self = self.u32(u32::try_from(offsets.len()).unwrap());
            for &offset in offsets {
                self = self.u32(offset);
            }
            self = self.u32(u32::try_from(targets.len()).unwrap());
            for &target in targets {
                self = self.u32(target);
            }
            match edge_data {
                Some(positions) => {
                    self = self.byte(1).u32(u32::try_from(positions.len()).unwrap());
                    for &position in positions {
                        self = self.u32(position);
                    }
                    self
                }
                None => self.byte(0),
            }
        }

        /// An id map entry: an id at a row of a table.
        fn entry(self, id: u64, table: u16, row: u64) -> Self {
            self.u64(id).u16(table).u64(row)
        }

        /// The bytes and their CRC32.
        fn sealed(self) -> Vec<u8> {
            let mut bytes = self.0;
            let crc = crc32fast::hash(&bytes);
            bytes.extend_from_slice(&crc.to_le_bytes());
            bytes
        }
    }

    /// The parts of a small valid encoding, each case below changing one:
    /// two `:Person` rows (ids 3 and 19) and one `KNOWS` edge (id 88) from the
    /// first to the second.
    struct Parts {
        node_tables: Fields,
        rel_tables: Fields,
        node_map: Fields,
        edge_map: Fields,
    }

    impl Parts {
        fn valid() -> Self {
            Self {
                node_tables: Fields(Vec::new()).u32(1).node_table("Person", 2),
                rel_tables: Fields(Vec::new())
                    .u32(1)
                    .name("KNOWS")
                    .u16(0)
                    .u16(0)
                    .csr(&[0, 1, 1], &[1], None)
                    .byte(1)
                    .csr(&[0, 0, 1], &[0], Some(&[0]))
                    .u32(0),
                node_map: Fields(Vec::new()).u32(2).entry(3, 0, 0).entry(19, 0, 1),
                edge_map: Fields(Vec::new()).u32(1).entry(88, 0, 0),
            }
        }

        fn sealed(self) -> Vec<u8> {
            let mut fields = Fields::header();
            for part in [
                self.node_tables,
                self.rel_tables,
                self.node_map,
                self.edge_map,
            ] {
                fields.0.extend_from_slice(&part.0);
            }
            fields.sealed()
        }
    }

    /// A rel table part with `fwd` and `bwd` as its adjacencies.
    fn rel_tables(
        src: u16,
        dst: u16,
        fwd: (&[u32], &[u32]),
        bwd: (&[u32], &[u32], &[u32]),
    ) -> Fields {
        Fields(Vec::new())
            .u32(1)
            .name("KNOWS")
            .u16(src)
            .u16(dst)
            .csr(fwd.0, fwd.1, None)
            .byte(1)
            .csr(bwd.0, bwd.1, Some(bwd.2))
            .u32(0)
    }

    /// An encoding of one more node table (each empty) than table ids can
    /// name.
    fn more_tables_than_ids() -> Vec<u8> {
        let tables = u32::from(super::super::id::MAX_TABLE_ID) + 2;
        let mut fields = Fields::header().u32(tables);
        for _ in 0..tables {
            fields = fields.node_table("P", 0);
        }
        fields.u32(0).u32(0).u32(0).sealed()
    }

    /// Loads `encoding` as the stream of a version 5 section.
    fn load_encoding_of(
        encoding: &[u8],
    ) -> grafeo_common::utils::error::Result<CompactStoreSection> {
        let piece = (ChunkMeta::stream_piece(0, 0, 0), encoding.to_vec());
        load(&crafted(5, &[meta_chunk(1), piece]))
    }

    /// Counts, table ids and rows read from a stream are untrusted: a count
    /// no file could hold, a table that does not exist or a row past its
    /// table is an error naming the section, never an abort, a huge
    /// allocation or a silent misread.
    #[test]
    fn crafted_counts_tables_and_rows_are_refused() {
        let store = load_encoding_of(&Parts::valid().sealed())
            .unwrap()
            .store()
            .unwrap();
        assert_eq!((store.node_count(), store.edge_count()), (2, 1));
        assert_eq!(
            store.neighbors(NodeId::new(3), crate::graph::Direction::Outgoing),
            vec![NodeId::new(19)]
        );

        let with = |change: &dyn Fn(&mut Parts)| {
            let mut parts = Parts::valid();
            change(&mut parts);
            parts.sealed()
        };
        let cases: Vec<(&str, Vec<u8>, &str)> = vec![
            (
                "a node table count",
                Fields::header().u32(u32::MAX).sealed(),
                "node tables",
            ),
            (
                "a column count",
                with(&|parts| {
                    parts.node_tables = Fields(Vec::new())
                        .u32(1)
                        .name("Person")
                        .u32(2)
                        .u32(u32::MAX);
                }),
                "columns",
            ),
            (
                "a rel table count",
                with(&|parts| parts.rel_tables = Fields(Vec::new()).u32(u32::MAX)),
                "relationship tables",
            ),
            (
                "an adjacency offset count",
                with(&|parts| {
                    parts.rel_tables = Fields(Vec::new())
                        .u32(1)
                        .name("KNOWS")
                        .u16(0)
                        .u16(0)
                        .u32(u32::MAX);
                }),
                "offsets",
            ),
            (
                "a node id map length",
                with(&|parts| parts.node_map = Fields(Vec::new()).u32(u32::MAX)),
                "node id map entries",
            ),
            (
                "an edge id map length",
                with(&|parts| parts.edge_map = Fields(Vec::new()).u32(u32::MAX)),
                "edge id map entries",
            ),
            (
                "a table that does not exist",
                with(&|parts| {
                    parts.node_map = Fields(Vec::new()).u32(2).entry(3, 7, 0).entry(19, 0, 1);
                }),
                "table 7",
            ),
            (
                "a row past its table",
                with(&|parts| {
                    parts.node_map = Fields(Vec::new())
                        .u32(2)
                        .entry(3, 0, 0)
                        .entry(19, 0, 10_000_000);
                }),
                "row 10000000",
            ),
            (
                "the last row there is",
                with(&|parts| {
                    parts.node_map =
                        Fields(Vec::new())
                            .u32(2)
                            .entry(3, 0, 0)
                            .entry(19, 0, u64::MAX);
                }),
                "row 18446744073709551615",
            ),
            (
                "two ids at one row",
                with(&|parts| {
                    parts.node_map = Fields(Vec::new()).u32(2).entry(3, 0, 0).entry(19, 0, 0);
                }),
                "both",
            ),
            (
                "fewer ids than rows",
                with(&|parts| parts.node_map = Fields(Vec::new()).u32(1).entry(3, 0, 0)),
                "1 entries for 2 rows",
            ),
            (
                "an edge past its table",
                with(&|parts| parts.edge_map = Fields(Vec::new()).u32(1).entry(88, 0, 1)),
                "row 1",
            ),
            (
                "an edge table that does not exist",
                with(&|parts| parts.edge_map = Fields(Vec::new()).u32(1).entry(88, 3, 0)),
                "table 3",
            ),
            (
                "a rel table between tables that do not exist",
                with(&|parts| {
                    parts.rel_tables =
                        rel_tables(5, 0, (&[0, 1, 1], &[1]), (&[0, 0, 1], &[0], &[0]));
                }),
                "tables 5 and 0",
            ),
            (
                "an adjacency whose offsets run backwards",
                with(&|parts| {
                    parts.rel_tables =
                        rel_tables(0, 0, (&[0, 1, 0], &[1]), (&[0, 0, 1], &[0], &[0]));
                }),
                "offsets",
            ),
            (
                "an adjacency of another table's size",
                with(&|parts| {
                    parts.rel_tables = rel_tables(0, 0, (&[0, 1], &[1]), (&[0, 0, 1], &[0], &[0]));
                }),
                "1 nodes",
            ),
            (
                "a target past its table",
                with(&|parts| {
                    parts.rel_tables =
                        rel_tables(0, 0, (&[0, 1, 1], &[7]), (&[0, 0, 1], &[0], &[0]));
                }),
                "row 7",
            ),
            (
                "a backward edge without its forward edge",
                with(&|parts| {
                    parts.rel_tables =
                        rel_tables(0, 0, (&[0, 1, 1], &[1]), (&[0, 0, 1], &[0], &[5]));
                }),
                "forward edge 5",
            ),
            (
                "a backward adjacency with other edges",
                with(&|parts| {
                    parts.rel_tables =
                        rel_tables(0, 0, (&[0, 1, 1], &[1]), (&[0, 0, 2], &[0, 0], &[0, 0]));
                }),
                "2 edges backward and 1 forward",
            ),
            (
                "one id at two rows",
                with(&|parts| {
                    parts.node_map = Fields(Vec::new()).u32(2).entry(3, 0, 0).entry(3, 0, 1);
                }),
                "lists id 3 twice",
            ),
            (
                "the invalid id",
                with(&|parts| {
                    parts.node_map = Fields(Vec::new())
                        .u32(2)
                        .entry(3, 0, 0)
                        .entry(u64::MAX, 0, 1);
                }),
                "invalid id",
            ),
            (
                "more tables than table ids name",
                more_tables_than_ids(),
                "more than the 32768",
            ),
        ];
        for (case, encoding, expected) in cases {
            let error = load_encoding_of(&encoding).map(drop).unwrap_err();
            assert!(
                matches!(&error, Error::Serialization(message)
                    if message.starts_with("section CompactStore: ") && message.contains(expected)),
                "{case}: {error:?}"
            );
        }
    }

    /// A 0.5.x section is checked as a streamed one is: a raw chunk with a
    /// crafted count is an error, never an abort.
    #[test]
    fn a_crafted_0_5_compact_section_is_refused() {
        let mut bytes = Parts::valid().sealed();
        bytes[4] = FORMAT_VERSION_V3;
        let crc_at = bytes.len() - 4;
        bytes[6..10].copy_from_slice(&u32::MAX.to_le_bytes());
        let crc = crc32fast::hash(&bytes[..crc_at]);
        bytes[crc_at..].copy_from_slice(&crc.to_le_bytes());
        let image = MemoryImage::from_raw(vec![(SectionType::CompactStore, bytes)]).unwrap();
        let error = load(&image).map(drop).unwrap_err();
        assert!(
            matches!(&error, Error::Serialization(message)
                if message.starts_with("section CompactStore: ") && message.contains("node tables")),
            "{error:?}"
        );
    }

    // ── Nodes with several labels ──────────────────────────────────

    /// The labels of node `id` in `store`, sorted.
    fn sorted_labels(store: &CompactStore, id: NodeId) -> Vec<String> {
        described_node(store, id)
            .map(|(labels, _)| labels)
            .unwrap_or_default()
    }

    fn sorted_ids(mut ids: Vec<NodeId>) -> Vec<NodeId> {
        ids.sort_unstable();
        ids
    }

    /// Each node keeps its labels through a checkpoint and a reopen: two
    /// labels, labels holding the separator (`|`) and the escape character
    /// (`\`) of a label set's key, and the set {In, Out} beside the label
    /// `In|Out`.
    #[test]
    fn the_labels_of_compacted_nodes_round_trip_through_the_section() {
        let store = LpgStore::new().unwrap();
        let vincent = store.create_node(&["Person", "Actor"]);
        let pipe = store.create_node(&["In|Out"]);
        let pair = store.create_node(&["In", "Out"]);
        let both = store.create_node(&["In|Out", "Person"]);
        let slash = store.create_node(&["C:\\Data\\", "Person"]);
        let expected: [(NodeId, &[&str]); 5] = [
            (vincent, &["Actor", "Person"]),
            (pipe, &["In|Out"]),
            (pair, &["In", "Out"]),
            (both, &["In|Out", "Person"]),
            (slash, &["C:\\Data\\", "Person"]),
        ];
        let section =
            CompactStoreSection::new(Arc::new(from_graph_store_preserving_ids(&store).unwrap()));
        let restored = load(&image_of(&section)).unwrap().store().unwrap();
        for (id, labels) in expected {
            assert_eq!(sorted_labels(&restored, id), labels, "node {id:?}");
        }
        assert_eq!(
            sorted_ids(restored.nodes_by_label("In|Out")),
            sorted_ids(vec![pipe, both])
        );
        assert_eq!(restored.nodes_by_label("In"), vec![pair]);
        assert_eq!(
            sorted_ids(restored.nodes_by_label("Person")),
            sorted_ids(vec![vincent, both, slash])
        );
        assert_eq!(restored.nodes_by_label_count("Person"), 3);

        // The 0.5.x encoding a test may still write joins the labels as
        // 0.5.x did, and reads back the same for labels without `|`.
        let version_3 = section.serialize_with_version(FORMAT_VERSION_V3).unwrap();
        let image = MemoryImage::from_raw(vec![(SectionType::CompactStore, version_3)]).unwrap();
        let restored = load(&image).unwrap().store().unwrap();
        assert_eq!(sorted_labels(&restored, vincent), ["Actor", "Person"]);
        assert_eq!(
            sorted_labels(&restored, slash),
            ["C:\\Data\\", "Person"],
            "0.5.x keys have no escapes"
        );
    }

    /// The empty label is not the absence of labels: a node labeled `` and a
    /// node without labels compact into tables of their own and keep their
    /// labels, in memory and through the section.
    #[test]
    fn an_empty_label_is_not_the_absence_of_labels() {
        let store = LpgStore::new().unwrap();
        let mia = store.create_node(&[""]);
        let gus = store.create_node(&[]);
        let jules = store.create_node(&["", "Person"]);
        let section =
            CompactStoreSection::new(Arc::new(from_graph_store_preserving_ids(&store).unwrap()));
        let restored = load(&image_of(&section)).unwrap().store().unwrap();
        for (stage, compact) in [
            ("compacted", section.store().unwrap()),
            ("read back", restored),
        ] {
            assert_eq!(sorted_labels(&compact, mia), [""], "{stage}: Mia");
            assert!(sorted_labels(&compact, gus).is_empty(), "{stage}: Gus");
            assert_eq!(
                sorted_labels(&compact, jules),
                ["", "Person"],
                "{stage}: Jules"
            );
            assert_eq!(
                sorted_ids(compact.nodes_by_label("")),
                sorted_ids(vec![mia, jules]),
                "{stage}: the nodes labeled ``"
            );
            assert_eq!(compact.node_count(), 3, "{stage}");
        }
    }

    /// The nodes without labels (the table of the empty key) round trip
    /// through the section with their properties and their edges, in both
    /// directions, to and from a labeled node, in the encoding this build
    /// writes and the 0.5.x one, and the store writes the bytes it was read
    /// from.
    #[test]
    fn nodes_without_labels_round_trip_through_the_section() {
        let store = LpgStore::new().unwrap();
        let gus = store.create_node(&[]);
        store.set_node_property(gus, "name", Value::from("Gus"));
        store.set_node_property(gus, "age", Value::Int64(19));
        let alix = store.create_node(&["Person"]);
        store.set_node_property(alix, "name", Value::from("Alix"));
        let mia = store.create_node(&[]);
        store.set_node_property(mia, "name", Value::from("Mia"));
        store.set_node_property(mia, "age", Value::Int64(88));
        let to_alix = store.create_edge(gus, alix, "KNOWS");
        store.set_edge_property(to_alix, "since", Value::Int64(3));
        let to_mia = store.create_edge(alix, mia, "KNOWS");
        let section =
            CompactStoreSection::new(Arc::new(from_graph_store_preserving_ids(&store).unwrap()));
        let first = section.serialize().unwrap();
        let version_3 = section.serialize_with_version(FORMAT_VERSION_V3).unwrap();

        for (encoding, image) in [
            ("this build's", image_of(&section)),
            (
                "the 0.5.x",
                MemoryImage::from_raw(vec![(SectionType::CompactStore, version_3)]).unwrap(),
            ),
        ] {
            let restored = load(&image).unwrap().store().unwrap();
            assert_eq!(
                sorted_ids(restored.node_ids()),
                sorted_ids(vec![alix, gus, mia]),
                "{encoding} encoding: every node"
            );
            assert_eq!(
                described_node(&restored, gus),
                Some((
                    Vec::new(),
                    vec![
                        ("age".to_string(), Value::Int64(19)),
                        ("name".to_string(), Value::from("Gus"))
                    ]
                )),
                "{encoding} encoding: Gus, without labels"
            );
            assert_eq!(
                described_node(&restored, mia),
                Some((
                    Vec::new(),
                    vec![
                        ("age".to_string(), Value::Int64(88)),
                        ("name".to_string(), Value::from("Mia"))
                    ]
                )),
                "{encoding} encoding: Mia, without labels"
            );
            assert_eq!(restored.nodes_by_label("Person"), vec![alix]);
            assert_eq!(restored.statistics().total_nodes, 3, "{encoding} encoding");
            assert_eq!(
                restored.edges_from(gus, crate::graph::Direction::Outgoing),
                vec![(alix, to_alix)],
                "{encoding} encoding: out of Gus"
            );
            assert_eq!(
                restored.edges_from(alix, crate::graph::Direction::Incoming),
                vec![(gus, to_alix)],
                "{encoding} encoding: into Alix"
            );
            assert_eq!(
                restored.edges_from(mia, crate::graph::Direction::Incoming),
                vec![(alix, to_mia)],
                "{encoding} encoding: into Mia"
            );
            assert_eq!(
                restored.get_edge_property(to_alix, &PropertyKey::new("since")),
                Some(Value::Int64(3))
            );
        }

        let mut section = CompactStoreSection::empty();
        section.deserialize(&first).unwrap();
        let again = CompactStoreSection::new(section.store().unwrap())
            .serialize()
            .unwrap();
        assert_eq!(first_difference(&first, &again), None, "the same bytes");
    }

    /// 0.5.44 (the version 3 encoding, a raw chunk) and earlier 0.6 builds
    /// (version 4) named the table of a node with several labels by the
    /// labels joined with `|`, without escapes: its nodes read back with each
    /// label, and a backslash in a label stays one.
    #[test]
    fn a_joined_label_key_of_an_older_build_reads_as_separate_labels() {
        let encoding = |key: &str, version: u8| {
            let mut parts = Parts::valid();
            parts.node_tables = Fields(Vec::new()).u32(1).node_table(key, 2);
            let mut bytes = parts.sealed();
            bytes[4] = version;
            let crc_at = bytes.len() - 4;
            let crc = crc32fast::hash(&bytes[..crc_at]);
            bytes[crc_at..].copy_from_slice(&crc.to_le_bytes());
            bytes
        };
        let (alix, gus) = (NodeId::new(3), NodeId::new(19));
        for key in ["Employee|Person", "C:\\Data|Person"] {
            let first = key.split('|').next().unwrap();
            let version_3 = MemoryImage::from_raw(vec![(
                SectionType::CompactStore,
                encoding(key, FORMAT_VERSION_V3),
            )])
            .unwrap();
            for (build, store) in [
                ("0.5.44", load(&version_3).unwrap().store().unwrap()),
                (
                    "an earlier 0.6 build",
                    load_encoding_of(&encoding(key, FORMAT_VERSION))
                        .unwrap()
                        .store()
                        .unwrap(),
                ),
            ] {
                for id in [alix, gus] {
                    assert_eq!(
                        sorted_labels(&store, id),
                        [first, "Person"],
                        "{build}: {key}"
                    );
                }
                for label in [first, "Person"] {
                    assert_eq!(
                        sorted_ids(store.nodes_by_label(label)),
                        vec![alix, gus],
                        "{build}: {label} of {key}"
                    );
                    assert_eq!(store.nodes_by_label_count(label), 2, "{build}: {label}");
                }
                assert!(store.nodes_by_label(key).is_empty(), "{build}: {key}");
                assert_eq!(
                    store.statistics().get_label("Person").map(|s| s.node_count),
                    Some(2),
                    "{build}"
                );
            }
        }
    }
}
