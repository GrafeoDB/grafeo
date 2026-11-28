//! Saved index sections (#711): an index of the store as a checkpoint
//! writes it into the database file, so an open restores it instead of
//! building it from the rows.
//!
//! Each kind of index has its own section; the vector indexes are
//! `VectorStore` version 4 ([`VECTOR_INDEX_SECTION_VERSION`]). A section
//! holds, for each graph (by the LPG section's graph id) and each of its
//! indexes (by a column id the metadata gives it):
//!
//! - **Pieces**: [`ChunkKind::Column`] chunks of the index's column. A piece
//!   covers the rows from `row_start` (the raw id of its first node) for
//!   `row_count` rows (to its last node), within one store row group of
//!   65,536 rows; pieces of an index ascend and do not overlap. The pieces
//!   of a row group are cut at node boundaries at the byte cap, a node whose
//!   entry passes the cap alone taking a piece of its own.
//! - **The list stream**: [`ChunkKind::Stream`] of the index's column,
//!   `[count u64]` then the raw ids of the *listed* nodes, ascending, as
//!   LEB128 deltas (the first from 0). A listed node's content in the
//!   pieces is not to be trusted: the open derives it again from the rows.
//! - **The quantizer stream** (a quantized vector index only): stream
//!   `quantizer_column`, `[kind u8][fitted u8]` (kind 1 scalar, 2 binary, 3
//!   product), then for a fitted scalar quantizer `[dimensions u32]` and its
//!   minimum, scale and inverse scale per dimension (`f32` each, the three
//!   arrays in turn); for a fitted product quantizer `[subvectors
//!   u32][centroids u32][subvector dimensions u32]` and the centroid values
//!   (`f32`); for a quantizer not fitted yet `[samples u64]` and the raw ids
//!   of its training samples in insert order (`u64`). Binary quantization
//!   has nothing to fit and is always fitted.
//! - **The metadata chunk** ([`ChunkMeta::meta`]), last, vouching for every
//!   chunk before it: the bincode (standard configuration) of [`SavedMeta`].
//!
//! A vector index piece is `[nodes u32]`, then per node `[offset u32][tag
//! u8]`, the offset from `row_start` strictly increasing and below
//! `row_count`: tag 0, a graph node, `[hash u64][levels u8]` (1 or more)
//! and per level `[count u32]` and as many `[neighbor u64]` raw ids (never
//! the node itself); tag 1, an exact duplicate (alias), `[representative
//! u64]` (never the node itself). Then, while the index keeps codes,
//! `[code length u32]` and the code, of the length the quantization fixes.
//! All little-endian.

use bytes::Bytes;
use serde::{Deserialize, Serialize};

use grafeo_common::storage::catalog_record::IndexKindRecord;
use grafeo_common::storage::section::{ChunkKind, ChunkMeta, SectionSink, SectionSource};
use grafeo_common::storage::{ChunkCaps, ChunkStreamWriter, read_stream};
use grafeo_common::types::NodeId;
use grafeo_common::utils::error::{Error, Result};

use crate::index::vector::{
    ProductQuantizer, SavedNode, SavedQuantizer, ScalarQuantizer, VectorIndexKind,
};

use super::super::ROWS_PER_GROUP;

/// The version of the `VectorStore` section this module writes and reads.
pub const VECTOR_INDEX_SECTION_VERSION: u8 = 4;

/// The layout byte of the metadata chunk.
const SAVED_META_LAYOUT: u8 = 1;

/// The most bytes the metadata's decoding may claim, so a corrupt length
/// prefix fails the decode instead of requesting a huge allocation.
const META_DECODE_LIMIT: usize = 1 << 24;

/// The most entries a decode allocates for up front: a count read from the
/// file is not trusted with an allocation.
const PREALLOCATE_AT_MOST: usize = 64;

/// An index's definition as its section records it, in the catalog's own
/// record type, so an open compares it with the catalog's definition field
/// for field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SavedDefinition {
    /// A property index on every label, a text index or a vector index.
    Index(IndexKindRecord),
}

/// What a saved index section's metadata chunk holds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SavedMeta {
    /// The layout of this metadata: `SAVED_META_LAYOUT` (1) today.
    pub layout: u8,
    /// The epoch of the LPG section of the same image: the section is used
    /// only with it.
    pub epoch: u64,
    /// The byte cap the pieces and streams were cut with (readers accept any).
    pub max_bytes: u32,
    /// The next column id to give an index: ids are kept from image to
    /// image, so a reused piece keeps its identity, and never given twice.
    pub next_column: u32,
    /// By graph id, then column, ascending.
    pub indexes: Vec<SavedIndexMeta>,
}

/// One index of a saved index section.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SavedIndexMeta {
    /// The graph, by the LPG section's graph id.
    pub graph_id: u32,
    /// The column of its pieces and its list stream.
    pub column: u32,
    /// The stream of its quantizer, for a quantized vector index.
    pub quantizer_column: Option<u32>,
    /// The definition it was built by.
    pub definition: SavedDefinition,
    /// The members its pieces hold (a vector index's aliases included),
    /// listed nodes included.
    pub members: u64,
    /// A vector index's entry point and top level when its pieces were
    /// written; `None` when it had no node.
    pub entry: Option<(u64, u32)>,
}

/// What one index's pieces and streams held, for its metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WrittenIndex {
    /// The members written.
    pub members: u64,
    /// The entry point and top level, read after the pieces.
    pub entry: Option<(u64, u32)>,
}

/// A corruption of the `VectorStore` section.
fn corrupt(text: impl std::fmt::Display) -> Error {
    Error::corruption(format!("section VectorStore: {text}"))
}

/// `count` as a `u32` of the format, or an error naming `what`.
fn count_u32(count: usize, what: &str) -> Result<u32> {
    u32::try_from(count).map_err(|_| {
        Error::Serialization(format!(
            "section VectorStore: {count} {what}, more than a saved index holds"
        ))
    })
}

/// `value`, a row offset or row count inside one row group, as a `u32`.
fn within_group(value: u64) -> Result<u32> {
    u32::try_from(value)
        .ok()
        .filter(|&value| u64::from(value) <= ROWS_PER_GROUP)
        .ok_or_else(|| {
            Error::Internal(format!("a piece spans {value} rows, more than a row group"))
        })
}

/// The metadata chunk's bytes.
///
/// # Errors
///
/// [`Error::Serialization`] when the metadata does not encode.
pub fn encode_meta(meta: &SavedMeta) -> Result<Vec<u8>> {
    bincode::serde::encode_to_vec(meta, bincode::config::standard()).map_err(|error| {
        Error::Serialization(format!(
            "section VectorStore: the metadata does not encode: {error}"
        ))
    })
}

/// The metadata of a section: its last chunk, decoded and checked (its
/// layout, its indexes ascending by graph and column, every column and
/// quantizer stream distinct and below `next_column`, every other chunk a
/// piece or stream of an index it lists).
///
/// # Errors
///
/// [`Error::Corruption`] naming what is wrong, or the fetch's error.
pub fn read_meta(source: &dyn SectionSource) -> Result<SavedMeta> {
    let chunks = source.chunks();
    let Some((last, rest)) = chunks.split_last() else {
        return Err(corrupt("the section holds no chunk, not even its metadata"));
    };
    if *last != ChunkMeta::meta() {
        return Err(corrupt(format!(
            "the last chunk is {last:?}, where the metadata chunk comes last"
        )));
    }
    let bytes = source.fetch(chunks.len() - 1)?;
    let (meta, read): (SavedMeta, usize) = bincode::serde::decode_from_slice(
        &bytes,
        bincode::config::standard().with_limit::<META_DECODE_LIMIT>(),
    )
    .map_err(|error| corrupt(format!("the metadata does not decode: {error}")))?;
    if read != bytes.len() {
        return Err(corrupt(format!(
            "the metadata chunk holds {} bytes after its {read} bytes of metadata",
            bytes.len() - read
        )));
    }
    if meta.layout != SAVED_META_LAYOUT {
        return Err(corrupt(format!(
            "the metadata has layout {}, this build reads layout {SAVED_META_LAYOUT}",
            meta.layout
        )));
    }
    for pair in meta.indexes.windows(2) {
        if (pair[0].graph_id, pair[0].column) >= (pair[1].graph_id, pair[1].column) {
            return Err(corrupt(format!(
                "the metadata lists column {} of graph {} after column {} of graph {}, but \
                 indexes ascend by graph and column",
                pair[1].column, pair[1].graph_id, pair[0].column, pair[0].graph_id
            )));
        }
    }
    let mut columns: Vec<u32> = Vec::with_capacity(meta.indexes.len() * 2);
    for index in &meta.indexes {
        columns.push(index.column);
        columns.extend(index.quantizer_column);
    }
    if let Some(&column) = columns.iter().find(|&&column| column >= meta.next_column) {
        return Err(corrupt(format!(
            "column {column} is not below the next column {}",
            meta.next_column
        )));
    }
    columns.sort_unstable();
    if let Some(&[column, ..]) = columns.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(corrupt(format!("column {column} is given twice")));
    }
    for chunk in rest {
        let owned = meta.indexes.iter().any(|index| {
            index.graph_id == chunk.graph_id
                && match chunk.kind {
                    ChunkKind::Column => chunk.column_id == index.column,
                    ChunkKind::Stream => {
                        chunk.column_id == index.column
                            || index.quantizer_column == Some(chunk.column_id)
                    }
                    _ => false,
                }
        });
        if !owned {
            return Err(corrupt(format!(
                "the chunk {chunk:?} belongs to no index the metadata lists"
            )));
        }
    }
    Ok(meta)
}

// ── Pieces ──────────────────────────────────────────────────────────

/// One member of a vector index piece: its id, its saved form and its code.
type PieceEntry = (NodeId, SavedNode, Option<Vec<u8>>);

/// Appends `entry`, a node at `offset` from its piece's first row, to `out`.
fn encode_entry(out: &mut Vec<u8>, offset: u32, entry: &PieceEntry) -> Result<()> {
    let (_, node, code) = entry;
    out.extend_from_slice(&offset.to_le_bytes());
    match node {
        SavedNode::Graph { hash, levels } => {
            out.push(0);
            out.extend_from_slice(&hash.to_le_bytes());
            let count = u8::try_from(levels.len())
                .ok()
                .filter(|&count| count > 0)
                .ok_or_else(|| {
                    Error::Serialization(format!(
                        "section VectorStore: a graph node with {} levels, where a saved index \
                         holds 1 to 255",
                        levels.len()
                    ))
                })?;
            out.push(count);
            for level in levels {
                out.extend_from_slice(&count_u32(level.len(), "neighbors")?.to_le_bytes());
                for neighbor in level {
                    out.extend_from_slice(&neighbor.as_u64().to_le_bytes());
                }
            }
        }
        SavedNode::Alias { representative } => {
            out.push(1);
            out.extend_from_slice(&representative.as_u64().to_le_bytes());
        }
    }
    if let Some(code) = code {
        out.extend_from_slice(&count_u32(code.len(), "code bytes")?.to_le_bytes());
        out.extend_from_slice(code);
    }
    Ok(())
}

/// Reads a piece's fields in order, refusing a piece that ends early.
struct PieceReader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl PieceReader<'_> {
    fn take(&mut self, count: usize) -> Result<&[u8]> {
        let end = self
            .at
            .checked_add(count)
            .filter(|&end| end <= self.bytes.len())
            .ok_or_else(|| {
                corrupt(format!(
                    "a piece of {} bytes ends at byte {} in the middle of a field",
                    self.bytes.len(),
                    self.at
                ))
            })?;
        let taken = &self.bytes[self.at..end];
        self.at = end;
        Ok(taken)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("8 bytes"),
        ))
    }

    fn left(&self) -> usize {
        self.bytes.len() - self.at
    }
}

/// The entries of a piece covering `row_count` rows from `row_start`, each
/// with a code of `code_length` bytes when the index keeps codes.
///
/// # Errors
///
/// [`Error::Corruption`] when the piece breaks a rule of the format.
fn decode_piece(
    bytes: &[u8],
    row_start: u64,
    row_count: u32,
    code_length: Option<usize>,
) -> Result<Vec<PieceEntry>> {
    let mut reader = PieceReader { bytes, at: 0 };
    let nodes = reader.u32()? as usize;
    let mut entries: Vec<PieceEntry> = Vec::with_capacity(nodes.min(PREALLOCATE_AT_MOST));
    let mut previous: Option<u32> = None;
    for _ in 0..nodes {
        let offset = reader.u32()?;
        if offset >= row_count || previous.is_some_and(|previous| offset <= previous) {
            return Err(corrupt(format!(
                "a piece from row {row_start} of {row_count} rows holds a node at offset \
                 {offset}{}",
                previous.map_or_else(String::new, |previous| format!(" after offset {previous}"))
            )));
        }
        previous = Some(offset);
        let id = NodeId::new(row_start + u64::from(offset));
        let node = match reader.u8()? {
            0 => {
                let hash = reader.u64()?;
                let level_count = reader.u8()?;
                if level_count == 0 {
                    return Err(corrupt(format!("graph node {} has no level", id.as_u64())));
                }
                let mut levels: Vec<Vec<NodeId>> = Vec::with_capacity(usize::from(level_count));
                for _ in 0..level_count {
                    let count = reader.u32()? as usize;
                    if count > reader.left() / 8 {
                        return Err(corrupt(format!(
                            "graph node {} claims {count} neighbors at a level, more than the \
                             piece holds",
                            id.as_u64()
                        )));
                    }
                    let mut level = Vec::with_capacity(count);
                    for _ in 0..count {
                        let neighbor = NodeId::new(reader.u64()?);
                        if neighbor == id {
                            return Err(corrupt(format!(
                                "graph node {} lists itself",
                                id.as_u64()
                            )));
                        }
                        level.push(neighbor);
                    }
                    levels.push(level);
                }
                SavedNode::Graph { hash, levels }
            }
            1 => {
                let representative = NodeId::new(reader.u64()?);
                if representative == id {
                    return Err(corrupt(format!(
                        "alias {} is its own representative",
                        id.as_u64()
                    )));
                }
                SavedNode::Alias { representative }
            }
            tag => {
                return Err(corrupt(format!(
                    "node {} has tag {tag}, where 0 is a graph node and 1 an alias",
                    id.as_u64()
                )));
            }
        };
        let code = match code_length {
            None => None,
            Some(expected) => {
                let length = reader.u32()? as usize;
                if length != expected {
                    return Err(corrupt(format!(
                        "node {} has a code of {length} bytes, where the index's are \
                         {expected}",
                        id.as_u64()
                    )));
                }
                Some(reader.take(length)?.to_vec())
            }
        };
        entries.push((id, node, code));
    }
    if reader.left() != 0 {
        return Err(corrupt(format!(
            "a piece from row {row_start} holds {} bytes after its {nodes} nodes",
            reader.left()
        )));
    }
    Ok(entries)
}

/// The pieces of `entries` (ascending ids): per row group, cut at node
/// boundaries once a piece passes `max_bytes`, each with its first row, its
/// row count and its bytes.
fn cut_pieces(entries: &[PieceEntry], max_bytes: usize) -> Result<Vec<(u64, u32, Vec<u8>)>> {
    // The piece being cut: its first row, its last, its nodes and their bytes.
    struct Open {
        row_start: u64,
        last: u64,
        nodes: u32,
        body: Vec<u8>,
    }
    let close = |open: Open| -> Result<(u64, u32, Vec<u8>)> {
        let mut bytes = Vec::with_capacity(4 + open.body.len());
        bytes.extend_from_slice(&open.nodes.to_le_bytes());
        bytes.extend_from_slice(&open.body);
        let rows = within_group(open.last - open.row_start + 1)?;
        Ok((open.row_start, rows, bytes))
    };
    let mut pieces = Vec::new();
    let mut open: Option<Open> = None;
    let mut one = Vec::new();
    for entry in entries {
        let id = entry.0.as_u64();
        let continues = open
            .as_ref()
            .is_some_and(|piece| piece.row_start / ROWS_PER_GROUP == id / ROWS_PER_GROUP);
        if continues {
            let piece = open.as_mut().expect("continues");
            one.clear();
            encode_entry(&mut one, within_group(id - piece.row_start)?, entry)?;
            if 4 + piece.body.len() + one.len() <= max_bytes {
                piece.body.extend_from_slice(&one);
                piece.nodes += 1;
                piece.last = id;
                continue;
            }
        }
        if let Some(piece) = open.take() {
            pieces.push(close(piece)?);
        }
        let mut body = Vec::new();
        encode_entry(&mut body, 0, entry)?;
        open = Some(Open {
            row_start: id,
            last: id,
            nodes: 1,
            body,
        });
    }
    if let Some(piece) = open {
        pieces.push(close(piece)?);
    }
    Ok(pieces)
}

// ── Streams ─────────────────────────────────────────────────────────

/// Appends `value` as LEB128 to `out`.
fn push_leb128(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Reads a LEB128 value at `*at`, at most 10 bytes.
fn read_leb128(bytes: &[u8], at: &mut usize) -> Option<u64> {
    let mut value = 0_u64;
    for shift in (0..64).step_by(7) {
        let byte = *bytes.get(*at)?;
        *at += 1;
        let part = u64::from(byte & 0x7f);
        if shift == 63 && part > 1 {
            return None;
        }
        value |= part << shift;
        if byte & 0x80 == 0 {
            return Some(value);
        }
    }
    None
}

/// The list stream's bytes: `[count u64]` and the ids as deltas.
fn encode_list(listed: &[NodeId]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + listed.len() * 2);
    out.extend_from_slice(&(listed.len() as u64).to_le_bytes());
    let mut previous = 0;
    for id in listed {
        push_leb128(&mut out, id.as_u64() - previous);
        previous = id.as_u64();
    }
    out
}

/// The listed ids of a list stream: strictly ascending, as many as it says.
fn decode_list(bytes: &[u8]) -> Result<Vec<NodeId>> {
    let head: [u8; 8] = bytes
        .get(..8)
        .and_then(|head| head.try_into().ok())
        .ok_or_else(|| {
            corrupt(format!(
                "a list stream of {} bytes has no count",
                bytes.len()
            ))
        })?;
    let count = u64::from_le_bytes(head);
    let mut at = 8;
    let mut listed: Vec<NodeId> = Vec::with_capacity(
        usize::try_from(count)
            .map_or(bytes.len(), |count| count.min(bytes.len()))
            .min(1 << 20),
    );
    let mut previous: Option<u64> = None;
    for _ in 0..count {
        let delta = read_leb128(bytes, &mut at)
            .ok_or_else(|| corrupt("a list stream ends before its count of ids"))?;
        let id = match previous {
            None => delta,
            Some(_) if delta == 0 => {
                return Err(corrupt("a list stream lists an id twice"));
            }
            Some(previous) => previous
                .checked_add(delta)
                .ok_or_else(|| corrupt("a list stream's ids pass the largest id"))?,
        };
        previous = Some(id);
        listed.push(NodeId::new(id));
    }
    if at != bytes.len() {
        return Err(corrupt(format!(
            "a list stream holds {} bytes after its {count} ids",
            bytes.len() - at
        )));
    }
    Ok(listed)
}

/// The quantizer stream's bytes for a quantized index of `kind` (1 scalar,
/// 2 binary, 3 product).
fn encode_quantizer(kind: u8, saved: &SavedQuantizer) -> Result<Vec<u8>> {
    let mut out = vec![kind];
    let floats = |out: &mut Vec<u8>, values: &[f32]| {
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
    };
    match saved {
        SavedQuantizer::Stateless => out.push(1),
        SavedQuantizer::Unfitted(samples) => {
            out.push(0);
            out.extend_from_slice(&(samples.len() as u64).to_le_bytes());
            for sample in samples {
                out.extend_from_slice(&sample.as_u64().to_le_bytes());
            }
        }
        SavedQuantizer::Scalar(quantizer) => {
            out.push(1);
            out.extend_from_slice(&count_u32(quantizer.dimensions(), "dimensions")?.to_le_bytes());
            floats(&mut out, quantizer.min_values());
            floats(&mut out, quantizer.scales());
            floats(&mut out, quantizer.inverse_scales());
        }
        SavedQuantizer::Product(quantizer) => {
            out.push(1);
            for count in [
                quantizer.num_subvectors(),
                quantizer.num_centroids(),
                quantizer.subvector_dim(),
            ] {
                out.extend_from_slice(&count_u32(count, "quantizer parts")?.to_le_bytes());
            }
            floats(&mut out, quantizer.centroids());
        }
    }
    Ok(out)
}

/// The quantizer of a quantizer stream of an index of `kind`.
fn decode_quantizer(bytes: &[u8], kind: u8) -> Result<SavedQuantizer> {
    let mut reader = PieceReader { bytes, at: 0 };
    let floats = |reader: &mut PieceReader<'_>, count: usize| -> Result<Vec<f32>> {
        if count > reader.left() / 4 {
            return Err(corrupt(format!(
                "a quantizer claims {count} values, more than its stream holds"
            )));
        }
        (0..count)
            .map(|_| {
                Ok(f32::from_le_bytes(
                    reader.take(4)?.try_into().expect("4 bytes"),
                ))
            })
            .collect()
    };
    let stored = reader.u8()?;
    if stored != kind {
        return Err(corrupt(format!(
            "a quantizer of kind {stored} for an index of kind {kind}"
        )));
    }
    let fitted = reader.u8()?;
    let saved = match (kind, fitted) {
        (2, 1) => SavedQuantizer::Stateless,
        (1 | 3, 0) => {
            let count = reader.u64()?;
            if count > (reader.left() / 8) as u64 {
                return Err(corrupt(format!(
                    "a quantizer claims {count} training samples, more than its stream holds"
                )));
            }
            SavedQuantizer::Unfitted(
                (0..count)
                    .map(|_| reader.u64().map(NodeId::new))
                    .collect::<Result<_>>()?,
            )
        }
        (1, 1) => {
            let dimensions = reader.u32()? as usize;
            let min = floats(&mut reader, dimensions)?;
            let scale = floats(&mut reader, dimensions)?;
            let inverse = floats(&mut reader, dimensions)?;
            SavedQuantizer::Scalar(
                ScalarQuantizer::from_parts(min, scale, inverse)
                    .ok_or_else(|| corrupt("a scalar quantizer's parts differ in length"))?,
            )
        }
        (3, 1) => {
            let subvectors = reader.u32()? as usize;
            let centroids = reader.u32()? as usize;
            let subvector_dim = reader.u32()? as usize;
            let count = subvectors
                .checked_mul(centroids)
                .and_then(|count| count.checked_mul(subvector_dim))
                .ok_or_else(|| corrupt("a product quantizer's parts overflow"))?;
            let values = floats(&mut reader, count)?;
            SavedQuantizer::Product(
                ProductQuantizer::from_parts(subvectors, centroids, subvector_dim, values)
                    .ok_or_else(|| {
                        corrupt(format!(
                            "a product quantizer of {subvectors} subvectors of {subvector_dim} \
                             dimensions with {centroids} centroids"
                        ))
                    })?,
            )
        }
        (kind, fitted) => {
            return Err(corrupt(format!(
                "a quantizer of kind {kind} with fitted byte {fitted}"
            )));
        }
    };
    if reader.left() != 0 {
        return Err(corrupt(format!(
            "a quantizer stream holds {} bytes after its quantizer",
            reader.left()
        )));
    }
    Ok(saved)
}

/// The quantizer stream kind of `index`: `None` for an index without
/// quantization.
fn quantizer_kind(index: &VectorIndexKind) -> Option<u8> {
    use crate::index::vector::QuantizationType;
    match index.quantization_type()? {
        QuantizationType::None => None,
        QuantizationType::Scalar => Some(1),
        QuantizationType::Binary => Some(2),
        QuantizationType::Product { .. } => Some(3),
    }
}

// ── One index ───────────────────────────────────────────────────────

/// Writes `index` into `sink` as column `column` of graph `graph_id`: its
/// pieces, its list stream with `listed` (ascending), and for a quantized
/// index its quantizer as stream `quantizer_column`. The entry point is
/// read after the pieces.
///
/// # Errors
///
/// The sink's errors; [`Error::Internal`] for a quantized index without a
/// quantizer column.
pub fn write_vector_index(
    sink: &mut dyn SectionSink,
    graph_id: u32,
    column: u32,
    quantizer_column: Option<u32>,
    index: &VectorIndexKind,
    listed: &[NodeId],
    caps: ChunkCaps,
) -> Result<WrittenIndex> {
    caps.validate()?;
    let hnsw = index.hnsw();
    let quantized = index.as_quantized();
    let saved_quantizer = quantized.map(crate::index::vector::QuantizedHnswIndex::saved_quantizer);
    let entries: Vec<PieceEntry> = hnsw
        .member_ids()
        .into_iter()
        .filter_map(|id| {
            let node = hnsw.saved_form(id)?;
            let code = quantized.and_then(|quantized| quantized.code_of(id));
            Some((id, node, code))
        })
        .collect();
    // A member written while its quantizer is fitted has a code, one before
    // has none: a piece holds codes for all its nodes or for none.
    let code_length = quantized.and_then(crate::index::vector::QuantizedHnswIndex::code_length);
    let mut entries = entries;
    for entry in &mut entries {
        if code_length.is_none() {
            entry.2 = None;
        } else if entry.2.as_ref().map(Vec::len) != code_length {
            return Err(Error::Internal(format!(
                "vector index member {} has no code of the index's {} bytes",
                entry.0.as_u64(),
                code_length.unwrap_or(0)
            )));
        }
    }
    for (row_start, row_count, bytes) in cut_pieces(&entries, caps.max_bytes as usize)? {
        sink.write_chunk(
            ChunkMeta::column(graph_id, column, row_start, row_count, 0),
            &bytes,
        )?;
    }
    write_stream(sink, graph_id, column, &encode_list(listed), caps)?;
    if let (Some(kind), Some(saved)) = (quantizer_kind(index), saved_quantizer) {
        let stream = quantizer_column.ok_or_else(|| {
            Error::Internal("a quantized vector index without a quantizer column".to_string())
        })?;
        write_stream(
            sink,
            graph_id,
            stream,
            &encode_quantizer(kind, &saved)?,
            caps,
        )?;
    }
    let (entry, top) = hnsw.entry_and_top();
    Ok(WrittenIndex {
        members: entries.len() as u64,
        entry: entry
            .map(|entry| Ok::<_, Error>((entry.as_u64(), count_u32(top, "levels")?)))
            .transpose()?,
    })
}

/// Writes `bytes` as stream `stream` of graph `graph_id`.
fn write_stream(
    sink: &mut dyn SectionSink,
    graph_id: u32,
    stream: u32,
    bytes: &[u8],
    caps: ChunkCaps,
) -> Result<()> {
    use std::io::Write as _;
    let mut writer = ChunkStreamWriter::new(sink, graph_id, stream, caps);
    // The writer keeps its first error for `finish`.
    let _ = writer.write_all(bytes);
    writer.finish().map(|_| ())
}

/// Restores the index `meta` describes from `source` into `index`, a fresh
/// index of its definition, and returns its listed nodes (ascending), which
/// the caller derives again from the rows.
///
/// Checks every rule of the format: pieces ascending and apart, each within
/// one row group, the members as many as `meta` says, every neighbor at a
/// level a graph node with more levels or a listed node, every alias's
/// representative a graph node, the entry point a graph node with the top
/// level, or a listed node (then the graph node with the most levels, the
/// lowest id among them, becomes it).
///
/// # Errors
///
/// [`Error::Corruption`] naming what is wrong, or the fetch's error. The
/// index is then left part restored: the caller drops it.
pub fn read_vector_index(
    source: &dyn SectionSource,
    meta: &SavedIndexMeta,
    index: &VectorIndexKind,
) -> Result<Vec<NodeId>> {
    let hnsw = index.hnsw();
    let quantized = index.as_quantized();
    let top = meta.entry.map_or(0, |(_, top)| top as usize);
    hnsw.begin_restore(meta.entry.map(|(id, _)| NodeId::new(id)), top, 0);
    match (quantized, quantizer_kind(index), meta.quantizer_column) {
        (Some(quantized), Some(kind), Some(stream)) => {
            let bytes = read_stream(source, meta.graph_id, stream)?;
            quantized.restore_quantizer(decode_quantizer(&bytes, kind)?)?;
        }
        (_, None, None) => {}
        (_, kind, column) => {
            return Err(corrupt(format!(
                "column {} has quantizer stream {column:?} for an index of quantizer kind \
                 {kind:?}",
                meta.column
            )));
        }
    }
    let code_length = quantized.and_then(crate::index::vector::QuantizedHnswIndex::code_length);
    let mut members = 0_u64;
    let mut end: Option<u64> = None;
    for (position, chunk) in source.chunks().iter().enumerate() {
        if chunk.kind != ChunkKind::Column
            || chunk.graph_id != meta.graph_id
            || chunk.column_id != meta.column
        {
            continue;
        }
        let rows_end = chunk.row_start.checked_add(u64::from(chunk.row_count));
        let in_one_group = chunk.row_count > 0
            && rows_end.is_some_and(|rows_end| {
                chunk.row_start / ROWS_PER_GROUP == (rows_end - 1) / ROWS_PER_GROUP
            });
        if !in_one_group || end.is_some_and(|end| chunk.row_start < end) || chunk.codec != 0 {
            return Err(corrupt(format!(
                "piece {chunk:?} of column {} is not within one row group, overlaps the \
                 piece before it, or has a codec",
                meta.column
            )));
        }
        end = rows_end;
        let bytes: Bytes = source.fetch(position)?;
        for (id, node, code) in decode_piece(&bytes, chunk.row_start, chunk.row_count, code_length)?
        {
            hnsw.restore_saved(id, node);
            if let (Some(quantized), Some(code)) = (quantized, code) {
                quantized.restore_code(id, &code)?;
            }
            members += 1;
        }
    }
    if members != meta.members {
        return Err(corrupt(format!(
            "column {} holds {members} members, where its metadata says {}",
            meta.column, meta.members
        )));
    }
    let listed = decode_list(&read_stream(source, meta.graph_id, meta.column)?)?;
    let is_listed = |id: NodeId| listed.binary_search(&id).is_ok();
    if let Some(broken) = hnsw.first_broken_link_unless(is_listed) {
        return Err(corrupt(format!(
            "node {} lists node {} at level {}, which is no graph node with that level",
            broken.node.as_u64(),
            broken.neighbor.as_u64(),
            broken.level
        )));
    }
    if let Some((alias, representative)) = hnsw.first_stray_alias() {
        return Err(corrupt(format!(
            "alias {} has representative {}, which is no graph node",
            alias.as_u64(),
            representative.as_u64()
        )));
    }
    match meta.entry {
        None if !hnsw.is_empty() => {
            return Err(corrupt(format!(
                "column {} has graph nodes but no entry point",
                meta.column
            )));
        }
        None => {}
        Some((entry, entry_top)) => {
            let entry = NodeId::new(entry);
            let levels = match hnsw.saved_form(entry) {
                Some(SavedNode::Graph { levels, .. }) => Some(levels.len()),
                _ => None,
            };
            if levels != Some(entry_top as usize + 1) {
                if !is_listed(entry) {
                    return Err(corrupt(format!(
                        "the entry point {} of column {} is no graph node with its top level \
                         {entry_top}",
                        entry.as_u64(),
                        meta.column
                    )));
                }
                hnsw.repick_entry_point();
            }
        }
    }
    Ok(listed)
}

#[cfg(test)]
mod tests;
