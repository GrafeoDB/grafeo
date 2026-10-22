//! The row-group store's `LPG_STORE` section, version 4 (FD4, FD5, FD6).
//!
//! Per graph in id order, the node table and then the edge table, each cut
//! into row groups of the caps' `max_rows` rows (the file's row groups; the
//! store's own groups hold 65,536), the chunks of a group together, and the
//! metadata chunk last. In a node row group:
//!
//! - which rows are nodes: column 0 of the node structure, `true` per node;
//! - one column per label a node of the group has, in the node label
//!   namespace, column id = the label's id, `true` per node that has it;
//! - the property columns, column id = the key's id;
//! - the adjacency chunks, outgoing then incoming (`adjacency_chunk`).
//!
//! In an edge row group: the source, target and type columns as three chunks
//! of one range in a row, then the property columns. A delete chunk (node or
//! edge delete namespace, column 0) names rows deleted since their group was
//! written; no writer writes one before chunk reuse (H3), every reader
//! applies them.
//!
//! A checkpoint writes the committed state at the store's epoch: the rows
//! visible then, with their values and labels as they are. A null value is
//! not written. The metadata chunk (layout 2) holds the caps, the epoch, and
//! per graph its name, next ids, dictionaries, whether adjacency was written
//! and each row group's node and edge counts, which the reader checks.
//!
//! A reader refuses everything a writer does not write with
//! `Error::Corruption`, and a section of another version as such.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::Ordering;

use bytes::Bytes;
use grafeo_common::storage::{
    ChunkCaps, ChunkKind, ChunkMeta, ChunkNamespace, SectionSink, SectionSource,
};
use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, Result};
use grafeo_common::utils::hash::FxHashMap;

use super::adjacency::{Adjacency, Adjacent};
use super::adjacency_chunk;
use super::{ABSENT, Inner, ROWS_PER_GROUP, Read, RowGroupStore, RowVersion, locate};
use crate::codec::column_chunk::{ChunkCodec, decode_column_chunk_bytes};
use crate::codec::{ChunkColumn, RowsChunker};
use crate::graph::lpg::dictionary::NameDictionary;

/// The section version this module writes and reads.
pub const LPG_SECTION_VERSION: u8 = 4;

/// The layout byte of the metadata chunk.
const META_LAYOUT: u8 = 2;

/// Column 0 of the node structure namespace: which rows are nodes.
const COLUMN_EXISTS: u32 = 0;
/// The edge structure's columns.
const COLUMN_SOURCE: u32 = 1;
const COLUMN_TARGET: u32 = 2;
const COLUMN_EDGE_TYPE: u32 = 3;

/// An upper bound of an adjacency chunk's bytes: its header and bases, at
/// most three bytes per row of degree and twenty per entry.
fn adjacency_bound(rows: usize, entries: usize) -> usize {
    96 + 3 * rows + 20 * entries
}

/// A graph to write: its stable id, its name (empty for the default graph)
/// and its store.
#[derive(Debug, Clone, Copy)]
pub struct GraphToWrite<'s> {
    /// The graph's id.
    pub id: u32,
    /// Its name, empty for the default graph.
    pub name: &'s str,
    /// Its store.
    pub store: &'s RowGroupStore,
}

/// A graph a section held.
#[derive(Debug)]
pub struct LoadedGraph {
    /// The graph's id.
    pub id: u32,
    /// Its name, empty for the default graph.
    pub name: String,
    /// Its store, as the section held it.
    pub store: RowGroupStore,
}

// ── Metadata ────────────────────────────────────────────────────────

/// A dictionary as the metadata holds it.
#[derive(Debug, Default, PartialEq, Eq)]
struct DictionaryMeta {
    next_id: u32,
    names: Vec<(u32, String)>,
}

impl DictionaryMeta {
    fn of(dictionary: &NameDictionary) -> Self {
        Self {
            next_id: dictionary.next_id(),
            names: dictionary
                .iter()
                .map(|(id, name)| (id, name.to_string()))
                .collect(),
        }
    }
}

/// A graph as the metadata holds it.
#[derive(Debug, Default, PartialEq, Eq)]
struct GraphMeta {
    id: u32,
    name: String,
    next_node_id: u64,
    next_edge_id: u64,
    labels: DictionaryMeta,
    edge_types: DictionaryMeta,
    keys: DictionaryMeta,
    adjacency_written: bool,
    /// The nodes of each file row group that has some, groups ascending.
    node_groups: Vec<(u64, u32)>,
    /// The edges of each file row group that has some.
    edge_groups: Vec<(u64, u32)>,
}

/// The metadata chunk.
#[derive(Debug, PartialEq, Eq)]
struct Meta {
    caps: ChunkCaps,
    epoch: u64,
    next_graph_id: u32,
    graphs: Vec<GraphMeta>,
}

fn put_count(out: &mut Vec<u8>, count: usize, what: &str) -> Result<()> {
    let count = u32::try_from(count)
        .map_err(|_| Error::Internal(format!("cannot write the LPG metadata: {count} {what}")))?;
    out.extend_from_slice(&count.to_le_bytes());
    Ok(())
}

fn put_name(out: &mut Vec<u8>, name: &str) -> Result<()> {
    put_count(out, name.len(), "bytes in a name")?;
    out.extend_from_slice(name.as_bytes());
    Ok(())
}

fn put_dictionary(out: &mut Vec<u8>, dictionary: &DictionaryMeta) -> Result<()> {
    out.extend_from_slice(&dictionary.next_id.to_le_bytes());
    put_count(out, dictionary.names.len(), "names in a dictionary")?;
    for (id, name) in &dictionary.names {
        out.extend_from_slice(&id.to_le_bytes());
        put_name(out, name)?;
    }
    Ok(())
}

fn put_groups(out: &mut Vec<u8>, groups: &[(u64, u32)]) -> Result<()> {
    put_count(out, groups.len(), "row groups")?;
    for (group, rows) in groups {
        out.extend_from_slice(&group.to_le_bytes());
        out.extend_from_slice(&rows.to_le_bytes());
    }
    Ok(())
}

fn encode_meta(meta: &Meta) -> Result<Vec<u8>> {
    let mut out = vec![META_LAYOUT];
    out.extend_from_slice(&meta.caps.max_rows.to_le_bytes());
    out.extend_from_slice(&meta.caps.max_bytes.to_le_bytes());
    out.extend_from_slice(&meta.epoch.to_le_bytes());
    out.extend_from_slice(&meta.next_graph_id.to_le_bytes());
    put_count(&mut out, meta.graphs.len(), "graphs")?;
    for graph in &meta.graphs {
        out.extend_from_slice(&graph.id.to_le_bytes());
        put_name(&mut out, &graph.name)?;
        out.extend_from_slice(&graph.next_node_id.to_le_bytes());
        out.extend_from_slice(&graph.next_edge_id.to_le_bytes());
        put_dictionary(&mut out, &graph.labels)?;
        put_dictionary(&mut out, &graph.edge_types)?;
        put_dictionary(&mut out, &graph.keys)?;
        out.push(u8::from(graph.adjacency_written));
        put_groups(&mut out, &graph.node_groups)?;
        put_groups(&mut out, &graph.edge_groups)?;
    }
    Ok(out)
}

/// A reader over the metadata chunk's bytes.
struct MetaReader<'b> {
    bytes: &'b [u8],
    pos: usize,
}

impl MetaReader<'_> {
    fn refuse(&self, what: impl std::fmt::Display) -> Error {
        Error::corruption(format!("LPG metadata chunk, byte {}: {what}", self.pos))
    }

    fn take<const N: usize>(&mut self, what: &str) -> Result<[u8; N]> {
        let taken = self
            .bytes
            .get(self.pos..self.pos + N)
            .and_then(|bytes| <[u8; N]>::try_from(bytes).ok())
            .ok_or_else(|| self.refuse(format!("the chunk ends inside {what}")))?;
        self.pos += N;
        Ok(taken)
    }

    fn u32(&mut self, what: &str) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(what)?))
    }

    fn u64(&mut self, what: &str) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(what)?))
    }

    /// A count of items of at least `item` bytes each, checked against the
    /// bytes left, so a crafted count allocates nothing.
    fn count(&mut self, item: usize, what: &str) -> Result<usize> {
        let count = self.u32(what)? as usize;
        if count > (self.bytes.len() - self.pos) / item.max(1) {
            return Err(self.refuse(format!("{count} {what}, more than the chunk has bytes for")));
        }
        Ok(count)
    }

    fn name(&mut self, what: &str) -> Result<String> {
        let length = self.count(1, "bytes of a name")?;
        let bytes = &self.bytes[self.pos..self.pos + length];
        let name = std::str::from_utf8(bytes)
            .map_err(|_| self.refuse(format!("{what} is not UTF-8")))?
            .to_string();
        self.pos += length;
        Ok(name)
    }

    fn dictionary(&mut self, what: &str) -> Result<DictionaryMeta> {
        let next_id = self.u32(what)?;
        let count = self.count(8, what)?;
        let mut names: Vec<(u32, String)> = Vec::with_capacity(count);
        for _ in 0..count {
            let id = self.u32(what)?;
            let name = self.name(what)?;
            if id >= next_id || names.last().is_some_and(|(last, _)| *last >= id) {
                return Err(self.refuse(format!(
                    "id {id} of {what} is not ascending below the next id {next_id}"
                )));
            }
            names.push((id, name));
        }
        Ok(DictionaryMeta { next_id, names })
    }

    fn groups(&mut self, what: &str, max_rows: u32) -> Result<Vec<(u64, u32)>> {
        let count = self.count(12, what)?;
        let mut groups: Vec<(u64, u32)> = Vec::with_capacity(count);
        for _ in 0..count {
            let group = self.u64(what)?;
            let rows = self.u32(what)?;
            if rows == 0 || rows > max_rows || groups.last().is_some_and(|(last, _)| *last >= group)
            {
                return Err(self.refuse(format!("{what}: group {group} with {rows} rows")));
            }
            groups.push((group, rows));
        }
        Ok(groups)
    }
}

fn decode_meta(bytes: &[u8]) -> Result<Meta> {
    let mut reader = MetaReader { bytes, pos: 0 };
    let layout = reader.take::<1>("the layout")?[0];
    if layout != META_LAYOUT {
        return Err(Error::Serialization(format!(
            "LPG metadata chunk: layout {layout}, where this version reads {META_LAYOUT}: written by another version"
        )));
    }
    let caps = ChunkCaps {
        max_rows: reader.u32("the row cap")?,
        max_bytes: reader.u32("the byte cap")?,
    };
    caps.validate()
        .map_err(|error| reader.refuse(format!("the caps: {error}")))?;
    let epoch = reader.u64("the epoch")?;
    let next_graph_id = reader.u32("the next graph id")?;
    let count = reader.count(60, "graphs")?;
    let mut graphs: Vec<GraphMeta> = Vec::with_capacity(count);
    for _ in 0..count {
        let id = reader.u32("a graph id")?;
        if id >= next_graph_id || graphs.last().is_some_and(|last| last.id >= id) {
            return Err(reader.refuse(format!(
                "graph id {id} is not ascending below the next id {next_graph_id}"
            )));
        }
        let name = reader.name("a graph name")?;
        let next_node_id = reader.u64("the next node id")?;
        let next_edge_id = reader.u64("the next edge id")?;
        let labels = reader.dictionary("labels")?;
        let edge_types = reader.dictionary("edge types")?;
        let keys = reader.dictionary("property keys")?;
        let adjacency_written = match reader.take::<1>("the adjacency flag")?[0] {
            0 => false,
            1 => true,
            other => return Err(reader.refuse(format!("adjacency flag {other}"))),
        };
        graphs.push(GraphMeta {
            id,
            name,
            next_node_id,
            next_edge_id,
            labels,
            edge_types,
            keys,
            adjacency_written,
            node_groups: reader.groups("node row groups", caps.max_rows)?,
            edge_groups: reader.groups("edge row groups", caps.max_rows)?,
        });
    }
    if reader.pos != bytes.len() {
        return Err(reader.refuse(format!(
            "{} bytes after the last graph",
            bytes.len() - reader.pos
        )));
    }
    Ok(Meta {
        caps,
        epoch,
        next_graph_id,
        graphs,
    })
}

// ── Writing ─────────────────────────────────────────────────────────

fn column(namespace: ChunkNamespace, column_id: u32) -> ChunkColumn {
    ChunkColumn {
        kind: ChunkKind::Column,
        namespace,
        column_id,
    }
}

/// An id as the `Int64` an edge column holds.
fn int(id: u64, what: &str) -> Result<Value> {
    i64::try_from(id).map(Value::Int64).map_err(|_| {
        Error::Internal(format!(
            "cannot write the LPG section: {what} {id} is past the ids a file holds"
        ))
    })
}

/// Writes the adjacency chunks of one file row group and direction:
/// `lists` holds each node row of the group with edges, ascending, with its
/// sorted list.
fn write_adjacency(
    sink: &mut dyn SectionSink,
    graph_id: u32,
    namespace: ChunkNamespace,
    group_start: u64,
    lists: &[(u32, Vec<Adjacent>)],
    caps: ChunkCaps,
) -> Result<()> {
    let max_bytes = caps.max_bytes as usize;
    let cap = caps.max_rows as usize;
    let meta = |row: u32, rows: usize, piece: u32| -> Result<ChunkMeta> {
        Ok(ChunkMeta {
            kind: ChunkKind::Adjacency,
            ..ChunkMeta::column(
                graph_id,
                piece,
                group_start + u64::from(row),
                u32::try_from(rows).map_err(|_| {
                    Error::Internal("an adjacency chunk past the row cap".to_string())
                })?,
                0,
            )
            .in_namespace(namespace)
        })
    };
    let mut open: Vec<&(u32, Vec<Adjacent>)> = Vec::new();
    let mut entries = 0;
    let flush = |open: &mut Vec<&(u32, Vec<Adjacent>)>, sink: &mut dyn SectionSink| -> Result<()> {
        let (Some(first), Some(last)) = (open.first(), open.last()) else {
            return Ok(());
        };
        let rows = (last.0 - first.0) as usize + 1;
        let mut dense: Vec<&[Adjacent]> = vec![&[]; rows];
        for (row, list) in open.iter() {
            dense[(row - first.0) as usize] = list;
        }
        let bytes = adjacency_chunk::encode(&dense, 0, caps.max_rows)?;
        sink.write_chunk(meta(first.0, rows, 0)?, &bytes)?;
        open.clear();
        Ok(())
    };
    for item in lists {
        let (row, list) = item;
        // A list that fills a chunk alone goes in pieces of its own.
        let piece = cap.min((max_bytes.saturating_sub(adjacency_bound(1, 0)) / 20).max(1));
        if list.len() > piece {
            flush(&mut open, sink)?;
            entries = 0;
            for (number, part) in list.chunks(piece).enumerate() {
                let first_entry = u32::try_from(number * piece).map_err(|_| {
                    Error::Internal(
                        "a node's adjacency list past 4,294,967,295 entries".to_string(),
                    )
                })?;
                let bytes = adjacency_chunk::encode(&[part], first_entry, caps.max_rows)?;
                let number = u32::try_from(number)
                    .map_err(|_| Error::Internal("too many adjacency pieces".to_string()))?;
                sink.write_chunk(meta(*row, 1, number)?, &bytes)?;
            }
            continue;
        }
        if let Some(first) = open.first() {
            let rows = (row - first.0) as usize + 1;
            if entries + list.len() > cap || adjacency_bound(rows, entries + list.len()) > max_bytes
            {
                flush(&mut open, sink)?;
                entries = 0;
            }
        }
        entries += list.len();
        open.push(item);
    }
    flush(&mut open, sink)
}

impl Inner {
    /// A node's edges in one direction that `read` sees, sorted.
    fn visible_adjacency(&self, id: u64, outgoing: bool, read: Read) -> Vec<Adjacent> {
        let Some((group, row)) = self.node(id) else {
            return Vec::new();
        };
        let mut list: Vec<Adjacent> = group
            .adjacency(outgoing)
            .of(row)
            .filter(|adjacent| self.edge_visible(adjacent.edge, read))
            .collect();
        list.sort_unstable();
        list
    }

    /// Writes one graph's tables as `read` sees them, and returns its
    /// metadata.
    fn write_graph(
        &self,
        graph: &GraphToWrite<'_>,
        read: Read,
        caps: ChunkCaps,
        sink: &mut dyn SectionSink,
    ) -> Result<GraphMeta> {
        let max_rows = u64::from(caps.max_rows);
        let mut meta = GraphMeta {
            id: graph.id,
            name: graph.name.to_string(),
            next_node_id: graph.store.next_node_id.load(Ordering::Acquire),
            next_edge_id: graph.store.next_edge_id.load(Ordering::Acquire),
            labels: DictionaryMeta::of(&self.labels),
            edge_types: DictionaryMeta::of(&self.edge_types),
            keys: DictionaryMeta::of(&self.keys),
            adjacency_written: true,
            ..GraphMeta::default()
        };

        // The node table, a file row group at a time.
        let mut nodes: Vec<u64> = Vec::new();
        for (index, group) in &self.nodes {
            for (row, version) in group.versions.iter().enumerate() {
                if version.visible(read) {
                    nodes.push(index * ROWS_PER_GROUP + row as u64);
                }
            }
        }
        for rows in nodes.chunk_by(|a, b| a / max_rows == b / max_rows) {
            let file_group = rows[0] / max_rows;
            let group_start = file_group * max_rows;
            let mut labels: BTreeSet<u32> = BTreeSet::new();
            let mut keys: BTreeSet<u32> = BTreeSet::new();
            let mut cells: Vec<(u64, Vec<u32>, FxHashMap<u32, Value>)> =
                Vec::with_capacity(rows.len());
            for &id in rows {
                let (group, row) = self.node(id).expect("a row it listed");
                let row_labels = group.labels_of(row);
                let values: FxHashMap<u32, Value> = group
                    .columns
                    .row(row)
                    .filter(|(_, value)| !value.is_null())
                    .collect();
                labels.extend(&row_labels);
                keys.extend(values.keys());
                cells.push((id, row_labels, values));
            }
            let columns: Vec<ChunkColumn> =
                std::iter::once(column(ChunkNamespace::NodeStructure, COLUMN_EXISTS))
                    .chain(
                        labels
                            .iter()
                            .map(|label| column(ChunkNamespace::NodeLabels, *label)),
                    )
                    .chain(
                        keys.iter()
                            .map(|key| column(ChunkNamespace::NodeProperties, *key)),
                    )
                    .collect();
            let mut chunker = RowsChunker::new(graph.id, columns, group_start, caps);
            for (id, row_labels, mut values) in cells {
                let row: Vec<Option<(Value, u64)>> =
                    std::iter::once(Some((Value::Bool(true), 0)))
                        .chain(labels.iter().map(|label| {
                            row_labels.contains(label).then_some((Value::Bool(true), 0))
                        }))
                        .chain(
                            keys.iter()
                                .map(|key| values.remove(key).map(|value| (value, 0))),
                        )
                        .collect();
                chunker.push(sink, id, row)?;
            }
            chunker.finish(sink)?;
            for (namespace, outgoing) in [
                (ChunkNamespace::OutgoingAdjacency, true),
                (ChunkNamespace::IncomingAdjacency, false),
            ] {
                let lists: Vec<(u32, Vec<Adjacent>)> = rows
                    .iter()
                    .map(|id| {
                        (
                            u32::try_from(id - group_start).expect("a row below the row cap"),
                            self.visible_adjacency(*id, outgoing, read),
                        )
                    })
                    .filter(|(_, list)| !list.is_empty())
                    .collect();
                write_adjacency(sink, graph.id, namespace, group_start, &lists, caps)?;
            }
            meta.node_groups.push((
                file_group,
                u32::try_from(rows.len()).expect("rows of one group"),
            ));
        }

        // The edge table.
        let mut edges: Vec<u64> = Vec::new();
        for (index, group) in &self.edges {
            for (row, version) in group.versions.iter().enumerate() {
                if version.visible(read) {
                    edges.push(index * ROWS_PER_GROUP + row as u64);
                }
            }
        }
        for rows in edges.chunk_by(|a, b| a / max_rows == b / max_rows) {
            let file_group = rows[0] / max_rows;
            let mut keys: BTreeSet<u32> = BTreeSet::new();
            let mut cells: Vec<(u64, (u64, u64, u32), FxHashMap<u32, Value>)> =
                Vec::with_capacity(rows.len());
            for &id in rows {
                let (group, row) = self.edge(id).expect("a row it listed");
                let values: FxHashMap<u32, Value> = group
                    .columns
                    .row(row)
                    .filter(|(_, value)| !value.is_null())
                    .collect();
                keys.extend(values.keys());
                cells.push((id, group.ends(row), values));
            }
            let columns: Vec<ChunkColumn> = [COLUMN_SOURCE, COLUMN_TARGET, COLUMN_EDGE_TYPE]
                .into_iter()
                .map(|id| column(ChunkNamespace::EdgeStructure, id))
                .chain(
                    keys.iter()
                        .map(|key| column(ChunkNamespace::EdgeProperties, *key)),
                )
                .collect();
            let mut chunker = RowsChunker::new(graph.id, columns, file_group * max_rows, caps);
            for (id, (src, dst, edge_type), mut values) in cells {
                let row: Vec<Option<(Value, u64)>> = [
                    int(src, "source node")?,
                    int(dst, "target node")?,
                    Value::Int64(i64::from(edge_type)),
                ]
                .into_iter()
                .map(|value| Some((value, 0)))
                .chain(
                    keys.iter()
                        .map(|key| values.remove(key).map(|value| (value, 0))),
                )
                .collect();
                chunker.push(sink, id, row)?;
            }
            chunker.finish(sink)?;
            meta.edge_groups.push((
                file_group,
                u32::try_from(rows.len()).expect("rows of one group"),
            ));
        }
        Ok(meta)
    }
}

/// Writes the `LPG_STORE` section, version 4, of `graphs` (ids ascending):
/// each graph's committed state at its store's epoch.
///
/// # Errors
///
/// Returns [`Error::InvalidValue`] for caps the format refuses,
/// [`Error::Internal`] for graphs out of id order or an id at or above
/// `next_graph_id`, and the encoder's or the sink's errors.
pub fn write_lpg_section(
    graphs: &[GraphToWrite<'_>],
    next_graph_id: u32,
    caps: ChunkCaps,
    sink: &mut dyn SectionSink,
) -> Result<()> {
    caps.validate()?;
    if !graphs.windows(2).all(|pair| pair[0].id < pair[1].id)
        || graphs.iter().any(|graph| graph.id >= next_graph_id)
    {
        return Err(Error::Internal(
            "cannot write the LPG section: graph ids must ascend below the next graph id"
                .to_string(),
        ));
    }
    let mut meta = Meta {
        caps,
        epoch: 0,
        next_graph_id,
        graphs: Vec::with_capacity(graphs.len()),
    };
    for graph in graphs {
        let inner = graph.store.inner.read();
        let epoch = graph.store.epoch.load(Ordering::Acquire);
        meta.epoch = meta.epoch.max(epoch);
        meta.graphs
            .push(inner.write_graph(graph, Read::At(epoch), caps, sink)?);
    }
    sink.write_chunk(ChunkMeta::meta(), &encode_meta(&meta)?)
}

// ── Reading ─────────────────────────────────────────────────────────

/// A graph being loaded.
struct Loading {
    meta_index: usize,
    inner: Inner,
    /// Each direction's lists as the chunks gave them: node id and list,
    /// nodes ascending.
    adjacency: [Vec<(u64, Vec<Adjacent>)>; 2],
    /// Rows the delete chunks name.
    deleted_nodes: Vec<u64>,
    deleted_edges: Vec<u64>,
    /// The nodes and edges read per file row group.
    node_groups: BTreeMap<u64, u32>,
    edge_groups: BTreeMap<u64, u32>,
}

/// Where a chunk is, for an error.
fn place(index: usize, chunk: &ChunkMeta) -> String {
    format!(
        "LPG section, chunk {index} ({:?} in {:?}, graph {}, column {}, rows {} to {})",
        chunk.kind,
        chunk.namespace,
        chunk.graph_id,
        chunk.column_id,
        chunk.row_start,
        chunk.row_start + u64::from(chunk.row_count).saturating_sub(1)
    )
}

/// The rows of a `Bitmap` chunk of `true` values, as ids.
fn true_rows(bytes: &Bytes, chunk: &ChunkMeta) -> std::result::Result<Vec<u64>, String> {
    if chunk.codec != ChunkCodec::Bitmap.to_byte() {
        return Err(format!(
            "codec {}, where these chunks are bitmaps",
            chunk.codec
        ));
    }
    let decoded = decode_column_chunk_bytes(bytes, chunk.codec, chunk.row_count)
        .map_err(|error| error.to_string())?;
    if decoded.epochs.is_some() {
        return Err("epochs, which these chunks never carry".to_string());
    }
    decoded
        .values
        .into_iter()
        .map(|(row, value)| match value {
            Value::Bool(true) => Ok(chunk.row_start + u64::from(row)),
            other => Err(format!(
                "row {} holds {other:?}, where every row holds true",
                chunk.row_start + u64::from(row)
            )),
        })
        .collect()
}

/// The rows and `Int64` values of an edge structure chunk.
fn id_rows(bytes: &Bytes, chunk: &ChunkMeta) -> std::result::Result<Vec<(u64, u64)>, String> {
    let decoded = decode_column_chunk_bytes(bytes, chunk.codec, chunk.row_count)
        .map_err(|error| error.to_string())?;
    if decoded.epochs.is_some() {
        return Err("epochs, which the edge columns never carry".to_string());
    }
    decoded
        .values
        .into_iter()
        .map(|(row, value)| match value {
            Value::Int64(id) if id >= 0 => {
                Ok((chunk.row_start + u64::from(row), id.cast_unsigned()))
            }
            other => Err(format!(
                "row {} holds {other:?}, where it holds an id",
                chunk.row_start + u64::from(row)
            )),
        })
        .collect()
}

fn restore_dictionary(
    target: &mut NameDictionary,
    meta: &DictionaryMeta,
    what: &str,
) -> Result<()> {
    for (id, name) in &meta.names {
        target
            .insert_at(*id, name)
            .map_err(|error| Error::corruption(format!("LPG metadata chunk: {what}: {error}")))?;
    }
    target.reserve_below(meta.next_id);
    Ok(())
}

/// Reads an `LPG_STORE` section of version 4: the next graph id and each
/// graph's store, as the checkpoint that wrote it saw them.
///
/// # Errors
///
/// Returns [`Error::Serialization`] for a section of another version (written
/// by another release), and [`Error::Corruption`] for everything else a
/// writer does not write, naming the chunk.
#[expect(
    clippy::too_many_lines,
    reason = "one pass over the chunks, a check per kind"
)]
pub fn read_lpg_section(source: &dyn SectionSource) -> Result<(u32, Vec<LoadedGraph>)> {
    if source.section_version() != LPG_SECTION_VERSION {
        return Err(Error::Serialization(format!(
            "LPG section version {}, where this reader reads version {LPG_SECTION_VERSION}: written by another version",
            source.section_version()
        )));
    }
    let chunks = source.chunks();
    let Some((last, data)) = chunks.split_last() else {
        return Err(Error::corruption("LPG section: no metadata chunk"));
    };
    if last.kind != ChunkKind::Meta || data.iter().any(|chunk| chunk.kind == ChunkKind::Meta) {
        return Err(Error::corruption(
            "LPG section: its last chunk, and only that one, is the metadata chunk",
        ));
    }
    let meta = decode_meta(&source.fetch(chunks.len() - 1)?)?;
    let max_rows = u64::from(meta.caps.max_rows);

    let mut loading: Vec<Loading> = Vec::with_capacity(meta.graphs.len());
    for (meta_index, graph) in meta.graphs.iter().enumerate() {
        let mut inner = Inner::default();
        restore_dictionary(&mut inner.labels, &graph.labels, "labels")?;
        restore_dictionary(&mut inner.edge_types, &graph.edge_types, "edge types")?;
        restore_dictionary(&mut inner.keys, &graph.keys, "property keys")?;
        loading.push(Loading {
            meta_index,
            inner,
            adjacency: [Vec::new(), Vec::new()],
            deleted_nodes: Vec::new(),
            deleted_edges: Vec::new(),
            node_groups: BTreeMap::new(),
            edge_groups: BTreeMap::new(),
        });
    }
    let position: FxHashMap<u32, usize> = meta
        .graphs
        .iter()
        .enumerate()
        .map(|(index, graph)| (graph.id, index))
        .collect();

    // The end of the last chunk of each column, against overlaps.
    let mut ends: FxHashMap<(u32, u8, u8, u32), u64> = FxHashMap::default();
    // Per graph and direction: the last column-0 adjacency chunk (row, rows)
    // and the pieces and entries of its list so far.
    let mut pieces: FxHashMap<(u32, u8), (u64, u32, u32, usize)> = FxHashMap::default();
    let mut order = (0_usize, 0_u8, 0_u64);
    let loaded = RowVersion {
        created: 0,
        deleted: ABSENT,
    };

    let mut index = 0;
    while index < data.len() {
        let chunk = &data[index];
        let refuse = |what: String| Error::corruption(format!("{}: {what}", place(index, chunk)));
        let Some(&graph_at) = position.get(&chunk.graph_id) else {
            return Err(refuse("a graph the metadata does not list".to_string()));
        };
        let graph = &mut loading[graph_at];
        let graph_meta = &meta.graphs[graph.meta_index];
        let node_table = match chunk.namespace {
            ChunkNamespace::NodeStructure
            | ChunkNamespace::NodeProperties
            | ChunkNamespace::NodeDeletes
            | ChunkNamespace::NodeLabels
            | ChunkNamespace::OutgoingAdjacency
            | ChunkNamespace::IncomingAdjacency => true,
            ChunkNamespace::EdgeStructure
            | ChunkNamespace::EdgeProperties
            | ChunkNamespace::EdgeDeletes => false,
            _ => {
                return Err(refuse(
                    "a namespace an LPG section does not have".to_string(),
                ));
            }
        };
        let next_id = if node_table {
            graph_meta.next_node_id
        } else {
            graph_meta.next_edge_id
        };
        let end = chunk.row_start + u64::from(chunk.row_count);
        let file_group = chunk.row_start / max_rows;
        if chunk.row_count == 0
            || u64::from(chunk.row_count) > max_rows
            || (end - 1) / max_rows != file_group
        {
            return Err(refuse(format!(
                "rows that do not lie in one row group of {max_rows}"
            )));
        }
        if end > next_id {
            return Err(refuse(format!("rows past the table's next id {next_id}")));
        }
        let here = (graph_at, u8::from(!node_table), file_group);
        if here < order {
            return Err(refuse(
                "out of order: graphs, then the node and the edge table, then row groups"
                    .to_string(),
            ));
        }
        order = here;
        let adjacency = matches!(
            chunk.namespace,
            ChunkNamespace::OutgoingAdjacency | ChunkNamespace::IncomingAdjacency
        );
        if adjacency != (chunk.kind == ChunkKind::Adjacency)
            || !matches!(chunk.kind, ChunkKind::Column | ChunkKind::Adjacency)
        {
            return Err(refuse(
                "a chunk kind its namespace does not hold".to_string(),
            ));
        }
        let column_key = if adjacency { 0 } else { chunk.column_id };
        let identity = (
            chunk.graph_id,
            chunk.namespace.to_byte(),
            chunk.kind.to_byte(),
            column_key,
        );
        let piece = adjacency && chunk.column_id > 0;
        if !piece {
            if ends
                .get(&identity)
                .is_some_and(|last_end| chunk.row_start < *last_end)
            {
                return Err(refuse(
                    "rows that overlap or come before the column's chunk before it".to_string(),
                ));
            }
            ends.insert(identity, end);
        }
        let bytes = source.fetch(index)?;

        match chunk.namespace {
            ChunkNamespace::NodeStructure => {
                if chunk.column_id != COLUMN_EXISTS {
                    return Err(refuse(format!(
                        "the node structure holds column {COLUMN_EXISTS} only"
                    )));
                }
                let rows = true_rows(&bytes, chunk).map_err(refuse)?;
                *graph.node_groups.entry(file_group).or_default() +=
                    u32::try_from(rows.len()).unwrap_or(u32::MAX);
                for id in rows {
                    let (group, row) = locate(id);
                    *graph.inner.nodes.entry(group).or_default().version_mut(row) = loaded;
                }
            }
            ChunkNamespace::NodeLabels => {
                if graph.inner.labels.get_name(chunk.column_id).is_none() {
                    return Err(refuse(format!(
                        "label {} is not in the graph's dictionary",
                        chunk.column_id
                    )));
                }
                for id in true_rows(&bytes, chunk).map_err(refuse)? {
                    let Some((group, row)) = graph.inner.node_mut(id) else {
                        return Err(refuse(format!("row {id} is no node of its group")));
                    };
                    group.labels.entry(chunk.column_id).or_default().set(row);
                }
            }
            ChunkNamespace::NodeProperties | ChunkNamespace::EdgeProperties => {
                if graph.inner.keys.get_name(chunk.column_id).is_none() {
                    return Err(refuse(format!(
                        "column {} is no property key of the graph",
                        chunk.column_id
                    )));
                }
                let decoded = decode_column_chunk_bytes(&bytes, chunk.codec, chunk.row_count)
                    .map_err(|error| refuse(error.to_string()))?;
                if decoded.epochs.is_some() {
                    return Err(refuse(
                        "epochs, which version 4 does not write yet".to_string(),
                    ));
                }
                // A chunk inside one of the store's row groups stays as
                // its bytes until the column's first use; one across two
                // (caps below the store's group) is decoded here.
                let (group, first_row) = locate(chunk.row_start);
                let cold = locate(end - 1).0 == group;
                for (offset, value) in decoded.values {
                    let id = chunk.row_start + u64::from(offset);
                    if value.is_null() {
                        return Err(refuse(format!(
                            "row {id} holds a null, which is not written"
                        )));
                    }
                    let columns = if node_table {
                        graph
                            .inner
                            .node_mut(id)
                            .map(|(group, row)| (&mut group.columns, row))
                    } else {
                        graph
                            .inner
                            .edge_mut(id)
                            .map(|(group, row)| (&mut group.columns, row))
                    };
                    let Some((columns, row)) = columns else {
                        return Err(refuse(format!("row {id} is no node or edge of its group")));
                    };
                    if !cold {
                        columns.set(chunk.column_id, row, &value);
                    }
                }
                if cold {
                    let columns = if node_table {
                        graph
                            .inner
                            .nodes
                            .get_mut(&group)
                            .map(|group| &mut group.columns)
                    } else {
                        graph
                            .inner
                            .edges
                            .get_mut(&group)
                            .map(|group| &mut group.columns)
                    };
                    columns
                        .expect("the group of the rows checked above")
                        .add_cold(
                            chunk.column_id,
                            first_row,
                            chunk.row_count,
                            chunk.codec,
                            bytes,
                        );
                }
            }
            ChunkNamespace::NodeDeletes | ChunkNamespace::EdgeDeletes => {
                if chunk.column_id != 0 {
                    return Err(refuse("a delete chunk has column 0".to_string()));
                }
                let rows = true_rows(&bytes, chunk).map_err(refuse)?;
                if rows.first() != Some(&chunk.row_start) || rows.last() != Some(&(end - 1)) {
                    return Err(refuse(
                        "a delete chunk runs from its first to its last deleted row".to_string(),
                    ));
                }
                for id in &rows {
                    let exists = if node_table {
                        graph.inner.node(*id).is_some()
                    } else {
                        graph.inner.edge(*id).is_some()
                    };
                    if !exists {
                        return Err(refuse(format!("row {id} is no node or edge of its group")));
                    }
                }
                if node_table {
                    graph.deleted_nodes.extend(rows);
                } else {
                    graph.deleted_edges.extend(rows);
                }
            }
            ChunkNamespace::EdgeStructure => {
                if chunk.column_id != COLUMN_SOURCE {
                    return Err(refuse(format!(
                        "column {} comes without column {COLUMN_SOURCE} of the same rows before it",
                        chunk.column_id
                    )));
                }
                let sources = id_rows(&bytes, chunk).map_err(refuse)?;
                let mut partners = Vec::with_capacity(2);
                for (step, column_id) in [(1, COLUMN_TARGET), (2, COLUMN_EDGE_TYPE)] {
                    let partner = data.get(index + step).filter(|partner| {
                        partner.kind == ChunkKind::Column
                            && partner.namespace == ChunkNamespace::EdgeStructure
                            && partner.graph_id == chunk.graph_id
                            && partner.column_id == column_id
                            && partner.row_start == chunk.row_start
                            && partner.row_count == chunk.row_count
                    });
                    let Some(partner) = partner else {
                        return Err(refuse(format!(
                            "column {column_id} of the same rows does not follow"
                        )));
                    };
                    let rows = id_rows(&source.fetch(index + step)?, partner).map_err(|what| {
                        Error::corruption(format!("{}: {what}", place(index + step, partner)))
                    })?;
                    if rows
                        .iter()
                        .map(|(id, _)| id)
                        .ne(sources.iter().map(|(id, _)| id))
                    {
                        return Err(refuse(format!(
                            "column {column_id} holds other rows than the sources"
                        )));
                    }
                    partners.push(rows);
                }
                *graph.edge_groups.entry(file_group).or_default() +=
                    u32::try_from(sources.len()).unwrap_or(u32::MAX);
                for (at, (id, src)) in sources.iter().enumerate() {
                    let (dst, edge_type) = (partners[0][at].1, partners[1][at].1);
                    let Some(edge_type) = u32::try_from(edge_type)
                        .ok()
                        .filter(|edge_type| graph.inner.edge_types.get_name(*edge_type).is_some())
                    else {
                        return Err(refuse(format!(
                            "edge {id} has type {edge_type}, which is not in the dictionary"
                        )));
                    };
                    if graph.inner.node(*src).is_none() || graph.inner.node(dst).is_none() {
                        return Err(refuse(format!(
                            "edge {id} runs {src} to {dst}, which are not both nodes"
                        )));
                    }
                    let (group, row) = locate(*id);
                    let group = graph.inner.edges.entry(group).or_default();
                    *group.version_mut(row) = loaded;
                    group.set_ends(row, *src, dst, edge_type);
                }
                index += 2;
            }
            ChunkNamespace::OutgoingAdjacency | ChunkNamespace::IncomingAdjacency => {
                if !graph_meta.adjacency_written {
                    return Err(refuse(
                        "an adjacency chunk in a graph whose metadata says none was written"
                            .to_string(),
                    ));
                }
                let direction = usize::from(chunk.namespace == ChunkNamespace::IncomingAdjacency);
                let decoded = adjacency_chunk::decode(&bytes, chunk.row_count, meta.caps.max_rows)
                    .map_err(|error| error.wrapped(place(index, chunk)))?;
                let key = (chunk.graph_id, chunk.namespace.to_byte());
                if piece {
                    let Some((row, rows, next_piece, entries)) = pieces.get_mut(&key) else {
                        return Err(refuse("a piece without its list's first chunk".to_string()));
                    };
                    if *row != chunk.row_start
                        || *rows != 1
                        || chunk.row_count != 1
                        || *next_piece != chunk.column_id
                        || decoded.first_entry as usize != *entries
                    {
                        return Err(refuse(format!(
                            "piece {} from entry {} does not continue the list of row {row}",
                            chunk.column_id, decoded.first_entry
                        )));
                    }
                    let list = &decoded.lists[0];
                    let (_, whole) = graph.adjacency[direction]
                        .last_mut()
                        .expect("the list's first chunk");
                    if whole
                        .last()
                        .is_some_and(|last| list.first().is_some_and(|first| last >= first))
                    {
                        return Err(refuse(
                            "a piece that does not sort after the piece before it".to_string(),
                        ));
                    }
                    *next_piece += 1;
                    *entries += list.len();
                    whole.extend(list);
                } else {
                    if decoded.first_entry != 0 {
                        return Err(refuse(format!(
                            "a first chunk from entry {}",
                            decoded.first_entry
                        )));
                    }
                    let entries = decoded.lists.first().map_or(0, Vec::len);
                    pieces.insert(key, (chunk.row_start, chunk.row_count, 1, entries));
                    for (offset, list) in decoded.lists.into_iter().enumerate() {
                        if list.is_empty() {
                            continue;
                        }
                        let id = chunk.row_start + offset as u64;
                        if graph.inner.node(id).is_none() {
                            return Err(refuse(format!("row {id} is no node of its group")));
                        }
                        graph.adjacency[direction].push((id, list));
                    }
                }
            }
            _ => unreachable!("refused above"),
        }
        index += 1;
    }

    let mut graphs = Vec::with_capacity(loading.len());
    for mut graph in loading {
        let graph_meta = &meta.graphs[graph.meta_index];
        let refuse = |what: String| {
            Error::corruption(format!("LPG section, graph {}: {what}", graph_meta.id))
        };
        let read: Vec<(u64, u32)> = graph
            .node_groups
            .iter()
            .map(|(group, rows)| (*group, *rows))
            .collect();
        if read != graph_meta.node_groups {
            return Err(refuse(format!(
                "its node row groups hold {read:?}, where the metadata counts {:?}",
                graph_meta.node_groups
            )));
        }
        let read: Vec<(u64, u32)> = graph
            .edge_groups
            .iter()
            .map(|(group, rows)| (*group, *rows))
            .collect();
        if read != graph_meta.edge_groups {
            return Err(refuse(format!(
                "its edge row groups hold {read:?}, where the metadata counts {:?}",
                graph_meta.edge_groups
            )));
        }

        // Adjacency against the edge table: every entry names an edge with
        // these ends and type, and every edge is in both directions.
        let edge_count: usize = graph_meta
            .edge_groups
            .iter()
            .map(|(_, rows)| *rows as usize)
            .sum();
        if graph_meta.adjacency_written {
            for (direction, lists) in graph.adjacency.iter().enumerate() {
                let mut entries = 0;
                for (node, list) in lists {
                    for adjacent in list {
                        let ends = graph
                            .inner
                            .edge(adjacent.edge)
                            .map(|(group, row)| group.ends(row));
                        let expected = if direction == 0 {
                            (*node, adjacent.other, adjacent.edge_type)
                        } else {
                            (adjacent.other, *node, adjacent.edge_type)
                        };
                        if ends != Some(expected) {
                            return Err(refuse(format!(
                                "node {node}'s adjacency names edge {} as {expected:?}, where the edge table has {ends:?}",
                                adjacent.edge
                            )));
                        }
                    }
                    entries += list.len();
                }
                if entries != edge_count {
                    return Err(refuse(format!(
                        "its adjacency lists {entries} edges in direction {direction}, where the edge table has {edge_count}"
                    )));
                }
            }
            for (direction, lists) in graph.adjacency.into_iter().enumerate() {
                let mut by_group: BTreeMap<u64, Vec<(usize, Vec<Adjacent>)>> = BTreeMap::new();
                for (node, list) in lists {
                    let (group, row) = locate(node);
                    by_group.entry(group).or_default().push((row, list));
                }
                for (group, lists) in by_group {
                    let group = graph
                        .inner
                        .nodes
                        .get_mut(&group)
                        .expect("a group of its nodes");
                    *group.adjacency_mut(direction == 0) = Adjacency::from_sorted(lists);
                }
            }
        } else {
            // A file written without adjacency: build it from the edge table.
            let edges: Vec<(u64, (u64, u64, u32))> = graph
                .inner
                .edges
                .iter()
                .flat_map(|(index, group)| {
                    group
                        .versions
                        .iter()
                        .enumerate()
                        .filter(|(_, version)| version.exists())
                        .map(move |(row, _)| (index * ROWS_PER_GROUP + row as u64, group.ends(row)))
                })
                .collect();
            for (edge, (src, dst, edge_type)) in edges {
                for (node, other, outgoing) in [(src, dst, true), (dst, src, false)] {
                    let (group, row) = locate(node);
                    let group = graph
                        .inner
                        .nodes
                        .get_mut(&group)
                        .expect("an endpoint it checked");
                    group.adjacency_mut(outgoing).push(
                        row,
                        Adjacent {
                            edge_type,
                            other,
                            edge,
                        },
                    );
                }
            }
            for group in graph.inner.nodes.values_mut() {
                group.adjacency_mut(true).merge();
                group.adjacency_mut(false).merge();
            }
        }

        // The rows deleted since their group was written: edges first.
        for edge in std::mem::take(&mut graph.deleted_edges) {
            graph.inner.remove_edge_row(edge);
        }
        for node in std::mem::take(&mut graph.deleted_nodes) {
            if graph.inner.sees_an_edge_of(node, Read::At(0)) {
                return Err(refuse(format!(
                    "node {node} is deleted, an edge of it is not"
                )));
            }
            if let Some((group, row)) = graph.inner.node_mut(node) {
                group.remove(row);
            }
        }

        // The counts of what was loaded.
        let mut counts = super::Counts::default();
        for group in graph.inner.nodes.values() {
            for (row, version) in group.versions.iter().enumerate() {
                if version.exists() {
                    counts.nodes += 1;
                    for label in group.labels_of(row) {
                        counts.label(label, 1);
                    }
                }
            }
        }
        for group in graph.inner.edges.values() {
            for (row, version) in group.versions.iter().enumerate() {
                if version.exists() {
                    counts.edges += 1;
                    counts.edge_type(group.ends(row).2, 1);
                }
            }
        }
        graph.inner.counts = counts;

        let store = RowGroupStore::default();
        *store.inner.write() = graph.inner;
        store
            .next_node_id
            .store(graph_meta.next_node_id, Ordering::Release);
        store
            .next_edge_id
            .store(graph_meta.next_edge_id, Ordering::Release);
        store.epoch.store(meta.epoch, Ordering::Release);
        graphs.push(LoadedGraph {
            id: graph_meta.id,
            name: graph_meta.name.clone(),
            store,
        });
    }
    Ok((meta.next_graph_id, graphs))
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use grafeo_common::storage::{
        ChunkCaps, ChunkKind, ChunkMeta, ChunkNamespace, SectionSink, SectionSource,
    };
    use grafeo_common::types::{NodeId, Value};
    use grafeo_common::utils::error::{Error, Result};

    use super::{
        GraphToWrite, LPG_SECTION_VERSION, decode_meta, encode_meta, read_lpg_section,
        write_lpg_section,
    };
    use crate::codec::column_chunk::encode_column_chunk;
    use crate::graph::Direction;
    use crate::graph::apply::ChangeTarget;
    use crate::graph::conformance::{LABELS, Rng, Tx, committed, counts, people, random_write};
    use crate::graph::rowgroup::RowGroupStore;
    use crate::graph::traits::GraphStore;

    /// A section in memory: what a writer wrote, for a reader, open to
    /// tampering.
    #[derive(Default, Clone)]
    struct Chunks {
        metas: Vec<ChunkMeta>,
        bytes: Vec<Bytes>,
        version: u8,
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
            self.version
        }
    }

    impl Chunks {
        fn remove(&mut self, index: usize) {
            self.metas.remove(index);
            self.bytes.remove(index);
        }

        fn insert(&mut self, index: usize, meta: ChunkMeta, bytes: Vec<u8>) {
            self.metas.insert(index, meta);
            self.bytes.insert(index, Bytes::from(bytes));
        }

        fn position(&self, wanted: impl Fn(&ChunkMeta) -> bool) -> usize {
            self.metas.iter().position(wanted).expect("such a chunk")
        }
    }

    fn write(store: &RowGroupStore, caps: ChunkCaps) -> Chunks {
        let mut chunks = Chunks {
            version: LPG_SECTION_VERSION,
            ..Chunks::default()
        };
        let graphs = [GraphToWrite {
            id: 0,
            name: "",
            store,
        }];
        write_lpg_section(&graphs, 1, caps, &mut chunks).unwrap();
        chunks
    }

    fn read(chunks: &Chunks) -> RowGroupStore {
        let (next_graph_id, mut graphs) = read_lpg_section(chunks).unwrap();
        assert_eq!((next_graph_id, graphs.len()), (1, 1));
        graphs.remove(0).store
    }

    /// Checks that `loaded` holds what `store` had committed: the data
    /// through every access path, the counts, the names, the epoch and ids.
    fn assert_same(loaded: &RowGroupStore, store: &RowGroupStore, what: &str) {
        let (found, expected) = (committed(loaded), committed(store));
        found.consistent(what).unwrap();
        assert_eq!(
            (&found.nodes, &found.edges),
            (&expected.nodes, &expected.edges),
            "{what}"
        );
        assert_eq!(
            counts(loaded, &LABELS),
            counts(store, &LABELS),
            "{what}: the counts"
        );
        assert_eq!(
            loaded.all_labels(),
            store.all_labels(),
            "{what}: the labels"
        );
        assert_eq!(
            loaded.all_edge_types(),
            store.all_edge_types(),
            "{what}: the edge types"
        );
        assert_eq!(
            loaded.current_epoch(),
            store.current_epoch(),
            "{what}: the epoch"
        );
        let next = |store: &RowGroupStore| {
            (
                store.reserve_node_ids(1).unwrap().start,
                store.reserve_edge_ids(1).unwrap().start,
            )
        };
        assert_eq!(next(loaded), next(store), "{what}: the next ids");
    }

    /// A store after a seeded workload of commits and rollbacks.
    fn workload(seed: u64) -> RowGroupStore {
        let store = RowGroupStore::new();
        let mut rng = Rng::new(seed);
        for transaction in 0..14_u64 {
            let mut tx = Tx::begin(&store, 100 + transaction);
            for _ in 0..12 {
                random_write(&mut tx, &mut rng).unwrap();
            }
            if rng.below(4) == 0 {
                tx.roll_back().unwrap();
            } else {
                tx.commit().unwrap();
            }
        }
        store
    }

    /// What a checkpoint wrote reads back as the committed state, at the
    /// default caps and at caps that cut every table into many row groups
    /// and chunks; and the loaded store writes the same chunks again.
    #[test]
    fn a_seeded_workload_round_trips_at_every_cap() {
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
            for seed in 1..=10 {
                let store = workload(seed);
                let chunks = write(&store, caps);
                let loaded = read(&chunks);
                let what = format!("seed {seed}, caps {caps:?}");
                // Before the comparison, which takes an id from each store.
                let again = write(&loaded, caps);
                assert_same(&loaded, &store, &what);
                assert!(
                    again.metas == chunks.metas && again.bytes == chunks.bytes,
                    "{what}: the loaded store writes the same chunks"
                );
            }
        }
    }

    /// A checkpoint writes the committed state: what an open transaction
    /// created is not in it, and what it deleted still is.
    #[test]
    fn an_open_transactions_writes_are_not_in_the_section() {
        let store = RowGroupStore::new();
        let people = people(&store).unwrap();
        let before = committed(&store);
        let mut tx = Tx::begin(&store, 10);
        let jules = tx
            .create_node(&["Person"], &[("name", Value::from("Jules"))])
            .unwrap();
        tx.create_edge(jules, people.alix, "KNOWS", &[]).unwrap();
        tx.delete_edge(people.gus_mia).unwrap();
        tx.delete_node(people.mia).unwrap();
        let loaded = read(&write(&store, ChunkCaps::DEFAULT));
        let found = committed(&loaded);
        found.consistent("the loaded store").unwrap();
        assert_eq!((&found.nodes, &found.edges), (&before.nodes, &before.edges));
    }

    /// A node whose list fills a chunk alone is written in pieces with
    /// their own column ids, and reads back whole and sorted.
    #[test]
    fn a_long_adjacency_list_goes_in_pieces() {
        let store = RowGroupStore::new();
        let mut tx = Tx::begin(&store, 10);
        let hub = tx.create_node(&["City"], &[]).unwrap();
        let others: Vec<NodeId> = (0..40)
            .map(|_| tx.create_node(&["Person"], &[]).unwrap())
            .collect();
        for round in 0..12 {
            for other in &others {
                let edge_type = ["LIVES_IN", "KNOWS"][round % 2];
                tx.create_edge(*other, hub, edge_type, &[]).unwrap();
            }
        }
        tx.commit().unwrap();
        let caps = ChunkCaps {
            max_rows: 65_536,
            max_bytes: 512,
        };
        let chunks = write(&store, caps);
        let pieces: Vec<u32> = chunks
            .metas
            .iter()
            .filter(|meta| {
                meta.namespace == ChunkNamespace::IncomingAdjacency
                    && meta.row_start == hub.as_u64()
            })
            .map(|meta| meta.column_id)
            .collect();
        assert!(
            pieces.len() > 5,
            "the hub's 480 incoming edges take pieces: {pieces:?}"
        );
        let numbered: Vec<u32> = (0..u32::try_from(pieces.len()).unwrap()).collect();
        assert_eq!(pieces, numbered, "numbered from 0");
        let loaded = read(&chunks);
        assert_same(&loaded, &store, "pieces");
        assert_eq!(loaded.edges_from(hub, Direction::Incoming).len(), 480);
        assert_eq!(
            loaded
                .edges_of_type(hub, Direction::Incoming, "KNOWS")
                .len(),
            240
        );
    }

    /// A section whose metadata says adjacency was not written reads with
    /// adjacency built from the edge table.
    #[test]
    fn a_section_without_adjacency_is_rebuilt_from_the_edges() {
        let store = workload(3);
        let mut chunks = write(&store, ChunkCaps::DEFAULT);
        while let Some(index) = chunks
            .metas
            .iter()
            .position(|meta| meta.kind == ChunkKind::Adjacency)
        {
            chunks.remove(index);
        }
        let last = chunks.bytes.len() - 1;
        let mut meta = decode_meta(&chunks.bytes[last]).unwrap();
        meta.graphs[0].adjacency_written = false;
        chunks.bytes[last] = Bytes::from(encode_meta(&meta).unwrap());
        assert_same(&read(&chunks), &store, "rebuilt adjacency");
    }

    /// A loaded store keeps its property columns as the file's chunks until
    /// their first use: a read decodes its own column only, and a write to a
    /// cold column keeps the column's other rows.
    #[test]
    fn loaded_columns_stay_encoded_until_their_first_use() {
        use grafeo_common::change::DataOp;
        use grafeo_common::types::{EpochId, PropertyKey};

        use crate::graph::apply::Writer;

        let (store, [alix, gus, mia, _]) = four_people();
        let loaded = read(&write(&store, ChunkCaps::DEFAULT));
        // The nodes' name, age and city, and the edges' since.
        assert_eq!(loaded.cold_columns(), 4, "nothing is decoded by the load");
        assert_eq!((loaded.node_count(), loaded.edge_count()), (4, 2));
        assert_eq!(loaded.nodes_by_label("Person").len(), 4);
        assert_eq!(loaded.edges_from(gus, Direction::Outgoing).len(), 1);
        assert_eq!(
            loaded.cold_columns(),
            4,
            "counts, labels and adjacency read no property column"
        );
        let cold_bytes = loaded.heap_bytes();

        let name = PropertyKey::new("name");
        assert_eq!(
            loaded.get_node_property(mia, &name),
            Some(Value::from("Mia"))
        );
        assert_eq!(
            loaded.cold_columns(),
            3,
            "the read decoded the name column alone"
        );

        // A write to the cold age column: Alix's age changes, Gus keeps his.
        let op = DataOp::SetNodeProperty {
            id: alix,
            key: PropertyKey::new("age"),
            value: Value::Int64(88),
        };
        let epoch = EpochId::new(loaded.current_epoch().as_u64() + 1);
        loaded
            .apply(
                &op,
                Writer::Immediate {
                    epoch,
                    before_images: false,
                },
            )
            .unwrap();
        assert_eq!(loaded.cold_columns(), 2);
        let age = PropertyKey::new("age");
        assert_eq!(loaded.get_node_property(alix, &age), Some(Value::Int64(88)));
        assert_eq!(loaded.get_node_property(gus, &age), Some(Value::Int64(3)));

        // A whole node reads every column of its group.
        assert!(loaded.get_node(alix).is_some());
        assert!(
            loaded
                .get_edge(grafeo_common::types::EdgeId::new(0))
                .is_some()
        );
        assert_eq!(loaded.cold_columns(), 0);
        assert!(
            loaded.heap_bytes() != cold_bytes,
            "the decoded columns are counted"
        );
    }

    /// A bitmap chunk of `rows` (ids) in `namespace`, column `column`.
    fn bitmap(namespace: ChunkNamespace, column: u32, rows: &[u64]) -> (ChunkMeta, Vec<u8>) {
        let first = rows[0];
        let values: Vec<(u32, Value)> = rows
            .iter()
            .map(|row| (u32::try_from(row - first).unwrap(), Value::Bool(true)))
            .collect();
        let row_count = u32::try_from(rows[rows.len() - 1] - first + 1).unwrap();
        let (codec, bytes) = encode_column_chunk(row_count, &values, None).unwrap();
        (
            ChunkMeta::column(0, column, first, row_count, codec.to_byte()).in_namespace(namespace),
            bytes,
        )
    }

    /// Alix knows Gus, who knows Mia; Vincent stands alone.
    fn four_people() -> (RowGroupStore, [NodeId; 4]) {
        let store = RowGroupStore::new();
        let people = people(&store).unwrap();
        let mut tx = Tx::begin(&store, 10);
        let vincent = tx
            .create_node(&["Person"], &[("name", Value::from("Vincent"))])
            .unwrap();
        tx.commit().unwrap();
        (store, [people.alix, people.gus, people.mia, vincent])
    }

    /// The first chunk of the edge table: delete chunks of nodes go before it.
    fn first_edge_chunk(chunks: &Chunks) -> usize {
        chunks.position(|meta| meta.namespace == ChunkNamespace::EdgeStructure)
    }

    /// A reader applies delete chunks: the rows they name are no nodes or
    /// edges, with no adjacency entry left.
    #[test]
    fn delete_chunks_take_their_rows_out() {
        let (store, [alix, gus, _, vincent]) = four_people();
        let mut chunks = write(&store, ChunkCaps::DEFAULT);
        let at = first_edge_chunk(&chunks);
        let (meta, bytes) = bitmap(ChunkNamespace::NodeDeletes, 0, &[vincent.as_u64()]);
        chunks.insert(at, meta, bytes);
        // The edge from Alix to Gus (edge 0) goes too.
        let end = chunks.metas.len() - 1;
        let (meta, bytes) = bitmap(ChunkNamespace::EdgeDeletes, 0, &[0]);
        chunks.insert(end, meta, bytes);
        let loaded = read(&chunks);
        let found = committed(&loaded);
        found.consistent("after the delete chunks").unwrap();
        assert!(!found.nodes.contains_key(&vincent.as_u64()) && found.nodes.len() == 3);
        assert_eq!(found.edges.keys().copied().collect::<Vec<_>>(), [1]);
        assert_eq!(loaded.edges_from(alix, Direction::Outgoing), Vec::new());
        assert_eq!(loaded.edges_from(gus, Direction::Incoming), Vec::new());
        assert_eq!((loaded.node_count(), loaded.edge_count()), (3, 1));
    }

    /// Every section a writer does not write is refused as corruption,
    /// naming what is wrong; a section of another version as such.
    #[test]
    fn tampered_sections_are_refused_as_corruption() {
        let (store, [alix, gus, _, _]) = four_people();
        let good = write(&store, ChunkCaps::DEFAULT);
        assert_same(&read(&good), &store, "the untampered section");
        let first = |namespace: ChunkNamespace| good.position(|meta| meta.namespace == namespace);
        let without = |index: usize| {
            let mut chunks = good.clone();
            chunks.remove(index);
            chunks
        };
        let with_meta = |change: fn(&mut super::Meta)| {
            let mut chunks = good.clone();
            let last = chunks.bytes.len() - 1;
            let mut meta = decode_meta(&chunks.bytes[last]).unwrap();
            change(&mut meta);
            chunks.bytes[last] = Bytes::from(encode_meta(&meta).unwrap());
            chunks
        };
        let with_chunk = |at: usize, (meta, bytes): (ChunkMeta, Vec<u8>)| {
            let mut chunks = good.clone();
            chunks.insert(at, meta, bytes);
            chunks
        };
        let label = first(ChunkNamespace::NodeLabels);
        let edges = first_edge_chunk(&good);
        let cases: Vec<(&str, Chunks, &str)> = vec![
            (
                "the node existence chunk missing",
                without(first(ChunkNamespace::NodeStructure)),
                "is no node of its group",
            ),
            (
                "an outgoing adjacency chunk missing",
                without(first(ChunkNamespace::OutgoingAdjacency)),
                "adjacency lists 0 edges in direction 0",
            ),
            (
                "a label chunk twice",
                with_chunk(label + 1, (good.metas[label], good.bytes[label].to_vec())),
                "overlap",
            ),
            (
                "a label the dictionary does not hold",
                {
                    let mut chunks = good.clone();
                    chunks.metas[label].column_id = 999;
                    chunks
                },
                "label 999 is not in the graph's dictionary",
            ),
            (
                "an edge target without its source",
                without(edges),
                "comes without column 1",
            ),
            (
                "the edge table before the node table",
                {
                    let mut chunks = good.clone();
                    for step in 0..3 {
                        let meta = chunks.metas.remove(edges + step);
                        let bytes = chunks.bytes.remove(edges + step);
                        chunks.insert(step, meta, bytes.to_vec());
                    }
                    chunks
                },
                "are not both nodes",
            ),
            (
                "a delete chunk of a node with an edge",
                with_chunk(
                    edges,
                    bitmap(ChunkNamespace::NodeDeletes, 0, &[gus.as_u64()]),
                ),
                "is deleted, an edge of it is not",
            ),
            (
                "a delete chunk past the table",
                with_chunk(edges, bitmap(ChunkNamespace::NodeDeletes, 0, &[3_000])),
                "past the table's next id",
            ),
            (
                "a column chunk in an adjacency namespace",
                with_chunk(
                    edges,
                    bitmap(ChunkNamespace::IncomingAdjacency, 0, &[alix.as_u64()]),
                ),
                "a chunk kind its namespace does not hold",
            ),
            (
                "adjacency chunks in a graph that wrote none",
                with_meta(|meta| meta.graphs[0].adjacency_written = false),
                "says none was written",
            ),
            (
                "a node count the chunks do not hold",
                with_meta(|meta| meta.graphs[0].node_groups[0].1 += 1),
                "where the metadata counts",
            ),
            (
                "an edge group the chunks do not hold",
                with_meta(|meta| meta.graphs[0].edge_groups.push((9, 1))),
                "where the metadata counts",
            ),
            (
                "rows past the next node id",
                with_meta(|meta| meta.graphs[0].next_node_id = 2),
                "past the table's next id",
            ),
            (
                "no metadata chunk",
                without(good.metas.len() - 1),
                "metadata chunk",
            ),
        ];
        for (case, chunks, expected) in cases {
            let error = read_lpg_section(&chunks)
                .err()
                .unwrap_or_else(|| panic!("{case}: the reader took it"));
            assert!(
                matches!(error, Error::Corruption(_)) && error.to_string().contains(expected),
                "{case}: {error}"
            );
        }

        let mut older = good.clone();
        older.version = 3;
        let error = read_lpg_section(&older).unwrap_err();
        assert!(
            matches!(error, Error::Serialization(_))
                && error.to_string().contains("another version"),
            "{error}"
        );
    }

    /// Adjacency that disagrees with the edge table is refused: an entry
    /// naming another end, and a piece that does not continue its list.
    #[test]
    fn adjacency_that_disagrees_with_the_edges_is_refused() {
        use crate::graph::rowgroup::adjacency::Adjacent;
        use crate::graph::rowgroup::adjacency_chunk;

        let (store, [alix, _, mia, _]) = four_people();
        let good = write(&store, ChunkCaps::DEFAULT);
        let at = good.position(|meta| meta.namespace == ChunkNamespace::OutgoingAdjacency);
        // Alix's edge 0 to Gus, said to end at Mia.
        let to_mia = |edge: u64| {
            vec![Adjacent {
                edge_type: 0,
                other: mia.as_u64(),
                edge,
            }]
        };
        let lists = [to_mia(0), to_mia(1)];
        let borrowed: Vec<&[Adjacent]> = lists.iter().map(Vec::as_slice).collect();
        let mut chunks = good.clone();
        chunks.bytes[at] = Bytes::from(adjacency_chunk::encode(&borrowed, 0, 65_536).unwrap());
        let error = read_lpg_section(&chunks).unwrap_err();
        let expected = format!("node {}'s adjacency names edge 0", alix.as_u64());
        assert!(
            matches!(error, Error::Corruption(_)) && error.to_string().contains(&expected),
            "{error}"
        );

        let mut chunks = good.clone();
        let mut piece = good.metas[at];
        piece.column_id = 1;
        chunks.insert(at + 1, piece, good.bytes[at].to_vec());
        let error = read_lpg_section(&chunks).unwrap_err();
        assert!(
            matches!(error, Error::Corruption(_))
                && error.to_string().contains("does not continue the list"),
            "{error}"
        );
    }
}
