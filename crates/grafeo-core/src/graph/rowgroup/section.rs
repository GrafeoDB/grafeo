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
//! - the adjacency chunks, outgoing then incoming (`adjacency_chunk`); a
//!   store that keeps no incoming edges (`backward_edges` off) writes the
//!   outgoing ones only.
//!
//! In an edge row group: the source, target and type columns as three chunks
//! of one range in a row, then the property columns. A delete chunk (node or
//! edge delete namespace, column 0) names rows deleted since their group was
//! written; no writer writes one before chunk reuse (H3), every reader
//! applies them.
//!
//! A checkpoint writes the committed state at an epoch E: the rows, values
//! and labels a reader at E sees, read in place through the update chains,
//! so nothing a transaction still open wrote and nothing committed after E
//! is in it. A null value is not written. The metadata chunk (layout 2)
//! holds the caps, the epoch E, and per graph its name, next ids,
//! dictionaries, which adjacency was written (none, both directions, or the
//! outgoing one only) and each row group's node and edge counts, which the
//! reader checks. A reader builds the adjacency its store keeps: what the
//! file lacks from the edge table, and the incoming lists of a file written
//! with them are dropped by a store that keeps none (after the reader
//! checked them).
//!
//! A database that keeps history (a retention window, H2c) checkpoints its
//! history from an epoch H before E too: every row a reader at H or later
//! sees (a row a commit in the window deleted included, with the state a
//! reader saw last, and its edges in the adjacency), and per row group the
//! `History` chunks of the versions a reader in the window tells apart,
//! after the group's other chunks:
//!
//! - node structure, column 0, and edge structure, column 0: per row whose
//!   create or delete a reader in the window sees differently, `[created,
//!   deleted]` (`created` 0 for a row every reader there sees created,
//!   `deleted` null for one not deleted by E);
//! - node structure, column 1: per row whose labels a commit in the window
//!   changed, the `[epoch, label ids]` of each such commit, oldest first:
//!   the labels the row had before it;
//! - a property column (node or edge): per row, the `[epoch, value]` of
//!   each commit in the window that wrote the key, oldest first: the value
//!   the row held before it, null for none.
//!
//! A reader rebuilds the versions and the update chains from them, so a
//! reader at any epoch from H to E reads what it read before the
//! checkpoint. Without history (H = E) nothing of this is written.
//!
//! A reader refuses everything a writer does not write with
//! `Error::Corruption`, and a section of another version as such.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bytes::Bytes;
use grafeo_common::storage::value_codec::{MAX_PROPERTY_VALUE_DEPTH, nests_too_deep};
use grafeo_common::storage::{
    ChunkCaps, ChunkKind, ChunkMeta, ChunkNamespace, Section, SectionSink, SectionSource,
    SectionType, legacy_bytes,
};
use grafeo_common::types::{EpochId, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_common::utils::hash::FxHashMap;

use super::adjacency::{Adjacency, Adjacent};
use super::adjacency_chunk;
use super::{
    ABSENT, IncomingEdges, Inner, PENDING, ROWS_PER_GROUP, Read, RowGroupStore, RowVersion, locate,
};
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
/// The `History` columns of the node and edge structure: the rows'
/// versions, and (nodes only) the label sets label writes replaced.
const HISTORY_VERSIONS: u32 = 0;
const HISTORY_LABELS: u32 = 1;

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

/// Which adjacency a graph's chunks hold: the metadata's adjacency byte.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum WrittenAdjacency {
    /// None (byte 0): a reader builds it from the edge table.
    None,
    /// Both directions (byte 1).
    #[default]
    Both,
    /// The outgoing direction only (byte 2): the store kept no incoming
    /// edges (`backward_edges` off).
    Outgoing,
}

impl WrittenAdjacency {
    fn to_byte(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Both => 1,
            Self::Outgoing => 2,
        }
    }

    fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::None),
            1 => Some(Self::Both),
            2 => Some(Self::Outgoing),
            _ => None,
        }
    }

    /// Whether the chunks hold the direction (`outgoing`, or incoming).
    fn holds(self, outgoing: bool) -> bool {
        match self {
            Self::None => false,
            Self::Both => true,
            Self::Outgoing => outgoing,
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
    adjacency_written: WrittenAdjacency,
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
        out.push(graph.adjacency_written.to_byte());
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
    // A graph takes at least 57 bytes: its id, an empty name, the next
    // node and edge ids, three empty dictionaries, the adjacency flag and
    // two empty row group lists (an empty root graph is that small).
    let count = reader.count(57, "graphs")?;
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
        let flag = reader.take::<1>("the adjacency flag")?[0];
        let Some(adjacency_written) = WrittenAdjacency::from_byte(flag) else {
            return Err(reader.refuse(format!("adjacency flag {flag}")));
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

fn history_column(namespace: ChunkNamespace, column_id: u32) -> ChunkColumn {
    ChunkColumn {
        kind: ChunkKind::History,
        namespace,
        column_id,
    }
}

/// Whether `stamp` is a commit epoch (not absent, not a transaction's).
fn committed(stamp: u64) -> bool {
    stamp != ABSENT && stamp & PENDING == 0
}

/// An epoch as the `Int64` a history cell holds; every commit epoch is
/// below [`PENDING`], so it fits.
fn epoch_cell(epoch: u64) -> Value {
    Value::Int64(i64::try_from(epoch).expect("a commit epoch is below 2^63"))
}

/// A history cell: its entries, each a list.
fn list(items: Vec<Value>) -> Value {
    Value::List(Arc::from(items))
}

/// What a checkpoint writes of a graph's rows: the committed state at `at`,
/// and the versions a reader at `from` (at most `at`) or later tells apart.
#[derive(Debug, Clone, Copy)]
struct Window {
    at: u64,
    from: u64,
}

impl Window {
    /// Whether the window holds history: the versions between `from` and
    /// `at`.
    fn has_history(self) -> bool {
        self.from < self.at
    }

    /// The epoch whose state the row of `version` is written in: `at`, or
    /// the last epoch before a delete in the window; `None` for a row no
    /// reader in the window sees.
    fn last(self, version: RowVersion) -> Option<u64> {
        let created = version.created;
        if !committed(created) || created > self.at {
            return None;
        }
        let deleted = version.deleted;
        if committed(deleted) && deleted <= self.at {
            if deleted <= self.from.max(created) {
                return None;
            }
            return Some(deleted - 1);
        }
        Some(self.at)
    }

    /// The `[created, deleted]` history cell of a written row, `None` when
    /// every reader in the window sees the row alike.
    fn versions_cell(self, version: RowVersion) -> Option<Value> {
        let created = if version.created > self.from {
            version.created
        } else {
            0
        };
        let deleted =
            (committed(version.deleted) && version.deleted <= self.at).then_some(version.deleted);
        if created == 0 && deleted.is_none() {
            return None;
        }
        Some(list(vec![
            epoch_cell(created),
            deleted.map_or(Value::Null, epoch_cell),
        ]))
    }

    /// The history cell of what a chain kept of a row whose state is
    /// written at `last`: the `[epoch, kept]` of each commit in the window
    /// up to `last`, oldest first; `None` when there is none.
    fn chain_cell<'k, T: 'k>(
        self,
        last: u64,
        kept: impl Iterator<Item = (u64, &'k T)>,
        cell: impl Fn(&T) -> Value,
    ) -> Option<Value> {
        let entries: Vec<Value> = kept
            .filter(|(stamp, _)| committed(*stamp) && *stamp > self.from && *stamp <= last)
            .map(|(stamp, kept)| list(vec![epoch_cell(stamp), cell(kept)]))
            .collect();
        (!entries.is_empty()).then(|| list(entries))
    }
}

/// The history cells of one node row: its id, versions, labels and values
/// by key.
type NodeHistoryCells = (u64, Option<Value>, Option<Value>, FxHashMap<u32, Value>);

/// A kept value as a history cell holds it: none as null (the store holds
/// no null, #412 decision Q8).
fn value_cell(kept: &Option<Value>) -> Value {
    kept.clone().unwrap_or(Value::Null)
}

/// A kept label set as a history cell holds it: its label ids.
fn labels_cell(kept: &[u32]) -> Value {
    list(kept.iter().map(|id| Value::Int64(i64::from(*id))).collect())
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
    /// A node's edges in one direction that the window writes, sorted.
    fn written_adjacency(&self, id: u64, outgoing: bool, window: Window) -> Vec<Adjacent> {
        let Some((group, row)) = self.node(id) else {
            return Vec::new();
        };
        let mut list: Vec<Adjacent> = group
            .adjacency(outgoing)
            .of(row)
            .filter(|adjacent| {
                self.edge(adjacent.edge)
                    .is_some_and(|(group, row)| window.last(group.version(row)).is_some())
            })
            .collect();
        list.sort_unstable();
        list
    }

    /// Writes the `History` chunks of one file row group of the node table:
    /// each written row's versions, label and value chains in the window.
    fn write_node_history(
        &self,
        graph_id: u32,
        rows: &[(u64, u64)],
        group_start: u64,
        window: Window,
        caps: ChunkCaps,
        sink: &mut dyn SectionSink,
    ) -> Result<()> {
        let mut keys: BTreeSet<u32> = BTreeSet::new();
        let mut cells: Vec<NodeHistoryCells> = Vec::new();
        for &(id, last) in rows {
            let (group, row) = self.node(id).expect("a row it listed");
            let versions = window.versions_cell(group.version(row));
            let labels = if group.label_updates.is_empty() {
                None
            } else {
                window.chain_cell(last, group.label_updates.kept_stamped(row), |kept| {
                    labels_cell(kept)
                })
            };
            let values: FxHashMap<u32, Value> = group
                .columns
                .keys()
                .filter_map(|key| {
                    Some((
                        key,
                        window.chain_cell(last, group.columns.kept(key, row), value_cell)?,
                    ))
                })
                .collect();
            if versions.is_some() || labels.is_some() || !values.is_empty() {
                keys.extend(values.keys());
                cells.push((id, versions, labels, values));
            }
        }
        if cells.is_empty() {
            return Ok(());
        }
        let columns: Vec<ChunkColumn> = [HISTORY_VERSIONS, HISTORY_LABELS]
            .into_iter()
            .map(|id| history_column(ChunkNamespace::NodeStructure, id))
            .chain(
                keys.iter()
                    .map(|key| history_column(ChunkNamespace::NodeProperties, *key)),
            )
            .collect();
        let mut chunker = RowsChunker::new(graph_id, columns, group_start, caps);
        for (id, versions, labels, mut values) in cells {
            let row: Vec<Option<(Value, u64)>> = [versions, labels]
                .into_iter()
                .map(|cell| cell.map(|cell| (cell, 0)))
                .chain(
                    keys.iter()
                        .map(|key| values.remove(key).map(|cell| (cell, 0))),
                )
                .collect();
            chunker.push(sink, id, row)?;
        }
        chunker.finish(sink)
    }

    /// Writes the `History` chunks of one file row group of the edge table:
    /// each written row's versions and value chains in the window.
    fn write_edge_history(
        &self,
        graph_id: u32,
        rows: &[(u64, u64)],
        group_start: u64,
        window: Window,
        caps: ChunkCaps,
        sink: &mut dyn SectionSink,
    ) -> Result<()> {
        let mut keys: BTreeSet<u32> = BTreeSet::new();
        let mut cells: Vec<(u64, Option<Value>, FxHashMap<u32, Value>)> = Vec::new();
        for &(id, last) in rows {
            let (group, row) = self.edge(id).expect("a row it listed");
            let versions = window.versions_cell(group.version(row));
            let values: FxHashMap<u32, Value> = group
                .columns
                .keys()
                .filter_map(|key| {
                    Some((
                        key,
                        window.chain_cell(last, group.columns.kept(key, row), value_cell)?,
                    ))
                })
                .collect();
            if versions.is_some() || !values.is_empty() {
                keys.extend(values.keys());
                cells.push((id, versions, values));
            }
        }
        if cells.is_empty() {
            return Ok(());
        }
        let columns: Vec<ChunkColumn> = std::iter::once(history_column(
            ChunkNamespace::EdgeStructure,
            HISTORY_VERSIONS,
        ))
        .chain(
            keys.iter()
                .map(|key| history_column(ChunkNamespace::EdgeProperties, *key)),
        )
        .collect();
        let mut chunker = RowsChunker::new(graph_id, columns, group_start, caps);
        for (id, versions, mut values) in cells {
            let row: Vec<Option<(Value, u64)>> = std::iter::once(versions.map(|cell| (cell, 0)))
                .chain(
                    keys.iter()
                        .map(|key| values.remove(key).map(|cell| (cell, 0))),
                )
                .collect();
            chunker.push(sink, id, row)?;
        }
        chunker.finish(sink)
    }

    /// Refuses a value a file cannot hold: one that nests lists, maps and
    /// paths deeper than [`MAX_PROPERTY_VALUE_DEPTH`], which no write takes
    /// but a 0.5.x database can hold (the migration then fails, naming it).
    fn refuse_too_deep(
        &self,
        graph: &str,
        entity: &str,
        id: u64,
        values: &FxHashMap<u32, Value>,
    ) -> Result<()> {
        let Some(key) = values
            .iter()
            .find(|(_, value)| nests_too_deep(value))
            .map(|(key, _)| *key)
        else {
            return Ok(());
        };
        let graph = if graph.is_empty() {
            "the default graph".to_string()
        } else {
            format!("graph {graph}")
        };
        let key = self.keys.get_name(key).map_or("?", |name| name.as_str());
        Err(Error::Serialization(format!(
            "{graph}, {entity} {id}, property {key:?}: the value nests lists, maps and paths \
             more than {MAX_PROPERTY_VALUE_DEPTH} levels deep, deeper than a database file can \
             hold"
        )))
    }

    /// Writes one graph's tables as the window holds them, and returns its
    /// metadata.
    #[expect(
        clippy::too_many_lines,
        reason = "the node table, then the edge table, each a file row group at a time"
    )]
    fn write_graph(
        &self,
        graph: &GraphToWrite<'_>,
        window: Window,
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
            adjacency_written: if self.keeps_incoming() {
                WrittenAdjacency::Both
            } else {
                WrittenAdjacency::Outgoing
            },
            ..GraphMeta::default()
        };

        // The node table, a file row group at a time: each row with the
        // epoch its state is written at.
        let mut nodes: Vec<(u64, u64)> = Vec::new();
        for (index, group) in &self.nodes {
            for (row, version) in group.versions.rows() {
                if let Some(last) = window.last(version) {
                    nodes.push((index * ROWS_PER_GROUP + row as u64, last));
                }
            }
        }
        for written in nodes.chunk_by(|a, b| a.0 / max_rows == b.0 / max_rows) {
            let file_group = written[0].0 / max_rows;
            let group_start = file_group * max_rows;
            let rows: Vec<u64> = written.iter().map(|(id, _)| *id).collect();
            let mut labels: BTreeSet<u32> = BTreeSet::new();
            let mut keys: BTreeSet<u32> = BTreeSet::new();
            let mut cells: Vec<(u64, Vec<u32>, FxHashMap<u32, Value>)> =
                Vec::with_capacity(rows.len());
            for &(id, last) in written {
                let (group, row) = self.node(id).expect("a row it listed");
                let read = Read::At(last);
                let row_labels = group.labels_at(row, read);
                let values: FxHashMap<u32, Value> = group
                    .columns
                    .row_at(row, read)
                    .filter(|(_, value)| !value.is_null())
                    .collect();
                self.refuse_too_deep(graph.name, "node", id, &values)?;
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
            if window.has_history() {
                self.write_node_history(graph.id, written, group_start, window, caps, sink)?;
            }
            for (namespace, outgoing) in [
                (ChunkNamespace::OutgoingAdjacency, true),
                (ChunkNamespace::IncomingAdjacency, false),
            ] {
                if !meta.adjacency_written.holds(outgoing) {
                    continue;
                }
                let lists: Vec<(u32, Vec<Adjacent>)> = rows
                    .iter()
                    .map(|id| {
                        (
                            u32::try_from(id - group_start).expect("a row below the row cap"),
                            self.written_adjacency(*id, outgoing, window),
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
        let mut edges: Vec<(u64, u64)> = Vec::new();
        for (index, group) in &self.edges {
            for (row, version) in group.versions.rows() {
                if let Some(last) = window.last(version) {
                    edges.push((index * ROWS_PER_GROUP + row as u64, last));
                }
            }
        }
        for written in edges.chunk_by(|a, b| a.0 / max_rows == b.0 / max_rows) {
            let file_group = written[0].0 / max_rows;
            let mut keys: BTreeSet<u32> = BTreeSet::new();
            let mut cells: Vec<(u64, (u64, u64, u32), FxHashMap<u32, Value>)> =
                Vec::with_capacity(written.len());
            for &(id, last) in written {
                let (group, row) = self.edge(id).expect("a row it listed");
                let values: FxHashMap<u32, Value> = group
                    .columns
                    .row_at(row, Read::At(last))
                    .filter(|(_, value)| !value.is_null())
                    .collect();
                self.refuse_too_deep(graph.name, "edge", id, &values)?;
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
            if window.has_history() {
                self.write_edge_history(
                    graph.id,
                    written,
                    file_group * max_rows,
                    window,
                    caps,
                    sink,
                )?;
            }
            meta.edge_groups.push((
                file_group,
                u32::try_from(written.len()).expect("rows of one group"),
            ));
        }
        Ok(meta)
    }
}

/// Writes the `LPG_STORE` section, version 4, of `graphs` (ids ascending):
/// each graph's committed state at `epoch`, as a reader at it sees the
/// rows, their values and their labels. What a transaction still open wrote
/// is not in it, nor what a commit after `epoch` wrote; the section's epoch
/// is `epoch`.
///
/// `epoch` is at or after the epoch the graphs were last collected at (see
/// [`RowGroupStore::collect`]): a collect drops the versions a reader before
/// its horizon would read.
///
/// # Errors
///
/// Returns [`Error::InvalidValue`] for caps the format refuses,
/// [`Error::Internal`] for graphs out of id order or an id at or above
/// `next_graph_id`, and the encoder's or the sink's errors.
pub fn write_lpg_section(
    graphs: &[GraphToWrite<'_>],
    next_graph_id: u32,
    epoch: EpochId,
    caps: ChunkCaps,
    sink: &mut dyn SectionSink,
) -> Result<()> {
    write_lpg_section_with_history(graphs, next_graph_id, epoch, None, caps, sink)
}

/// Writes the `LPG_STORE` section as [`write_lpg_section`] does, and with
/// `history_from` (an epoch H before `epoch`) the history a reader at H or
/// later reads too: the rows such a reader sees, deleted ones included, and
/// the `History` chunks of the versions it tells apart (see the module
/// docs). `None`, or an epoch at or after `epoch`, writes no history.
///
/// H is at or after the horizon the graphs were last collected at: the
/// versions before it are gone.
///
/// # Errors
///
/// As [`write_lpg_section`].
pub fn write_lpg_section_with_history(
    graphs: &[GraphToWrite<'_>],
    next_graph_id: u32,
    epoch: EpochId,
    history_from: Option<EpochId>,
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
        epoch: epoch.as_u64(),
        next_graph_id,
        graphs: Vec::with_capacity(graphs.len()),
    };
    let window = Window {
        at: epoch.as_u64(),
        from: history_from.map_or(epoch, |from| from.min(epoch)).as_u64(),
    };
    for graph in graphs {
        let inner = graph.store.inner.read();
        meta.graphs
            .push(inner.write_graph(graph, window, caps, sink)?);
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
    /// What the `History` chunks hold, applied once every chunk is read.
    history: LoadedHistory,
}

/// What a section's `History` chunks hold of one graph.
#[derive(Debug, Default)]
struct LoadedHistory {
    /// Per node row and per edge row: who created it (0: every reader in
    /// the window) and who deleted it ([`ABSENT`]: no one by the epoch).
    node_versions: Vec<(u64, u64, u64)>,
    edge_versions: Vec<(u64, u64, u64)>,
    /// Per node row and label write: its epoch and the label ids before.
    labels: Vec<(u64, u64, Box<[u32]>)>,
    /// Per row, key and value write: the key, the row, the write's epoch and
    /// the value before.
    node_values: Vec<(u32, u64, u64, Option<Value>)>,
    edge_values: Vec<(u32, u64, u64, Option<Value>)>,
}

impl LoadedHistory {
    fn is_empty(&self) -> bool {
        self.node_versions.is_empty()
            && self.edge_versions.is_empty()
            && self.labels.is_empty()
            && self.node_values.is_empty()
            && self.edge_values.is_empty()
    }
}

/// An epoch a history cell holds, at most `epoch` (the section's).
fn cell_epoch(value: &Value, epoch: u64) -> std::result::Result<u64, String> {
    match value {
        Value::Int64(at) => u64::try_from(*at)
            .ok()
            .filter(|at| *at <= epoch)
            .ok_or_else(|| format!("epoch {at}, past the section's epoch {epoch}")),
        other => Err(format!("{other:?} where an epoch is")),
    }
}

/// The `[created, deleted]` of a versions cell, `deleted` [`ABSENT`] for
/// null: created at most the section's epoch `epoch`, deleted after it and
/// at most `epoch`, not both "every reader".
fn versions_of(cell: &Value, epoch: u64) -> std::result::Result<(u64, u64), String> {
    let Value::List(pair) = cell else {
        return Err(format!("{cell:?} where a [created, deleted] pair is"));
    };
    let [created, deleted] = &pair[..] else {
        return Err(format!("{cell:?} where a [created, deleted] pair is"));
    };
    let created = cell_epoch(created, epoch)?;
    let deleted = match deleted {
        Value::Null => ABSENT,
        deleted => cell_epoch(deleted, epoch)?,
    };
    if deleted != ABSENT && deleted <= created {
        return Err(format!(
            "deleted at {deleted}, not after its create at {created}"
        ));
    }
    if created == 0 && deleted == ABSENT {
        return Err("a version every reader sees alike, which is not written".to_string());
    }
    Ok((created, deleted))
}

/// The `[epoch, kept]` entries of a chain cell: at least one, epochs from 1
/// to the section's epoch `epoch`, ascending.
fn chain_of(cell: &Value, epoch: u64) -> std::result::Result<Vec<(u64, Value)>, String> {
    let shape = || format!("{cell:?} where a list of [epoch, kept] pairs is");
    let Value::List(entries) = cell else {
        return Err(shape());
    };
    if entries.is_empty() {
        return Err(shape());
    }
    let mut chain: Vec<(u64, Value)> = Vec::with_capacity(entries.len());
    for entry in entries.iter() {
        let Value::List(pair) = entry else {
            return Err(shape());
        };
        let [at, kept] = &pair[..] else {
            return Err(shape());
        };
        let at = cell_epoch(at, epoch)?;
        if at == 0 || chain.last().is_some_and(|(last, _)| *last >= at) {
            return Err(format!(
                "history epochs that are not ascending from 1: {cell:?}"
            ));
        }
        chain.push((at, kept.clone()));
    }
    Ok(chain)
}

impl Loading {
    /// Takes in the cells of a `History` chunk, after the rows they name
    /// are read; the section's epoch is `epoch`.
    fn read_history(
        &mut self,
        chunk: &ChunkMeta,
        bytes: &Bytes,
        epoch: u64,
    ) -> std::result::Result<(), String> {
        enum Kind {
            Versions(bool),
            Labels,
            Values(bool),
        }
        let kind = match (chunk.namespace, chunk.column_id) {
            (ChunkNamespace::NodeStructure, HISTORY_VERSIONS) => Kind::Versions(true),
            (ChunkNamespace::EdgeStructure, HISTORY_VERSIONS) => Kind::Versions(false),
            (ChunkNamespace::NodeStructure, HISTORY_LABELS) => Kind::Labels,
            (ChunkNamespace::NodeProperties, key) | (ChunkNamespace::EdgeProperties, key) => {
                if self.inner.keys.get_name(key).is_none() {
                    return Err(format!("column {key} is no property key of the graph"));
                }
                Kind::Values(chunk.namespace == ChunkNamespace::NodeProperties)
            }
            _ => {
                return Err("a history chunk of a column that has no history".to_string());
            }
        };
        let decoded = decode_column_chunk_bytes(bytes, chunk.codec, chunk.row_count)
            .map_err(|error| error.to_string())?;
        if decoded.epochs.is_some() {
            return Err("epochs, which a history chunk never carries".to_string());
        }
        for (offset, cell) in decoded.values {
            let id = chunk.row_start + u64::from(offset);
            let node = match kind {
                Kind::Versions(node) | Kind::Values(node) => node,
                Kind::Labels => true,
            };
            let exists = if node {
                self.inner.node(id).is_some()
            } else {
                self.inner.edge(id).is_some()
            };
            if !exists {
                return Err(format!("row {id} is no node or edge of its group"));
            }
            match kind {
                Kind::Versions(node) => {
                    let (created, deleted) = versions_of(&cell, epoch)?;
                    let versions = if node {
                        &mut self.history.node_versions
                    } else {
                        &mut self.history.edge_versions
                    };
                    versions.push((id, created, deleted));
                }
                Kind::Labels => {
                    for (at, kept) in chain_of(&cell, epoch)? {
                        let ids = label_ids_of(&kept, &self.inner.labels)?;
                        self.history.labels.push((id, at, ids));
                    }
                }
                Kind::Values(node) => {
                    let values = if node {
                        &mut self.history.node_values
                    } else {
                        &mut self.history.edge_values
                    };
                    for (at, kept) in chain_of(&cell, epoch)? {
                        let kept = (!kept.is_null()).then_some(kept);
                        values.push((chunk.column_id, id, at, kept));
                    }
                }
            }
        }
        Ok(())
    }
}

impl LoadedHistory {
    /// Rebuilds in `inner` the versions and update chains the `History`
    /// chunks held, once every row is in, and lists the vectors they are in
    /// for `collect`.
    fn apply(self, inner: &mut Inner) -> std::result::Result<(), String> {
        let history = self;
        if history.is_empty() {
            return Ok(());
        }
        for (id, created, deleted) in history.node_versions {
            let (group, row) = inner
                .node_mut(id)
                .ok_or_else(|| format!("row {id} is no node of its group"))?;
            restore_version(&mut group.versions, row, created, deleted);
        }
        for (id, created, deleted) in history.edge_versions {
            let (group, row) = inner
                .edge_mut(id)
                .ok_or_else(|| format!("row {id} is no edge of its group"))?;
            restore_version(&mut group.versions, row, created, deleted);
        }
        let mut labels: BTreeMap<u64, Vec<(usize, u64, Box<[u32]>)>> = BTreeMap::new();
        for (id, at, kept) in history.labels {
            let (group, row) = locate(id);
            labels.entry(group).or_default().push((row, at, kept));
        }
        for (index, kept) in labels {
            inner
                .nodes
                .get_mut(&index)
                .ok_or_else(|| format!("node row group {index} holds no rows"))?
                .label_updates
                .load(kept);
        }
        for (node, values) in [(true, history.node_values), (false, history.edge_values)] {
            let mut by_column: BTreeMap<(u64, u32), Vec<(usize, u64, Option<Value>)>> =
                BTreeMap::new();
            for (key, id, at, kept) in values {
                let (group, row) = locate(id);
                by_column
                    .entry((group, key))
                    .or_default()
                    .push((row, at, kept));
            }
            for ((index, key), kept) in by_column {
                let columns = if node {
                    inner.nodes.get_mut(&index).map(|group| &mut group.columns)
                } else {
                    inner.edges.get_mut(&index).map(|group| &mut group.columns)
                };
                columns
                    .ok_or_else(|| format!("row group {index} holds no rows"))?
                    .load_kept(key, kept);
            }
        }
        inner.relist();
        Ok(())
    }
}

/// Gives a loaded row the versions its history cell holds.
fn restore_version(
    versions: &mut super::versions::Versions,
    row: usize,
    created: u64,
    deleted: u64,
) {
    if created != 0 {
        versions.set_created(row, created);
    }
    if deleted != ABSENT {
        versions.set_deleted(row, deleted);
    }
}

/// The label ids a label history entry holds: ascending ids of the graph's
/// dictionary.
fn label_ids_of(
    kept: &Value,
    dictionary: &NameDictionary,
) -> std::result::Result<Box<[u32]>, String> {
    let Value::List(items) = kept else {
        return Err(format!("{kept:?} where label ids are"));
    };
    let mut ids: Vec<u32> = Vec::with_capacity(items.len());
    for item in items.iter() {
        let id = match item {
            Value::Int64(id) => u32::try_from(*id).ok(),
            _ => None,
        }
        .filter(|id| dictionary.get_name(*id).is_some())
        .ok_or_else(|| format!("{item:?} is no label of the graph"))?;
        if ids.last().is_some_and(|last| *last >= id) {
            return Err(format!("label ids {kept:?} are not ascending"));
        }
        ids.push(id);
    }
    Ok(ids.into_boxed_slice())
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
pub fn read_lpg_section(source: &dyn SectionSource) -> Result<(u32, Vec<LoadedGraph>)> {
    read_lpg_section_with_backward_edges(source, true)
}

/// Reads an `LPG_STORE` section of version 4 as [`read_lpg_section`] does,
/// into stores that keep each node's incoming edges or, with
/// `backward_edges` false, keep none (see
/// [`RowGroupStore::with_backward_edges`]): the reader builds the adjacency
/// the stores keep, from the edge table where the section holds none of a
/// direction, and checks the incoming lists a section holds before such a
/// store drops them.
///
/// # Errors
///
/// As [`read_lpg_section`].
#[expect(
    clippy::too_many_lines,
    reason = "one pass over the chunks, a check per kind"
)]
pub fn read_lpg_section_with_backward_edges(
    source: &dyn SectionSource,
    backward_edges: bool,
) -> Result<(u32, Vec<LoadedGraph>)> {
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
        let mut inner = Inner {
            incoming_edges: if backward_edges {
                IncomingEdges::Kept
            } else {
                IncomingEdges::NotKept
            },
            ..Inner::default()
        };
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
            history: LoadedHistory::default(),
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
            || !matches!(
                chunk.kind,
                ChunkKind::Column | ChunkKind::Adjacency | ChunkKind::History
            )
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
        if chunk.kind == ChunkKind::History {
            graph
                .read_history(chunk, &bytes, meta.epoch)
                .map_err(refuse)?;
            index += 1;
            continue;
        }

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
                    graph
                        .inner
                        .nodes
                        .entry(group)
                        .or_default()
                        .versions
                        .set_always(row);
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
                    // The rows were checked above: their group is there.
                    let Some(columns) = columns else {
                        return Err(refuse(format!("group {group} holds no rows")));
                    };
                    columns.add_cold(
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
                    group.versions.set_always(row);
                    group.set_ends(row, *src, dst, edge_type);
                }
                index += 2;
            }
            ChunkNamespace::OutgoingAdjacency | ChunkNamespace::IncomingAdjacency => {
                match graph_meta.adjacency_written {
                    WrittenAdjacency::None => {
                        return Err(refuse(
                            "an adjacency chunk in a graph whose metadata says none was written"
                                .to_string(),
                        ));
                    }
                    WrittenAdjacency::Outgoing
                        if chunk.namespace == ChunkNamespace::IncomingAdjacency =>
                    {
                        return Err(refuse(
                            "an incoming adjacency chunk in a graph whose metadata says only \
                             the outgoing one was written"
                                .to_string(),
                        ));
                    }
                    WrittenAdjacency::Both | WrittenAdjacency::Outgoing => {}
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
                    // A piece follows its list's first chunk (checked above).
                    let Some((_, whole)) = graph.adjacency[direction].last_mut() else {
                        return Err(refuse("a piece without its list's first chunk".to_string()));
                    };
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
        let written = graph_meta.adjacency_written;
        for (direction, lists) in graph.adjacency.iter().enumerate() {
            if !written.holds(direction == 0) {
                continue;
            }
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
        let keeps_incoming = graph.inner.keeps_incoming();
        for (direction, lists) in graph.adjacency.into_iter().enumerate() {
            // The incoming lists of a store that keeps none are dropped,
            // once checked.
            if direction == 1 && !keeps_incoming {
                continue;
            }
            let mut by_group: BTreeMap<u64, Vec<(usize, Vec<Adjacent>)>> = BTreeMap::new();
            for (node, list) in lists {
                let (group, row) = locate(node);
                by_group.entry(group).or_default().push((row, list));
            }
            for (group, lists) in by_group {
                // Every list's node was checked to be a node of the graph.
                let Some(group) = graph.inner.nodes.get_mut(&group) else {
                    return Err(Error::corruption(format!(
                        "LPG section: adjacency of group {group}, which holds no nodes"
                    )));
                };
                *group.adjacency_mut(direction == 0) = Adjacency::from_sorted(lists);
            }
        }
        // What the file lacks of the adjacency the store keeps (a file
        // written without adjacency, or without the incoming one): built
        // from the edge table, outgoing then incoming.
        let build = [
            !written.holds(true),
            keeps_incoming && !written.holds(false),
        ];
        if build.contains(&true) {
            let edges: Vec<(u64, (u64, u64, u32))> = graph
                .inner
                .edges
                .iter()
                .flat_map(|(index, group)| {
                    group
                        .versions
                        .rows()
                        .map(move |(row, _)| (index * ROWS_PER_GROUP + row as u64, group.ends(row)))
                })
                .collect();
            for (edge, (src, dst, edge_type)) in edges {
                for (node, other, outgoing) in [(src, dst, true), (dst, src, false)] {
                    if !build[usize::from(!outgoing)] {
                        continue;
                    }
                    let (group, row) = locate(node);
                    // Every edge's ends were checked to be nodes of the graph.
                    let Some(group) = graph.inner.nodes.get_mut(&group) else {
                        return Err(Error::corruption(format!(
                            "LPG section: edge {edge} ends at node {node}, which is no node"
                        )));
                    };
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
                for (direction, built) in build.iter().enumerate() {
                    if *built {
                        group.adjacency_mut(direction == 0).merge();
                    }
                }
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

        // The versions and chains of the history, once every row is in.
        std::mem::take(&mut graph.history)
            .apply(&mut graph.inner)
            .map_err(refuse)?;

        // The counts of what a reader at the section's epoch sees.
        let at = Read::At(meta.epoch);
        let mut counts = super::Counts::default();
        for group in graph.inner.nodes.values() {
            for (row, version) in group.versions.rows() {
                if !version.visible(at) {
                    continue;
                }
                counts.nodes += 1;
                for label in group.labels_at(row, at) {
                    counts.label(label, 1);
                }
            }
        }
        for group in graph.inner.edges.values() {
            for (row, version) in group.versions.rows() {
                if !version.visible(at) {
                    continue;
                }
                counts.edges += 1;
                counts.edge_type(group.ends(row).2, 1);
            }
        }
        graph.inner.counts = counts;

        let store = RowGroupStore::default();
        if !graph.inner.listed.is_empty() {
            store.note_listed();
        }
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

// ── The section of a database file ──────────────────────────────────

/// The row-group store's `LPG_STORE` section (version 4) in a `.grafeo`
/// container, the part `LpgStoreSection` plays for `LpgStore`: it writes a
/// root and its named graphs (see [`RowGroupStore::write_section_at`]) and
/// reads them back into its root ([`RowGroupStore::read_section`]).
///
/// A checkpoint builds it with the epoch it records ([`at`](Self::at)) while
/// it holds commits off: every commit it holds is stamped at or before that
/// epoch, and the section holds the committed state at it, with nothing a
/// transaction still open wrote. No committed copy and no change sets of
/// open transactions are needed, as the writer reads each row as a reader
/// at the epoch does.
///
/// The section holds data only: the indexes are the catalog's, which the
/// loader builds from the rows once the section is read.
pub struct RowGroupStoreSection {
    store: Arc<RowGroupStore>,
    /// The epoch to write at; `None`: the latest any graph reached.
    epoch: Option<EpochId>,
    /// The epoch the history it writes starts at; `None`: no history.
    history_from: Option<EpochId>,
    caps: ChunkCaps,
    dirty: AtomicBool,
}

impl RowGroupStoreSection {
    /// The section of `store`, a root, writing chunks of at most
    /// [`ChunkCaps::current`] (the caps a test set on this thread, or the
    /// default ones) at the latest epoch any of its graphs reached.
    #[must_use]
    pub fn new(store: Arc<RowGroupStore>) -> Self {
        Self::with_caps(store, ChunkCaps::current())
    }

    /// The section of `store`, a root, writing chunks of at most `caps`.
    #[must_use]
    pub fn with_caps(store: Arc<RowGroupStore>, caps: ChunkCaps) -> Self {
        Self {
            store,
            epoch: None,
            history_from: None,
            caps,
            dirty: AtomicBool::new(false),
        }
    }

    /// Writes the committed state at `epoch` (see
    /// [`RowGroupStore::write_section_at`]).
    #[must_use]
    pub fn at(mut self, epoch: EpochId) -> Self {
        self.epoch = Some(epoch);
        self
    }

    /// Writes the history from `from` too, with the epoch [`at`](Self::at)
    /// gave (see [`RowGroupStore::write_section_with_history`]): a
    /// checkpoint of a database that keeps history passes the start of its
    /// retention window.
    #[must_use]
    pub fn with_history_from(mut self, from: EpochId) -> Self {
        self.history_from = Some(from);
        self
    }

    /// The root store it writes and reads into.
    #[must_use]
    pub fn store(&self) -> &Arc<RowGroupStore> {
        &self.store
    }
}

impl Section for RowGroupStoreSection {
    fn section_type(&self) -> SectionType {
        SectionType::LpgStore
    }

    fn version(&self) -> u8 {
        LPG_SECTION_VERSION
    }

    /// Version 4 has no form as one blob: it is written as chunks
    /// ([`write_to`](Section::write_to)).
    fn serialize(&self) -> Result<Vec<u8>> {
        Err(Error::Internal(
            "the row-group LPG section (version 4) is written as chunks, not as one blob"
                .to_string(),
        ))
    }

    /// Migrates the LPG section of 0.5.x, the block layout `data`, into the
    /// root and its named graphs, every row through the replay writer (see
    /// [`LoadTarget`](crate::graph::load::LoadTarget)).
    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        crate::graph::lpg::load_0_5_blocks(&*self.store, data)
    }

    fn write_to(&self, sink: &mut dyn SectionSink) -> Result<()> {
        match self.epoch {
            Some(epoch) => {
                self.store
                    .write_section_with_history(epoch, self.history_from, self.caps, sink)
            }
            None => self.store.write_section(self.caps, sink),
        }
    }

    /// Reads the chunks of version 4 into the root (see
    /// [`RowGroupStore::read_section`]), and migrates the sections earlier
    /// releases wrote: the chunks of version 3 (0.6.0-beta), read by
    /// `LpgStoreSection`'s reader, and the one raw chunk of 0.5.x
    /// ([`deserialize`](Section::deserialize)). A migrated section's rows
    /// come in through the replay writer, its dictionaries and ids as the
    /// file holds them, and the versions they were stamped with collapse
    /// once each graph is in, so the root holds what a section of version 4
    /// with the same rows would load; the older versions a temporal file
    /// keeps are dropped: the history of a migrated file starts at its
    /// epoch.
    ///
    /// # Errors
    ///
    /// [`Error::Serialization`] for a section of another version;
    /// [`Error::Corruption`] for chunks the reader refuses.
    fn read_from(&mut self, source: &dyn SectionSource) -> Result<()> {
        if let Some(bytes) = legacy_bytes(source)? {
            return self.deserialize(&bytes);
        }
        if source.section_version() == crate::graph::lpg::LPG_SECTION_VERSION_3 {
            return crate::graph::lpg::read_version_3_chunks(&*self.store, source);
        }
        self.store.read_section(source)
    }

    fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    fn mark_clean(&self) {
        self.dirty.store(false, Ordering::Release);
    }

    fn memory_usage(&self) -> usize {
        self.store.heap_bytes()
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use grafeo_common::storage::{
        ChunkCaps, ChunkKind, ChunkMeta, ChunkNamespace, SectionSink, SectionSource,
    };
    use grafeo_common::types::{EdgeId, NodeId, Value};
    use grafeo_common::utils::error::{Error, Result};

    use super::{
        GraphToWrite, LPG_SECTION_VERSION, WrittenAdjacency, decode_meta, encode_meta,
        read_lpg_section, read_lpg_section_with_backward_edges, write_lpg_section,
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
        write_lpg_section(&graphs, 1, store.current_epoch(), caps, &mut chunks).unwrap();
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

    /// An empty store, the root of a new database, writes the smallest
    /// metadata a graph has (57 bytes for it), and reads back: the reader
    /// refused it, taking a graph to need 60 bytes.
    #[test]
    fn an_empty_store_round_trips() {
        let store = RowGroupStore::new();
        let chunks = write(&store, ChunkCaps::DEFAULT);
        assert_same(&read(&chunks), &store, "an empty store");
    }

    /// A store after a seeded workload of commits and rollbacks.
    fn workload(seed: u64) -> RowGroupStore {
        workload_on(RowGroupStore::new(), seed)
    }

    /// `store` after the seeded workload of [`workload`].
    fn workload_on(store: RowGroupStore, seed: u64) -> RowGroupStore {
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

    /// The store a section of one graph loads into, keeping backward edges
    /// or not.
    fn read_keeping(chunks: &Chunks, backward_edges: bool) -> RowGroupStore {
        let (next_graph_id, mut graphs) =
            read_lpg_section_with_backward_edges(chunks, backward_edges).unwrap();
        assert_eq!((next_graph_id, graphs.len()), (1, 1));
        graphs.remove(0).store
    }

    /// The namespaces of a section's adjacency chunks, and its graph's
    /// adjacency byte.
    fn adjacency_of(chunks: &Chunks) -> (usize, usize, WrittenAdjacency) {
        let count = |namespace: ChunkNamespace| {
            chunks
                .metas
                .iter()
                .filter(|meta| meta.namespace == namespace)
                .count()
        };
        (
            count(ChunkNamespace::OutgoingAdjacency),
            count(ChunkNamespace::IncomingAdjacency),
            decode_meta(chunks.bytes.last().unwrap()).unwrap().graphs[0].adjacency_written,
        )
    }

    /// A store without backward edges (10b) checkpoints its outgoing
    /// adjacency only, and says so in the metadata (byte 2); the section
    /// reads back into such a store as the committed state, every incoming
    /// read included (the image checks them), with no incoming adjacency,
    /// and the loaded store writes the same chunks again. At the default
    /// caps and at caps that cut the tables into many row groups.
    #[test]
    fn a_store_without_backward_edges_checkpoints_no_incoming_adjacency() {
        for caps in [
            ChunkCaps::DEFAULT,
            ChunkCaps {
                max_rows: 3,
                max_bytes: 4_096,
            },
        ] {
            for seed in 1..=6 {
                let store = workload_on(RowGroupStore::with_backward_edges(false), seed);
                let chunks = write(&store, caps);
                let what = format!("seed {seed}, caps {caps:?}");
                let (outgoing, incoming, written) = adjacency_of(&chunks);
                assert_eq!(
                    outgoing > 0,
                    !committed(&store).edges.is_empty(),
                    "{what}: the outgoing adjacency is written"
                );
                assert_eq!(
                    (incoming, written),
                    (0, WrittenAdjacency::Outgoing),
                    "{what}"
                );
                let loaded = read_keeping(&chunks, false);
                assert!(!loaded.keeps_backward_edges(), "{what}");
                let again = write(&loaded, caps);
                assert_same(&loaded, &store, &what);
                assert_eq!(loaded.memory().incoming_bytes, 0, "{what}");
                assert!(
                    again.metas == chunks.metas && again.bytes == chunks.bytes,
                    "{what}: the loaded store writes the same chunks"
                );
            }
        }
    }

    /// The store decides the adjacency a section loads into (10b ruling): a
    /// section written with incoming adjacency loads into a store without
    /// backward edges with its incoming lists checked and dropped; one
    /// written without it loads into a store with backward edges with them
    /// built from the edge table, which then writes what a store that
    /// always kept them writes.
    #[test]
    fn the_store_decides_the_adjacency_a_section_loads() {
        for seed in 1..=6 {
            let what = format!("seed {seed}");
            let kept = workload(seed);
            let scanned = workload_on(RowGroupStore::with_backward_edges(false), seed);
            let (with_incoming, without_incoming) = (
                write(&kept, ChunkCaps::DEFAULT),
                write(&scanned, ChunkCaps::DEFAULT),
            );
            assert_eq!(adjacency_of(&with_incoming).2, WrittenAdjacency::Both);

            let dropped = read_keeping(&with_incoming, false);
            assert!(!dropped.keeps_backward_edges());
            let rewritten = write(&dropped, ChunkCaps::DEFAULT);
            assert_same(&dropped, &kept, &what);
            assert_eq!(dropped.memory().incoming_bytes, 0, "{what}: dropped");
            assert!(
                rewritten.metas == without_incoming.metas
                    && rewritten.bytes == without_incoming.bytes,
                "{what}: it writes what a store without backward edges writes"
            );

            let built = read_keeping(&without_incoming, true);
            assert!(built.keeps_backward_edges());
            let rewritten = write(&built, ChunkCaps::DEFAULT);
            // `assert_same` took an id of `kept`: compared with its twin.
            assert_same(&built, &scanned, &what);
            // The writer reads the incoming lists from the built adjacency.
            assert!(
                rewritten.metas == with_incoming.metas && rewritten.bytes == with_incoming.bytes,
                "{what}: it writes what a store that kept them writes"
            );
        }
    }

    /// The adjacency byte names none, both directions or the outgoing one
    /// only; any other byte is no adjacency the format knows.
    #[test]
    fn the_adjacency_byte_names_what_was_written() {
        for written in [
            WrittenAdjacency::None,
            WrittenAdjacency::Both,
            WrittenAdjacency::Outgoing,
        ] {
            assert_eq!(
                WrittenAdjacency::from_byte(written.to_byte()),
                Some(written)
            );
        }
        assert_eq!(
            [0, 1, 2]
                .map(|byte| WrittenAdjacency::from_byte(byte)
                    .map(|w| (w.holds(true), w.holds(false)))),
            [
                Some((false, false)),
                Some((true, true)),
                Some((true, false))
            ]
        );
        assert_eq!(WrittenAdjacency::from_byte(3), None);
    }

    /// After `collect` at the store's epoch, with no transaction open, the
    /// version info costs what a store loaded from its checkpoint holds: one
    /// bit per row, no stamps.
    #[test]
    fn after_collect_the_versions_cost_what_a_loaded_stores_do() {
        for seed in 1..=10 {
            let store = workload(seed);
            let flat = read(&write(&store, ChunkCaps::DEFAULT)).version_bytes();
            assert!(
                store.stamped_vectors() > 0,
                "seed {seed}: the workload stamped its vectors"
            );
            store.collect(store.current_epoch());
            assert_eq!(store.version_bytes(), flat, "seed {seed}");
            assert_same(
                &read(&write(&store, ChunkCaps::DEFAULT)),
                &store,
                "after it",
            );
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

    /// The labels and property keys of the section's label and property
    /// columns, by name.
    fn written_names(chunks: &Chunks) -> (Vec<String>, Vec<String>) {
        let meta = decode_meta(chunks.bytes.last().unwrap()).unwrap();
        let name = |names: &[(u32, String)], id: u32| {
            names
                .iter()
                .find(|(named, _)| *named == id)
                .map(|(_, name)| name.clone())
                .unwrap()
        };
        let mut labels = Vec::new();
        let mut keys = Vec::new();
        for chunk in &chunks.metas {
            match chunk.namespace {
                ChunkNamespace::NodeLabels => {
                    labels.push(name(&meta.graphs[0].labels.names, chunk.column_id));
                }
                ChunkNamespace::NodeProperties | ChunkNamespace::EdgeProperties => {
                    keys.push(name(&meta.graphs[0].keys.names, chunk.column_id));
                }
                _ => {}
            }
        }
        labels.sort();
        labels.dedup();
        keys.sort();
        keys.dedup();
        (labels, keys)
    }

    /// A checkpoint during an open write writes values and labels as they
    /// were committed: what the open transaction set, removed, added or took
    /// away is not in the section, on nodes and edges, nor a label or key
    /// only it wrote; after its commit the next checkpoint holds it all.
    #[test]
    fn an_open_transactions_values_and_labels_are_not_in_the_section() {
        let (store, [alix, gus, mia, vincent]) = four_people();
        let before = committed(&store);
        let mut tx = Tx::begin(&store, 20);
        tx.set_node(alix, "age", Value::Int64(88)).unwrap();
        tx.set_node(vincent, "nickname", Value::from("Vince"))
            .unwrap();
        tx.remove_node_value(gus, "name").unwrap();
        tx.add_label(mia, "Traveller").unwrap();
        tx.remove_label(gus, "Employee").unwrap();
        tx.set_edge(EdgeId::new(0), "since", Value::Int64(2030))
            .unwrap();
        tx.remove_edge_value(EdgeId::new(1), "since").unwrap();

        let chunks = write(&store, ChunkCaps::DEFAULT);
        let loaded = read(&chunks);
        let found = committed(&loaded);
        found.consistent("the loaded store").unwrap();
        assert_eq!((&found.nodes, &found.edges), (&before.nodes, &before.edges));
        let (labels, keys) = written_names(&chunks);
        assert_eq!(labels, ["Employee", "Person"], "no Traveller column");
        assert_eq!(keys, ["age", "city", "name", "since"], "no nickname column");
        assert_eq!(
            counts(&loaded, &LABELS),
            counts(&store, &LABELS),
            "the counts of the committed state"
        );

        tx.commit().unwrap();
        assert_same(
            &read(&write(&store, ChunkCaps::DEFAULT)),
            &store,
            "after the commit",
        );
    }

    /// A checkpoint at an epoch before the last commit writes what a reader
    /// at that epoch sees: the creates, deletes, values and labels of the
    /// later commit are not in it, though its versions are still chained.
    #[test]
    fn a_checkpoint_at_an_earlier_epoch_writes_what_a_reader_then_saw() {
        use crate::graph::conformance::{Reader, image};

        let (store, [alix, gus, mia, _]) = four_people();
        let earlier = store.current_epoch();
        let then = image(&store, Reader::At(earlier));
        let mut tx = Tx::begin(&store, 20);
        tx.set_node(alix, "age", Value::Int64(88)).unwrap();
        tx.add_label(alix, "Traveller").unwrap();
        tx.remove_label(gus, "Employee").unwrap();
        tx.delete_edge(EdgeId::new(1)).unwrap();
        tx.delete_node(mia).unwrap();
        let jules = tx
            .create_node(&["Person"], &[("name", Value::from("Jules"))])
            .unwrap();
        tx.create_edge(jules, alix, "KNOWS", &[]).unwrap();
        tx.commit().unwrap();
        assert!(
            store.update_nodes() > 0,
            "the commit's versions are chained"
        );

        let mut chunks = Chunks {
            version: LPG_SECTION_VERSION,
            ..Chunks::default()
        };
        let graphs = [GraphToWrite {
            id: 0,
            name: "",
            store: &store,
        }];
        write_lpg_section(&graphs, 1, earlier, ChunkCaps::DEFAULT, &mut chunks).unwrap();
        let loaded = read(&chunks);
        let found = committed(&loaded);
        found.consistent("the loaded store").unwrap();
        assert_eq!((&found.nodes, &found.edges), (&then.nodes, &then.edges));
        assert_eq!(loaded.current_epoch(), earlier, "the section's epoch");
        assert_eq!(
            (loaded.node_count(), loaded.edge_count()),
            (then.nodes.len(), then.edges.len())
        );
    }

    /// The section of a database file writes its root at the epoch it was
    /// given, as the container holds sections, and reads it back into a
    /// root; it refuses bytes that are no 0.5.x section and has no form as
    /// one blob.
    #[test]
    fn the_file_section_writes_at_its_epoch_and_reads_into_a_root() {
        use std::sync::Arc;

        use grafeo_common::storage::{ImageSource, MemoryImage, Section, SectionType};
        use grafeo_common::types::ReadView;

        use super::RowGroupStoreSection;

        let (store, [alix, ..]) = four_people();
        let store = Arc::new(store);
        let earlier = store.current_epoch();
        let mut tx = Tx::begin(&*store, 20);
        tx.set_node(alix, "age", Value::Int64(88)).unwrap();
        tx.commit().unwrap();

        let section =
            RowGroupStoreSection::with_caps(Arc::clone(&store), ChunkCaps::DEFAULT).at(earlier);
        assert_eq!(
            (section.section_type(), section.version()),
            (SectionType::LpgStore, LPG_SECTION_VERSION)
        );
        let mut image = MemoryImage::new();
        image
            .begin_section(section.section_type(), section.version())
            .unwrap();
        section.write_to(&mut image).unwrap();
        let root = Arc::new(RowGroupStore::new());
        RowGroupStoreSection::new(Arc::clone(&root))
            .read_from(&*image.section_source(SectionType::LpgStore).unwrap())
            .unwrap();
        assert_eq!(root.current_epoch(), earlier);
        assert_eq!(
            root.node_property_at(alix, "age", ReadView::committed(earlier)),
            Some(Value::Int64(19)),
            "Alix's age before the later commit"
        );
        assert_eq!(root.node_count(), 4);

        // A 0.5.x blob is migrated (see `load`); bytes that are none are
        // refused as damage, and load nothing.
        let legacy =
            MemoryImage::from_raw(vec![(SectionType::LpgStore, vec![1, 9, 8, 8])]).unwrap();
        let empty = Arc::new(RowGroupStore::new());
        let error = RowGroupStoreSection::new(Arc::clone(&empty))
            .read_from(&*legacy.section_source(SectionType::LpgStore).unwrap())
            .unwrap_err();
        assert!(matches!(error, Error::Corruption(_)), "{error}");
        assert_eq!(empty.node_count(), 0);
        assert!(section.serialize().is_err(), "no blob form");
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
        meta.graphs[0].adjacency_written = WrittenAdjacency::None;
        chunks.bytes[last] = Bytes::from(encode_meta(&meta).unwrap());
        assert_same(&read(&chunks), &store, "rebuilt adjacency");
    }

    /// A loaded store keeps its property columns as the file's chunks until
    /// a write or a read of a complex value needs decoding. Scalar reads
    /// stay encoded, and writes preserve the column's other rows.
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

        assert_eq!(
            loaded.get_node_property(gus, &PropertyKey::new("age")),
            Some(Value::Int64(3))
        );
        assert_eq!(loaded.cold_columns(), 4, "a scalar read stays encoded");
        assert_eq!(loaded.heap_bytes(), cold_bytes);

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

        // Whole entities use the same scalar path; the city string decodes.
        assert!(loaded.get_node(alix).is_some());
        assert!(
            loaded
                .get_edge(grafeo_common::types::EdgeId::new(0))
                .is_some()
        );
        assert_eq!(
            loaded.cold_columns(),
            1,
            "the edge's scalar column stays encoded"
        );
        assert!(
            loaded.heap_bytes() != cold_bytes,
            "the decoded columns are counted"
        );
    }

    /// A transaction's write to a cold column thaws it (#412 decision Q9)
    /// and keeps the loaded value for every other reader until its commit,
    /// and for readers before the commit until a collect at its epoch.
    #[test]
    fn a_write_to_a_cold_column_keeps_the_loaded_value_for_other_readers() {
        use grafeo_common::types::{PropertyKey, ReadView};

        let (store, [alix, gus, ..]) = four_people();
        let age = PropertyKey::new("age");
        let (alix_age, gus_age) = (
            store.get_node_property(alix, &age),
            store.get_node_property(gus, &age),
        );
        assert!(alix_age.is_some());
        let loaded = read(&write(&store, ChunkCaps::DEFAULT));
        let before = ReadView::committed(loaded.current_epoch());
        let cold = loaded.cold_columns();

        let mut tx = Tx::begin(&loaded, 20);
        tx.set_node(alix, "age", Value::Int64(88)).unwrap();
        assert_eq!(
            loaded.cold_columns(),
            cold - 1,
            "the write thawed the column"
        );
        let own = ReadView::transaction(tx.id, tx.snapshot);
        assert_eq!(
            loaded.node_property_at(alix, "age", own),
            Some(Value::Int64(88))
        );
        assert_eq!(loaded.node_property_at(alix, "age", before), alix_age);
        assert_eq!(loaded.node_property_at(gus, "age", before), gus_age);

        let epoch = tx.commit().unwrap();
        let after = ReadView::committed(epoch);
        assert_eq!(loaded.node_property_at(alix, "age", before), alix_age);
        assert_eq!(
            loaded.node_property_at(alix, "age", after),
            Some(Value::Int64(88))
        );
        assert_eq!(loaded.update_nodes(), 1);
        loaded.collect(epoch);
        assert_eq!(loaded.update_nodes(), 0);
        assert_eq!(
            loaded.node_property_at(alix, "age", after),
            Some(Value::Int64(88))
        );
        assert_eq!(loaded.node_property_at(gus, "age", after), gus_age);
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
        let flat = read(&chunks).version_bytes();
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
        assert_eq!(
            loaded.version_bytes(),
            flat,
            "the deleted rows leave every vector flat, with one bit per row"
        );
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
                with_meta(|meta| meta.graphs[0].adjacency_written = WrittenAdjacency::None),
                "says none was written",
            ),
            (
                "incoming adjacency chunks in a graph that wrote the outgoing only",
                with_meta(|meta| meta.graphs[0].adjacency_written = WrittenAdjacency::Outgoing),
                "says only the outgoing one was written",
            ),
            (
                "an incoming adjacency chunk missing",
                without(first(ChunkNamespace::IncomingAdjacency)),
                "adjacency lists 0 edges in direction 1",
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
