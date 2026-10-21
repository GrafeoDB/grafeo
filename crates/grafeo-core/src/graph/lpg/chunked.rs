//! The reader of the LPG section in chunks of version 3, which
//! 0.6.0-beta.1 wrote. This release writes the row-group store's section
//! (version 4) and reads version 3 to migrate it into a root, every row
//! through [`LoadTarget`]; no build writes version 3 any more.
//!
//! Per graph in id order (graph 0 is the default graph, then the named graphs
//! by their ids), the section holds its node table and then its edge table.
//! A row is a node or edge id, and a table was written in row groups of
//! `max_rows` rows, `[k * max_rows, (k + 1) * max_rows)`; only the groups
//! that hold a node (or an edge) were written, and the chunks of one group
//! come together. In a group:
//!
//! - the node table holds [`COLUMN_LABELS`] (node structure namespace): the
//!   node's label ids, ascending, joined by `,` as a string (`""` for a node
//!   without labels);
//! - the edge table holds [`COLUMN_SOURCE`], [`COLUMN_TARGET`] and
//!   [`COLUMN_EDGE_TYPE`] (edge structure namespace): `Int64` values of the
//!   source and target node ids and the edge type id, as three chunks of one
//!   range in a row;
//! - each property column with a value in the group holds `Column` chunks of
//!   the current values, in the table's property namespace, with the property
//!   key's id as the column id. A build with the `temporal` feature wrote
//!   each value with the epoch it was set at, and the older versions in
//!   `History` chunks, each right before the `Column` chunk of its rows: a
//!   history value is a list of `[epoch, value]` lists, epochs ascending. A
//!   property removed last (its latest version null) has no current value;
//!   its history holds every version. The reader checks the history and
//!   hands it to the store as older versions, which the row-group store
//!   drops: the history of a migrated file starts at its epoch.
//!
//! Ids are permanent: a graph's id, and the ids of each graph's labels, edge
//! types and property keys, are given once and never reassigned or reused,
//! so a load restores every graph and every name with its id. The labels and
//! edge types are the graph's own dictionaries, and one dictionary of keys
//! serves the node and the edge table (the namespace tells them apart). A
//! name nothing uses keeps its id, and an id no name holds is a gap.
//!
//! The metadata chunk ([`ChunkMeta::meta`]) comes last: [`LpgMeta`], with the
//! caps the section was written with, the id the next graph gets, and each
//! graph with its id, name, next node and edge ids and its three dictionaries
//! (each its next id and its names with their ids). The reader fetches it
//! first, restores every graph and its dictionaries id for id, then reads the
//! other chunks in order, one at a time.

use std::collections::BTreeMap;
use std::sync::Arc;

use bytes::Bytes;
use grafeo_common::storage::{ChunkCaps, ChunkKind, ChunkMeta, ChunkNamespace, SectionSource};
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};

use crate::codec::column_chunk::{ColumnChunk, decode_column_chunk_bytes};
use crate::graph::load::{LoadTarget, LoadedNames};

/// The LPG section's version: chunks, as this module writes them.
pub(crate) const LPG_SECTION_VERSION: u8 = 3;
/// The layout byte of the metadata chunk.
pub(crate) const LPG_META_LAYOUT: u8 = 1;
/// The node table's labels column.
pub(crate) const COLUMN_LABELS: u32 = 0;
/// The edge table's source node column.
pub(crate) const COLUMN_SOURCE: u32 = 1;
/// The edge table's target node column.
pub(crate) const COLUMN_TARGET: u32 = 2;
/// The edge table's edge type column.
pub(crate) const COLUMN_EDGE_TYPE: u32 = 3;

/// The largest epoch a section holds: history values store epochs as
/// `Int64`.
const MAX_EPOCH: u64 = i64::MAX.unsigned_abs();

/// The metadata chunk of an LPG section.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LpgMeta {
    /// [`LPG_META_LAYOUT`].
    pub layout: u8,
    /// The rows per row group and chunk the section was written with.
    pub max_rows: u32,
    /// The byte cap the section was written with.
    pub max_bytes: u32,
    /// The store's epoch when a build with `temporal` wrote it, 0 otherwise.
    pub epoch: u64,
    /// The id the next named graph gets: above every graph id given out.
    pub next_graph_id: u32,
    /// The default graph (id 0, an empty name) first, then the named graphs
    /// by id, ascending (one of them may have the empty name: the default
    /// graph is the one with id 0).
    pub graphs: Vec<GraphMeta>,
}

/// One graph of an LPG section.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GraphMeta {
    /// The graph's id: 0 for the default graph.
    pub id: u32,
    /// The graph's name; empty for the default graph.
    pub name: String,
    /// The id the graph gives its next node: the node table's rows are below it.
    pub next_node_id: u64,
    /// The id the graph gives its next edge: the edge table's rows are below it.
    pub next_edge_id: u64,
    /// The graph's labels.
    pub labels: DictionaryMeta,
    /// The graph's edge types.
    pub edge_types: DictionaryMeta,
    /// The graph's property keys, of the node and the edge table.
    pub keys: DictionaryMeta,
}

/// A name dictionary of a graph, as the metadata chunk lists it.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct DictionaryMeta {
    /// The id the next new name gets: above every id given out.
    pub next_id: u32,
    /// The names with their ids, ids ascending.
    pub names: Vec<(u32, String)>,
}

/// The table a column belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Table {
    /// Rows are node ids.
    Node,
    /// Rows are edge ids.
    Edge,
}

impl Table {
    /// "node" or "edge", for errors.
    fn entity(self) -> &'static str {
        match self {
            Self::Node => "node",
            Self::Edge => "edge",
        }
    }

    /// The node or edge `row` of the table.
    fn entity_at(self, row: u64) -> grafeo_common::change::Entity {
        match self {
            Self::Node => grafeo_common::change::Entity::Node(NodeId::new(row)),
            Self::Edge => grafeo_common::change::Entity::Edge(EdgeId::new(row)),
        }
    }
}

// ── Reading ─────────────────────────────────────────────────────────

/// Encodes a metadata chunk, for the tests that craft sections (no build
/// writes version 3). Little-endian:
///
/// | Field | Encoding |
/// | --- | --- |
/// | layout | u8, [`LPG_META_LAYOUT`] |
/// | max_rows, max_bytes | u32 each |
/// | epoch | u64 |
/// | next graph id | u32 |
/// | graphs | a count u32, then per graph its id u32, name (its length u32 and UTF-8), next node id u64, next edge id u64, and its labels, edge types and property keys |
/// | a dictionary | its next id u32, a count u32, then per name its id u32 and the name (as above) |
///
/// Nothing in the layout limits its size: a reader checks every count and
/// length against the bytes left, so the chunk holds as many names as the
/// store has.
///
/// # Errors
///
/// Returns [`Error::Serialization`] when a list holds more than `u32::MAX`
/// entries or a name is longer than `u32::MAX` bytes.
#[cfg(test)]
pub(crate) fn encode_lpg_meta(meta: &LpgMeta) -> Result<Vec<u8>> {
    let mut out = vec![meta.layout];
    out.extend_from_slice(&meta.max_rows.to_le_bytes());
    out.extend_from_slice(&meta.max_bytes.to_le_bytes());
    out.extend_from_slice(&meta.epoch.to_le_bytes());
    out.extend_from_slice(&meta.next_graph_id.to_le_bytes());
    put_count(meta.graphs.len(), "graph", &mut out)?;
    for graph in &meta.graphs {
        out.extend_from_slice(&graph.id.to_le_bytes());
        put_name(&graph.name, "graph", &mut out)?;
        out.extend_from_slice(&graph.next_node_id.to_le_bytes());
        out.extend_from_slice(&graph.next_edge_id.to_le_bytes());
        for (dictionary, what) in [
            (&graph.labels, "label"),
            (&graph.edge_types, "edge type"),
            (&graph.keys, "property key"),
        ] {
            out.extend_from_slice(&dictionary.next_id.to_le_bytes());
            put_count(dictionary.names.len(), what, &mut out)?;
            for (id, name) in &dictionary.names {
                out.extend_from_slice(&id.to_le_bytes());
                put_name(name, what, &mut out)?;
            }
        }
    }
    Ok(out)
}

/// Appends the count of a list of `what`s.
#[cfg(test)]
fn put_count(count: usize, what: &str, out: &mut Vec<u8>) -> Result<()> {
    let count = u32::try_from(count).map_err(|_| {
        Error::Serialization(format!(
            "LPG metadata chunk: {count} {what}s, more than {} a section holds",
            u32::MAX
        ))
    })?;
    out.extend_from_slice(&count.to_le_bytes());
    Ok(())
}

/// Appends a name: its length and UTF-8.
#[cfg(test)]
fn put_name(name: &str, what: &str, out: &mut Vec<u8>) -> Result<()> {
    let length = u32::try_from(name.len()).map_err(|_| {
        Error::Serialization(format!(
            "LPG metadata chunk: a {what} name of {} bytes, longer than the {} a section holds",
            name.len(),
            u32::MAX
        ))
    })?;
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(name.as_bytes());
    Ok(())
}

/// Decodes a metadata chunk of [`encode_lpg_meta`]'s layout. Every count and
/// length is checked against the bytes left before anything is allocated.
///
/// # Errors
///
/// Returns [`Error::Corruption`] naming the byte offset of what is wrong:
/// another layout than [`LPG_META_LAYOUT`], a count or length past the bytes
/// left, a name that is not UTF-8, an unknown table, bytes after the
/// metadata.
pub(crate) fn decode_lpg_meta(bytes: &[u8]) -> Result<LpgMeta> {
    let mut reader = MetaReader { bytes, pos: 0 };
    let layout = reader.u8("layout", "")?;
    if layout != LPG_META_LAYOUT {
        return Err(reader.refuse(
            0,
            format!("layout {layout}, this build reads layout {LPG_META_LAYOUT}"),
        ));
    }
    let max_rows = reader.u32("max_rows", "")?;
    let max_bytes = reader.u32("max_bytes", "")?;
    let epoch = reader.u64("epoch", "")?;
    let next_graph_id = reader.u32("next graph id", "")?;
    // An id, a name length, two next ids and three dictionaries (a next id
    // and a count each) at least.
    let count = reader.count("graph", 4 + 4 + 8 + 8 + 3 * (4 + 4))?;
    let mut graphs = Vec::with_capacity(count);
    for _ in 0..count {
        graphs.push(GraphMeta {
            id: reader.u32("graph id", "")?,
            name: reader.name("graph")?,
            next_node_id: reader.u64("next node id", "")?,
            next_edge_id: reader.u64("next edge id", "")?,
            labels: reader.dictionary("label")?,
            edge_types: reader.dictionary("edge type")?,
            keys: reader.dictionary("property key")?,
        });
    }
    if reader.pos != bytes.len() {
        return Err(reader.refuse(
            reader.pos,
            format!("{} bytes after the metadata", bytes.len() - reader.pos),
        ));
    }
    Ok(LpgMeta {
        layout,
        max_rows,
        max_bytes,
        epoch,
        next_graph_id,
        graphs,
    })
}

/// Reads a metadata chunk from its first byte.
struct MetaReader<'b> {
    bytes: &'b [u8],
    pos: usize,
}

impl MetaReader<'_> {
    /// The error of what is wrong at byte `at`.
    fn refuse(&self, at: usize, what: String) -> Error {
        Error::corruption(format!("LPG metadata chunk, byte {at}: {what}"))
    }

    /// The next `N` bytes, the field `what` (followed by `part`, when not
    /// empty); the field's name is only put together for an error.
    fn take<const N: usize>(&mut self, what: &str, part: &str) -> Result<[u8; N]> {
        let taken: [u8; N] = self
            .bytes
            .get(self.pos..)
            .and_then(|rest| rest.get(..N))
            .and_then(|bytes| bytes.try_into().ok())
            .ok_or_else(|| {
                let space = if part.is_empty() { "" } else { " " };
                self.refuse(
                    self.pos,
                    format!("the chunk ends inside its {what}{space}{part}"),
                )
            })?;
        self.pos += N;
        Ok(taken)
    }

    fn u8(&mut self, what: &str, part: &str) -> Result<u8> {
        Ok(self.take::<1>(what, part)?[0])
    }

    fn u32(&mut self, what: &str, part: &str) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take::<4>(what, part)?))
    }

    fn u64(&mut self, what: &str, part: &str) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take::<8>(what, part)?))
    }

    /// The count of a list of `what`s, each taking at least `least` bytes,
    /// refused when the bytes left cannot hold that many.
    fn count(&mut self, what: &str, least: usize) -> Result<usize> {
        let at = self.pos;
        let count = self.u32(what, "count")?;
        let left = self.bytes.len() - self.pos;
        let count = usize::try_from(count).unwrap_or(usize::MAX);
        if count.saturating_mul(least) > left {
            return Err(self.refuse(
                at,
                format!("{count} {what}s, but only {left} bytes are left for them"),
            ));
        }
        Ok(count)
    }

    /// A name: its length (checked against the bytes left) and UTF-8.
    fn name(&mut self, what: &str) -> Result<String> {
        let at = self.pos;
        let length = self.u32(what, "name length")?;
        let left = self.bytes.len() - self.pos;
        let length = usize::try_from(length).unwrap_or(usize::MAX);
        if length > left {
            return Err(self.refuse(
                at,
                format!("a {what} name of {length} bytes, but only {left} bytes are left"),
            ));
        }
        let text = &self.bytes[self.pos..self.pos + length];
        let name = std::str::from_utf8(text)
            .map_err(|error| {
                self.refuse(
                    self.pos,
                    format!("a {what} name that is not UTF-8: {error}"),
                )
            })?
            .to_string();
        self.pos += length;
        Ok(name)
    }

    /// A dictionary: its next id, then its names with their ids.
    fn dictionary(&mut self, what: &str) -> Result<DictionaryMeta> {
        let next_id = self.u32(what, "next id")?;
        // An id and a name length at least.
        let count = self.count(what, 4 + 4)?;
        let mut names = Vec::with_capacity(count);
        for _ in 0..count {
            names.push((self.u32(what, "id")?, self.name(what)?));
        }
        Ok(DictionaryMeta { next_id, names })
    }
}

/// Applies the chunks of an LPG section of version 3 to `store`, one chunk at
/// a time: the metadata chunk (the last one) first, then the others in order,
/// each fetched when it is reached.
///
/// Rules (every failure an [`Error::Corruption`] naming the graph, the column
/// and the rows):
///
/// 1. The last chunk is the metadata chunk, the only one: layout 1, caps of 1
///    to 65,536 rows (the format's row cap) and at least one byte, an epoch
///    of at most `i64::MAX`, the default graph first (id 0, no name), the
///    named graphs with ids from 1, strictly ascending, below the next graph
///    id, each name once; per graph, each dictionary's ids strictly
///    ascending and below its next id, each name once.
/// 2. Every other chunk is a `Column` or `History` chunk (`History` only for
///    property columns) of a known graph, in a namespace of the node or the
///    edge table, of a fixed column of that table or a property key of the
///    graph, with `1 <= row_count <= max_rows`, inside one row group, ending
///    below its table's next id.
/// 3. Chunks come grouped by (graph, table, row group), in that order (the
///    node table before the edge table); a change of group closes the one
///    before. Within a group the chunks of one (kind, namespace, column)
///    ascend without overlap, and a `History` chunk never covers a row of a
///    `Column` chunk of its column that came before it.
/// 4. The edge columns 1, 2 and 3 come as three consecutive chunks of one
///    range and the same rows.
/// 5. When a group closes, every row with a property or history value has
///    its node (a labels value) or edge (an endpoints value) in the group.
/// 6. Labels are ids of the graph's labels, ascending, in decimal joined by
///    `,`; endpoints are non-negative `Int64`s and types `Int64` ids of the
///    graph's edge types. Fixed columns and `History` chunks carry no
///    epochs; a property value is never null; a history value is a non-empty
///    list of `[epoch, value]` lists with `Int64` epochs from 0, never going
///    back (equal epochs are versions of one commit), and the value after it
///    (if any) has an epoch at least the last one; epochs are at most
///    `i64::MAX`.
///
/// The rows go into `store` through [`LoadTarget`] (the row-group store,
/// which migrates the section). Before any row, every graph is restored with
/// its id (the named ones created in `store`) and its dictionaries id for
/// id, and the store's next graph id becomes the metadata's at least. The
/// history versions are checked and given to the store as older versions,
/// which the row-group store drops, as it drops the epochs values carry. An
/// edge whose end the graph does not hold is left out (see
/// [`LoadTarget::load_edge`]). At the end each graph's next ids become the
/// larger of its own and the metadata's, its epoch the metadata's, and its
/// load is finished ([`LoadTarget::load_finish`]).
///
/// # Errors
///
/// [`Error::Corruption`] for a section the rules refuse, for a node or edge
/// the store cannot allocate (naming it), and for a graph or name the store
/// holds already with another id; the source's error when a chunk cannot be
/// fetched.
pub(crate) fn read_lpg_chunks<S: LoadTarget>(store: &S, source: &dyn SectionSource) -> Result<()> {
    let chunks = source.chunks();
    let Some((last, data)) = chunks.split_last() else {
        return Err(Error::corruption(
            "LPG section: no chunk, where the last chunk is the metadata chunk",
        ));
    };
    if *last != ChunkMeta::meta() {
        return Err(Error::corruption(format!(
            "LPG section: the last chunk is a {:?} chunk of graph {}, column {}, first row {}, \
             where the metadata chunk comes last",
            last.kind, last.graph_id, last.column_id, last.row_start
        )));
    }
    if let Some(index) = data.iter().position(|chunk| chunk.kind == ChunkKind::Meta) {
        return Err(Error::corruption(format!(
            "LPG section: chunk {index} is a second metadata chunk, where only the last chunk \
             is one"
        )));
    }
    let meta = decode_lpg_meta(&source.fetch(data.len())?)?;
    let layout = Layout::new(&meta)?;

    // Every graph and its names, id for id, before any row names them.
    let named: Vec<(u32, &str)> = meta.graphs[1..]
        .iter()
        .map(|graph| (graph.id, graph.name.as_str()))
        .collect();
    let restored = store
        .load_graphs(&named, meta.next_graph_id)
        .map_err(|error| Error::corruption(format!("LPG section: {error}")))?;
    let graphs: Vec<GraphTarget<'_, S>> = std::iter::once(GraphTarget::Default(store))
        .chain(restored.into_iter().map(GraphTarget::Named))
        .collect();
    for (graph, target) in meta.graphs.iter().zip(&graphs) {
        restore_dictionaries(target.store(), graph)?;
    }

    let mut reader = Reader {
        layout,
        graphs,
        group: None,
    };
    let mut index = 0;
    while index < data.len() {
        index += reader.read(source, data, index)?;
    }
    reader.close_group()?;

    for (graph, target) in meta.graphs.iter().zip(&reader.graphs) {
        let target = target.store();
        target.load_next_ids(graph.next_node_id, graph.next_edge_id);
        target.load_section_epoch(EpochId::new(meta.epoch));
        target.load_finish();
    }
    Ok(())
}

/// Gives `target` the dictionaries of `graph`, id for id, and their next ids.
///
/// # Errors
///
/// Returns [`Error::Corruption`] when `target` holds one of the names, or one
/// of the ids, already.
fn restore_dictionaries<S: LoadTarget>(target: &S, graph: &GraphMeta) -> Result<()> {
    target
        .load_names(LoadedNames {
            labels: &graph.labels.names,
            edge_types: &graph.edge_types.names,
            keys: &graph.keys.names,
            next_ids: [
                graph.labels.next_id,
                graph.edge_types.next_id,
                graph.keys.next_id,
            ],
        })
        .map_err(|error| Error::corruption(format!("LPG section, graph {}, {error}", graph.id)))
}

/// The store a graph of the section loads into.
enum GraphTarget<'s, S> {
    Default(&'s S),
    Named(Arc<S>),
}

impl<S> GraphTarget<'_, S> {
    fn store(&self) -> &S {
        match self {
            Self::Default(store) => store,
            Self::Named(store) => store,
        }
    }
}

/// What a chunk's column holds.
enum Role {
    /// The labels column.
    Labels,
    /// The source column, the first of the three edge columns.
    Endpoints,
    /// A property column.
    Property { table: Table, key: PropertyKey },
}

impl Role {
    fn table(&self) -> Table {
        match self {
            Self::Labels => Table::Node,
            Self::Endpoints => Table::Edge,
            Self::Property { table, .. } => *table,
        }
    }
}

/// The names of one graph by id, from the metadata chunk.
struct GraphNames {
    labels: FxHashMap<u32, String>,
    edge_types: FxHashMap<u32, String>,
    keys: FxHashMap<u32, PropertyKey>,
}

/// The metadata chunk, checked.
struct Layout<'m> {
    meta: &'m LpgMeta,
    max_rows: u64,
    /// The position of each graph in `meta.graphs`, by graph id.
    positions: FxHashMap<u32, usize>,
    /// The names of each graph, by position.
    names: Vec<GraphNames>,
}

impl<'m> Layout<'m> {
    /// Checks rule 1 on `meta`.
    fn new(meta: &'m LpgMeta) -> Result<Self> {
        let refuse = |what: String| Err(Error::corruption(format!("LPG metadata chunk: {what}")));
        let caps = ChunkCaps {
            max_rows: meta.max_rows,
            max_bytes: meta.max_bytes,
        };
        if let Err(error) = caps.validate() {
            return refuse(format!(
                "max_rows {} and max_bytes {}: {error}",
                meta.max_rows, meta.max_bytes
            ));
        }
        if meta.epoch > MAX_EPOCH {
            return refuse(format!("epoch {} is above {MAX_EPOCH}", meta.epoch));
        }
        let Some(default) = meta.graphs.first() else {
            return refuse("no graph, where the default graph comes first".to_string());
        };
        if default.id != 0 || !default.name.is_empty() {
            return refuse(format!(
                "the first graph is graph {} named {:?}, where the default graph (id 0, without \
                 a name) comes first",
                default.id, default.name
            ));
        }
        let mut names_seen = FxHashSet::default();
        for pair in meta.graphs.windows(2) {
            if pair[0].id >= pair[1].id {
                return refuse(format!(
                    "graph {} comes after graph {}: the graphs come by id, each once",
                    pair[1].id, pair[0].id
                ));
            }
        }
        for graph in &meta.graphs[1..] {
            if graph.id >= meta.next_graph_id {
                return refuse(format!(
                    "graph {} is not below the next graph id {}",
                    graph.id, meta.next_graph_id
                ));
            }
            if !names_seen.insert(graph.name.as_str()) {
                return refuse(format!("two named graphs are named {:?}", graph.name));
            }
        }
        let mut positions = FxHashMap::default();
        let mut names = Vec::with_capacity(meta.graphs.len());
        for (position, graph) in meta.graphs.iter().enumerate() {
            positions.insert(graph.id, position);
            let checked = |dictionary: &DictionaryMeta, what: &str| {
                let mut by_id = FxHashMap::default();
                let mut seen = FxHashSet::default();
                for pair in dictionary.names.windows(2) {
                    if pair[0].0 >= pair[1].0 {
                        return Err(format!(
                            "graph {}: {what} id {} comes after {what} id {}: the ids ascend, \
                             each once",
                            graph.id, pair[1].0, pair[0].0
                        ));
                    }
                }
                for (id, name) in &dictionary.names {
                    if *id >= dictionary.next_id {
                        return Err(format!(
                            "graph {}: {what} id {id} is not below the next {what} id {}",
                            graph.id, dictionary.next_id
                        ));
                    }
                    if !seen.insert(name.as_str()) {
                        return Err(format!(
                            "graph {}: {what} {name:?} is listed twice",
                            graph.id
                        ));
                    }
                    by_id.insert(*id, name.clone());
                }
                Ok(by_id)
            };
            let labels = checked(&graph.labels, "label");
            let edge_types = checked(&graph.edge_types, "edge type");
            let keys = checked(&graph.keys, "property key");
            match (labels, edge_types, keys) {
                (Ok(labels), Ok(edge_types), Ok(keys)) => names.push(GraphNames {
                    labels,
                    edge_types,
                    keys: keys
                        .into_iter()
                        .map(|(id, key)| (id, PropertyKey::new(key.as_str())))
                        .collect(),
                }),
                (Err(what), _, _) | (_, Err(what), _) | (_, _, Err(what)) => return refuse(what),
            }
        }
        Ok(Self {
            meta,
            max_rows: u64::from(meta.max_rows),
            positions,
            names,
        })
    }

    /// What column `column_id` of `namespace` holds in the graph at
    /// `position`.
    fn role(
        &self,
        position: usize,
        namespace: ChunkNamespace,
        column_id: u32,
    ) -> std::result::Result<Role, String> {
        let table = match namespace {
            ChunkNamespace::NodeStructure if column_id == COLUMN_LABELS => {
                return Ok(Role::Labels);
            }
            ChunkNamespace::NodeStructure => {
                return Err(format!(
                    "column {column_id} of the node structure, which holds column \
                     {COLUMN_LABELS} (the labels) only"
                ));
            }
            ChunkNamespace::EdgeStructure => {
                return match column_id {
                    COLUMN_SOURCE => Ok(Role::Endpoints),
                    COLUMN_TARGET | COLUMN_EDGE_TYPE => Err(format!(
                        "column {column_id} comes without column {COLUMN_SOURCE} of the same \
                         rows before it"
                    )),
                    _ => Err(format!(
                        "column {column_id} of the edge structure, which holds columns \
                         {COLUMN_SOURCE} to {COLUMN_EDGE_TYPE} (source, target, edge type) only"
                    )),
                };
            }
            ChunkNamespace::NodeProperties => Table::Node,
            ChunkNamespace::EdgeProperties => Table::Edge,
            _ => {
                return Err(format!(
                    "a chunk in namespace {namespace:?}, where an LPG section has the node \
                     and edge namespaces only"
                ));
            }
        };
        match self.names[position].keys.get(&column_id) {
            Some(key) => Ok(Role::Property {
                table,
                key: key.clone(),
            }),
            None => Err(format!(
                "column {column_id} is no property key of the graph in the metadata chunk"
            )),
        }
    }
}

/// The rows of one row group set by a column, as bits from the group's first
/// row, grown as rows are set.
#[derive(Default)]
struct RowBits(Vec<u64>);

impl RowBits {
    fn set(&mut self, offset: u64) {
        // The offset is below a chunk's row count, which a decoded chunk's
        // bytes cover.
        let word = usize::try_from(offset / 64).unwrap_or(usize::MAX);
        if self.0.len() <= word {
            self.0.resize(word + 1, 0);
        }
        self.0[word] |= 1 << (offset % 64);
    }

    /// The first offset set here and not in `other`.
    fn first_outside(&self, other: &Self) -> Option<u64> {
        self.0.iter().enumerate().find_map(|(at, word)| {
            let missing = word & !other.0.get(at).copied().unwrap_or(0);
            (missing != 0).then(|| {
                u64::try_from(at).unwrap_or(u64::MAX) * 64 + u64::from(missing.trailing_zeros())
            })
        })
    }
}

/// The row group being read.
struct ReadGroup {
    /// Graph, table and row group number.
    key: (u32, Table, u64),
    /// The group's first row.
    start: u64,
    /// One past the last row of the last chunk of each (kind, namespace,
    /// column).
    ends: BTreeMap<(u8, u8, u32), u64>,
    /// The rows with a node or an edge.
    entities: RowBits,
    /// The rows with a property or history value.
    values: RowBits,
    /// The last history epoch of each (column, row) whose current value has
    /// not come yet.
    history_epochs: FxHashMap<(u32, u64), u64>,
}

/// The state of a read: the checked metadata, the graph stores and the group
/// being read.
struct Reader<'m, 's, S> {
    layout: Layout<'m>,
    graphs: Vec<GraphTarget<'s, S>>,
    group: Option<ReadGroup>,
}

impl<S: LoadTarget> Reader<'_, '_, S> {
    /// Reads the chunk at `index` of `data` (three for the edge columns);
    /// returns how many chunks it read.
    fn read(
        &mut self,
        source: &dyn SectionSource,
        data: &[ChunkMeta],
        index: usize,
    ) -> Result<usize> {
        let chunk = data[index];
        let place = format!(
            "LPG section, chunk {index} (graph {}, namespace {:?}, column {}, rows from {})",
            chunk.graph_id, chunk.namespace, chunk.column_id, chunk.row_start
        );
        let refuse = |what: String| Error::corruption(format!("{place}: {what}"));
        if !matches!(chunk.kind, ChunkKind::Column | ChunkKind::History) {
            return Err(refuse(format!(
                "a {:?} chunk, where the chunks before the metadata chunk are Column and \
                 History chunks",
                chunk.kind
            )));
        }
        let graph_id = chunk.graph_id;
        let position = *self
            .layout
            .positions
            .get(&graph_id)
            .ok_or_else(|| refuse(format!("graph {graph_id} is not in the metadata chunk")))?;
        let graph_meta = &self.layout.meta.graphs[position];
        let role = self
            .layout
            .role(position, chunk.namespace, chunk.column_id)
            .map_err(refuse)?;
        if chunk.kind == ChunkKind::History && !matches!(role, Role::Property { .. }) {
            return Err(refuse(
                "a history chunk of a fixed column, where only property columns have history"
                    .to_string(),
            ));
        }
        let table = role.table();
        let next_id = match table {
            Table::Node => graph_meta.next_node_id,
            Table::Edge => graph_meta.next_edge_id,
        };
        let last_row = self.check_rows(&chunk, next_id).map_err(refuse)?;
        let group_number = chunk.row_start / self.layout.max_rows;
        self.enter_group((graph_id, table, group_number))
            .map_err(refuse)?;
        self.check_order(&chunk, last_row).map_err(refuse)?;
        let graph_index = position;
        match role {
            Role::Labels => {
                let decoded =
                    decode(source, index, &chunk).map_err(|error| prefixed(&place, error))?;
                self.apply_labels(graph_index, &chunk, decoded)
                    .map_err(refuse)?;
                Ok(1)
            }
            Role::Endpoints => {
                let partners = [index + 1, index + 2].map(|at| data.get(at).copied());
                for (partner, column_id) in partners.iter().zip([COLUMN_TARGET, COLUMN_EDGE_TYPE]) {
                    let fits = partner.is_some_and(|partner| {
                        partner.kind == ChunkKind::Column
                            && partner.namespace == ChunkNamespace::EdgeStructure
                            && partner.column_id == column_id
                            && partner.graph_id == chunk.graph_id
                            && partner.row_start == chunk.row_start
                            && partner.row_count == chunk.row_count
                    });
                    if !fits {
                        return Err(refuse(format!(
                            "column {COLUMN_SOURCE} is not followed by column {COLUMN_TARGET} \
                             and column {COLUMN_EDGE_TYPE} of the same rows"
                        )));
                    }
                }
                let mut columns = Vec::with_capacity(3);
                for at in [index, index + 1, index + 2] {
                    columns.push(
                        decode(source, at, &data[at]).map_err(|error| prefixed(&place, error))?,
                    );
                }
                self.apply_edges(graph_index, &chunk, columns)
                    .map_err(refuse)?;
                Ok(3)
            }
            Role::Property { table, key } => {
                let decoded =
                    decode(source, index, &chunk).map_err(|error| prefixed(&place, error))?;
                self.apply_values(graph_index, &chunk, table, &key, decoded)
                    .map_err(refuse)?;
                Ok(1)
            }
        }
    }

    /// Checks rule 2 on the rows of `chunk`; returns its last row.
    fn check_rows(&self, chunk: &ChunkMeta, next_id: u64) -> std::result::Result<u64, String> {
        let max_rows = self.layout.max_rows;
        if chunk.row_count == 0 || u64::from(chunk.row_count) > max_rows {
            return Err(format!(
                "{} rows, where a chunk holds 1 to {max_rows} rows",
                chunk.row_count
            ));
        }
        let last_row = chunk
            .row_start
            .checked_add(u64::from(chunk.row_count) - 1)
            .ok_or_else(|| {
                format!(
                    "{} rows from row {} pass the id space",
                    chunk.row_count, chunk.row_start
                )
            })?;
        if chunk.row_start / max_rows != last_row / max_rows {
            return Err(format!(
                "rows {}..={last_row} cross a row group of {max_rows} rows",
                chunk.row_start
            ));
        }
        if last_row >= next_id {
            return Err(format!(
                "rows {}..={last_row} reach the table's next id {next_id}",
                chunk.row_start
            ));
        }
        Ok(last_row)
    }

    /// Moves to the group `key` (rule 3), closing the group before it.
    fn enter_group(&mut self, key: (u32, Table, u64)) -> std::result::Result<(), String> {
        if let Some(group) = &self.group {
            if group.key == key {
                return Ok(());
            }
            if group.key > key {
                let (graph, table, number) = group.key;
                return Err(format!(
                    "the chunk is out of order: it belongs to the {} table's row group {} of \
                     graph {}, after row group {number} of the {} table of graph {graph}",
                    key.1.entity(),
                    key.2,
                    key.0,
                    table.entity()
                ));
            }
        }
        self.close_group().map_err(|error| error.to_string())?;
        self.group = Some(ReadGroup {
            key,
            start: key.2 * self.layout.max_rows,
            ends: BTreeMap::new(),
            entities: RowBits::default(),
            values: RowBits::default(),
            history_epochs: FxHashMap::default(),
        });
        Ok(())
    }

    /// Checks rule 5 on the group being read and ends it.
    fn close_group(&mut self) -> Result<()> {
        let Some(group) = self.group.take() else {
            return Ok(());
        };
        if let Some(offset) = group.values.first_outside(&group.entities) {
            let (graph, table, _) = group.key;
            let entity = table.entity();
            let row = group.start + offset;
            return Err(Error::corruption(format!(
                "LPG section, graph {graph}: {entity} {row} has a property or history value, \
                 but its row group holds no {entity} {row}"
            )));
        }
        Ok(())
    }

    /// Checks rule 3's order within the group for `chunk`, which ends at
    /// `last_row`.
    fn check_order(&mut self, chunk: &ChunkMeta, last_row: u64) -> std::result::Result<(), String> {
        let group = self.group.as_mut().ok_or("no row group")?;
        let identity = (
            chunk.kind.to_byte(),
            chunk.namespace.to_byte(),
            chunk.column_id,
        );
        if let Some(end) = group.ends.get(&identity)
            && chunk.row_start < *end
        {
            return Err(format!(
                "rows {}..={last_row} overlap the chunk before it, which ends at row {}",
                chunk.row_start,
                end - 1
            ));
        }
        if chunk.kind == ChunkKind::History
            && let Some(end) = group.ends.get(&(
                ChunkKind::Column.to_byte(),
                chunk.namespace.to_byte(),
                chunk.column_id,
            ))
            && chunk.row_start < *end
        {
            return Err(format!(
                "a history chunk of rows {}..={last_row} comes after the column chunk that \
                 ends at row {}: older versions come before the value",
                chunk.row_start,
                end - 1
            ));
        }
        group.ends.insert(identity, last_row + 1);
        Ok(())
    }

    /// Creates the nodes of a labels chunk.
    fn apply_labels(
        &mut self,
        graph: usize,
        chunk: &ChunkMeta,
        decoded: ColumnChunk,
    ) -> std::result::Result<(), String> {
        no_epochs(&decoded, "the labels column")?;
        let names = &self.layout.names[graph].labels;
        let target = self.graphs[graph].store();
        let group = self.group.as_mut().ok_or("no row group")?;
        for (offset, value) in decoded.values {
            let row = chunk.row_start + u64::from(offset);
            let Value::String(text) = &value else {
                return Err(format!(
                    "node {row}: labels {value:?}, where labels are a string"
                ));
            };
            let labels =
                label_names(text, names).map_err(|error| format!("node {row}: {error}"))?;
            target
                .load_node(NodeId::new(row), &labels)
                .map_err(|error| format!("node {row}: {error}"))?;
            group.entities.set(row - group.start);
        }
        Ok(())
    }

    /// Creates the edges of the three edge columns.
    fn apply_edges(
        &mut self,
        graph: usize,
        chunk: &ChunkMeta,
        columns: Vec<ColumnChunk>,
    ) -> std::result::Result<(), String> {
        for (decoded, what) in columns.iter().zip([
            "the source column",
            "the target column",
            "the edge type column",
        ]) {
            no_epochs(decoded, what)?;
        }
        let [sources, targets, types]: [ColumnChunk; 3] = columns
            .try_into()
            .map_err(|_| "three edge columns".to_string())?;
        let same_rows = |other: &ColumnChunk| {
            other.values.len() == sources.values.len()
                && other
                    .values
                    .iter()
                    .zip(&sources.values)
                    .all(|((a, _), (b, _))| a == b)
        };
        if !same_rows(&targets) || !same_rows(&types) {
            return Err(format!(
                "columns {COLUMN_SOURCE}, {COLUMN_TARGET} and {COLUMN_EDGE_TYPE} hold values for \
                 different rows"
            ));
        }
        let edge_types = &self.layout.names[graph].edge_types;
        let target_store = self.graphs[graph].store();
        let group = self.group.as_mut().ok_or("no row group")?;
        for (((offset, source), (_, target)), (_, edge_type)) in sources
            .values
            .into_iter()
            .zip(targets.values)
            .zip(types.values)
        {
            let row = chunk.row_start + u64::from(offset);
            let id = |value: &Value, what: &str| match value {
                Value::Int64(id) => {
                    u64::try_from(*id).map_err(|_| format!("edge {row}: {what} {id} is negative"))
                }
                other => Err(format!(
                    "edge {row}: {what} {other:?}, where it is an Int64"
                )),
            };
            let source = id(&source, "source node")?;
            let target = id(&target, "target node")?;
            let type_id = id(&edge_type, "edge type")?;
            let name = u32::try_from(type_id)
                .ok()
                .and_then(|type_id| edge_types.get(&type_id))
                .ok_or_else(|| {
                    format!("edge {row}: edge type {type_id} is not an edge type of the graph")
                })?;
            // The row-group store leaves out an edge whose end the graph
            // does not hold (see `LoadTarget::load_edge`), and its values
            // with it; the row still counts as an edge of the file.
            target_store
                .load_edge(
                    EdgeId::new(row),
                    NodeId::new(source),
                    NodeId::new(target),
                    name,
                )
                .map_err(|error| format!("edge {row}: {error}"))?;
            group.entities.set(row - group.start);
        }
        Ok(())
    }

    /// Sets the values (or, for a history chunk, the older versions) of a
    /// property column chunk.
    fn apply_values(
        &mut self,
        graph: usize,
        chunk: &ChunkMeta,
        table: Table,
        key: &PropertyKey,
        decoded: ColumnChunk,
    ) -> std::result::Result<(), String> {
        let target = self.graphs[graph].store();
        let group = self.group.as_mut().ok_or("no row group")?;
        let entity = table.entity();
        let column_id = chunk.column_id;
        if chunk.kind == ChunkKind::History {
            no_epochs(&decoded, "a history chunk")?;
            for (offset, value) in decoded.values {
                let row = chunk.row_start + u64::from(offset);
                let versions =
                    history_versions(value).map_err(|error| format!("{entity} {row}: {error}"))?;
                if let Some((epoch, _)) = versions.last() {
                    group.history_epochs.insert((column_id, row), *epoch);
                }
                for (epoch, value) in versions {
                    target.load_old_value(
                        table.entity_at(row),
                        key.as_str(),
                        value,
                        EpochId::new(epoch),
                    );
                }
                group.values.set(row - group.start);
            }
            return Ok(());
        }
        let epochs = decoded.epochs.unwrap_or_default();
        for (index, (offset, value)) in decoded.values.into_iter().enumerate() {
            let row = chunk.row_start + u64::from(offset);
            if value.is_null() {
                return Err(format!(
                    "{entity} {row}: a null value, where a removed property has no value"
                ));
            }
            let epoch = epochs.get(index).copied().unwrap_or(0);
            if epoch > MAX_EPOCH {
                return Err(format!(
                    "{entity} {row}: epoch {epoch} is above {MAX_EPOCH}"
                ));
            }
            if let Some(last) = group.history_epochs.remove(&(column_id, row))
                && epoch < last
            {
                return Err(format!(
                    "{entity} {row}: the value's epoch {epoch} is before the last epoch {last} \
                     of its history"
                ));
            }
            target
                .load_value(
                    table.entity_at(row),
                    key.as_str(),
                    value,
                    Some(EpochId::new(epoch)),
                )
                .map_err(|error| format!("{entity} {row}: {error}"))?;
            group.values.set(row - group.start);
        }
        Ok(())
    }
}

/// Fetches and decodes the column chunk at `index`.
fn decode(source: &dyn SectionSource, index: usize, chunk: &ChunkMeta) -> Result<ColumnChunk> {
    let bytes: Bytes = source.fetch(index)?;
    decode_column_chunk_bytes(&bytes, chunk.codec, chunk.row_count)
}

/// `error` with `place` before its message, when it is a decoding error
/// (damage, or what a newer release wrote).
fn prefixed(place: &str, error: Error) -> Error {
    match error {
        Error::Serialization(_) | Error::Corruption(_) => error.wrapped(place),
        other => other,
    }
}

/// Refuses epochs on a chunk that carries none (fixed columns, history).
fn no_epochs(chunk: &ColumnChunk, what: &str) -> std::result::Result<(), String> {
    if chunk.epochs.is_some() {
        return Err(format!(
            "{what} carries epochs, which only property values have"
        ));
    }
    Ok(())
}

/// The label names of a labels value: decimal ids of the graph's `labels`,
/// ascending, joined by `,` (empty for none).
fn label_names<'n>(
    text: &str,
    labels: &'n FxHashMap<u32, String>,
) -> std::result::Result<Vec<&'n str>, String> {
    if text.is_empty() {
        return Ok(Vec::new());
    }
    let mut names = Vec::new();
    let mut last: Option<u32> = None;
    for part in text.split(',') {
        let canonical = !part.is_empty()
            && part.bytes().all(|byte| byte.is_ascii_digit())
            && (part == "0" || !part.starts_with('0'));
        let id = canonical
            .then(|| part.parse::<u32>().ok())
            .flatten()
            .ok_or_else(|| format!("labels {text:?}: {part:?} is not a label id"))?;
        if last.is_some_and(|last| last >= id) {
            return Err(format!("labels {text:?}: label ids are not ascending"));
        }
        last = Some(id);
        let name = labels
            .get(&id)
            .ok_or_else(|| format!("label {id} is not a label of the graph"))?;
        names.push(name.as_str());
    }
    Ok(names)
}

/// The versions of a history value: `[epoch, value]` lists, epochs from 0
/// and never going back.
fn history_versions(value: Value) -> std::result::Result<Vec<(u64, Value)>, String> {
    let shape = || "a history value is a non-empty list of [epoch, value] lists".to_string();
    let Value::List(items) = value else {
        return Err(shape());
    };
    if items.is_empty() {
        return Err(shape());
    }
    let mut versions: Vec<(u64, Value)> = Vec::new();
    for item in items.iter() {
        let Value::List(pair) = item else {
            return Err(shape());
        };
        let [Value::Int64(epoch), value] = &pair[..] else {
            return Err(shape());
        };
        let epoch =
            u64::try_from(*epoch).map_err(|_| format!("history epoch {epoch} is negative"))?;
        if let Some((last, _)) = versions.last()
            && epoch < *last
        {
            return Err(format!("history epochs go back from {last} to {epoch}"));
        }
        versions.push((epoch, value.clone()));
    }
    Ok(versions)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;
    use grafeo_common::storage::{
        ChunkCaps, ChunkKind, ChunkMeta, ChunkNamespace, ImageSource, MemoryImage, SectionSink,
        SectionSource, SectionType,
    };
    use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, Value};
    use grafeo_common::utils::error::{Error, Result};

    use super::{
        COLUMN_EDGE_TYPE, COLUMN_LABELS, COLUMN_SOURCE, COLUMN_TARGET, DictionaryMeta, GraphMeta,
        LPG_SECTION_VERSION, LpgMeta, decode_lpg_meta, encode_lpg_meta, read_lpg_chunks,
    };
    use crate::graph::load::testing::version_3_fixture;
    use crate::graph::rowgroup::RowGroupStore;
    use crate::graph::{GraphStore, GraphStoreMut};

    // ── Sections 0.6.0-beta.1 wrote (fixtures) ──────────────────────────

    /// Loads the section of the fixture `name` into a new root.
    fn load_fixture(name: &str) -> RowGroupStore {
        let image = version_3_fixture(name);
        let source = image
            .section_source(SectionType::LpgStore)
            .expect("the fixture holds a section");
        assert_eq!(source.section_version(), LPG_SECTION_VERSION);
        let root = RowGroupStore::new();
        read_lpg_chunks(&root, &*source).unwrap_or_else(|error| panic!("{name}: {error}"));
        root
    }

    /// The value codec's bytes of `value`: equal bytes, equal values (floats
    /// by their bits, map and counter entries in order).
    fn bits(value: &Value) -> Vec<u8> {
        let mut bytes = Vec::new();
        grafeo_common::storage::value_codec::encode_value(value, &mut bytes).unwrap();
        bytes
    }

    /// The properties of a map, by key, as codec bytes.
    fn property_bits(properties: &grafeo_common::types::PropertyMap) -> Vec<(String, Vec<u8>)> {
        properties
            .to_btree_map()
            .iter()
            .map(|(key, value)| (key.as_str().to_string(), bits(value)))
            .collect()
    }

    fn sorted_labels(node: &crate::graph::lpg::Node) -> Vec<String> {
        let mut labels: Vec<String> = node.labels.iter().map(ToString::to_string).collect();
        labels.sort();
        labels
    }

    /// Asserts that `a` and `b` hold the same nodes and edges (labels,
    /// endpoints, types and property values, floats by bits) and next ids,
    /// in the default graph and in every named graph.
    fn assert_same_graph(a: &RowGroupStore, b: &RowGroupStore) {
        let mut names_a = a.graph_names();
        let mut names_b = b.graph_names();
        names_a.sort();
        names_b.sort();
        assert_eq!(names_a, names_b, "named graphs");
        assert_same_store(a, b, "the default graph");
        for name in names_a {
            assert_same_store(
                &a.graph(&name).unwrap(),
                &b.graph(&name).unwrap(),
                &format!("graph {name:?}"),
            );
        }
    }

    fn assert_same_store(a: &RowGroupStore, b: &RowGroupStore, graph: &str) {
        assert_eq!(a.node_ids(), b.node_ids(), "{graph}: node ids");
        for id in a.node_ids() {
            let (x, y) = (a.get_node(id).unwrap(), b.get_node(id).unwrap());
            assert_eq!(
                sorted_labels(&x),
                sorted_labels(&y),
                "{graph}: node {}",
                id.0
            );
            assert_eq!(
                property_bits(&x.properties),
                property_bits(&y.properties),
                "{graph}: properties of node {}",
                id.0
            );
        }
        let edges = |store: &RowGroupStore| -> Vec<_> {
            let mut edges: Vec<_> = store
                .all_edges()
                .map(|edge| {
                    (
                        edge.id,
                        edge.src,
                        edge.dst,
                        edge.edge_type.to_string(),
                        property_bits(&edge.properties),
                    )
                })
                .collect();
            edges.sort_by_key(|edge| edge.0);
            edges
        };
        assert_eq!(edges(a), edges(b), "{graph}: edges");
        assert_eq!(
            (a.next_node_id(), a.next_edge_id()),
            (b.next_node_id(), b.next_edge_id()),
            "{graph}: the next ids"
        );
    }

    /// One value of every kind a property can hold, by key.
    fn value_kinds() -> Vec<(&'static str, Value)> {
        use grafeo_common::types::{Date, Duration, Time, Timestamp, ZonedDatetime};
        use std::collections::HashMap;

        let list = |items: Vec<Value>| Value::List(Arc::from(items));
        let map = |entries: Vec<(&str, Value)>| {
            Value::Map(Arc::new(
                entries
                    .into_iter()
                    .map(|(key, value)| (PropertyKey::new(key), value))
                    .collect(),
            ))
        };
        let counter = |entries: &[(&str, u64)]| {
            Arc::new(
                entries
                    .iter()
                    .map(|(replica, count)| ((*replica).to_string(), *count))
                    .collect::<HashMap<_, _>>(),
            )
        };
        let instant = Timestamp::from_micros(1_696_500_000_123_457);
        vec![
            ("bool", Value::Bool(true)),
            ("int", Value::Int64(-19)),
            ("count", Value::Int64(88)),
            ("nan", Value::Float64(f64::from_bits(0x7FF8_0000_0000_0058))),
            ("zero", Value::Float64(-0.0)),
            ("name", Value::from("Amsterdam")),
            ("empty", Value::from("")),
            ("long", Value::from("Prague ".repeat(19))),
            ("bytes", Value::Bytes(Arc::from(vec![3u8, 19, 88]))),
            ("date", Value::Date(Date::from_days(-3))),
            ("time", Value::Time(Time::from_nanos(88).unwrap())),
            (
                "zoned_time",
                Value::Time(
                    Time::from_nanos(3_600_000_000_019)
                        .unwrap()
                        .with_offset(3600),
                ),
            ),
            ("timestamp", Value::Timestamp(instant)),
            (
                "zoned",
                Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(instant, 7200)),
            ),
            ("duration", Value::Duration(Duration::new(3, 19, 88))),
            (
                "list",
                list(vec![
                    Value::Int64(3),
                    list(vec![Value::from("Berlin")]),
                    Value::Null,
                ]),
            ),
            (
                "map",
                map(vec![
                    ("city", Value::from("Prague")),
                    ("stops", list(vec![Value::Int64(19)])),
                ]),
            ),
            (
                "vector",
                Value::Vector(Arc::from(vec![3.0f32, -19.5, f32::NAN])),
            ),
            (
                "path",
                Value::Path {
                    nodes: Arc::from(vec![map(vec![("_id", Value::Int64(3))])]),
                    edges: Arc::from(Vec::<Value>::new()),
                },
            ),
            (
                "visits",
                Value::GCounter(counter(&[("Alix", 3), ("Gus", 19)])),
            ),
            (
                "balance",
                Value::OnCounter {
                    pos: counter(&[("Mia", 88)]),
                    neg: counter(&[("Jules", 3)]),
                },
            ),
        ]
    }

    /// The graph of the `kinds` fixtures: every value kind as properties,
    /// nodes with 0, 1 and 3 labels, edges of two types with properties,
    /// and named graphs "trips", "" and "travel/__default__" (the last one
    /// empty).
    fn kinds_store() -> RowGroupStore {
        let store = RowGroupStore::new();
        let kinds = value_kinds();
        let mut nodes = Vec::new();
        for (i, (key, value)) in kinds.iter().enumerate() {
            let labels: &[&str] = match i % 3 {
                0 => &[],
                1 => &["Person"],
                _ => &["Person", "Employee", "Traveller"],
            };
            let id = store.create_node(labels);
            store.set_node_property(id, key, value.clone());
            store.set_node_property(id, "seen", Value::Int64(i64::try_from(i).unwrap()));
            nodes.push(id);
        }
        // One key with an Int64 and a Float64 value.
        store.set_node_property(nodes[0], "mixed", Value::Int64(3));
        store.set_node_property(nodes[1], "mixed", Value::Float64(19.88));
        for (i, pair) in nodes.windows(2).enumerate() {
            let edge_type = if i % 2 == 0 { "KNOWS" } else { "VISITED" };
            let edge = store.create_edge(pair[0], pair[1], edge_type);
            let (key, value) = &kinds[i];
            store.set_edge_property(edge, key, value.clone());
        }
        for name in ["trips", "", "travel/__default__"] {
            store.create_graph(name).unwrap();
        }
        let trips = store.graph("trips").unwrap();
        let berlin = trips.create_node(&["City"]);
        let paris = trips.create_node(&["City", "Capital"]);
        trips.set_node_property(paris, "population", Value::Int64(2_100_000));
        let route = trips.create_edge(berlin, paris, "ROUTE");
        trips.set_edge_property(route, "hours", Value::Float64(8.8));
        let unnamed = store.graph("").unwrap();
        let prague = unnamed.create_node(&["City"]);
        unnamed.set_node_property(prague, "name", Value::from("Prague"));
        store
    }

    /// Sections of every value kind, written with caps of 4 rows and 512
    /// bytes, 1 row and 64 bytes and the default caps, load the same rows,
    /// values (floats by their bits) and next ids.
    #[test]
    fn every_value_kind_loads_at_every_cap() {
        let expected = kinds_store();
        for name in [
            "kinds_caps_4_512.chunks",
            "kinds_caps_1_64.chunks",
            "kinds.chunks",
        ] {
            let root = load_fixture(name);
            assert_same_graph(&expected, &root);
            assert_eq!(root.stamped_vectors(), 0, "{name}: the load is collected");
        }
    }

    /// Each graph keeps the labels and edge types it registered that no row
    /// of it uses (a deleted node's label, a deleted edge's type), with
    /// their ids, in the default graph and in a named one.
    #[test]
    fn names_no_row_uses_load_with_their_ids() {
        let root = load_fixture("unused_names.chunks");
        let names = |pairs: &[(u32, &str)]| -> Vec<(u32, String)> {
            pairs
                .iter()
                .map(|(id, name)| (*id, (*name).to_string()))
                .collect()
        };
        let [labels, edge_types, _] = root.dictionaries();
        assert_eq!(labels.0, names(&[(0, "Person"), (1, "Ghost"), (2, "City")]));
        assert_eq!(edge_types.0, names(&[(0, "OLD"), (1, "KNOWS")]));
        let trips = root.graph("trips").unwrap();
        let [labels, edge_types, _] = trips.dictionaries();
        assert_eq!(
            labels.0,
            names(&[(0, "City"), (1, "Museum"), (2, "Person")])
        );
        assert_eq!(edge_types.0, names(&[(0, "FERRY"), (1, "KNOWS")]));
        assert_eq!((root.node_count(), root.edge_count()), (2, 1));
        assert_eq!((trips.node_count(), trips.edge_count()), (2, 1));
        assert_eq!(
            root.next_node_id(),
            3,
            "the deleted node's id stays retired"
        );
    }

    /// Graph ids load as written: a gap where a dropped graph was, and the
    /// next graph id past it, so a graph created again under the dropped
    /// name gets a new id.
    #[test]
    fn graph_ids_load_with_their_gaps() {
        let root = load_fixture("dropped_graph.chunks");
        let mut ids: Vec<(u32, String)> = root
            .graph_names()
            .into_iter()
            .map(|name| (root.graph(&name).unwrap().graph_id(), name))
            .collect();
        ids.sort();
        assert_eq!(ids, [(1, "trips".to_string()), (3, "archive".to_string())]);
        assert_eq!(root.next_graph_id(), 4);
        assert_eq!(root.graph("trips").unwrap().node_count(), 1);
        root.create_graph("travel").unwrap();
        assert_eq!(root.graph("travel").unwrap().graph_id(), 4);
    }

    /// Ids across row groups of 4 rows load with their gaps (3 and 4 cross a
    /// group's end, 8 to 11 are a whole group), and so do ids far above 0.
    #[test]
    fn sparse_ids_load_across_row_groups() {
        let expected = RowGroupStore::new();
        let ids: Vec<NodeId> = (0..40).map(|_| expected.create_node(&["Person"])).collect();
        for id in [3, 4, 7, 8, 9, 10, 11] {
            assert!(expected.delete_node(ids[id]));
        }
        expected.set_node_property(ids[39], "city", Value::from("Berlin"));
        assert_same_graph(&expected, &load_fixture("sparse_caps_4.chunks"));

        let far = load_fixture("far_ids_caps_4.chunks");
        assert_eq!(far.node_ids(), [NodeId::new(1_000)]);
        assert_eq!(far.nodes_by_label("City"), [NodeId::new(1_000)]);
        assert_eq!(far.next_node_id(), 1_001);
    }

    /// The ids of deleted nodes and edges are never given again after a load.
    #[test]
    fn retired_ids_stay_retired() {
        let root = load_fixture("retired_ids.chunks");
        assert_eq!(root.node_count(), 3);
        assert_eq!(
            root.create_node(&["Jules"]),
            NodeId::new(5),
            "3 and 4 stay retired"
        );
        assert_eq!(
            root.create_edge(NodeId::new(0), NodeId::new(2), "KNOWS"),
            EdgeId::new(2),
            "edge 1 stays retired"
        );
    }

    /// A section a build with `temporal` wrote holds older versions in
    /// `History` chunks and its epoch in the metadata: the load checks them,
    /// keeps the current values (a property removed last has none) and
    /// continues at the section's epoch.
    #[test]
    fn a_section_with_history_loads_its_current_values() {
        let city = PropertyKey::new("city");
        for name in ["history_caps_1_64.chunks", "history.chunks"] {
            let root = load_fixture(name);
            let (alix, gus) = (NodeId::new(0), NodeId::new(1));
            assert_eq!(
                root.get_node_property(alix, &city),
                Some(Value::from("Paris")),
                "{name}"
            );
            assert_eq!(
                root.get_node_property(alix, &PropertyKey::new("deep")),
                Some(Value::Int64(19)),
                "{name}"
            );
            assert_eq!(root.get_node_property(gus, &city), None, "{name}: removed");
            assert_eq!(
                root.get_edge_property(EdgeId::new(0), &PropertyKey::new("since")),
                Some(Value::Int64(19)),
                "{name}"
            );
            assert_eq!(root.current_epoch(), EpochId::new(88), "{name}");
        }
    }

    /// Serves an image's chunks and records each fetch.
    struct Counting<'s> {
        inner: Box<dyn SectionSource + 's>,
        fetched: std::cell::RefCell<Vec<usize>>,
    }

    impl SectionSource for Counting<'_> {
        fn chunks(&self) -> &[ChunkMeta] {
            self.inner.chunks()
        }

        fn fetch(&self, index: usize) -> Result<Bytes> {
            self.fetched.borrow_mut().push(index);
            self.inner.fetch(index)
        }

        fn stored_length(&self, index: usize) -> Result<u64> {
            self.inner.stored_length(index)
        }

        fn section_version(&self) -> u8 {
            self.inner.section_version()
        }
    }

    #[test]
    fn the_reader_fetches_the_metadata_then_each_chunk_once_in_order() {
        let image = version_3_fixture("kinds_caps_4_512.chunks");
        let source = Counting {
            inner: image.section_source(SectionType::LpgStore).unwrap(),
            fetched: std::cell::RefCell::new(Vec::new()),
        };
        let count = source.chunks().len();
        assert!(count > 3, "{count} chunks");
        read_lpg_chunks(&RowGroupStore::new(), &source).unwrap();
        let mut expected = vec![count - 1];
        expected.extend(0..count - 1);
        assert_eq!(*source.fetched.borrow(), expected);
    }

    // ── Metadata and crafted sections ───────────────────────────────────

    /// A metadata chunk alone (no rows), loaded into a new store.
    fn load_meta(meta: &LpgMeta) -> RowGroupStore {
        let mut image = MemoryImage::new();
        image
            .begin_section(SectionType::LpgStore, LPG_SECTION_VERSION)
            .unwrap();
        image
            .write_chunk(ChunkMeta::meta(), &encode_lpg_meta(meta).unwrap())
            .unwrap();
        let source = image.section_source(SectionType::LpgStore).unwrap();
        let store = RowGroupStore::new();
        read_lpg_chunks(&store, &*source).unwrap();
        store
    }

    /// A load into a store that holds graphs already (an earlier step of the
    /// open creates some, as the catalog does for a schema's graphs) gives a
    /// graph the file lists its id from the file, whatever id this process
    /// gave it, and moves one the file does not list past the file's ids.
    #[test]
    fn a_load_gives_graphs_created_before_it_the_file_s_ids() {
        let mut crafted = crafted_meta();
        crafted.next_graph_id = 3;
        crafted.graphs.push(named(1, "social"));
        crafted.graphs.push(named(2, "reporting"));
        let mut image = MemoryImage::new();
        image
            .begin_section(SectionType::LpgStore, LPG_SECTION_VERSION)
            .unwrap();
        image
            .write_chunk(ChunkMeta::meta(), &encode_lpg_meta(&crafted).unwrap())
            .unwrap();
        let source = image.section_source(SectionType::LpgStore).unwrap();

        let store = RowGroupStore::new();
        store.create_graph("reporting").unwrap();
        store.create_graph("scratch").unwrap();
        let reporting = store.graph("reporting").unwrap();
        assert_eq!(reporting.graph_id(), 1, "before the load");
        read_lpg_chunks(&store, &*source).unwrap();
        assert_eq!(
            (
                store.graph("social").unwrap().graph_id(),
                reporting.graph_id(),
                store.graph("scratch").unwrap().graph_id()
            ),
            (1, 2, 3),
            "the file's ids, and the unlisted graph past them"
        );
        assert!(
            Arc::ptr_eq(&reporting, &store.graph("reporting").unwrap()),
            "the graph created before the load is the one loaded"
        );
        assert_eq!(store.next_graph_id(), 4);
    }

    /// A file's ids may have gaps (ids no name holds): they load as they
    /// are, and a new name gets the next id the file names, never a gap's.
    #[test]
    fn ids_with_gaps_load_and_are_never_given_again() {
        let mut crafted = crafted_meta();
        crafted.graphs[0].labels = DictionaryMeta {
            next_id: 10,
            names: vec![
                (0, "Person".into()),
                (5, "City".into()),
                (9, "Museum".into()),
            ],
        };
        crafted.graphs[0].keys = DictionaryMeta {
            next_id: 12,
            names: vec![(3, "name".into())],
        };
        let store = load_meta(&crafted);
        let alix = store.create_node(&["Person", "Station"]);
        store.set_node_property(alix, "age", Value::Int64(19));
        let [labels, _, keys] = store.dictionaries();
        assert_eq!(
            labels,
            (
                vec![
                    (0, "Person".to_string()),
                    (5, "City".into()),
                    (9, "Museum".into()),
                    (10, "Station".into())
                ],
                11
            ),
            "Station after the file's next id"
        );
        assert_eq!(
            keys,
            (vec![(3, "name".to_string()), (12, "age".into())], 13),
            "age after the file's next id"
        );
    }

    #[test]
    fn the_metadata_decoder_refuses_crafted_lengths_layouts_and_trailing_bytes() {
        let mut meta = crafted_meta();
        meta.graphs[0].labels = DictionaryMeta {
            next_id: 2,
            names: vec![(0, "Person".into()), (1, "City".into())],
        };
        let bytes = encode_lpg_meta(&meta).unwrap();
        assert_eq!(decode_lpg_meta(&bytes).unwrap(), meta);
        // Layout, max_rows, max_bytes, epoch and the next graph id take 21
        // bytes, the graph count 4; graph 0 then has its id, its empty name
        // (a length), its next node and edge ids (29 bytes up to its labels),
        // then the labels' next id, their count, the first label's id and its
        // name's length.
        let at_count = 21 + 4 + 4 + 4 + 8 + 8 + 4;
        let at_length = at_count + 4 + 4;
        // Refused at the count or length itself, before anything sized by it
        // is allocated.
        for (at, what, words) in [
            (at_count, "label count", "4294967295 labels, but only"),
            (
                at_length,
                "label length",
                "a label name of 4294967295 bytes, but only",
            ),
        ] {
            let mut crafted = bytes.clone();
            crafted[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
            let error = decode_lpg_meta(&crafted).unwrap_err();
            assert!(matches!(error, Error::Corruption(_)), "{what}: {error:?}");
            let error = error.to_string();
            assert!(
                error.contains(words) && error.contains(&format!("byte {at}")),
                "{what}: {error}"
            );
        }
        for cut in 0..bytes.len() {
            assert!(decode_lpg_meta(&bytes[..cut]).is_err(), "cut at {cut}");
        }

        let mut layout = bytes.clone();
        layout[0] = 2;
        let error = decode_lpg_meta(&layout).unwrap_err().to_string();
        assert!(error.contains("layout 2"), "{error}");
        let mut utf8 = bytes.clone();
        utf8[at_length + 4] = 0xFF;
        let error = decode_lpg_meta(&utf8).unwrap_err().to_string();
        assert!(error.contains("UTF-8"), "{error}");
        let mut trailing = bytes;
        trailing.push(0);
        let error = decode_lpg_meta(&trailing).unwrap_err().to_string();
        assert!(error.contains("after the metadata"), "{error}");
    }

    /// The metadata chunk has no fixed size limit: its lists are read with
    /// every count and length checked against the bytes left, so it holds
    /// as many names as the store has (the bincode limit of 64 MiB refused
    /// some millions of names, and every checkpoint after them).
    #[test]
    fn a_metadata_chunk_past_the_old_size_limit_round_trips() {
        let mut meta = crafted_meta();
        meta.graphs[0].labels = DictionaryMeta {
            next_id: 1_000_000,
            names: (0..1_000_000u32).map(|i| (i, format!("L{i:07}"))).collect(),
        };
        meta.graphs[0].edge_types = DictionaryMeta {
            next_id: 66,
            names: (0..66u8)
                .map(|i| (u32::from(i), format!("{i:02}{}", "x".repeat(1 << 20))))
                .collect(),
        };
        let bytes = encode_lpg_meta(&meta).unwrap();
        assert!(bytes.len() > 1 << 26, "{} bytes", bytes.len());
        assert_eq!(decode_lpg_meta(&bytes).unwrap(), meta);
    }
    /// The metadata chunk of a crafted section: caps of 4 rows and 1 MiB, one
    /// graph (next node id 4, next edge id 2), label "Person", edge type
    /// "KNOWS" and node property column 16 "name".
    /// The id of property key "name" in [`crafted_meta`].
    const NAME: u32 = 0;

    /// A named graph `name` with the id `id`, without nodes, edges or names.
    fn named(id: u32, name: &str) -> GraphMeta {
        GraphMeta {
            id,
            name: name.into(),
            next_node_id: 0,
            next_edge_id: 0,
            labels: DictionaryMeta::default(),
            edge_types: DictionaryMeta::default(),
            keys: DictionaryMeta::default(),
        }
    }

    fn crafted_meta() -> LpgMeta {
        let one = |name: &str| DictionaryMeta {
            next_id: 1,
            names: vec![(0, name.into())],
        };
        LpgMeta {
            layout: 1,
            max_rows: 4,
            max_bytes: 1 << 20,
            epoch: 0,
            next_graph_id: 1,
            graphs: vec![GraphMeta {
                id: 0,
                name: String::new(),
                next_node_id: 4,
                next_edge_id: 2,
                labels: one("Person"),
                edge_types: one("KNOWS"),
                keys: one("name"),
            }],
        }
    }

    /// One chunk of a crafted LPG section.
    enum Crafted {
        /// The metadata chunk of [`crafted_meta`].
        Meta,
        /// A metadata chunk of other metadata.
        MetaOf(LpgMeta),
        /// A labels chunk of graph 0 (node structure column 0).
        Labels {
            row_start: u64,
            row_count: u32,
            rows: Vec<(u32, &'static str)>,
        },
        /// A chunk of node property column [`NAME`] of graph 0, every row
        /// "Alix".
        Name {
            row_start: u64,
            row_count: u32,
            rows: Vec<u32>,
        },
        /// A chunk of edge structure column `column_id` of graph 0 with Int64
        /// values.
        Column {
            column_id: u32,
            row_start: u64,
            row_count: u32,
            rows: Vec<(u32, i64)>,
        },
        /// Any chunk: its kind, namespace, graph, column and rows from
        /// `meta`, its values and epochs encoded.
        Raw {
            meta: ChunkMeta,
            values: Vec<(u32, Value)>,
            epochs: Option<Vec<u64>>,
        },
    }

    /// The bytes of a column chunk of `values`, and its entry with `kind`
    /// in `namespace`.
    #[expect(clippy::too_many_arguments, reason = "one argument per entry field")]
    fn crafted_chunk(
        kind: ChunkKind,
        namespace: ChunkNamespace,
        graph: u32,
        column: u32,
        row_start: u64,
        row_count: u32,
        values: &[(u32, Value)],
        epochs: Option<&[u64]>,
    ) -> (ChunkMeta, Vec<u8>) {
        let (codec, bytes) =
            crate::codec::column_chunk::encode_column_chunk(row_count, values, epochs).unwrap();
        let meta = if kind == ChunkKind::History {
            ChunkMeta::history(graph, column, row_start, row_count, codec.to_byte())
        } else {
            ChunkMeta::column(graph, column, row_start, row_count, codec.to_byte())
        };
        (meta.in_namespace(namespace), bytes)
    }

    /// Writes `chunks` in order into a memory image (LpgStore, version 3) and
    /// loads them into a new store.
    fn load_crafted(chunks: Vec<Crafted>) -> Result<()> {
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::LpgStore, LPG_SECTION_VERSION)?;
        for chunk in chunks {
            let (meta, bytes) = match chunk {
                Crafted::Meta => (ChunkMeta::meta(), encode_lpg_meta(&crafted_meta()).unwrap()),
                Crafted::MetaOf(meta) => (ChunkMeta::meta(), encode_lpg_meta(&meta).unwrap()),
                Crafted::Labels {
                    row_start,
                    row_count,
                    rows,
                } => {
                    let values: Vec<(u32, Value)> = rows
                        .into_iter()
                        .map(|(row, labels)| (row, Value::from(labels)))
                        .collect();
                    crafted_chunk(
                        ChunkKind::Column,
                        ChunkNamespace::NodeStructure,
                        0,
                        COLUMN_LABELS,
                        row_start,
                        row_count,
                        &values,
                        None,
                    )
                }
                Crafted::Name {
                    row_start,
                    row_count,
                    rows,
                } => {
                    let values: Vec<(u32, Value)> = rows
                        .into_iter()
                        .map(|row| (row, Value::from("Alix")))
                        .collect();
                    crafted_chunk(
                        ChunkKind::Column,
                        ChunkNamespace::NodeProperties,
                        0,
                        NAME,
                        row_start,
                        row_count,
                        &values,
                        None,
                    )
                }
                Crafted::Column {
                    column_id,
                    row_start,
                    row_count,
                    rows,
                } => {
                    let values: Vec<(u32, Value)> = rows
                        .into_iter()
                        .map(|(row, value)| (row, Value::Int64(value)))
                        .collect();
                    crafted_chunk(
                        ChunkKind::Column,
                        ChunkNamespace::EdgeStructure,
                        0,
                        column_id,
                        row_start,
                        row_count,
                        &values,
                        None,
                    )
                }
                Crafted::Raw {
                    meta,
                    values,
                    epochs,
                } => {
                    let (encoded, bytes) = crafted_chunk(
                        meta.kind,
                        meta.namespace,
                        meta.graph_id,
                        meta.column_id,
                        meta.row_start,
                        meta.row_count,
                        &values,
                        epochs.as_deref(),
                    );
                    (
                        ChunkMeta {
                            codec: encoded.codec,
                            ..meta
                        },
                        bytes,
                    )
                }
            };
            image.write_chunk(meta, &bytes)?;
        }
        let source = image
            .section_source(SectionType::LpgStore)
            .expect("crafted chunks");
        read_lpg_chunks(&RowGroupStore::new(), &*source)
    }

    /// `[epoch, value]` versions as a history value.
    fn history_value(versions: &[(i64, Value)]) -> Value {
        Value::List(Arc::from(
            versions
                .iter()
                .map(|(epoch, value)| {
                    Value::List(Arc::from(vec![Value::Int64(*epoch), value.clone()]))
                })
                .collect::<Vec<_>>(),
        ))
    }

    /// The three edge columns of rows `rows` (source, target, type).
    fn edge_columns(rows: Vec<(u32, i64, i64, i64)>, row_count: u32) -> Vec<Crafted> {
        let column = |column_id: u32, pick: fn(&(u32, i64, i64, i64)) -> i64| Crafted::Column {
            column_id,
            row_start: 0,
            row_count,
            rows: rows.iter().map(|row| (row.0, pick(row))).collect(),
        };
        vec![
            column(COLUMN_SOURCE, |row| row.1),
            column(COLUMN_TARGET, |row| row.2),
            column(COLUMN_EDGE_TYPE, |row| row.3),
        ]
    }

    #[test]
    fn crafted_chunk_sequences_are_refused() {
        use Crafted::{Column, Labels, Meta, MetaOf, Name, Raw};

        let alix = || Labels {
            row_start: 0,
            row_count: 1,
            rows: vec![(0, "0")],
        };
        let history = |column: u32, values: Vec<(u32, Value)>, epochs: Option<Vec<u64>>| Raw {
            meta: ChunkMeta::history(0, column, 0, 1, 0)
                .in_namespace(ChunkNamespace::NodeProperties),
            values,
            epochs,
        };
        let column = |column: u32, values: Vec<(u32, Value)>, epochs: Option<Vec<u64>>| Raw {
            meta: ChunkMeta::column(0, column, 0, 1, 0)
                .in_namespace(ChunkNamespace::NodeProperties),
            values,
            epochs,
        };
        let with = |change: fn(&mut LpgMeta)| {
            let mut meta = crafted_meta();
            change(&mut meta);
            MetaOf(meta)
        };
        let two_nodes = || Labels {
            row_start: 0,
            row_count: 2,
            rows: vec![(0, "0"), (1, "0")],
        };
        let mut cases: Vec<(&str, Vec<Crafted>, &str)> = vec![
            ("no metadata chunk", vec![alix()], "metadata"),
            (
                "the metadata chunk before the others",
                vec![Meta, alix()],
                "metadata",
            ),
            (
                "a property without its node",
                vec![
                    alix(),
                    Name {
                        row_start: 0,
                        row_count: 3,
                        rows: vec![0, 2],
                    },
                    Meta,
                ],
                "node 2",
            ),
            (
                "an edge source without its target and type",
                vec![
                    two_nodes(),
                    Column {
                        column_id: 1,
                        row_start: 0,
                        row_count: 1,
                        rows: vec![(0, 0)],
                    },
                    Meta,
                ],
                "column 2",
            ),
            (
                "an edge target without its source",
                vec![
                    two_nodes(),
                    Column {
                        column_id: 2,
                        row_start: 0,
                        row_count: 1,
                        rows: vec![(0, 1)],
                    },
                    Meta,
                ],
                "column 1",
            ),
            (
                "overlapping chunks of one column",
                vec![
                    Labels {
                        row_start: 0,
                        row_count: 3,
                        rows: vec![(0, "0")],
                    },
                    Labels {
                        row_start: 2,
                        row_count: 2,
                        rows: vec![(0, "0")],
                    },
                    Meta,
                ],
                "overlap",
            ),
            (
                "an unknown edge structure column",
                vec![
                    Column {
                        column_id: 99,
                        row_start: 0,
                        row_count: 1,
                        rows: vec![(0, 3)],
                    },
                    Meta,
                ],
                "column 99 of the edge structure",
            ),
            (
                "an unknown property column",
                vec![
                    alix(),
                    column(99, vec![(0, Value::from("Gus"))], None),
                    Meta,
                ],
                "column 99 is no property key of the graph",
            ),
            (
                "a node structure column other than the labels",
                vec![
                    Raw {
                        meta: ChunkMeta::column(0, COLUMN_SOURCE, 0, 1, 0)
                            .in_namespace(ChunkNamespace::NodeStructure),
                        values: vec![(0, Value::Int64(3))],
                        epochs: None,
                    },
                    Meta,
                ],
                "column 1 of the node structure",
            ),
            (
                "a chunk in the section's own namespace",
                vec![
                    Raw {
                        meta: ChunkMeta::column(0, COLUMN_LABELS, 0, 1, 0),
                        values: vec![(0, Value::from("0"))],
                        epochs: None,
                    },
                    Meta,
                ],
                "namespace Section",
            ),
            (
                "a chunk in the node delete namespace, which version 3 does not have",
                vec![
                    alix(),
                    Raw {
                        meta: ChunkMeta::column(0, 0, 0, 1, 0)
                            .in_namespace(ChunkNamespace::NodeDeletes),
                        values: vec![(0, Value::Bool(true))],
                        epochs: None,
                    },
                    Meta,
                ],
                "namespace NodeDeletes",
            ),
            (
                "a chunk in the node label namespace, which version 3 does not have",
                vec![
                    alix(),
                    Raw {
                        meta: ChunkMeta::column(0, 0, 0, 1, 0)
                            .in_namespace(ChunkNamespace::NodeLabels),
                        values: vec![(0, Value::Bool(true))],
                        epochs: None,
                    },
                    Meta,
                ],
                "namespace NodeLabels",
            ),
            (
                "a chunk in the edge delete namespace, which version 3 does not have",
                vec![
                    alix(),
                    Raw {
                        meta: ChunkMeta::column(0, 0, 0, 1, 0)
                            .in_namespace(ChunkNamespace::EdgeDeletes),
                        values: vec![(0, Value::Bool(true))],
                        epochs: None,
                    },
                    Meta,
                ],
                "namespace EdgeDeletes",
            ),
            (
                "a chunk in the outgoing adjacency namespace, which version 3 does not have",
                vec![
                    alix(),
                    Raw {
                        meta: ChunkMeta::column(0, 0, 0, 1, 0)
                            .in_namespace(ChunkNamespace::OutgoingAdjacency),
                        values: vec![(0, Value::Bool(true))],
                        epochs: None,
                    },
                    Meta,
                ],
                "namespace OutgoingAdjacency",
            ),
            (
                "a chunk in the incoming adjacency namespace, which version 3 does not have",
                vec![
                    alix(),
                    Raw {
                        meta: ChunkMeta::column(0, 0, 0, 1, 0)
                            .in_namespace(ChunkNamespace::IncomingAdjacency),
                        values: vec![(0, Value::Bool(true))],
                        epochs: None,
                    },
                    Meta,
                ],
                "namespace IncomingAdjacency",
            ),
            (
                "an adjacency chunk, which version 3 does not have",
                vec![
                    alix(),
                    Raw {
                        meta: ChunkMeta {
                            kind: ChunkKind::Adjacency,
                            ..ChunkMeta::column(0, 0, 0, 1, 0)
                                .in_namespace(ChunkNamespace::OutgoingAdjacency)
                        },
                        values: vec![(0, Value::Bool(true))],
                        epochs: None,
                    },
                    Meta,
                ],
                "a Adjacency chunk",
            ),
            (
                // The namespace says which table a key's column is of: one
                // in the edge namespace holds edge rows.
                "a property value in the edge namespace for a row with no edge",
                vec![
                    two_nodes(),
                    Raw {
                        meta: ChunkMeta::column(0, NAME, 0, 1, 0)
                            .in_namespace(ChunkNamespace::EdgeProperties),
                        values: vec![(0, Value::from("Alix"))],
                        epochs: None,
                    },
                    Meta,
                ],
                "edge 0 has a property or history value",
            ),
            (
                "rows past the next id",
                vec![
                    Labels {
                        row_start: 4,
                        row_count: 1,
                        rows: vec![(0, "0")],
                    },
                    Meta,
                ],
                "next id",
            ),
            (
                "a chunk across a row group",
                vec![
                    Labels {
                        row_start: 3,
                        row_count: 2,
                        rows: vec![(0, "0")],
                    },
                    Meta,
                ],
                "row group",
            ),
            (
                "a label id past the label table",
                vec![
                    Labels {
                        row_start: 0,
                        row_count: 1,
                        rows: vec![(0, "7")],
                    },
                    Meta,
                ],
                "label 7",
            ),
            (
                "label ids not ascending",
                vec![
                    Labels {
                        row_start: 0,
                        row_count: 1,
                        rows: vec![(0, "0,0")],
                    },
                    Meta,
                ],
                "ascending",
            ),
            (
                "a label id that is not a number",
                vec![
                    Labels {
                        row_start: 0,
                        row_count: 1,
                        rows: vec![(0, "Person")],
                    },
                    Meta,
                ],
                "label",
            ),
            (
                "a negative endpoint",
                [
                    vec![two_nodes()],
                    edge_columns(vec![(0, -3, 1, 0)], 1),
                    vec![Meta],
                ]
                .into_iter()
                .flatten()
                .collect(),
                "-3",
            ),
            (
                "an edge type past the type table",
                [
                    vec![two_nodes()],
                    edge_columns(vec![(0, 0, 1, 5)], 1),
                    vec![Meta],
                ]
                .into_iter()
                .flatten()
                .collect(),
                "edge type 5",
            ),
            (
                "the edge table before the node table",
                [edge_columns(vec![(0, 0, 1, 0)], 1), vec![two_nodes(), Meta]]
                    .into_iter()
                    .flatten()
                    .collect(),
                "order",
            ),
            (
                "a history chunk of a fixed column",
                vec![
                    Raw {
                        meta: ChunkMeta::history(0, COLUMN_LABELS, 0, 1, 0)
                            .in_namespace(ChunkNamespace::NodeStructure),
                        values: vec![(0, Value::from("0"))],
                        epochs: None,
                    },
                    Meta,
                ],
                "a history chunk of a fixed column",
            ),
            (
                "a history chunk with epochs",
                vec![
                    alix(),
                    history(
                        NAME,
                        vec![(0, history_value(&[(3, Value::from("Gus"))]))],
                        Some(vec![3]),
                    ),
                    Meta,
                ],
                "epoch",
            ),
            (
                "a history value that is not a list of versions",
                vec![
                    alix(),
                    history(NAME, vec![(0, Value::Int64(3))], None),
                    Meta,
                ],
                "history",
            ),
            (
                "history epochs that go back",
                vec![
                    alix(),
                    history(
                        NAME,
                        vec![(
                            0,
                            history_value(&[(19, Value::from("Gus")), (3, Value::from("Mia"))]),
                        )],
                        None,
                    ),
                    Meta,
                ],
                "epoch",
            ),
            (
                "a history chunk after the column chunk of its rows",
                vec![
                    alix(),
                    column(NAME, vec![(0, Value::from("Alix"))], Some(vec![19])),
                    history(
                        NAME,
                        vec![(0, history_value(&[(3, Value::from("Gus"))]))],
                        None,
                    ),
                    Meta,
                ],
                "history",
            ),
            (
                "a value older than its history",
                vec![
                    alix(),
                    history(
                        NAME,
                        vec![(0, history_value(&[(19, Value::from("Gus"))]))],
                        None,
                    ),
                    column(NAME, vec![(0, Value::from("Alix"))], Some(vec![3])),
                    Meta,
                ],
                "epoch",
            ),
            (
                "a null property value",
                vec![alix(), column(NAME, vec![(0, Value::Null)], None), Meta],
                "null",
            ),
            (
                "epochs on a fixed column",
                vec![
                    Raw {
                        meta: ChunkMeta::column(0, COLUMN_LABELS, 0, 1, 0)
                            .in_namespace(ChunkNamespace::NodeStructure),
                        values: vec![(0, Value::from("0"))],
                        epochs: Some(vec![3]),
                    },
                    Meta,
                ],
                "epoch",
            ),
            (
                "an unknown graph",
                vec![
                    Raw {
                        meta: ChunkMeta::column(5, COLUMN_LABELS, 0, 1, 0)
                            .in_namespace(ChunkNamespace::NodeStructure),
                        values: vec![(0, Value::from("0"))],
                        epochs: None,
                    },
                    Meta,
                ],
                "graph 5",
            ),
            (
                "graph 0 with a name",
                vec![with(|meta| meta.graphs[0].name = "trips".into())],
                "graph 0",
            ),
            (
                "named graphs out of id order",
                vec![with(|meta| {
                    meta.next_graph_id = 3;
                    meta.graphs.push(named(2, "trips"));
                    meta.graphs.push(named(1, "travel"));
                })],
                "graph 1 comes after graph 2",
            ),
            (
                "a graph id not below the next graph id",
                vec![with(|meta| meta.graphs.push(named(1, "trips")))],
                "not below the next graph id 1",
            ),
            (
                "a default graph that is not first",
                vec![with(|meta| meta.graphs[0].id = 3)],
                "the first graph is graph 3",
            ),
            (
                "edge columns holding different rows",
                vec![
                    two_nodes(),
                    Column {
                        column_id: 1,
                        row_start: 0,
                        row_count: 2,
                        rows: vec![(0, 0)],
                    },
                    Column {
                        column_id: 2,
                        row_start: 0,
                        row_count: 2,
                        rows: vec![(1, 1)],
                    },
                    Column {
                        column_id: 3,
                        row_start: 0,
                        row_count: 2,
                        rows: vec![(0, 0)],
                    },
                    Meta,
                ],
                "different rows",
            ),
            (
                "a history row without its node",
                vec![
                    alix(),
                    Raw {
                        meta: ChunkMeta::history(0, NAME, 0, 3, 0)
                            .in_namespace(ChunkNamespace::NodeProperties),
                        values: vec![(2, history_value(&[(3, Value::from("Gus"))]))],
                        epochs: None,
                    },
                    Meta,
                ],
                "node 2",
            ),
            (
                "a value epoch above i64::MAX",
                vec![
                    alix(),
                    column(NAME, vec![(0, Value::from("Alix"))], Some(vec![u64::MAX])),
                    Meta,
                ],
                "above",
            ),
            (
                "a metadata epoch above i64::MAX",
                vec![with(|meta| meta.epoch = u64::MAX)],
                "epoch",
            ),
            (
                "a label id with a leading zero",
                vec![
                    Labels {
                        row_start: 0,
                        row_count: 1,
                        rows: vec![(0, "00")],
                    },
                    Meta,
                ],
                "is not a label id",
            ),
            (
                "a label id not below the next label id",
                vec![with(|meta| {
                    meta.graphs[0].labels.names.push((1, "City".into()));
                })],
                "label id 1 is not below the next label id 1",
            ),
            (
                "edge type ids not ascending",
                vec![with(|meta| {
                    meta.graphs[0].edge_types = DictionaryMeta {
                        next_id: 2,
                        names: vec![(1, "VISITED".into()), (0, "KNOWS".into())],
                    };
                })],
                "edge type id 0 comes after edge type id 1",
            ),
            (
                "a named graph listed twice",
                vec![with(|meta| {
                    meta.next_graph_id = 3;
                    meta.graphs.push(named(1, ""));
                    meta.graphs.push(named(2, ""));
                })],
                "two named graphs are named \"\"",
            ),
            (
                "a property key listed twice",
                vec![with(|meta| {
                    meta.graphs[0].keys = DictionaryMeta {
                        next_id: 2,
                        names: vec![(0, "name".into()), (1, "name".into())],
                    };
                })],
                "property key \"name\" is listed twice",
            ),
            (
                "a label listed twice",
                vec![with(|meta| {
                    meta.graphs[0].labels = DictionaryMeta {
                        next_id: 2,
                        names: vec![(0, "Person".into()), (1, "Person".into())],
                    };
                })],
                "label \"Person\" is listed twice",
            ),
            (
                "a label the graph does not hold",
                vec![
                    Labels {
                        row_start: 0,
                        row_count: 1,
                        rows: vec![(0, "1")],
                    },
                    Meta,
                ],
                "label 1 is not a label of the graph",
            ),
            (
                "an edge type the graph does not hold",
                [
                    vec![two_nodes()],
                    edge_columns(vec![(0, 0, 1, 5)], 1),
                    vec![Meta],
                ]
                .into_iter()
                .flatten()
                .collect(),
                "edge type 5 is not an edge type of the graph",
            ),
            (
                "rows of zero",
                vec![with(|meta| meta.max_rows = 0)],
                "max_rows",
            ),
            (
                "rows above the format's row cap",
                vec![with(|meta| meta.max_rows = ChunkCaps::DEFAULT.max_rows + 1)],
                "max_rows 65537",
            ),
        ];
        cases.push((
            "two metadata chunks",
            vec![
                Raw {
                    meta: ChunkMeta {
                        column_id: 1,
                        row_count: 1,
                        ..ChunkMeta::meta()
                    },
                    values: vec![(0, Value::Int64(3))],
                    epochs: None,
                },
                Meta,
            ],
            "metadata",
        ));
        for (name, chunks, words) in cases {
            let error = load_crafted(chunks).expect_err(name);
            assert!(matches!(error, Error::Corruption(_)), "{name}: {error:?}");
            let error = error.to_string();
            assert!(error.contains(words), "{name}: {error}");
        }

        // The well-formed crafted section loads: two nodes, an edge, a name.
        // A property chunk may come before the labels of its rows.
        let mut good = vec![
            Name {
                row_start: 0,
                row_count: 1,
                rows: vec![0],
            },
            two_nodes(),
        ];
        good.extend(edge_columns(vec![(0, 0, 1, 0)], 1));
        good.push(Meta);
        load_crafted(good).unwrap();
    }
}
