//! The LPG section in chunks (section version 3).
//!
//! Per graph in id order (graph 0 is the default graph, then the named graphs
//! in name order), the section holds its node table and then its edge table.
//! A row is a node or edge id, and a table is written in row groups of
//! `max_rows` rows, `[k * max_rows, (k + 1) * max_rows)`; only the groups
//! that hold a node (or an edge) are written, and the chunks of one group come
//! together. In a group:
//!
//! - the node table writes [`COLUMN_LABELS`]: the node's label ids, ascending,
//!   joined by `,` as a string (`""` for a node without labels);
//! - the edge table writes [`COLUMN_SOURCE`], [`COLUMN_TARGET`] and
//!   [`COLUMN_EDGE_TYPE`]: `Int64` values of the source and target node ids and
//!   the edge type id, as three chunks of one range in a row;
//! - each property column with a value in the group writes `Column` chunks of
//!   the current values. With `temporal`, a value carries the epoch it was set
//!   at, and the older versions go to `History` chunks, each written right
//!   before the `Column` chunk of its rows: a history value is a list of
//!   `[epoch, value]` lists, epochs ascending. A property removed last (its
//!   latest version null) has no current value; its history holds every
//!   version. A `History` chunk and the `Column` chunk after it cover the same
//!   rows; either can come alone: a `History` chunk when every row of its
//!   range had its property removed last, a `Column` chunk when no row of its
//!   range has an older version. Versions of transactions that did not commit
//!   are not written.
//!
//! The metadata chunk ([`ChunkMeta::meta`]) comes last: [`LpgMeta`], with the
//! caps the section was written with, the graphs with their next node and
//! edge ids, the label and edge type names (a label's or edge type's id in
//! the file is its position in these lists), and the property columns (ids
//! from [`FIRST_PROPERTY_COLUMN`]), in the layout of [`encode_lpg_meta`],
//! which has no size limit of its own: it holds as many names as the store
//! has. It comes last
//! because it lists every name the chunks use: commits are held while a
//! checkpoint writes, but a transaction still open can create a label, an
//! edge type or a property key meanwhile, and the write meets it. The names
//! are those the registries held when the write began, sorted (labels and
//! edge types by name, columns by `(table, key)`), followed by the names met
//! during the write, in the order met. Each graph also lists the labels and
//! edge types it has registered that none of its rows use (those of deleted
//! nodes and edges, or of a transaction that rolled back). A reader fetches
//! the metadata chunk first, then the others in order, and registers each
//! graph's unused names in that graph, so every graph's names come back as
//! they were. A loaded store then writes the same chunks again, with two
//! exceptions: names appended during a write are sorted into place by the
//! next write (their ids change), and a property column without a value is
//! not recreated (the next write lists one column fewer).
//!
//! The chunk sizes follow [`RowsChunker`]: at most `max_rows` rows and
//! `max_bytes` bytes, a larger value in a chunk of its own. Every order is
//! defined (ids, names and keys sorted), so the same data gives the same
//! bytes. Only the nodes and edges visible now are written, with their
//! properties; a property whose value is null does not exist and is not
//! written (the direct store API and a 0.5.x load can leave one in a store
//! without `temporal`). With `temporal` a node's labels are those of the
//! store's current epoch, so a label set a transaction still open wrote is
//! not written; without it the store keeps no versions, and such a
//! transaction's in-place changes reach the section.
//!
//! Memory: the writer holds one open chunk per column of the row group
//! being written, the sorted ids of the table being written (8 bytes per
//! node or edge) and, without `temporal`, the sorted ids of each of its
//! property columns (8 bytes per value); it reads values a batch at a time.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bytes::Bytes;
use grafeo_common::storage::value_codec::{MAX_PROPERTY_VALUE_DEPTH, nests_too_deep};
use grafeo_common::storage::{ChunkCaps, ChunkKind, ChunkMeta, SectionSink, SectionSource};
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};

use super::LpgStore;
use super::property::{EntityId, PropertyStorage};
use crate::codec::column_chunk::{ColumnChunk, decode_column_chunk_bytes};
use crate::codec::{ChunkColumn, RowsChunker};

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
/// The id of the first property column; ids below it are fixed columns.
pub(crate) const FIRST_PROPERTY_COLUMN: u32 = 16;

/// The largest epoch a section holds: history values store epochs as
/// `Int64`.
const MAX_EPOCH: u64 = i64::MAX.unsigned_abs();

/// How many values of one property column the writer reads at a time.
#[cfg(not(feature = "temporal"))]
const READ_BATCH_ROWS: usize = 256;

/// The metadata chunk of an LPG section.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LpgMeta {
    /// [`LPG_META_LAYOUT`].
    pub layout: u8,
    /// The rows per row group and chunk the section was written with.
    pub max_rows: u32,
    /// The byte cap the section was written with.
    pub max_bytes: u32,
    /// The store's epoch with `temporal`, 0 without.
    pub epoch: u64,
    /// Label names; a label's id in the file is its position.
    pub labels: Vec<String>,
    /// Edge type names; an edge type's id in the file is its position.
    pub edge_types: Vec<String>,
    /// Graph `i` has graph id `i`; graph 0 is the default graph, with an
    /// empty name. The named graphs follow in name order (one of them may
    /// have the empty name: the default graph is graph 0 by its position).
    pub graphs: Vec<GraphMeta>,
    /// The property columns, ids ascending from [`FIRST_PROPERTY_COLUMN`].
    pub columns: Vec<ColumnMeta>,
}

/// One graph of an LPG section.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GraphMeta {
    /// The graph's name; empty for the default graph.
    pub name: String,
    /// The id the graph gives its next node: the node table's rows are below it.
    pub next_node_id: u64,
    /// The id the graph gives its next edge: the edge table's rows are below it.
    pub next_edge_id: u64,
    /// The ids of the labels the graph has registered that none of its nodes
    /// has, ascending.
    pub unused_labels: Vec<u32>,
    /// The ids of the edge types the graph has registered that none of its
    /// edges has, ascending.
    pub unused_edge_types: Vec<u32>,
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
}

/// A property column of an LPG section.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ColumnMeta {
    /// The column's id in its chunks, at least [`FIRST_PROPERTY_COLUMN`].
    pub column_id: u32,
    /// The table whose rows the column holds values for.
    pub table: Table,
    /// The property key.
    pub key: String,
}

// ── Writing ─────────────────────────────────────────────────────────

/// Writes per graph (in id order) the node table's row groups, then the edge
/// table's, then the metadata chunk.
///
/// # Errors
///
/// Returns [`Error::InvalidValue`] for caps of zero; [`Error::Serialization`]
/// for what a file cannot hold (each naming what holds it): a value nested
/// deeper than [`MAX_PROPERTY_VALUE_DEPTH`], an edge endpoint, an epoch
/// (with `temporal`) or the store's epoch above `i64::MAX`, a node or edge
/// with the largest id (no next id can follow it), more names than ids, or a
/// name longer than `u32::MAX` bytes; the error of reading a node or edge
/// record or a spilled property value; and the sink's error.
pub(crate) fn write_lpg_chunks(
    store: &LpgStore,
    caps: ChunkCaps,
    sink: &mut dyn SectionSink,
) -> Result<()> {
    caps.validate()?;
    let mut graph_names = store.graph_names();
    graph_names.sort_unstable();
    // A graph dropped since the names were read is left out.
    let named: Vec<(String, Arc<LpgStore>)> = graph_names
        .into_iter()
        .filter_map(|name| store.graph(&name).map(|graph| (name, graph)))
        .collect();
    let graphs: Vec<(&str, &LpgStore)> = std::iter::once(("", store))
        .chain(named.iter().map(|(name, graph)| (name.as_str(), &**graph)))
        .collect();

    let epoch = section_epoch(store);
    if epoch > MAX_EPOCH {
        return Err(Error::Serialization(format!(
            "the store epoch {epoch} is above {MAX_EPOCH}, the largest epoch a section holds"
        )));
    }
    let names = Names::begin(&graphs);
    let mut graph_metas = Vec::with_capacity(graphs.len());
    for (index, (name, graph)) in graphs.iter().enumerate() {
        let place = Place {
            graph_id: u32::try_from(index).map_err(|_| {
                Error::Serialization(format!("an LPG section holds {index} graphs or more"))
            })?,
            graph: describe_graph(index, name),
            caps,
        };
        // With `temporal`, the labels of the current epoch: every committed
        // label set is at or below it (the default graph's epoch follows
        // every commit), a pending one is above it.
        let labels_at = EpochId::new(graph.current_epoch().as_u64().max(epoch));
        let (nodes_end, used_labels) = write_node_table(graph, labels_at, &names, &place, sink)?;
        let (edges_end, used_edge_types) = write_edge_table(graph, &names, &place, sink)?;
        // Read after the rows: every row written is below the next ids, and
        // the registries hold every name the rows use.
        graph_metas.push(GraphMeta {
            name: (*name).to_string(),
            next_node_id: graph.next_node_id().max(nodes_end),
            next_edge_id: graph.next_edge_id().max(edges_end),
            unused_labels: unused(graph.all_labels(), &names.labels, &used_labels)?,
            unused_edge_types: unused(graph.all_edge_types(), &names.edge_types, &used_edge_types)?,
        });
    }

    let meta = names.finish(caps, epoch, graph_metas)?;
    sink.write_chunk(ChunkMeta::meta(), &encode_lpg_meta(&meta)?)
}

/// The ids of the names in `registered` that are not in `used`, ascending.
fn unused(registered: Vec<String>, list: &NameList, used: &BTreeSet<u32>) -> Result<Vec<u32>> {
    let mut ids = BTreeSet::new();
    for name in registered {
        let id = list.id(&name)?;
        if !used.contains(&id) {
            ids.insert(id);
        }
    }
    Ok(ids.into_iter().collect())
}

/// Writes the node table of `graph`, each node with the labels it has at
/// `labels_at` (see [`LpgStore::node_with_labels_at`]); returns one past its
/// last row (0 for none) and the ids of the labels its nodes have.
fn write_node_table(
    graph: &LpgStore,
    labels_at: EpochId,
    names: &Names,
    place: &Place,
    sink: &mut dyn SectionSink,
) -> Result<(u64, BTreeSet<u32>)> {
    let used = RefCell::new(BTreeSet::new());
    let nodes = graph.try_node_ids()?.into_iter().map(|id| {
        let node = graph.node_with_labels_at(id, labels_at);
        let mut labels = node
            .labels
            .iter()
            .map(|label| names.labels.id(label))
            .collect::<Result<Vec<u32>>>()?;
        labels.sort_unstable();
        used.borrow_mut().extend(labels.iter().copied());
        let text = labels
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        Ok((node.id, vec![Some((Value::from(text), 0))]))
    });
    let table = TableWriter {
        table: Table::Node,
        fixed: vec![column(COLUMN_LABELS)],
        properties: &graph.node_properties,
        names,
    };
    let end = table.write(place, nodes, sink)?;
    Ok((end, used.into_inner()))
}

/// Writes the edge table of `graph`; returns one past its last row (0 for
/// none) and the ids of the edge types its edges have.
fn write_edge_table(
    graph: &LpgStore,
    names: &Names,
    place: &Place,
    sink: &mut dyn SectionSink,
) -> Result<(u64, BTreeSet<u32>)> {
    let used = RefCell::new(BTreeSet::new());
    let edges = graph
        .try_edge_ids()?
        .into_iter()
        .filter_map(|id| graph.try_edge_without_properties(id).transpose())
        .map(|edge| {
            let edge = edge?;
            let endpoint = |node: NodeId, end: &str| {
                i64::try_from(node.as_u64()).map_err(|_| {
                    Error::Serialization(format!(
                        "{}, edge {}: its {end} node {} is above {}, the largest id an edge \
                         table holds",
                        place.graph,
                        edge.id.as_u64(),
                        node.as_u64(),
                        i64::MAX
                    ))
                })
            };
            let source = endpoint(edge.src, "source")?;
            let target = endpoint(edge.dst, "target")?;
            let edge_type = names.edge_types.id(&edge.edge_type)?;
            used.borrow_mut().insert(edge_type);
            Ok((
                edge.id,
                vec![
                    Some((Value::Int64(source), 0)),
                    Some((Value::Int64(target), 0)),
                    Some((Value::Int64(i64::from(edge_type)), 0)),
                ],
            ))
        });
    let table = TableWriter {
        table: Table::Edge,
        fixed: vec![
            column(COLUMN_SOURCE),
            column(COLUMN_TARGET),
            column(COLUMN_EDGE_TYPE),
        ],
        properties: &graph.edge_properties,
        names,
    };
    let end = table.write(place, edges, sink)?;
    Ok((end, used.into_inner()))
}

/// The epoch the section records: the store's with `temporal`.
#[cfg(feature = "temporal")]
fn section_epoch(store: &LpgStore) -> u64 {
    store.current_epoch().as_u64()
}

/// The epoch the section records: 0 without `temporal`, as the 0.5.x block
/// layout wrote it.
#[cfg(not(feature = "temporal"))]
fn section_epoch(_store: &LpgStore) -> u64 {
    0
}

/// How graph `index` (named `name`) is named in errors.
fn describe_graph(index: usize, name: &str) -> String {
    if index == 0 {
        "the default graph".to_string()
    } else {
        format!("graph {name:?}")
    }
}

/// A `Column` chunk column.
fn column(column_id: u32) -> ChunkColumn {
    ChunkColumn {
        kind: ChunkKind::Column,
        column_id,
    }
}

/// The names a write gives ids: those the registries held when it began,
/// sorted, then those it meets, in the order met.
struct Names {
    labels: NameList,
    edge_types: NameList,
    columns: ColumnList,
}

impl Names {
    /// The names `graphs` hold now: the union over every graph.
    fn begin(graphs: &[(&str, &LpgStore)]) -> Self {
        let mut labels = Vec::new();
        let mut edge_types = Vec::new();
        let mut columns = BTreeSet::new();
        for (_, graph) in graphs {
            labels.extend(graph.all_labels());
            edge_types.extend(graph.all_edge_types());
            columns.extend(
                graph
                    .node_property_keys()
                    .into_iter()
                    .map(|key| (Table::Node, key)),
            );
            columns.extend(
                graph
                    .edge_property_keys()
                    .into_iter()
                    .map(|key| (Table::Edge, key)),
            );
        }
        Self {
            labels: NameList::new(labels, "label"),
            edge_types: NameList::new(edge_types, "edge type"),
            columns: ColumnList {
                sorted: columns.into_iter().collect(),
                added: RefCell::new(Vec::new()),
            },
        }
    }

    /// The metadata of a write with these names.
    fn finish(self, caps: ChunkCaps, epoch: u64, graphs: Vec<GraphMeta>) -> Result<LpgMeta> {
        Ok(LpgMeta {
            layout: LPG_META_LAYOUT,
            max_rows: caps.max_rows,
            max_bytes: caps.max_bytes,
            epoch,
            labels: self.labels.into_names(),
            edge_types: self.edge_types.into_names(),
            graphs,
            columns: self.columns.into_columns()?,
        })
    }
}

/// Label or edge type names with their ids in the file: the names a write
/// began with, sorted, then those it met, in the order met.
pub(super) struct NameList {
    sorted: Vec<String>,
    added: RefCell<Vec<String>>,
    /// "label" or "edge type", for errors.
    what: &'static str,
}

impl NameList {
    /// `names`, sorted, each once.
    pub(super) fn new(names: impl IntoIterator<Item = String>, what: &'static str) -> Self {
        let sorted: BTreeSet<String> = names.into_iter().collect();
        Self {
            sorted: sorted.into_iter().collect(),
            added: RefCell::new(Vec::new()),
            what,
        }
    }

    /// The id of `name`, given the next free id when the list lacks it.
    pub(super) fn id(&self, name: &str) -> Result<u32> {
        let at = match self
            .sorted
            .binary_search_by(|known| known.as_str().cmp(name))
        {
            Ok(at) => at,
            Err(_) => {
                let mut added = self.added.borrow_mut();
                let position = match added.iter().position(|known| known == name) {
                    Some(position) => position,
                    None => {
                        added.push(name.to_string());
                        added.len() - 1
                    }
                };
                self.sorted.len() + position
            }
        };
        u32::try_from(at).map_err(|_| {
            Error::Serialization(format!(
                "an LPG section holds more than {} {} names",
                u32::MAX,
                self.what
            ))
        })
    }

    /// Every name, in id order.
    pub(super) fn into_names(self) -> Vec<String> {
        let mut names = self.sorted;
        names.extend(self.added.into_inner());
        names
    }
}

/// The property columns of a write, like [`NameList`]: those it began with,
/// by `(table, key)`, then those it met.
struct ColumnList {
    sorted: Vec<(Table, PropertyKey)>,
    added: RefCell<Vec<(Table, PropertyKey)>>,
}

impl ColumnList {
    /// The column id of property `key` of `table`, given the next free id
    /// when the list lacks it.
    fn id(&self, table: Table, key: &PropertyKey) -> Result<u32> {
        let at = match self
            .sorted
            .binary_search_by(|(known_table, known)| (*known_table, known).cmp(&(table, key)))
        {
            Ok(at) => at,
            Err(_) => {
                let mut added = self.added.borrow_mut();
                let position = match added
                    .iter()
                    .position(|(known_table, known)| *known_table == table && known == key)
                {
                    Some(position) => position,
                    None => {
                        added.push((table, key.clone()));
                        added.len() - 1
                    }
                };
                self.sorted.len() + position
            }
        };
        column_id(at)
    }

    /// Every column, in id order.
    fn into_columns(self) -> Result<Vec<ColumnMeta>> {
        self.sorted
            .into_iter()
            .chain(self.added.into_inner())
            .enumerate()
            .map(|(at, (table, key))| {
                Ok(ColumnMeta {
                    column_id: column_id(at)?,
                    table,
                    key: key.as_str().to_string(),
                })
            })
            .collect()
    }
}

/// The id of the property column at position `at`.
fn column_id(at: usize) -> Result<u32> {
    u32::try_from(at)
        .ok()
        .and_then(|at| FIRST_PROPERTY_COLUMN.checked_add(at))
        .ok_or_else(|| {
            Error::Serialization(format!(
                "an LPG section holds {at} property columns or more"
            ))
        })
}

/// Where a table is written.
struct Place {
    graph_id: u32,
    /// The graph as errors name it.
    graph: String,
    caps: ChunkCaps,
}

/// A row of a table: the node or edge id with its cells of the fixed columns.
type Row<Id> = (Id, Vec<Option<(Value, u64)>>);

/// One table of one graph.
struct TableWriter<'s, Id: EntityId> {
    table: Table,
    /// The fixed columns of the table.
    fixed: Vec<ChunkColumn>,
    properties: &'s PropertyStorage<Id>,
    names: &'s Names,
}

impl<Id: EntityId> TableWriter<'_, Id> {
    /// Writes the row groups holding `rows` (ascending by id), one group at a
    /// time; returns one past the last row (0 for none).
    fn write(
        &self,
        place: &Place,
        mut rows: impl Iterator<Item = Result<Row<Id>>>,
        sink: &mut dyn SectionSink,
    ) -> Result<u64> {
        let max_rows = u64::from(place.caps.max_rows);
        #[cfg(not(feature = "temporal"))]
        let mut cursors = self.cursors()?;
        let mut end = 0;
        let mut pending: Option<Row<Id>> = None;
        loop {
            let (first, cells) = match pending.take() {
                Some(row) => row,
                None => match rows.next() {
                    Some(row) => row?,
                    None => return Ok(end),
                },
            };
            let first_row = first.as_u64();
            let mut group = Group::new(self, place, first_row - first_row % max_rows);
            let mut last = first_row;
            group.push(self, sink, first, cells)?;
            for row in rows.by_ref() {
                let (id, cells) = row?;
                let in_group = id
                    .as_u64()
                    .checked_sub(group.start)
                    .is_some_and(|offset| offset < max_rows);
                if in_group {
                    last = id.as_u64();
                    group.push(self, sink, id, cells)?;
                } else {
                    pending = Some((id, cells));
                    break;
                }
            }
            // A row is below `u64::MAX` (`Group::push` refuses that id).
            end = last + 1;
            #[cfg(not(feature = "temporal"))]
            group.finish(self, &mut cursors, sink)?;
            #[cfg(feature = "temporal")]
            group.finish(sink)?;
            if pending.is_none() {
                return Ok(end);
            }
        }
    }

    /// The table's property columns as the store holds them now, by key,
    /// each with the ids that have a value.
    #[cfg(not(feature = "temporal"))]
    fn cursors(&self) -> Result<Vec<ColumnCursor<Id>>> {
        let mut keys = self.properties.keys();
        keys.sort_unstable();
        keys.into_iter()
            .map(|key| {
                Ok(ColumnCursor {
                    column_id: self.names.columns.id(self.table, &key)?,
                    ids: self.properties.column_ids(&key),
                    key,
                    next: 0,
                })
            })
            .collect()
    }

    /// Refuses a value a file cannot hold, naming where it is.
    fn refuse_too_deep(
        &self,
        place: &Place,
        id: Id,
        key: &PropertyKey,
        value: &Value,
        what: &str,
    ) -> Result<()> {
        if nests_too_deep(value) {
            return Err(Error::Serialization(format!(
                "{}, {} {}, property {:?}: {what} nests lists, maps and paths more than \
                 {MAX_PROPERTY_VALUE_DEPTH} levels deep, deeper than a database file can hold",
                place.graph,
                self.table.entity(),
                id.as_u64(),
                key.as_str()
            )));
        }
        Ok(())
    }
}

/// The row group being written.
struct Group<'p, Id: EntityId> {
    place: &'p Place,
    /// The group's first row.
    start: u64,
    /// The fixed columns.
    fixed: RowsChunker,
    /// The ids of the group's nodes or edges, ascending.
    #[cfg(not(feature = "temporal"))]
    present: Vec<Id>,
    /// One chunker per property column met in the group, `[History, Column]`.
    #[cfg(feature = "temporal")]
    properties: BTreeMap<u32, RowsChunker>,
    #[cfg(feature = "temporal")]
    entities: std::marker::PhantomData<Id>,
}

impl<'p, Id: EntityId> Group<'p, Id> {
    fn new(table: &TableWriter<'_, Id>, place: &'p Place, start: u64) -> Self {
        Self {
            place,
            start,
            fixed: RowsChunker::new(place.graph_id, table.fixed.clone(), start, place.caps),
            #[cfg(not(feature = "temporal"))]
            present: Vec::new(),
            #[cfg(feature = "temporal")]
            properties: BTreeMap::new(),
            #[cfg(feature = "temporal")]
            entities: std::marker::PhantomData,
        }
    }

    /// Adds the row of `id`: its fixed cells, and with `temporal` its
    /// property versions.
    fn push(
        &mut self,
        table: &TableWriter<'_, Id>,
        sink: &mut dyn SectionSink,
        id: Id,
        cells: Vec<Option<(Value, u64)>>,
    ) -> Result<()> {
        if id.as_u64() == u64::MAX {
            let entity = table.table.entity();
            return Err(Error::Serialization(format!(
                "{}, {entity} {}: the largest id, after which the section cannot store the \
                 graph's next {entity} id",
                self.place.graph,
                u64::MAX
            )));
        }
        self.fixed.push(sink, id.as_u64(), cells)?;
        #[cfg(not(feature = "temporal"))]
        self.present.push(id);
        #[cfg(feature = "temporal")]
        self.push_versions(table, sink, id)?;
        Ok(())
    }

    /// Adds the committed property versions of `id` to the chunkers of
    /// their columns, in column order.
    #[cfg(feature = "temporal")]
    fn push_versions(
        &mut self,
        table: &TableWriter<'_, Id>,
        sink: &mut dyn SectionSink,
        id: Id,
    ) -> Result<()> {
        let mut histories = Vec::new();
        for (key, mut versions) in table.properties.get_all_history(id) {
            // A checkpoint holds committed versions only.
            versions.retain(|(epoch, _)| *epoch != EpochId::PENDING);
            if !versions.is_empty() {
                histories.push((table.names.columns.id(table.table, &key)?, key, versions));
            }
        }
        histories.sort_unstable_by_key(|(column_id, _, _)| *column_id);
        for (column_id, key, versions) in histories {
            let Some((epoch, latest)) = versions.last() else {
                continue;
            };
            let (current, older) = if latest.is_null() {
                (None, versions.as_slice())
            } else {
                table.refuse_too_deep(self.place, id, &key, latest, "the value")?;
                (
                    Some((
                        latest.clone(),
                        stored_epoch(self.place, table, id, &key, *epoch)?,
                    )),
                    &versions[..versions.len() - 1],
                )
            };
            let history = if older.is_empty() {
                None
            } else {
                for (_, value) in older {
                    table.refuse_too_deep(self.place, id, &key, value, "an older version")?;
                }
                Some((history_cell(self.place, table, id, &key, older)?, 0))
            };
            let (place, start) = (self.place, self.start);
            self.properties
                .entry(column_id)
                .or_insert_with(|| {
                    let columns = vec![
                        ChunkColumn {
                            kind: ChunkKind::History,
                            column_id,
                        },
                        column(column_id),
                    ];
                    RowsChunker::new(place.graph_id, columns, start, place.caps)
                })
                .push(sink, id.as_u64(), vec![history, current])?;
        }
        Ok(())
    }

    /// Writes the group's open chunks: the fixed columns', then the property
    /// columns' in column order.
    #[cfg(feature = "temporal")]
    fn finish(self, sink: &mut dyn SectionSink) -> Result<()> {
        self.fixed.finish(sink)?;
        for chunker in self.properties.into_values() {
            chunker.finish(sink)?;
        }
        Ok(())
    }

    /// Writes the group's open fixed chunks, then its property columns, one
    /// column at a time, reading the values a batch at a time.
    #[cfg(not(feature = "temporal"))]
    fn finish(
        self,
        table: &TableWriter<'_, Id>,
        cursors: &mut [ColumnCursor<Id>],
        sink: &mut dyn SectionSink,
    ) -> Result<()> {
        self.fixed.finish(sink)?;
        let max_rows = u64::from(self.place.caps.max_rows);
        for cursor in cursors {
            let ids = intersect(cursor.take_group(self.start, max_rows), &self.present);
            if ids.is_empty() {
                continue;
            }
            let mut chunker = RowsChunker::new(
                self.place.graph_id,
                vec![column(cursor.column_id)],
                self.start,
                self.place.caps,
            );
            for batch in ids.chunks(READ_BATCH_ROWS) {
                let values = table.properties.try_get_batch(batch, &cursor.key)?;
                for (&id, value) in batch.iter().zip(values) {
                    // Removed since the ids were listed, or null (a property
                    // whose value is null does not exist): nothing to write.
                    let Some(value) = value.filter(|value| !value.is_null()) else {
                        continue;
                    };
                    table.refuse_too_deep(self.place, id, &cursor.key, &value, "the value")?;
                    chunker.push(sink, id.as_u64(), vec![Some((value, 0))])?;
                }
            }
            chunker.finish(sink)?;
        }
        Ok(())
    }
}

/// The epoch a section stores for a version of property `key` of `id`:
/// at most [`MAX_EPOCH`], as a reader requires (history values hold epochs
/// as `Int64`).
#[cfg(feature = "temporal")]
fn stored_epoch<Id: EntityId>(
    place: &Place,
    table: &TableWriter<'_, Id>,
    id: Id,
    key: &PropertyKey,
    epoch: EpochId,
) -> Result<u64> {
    let epoch = epoch.as_u64();
    if epoch > MAX_EPOCH {
        return Err(Error::Serialization(format!(
            "{}, {} {}, property {:?}: epoch {epoch} is above {MAX_EPOCH}, the largest epoch a \
             section holds",
            place.graph,
            table.table.entity(),
            id.as_u64(),
            key.as_str()
        )));
    }
    Ok(epoch)
}

/// A history value: the versions as `[epoch, value]` lists, ascending.
#[cfg(feature = "temporal")]
fn history_cell<Id: EntityId>(
    place: &Place,
    table: &TableWriter<'_, Id>,
    id: Id,
    key: &PropertyKey,
    versions: &[(EpochId, Value)],
) -> Result<Value> {
    let versions = versions
        .iter()
        .map(|(epoch, value)| {
            let epoch = stored_epoch(place, table, id, key, *epoch)?;
            let epoch = i64::try_from(epoch).unwrap_or(i64::MAX);
            Ok(Value::List(Arc::from(vec![
                Value::Int64(epoch),
                value.clone(),
            ])))
        })
        .collect::<Result<Vec<Value>>>()?;
    Ok(Value::List(Arc::from(versions)))
}

/// The ids of one property column, walked group by group.
#[cfg(not(feature = "temporal"))]
struct ColumnCursor<Id> {
    key: PropertyKey,
    column_id: u32,
    /// The ids with a value, ascending.
    ids: Vec<Id>,
    /// The first id not walked yet.
    next: usize,
}

#[cfg(not(feature = "temporal"))]
impl<Id: EntityId> ColumnCursor<Id> {
    /// The ids of the group `[start, start + max_rows)`, after skipping the
    /// ids before it (those of groups without a node or edge).
    fn take_group(&mut self, start: u64, max_rows: u64) -> &[Id] {
        while self
            .ids
            .get(self.next)
            .is_some_and(|id| id.as_u64() < start)
        {
            self.next += 1;
        }
        let first = self.next;
        while self
            .ids
            .get(self.next)
            .is_some_and(|id| id.as_u64() - start < max_rows)
        {
            self.next += 1;
        }
        &self.ids[first..self.next]
    }
}

/// The ids in both `left` and `right`, both ascending.
#[cfg(not(feature = "temporal"))]
fn intersect<Id: EntityId>(left: &[Id], right: &[Id]) -> Vec<Id> {
    let mut both = Vec::new();
    let (mut at_left, mut at_right) = (0, 0);
    while let (Some(a), Some(b)) = (left.get(at_left), right.get(at_right)) {
        match a.as_u64().cmp(&b.as_u64()) {
            std::cmp::Ordering::Less => at_left += 1,
            std::cmp::Ordering::Greater => at_right += 1,
            std::cmp::Ordering::Equal => {
                both.push(*a);
                at_left += 1;
                at_right += 1;
            }
        }
    }
    both
}

// ── Reading ─────────────────────────────────────────────────────────

/// Encodes a metadata chunk. Little-endian:
///
/// | Field | Encoding |
/// | --- | --- |
/// | layout | u8, [`LPG_META_LAYOUT`] |
/// | max_rows, max_bytes | u32 each |
/// | epoch | u64 |
/// | labels, edge types | each a count u32, then per name its length u32 and UTF-8 |
/// | graphs | a count u32, then per graph its name (as above), next node id u64, next edge id u64, unused label ids and unused edge type ids (each a count u32 and u32 ids) |
/// | columns | a count u32, then per column its id u32, table u8 (0 node, 1 edge) and key (as above) |
///
/// Nothing in the layout limits its size: a reader checks every count and
/// length against the bytes left, so the chunk holds as many names as the
/// store has.
///
/// # Errors
///
/// Returns [`Error::Serialization`] when a list holds more than `u32::MAX`
/// entries or a name is longer than `u32::MAX` bytes.
pub(crate) fn encode_lpg_meta(meta: &LpgMeta) -> Result<Vec<u8>> {
    let mut out = vec![meta.layout];
    out.extend_from_slice(&meta.max_rows.to_le_bytes());
    out.extend_from_slice(&meta.max_bytes.to_le_bytes());
    out.extend_from_slice(&meta.epoch.to_le_bytes());
    for (names, what) in [(&meta.labels, "label"), (&meta.edge_types, "edge type")] {
        put_count(names.len(), what, &mut out)?;
        for name in names {
            put_name(name, what, &mut out)?;
        }
    }
    put_count(meta.graphs.len(), "graph", &mut out)?;
    for graph in &meta.graphs {
        put_name(&graph.name, "graph", &mut out)?;
        out.extend_from_slice(&graph.next_node_id.to_le_bytes());
        out.extend_from_slice(&graph.next_edge_id.to_le_bytes());
        for (ids, what) in [
            (&graph.unused_labels, "unused label"),
            (&graph.unused_edge_types, "unused edge type"),
        ] {
            put_count(ids.len(), what, &mut out)?;
            for id in ids {
                out.extend_from_slice(&id.to_le_bytes());
            }
        }
    }
    put_count(meta.columns.len(), "property column", &mut out)?;
    for column in &meta.columns {
        out.extend_from_slice(&column.column_id.to_le_bytes());
        out.push(match column.table {
            Table::Node => 0,
            Table::Edge => 1,
        });
        put_name(&column.key, "property", &mut out)?;
    }
    Ok(out)
}

/// Appends the count of a list of `what`s.
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
/// Returns [`Error::Serialization`] naming the byte offset of what is wrong:
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
    let labels = reader.names("label")?;
    let edge_types = reader.names("edge type")?;
    // A name length, two ids and two counts at least.
    let count = reader.count("graph", 4 + 8 + 8 + 4 + 4)?;
    let mut graphs = Vec::with_capacity(count);
    for _ in 0..count {
        graphs.push(GraphMeta {
            name: reader.name("graph")?,
            next_node_id: reader.u64("next node id", "")?,
            next_edge_id: reader.u64("next edge id", "")?,
            unused_labels: reader.ids("unused label")?,
            unused_edge_types: reader.ids("unused edge type")?,
        });
    }
    // An id, a table and a key length at least.
    let count = reader.count("property column", 4 + 1 + 4)?;
    let mut columns = Vec::with_capacity(count);
    for _ in 0..count {
        let column_id = reader.u32("column id", "")?;
        let at = reader.pos;
        let table = match reader.u8("table", "")? {
            0 => Table::Node,
            1 => Table::Edge,
            other => {
                return Err(reader.refuse(
                    at,
                    format!("table {other}, where 0 is the node table and 1 the edge table"),
                ));
            }
        };
        columns.push(ColumnMeta {
            column_id,
            table,
            key: reader.name("property")?,
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
        labels,
        edge_types,
        graphs,
        columns,
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
        Error::Serialization(format!("LPG metadata chunk, byte {at}: {what}"))
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

    /// A list of names.
    fn names(&mut self, what: &str) -> Result<Vec<String>> {
        let count = self.count(what, 4)?;
        let mut names = Vec::with_capacity(count);
        for _ in 0..count {
            names.push(self.name(what)?);
        }
        Ok(names)
    }

    /// A list of u32 ids.
    fn ids(&mut self, what: &str) -> Result<Vec<u32>> {
        let count = self.count(what, 4)?;
        let mut ids = Vec::with_capacity(count);
        for _ in 0..count {
            ids.push(self.u32(what, "id")?);
        }
        Ok(ids)
    }
}

/// Applies the chunks of an LPG section of version 3 to `store`, one chunk at
/// a time: the metadata chunk (the last one) first, then the others in order,
/// each fetched when it is reached.
///
/// Rules (every failure an [`Error::Serialization`] naming the graph, the
/// column and the rows):
///
/// 1. The last chunk is the metadata chunk, the only one: layout 1, caps of at
///    least one row and byte, an epoch of at most `i64::MAX`, graph 0 without
///    a name, the named graphs in strictly ascending name order, each label,
///    edge type and `(table, key)` listed once, property column ids unique and
///    at least [`FIRST_PROPERTY_COLUMN`].
/// 2. Every other chunk is a `Column` or `History` chunk (`History` only for
///    property columns) of a known graph and column, with `1 <= row_count <=
///    max_rows`, inside one row group, ending below its table's next id.
/// 3. Chunks come grouped by (graph, table, row group), in that order (the
///    node table before the edge table); a change of group closes the one
///    before. Within a group the chunks of one (kind, column) ascend without
///    overlap, and a `History` chunk never covers a row of a `Column` chunk of
///    its column that came before it.
/// 4. The edge columns 1, 2 and 3 come as three consecutive chunks of one
///    range and the same rows.
/// 5. When a group closes, every row with a property or history value has
///    its node (a labels value) or edge (an endpoints value) in the group.
/// 6. Labels are label ids below the label count, ascending, in decimal
///    joined by `,`; endpoints are non-negative `Int64`s and types `Int64`s
///    below the edge type count. Fixed columns and `History` chunks carry no
///    epochs; a property value is never null; a history value is a non-empty
///    list of `[epoch, value]` lists with `Int64` epochs from 0, never going
///    back (equal epochs are versions of one commit), and the value after it
///    (if any) has an epoch at least the last one; epochs are at most
///    `i64::MAX`.
///
/// Without `temporal` the history versions are checked and not applied, and
/// values are set without their epochs. At the end each graph's next ids
/// become the larger of its own and the metadata's, each graph registers the
/// labels and edge types the metadata lists as its unused ones (refusing one
/// its rows use), and with `temporal` every graph store's epoch is synced to
/// the metadata's.
///
/// # Errors
///
/// [`Error::Serialization`] for a section the rules refuse, and for a node
/// or edge the store cannot allocate (naming it); the source's error when a
/// chunk cannot be fetched; [`Error::Internal`] when the store cannot
/// allocate a named graph.
pub(crate) fn read_lpg_chunks(store: &LpgStore, source: &dyn SectionSource) -> Result<()> {
    let chunks = source.chunks();
    let Some((last, data)) = chunks.split_last() else {
        return Err(Error::Serialization(
            "LPG section: no chunk, where the last chunk is the metadata chunk".to_string(),
        ));
    };
    if *last != ChunkMeta::meta() {
        return Err(Error::Serialization(format!(
            "LPG section: the last chunk is a {:?} chunk of graph {}, column {}, first row {}, \
             where the metadata chunk comes last",
            last.kind, last.graph_id, last.column_id, last.row_start
        )));
    }
    if let Some(index) = data.iter().position(|chunk| chunk.kind == ChunkKind::Meta) {
        return Err(Error::Serialization(format!(
            "LPG section: chunk {index} is a second metadata chunk, where only the last chunk \
             is one"
        )));
    }
    let meta = decode_lpg_meta(&source.fetch(data.len())?)?;
    let layout = Layout::new(&meta)?;

    let mut graphs: Vec<GraphTarget<'_>> = vec![GraphTarget::Default(store)];
    for graph in &meta.graphs[1..] {
        store.create_graph(&graph.name).map_err(|error| {
            Error::Internal(format!("LPG section: graph {:?}: {error}", graph.name))
        })?;
        let named = store.graph(&graph.name).ok_or_else(|| {
            Error::Internal(format!(
                "LPG section: graph {:?} was not created",
                graph.name
            ))
        })?;
        graphs.push(GraphTarget::Named(named));
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

    for (graph_id, (graph, target)) in meta.graphs.iter().zip(&reader.graphs).enumerate() {
        let target = target.store();
        target.set_next_node_id(target.next_node_id().max(graph.next_node_id));
        target.set_next_edge_id(target.next_edge_id().max(graph.next_edge_id));
        #[cfg(feature = "temporal")]
        target.sync_epoch(EpochId::new(meta.epoch));
        // The graph's names no row uses (a deleted node's label stays
        // registered): the graph keeps them, so the next write lists them
        // again. A name its rows use is not unused.
        let used: FxHashSet<String> = target.all_labels().into_iter().collect();
        for &id in &graph.unused_labels {
            let label = &meta.labels[usize::try_from(id).unwrap_or(usize::MAX)];
            if used.contains(label) {
                return Err(Error::Serialization(format!(
                    "LPG section, graph {graph_id}: label {label:?} is listed as unused, but a \
                     node of the graph has it"
                )));
            }
            target.register_label(label);
        }
        let used: FxHashSet<String> = target.all_edge_types().into_iter().collect();
        for &id in &graph.unused_edge_types {
            let edge_type = &meta.edge_types[usize::try_from(id).unwrap_or(usize::MAX)];
            if used.contains(edge_type) {
                return Err(Error::Serialization(format!(
                    "LPG section, graph {graph_id}: edge type {edge_type:?} is listed as unused, \
                     but an edge of the graph has it"
                )));
            }
            target.register_edge_type(edge_type);
        }
    }
    Ok(())
}

/// The store a graph of the section loads into.
enum GraphTarget<'s> {
    Default(&'s LpgStore),
    Named(Arc<LpgStore>),
}

impl GraphTarget<'_> {
    fn store(&self) -> &LpgStore {
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

/// The metadata chunk, checked.
struct Layout<'m> {
    meta: &'m LpgMeta,
    max_rows: u64,
    /// The property columns by id.
    columns: FxHashMap<u32, (Table, PropertyKey)>,
}

impl<'m> Layout<'m> {
    /// Checks rule 1 on `meta`.
    fn new(meta: &'m LpgMeta) -> Result<Self> {
        let refuse =
            |what: String| Err(Error::Serialization(format!("LPG metadata chunk: {what}")));
        if meta.max_rows == 0 || meta.max_bytes == 0 {
            return refuse(format!(
                "max_rows {} and max_bytes {}: both must be at least 1",
                meta.max_rows, meta.max_bytes
            ));
        }
        if meta.epoch > MAX_EPOCH {
            return refuse(format!("epoch {} is above {MAX_EPOCH}", meta.epoch));
        }
        let Some(default) = meta.graphs.first() else {
            return refuse("no graph, where graph 0 is the default graph".to_string());
        };
        if !default.name.is_empty() {
            return refuse(format!(
                "graph 0 is the default graph, without a name, found {:?}",
                default.name
            ));
        }
        if u32::try_from(meta.graphs.len() - 1).is_err() {
            return refuse(format!("{} graphs", meta.graphs.len()));
        }
        for pair in meta.graphs[1..].windows(2) {
            if pair[0].name >= pair[1].name {
                return refuse(format!(
                    "named graphs {:?} and {:?} are not in name order, each once",
                    pair[0].name, pair[1].name
                ));
            }
        }
        for (graph_id, graph) in meta.graphs.iter().enumerate() {
            for (ids, names, what) in [
                (&graph.unused_labels, &meta.labels, "label"),
                (&graph.unused_edge_types, &meta.edge_types, "edge type"),
            ] {
                let in_order = ids.windows(2).all(|pair| pair[0] < pair[1]);
                let known = ids
                    .last()
                    .is_none_or(|&last| usize::try_from(last).is_ok_and(|last| last < names.len()));
                if !in_order || !known {
                    return refuse(format!(
                        "graph {graph_id}: unused {what} ids {ids:?} are not ascending ids \
                         below the {} {what}s listed",
                        names.len()
                    ));
                }
            }
        }
        for (names, what) in [(&meta.labels, "label"), (&meta.edge_types, "edge type")] {
            if u32::try_from(names.len()).is_err() {
                return refuse(format!("{} {what} names", names.len()));
            }
            let mut seen = FxHashSet::default();
            if let Some(name) = names.iter().find(|name| !seen.insert(name.as_str())) {
                return refuse(format!("{what} {name:?} is listed twice"));
            }
        }
        let mut columns = FxHashMap::default();
        let mut keys = FxHashSet::default();
        for column in &meta.columns {
            if column.column_id < FIRST_PROPERTY_COLUMN {
                return refuse(format!(
                    "property column {} is below {FIRST_PROPERTY_COLUMN}, the first property \
                     column",
                    column.column_id
                ));
            }
            if !keys.insert((column.table, column.key.as_str())) {
                return refuse(format!(
                    "{} property {:?} has two columns",
                    column.table.entity(),
                    column.key
                ));
            }
            let key = PropertyKey::new(column.key.as_str());
            if columns
                .insert(column.column_id, (column.table, key))
                .is_some()
            {
                return refuse(format!("column {} is listed twice", column.column_id));
            }
        }
        Ok(Self {
            meta,
            max_rows: u64::from(meta.max_rows),
            columns,
        })
    }

    /// What `column_id` holds.
    fn role(&self, column_id: u32) -> std::result::Result<Role, String> {
        match column_id {
            COLUMN_LABELS => Ok(Role::Labels),
            COLUMN_SOURCE => Ok(Role::Endpoints),
            COLUMN_TARGET | COLUMN_EDGE_TYPE => Err(format!(
                "column {column_id} comes without column {COLUMN_SOURCE} of the same rows \
                 before it"
            )),
            _ => match self.columns.get(&column_id) {
                Some((table, key)) => Ok(Role::Property {
                    table: *table,
                    key: key.clone(),
                }),
                None => Err(format!("column {column_id} is not in the metadata chunk")),
            },
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
    /// One past the last row of the last chunk of each (kind, column).
    ends: BTreeMap<(u8, u32), u64>,
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
struct Reader<'m, 's> {
    layout: Layout<'m>,
    graphs: Vec<GraphTarget<'s>>,
    group: Option<ReadGroup>,
}

impl Reader<'_, '_> {
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
            "LPG section, chunk {index} (graph {}, column {}, rows from {})",
            chunk.graph_id, chunk.column_id, chunk.row_start
        );
        let refuse = |what: String| Error::Serialization(format!("{place}: {what}"));
        if !matches!(chunk.kind, ChunkKind::Column | ChunkKind::History) {
            return Err(refuse(format!(
                "a {:?} chunk, where the chunks before the metadata chunk are Column and \
                 History chunks",
                chunk.kind
            )));
        }
        let graph_id = chunk.graph_id;
        let graph_meta = usize::try_from(graph_id)
            .ok()
            .and_then(|at| self.layout.meta.graphs.get(at))
            .ok_or_else(|| refuse(format!("graph {graph_id} is not in the metadata chunk")))?;
        let role = self.layout.role(chunk.column_id).map_err(refuse)?;
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
        let graph_index = usize::try_from(graph_id).unwrap_or(usize::MAX);
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
            return Err(Error::Serialization(format!(
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
        let identity = (chunk.kind.to_byte(), chunk.column_id);
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
            && let Some(end) = group
                .ends
                .get(&(ChunkKind::Column.to_byte(), chunk.column_id))
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
        let names = &self.layout.meta.labels;
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
                label_ids(text, names.len()).map_err(|error| format!("node {row}: {error}"))?;
            let labels: Vec<&str> = labels.iter().map(|&id| names[id].as_str()).collect();
            target
                .create_node_with_id(NodeId::new(row), &labels)
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
        let edge_types = &self.layout.meta.edge_types;
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
            let name = usize::try_from(type_id)
                .ok()
                .and_then(|at| edge_types.get(at))
                .ok_or_else(|| {
                    format!(
                        "edge {row}: edge type {type_id} is past the edge type table of {}",
                        edge_types.len()
                    )
                })?;
            target_store
                .create_edge_with_id(
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
                #[cfg(feature = "temporal")]
                for (epoch, value) in versions {
                    set_value(target, table, row, key, value, epoch);
                }
                #[cfg(not(feature = "temporal"))]
                let _ = (versions, target, key);
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
            set_value(target, table, row, key, value, epoch);
            group.values.set(row - group.start);
        }
        Ok(())
    }
}

/// Sets `value` of property `key` of the node or edge `row`, at `epoch`.
#[cfg(feature = "temporal")]
fn set_value(
    target: &LpgStore,
    table: Table,
    row: u64,
    key: &PropertyKey,
    value: Value,
    epoch: u64,
) {
    let epoch = EpochId::new(epoch);
    match table {
        Table::Node => {
            target.set_node_property_at_epoch(NodeId::new(row), key.as_str(), value, epoch);
        }
        Table::Edge => {
            target.set_edge_property_at_epoch(EdgeId::new(row), key.as_str(), value, epoch);
        }
    }
}

/// Sets `value` of property `key` of the node or edge `row` (a build without
/// `temporal` keeps no epochs).
#[cfg(not(feature = "temporal"))]
fn set_value(
    target: &LpgStore,
    table: Table,
    row: u64,
    key: &PropertyKey,
    value: Value,
    _epoch: u64,
) {
    match table {
        Table::Node => target.set_node_property(NodeId::new(row), key.as_str(), value),
        Table::Edge => target.set_edge_property(EdgeId::new(row), key.as_str(), value),
    }
}

/// Fetches and decodes the column chunk at `index`.
fn decode(source: &dyn SectionSource, index: usize, chunk: &ChunkMeta) -> Result<ColumnChunk> {
    let bytes: Bytes = source.fetch(index)?;
    decode_column_chunk_bytes(&bytes, chunk.codec, chunk.row_count)
}

/// `error` with `place` before its message, when it is a serialization error.
fn prefixed(place: &str, error: Error) -> Error {
    match error {
        Error::Serialization(message) => Error::Serialization(format!("{place}: {message}")),
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

/// The label ids of a labels value: decimal ids below `count`, ascending,
/// joined by `,` (empty for none).
fn label_ids(text: &str, count: usize) -> std::result::Result<Vec<usize>, String> {
    if text.is_empty() {
        return Ok(Vec::new());
    }
    let mut ids: Vec<usize> = Vec::new();
    for part in text.split(',') {
        let canonical = !part.is_empty()
            && part.bytes().all(|byte| byte.is_ascii_digit())
            && (part == "0" || !part.starts_with('0'));
        let id = canonical
            .then(|| part.parse::<usize>().ok())
            .flatten()
            .ok_or_else(|| format!("labels {text:?}: {part:?} is not a label id"))?;
        if id >= count {
            return Err(format!("label {id} is past the label table of {count}"));
        }
        if ids.last().is_some_and(|&last| last >= id) {
            return Err(format!("labels {text:?}: label ids are not ascending"));
        }
        ids.push(id);
    }
    Ok(ids)
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
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Arc;

    use bytes::Bytes;
    use grafeo_common::storage::SectionType;
    use grafeo_common::storage::value_codec::MAX_PROPERTY_VALUE_DEPTH;
    use grafeo_common::storage::{
        ChunkCaps, ChunkKind, ChunkMeta, ImageSource, MemoryImage, SectionSink, SectionSource,
    };
    use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};
    use grafeo_common::utils::error::{Error, Result};

    use super::{
        COLUMN_EDGE_TYPE, COLUMN_LABELS, COLUMN_SOURCE, COLUMN_TARGET, ColumnMeta,
        FIRST_PROPERTY_COLUMN, LPG_SECTION_VERSION, LpgMeta, Table, decode_lpg_meta,
        encode_lpg_meta, read_lpg_chunks, write_lpg_chunks,
    };
    use crate::codec::column_chunk::{ColumnChunk, decode_column_chunk};
    use crate::graph::lpg::LpgStore;

    /// The chunks `write_lpg_chunks` writes for `store`, in order.
    fn try_chunks(store: &LpgStore, caps: ChunkCaps) -> Result<Vec<(ChunkMeta, Bytes)>> {
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::LpgStore, LPG_SECTION_VERSION)?;
        write_lpg_chunks(store, caps, &mut image)?;
        let Some(section) = image.take_section(SectionType::LpgStore) else {
            return Ok(Vec::new());
        };
        let metas = section.chunks().to_vec();
        metas
            .into_iter()
            .enumerate()
            .map(|(index, meta)| Ok((meta, section.fetch(index)?)))
            .collect()
    }

    fn chunks(store: &LpgStore, caps: ChunkCaps) -> Vec<(ChunkMeta, Bytes)> {
        try_chunks(store, caps).unwrap()
    }

    /// The metadata chunk, decoded: the last chunk.
    fn meta(chunks: &[(ChunkMeta, Bytes)]) -> LpgMeta {
        let (last, bytes) = chunks.last().expect("a section holds its metadata chunk");
        assert_eq!(*last, ChunkMeta::meta(), "the metadata chunk comes last");
        decode_lpg_meta(bytes).unwrap()
    }

    /// The chunks before the metadata chunk.
    fn data(chunks: &[(ChunkMeta, Bytes)]) -> &[(ChunkMeta, Bytes)] {
        &chunks[..chunks.len() - 1]
    }

    /// The decoded `kind` chunk of `graph`, `column` and first row
    /// `row_start` (exactly one).
    fn decode_kind(
        chunks: &[(ChunkMeta, Bytes)],
        kind: ChunkKind,
        graph: u32,
        column: u32,
        row_start: u64,
    ) -> ColumnChunk {
        let found: Vec<&(ChunkMeta, Bytes)> = chunks
            .iter()
            .filter(|(meta, _)| {
                meta.kind == kind
                    && meta.graph_id == graph
                    && meta.column_id == column
                    && meta.row_start == row_start
            })
            .collect();
        assert_eq!(
            found.len(),
            1,
            "one {kind:?} chunk of graph {graph}, column {column}, row {row_start}"
        );
        let (meta, bytes) = found[0];
        decode_column_chunk(bytes, meta.codec, meta.row_count).unwrap()
    }

    /// The decoded `Column` chunk of `graph`, `column` and first row
    /// `row_start` (exactly one).
    fn decode(
        chunks: &[(ChunkMeta, Bytes)],
        graph: u32,
        column: u32,
        row_start: u64,
    ) -> ColumnChunk {
        decode_kind(chunks, ChunkKind::Column, graph, column, row_start)
    }

    /// Every value of the `kind` chunks of `graph` and `column`, at its row,
    /// with its epoch (0 when the chunk carries none).
    fn cells(
        chunks: &[(ChunkMeta, Bytes)],
        kind: ChunkKind,
        graph: u32,
        column: u32,
    ) -> Vec<(u64, Value, u64)> {
        let mut cells = Vec::new();
        for (meta, bytes) in chunks {
            if meta.kind != kind || meta.graph_id != graph || meta.column_id != column {
                continue;
            }
            let chunk = decode_column_chunk(bytes, meta.codec, meta.row_count).unwrap();
            for (index, (offset, value)) in chunk.values.into_iter().enumerate() {
                let epoch = chunk.epochs.as_ref().map_or(0, |epochs| epochs[index]);
                cells.push((meta.row_start + u64::from(offset), value, epoch));
            }
        }
        cells
    }

    /// The table of `column` in `meta`.
    fn table_of(meta: &LpgMeta, column: u32) -> Table {
        match column {
            COLUMN_LABELS => Table::Node,
            COLUMN_SOURCE | COLUMN_TARGET | COLUMN_EDGE_TYPE => Table::Edge,
            _ => {
                meta.columns
                    .iter()
                    .find(|known| known.column_id == column)
                    .unwrap_or_else(|| panic!("column {column} is not in the metadata"))
                    .table
            }
        }
    }

    /// The rows of a chunk that hold a value.
    fn rows_of(meta: &ChunkMeta, bytes: &Bytes) -> Vec<u64> {
        decode_column_chunk(bytes, meta.codec, meta.row_count)
            .unwrap()
            .values
            .iter()
            .map(|(offset, _)| meta.row_start + u64::from(*offset))
            .collect()
    }

    /// Asserts the rules a reader of the section checks: the metadata chunk
    /// first and only there; every other chunk a `Column` or `History`
    /// chunk (`History` only for property columns) of a known graph and
    /// column, inside one row group and below the table's next id; chunks
    /// grouped by (graph, table, row group) in that order; each (kind,
    /// column) ascending without overlap within its group; the edge columns
    /// as three consecutive chunks of one range and the same rows; and
    /// every property row's node or edge in its group.
    fn assert_layout(chunks: &[(ChunkMeta, Bytes)]) {
        let meta = meta(chunks);
        let max_rows = u64::from(meta.max_rows);
        let mut last_group: Option<(u32, Table, u64)> = None;
        let mut ends: BTreeMap<(u8, u32), u64> = BTreeMap::new();
        let mut entities: BTreeSet<u64> = BTreeSet::new();
        let mut property_rows: Vec<(u32, u64)> = Vec::new();
        let check_group = |entities: &BTreeSet<u64>, property_rows: &[(u32, u64)]| {
            for (column, row) in property_rows {
                assert!(
                    entities.contains(row),
                    "column {column}, row {row} has no entity in its group"
                );
            }
        };
        assert!(
            data(chunks)
                .iter()
                .all(|(chunk, _)| chunk.kind != ChunkKind::Meta),
            "one metadata chunk"
        );
        let mut index = 0;
        while index < chunks.len() - 1 {
            let (chunk, bytes) = &chunks[index];
            assert!(
                matches!(chunk.kind, ChunkKind::Column | ChunkKind::History),
                "chunk {index} is a {:?} chunk",
                chunk.kind
            );
            let graph = meta
                .graphs
                .get(chunk.graph_id as usize)
                .unwrap_or_else(|| panic!("chunk {index}: graph {} is unknown", chunk.graph_id));
            let table = table_of(&meta, chunk.column_id);
            if chunk.kind == ChunkKind::History {
                assert!(
                    chunk.column_id >= FIRST_PROPERTY_COLUMN,
                    "chunk {index}: history of a fixed column"
                );
            }
            let next_id = match table {
                Table::Node => graph.next_node_id,
                Table::Edge => graph.next_edge_id,
            };
            assert!(
                chunk.row_count >= 1 && chunk.row_count <= meta.max_rows,
                "chunk {index}: {} rows",
                chunk.row_count
            );
            let last_row = chunk.row_start + u64::from(chunk.row_count) - 1;
            assert_eq!(
                chunk.row_start / max_rows,
                last_row / max_rows,
                "chunk {index}: rows {}..={last_row} cross a row group",
                chunk.row_start
            );
            assert!(
                last_row < next_id,
                "chunk {index}: row {last_row} is past the next id {next_id}"
            );
            let group = (chunk.graph_id, table, chunk.row_start / max_rows);
            if last_group != Some(group) {
                assert!(
                    last_group.is_none_or(|last| last < group),
                    "chunk {index}: group {group:?} after {last_group:?}"
                );
                check_group(&entities, &property_rows);
                last_group = Some(group);
                ends.clear();
                entities.clear();
                property_rows.clear();
            }
            let end = ends
                .entry((chunk.kind.to_byte(), chunk.column_id))
                .or_insert(0);
            assert!(
                chunk.row_start >= *end,
                "chunk {index}: overlaps the chunk before it"
            );
            *end = last_row + 1;
            let rows = rows_of(chunk, bytes);
            match chunk.column_id {
                COLUMN_LABELS => entities.extend(rows),
                COLUMN_SOURCE => {
                    for (step, column) in [(1, COLUMN_TARGET), (2, COLUMN_EDGE_TYPE)] {
                        let (partner, partner_bytes) = &chunks[index + step];
                        assert_eq!(
                            (
                                partner.kind,
                                partner.column_id,
                                partner.row_start,
                                partner.row_count
                            ),
                            (ChunkKind::Column, column, chunk.row_start, chunk.row_count),
                            "chunk {index}: the edge columns come together"
                        );
                        assert_eq!(
                            rows_of(partner, partner_bytes),
                            rows,
                            "chunk {index}: edge rows"
                        );
                    }
                    entities.extend(rows);
                    index += 2;
                }
                COLUMN_TARGET | COLUMN_EDGE_TYPE => panic!("chunk {index}: an edge column alone"),
                column => property_rows.extend(rows.into_iter().map(|row| (column, row))),
            }
            index += 1;
        }
        check_group(&entities, &property_rows);
    }

    fn caps(max_rows: u32, max_bytes: u32) -> ChunkCaps {
        ChunkCaps {
            max_rows,
            max_bytes,
        }
    }

    /// A string inside `depth` lists.
    fn nested(depth: usize) -> Value {
        let mut value = Value::from("Prague");
        for _ in 0..depth {
            value = Value::List(Arc::from(vec![value]));
        }
        value
    }

    #[test]
    fn an_empty_store_writes_only_its_meta_chunk() {
        let written = chunks(&LpgStore::new().unwrap(), ChunkCaps::DEFAULT);
        assert_eq!(written.len(), 1);
        assert_eq!(written[0].0, ChunkMeta::meta(), "only the metadata chunk");
        let meta = meta(&written);
        assert_eq!(
            (meta.layout, meta.graphs.len(), meta.graphs[0].name.as_str()),
            (1, 1, "")
        );
        assert!(meta.labels.is_empty() && meta.columns.is_empty());
        assert_eq!(
            (meta.max_rows, meta.max_bytes, meta.epoch),
            (ChunkCaps::DEFAULT.max_rows, ChunkCaps::DEFAULT.max_bytes, 0)
        );
    }

    #[test]
    fn nodes_edges_and_properties_become_column_chunks() {
        let store = LpgStore::new().unwrap();
        let alix = store.create_node(&["Person", "Employee"]);
        let gus = store.create_node(&["Person"]);
        let amsterdam = store.create_node(&["City"]);
        store.set_node_property(alix, "name", Value::from("Alix"));
        store.set_node_property(gus, "name", Value::from("Gus"));
        store.set_node_property(amsterdam, "population", Value::Int64(921_402));
        let knows = store.create_edge(alix, gus, "KNOWS");
        store.set_edge_property(knows, "since", Value::Int64(2019));
        store.create_graph("trips").unwrap();
        store.graph("trips").unwrap().create_node(&["City"]);
        let written = chunks(&store, caps(2, 1 << 20));
        assert_layout(&written);
        let meta = meta(&written);
        assert_eq!(meta.labels, ["City", "Employee", "Person"]);
        assert_eq!(meta.edge_types, ["KNOWS"]);
        assert_eq!(
            meta.graphs
                .iter()
                .map(|g| (g.name.as_str(), g.next_node_id, g.next_edge_id))
                .collect::<Vec<_>>(),
            [("", 3, 1), ("trips", 1, 0)]
        );
        assert_eq!(
            meta.columns,
            [
                ColumnMeta {
                    column_id: 16,
                    table: Table::Node,
                    key: "name".into()
                },
                ColumnMeta {
                    column_id: 17,
                    table: Table::Node,
                    key: "population".into()
                },
                ColumnMeta {
                    column_id: 18,
                    table: Table::Edge,
                    key: "since".into()
                },
            ]
        );
        let labels = decode(&written, 0, COLUMN_LABELS, 0);
        assert_eq!(
            labels.values,
            [(0, Value::from("1,2")), (1, Value::from("2"))]
        );
        assert_eq!(
            decode(&written, 0, COLUMN_LABELS, 2).values,
            [(0, Value::from("0"))]
        );
        assert_eq!(
            decode(&written, 0, 16, 0).values,
            [(0, Value::from("Alix")), (1, Value::from("Gus"))]
        );
        assert_eq!(
            decode(&written, 0, 17, 2).values,
            [(0, Value::Int64(921_402))]
        );

        let edge_columns: Vec<(u32, u64, u32)> = written
            .iter()
            .filter(|(meta, _)| meta.graph_id == 0 && (1..=3).contains(&meta.column_id))
            .map(|(meta, _)| (meta.column_id, meta.row_start, meta.row_count))
            .collect();
        assert_eq!(edge_columns, [(1, 0, 1), (2, 0, 1), (3, 0, 1)]);
        assert_eq!(
            decode(&written, 0, COLUMN_SOURCE, 0).values,
            [(0, Value::Int64(0))]
        );
        assert_eq!(
            decode(&written, 0, COLUMN_TARGET, 0).values,
            [(0, Value::Int64(1))]
        );
        assert_eq!(
            decode(&written, 0, COLUMN_EDGE_TYPE, 0).values,
            [(0, Value::Int64(0))]
        );
        assert_eq!(decode(&written, 0, 18, 0).values, [(0, Value::Int64(2019))]);

        // The named graph has its own labels chunk, with the file's label ids.
        assert_eq!(
            decode(&written, 1, COLUMN_LABELS, 0).values,
            [(0, Value::from("0"))]
        );
        let graph_ids: BTreeSet<u32> = data(&written)
            .iter()
            .map(|(meta, _)| meta.graph_id)
            .collect();
        assert_eq!(graph_ids, BTreeSet::from([0, 1]));
    }

    #[test]
    fn an_empty_named_graph_is_listed_without_chunks() {
        let store = LpgStore::new().unwrap();
        store.create_node(&["Person"]);
        store.create_graph("travel").unwrap();
        let written = chunks(&store, ChunkCaps::DEFAULT);
        assert_layout(&written);
        let meta = meta(&written);
        assert_eq!(meta.graphs[1].name, "travel");
        assert!(
            data(&written).iter().all(|(chunk, _)| chunk.graph_id == 0),
            "the empty graph has no chunks"
        );
    }

    /// A store of 40 nodes with mixed properties and labels, some deleted,
    /// and edges between them.
    fn mixed_store() -> LpgStore {
        let store = LpgStore::new().unwrap();
        let names = ["Alix", "Gus", "Vincent", "Mia", "Jules"];
        let nodes: Vec<NodeId> = (0..40usize)
            .map(|i| {
                let labels: &[&str] = match i % 4 {
                    0 => &["Person"],
                    1 => &["Person", "Employee"],
                    2 => &[],
                    _ => &["City"],
                };
                let id = store.create_node(labels);
                let name = names[i % 5].repeat(1 + i % 7);
                store.set_node_property(id, "name", Value::from(name));
                if i % 2 == 0 {
                    store.set_node_property(id, "age", Value::Int64(i64::try_from(i).unwrap() * 3));
                }
                if i % 3 == 0 {
                    let i = f64::from(u32::try_from(i).unwrap());
                    store.set_node_property(id, "score", Value::Float64(1.88 * i));
                }
                if i % 5 == 0 {
                    let stops = vec![Value::from("Paris"), Value::Int64(19)];
                    store.set_node_property(id, "stops", Value::List(Arc::from(stops)));
                }
                id
            })
            .collect();
        for pair in nodes.windows(2).step_by(3) {
            let edge = store.create_edge(
                pair[0],
                pair[1],
                if pair[0].0 % 2 == 0 {
                    "KNOWS"
                } else {
                    "VISITED"
                },
            );
            store.set_edge_property(
                edge,
                "since",
                Value::Int64(1988 + i64::try_from(pair[0].0).unwrap()),
            );
        }
        for i in [3, 4, 8, 9, 10, 11, 39] {
            store.delete_node(nodes[i]);
        }
        store
    }

    #[test]
    fn every_chunk_stays_inside_its_row_group() {
        let store = mixed_store();
        let written = chunks(&store, caps(4, 200));
        assert_layout(&written);
        let meta = meta(&written);
        for (chunk, _) in data(&written) {
            let last = chunk.row_start + u64::from(chunk.row_count) - 1;
            assert_eq!(chunk.row_start / 4, last / 4, "{chunk:?}");
            let graph = &meta.graphs[chunk.graph_id as usize];
            let next = if chunk.column_id == COLUMN_LABELS
                || meta
                    .columns
                    .iter()
                    .any(|c| c.column_id == chunk.column_id && c.table == Table::Node)
            {
                graph.next_node_id
            } else {
                graph.next_edge_id
            };
            assert!(last < next, "{chunk:?} ends past the next id {next}");
        }

        // The byte cap cut some groups: more name chunks than groups with names.
        let name = meta
            .columns
            .iter()
            .find(|c| c.key == "name")
            .unwrap()
            .column_id;
        let name_chunks: Vec<u64> = written
            .iter()
            .filter(|(chunk, _)| chunk.column_id == name)
            .map(|(chunk, _)| chunk.row_start / 4)
            .collect();
        let groups: BTreeSet<u64> = name_chunks.iter().copied().collect();
        assert!(
            name_chunks.len() > groups.len(),
            "the byte cap cut a group: {name_chunks:?}"
        );

        // Every live value is written once, at its row; deleted nodes are left out.
        let expected: Vec<(u64, Value, u64)> = store
            .node_ids()
            .into_iter()
            .filter_map(|id| {
                store
                    .get_node_property(id, &PropertyKey::new("name"))
                    .map(|value| (id.0, value, 0))
            })
            .collect();
        assert_eq!(expected.len(), 33);
        assert_eq!(cells(&written, ChunkKind::Column, 0, name), expected);
        let labels = cells(&written, ChunkKind::Column, 0, COLUMN_LABELS);
        assert_eq!(labels.len(), 33, "one labels value per live node");
        assert!(
            labels.contains(&(2, Value::from(""), 0)),
            "a node without labels has an empty labels value"
        );
        let deleted: BTreeSet<u64> = [3, 4, 8, 9, 10, 11, 39].into();
        for column in [name, COLUMN_LABELS] {
            assert!(
                cells(&written, ChunkKind::Column, 0, column)
                    .iter()
                    .all(|(row, _, _)| !deleted.contains(row)),
                "column {column} holds a deleted node"
            );
        }
        let edges = cells(&written, ChunkKind::Column, 0, COLUMN_SOURCE);
        assert_eq!(
            edges.len(),
            store.edge_count(),
            "one endpoint value per live edge"
        );
        assert_eq!(meta.edge_types, ["KNOWS", "VISITED"]);
        for (row, edge_type, _) in cells(&written, ChunkKind::Column, 0, COLUMN_EDGE_TYPE) {
            let name = store.edge_type(EdgeId::new(row)).unwrap();
            let id = meta
                .edge_types
                .iter()
                .position(|known| known == name.as_str())
                .unwrap();
            assert_eq!(
                edge_type,
                Value::Int64(i64::try_from(id).unwrap()),
                "edge {row}"
            );
        }
        for (row, source, _) in edges {
            let edge = store.get_edge(EdgeId::new(row)).unwrap();
            assert_eq!(
                source,
                Value::Int64(i64::try_from(edge.src.0).unwrap()),
                "edge {row}"
            );
        }
    }

    #[test]
    fn edges_created_out_of_id_order_are_written_in_id_order() {
        let store = LpgStore::new().unwrap();
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        let ids = [88u64, 3, 19_000, 7, 1_000_003, 40, 5, 300_019, 8_803, 19];
        for id in ids {
            store
                .create_edge_with_id(EdgeId::new(id), alix, gus, "KNOWS")
                .unwrap();
            store.set_edge_property(EdgeId::new(id), "since", Value::Int64(1988));
        }
        let written = chunks(&store, caps(4, 1 << 20));
        assert_layout(&written);
        let mut sorted = ids.to_vec();
        sorted.sort_unstable();
        let rows = |column: u32| -> Vec<u64> {
            cells(&written, ChunkKind::Column, 0, column)
                .into_iter()
                .map(|(row, _, _)| row)
                .collect()
        };
        assert_eq!(rows(COLUMN_SOURCE), sorted, "endpoints");
        let since = meta(&written).columns[0].column_id;
        assert_eq!(rows(since), sorted, "the edge property");
    }

    #[test]
    fn nodes_and_edges_not_visible_now_are_left_out_with_their_values() {
        use grafeo_common::types::TransactionId;

        let store = LpgStore::new().unwrap();
        let alix = store.create_node(&["Person"]);
        store.set_node_property(alix, "name", Value::from("Alix"));
        // A transaction still open: its node, edge and values are not visible.
        let transaction = TransactionId::new(19);
        let epoch = store.current_epoch();
        let gus = store.create_node_versioned(&["Person"], epoch, transaction);
        store
            .set_node_property_versioned(gus, "name", Value::from("Gus"), transaction)
            .unwrap();
        let knows = store.create_edge_versioned(alix, gus, "KNOWS", epoch, transaction);
        store.set_edge_property_versioned(knows, "since", Value::Int64(3), transaction);
        // A value set on an id no node has.
        store.set_node_property(NodeId::new(88), "name", Value::from("Vincent"));

        let written = chunks(&store, caps(4, 1 << 20));
        assert_layout(&written);
        assert_eq!(
            cells(&written, ChunkKind::Column, 0, COLUMN_LABELS),
            [(0, Value::from("0"), 0)]
        );
        assert_eq!(
            cells(&written, ChunkKind::Column, 0, 16),
            [(0, Value::from("Alix"), 0)],
            "only the visible node's value"
        );
        assert!(
            data(&written)
                .iter()
                .all(|(chunk, _)| chunk.column_id == COLUMN_LABELS || chunk.column_id == 16),
            "no edge chunks: {:?}",
            written
                .iter()
                .map(|(chunk, _)| chunk.column_id)
                .collect::<Vec<_>>()
        );
    }

    /// A store whose next id is below a row it holds (never so in a store
    /// that allocated its ids) records the next id after that row, so a load
    /// never gives a new node an id in use.
    #[test]
    fn the_next_ids_cover_every_row_written() {
        let store = LpgStore::new().unwrap();
        let nodes: Vec<NodeId> = (0..3).map(|_| store.create_node(&["Person"])).collect();
        store.create_edge(nodes[0], nodes[2], "KNOWS");
        store.set_next_node_id(1);
        store.set_next_edge_id(0);
        let written = chunks(&store, ChunkCaps::DEFAULT);
        assert_layout(&written);
        let graph = &meta(&written).graphs[0];
        assert_eq!((graph.next_node_id, graph.next_edge_id), (3, 1));
    }

    #[test]
    fn ids_far_above_zero_write_only_the_groups_that_hold_them() {
        // An overlay above a compacted base starts its ids far above 0, here
        // in the middle of a row group: the groups stay aligned to max_rows.
        let overlay = LpgStore::new().unwrap();
        overlay.set_next_node_id((1 << 40) + 2);
        overlay.set_next_edge_id(1 << 41);
        let cities: Vec<NodeId> = (0..4).map(|_| overlay.create_node(&["City"])).collect();
        overlay.create_edge(cities[0], cities[1], "ROUTE");
        let written = chunks(&overlay, caps(4, 1 << 20));
        assert_layout(&written);
        let ranges: Vec<(u32, u64, u32)> = data(&written)
            .iter()
            .map(|(chunk, _)| (chunk.column_id, chunk.row_start, chunk.row_count))
            .collect();
        assert_eq!(
            ranges,
            [
                (0, (1 << 40) + 2, 2),
                (0, (1 << 40) + 4, 2),
                (1, 1 << 41, 1),
                (2, 1 << 41, 1),
                (3, 1 << 41, 1)
            ]
        );
        assert_eq!(meta(&written).graphs[0].next_node_id, (1 << 40) + 6);
    }

    #[test]
    fn ids_near_the_top_of_the_id_space_are_written_and_large_endpoints_refused() {
        let store = LpgStore::new().unwrap();
        let top = NodeId::new(u64::MAX - 1);
        store.create_node_with_id(top, &["Person"]).unwrap();
        let written = chunks(&store, caps(4, 1 << 20));
        assert_layout(&written);
        assert_eq!(
            decode(&written, 0, COLUMN_LABELS, u64::MAX - 1).values,
            [(0, Value::from("0"))]
        );

        let alix = store
            .create_node_with_id(NodeId::new(3), &["Person"])
            .map(|()| NodeId::new(3))
            .unwrap();
        store
            .create_edge_with_id(EdgeId::new(19), alix, top, "KNOWS")
            .unwrap();
        let error = try_chunks(&store, caps(4, 1 << 20))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("edge 19") && error.contains(&(u64::MAX - 1).to_string()),
            "an endpoint past i64::MAX is refused, naming the edge: {error}"
        );
    }

    /// What a reader refuses is refused at write, naming it: a node or edge
    /// with the largest id (no next id follows it), and with `temporal` a
    /// value epoch or a store epoch above `i64::MAX`.
    #[test]
    fn ids_and_epochs_a_reader_refuses_are_refused_at_write() {
        let store = LpgStore::new().unwrap();
        store
            .create_node_with_id(NodeId::new(u64::MAX - 1), &["Person"])
            .unwrap();
        assert!(try_chunks(&store, ChunkCaps::DEFAULT).is_ok());
        // The id counter wraps after giving out the largest id.
        let edges = LpgStore::new().unwrap();
        let alix = edges.create_node(&["Person"]);
        edges.set_next_edge_id(u64::MAX);
        assert_eq!(
            edges.create_edge(alix, alix, "KNOWS"),
            EdgeId::new(u64::MAX)
        );
        let error = try_chunks(&edges, ChunkCaps::DEFAULT).unwrap_err();
        assert!(matches!(error, Error::Serialization(_)), "{error:?}");
        assert!(
            error.to_string().contains(&format!("edge {}", u64::MAX)),
            "{error}"
        );

        #[cfg(feature = "temporal")]
        {
            use grafeo_common::types::EpochId;

            let late = LpgStore::new().unwrap();
            let gus = late.create_node(&["Person"]);
            late.set_node_property_at_epoch(
                gus,
                "city",
                Value::from("Paris"),
                EpochId::new(u64::MAX - 1),
            );
            let error = try_chunks(&late, ChunkCaps::DEFAULT)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("node 0, property \"city\": epoch"),
                "{error}"
            );
            let ahead = LpgStore::new().unwrap();
            ahead.sync_epoch(EpochId::new(u64::MAX - 1));
            let error = try_chunks(&ahead, ChunkCaps::DEFAULT)
                .unwrap_err()
                .to_string();
            assert!(error.contains("store epoch"), "{error}");
        }
    }

    #[cfg(feature = "temporal")]
    #[test]
    fn older_versions_go_to_history_chunks_before_their_column() {
        use grafeo_common::types::{EpochId, TransactionId};

        let store = LpgStore::new().unwrap();
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        let mia = store.create_node(&["Person"]);
        let at = EpochId::new;
        store.set_node_property_at_epoch(alix, "city", Value::from("Amsterdam"), at(3));
        store.set_node_property_at_epoch(alix, "city", Value::from("Berlin"), at(19));
        store.set_node_property_at_epoch(alix, "city", Value::from("Paris"), at(88));
        store.set_node_property_at_epoch(gus, "city", Value::from("Prague"), at(3));
        store.set_node_property_at_epoch(gus, "city", Value::Null, at(19));
        store.set_node_property_at_epoch(mia, "city", Value::from("Berlin"), at(19));
        // An uncommitted version: a checkpoint holds committed data only.
        store
            .set_node_property_versioned(mia, "city", Value::from("Prague"), TransactionId::new(88))
            .unwrap();
        let knows = store.create_edge(alix, gus, "KNOWS");
        store.set_edge_property_at_epoch(knows, "since", Value::Int64(3), at(3));
        store.set_edge_property_at_epoch(knows, "since", Value::Int64(19), at(19));
        store.sync_epoch(at(88));

        let written = chunks(&store, ChunkCaps::DEFAULT);
        assert_layout(&written);
        let meta = meta(&written);
        assert_eq!(meta.epoch, 88);
        let city = 16;
        let position = |kind: ChunkKind, column: u32| {
            written
                .iter()
                .position(|(chunk, _)| chunk.kind == kind && chunk.column_id == column)
                .unwrap()
        };
        assert_eq!(
            position(ChunkKind::History, city) + 1,
            position(ChunkKind::Column, city),
            "the history chunk comes right before its column chunk"
        );
        let column = decode(&written, 0, city, 0);
        assert_eq!(
            column.values,
            [(0, Value::from("Paris")), (2, Value::from("Berlin"))]
        );
        assert_eq!(column.epochs, Some(vec![88, 19]));
        let version =
            |epoch: i64, value: Value| Value::List(Arc::from(vec![Value::Int64(epoch), value]));
        let list = |items: Vec<Value>| Value::List(Arc::from(items));
        let history = decode_kind(&written, ChunkKind::History, 0, city, 0);
        assert_eq!(
            history.values,
            [
                (
                    0,
                    list(vec![
                        version(3, Value::from("Amsterdam")),
                        version(19, Value::from("Berlin"))
                    ])
                ),
                (
                    1,
                    list(vec![
                        version(3, Value::from("Prague")),
                        version(19, Value::Null)
                    ])
                ),
            ]
        );
        assert_eq!(history.epochs, None, "history values hold their epochs");

        let since = meta
            .columns
            .iter()
            .find(|c| c.key == "since")
            .unwrap()
            .column_id;
        assert_eq!(
            cells(&written, ChunkKind::Column, 0, since),
            [(0, Value::Int64(19), 19)]
        );
        assert_eq!(
            cells(&written, ChunkKind::History, 0, since),
            [(0, list(vec![version(3, Value::Int64(3))]), 0)]
        );
    }

    #[cfg(feature = "temporal")]
    #[test]
    fn a_range_whose_properties_were_all_removed_writes_its_history_chunk_alone() {
        use grafeo_common::types::EpochId;

        let store = LpgStore::new().unwrap();
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        store.set_node_property_at_epoch(alix, "city", Value::from("Amsterdam"), EpochId::new(3));
        store.set_node_property_at_epoch(alix, "city", Value::Null, EpochId::new(19));
        store.set_node_property_at_epoch(gus, "city", Value::from("Berlin"), EpochId::new(3));
        let written = chunks(&store, caps(1, 1 << 20));
        assert_layout(&written);
        let city: Vec<(ChunkKind, u64, u32)> = written
            .iter()
            .filter(|(chunk, _)| chunk.column_id == 16)
            .map(|(chunk, _)| (chunk.kind, chunk.row_start, chunk.row_count))
            .collect();
        assert_eq!(
            city,
            [(ChunkKind::History, 0, 1), (ChunkKind::Column, 1, 1)],
            "Alix's removed city has a history chunk alone, Gus's city a column chunk alone"
        );
        let version =
            |epoch: i64, value: Value| Value::List(Arc::from(vec![Value::Int64(epoch), value]));
        assert_eq!(
            decode_kind(&written, ChunkKind::History, 0, 16, 0).values,
            [(
                0,
                Value::List(Arc::from(vec![
                    version(3, Value::from("Amsterdam")),
                    version(19, Value::Null)
                ]))
            )]
        );
        assert_eq!(
            decode(&written, 0, 16, 1).values,
            [(0, Value::from("Berlin"))]
        );
    }

    /// Builds the same store every call, with many labels, keys and graphs,
    /// so hash maps seeded differently would order them differently.
    fn sample_store() -> LpgStore {
        let store = LpgStore::new().unwrap();
        let labels = [
            "Person", "City", "Employee", "Cafe", "Museum", "Station", "Park", "Bridge",
        ];
        let keys = [
            "name", "age", "score", "city", "since", "tags", "rank", "zone",
        ];
        for graph in [None, Some("trips"), Some("travel"), Some("archive")] {
            let named;
            let target: &LpgStore = match graph {
                None => &store,
                Some(name) => {
                    store.create_graph(name).unwrap();
                    named = store.graph(name).unwrap();
                    &named
                }
            };
            let mut previous = None;
            for i in 0..30usize {
                let node_labels: Vec<&str> = labels
                    .iter()
                    .copied()
                    .filter(|l| (l.len() + i) % 3 == 0)
                    .collect();
                let id = target.create_node(&node_labels);
                for (k, key) in keys.iter().enumerate() {
                    if (i + k) % 3 != 0 {
                        target.set_node_property(id, key, Value::from(format!("{key} {i}")));
                    }
                }
                if let Some(previous) = previous {
                    let edge = target.create_edge(previous, id, labels[i % 8]);
                    target.set_edge_property(
                        edge,
                        keys[i % 8],
                        Value::Int64(i64::try_from(i).unwrap()),
                    );
                }
                previous = Some(id);
            }
        }
        store
    }

    #[test]
    fn writing_the_same_data_twice_gives_the_same_bytes() {
        let caps = caps(8, 300);
        let store = sample_store();
        let first = chunks(&store, caps);
        assert_layout(&first);
        assert!(first.len() > 40, "many chunks: {}", first.len());
        assert_eq!(first, chunks(&store, caps), "two writes of one store");
        for _ in 0..3 {
            assert_eq!(
                first,
                chunks(&sample_store(), caps),
                "a store built the same way"
            );
        }
    }

    #[cfg(not(feature = "temporal"))]
    #[test]
    fn a_spilled_column_is_read_once_per_value_and_an_unreadable_one_fails_the_write() {
        use crate::graph::lpg::test_backing::MemoryBacking;
        use std::sync::atomic::Ordering;

        let store = LpgStore::new().unwrap();
        let key = PropertyKey::new("embedding");
        let mut expected = Vec::new();
        for i in 0..10u32 {
            let id = store.create_node(&["Item"]);
            if i % 3 != 1 {
                let vector = Value::Vector(
                    vec![3.0 * f32::from(u16::try_from(i).unwrap()), 19.0, 88.0].into(),
                );
                store.set_node_property(id, "embedding", vector.clone());
                expected.push((id.0, vector, 0));
            }
        }
        let snapshot = store.node_property_column_entries(&key).unwrap();
        let backing = MemoryBacking::of(&snapshot);
        assert!(store.spill_node_property_column(&key, backing.clone(), &snapshot));

        let written = chunks(&store, caps(4, 1 << 20));
        assert_layout(&written);
        assert_eq!(cells(&written, ChunkKind::Column, 0, 16), expected);
        assert_eq!(
            backing.copies.load(Ordering::Relaxed),
            expected.len(),
            "each spilled value is read once"
        );

        backing.fail_reads(true);
        let error = try_chunks(&store, caps(4, 1 << 20)).unwrap_err();
        assert!(
            matches!(error, Error::Io(_)),
            "a read error fails the write: {error:?}"
        );
    }

    #[test]
    fn a_value_nested_too_deep_fails_the_write_naming_the_node_and_the_property() {
        let store = LpgStore::new().unwrap();
        store.create_graph("travel").unwrap();
        let travel = store.graph("travel").unwrap();
        let berlin = travel.create_node(&["City"]);
        let paris = travel.create_node(&["City"]);
        travel.set_node_property(paris, "stops", nested(MAX_PROPERTY_VALUE_DEPTH));
        let route = travel.create_edge(berlin, paris, "ROUTE");
        assert!(
            try_chunks(&store, ChunkCaps::DEFAULT).is_ok(),
            "the deepest value a write accepts"
        );

        // The direct store API takes it, as a 0.5.x database could hold it.
        travel.set_edge_property(route, "legs", nested(MAX_PROPERTY_VALUE_DEPTH + 1));
        let error = try_chunks(&store, ChunkCaps::DEFAULT).unwrap_err();
        assert!(matches!(error, Error::Serialization(_)), "{error:?}");
        let error = error.to_string();
        assert!(
            error.contains("graph \"travel\", edge 0, property \"legs\""),
            "the error names the graph, the edge and the property: {error}"
        );
        travel.set_edge_property(route, "legs", Value::Int64(3));

        travel.set_node_property(berlin, "stops", nested(MAX_PROPERTY_VALUE_DEPTH + 1));
        let error = try_chunks(&store, ChunkCaps::DEFAULT)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("graph \"travel\", node 0, property \"stops\""),
            "the error names the graph, the node and the property: {error}"
        );
    }

    /// A history value nests each older version two lists deeper; the codec
    /// leaves room for that above the write limit (Ruling 52), so a value at
    /// the write limit with older versions is written, and an older version
    /// past it is refused naming it.
    #[cfg(feature = "temporal")]
    #[test]
    fn older_versions_at_the_write_limit_fit_a_history_value() {
        use grafeo_common::types::EpochId;

        let store = LpgStore::new().unwrap();
        let mia = store.create_node(&["Person"]);
        let set = |value: Value, epoch: u64| {
            store.set_node_property_at_epoch(mia, "trips", value, EpochId::new(epoch));
        };
        set(nested(MAX_PROPERTY_VALUE_DEPTH), 3);
        set(nested(MAX_PROPERTY_VALUE_DEPTH), 19);
        set(Value::Int64(88), 88);
        let written = chunks(&store, ChunkCaps::DEFAULT);
        assert_layout(&written);
        let history = decode_kind(&written, ChunkKind::History, 0, 16, 0);
        assert_eq!(
            history.values.len(),
            1,
            "both older versions in one history value"
        );

        set(nested(MAX_PROPERTY_VALUE_DEPTH + 1), 89);
        set(Value::Int64(3), 90);
        let error = try_chunks(&store, ChunkCaps::DEFAULT)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("the default graph, node 0, property \"trips\": an older version"),
            "{error}"
        );
    }

    #[test]
    fn caps_of_zero_are_refused() {
        let store = LpgStore::new().unwrap();
        store.create_node(&["Person"]);
        for caps in [caps(0, 1 << 20), caps(4, 0)] {
            let error = try_chunks(&store, caps).unwrap_err();
            assert!(
                matches!(error, Error::InvalidValue(_)),
                "{caps:?}: {error:?}"
            );
        }
    }

    /// A named graph may have the empty name: graph 0 is the default graph by
    /// its position, so the name cannot collide with it.
    #[test]
    fn a_named_graph_may_have_the_empty_name() {
        let store = LpgStore::new().unwrap();
        store.create_node(&["Person"]);
        store.create_graph("").unwrap();
        store.create_graph("trips").unwrap();
        let unnamed = store.graph("").unwrap();
        unnamed.create_node(&["City"]);
        unnamed.create_node(&["City"]);
        let written = chunks(&store, ChunkCaps::DEFAULT);
        assert_layout(&written);
        let names: Vec<String> = meta(&written).graphs.into_iter().map(|g| g.name).collect();
        assert_eq!(names, ["", "", "trips"]);
        assert_eq!(
            cells(&written, ChunkKind::Column, 0, COLUMN_LABELS).len(),
            1
        );
        assert_eq!(
            cells(&written, ChunkKind::Column, 1, COLUMN_LABELS).len(),
            2
        );
    }

    /// Names a write meets that the registries did not hold when it began
    /// (an open transaction creates them; commits are held) follow the sorted
    /// names, in the order met; the same name keeps its id.
    #[test]
    fn names_met_during_the_write_follow_the_sorted_names() {
        let list = super::NameList::new(["Person".to_string(), "City".to_string()], "label");
        assert_eq!(list.id("City").unwrap(), 0);
        assert_eq!(list.id("Person").unwrap(), 1);
        assert_eq!(list.id("Museum").unwrap(), 2);
        assert_eq!(list.id("Cafe").unwrap(), 3);
        assert_eq!(
            list.id("Museum").unwrap(),
            2,
            "a name met again keeps its id"
        );
        assert_eq!(list.into_names(), ["City", "Person", "Museum", "Cafe"]);
    }

    /// Runs a hook after the first chunk it passes on.
    struct Hooked<'h> {
        inner: MemoryImage,
        hook: Option<Box<dyn FnOnce() + 'h>>,
    }

    impl SectionSink for Hooked<'_> {
        fn write_chunk(&mut self, meta: ChunkMeta, bytes: &[u8]) -> Result<()> {
            self.inner.write_chunk(meta, bytes)?;
            if let Some(hook) = self.hook.take() {
                hook();
            }
            Ok(())
        }
    }

    /// An open transaction adds a label no node had, a property key and an
    /// edge type while a checkpoint writes the store (commits are held, not
    /// transactions): the write lists the label it meets after the names it
    /// began with, and writes the node with it.
    #[test]
    fn names_an_open_transaction_adds_during_the_write_are_written() {
        use grafeo_common::types::TransactionId;

        let store = LpgStore::new().unwrap();
        let nodes: Vec<NodeId> = (0..6).map(|_| store.create_node(&["Person"])).collect();
        store.set_node_property(nodes[0], "name", Value::from("Alix"));
        let mut image = Hooked {
            inner: MemoryImage::new(),
            hook: Some(Box::new(|| {
                let transaction = TransactionId::new(19);
                assert!(store.add_label_versioned(nodes[5], "Traveller", transaction));
                store
                    .set_node_property_versioned(
                        nodes[5],
                        "city",
                        Value::from("Prague"),
                        transaction,
                    )
                    .unwrap();
                store.create_edge_versioned(
                    nodes[0],
                    nodes[1],
                    "VISITED",
                    store.current_epoch(),
                    transaction,
                );
            })),
        };
        image
            .inner
            .begin_section(SectionType::LpgStore, LPG_SECTION_VERSION)
            .unwrap();
        write_lpg_chunks(&store, caps(2, 1 << 20), &mut image)
            .expect("the write tolerates new names");
        assert!(image.hook.is_none(), "the hook ran during the write");
        let section = image.inner.take_section(SectionType::LpgStore).unwrap();
        let written: Vec<(ChunkMeta, Bytes)> = section
            .chunks()
            .iter()
            .enumerate()
            .map(|(index, chunk)| (*chunk, section.fetch(index).unwrap()))
            .collect();
        assert_layout(&written);
        let meta = meta(&written);
        // The registry holds the new label (as it does after a rollback): it
        // is listed after the names the write began with. Without `temporal`
        // node 5's label set holds it; with it, no row has it, so the graph
        // lists it as unused.
        assert_eq!(meta.labels, ["Person", "Traveller"]);
        #[cfg(not(feature = "temporal"))]
        assert_eq!(meta.graphs[0].unused_labels, Vec::<u32>::new());
        #[cfg(feature = "temporal")]
        assert_eq!(meta.graphs[0].unused_labels, [1]);
        // Without `temporal` the transaction's label set is written in place
        // (the store keeps no versions); with it, the set an open
        // transaction wrote is pending and not written.
        #[cfg(not(feature = "temporal"))]
        let node_5 = "0,1";
        #[cfg(feature = "temporal")]
        let node_5 = "0";
        assert_eq!(
            decode(&written, 0, COLUMN_LABELS, 4).values,
            [(0, Value::from("0")), (1, Value::from(node_5))]
        );
    }

    /// With `temporal`, labels an open transaction adds or removes stay out
    /// of the section: it holds the label sets of the current epoch.
    #[cfg(feature = "temporal")]
    #[test]
    fn labels_of_an_open_transaction_are_not_written() {
        use grafeo_common::types::TransactionId;

        let store = LpgStore::new().unwrap();
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person", "Employee"]);
        let transaction = TransactionId::new(88);
        assert!(store.add_label_versioned(alix, "Traveller", transaction));
        assert!(store.remove_label_versioned(gus, "Employee", transaction));
        let written = chunks(&store, ChunkCaps::DEFAULT);
        assert_layout(&written);
        let meta = meta(&written);
        let label = |name: &str| meta.labels.iter().position(|known| known == name).unwrap();
        assert_eq!(
            decode(&written, 0, COLUMN_LABELS, 0).values,
            [
                (0, Value::from(label("Person").to_string())),
                (
                    1,
                    Value::from(format!("{},{}", label("Employee"), label("Person")))
                ),
            ],
            "the committed label sets"
        );
    }

    #[test]
    fn the_metadata_decoder_refuses_crafted_lengths_layouts_and_trailing_bytes() {
        let mut meta = crafted_meta();
        meta.labels = vec!["Person".into(), "City".into()];
        let bytes = encode_lpg_meta(&meta).unwrap();
        assert_eq!(decode_lpg_meta(&bytes).unwrap(), meta);
        // layout, max_rows, max_bytes and epoch take 17 bytes; the label
        // count follows, then the first label's length.
        let at_count = 17;
        let at_length = at_count + 4;
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
            assert!(
                matches!(error, Error::Serialization(_)),
                "{what}: {error:?}"
            );
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
        let mut table = bytes.clone();
        let at_table = bytes.len() - 4 - "name".len() - 1;
        table[at_table] = 2;
        let error = decode_lpg_meta(&table).unwrap_err().to_string();
        assert!(error.contains("table 2"), "{error}");
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
        meta.labels = (0..1_000_000u32).map(|i| format!("L{i:07}")).collect();
        meta.edge_types = (0..66u8)
            .map(|i| format!("{i:02}{}", "x".repeat(1 << 20)))
            .collect();
        let bytes = encode_lpg_meta(&meta).unwrap();
        assert!(bytes.len() > 1 << 26, "{} bytes", bytes.len());
        assert_eq!(decode_lpg_meta(&bytes).unwrap(), meta);
    }

    // ── Reading (F4) ────────────────────────────────────────────────

    /// Writes `store` with `caps` and loads the chunks into `into`.
    fn load_into(store: &LpgStore, caps: ChunkCaps, into: &LpgStore) -> Result<()> {
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::LpgStore, LPG_SECTION_VERSION)?;
        write_lpg_chunks(store, caps, &mut image)?;
        let source = image
            .section_source(SectionType::LpgStore)
            .expect("the section holds its metadata chunk");
        read_lpg_chunks(into, &*source)
    }

    /// `store` written with `caps` and loaded into a new store.
    fn round_trip(store: &LpgStore, caps: ChunkCaps) -> LpgStore {
        let back = LpgStore::new().unwrap();
        load_into(store, caps, &back).unwrap();
        back
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

    /// Asserts that `a` and `b` hold the same nodes and edges (labels,
    /// endpoints, types and property values, floats by bits) in the default
    /// graph and in every named graph.
    fn assert_same_graph(a: &LpgStore, b: &LpgStore) {
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

    fn assert_same_store(a: &LpgStore, b: &LpgStore, graph: &str) {
        assert_eq!(a.node_ids(), b.node_ids(), "{graph}: node ids");
        for id in a.node_ids() {
            let (x, y) = (a.get_node(id).unwrap(), b.get_node(id).unwrap());
            let labels = |node: &crate::graph::lpg::Node| {
                let mut labels: Vec<String> = node.labels.iter().map(|l| l.to_string()).collect();
                labels.sort();
                labels
            };
            assert_eq!(labels(&x), labels(&y), "{graph}: labels of node {}", id.0);
            assert_eq!(
                property_bits(&x.properties),
                property_bits(&y.properties),
                "{graph}: properties of node {}",
                id.0
            );
        }
        assert_eq!(
            a.try_edge_ids().unwrap(),
            b.try_edge_ids().unwrap(),
            "{graph}: edge ids"
        );
        for id in a.try_edge_ids().unwrap() {
            let (x, y) = (a.get_edge(id).unwrap(), b.get_edge(id).unwrap());
            assert_eq!(
                (x.src, x.dst, x.edge_type.as_str()),
                (y.src, y.dst, y.edge_type.as_str()),
                "{graph}: edge {}",
                id.0
            );
            assert_eq!(
                property_bits(&x.properties),
                property_bits(&y.properties),
                "{graph}: properties of edge {}",
                id.0
            );
        }
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

    /// Every value kind as properties, nodes with 0, 1 and 3 labels, edges of
    /// two types with properties, and named graphs "trips", "" and
    /// "travel/__default__" (the last one empty).
    fn round_trip_store() -> LpgStore {
        let store = LpgStore::new().unwrap();
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

    #[test]
    fn a_store_round_trips_through_chunks() {
        let store = round_trip_store();
        for caps in [caps(4, 512), caps(1, 64), ChunkCaps::DEFAULT] {
            let back = round_trip(&store, caps);
            assert_same_graph(&store, &back);
            assert_eq!(back.next_node_id(), store.next_node_id(), "{caps:?}");
            assert_eq!(back.next_edge_id(), store.next_edge_id(), "{caps:?}");
            assert_eq!(
                chunks(&back, caps),
                chunks(&store, caps),
                "a loaded store writes the same chunks: {caps:?}"
            );
        }
    }

    /// A property whose value is null does not exist: the direct store API
    /// (and a 0.5.x load, which reads counters as null) can leave a null in a
    /// column, which the section writes as no value.
    #[cfg(not(feature = "temporal"))]
    #[test]
    fn a_null_value_is_written_as_no_value() {
        let store = LpgStore::new().unwrap();
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        store.set_node_property(alix, "visits", Value::Null);
        store.set_node_property(gus, "visits", Value::Int64(3));
        let written = chunks(&store, ChunkCaps::DEFAULT);
        assert_layout(&written);
        assert_eq!(
            cells(&written, ChunkKind::Column, 0, 16),
            [(1, Value::Int64(3), 0)]
        );
        let back = round_trip(&store, ChunkCaps::DEFAULT);
        let visits = PropertyKey::new("visits");
        assert_eq!(back.get_node_property(alix, &visits), None);
        assert_eq!(back.get_node_property(gus, &visits), Some(Value::Int64(3)));
    }

    /// The labels and edge types of each graph, sorted.
    fn registries(store: &LpgStore) -> Vec<(String, Vec<String>, Vec<String>)> {
        let sorted = |mut names: Vec<String>| {
            names.sort();
            names
        };
        let mut graphs = vec![(
            String::new(),
            sorted(store.all_labels()),
            sorted(store.all_edge_types()),
        )];
        let mut names = store.graph_names();
        names.sort();
        for name in names {
            let graph = store.graph(&name).unwrap();
            graphs.push((
                name,
                sorted(graph.all_labels()),
                sorted(graph.all_edge_types()),
            ));
        }
        graphs
    }

    /// Each graph keeps the labels and edge types it registered that no row
    /// of it uses (a deleted node's label, a deleted edge's type): they come
    /// back in that graph after a load, in the default graph and in a named
    /// one, a name used in one graph and unused in the other both ways; and
    /// the loaded store writes the same chunks.
    #[test]
    fn names_no_row_uses_stay_with_their_graph() {
        let store = LpgStore::new().unwrap();
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        let ghost = store.create_node(&["Ghost", "City"]);
        let old = store.create_edge(alix, gus, "OLD");
        store.create_edge(alix, gus, "KNOWS");
        store.delete_edge(old);
        store.delete_node(ghost);
        store.create_graph("trips").unwrap();
        let trips = store.graph("trips").unwrap();
        let berlin = trips.create_node(&["City"]);
        let paris = trips.create_node(&["City"]);
        let museum = trips.create_node(&["Museum", "Person"]);
        let ferry = trips.create_edge(berlin, paris, "FERRY");
        trips.create_edge(berlin, paris, "KNOWS");
        trips.delete_edge(ferry);
        trips.delete_node(museum);

        let before = registries(&store);
        assert_eq!(
            before,
            [
                (
                    String::new(),
                    vec!["City".to_string(), "Ghost".into(), "Person".into()],
                    vec!["KNOWS".to_string(), "OLD".into()]
                ),
                (
                    "trips".to_string(),
                    vec!["City".to_string(), "Museum".into(), "Person".into()],
                    vec!["FERRY".to_string(), "KNOWS".into()]
                ),
            ],
            "City is unused in the default graph and used in trips, Person the other way"
        );
        let written = chunks(&store, ChunkCaps::DEFAULT);
        assert_layout(&written);
        let meta = meta(&written);
        let ids = |names: &[String], wanted: &[&str]| -> Vec<u32> {
            wanted
                .iter()
                .map(|name| {
                    u32::try_from(names.iter().position(|known| known == name).unwrap()).unwrap()
                })
                .collect()
        };
        assert_eq!(
            meta.graphs[0].unused_labels,
            ids(&meta.labels, &["City", "Ghost"])
        );
        assert_eq!(
            meta.graphs[0].unused_edge_types,
            ids(&meta.edge_types, &["OLD"])
        );
        assert_eq!(
            meta.graphs[1].unused_labels,
            ids(&meta.labels, &["Museum", "Person"])
        );
        assert_eq!(
            meta.graphs[1].unused_edge_types,
            ids(&meta.edge_types, &["FERRY"])
        );

        let back = round_trip(&store, ChunkCaps::DEFAULT);
        assert_eq!(registries(&back), before, "every graph's names come back");
        assert_eq!(
            chunks(&back, ChunkCaps::DEFAULT),
            written,
            "the loaded store writes the same chunks"
        );
    }

    #[test]
    fn sparse_ids_round_trip_across_row_groups() {
        let store = LpgStore::new().unwrap();
        let ids: Vec<NodeId> = (0..40).map(|_| store.create_node(&["Person"])).collect();
        // Ids 3 and 4 cross a group boundary; 8 to 11 are a whole group.
        for id in [3, 4, 7, 8, 9, 10, 11] {
            store.delete_node(ids[id]);
        }
        // A key on the last id only.
        store.set_node_property(ids[39], "city", Value::from("Berlin"));
        // An overlay above a compacted base.
        let overlay = LpgStore::new().unwrap();
        overlay.set_next_node_id(1_000);
        overlay.create_node(&["City"]);
        for source in [&store, &overlay] {
            let back = round_trip(source, caps(4, 1 << 20));
            assert_same_graph(source, &back);
            assert_eq!(back.next_node_id(), source.next_node_id());
        }
    }

    #[test]
    fn ids_are_not_reused_after_a_reopen() {
        let store = LpgStore::new().unwrap();
        let nodes: Vec<NodeId> = (0..5).map(|_| store.create_node(&["Mia"])).collect();
        let knows = store.create_edge(nodes[0], nodes[1], "KNOWS");
        store.create_edge(nodes[1], nodes[2], "KNOWS");
        store.delete_node(nodes[3]);
        store.delete_node(nodes[4]);
        store.delete_edge(EdgeId::new(knows.0 + 1));
        let back = round_trip(&store, ChunkCaps::DEFAULT);
        assert_eq!(
            back.create_node(&["Jules"]),
            NodeId::new(5),
            "ids 3 and 4 stay retired"
        );
        assert_eq!(
            back.create_edge(nodes[0], nodes[2], "KNOWS"),
            EdgeId::new(2),
            "edge 1 stays retired"
        );
    }

    #[test]
    fn names_an_open_transaction_added_during_the_write_load() {
        use grafeo_common::types::TransactionId;

        let store = LpgStore::new().unwrap();
        let nodes: Vec<NodeId> = (0..6).map(|_| store.create_node(&["Person"])).collect();
        let mut image = Hooked {
            inner: MemoryImage::new(),
            hook: Some(Box::new(|| {
                assert!(store.add_label_versioned(nodes[5], "Traveller", TransactionId::new(19)));
            })),
        };
        image
            .inner
            .begin_section(SectionType::LpgStore, LPG_SECTION_VERSION)
            .unwrap();
        write_lpg_chunks(&store, caps(2, 1 << 20), &mut image).unwrap();
        let back = LpgStore::new().unwrap();
        let source = image.inner.section_source(SectionType::LpgStore).unwrap();
        read_lpg_chunks(&back, &*source).unwrap();
        let mut labels: Vec<String> = back
            .get_node(nodes[5])
            .unwrap()
            .labels
            .iter()
            .map(|l| l.to_string())
            .collect();
        labels.sort();
        #[cfg(not(feature = "temporal"))]
        assert_eq!(labels, ["Person", "Traveller"], "written in place");
        #[cfg(feature = "temporal")]
        assert_eq!(
            labels,
            ["Person"],
            "the open transaction's label is pending"
        );
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

        fn section_version(&self) -> u8 {
            self.inner.section_version()
        }
    }

    #[test]
    fn the_reader_fetches_the_metadata_then_each_chunk_once_in_order() {
        let store = round_trip_store();
        let mut image = MemoryImage::new();
        image
            .begin_section(SectionType::LpgStore, LPG_SECTION_VERSION)
            .unwrap();
        write_lpg_chunks(&store, caps(4, 512), &mut image).unwrap();
        let source = Counting {
            inner: image.section_source(SectionType::LpgStore).unwrap(),
            fetched: std::cell::RefCell::new(Vec::new()),
        };
        let count = source.chunks().len();
        read_lpg_chunks(&LpgStore::new().unwrap(), &source).unwrap();
        let mut expected = vec![count - 1];
        expected.extend(0..count - 1);
        assert_eq!(*source.fetched.borrow(), expected);
    }

    #[cfg(feature = "temporal")]
    #[test]
    fn property_history_round_trips() {
        use grafeo_common::types::EpochId;

        let store = LpgStore::new().unwrap();
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        let at = EpochId::new;
        store.set_node_property_at_epoch(alix, "city", Value::from("Amsterdam"), at(3));
        store.set_node_property_at_epoch(alix, "city", Value::from("Berlin"), at(19));
        store.set_node_property_at_epoch(alix, "city", Value::from("Paris"), at(88));
        store.set_node_property_at_epoch(alix, "deep", nested(MAX_PROPERTY_VALUE_DEPTH), at(3));
        store.set_node_property_at_epoch(alix, "deep", Value::Int64(19), at(19));
        store.set_node_property_at_epoch(gus, "city", Value::from("Prague"), at(3));
        store.set_node_property_at_epoch(gus, "city", Value::Null, at(19));
        let knows = store.create_edge(alix, gus, "KNOWS");
        store.set_edge_property_at_epoch(knows, "since", Value::Int64(3), at(3));
        store.set_edge_property_at_epoch(knows, "since", Value::Int64(19), at(19));
        store.sync_epoch(at(88));

        for caps in [caps(1, 64), ChunkCaps::DEFAULT] {
            let back = round_trip(&store, caps);
            assert_eq!(back.current_epoch(), at(88));
            let history = |log: Vec<(PropertyKey, Vec<(EpochId, Value)>)>| {
                let mut log: Vec<(String, Vec<(u64, Vec<u8>)>)> = log
                    .into_iter()
                    .map(|(key, versions)| {
                        let versions = versions
                            .iter()
                            .map(|(epoch, value)| (epoch.as_u64(), bits(value)))
                            .collect();
                        (key.as_str().to_string(), versions)
                    })
                    .collect();
                log.sort();
                log
            };
            for id in [alix, gus] {
                assert_eq!(
                    history(back.node_property_history(id)),
                    history(store.node_property_history(id)),
                    "node {}: {caps:?}",
                    id.0
                );
            }
            assert_eq!(
                history(back.edge_property_history(knows)),
                history(store.edge_property_history(knows)),
                "{caps:?}"
            );
            assert_eq!(
                back.get_node_property_at_epoch(alix, &PropertyKey::new("city"), at(19)),
                Some(Value::from("Berlin"))
            );
            assert_eq!(back.get_node_property(gus, &PropertyKey::new("city")), None);
        }
    }

    /// A property set twice in one transaction holds two versions at one
    /// commit epoch, and its value can share the epoch of its last older
    /// version: equal epochs are no step back.
    #[cfg(feature = "temporal")]
    #[test]
    fn versions_at_one_epoch_round_trip() {
        use grafeo_common::types::EpochId;

        let store = LpgStore::new().unwrap();
        let alix = store.create_node(&["Person"]);
        for city in ["Amsterdam", "Berlin", "Paris"] {
            store.set_node_property_at_epoch(alix, "city", Value::from(city), EpochId::new(3));
        }
        store.set_node_property_at_epoch(alix, "visits", Value::Int64(3), EpochId::new(19));
        store.set_node_property_at_epoch(alix, "visits", Value::Null, EpochId::new(19));
        store.sync_epoch(EpochId::new(19));
        let back = round_trip(&store, ChunkCaps::DEFAULT);
        for key in ["city", "visits"] {
            assert_eq!(
                back.node_property_history_for_key(alix, key),
                store.node_property_history_for_key(alix, key),
                "{key}"
            );
        }
    }

    /// The metadata chunk of a crafted section: caps of 4 rows and 1 MiB, one
    /// graph (next node id 4, next edge id 2), label "Person", edge type
    /// "KNOWS" and node property column 16 "name".
    fn crafted_meta() -> LpgMeta {
        LpgMeta {
            layout: 1,
            max_rows: 4,
            max_bytes: 1 << 20,
            epoch: 0,
            labels: vec!["Person".into()],
            edge_types: vec!["KNOWS".into()],
            graphs: vec![super::GraphMeta {
                name: String::new(),
                next_node_id: 4,
                next_edge_id: 2,
                unused_labels: Vec::new(),
                unused_edge_types: Vec::new(),
            }],
            columns: vec![ColumnMeta {
                column_id: 16,
                table: Table::Node,
                key: "name".into(),
            }],
        }
    }

    /// One chunk of a crafted LPG section.
    enum Crafted {
        /// The metadata chunk of [`crafted_meta`].
        Meta,
        /// A metadata chunk of other metadata.
        MetaOf(LpgMeta),
        /// A labels chunk of graph 0.
        Labels {
            row_start: u64,
            row_count: u32,
            rows: Vec<(u32, &'static str)>,
        },
        /// A chunk of column 16 of graph 0, every row "Alix".
        Name {
            row_start: u64,
            row_count: u32,
            rows: Vec<u32>,
        },
        /// A chunk of `column_id` of graph 0 with Int64 values.
        Column {
            column_id: u32,
            row_start: u64,
            row_count: u32,
            rows: Vec<(u32, i64)>,
        },
        /// Any chunk: its kind, graph, column and rows from `meta`, its
        /// values and epochs encoded.
        Raw {
            meta: ChunkMeta,
            values: Vec<(u32, Value)>,
            epochs: Option<Vec<u64>>,
        },
    }

    /// The bytes of a column chunk of `values`, and its entry with `kind`.
    fn crafted_chunk(
        kind: ChunkKind,
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
        (meta, bytes)
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
                    crafted_chunk(ChunkKind::Column, 0, 0, row_start, row_count, &values, None)
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
                        0,
                        16,
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
        read_lpg_chunks(&LpgStore::new().unwrap(), &*source)
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
            meta: ChunkMeta::history(0, column, 0, 1, 0),
            values,
            epochs,
        };
        let column = |column: u32, values: Vec<(u32, Value)>, epochs: Option<Vec<u64>>| Raw {
            meta: ChunkMeta::column(0, column, 0, 1, 0),
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
                "an unknown column",
                vec![
                    Column {
                        column_id: 99,
                        row_start: 0,
                        row_count: 1,
                        rows: vec![(0, 3)],
                    },
                    Meta,
                ],
                "column 99",
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
                vec![history(0, vec![(0, Value::from("0"))], None), Meta],
                "history",
            ),
            (
                "a history chunk with epochs",
                vec![
                    alix(),
                    history(
                        16,
                        vec![(0, history_value(&[(3, Value::from("Gus"))]))],
                        Some(vec![3]),
                    ),
                    Meta,
                ],
                "epoch",
            ),
            (
                "a history value that is not a list of versions",
                vec![alix(), history(16, vec![(0, Value::Int64(3))], None), Meta],
                "history",
            ),
            (
                "history epochs that go back",
                vec![
                    alix(),
                    history(
                        16,
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
                    column(16, vec![(0, Value::from("Alix"))], Some(vec![19])),
                    history(
                        16,
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
                        16,
                        vec![(0, history_value(&[(19, Value::from("Gus"))]))],
                        None,
                    ),
                    column(16, vec![(0, Value::from("Alix"))], Some(vec![3])),
                    Meta,
                ],
                "epoch",
            ),
            (
                "a null property value",
                vec![alix(), column(16, vec![(0, Value::Null)], None), Meta],
                "null",
            ),
            (
                "epochs on a fixed column",
                vec![column(0, vec![(0, Value::from("0"))], Some(vec![3])), Meta],
                "epoch",
            ),
            (
                "an unknown graph",
                vec![
                    Raw {
                        meta: ChunkMeta::column(5, 0, 0, 1, 0),
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
                "named graphs out of name order",
                vec![with(|meta| {
                    for name in ["trips", "travel"] {
                        meta.graphs.push(super::GraphMeta {
                            name: name.into(),
                            next_node_id: 0,
                            next_edge_id: 0,
                            unused_labels: Vec::new(),
                            unused_edge_types: Vec::new(),
                        });
                    }
                })],
                "name order",
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
                        meta: ChunkMeta::history(0, 16, 0, 3, 0),
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
                    column(16, vec![(0, Value::from("Alix"))], Some(vec![u64::MAX])),
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
                "an unused label id past the label table",
                vec![with(|meta| meta.graphs[0].unused_labels = vec![1])],
                "unused label ids [1]",
            ),
            (
                "unused edge type ids not ascending",
                vec![with(|meta| {
                    meta.edge_types.push("VISITED".into());
                    meta.graphs[0].unused_edge_types = vec![1, 0];
                })],
                "unused edge type ids [1, 0]",
            ),
            (
                "a label listed as unused that a node has",
                vec![alix(), with(|meta| meta.graphs[0].unused_labels = vec![0])],
                "listed as unused",
            ),
            (
                "an edge type listed as unused that an edge has",
                [
                    vec![two_nodes()],
                    edge_columns(vec![(0, 0, 1, 0)], 1),
                    vec![with(|meta| meta.graphs[0].unused_edge_types = vec![0])],
                ]
                .into_iter()
                .flatten()
                .collect(),
                "listed as unused",
            ),
            (
                "a named graph listed twice",
                vec![with(|meta| {
                    for _ in 0..2 {
                        meta.graphs.push(super::GraphMeta {
                            name: String::new(),
                            next_node_id: 0,
                            next_edge_id: 0,
                            unused_labels: Vec::new(),
                            unused_edge_types: Vec::new(),
                        });
                    }
                })],
                "name order",
            ),
            (
                "a property column id below 16",
                vec![with(|meta| meta.columns[0].column_id = 3)],
                "column 3",
            ),
            (
                "two columns of one key",
                vec![with(|meta| {
                    meta.columns.push(ColumnMeta {
                        column_id: 17,
                        table: Table::Node,
                        key: "name".into(),
                    });
                })],
                "\"name\"",
            ),
            (
                "a label listed twice",
                vec![with(|meta| meta.labels.push("Person".into()))],
                "\"Person\"",
            ),
            (
                "rows of zero",
                vec![with(|meta| meta.max_rows = 0)],
                "max_rows",
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
            assert!(
                matches!(error, Error::Serialization(_)),
                "{name}: {error:?}"
            );
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
