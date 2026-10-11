//! The LPG section in chunks (section version 3).
//!
//! Per graph in id order (graph 0 is the default graph, then the named graphs
//! by their ids), the section holds its node table and then its edge table.
//! A row is a node or edge id, and a table is written in row groups of
//! `max_rows` rows, `[k * max_rows, (k + 1) * max_rows)`; only the groups
//! that hold a node (or an edge) are written, and the chunks of one group come
//! together. In a group:
//!
//! - the node table writes [`COLUMN_LABELS`] (node structure namespace): the
//!   node's label ids, ascending, joined by `,` as a string (`""` for a node
//!   without labels);
//! - the edge table writes [`COLUMN_SOURCE`], [`COLUMN_TARGET`] and
//!   [`COLUMN_EDGE_TYPE`] (edge structure namespace): `Int64` values of the
//!   source and target node ids and the edge type id, as three chunks of one
//!   range in a row;
//! - each property column with a value in the group writes `Column` chunks of
//!   the current values, in the table's property namespace, with the property
//!   key's id as the column id. With `temporal`, a value carries the epoch it was set
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
//! Ids are permanent: a graph's id, and the ids of each graph's labels, edge
//! types and property keys, are given once and never reassigned or reused
//! (see [`NameDictionary`]), so the chunks of one checkpoint and the next
//! name the same things by the same ids. The labels and edge types are the
//! graph's own dictionaries; a property key gets its id when a write first
//! meets its column, and one dictionary of keys serves the node and the edge
//! table (the namespace tells them apart). A name nothing uses any more keeps
//! its id, and an id no name holds is a gap. A dropped graph's id is never
//! given again: a graph created later under its name gets a new one.
//!
//! The metadata chunk ([`ChunkMeta::meta`]) comes last: [`LpgMeta`], with the
//! caps the section was written with, the id the next graph gets, and each
//! graph with its id, name, next node and edge ids and its three dictionaries
//! (each its next id and its names with their ids), in the layout of
//! [`encode_lpg_meta`], which has no size limit of its own: it holds as many
//! names as the store has. It comes last because it lists every name the
//! chunks use, and the dictionaries are read after the rows: commits are held
//! while a checkpoint writes, but a transaction still open can create a
//! label, an edge type or a property key meanwhile, and the dictionaries only
//! grow, so the ones read after the rows hold every id the rows name. A
//! reader fetches the metadata chunk first, restores every graph and its
//! dictionaries id for id, then reads the others in order, so the store comes
//! back with the ids it had.
//!
//! The chunk sizes follow [`RowsChunker`]: at most `max_rows` rows and
//! `max_bytes` bytes, a larger value in a chunk of its own. Every order is
//! defined (graphs, rows and columns by id), so the same store gives the same
//! bytes.
//!
//! The section holds the committed state, also while transactions are open
//! (a checkpoint holds their writes, not the transactions): the nodes and
//! edges visible now and those an open transaction deleted, with the labels
//! and values they were committed with, and nothing an open transaction
//! created. With `temporal` a node's labels are those of the store's current
//! epoch and a value's versions those a transaction committed, so what an
//! open transaction wrote is left out; without it the store keeps no
//! versions and an open transaction changes values and labels in place, so
//! the labels and values it replaced come from its change set
//! ([`OpenChanges`]), as do the nodes and edges it deleted in both builds. A
//! property whose value is null does not exist and is not written (the
//! direct store API and a 0.5.x load can leave one in a store without
//! `temporal`).
//!
//! Memory: the writer holds one open chunk per column of the row group
//! being written, the sorted ids of the table being written (8 bytes per
//! node or edge), the committed state of what open transactions changed
//! (O(open changes), see [`OpenChanges`]) and, without `temporal`, the
//! sorted ids of each of its property columns (8 bytes per value); it reads
//! values a batch at a time.

use std::collections::BTreeMap;
use std::sync::Arc;

use bytes::Bytes;
use grafeo_common::storage::value_codec::{MAX_PROPERTY_VALUE_DEPTH, nests_too_deep};
use grafeo_common::storage::{
    ChunkCaps, ChunkKind, ChunkMeta, ChunkNamespace, SectionSink, SectionSource,
};
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};

use super::dictionary::NameDictionary;
use super::property::{EntityId, PropertyStorage};
use super::store::{Labels, OpenChangeSource, OpenChanges};
use super::{Edge, LpgStore};
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

impl DictionaryMeta {
    /// `dictionary` as the metadata chunk lists it.
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

    /// The namespace of the table's fixed columns.
    pub(crate) fn structure(self) -> ChunkNamespace {
        match self {
            Self::Node => ChunkNamespace::NodeStructure,
            Self::Edge => ChunkNamespace::EdgeStructure,
        }
    }

    /// The namespace of the table's property columns.
    pub(crate) fn properties(self) -> ChunkNamespace {
        match self {
            Self::Node => ChunkNamespace::NodeProperties,
            Self::Edge => ChunkNamespace::EdgeProperties,
        }
    }
}

// ── Writing ─────────────────────────────────────────────────────────

/// Writes per graph (in id order) the node table's row groups, then the edge
/// table's, then the metadata chunk: the committed state (see the module
/// docs). While transactions are open, the caller holds their writes and
/// rollbacks and commits (a checkpoint's write freeze) for the whole write:
/// the committed state is read from the store and what the open
/// transactions changed together, from `open` (their change sets).
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
    open: OpenChangeSource<'_>,
    sink: &mut dyn SectionSink,
) -> Result<()> {
    caps.validate()?;
    // A graph dropped since the names were read is left out.
    let mut named: Vec<(String, Arc<LpgStore>)> = store
        .graph_names()
        .into_iter()
        .filter_map(|name| store.graph(&name).map(|graph| (name, graph)))
        .collect();
    named.sort_unstable_by_key(|(_, graph)| graph.graph_id());
    let graphs: Vec<(&str, &LpgStore)> = std::iter::once(("", store))
        .chain(named.iter().map(|(name, graph)| (name.as_str(), &**graph)))
        .collect();

    let epoch = section_epoch(store);
    if epoch > MAX_EPOCH {
        return Err(Error::Serialization(format!(
            "the store epoch {epoch} is above {MAX_EPOCH}, the largest epoch a section holds"
        )));
    }
    let mut graph_metas = Vec::with_capacity(graphs.len());
    for (name, graph) in &graphs {
        let place = Place {
            graph_id: graph.graph_id(),
            graph: describe_graph(graph.graph_id(), name),
            caps,
        };
        // Each graph is a graph of its own in a change set (a named graph's
        // name may be empty).
        let key = (!std::ptr::eq(*graph, store)).then_some(*name);
        let changes = open.changes_of(key);
        let nodes_end = write_node_table(graph, &changes, epoch, &place, sink)?;
        let edges_end = write_edge_table(graph, &changes, &place, sink)?;
        // Read after the rows: every row written is below the next ids, and
        // the dictionaries, which only grow, hold every id the rows name.
        let [labels, edge_types, keys] = graph.name_dictionaries();
        graph_metas.push(GraphMeta {
            id: graph.graph_id(),
            name: (*name).to_string(),
            next_node_id: graph.next_node_id().max(nodes_end),
            next_edge_id: graph.next_edge_id().max(edges_end),
            labels: DictionaryMeta::of(&labels),
            edge_types: DictionaryMeta::of(&edge_types),
            keys: DictionaryMeta::of(&keys),
        });
    }

    let meta = LpgMeta {
        layout: LPG_META_LAYOUT,
        max_rows: caps.max_rows,
        max_bytes: caps.max_bytes,
        epoch,
        // Read after the graphs: above every graph id written.
        next_graph_id: store.next_graph_id(),
        graphs: graph_metas,
    };
    sink.write_chunk(ChunkMeta::meta(), &encode_lpg_meta(&meta)?)
}

/// The nodes of `graph` as committed, ascending: those visible now and those
/// an open transaction deleted (`changes`, from the open transactions'
/// change sets), each with its committed labels. With
/// `temporal` these come from the label sets of the graph's epoch or
/// `epoch` (the root store's, see [`section_epoch`]), whichever is later:
/// every committed label set is at or below it (the default graph's epoch
/// follows every commit), a pending one is above it (see
/// [`LpgStore::node_with_labels_at`]).
///
/// The rows of a node table, and the nodes of
/// [`LpgStore::committed_copy`].
///
/// # Errors
///
/// Returns the error of reading a node record.
pub(super) fn committed_nodes<'g>(
    graph: &'g LpgStore,
    changes: &'g OpenChanges,
    epoch: u64,
) -> Result<impl Iterator<Item = (NodeId, Labels)> + 'g> {
    let labels_at = EpochId::new(graph.current_epoch().as_u64().max(epoch));
    let ids = merge_ids(graph.try_node_ids()?, changes.deleted_nodes());
    Ok(ids.into_iter().map(move |id| {
        let current = || graph.node_with_labels_at(id, labels_at).labels;
        (id, changes.committed_labels(id, current))
    }))
}

/// The edges of `graph` as committed, ascending, without their property
/// values: those visible now and those an open transaction deleted
/// (`changes`, from the open transactions' change sets), with the endpoints
/// and type they were committed with. Each record is read once: the ids are
/// those with a visible version, and a deleted record is left out where it
/// is read.
///
/// The rows of an edge table, and the edges of
/// [`LpgStore::committed_copy`]. An item is an error when the edge's record
/// cannot be read.
pub(super) fn committed_edges<'g>(
    graph: &'g LpgStore,
    changes: &'g OpenChanges,
) -> impl Iterator<Item = Result<Edge>> + 'g {
    let deleted: Vec<EdgeId> = changes.deleted_edges().iter().map(|edge| edge.id).collect();
    merge_ids(graph.edge_ids_with_a_visible_version(), &deleted)
        .into_iter()
        .filter_map(move |id| match changes.deleted_edge(id) {
            Some(edge) => Some(Ok(Edge::new(
                id,
                edge.src,
                edge.dst,
                edge.edge_type.clone(),
            ))),
            None => graph.try_edge_without_properties(id).transpose(),
        })
}

/// Writes the node table of `graph`: its [`committed_nodes`], each with the
/// ids its labels have in the graph's label dictionary; returns one past its
/// last row (0 for none).
fn write_node_table(
    graph: &LpgStore,
    changes: &OpenChanges,
    epoch: u64,
    place: &Place,
    sink: &mut dyn SectionSink,
) -> Result<u64> {
    let nodes = committed_nodes(graph, changes, epoch)?.map(|(id, labels)| {
        let mut labels = labels
            .iter()
            .map(|label| {
                graph.label_id(label).ok_or_else(|| {
                    Error::Internal(format!(
                        "{}, node {}: label {label:?} has no id in the graph's labels",
                        place.graph,
                        id.as_u64()
                    ))
                })
            })
            .collect::<Result<Vec<u32>>>()?;
        labels.sort_unstable();
        let text = labels
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        Ok((id, vec![Some((Value::from(text), 0))]))
    });
    let table = TableWriter {
        table: Table::Node,
        fixed: vec![column(Table::Node.structure(), COLUMN_LABELS)],
        properties: &graph.node_properties,
        #[cfg(not(feature = "temporal"))]
        committed: changes.node_values(),
        graph,
    };
    table.write(place, nodes, sink)
}

/// Writes the edge table of `graph`: its [`committed_edges`], each with the
/// id its type has in the graph's edge type dictionary; returns one past its
/// last row (0 for none).
fn write_edge_table(
    graph: &LpgStore,
    changes: &OpenChanges,
    place: &Place,
    sink: &mut dyn SectionSink,
) -> Result<u64> {
    let edges = committed_edges(graph, changes).map(|edge| {
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
        let edge_type = graph.edge_type_id(&edge.edge_type).ok_or_else(|| {
            Error::Internal(format!(
                "{}, edge {}: edge type {:?} has no id in the graph's edge types",
                place.graph,
                edge.id.as_u64(),
                edge.edge_type
            ))
        })?;
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
            column(Table::Edge.structure(), COLUMN_SOURCE),
            column(Table::Edge.structure(), COLUMN_TARGET),
            column(Table::Edge.structure(), COLUMN_EDGE_TYPE),
        ],
        properties: &graph.edge_properties,
        #[cfg(not(feature = "temporal"))]
        committed: changes.edge_values(),
        graph,
    };
    table.write(place, edges, sink)
}

/// The ids in `visible` and in `deleted` (both ascending), ascending and
/// each once: the rows of a table.
fn merge_ids<Id: EntityId>(visible: Vec<Id>, deleted: &[Id]) -> Vec<Id> {
    if deleted.is_empty() {
        return visible;
    }
    let mut ids = Vec::with_capacity(visible.len() + deleted.len());
    let (mut at_visible, mut at_deleted) = (0, 0);
    while let (Some(&a), Some(&b)) = (visible.get(at_visible), deleted.get(at_deleted)) {
        match a.as_u64().cmp(&b.as_u64()) {
            std::cmp::Ordering::Less => {
                ids.push(a);
                at_visible += 1;
            }
            std::cmp::Ordering::Greater => {
                ids.push(b);
                at_deleted += 1;
            }
            std::cmp::Ordering::Equal => {
                ids.push(a);
                at_visible += 1;
                at_deleted += 1;
            }
        }
    }
    ids.extend_from_slice(&visible[at_visible..]);
    ids.extend_from_slice(&deleted[at_deleted..]);
    ids
}

/// The epoch the section records: the store's with `temporal`.
#[cfg(feature = "temporal")]
pub(super) fn section_epoch(store: &LpgStore) -> u64 {
    store.current_epoch().as_u64()
}

/// The epoch the section records: 0 without `temporal`, as the 0.5.x block
/// layout wrote it.
#[cfg(not(feature = "temporal"))]
pub(super) fn section_epoch(_store: &LpgStore) -> u64 {
    0
}

/// How the graph with id `graph_id` (named `name`) is named in errors.
fn describe_graph(graph_id: u32, name: &str) -> String {
    if graph_id == 0 {
        "the default graph".to_string()
    } else {
        format!("graph {name:?}")
    }
}

/// A `Column` chunk column of `namespace`.
fn column(namespace: ChunkNamespace, column_id: u32) -> ChunkColumn {
    ChunkColumn {
        kind: ChunkKind::Column,
        namespace,
        column_id,
    }
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
    /// Per property key, the committed value of each node or edge whose
    /// value of it an open transaction replaced in place (`None` when it had
    /// none), by id ascending: written instead of the store's value.
    #[cfg(not(feature = "temporal"))]
    committed: &'s BTreeMap<PropertyKey, Vec<(Id, Option<Value>)>>,
    /// The graph whose property keys give the columns their ids.
    graph: &'s LpgStore,
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
        // The table's keys get their ids first, in key order, so a store
        // built the same way gets the same ids whatever order the rows meet
        // them in.
        let mut keys = self.properties.keys();
        keys.sort_unstable();
        for key in &keys {
            self.graph.property_key_id(key.as_str());
        }
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

#[cfg(not(feature = "temporal"))]
impl<'s, Id: EntityId> TableWriter<'s, Id> {
    /// The table's property columns as the store holds them now, by key,
    /// each with the ids that have a value and the committed values open
    /// transactions replaced; a key only a replaced value has joins them.
    fn cursors(&self) -> Result<Vec<ColumnCursor<'s, Id>>> {
        let committed: &'s BTreeMap<PropertyKey, Vec<(Id, Option<Value>)>> = self.committed;
        let mut keys = self.properties.keys();
        keys.extend(committed.keys().cloned());
        keys.sort_unstable();
        keys.dedup();
        let mut cursors: Vec<ColumnCursor<'s, Id>> = keys
            .into_iter()
            .map(|key| ColumnCursor {
                column_id: self.graph.property_key_id(key.as_str()),
                ids: self.properties.column_ids(&key),
                next: 0,
                committed: committed.get(&key).map_or(&[], Vec::as_slice),
                next_committed: 0,
                key,
            })
            .collect();
        // In column order: the keys' ids.
        cursors.sort_unstable_by_key(|cursor| cursor.column_id);
        Ok(cursors)
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
        for (key, versions) in committed_versions(table.properties, id) {
            histories.push((table.graph.property_key_id(key.as_str()), key, versions));
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
            let namespace = table.table.properties();
            self.properties
                .entry(column_id)
                .or_insert_with(|| {
                    let columns = vec![
                        ChunkColumn {
                            kind: ChunkKind::History,
                            namespace,
                            column_id,
                        },
                        column(namespace, column_id),
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
    /// column at a time, reading the values a batch at a time; a value an
    /// open transaction replaced is written as it was committed.
    #[cfg(not(feature = "temporal"))]
    fn finish(
        self,
        table: &TableWriter<'_, Id>,
        cursors: &mut [ColumnCursor<'_, Id>],
        sink: &mut dyn SectionSink,
    ) -> Result<()> {
        self.fixed.finish(sink)?;
        let max_rows = u64::from(self.place.caps.max_rows);
        for cursor in cursors {
            let (stored, committed) = cursor.take_group(self.start, max_rows);
            let rows = column_rows(stored, committed, &self.present);
            if rows.is_empty() {
                continue;
            }
            let mut chunker = RowsChunker::new(
                self.place.graph_id,
                vec![column(table.table.properties(), cursor.column_id)],
                self.start,
                self.place.caps,
            );
            let key = &cursor.key;
            for_each_committed_value(table.properties, key, &rows, |id, value| {
                table.refuse_too_deep(self.place, id, key, &value, "the value")?;
                chunker.push(sink, id.as_u64(), vec![Some((value, 0))])
            })?;
            chunker.finish(sink)?;
        }
        Ok(())
    }
}

/// The committed versions of the values of `id`, by key: its
/// [`PropertyStorage::get_all_history`] without the versions a transaction
/// still open wrote (at [`EpochId::PENDING`]), and without a key left with
/// none. A checkpoint holds committed versions only, and so does
/// [`LpgStore::committed_copy`].
#[cfg(feature = "temporal")]
pub(super) fn committed_versions<Id: EntityId>(
    properties: &PropertyStorage<Id>,
    id: Id,
) -> Vec<(PropertyKey, Vec<(EpochId, Value)>)> {
    let mut histories = properties.get_all_history(id);
    for (_, versions) in &mut histories {
        versions.retain(|(epoch, _)| *epoch != EpochId::PENDING);
    }
    histories.retain(|(_, versions)| !versions.is_empty());
    histories
}

/// Calls `each` with the committed value of each row of `rows` (see
/// [`column_rows`]), in order: the store's value of property `key` for a
/// [`Cell::Stored`] row, read [`READ_BATCH_ROWS`] at a time, and the value
/// an open transaction replaced for a [`Cell::Committed`] one. A row whose
/// value was removed since its ids were listed, or is null (a property whose
/// value is null does not exist), is passed over.
///
/// # Errors
///
/// Returns the error of reading a value (a spilled one that cannot be
/// read), or of `each`.
#[cfg(not(feature = "temporal"))]
pub(super) fn for_each_committed_value<Id: EntityId>(
    properties: &PropertyStorage<Id>,
    key: &PropertyKey,
    rows: &[(Id, Cell<'_>)],
    mut each: impl FnMut(Id, Value) -> Result<()>,
) -> Result<()> {
    for batch in rows.chunks(READ_BATCH_ROWS) {
        let read: Vec<Id> = batch
            .iter()
            .filter(|(_, cell)| matches!(cell, Cell::Stored))
            .map(|(id, _)| *id)
            .collect();
        let mut stored = if read.is_empty() {
            Vec::new()
        } else {
            properties.try_get_batch(&read, key)?
        }
        .into_iter();
        for (id, cell) in batch {
            let value = match cell {
                Cell::Stored => stored.next().flatten(),
                Cell::Committed(value) => Some((*value).clone()),
            };
            if let Some(value) = value.filter(|value| !value.is_null()) {
                each(*id, value)?;
            }
        }
    }
    Ok(())
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

/// The ids of one property column, walked group by group, with the
/// committed values open transactions replaced.
#[cfg(not(feature = "temporal"))]
struct ColumnCursor<'c, Id> {
    key: PropertyKey,
    column_id: u32,
    /// The ids with a value, ascending.
    ids: Vec<Id>,
    /// The first id not walked yet.
    next: usize,
    /// The committed value of each id whose value an open transaction
    /// replaced (`None` when it had none), by id ascending.
    committed: &'c [(Id, Option<Value>)],
    /// The first committed value not walked yet.
    next_committed: usize,
}

#[cfg(not(feature = "temporal"))]
impl<'c, Id: EntityId> ColumnCursor<'c, Id> {
    /// The ids and the committed values of the group
    /// `[start, start + max_rows)`, after skipping those before it (of
    /// groups without a node or edge).
    fn take_group(&mut self, start: u64, max_rows: u64) -> (&[Id], &'c [(Id, Option<Value>)]) {
        let in_group = |id: Id| id.as_u64() - start < max_rows;
        while self
            .ids
            .get(self.next)
            .is_some_and(|id| id.as_u64() < start)
        {
            self.next += 1;
        }
        let first = self.next;
        while self.ids.get(self.next).is_some_and(|id| in_group(*id)) {
            self.next += 1;
        }
        let committed: &'c [(Id, Option<Value>)] = self.committed;
        while committed
            .get(self.next_committed)
            .is_some_and(|(id, _)| id.as_u64() < start)
        {
            self.next_committed += 1;
        }
        let first_committed = self.next_committed;
        while committed
            .get(self.next_committed)
            .is_some_and(|(id, _)| in_group(*id))
        {
            self.next_committed += 1;
        }
        (
            &self.ids[first..self.next],
            &committed[first_committed..self.next_committed],
        )
    }
}

/// Where a value of a property column comes from.
#[cfg(not(feature = "temporal"))]
pub(super) enum Cell<'c> {
    /// The value the store holds.
    Stored,
    /// The committed value an open transaction replaced.
    Committed(&'c Value),
}

/// The rows of one property column, ascending: the ids with a value in the
/// store (`stored`) and, in place of those an open transaction changed, the
/// ids with a committed value (`committed`), of the nodes or edges written
/// (`present`: those of a row group for a write, all of them for
/// [`LpgStore::committed_copy`]). All three are ascending.
#[cfg(not(feature = "temporal"))]
pub(super) fn column_rows<'c, Id: EntityId>(
    stored: &[Id],
    committed: &'c [(Id, Option<Value>)],
    present: &[Id],
) -> Vec<(Id, Cell<'c>)> {
    let mut rows = Vec::with_capacity(stored.len());
    let (mut at_stored, mut at_committed) = (0, 0);
    loop {
        match (stored.get(at_stored), committed.get(at_committed)) {
            (None, None) => break,
            (Some(&id), None) => {
                rows.push((id, Cell::Stored));
                at_stored += 1;
            }
            (Some(&id), Some((other, _))) if id.as_u64() < other.as_u64() => {
                rows.push((id, Cell::Stored));
                at_stored += 1;
            }
            (stored_id, Some((id, value))) => {
                // The committed value replaces the store's value of the id.
                if stored_id.is_some_and(|stored_id| stored_id.as_u64() == id.as_u64()) {
                    at_stored += 1;
                }
                if let Some(value) = value {
                    rows.push((*id, Cell::Committed(value)));
                }
                at_committed += 1;
            }
        }
    }
    let mut at_present = 0;
    rows.retain(|(id, _)| {
        while present
            .get(at_present)
            .is_some_and(|known| known.as_u64() < id.as_u64())
        {
            at_present += 1;
        }
        present
            .get(at_present)
            .is_some_and(|known| known.as_u64() == id.as_u64())
    });
    rows
}

// ── Reading ─────────────────────────────────────────────────────────

/// Encodes a metadata chunk. Little-endian:
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
/// Before any row, every graph is restored with its id (the named ones
/// created in `store`) and its dictionaries id for id, and the store's next
/// graph id becomes the metadata's at least. Without `temporal` the history
/// versions are checked and not applied, and values are set without their
/// epochs. At the end each graph's next ids become the larger of its own and
/// the metadata's, and with `temporal` every graph store's epoch is synced
/// to the metadata's.
///
/// # Errors
///
/// [`Error::Corruption`] for a section the rules refuse, for a node or edge
/// the store cannot allocate (naming it), and for a graph or name the store
/// holds already with another id; the source's error when a chunk cannot be
/// fetched.
pub(crate) fn read_lpg_chunks(store: &LpgStore, source: &dyn SectionSource) -> Result<()> {
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
        .restore_graphs(&named, meta.next_graph_id)
        .map_err(|error| Error::corruption(format!("LPG section: {error}")))?;
    let graphs: Vec<GraphTarget<'_>> = std::iter::once(GraphTarget::Default(store))
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
        target.set_next_node_id(target.next_node_id().max(graph.next_node_id));
        target.set_next_edge_id(target.next_edge_id().max(graph.next_edge_id));
        #[cfg(feature = "temporal")]
        target.sync_epoch(EpochId::new(meta.epoch));
    }
    Ok(())
}

/// Gives `target` the dictionaries of `graph`, id for id, and their next ids.
///
/// # Errors
///
/// Returns [`Error::Corruption`] when `target` holds one of the names, or one
/// of the ids, already.
fn restore_dictionaries(target: &LpgStore, graph: &GraphMeta) -> Result<()> {
    let refuse = |what: &str, error: String| {
        Error::corruption(format!("LPG section, graph {}, {what}: {error}", graph.id))
    };
    for (id, name) in &graph.labels.names {
        target
            .restore_label(*id, name)
            .map_err(|error| refuse("label", error))?;
    }
    for (id, name) in &graph.edge_types.names {
        target
            .restore_edge_type(*id, name)
            .map_err(|error| refuse("edge type", error))?;
    }
    for (id, name) in &graph.keys.names {
        target
            .restore_property_key(*id, name)
            .map_err(|error| refuse("property key", error))?;
    }
    target.reserve_name_ids_below([
        graph.labels.next_id,
        graph.edge_types.next_id,
        graph.keys.next_id,
    ]);
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
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Arc;

    use bytes::Bytes;
    use grafeo_common::storage::SectionType;
    use grafeo_common::storage::value_codec::MAX_PROPERTY_VALUE_DEPTH;
    use grafeo_common::storage::{
        ChunkCaps, ChunkKind, ChunkMeta, ChunkNamespace, ImageSource, MemoryImage, SectionSink,
        SectionSource,
    };
    use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};
    use grafeo_common::utils::error::{Error, Result};

    use super::{
        COLUMN_EDGE_TYPE, COLUMN_LABELS, COLUMN_SOURCE, COLUMN_TARGET, DictionaryMeta, GraphMeta,
        LPG_SECTION_VERSION, LpgMeta, Table, decode_lpg_meta, encode_lpg_meta, read_lpg_chunks,
        write_lpg_chunks,
    };
    use crate::codec::column_chunk::{ColumnChunk, decode_column_chunk};
    use crate::graph::lpg::store::OpenChangeSource;
    use crate::graph::lpg::{LpgStore, OpenChangesByGraph};

    /// The chunks `write_lpg_chunks` writes for `store`, in order.
    fn try_chunks(store: &LpgStore, caps: ChunkCaps) -> Result<Vec<(ChunkMeta, Bytes)>> {
        try_chunks_from(store, caps, OpenChangeSource::None)
    }

    /// The chunks `write_lpg_chunks` writes for `store`, reading what open
    /// transactions changed from `open`.
    fn try_chunks_from(
        store: &LpgStore,
        caps: ChunkCaps,
        open: OpenChangeSource<'_>,
    ) -> Result<Vec<(ChunkMeta, Bytes)>> {
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::LpgStore, LPG_SECTION_VERSION)?;
        write_lpg_chunks(store, caps, open, &mut image)?;
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

    /// A column of a chunk: its namespace and its id there.
    type Column = (ChunkNamespace, u32);

    /// The node table's labels column.
    const LABELS: Column = (ChunkNamespace::NodeStructure, COLUMN_LABELS);
    /// The edge table's source column.
    const SOURCE: Column = (ChunkNamespace::EdgeStructure, COLUMN_SOURCE);
    /// The edge table's target column.
    const TARGET: Column = (ChunkNamespace::EdgeStructure, COLUMN_TARGET);
    /// The edge table's edge type column.
    const EDGE_TYPE: Column = (ChunkNamespace::EdgeStructure, COLUMN_EDGE_TYPE);

    /// The graph with id `graph` in `meta`.
    fn graph_of(meta: &LpgMeta, graph: u32) -> &GraphMeta {
        meta.graphs
            .iter()
            .find(|known| known.id == graph)
            .unwrap_or_else(|| panic!("graph {graph} is not in the metadata"))
    }

    /// The names of a dictionary, in id order.
    fn names_of(dictionary: &DictionaryMeta) -> Vec<&str> {
        dictionary
            .names
            .iter()
            .map(|(_, name)| name.as_str())
            .collect()
    }

    /// The id of `name` in `dictionary`.
    #[cfg(feature = "temporal")]
    fn id_in(dictionary: &DictionaryMeta, name: &str) -> u32 {
        dictionary
            .names
            .iter()
            .find(|(_, known)| known == name)
            .unwrap_or_else(|| panic!("{name:?} is not in {dictionary:?}"))
            .0
    }

    /// The column of property `key` of `table` in graph `graph`: its id in
    /// the graph's keys.
    fn key_column(meta: &LpgMeta, graph: u32, table: Table, key: &str) -> Column {
        let id = graph_of(meta, graph)
            .keys
            .names
            .iter()
            .find(|(_, name)| name == key)
            .unwrap_or_else(|| panic!("key {key:?} is not a key of graph {graph}"))
            .0;
        (table.properties(), id)
    }

    /// The decoded `kind` chunk of `graph`, `column` and first row
    /// `row_start` (exactly one).
    fn decode_kind(
        chunks: &[(ChunkMeta, Bytes)],
        kind: ChunkKind,
        graph: u32,
        (namespace, column): Column,
        row_start: u64,
    ) -> ColumnChunk {
        let found: Vec<&(ChunkMeta, Bytes)> = chunks
            .iter()
            .filter(|(meta, _)| {
                meta.kind == kind
                    && meta.graph_id == graph
                    && meta.namespace == namespace
                    && meta.column_id == column
                    && meta.row_start == row_start
            })
            .collect();
        assert_eq!(
            found.len(),
            1,
            "one {kind:?} chunk of graph {graph}, {namespace:?} column {column}, row {row_start}"
        );
        let (meta, bytes) = found[0];
        decode_column_chunk(bytes, meta.codec, meta.row_count).unwrap()
    }

    /// The decoded `Column` chunk of `graph`, `column` and first row
    /// `row_start` (exactly one).
    fn decode(
        chunks: &[(ChunkMeta, Bytes)],
        graph: u32,
        column: Column,
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
        (namespace, column): Column,
    ) -> Vec<(u64, Value, u64)> {
        let mut cells = Vec::new();
        for (meta, bytes) in chunks {
            if meta.kind != kind
                || meta.graph_id != graph
                || meta.namespace != namespace
                || meta.column_id != column
            {
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

    /// The table of a chunk and whether its column is a property column,
    /// from its namespace.
    fn table_of(chunk: &ChunkMeta) -> (Table, bool) {
        match chunk.namespace {
            ChunkNamespace::NodeStructure => (Table::Node, false),
            ChunkNamespace::NodeProperties => (Table::Node, true),
            ChunkNamespace::EdgeStructure => (Table::Edge, false),
            ChunkNamespace::EdgeProperties => (Table::Edge, true),
            other => panic!("a chunk in namespace {other:?}"),
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
    /// chunk (`History` only for property columns) of a known graph, in a
    /// namespace of the node or edge table, a property column being a key of
    /// its graph, inside one row group and below the table's next id; chunks
    /// grouped by (graph, table, row group) in that order; each (kind,
    /// namespace, column) ascending without overlap within its group; the
    /// edge columns as three consecutive chunks of one range and the same
    /// rows; and every property row's node or edge in its group.
    fn assert_layout(chunks: &[(ChunkMeta, Bytes)]) {
        let meta = meta(chunks);
        let max_rows = u64::from(meta.max_rows);
        let mut last_group: Option<(u32, Table, u64)> = None;
        let mut ends: BTreeMap<(u8, u8, u32), u64> = BTreeMap::new();
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
            let graph = graph_of(&meta, chunk.graph_id);
            let (table, property) = table_of(chunk);
            if property {
                assert!(
                    graph
                        .keys
                        .names
                        .iter()
                        .any(|(id, _)| *id == chunk.column_id),
                    "chunk {index}: column {} is a key of graph {}",
                    chunk.column_id,
                    graph.id
                );
            }
            if chunk.kind == ChunkKind::History {
                assert!(property, "chunk {index}: history of a fixed column");
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
                .entry((
                    chunk.kind.to_byte(),
                    chunk.namespace.to_byte(),
                    chunk.column_id,
                ))
                .or_insert(0);
            assert!(
                chunk.row_start >= *end,
                "chunk {index}: overlaps the chunk before it"
            );
            *end = last_row + 1;
            let rows = rows_of(chunk, bytes);
            match (chunk.namespace, chunk.column_id) {
                _ if property => {
                    property_rows.extend(rows.into_iter().map(|row| (chunk.column_id, row)));
                }
                LABELS => entities.extend(rows),
                SOURCE => {
                    for (step, column) in [(1, TARGET), (2, EDGE_TYPE)] {
                        let (partner, partner_bytes) = &chunks[index + step];
                        assert_eq!(
                            (
                                partner.kind,
                                (partner.namespace, partner.column_id),
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
                TARGET | EDGE_TYPE => panic!("chunk {index}: an edge column alone"),
                other => panic!("chunk {index}: fixed column {other:?}"),
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
            (
                meta.layout,
                meta.next_graph_id,
                meta.graphs.len(),
                meta.graphs[0].id,
                meta.graphs[0].name.as_str()
            ),
            (1, 1, 1, 0, "")
        );
        let default = &meta.graphs[0];
        for dictionary in [&default.labels, &default.edge_types, &default.keys] {
            assert_eq!(
                *dictionary,
                DictionaryMeta::default(),
                "no names, next id 0"
            );
        }
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
        let (default, trips) = (&meta.graphs[0], &meta.graphs[1]);
        // Each graph's own ids: labels and edge types in the order the store
        // met them, the keys in key order of the node table, then the edge
        // table's, as the write met them.
        assert_eq!(
            default.labels,
            DictionaryMeta {
                next_id: 3,
                names: vec![
                    (0, "Person".into()),
                    (1, "Employee".into()),
                    (2, "City".into())
                ],
            }
        );
        assert_eq!(names_of(&default.edge_types), ["KNOWS"]);
        assert_eq!(
            default.keys,
            DictionaryMeta {
                next_id: 3,
                names: vec![
                    (0, "name".into()),
                    (1, "population".into()),
                    (2, "since".into())
                ],
            }
        );
        assert_eq!(names_of(&trips.labels), ["City"], "trips' own labels");
        assert_eq!(trips.keys, DictionaryMeta::default());
        assert_eq!(
            meta.graphs
                .iter()
                .map(|g| (g.id, g.name.as_str(), g.next_node_id, g.next_edge_id))
                .collect::<Vec<_>>(),
            [(0, "", 3, 1), (1, "trips", 1, 0)]
        );
        assert_eq!(meta.next_graph_id, 2);
        let labels = decode(&written, 0, LABELS, 0);
        assert_eq!(
            labels.values,
            [(0, Value::from("0,1")), (1, Value::from("0"))]
        );
        assert_eq!(
            decode(&written, 0, LABELS, 2).values,
            [(0, Value::from("2"))]
        );
        assert_eq!(
            decode(&written, 0, key_column(&meta, 0, Table::Node, "name"), 0).values,
            [(0, Value::from("Alix")), (1, Value::from("Gus"))]
        );
        assert_eq!(
            decode(
                &written,
                0,
                key_column(&meta, 0, Table::Node, "population"),
                2
            )
            .values,
            [(0, Value::Int64(921_402))]
        );

        let edge_columns: Vec<(u32, u64, u32)> = written
            .iter()
            .filter(|(meta, _)| {
                meta.graph_id == 0 && meta.namespace == ChunkNamespace::EdgeStructure
            })
            .map(|(meta, _)| (meta.column_id, meta.row_start, meta.row_count))
            .collect();
        assert_eq!(edge_columns, [(1, 0, 1), (2, 0, 1), (3, 0, 1)]);
        assert_eq!(
            decode(&written, 0, SOURCE, 0).values,
            [(0, Value::Int64(0))]
        );
        assert_eq!(
            decode(&written, 0, TARGET, 0).values,
            [(0, Value::Int64(1))]
        );
        assert_eq!(
            decode(&written, 0, EDGE_TYPE, 0).values,
            [(0, Value::Int64(0))]
        );
        assert_eq!(
            decode(&written, 0, key_column(&meta, 0, Table::Edge, "since"), 0).values,
            [(0, Value::Int64(2019))]
        );

        // The named graph has its own labels chunk, with its own label ids.
        assert_eq!(
            decode(&written, 1, LABELS, 0).values,
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
        assert_eq!(
            (meta.graphs[1].id, meta.graphs[1].name.as_str()),
            (1, "travel")
        );
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
            let graph = graph_of(&meta, chunk.graph_id);
            let next = match table_of(chunk).0 {
                Table::Node => graph.next_node_id,
                Table::Edge => graph.next_edge_id,
            };
            assert!(last < next, "{chunk:?} ends past the next id {next}");
        }

        // The byte cap cut some groups: more name chunks than groups with names.
        let name = key_column(&meta, 0, Table::Node, "name");
        let name_chunks: Vec<u64> = written
            .iter()
            .filter(|(chunk, _)| (chunk.namespace, chunk.column_id) == name)
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
        let labels = cells(&written, ChunkKind::Column, 0, LABELS);
        assert_eq!(labels.len(), 33, "one labels value per live node");
        assert!(
            labels.contains(&(2, Value::from(""), 0)),
            "a node without labels has an empty labels value"
        );
        let deleted: BTreeSet<u64> = [3, 4, 8, 9, 10, 11, 39].into();
        for column in [name, LABELS] {
            assert!(
                cells(&written, ChunkKind::Column, 0, column)
                    .iter()
                    .all(|(row, _, _)| !deleted.contains(row)),
                "column {column:?} holds a deleted node"
            );
        }
        let edges = cells(&written, ChunkKind::Column, 0, SOURCE);
        assert_eq!(
            edges.len(),
            store.edge_count(),
            "one endpoint value per live edge"
        );
        let edge_types = &meta.graphs[0].edge_types;
        let mut type_names = names_of(edge_types);
        type_names.sort_unstable();
        assert_eq!(type_names, ["KNOWS", "VISITED"]);
        for (row, edge_type, _) in cells(&written, ChunkKind::Column, 0, EDGE_TYPE) {
            let name = store.edge_type(EdgeId::new(row)).unwrap();
            let id = edge_types
                .names
                .iter()
                .find(|(_, known)| known == name.as_str())
                .unwrap()
                .0;
            assert_eq!(edge_type, Value::Int64(i64::from(id)), "edge {row}");
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
        let rows = |column: Column| -> Vec<u64> {
            cells(&written, ChunkKind::Column, 0, column)
                .into_iter()
                .map(|(row, _, _)| row)
                .collect()
        };
        assert_eq!(rows(SOURCE), sorted, "endpoints");
        let since = key_column(&meta(&written), 0, Table::Edge, "since");
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
            cells(&written, ChunkKind::Column, 0, LABELS),
            [(0, Value::from("0"), 0)]
        );
        let name = key_column(&meta(&written), 0, Table::Node, "name");
        assert_eq!(
            cells(&written, ChunkKind::Column, 0, name),
            [(0, Value::from("Alix"), 0)],
            "only the visible node's value"
        );
        assert!(
            data(&written)
                .iter()
                .all(|(chunk, _)| [LABELS, name].contains(&(chunk.namespace, chunk.column_id))),
            "no edge chunks: {:?}",
            written
                .iter()
                .map(|(chunk, _)| chunk.column_id)
                .collect::<Vec<_>>()
        );
    }

    /// The edge table starts from the edges with a visible version and reads
    /// each record once, where it writes the row: a deleted edge is left
    /// out.
    #[test]
    fn a_deleted_edge_is_left_out_of_the_edge_table() {
        let store = LpgStore::new().unwrap();
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        let knows = store.create_edge(alix, gus, "KNOWS");
        let gone = store.create_edge(gus, alix, "KNOWS");
        assert!(store.delete_edge(gone));

        let loaded = round_trip(&store, caps(4, 1 << 20));
        assert_eq!(loaded.try_edge_ids().unwrap(), vec![knows]);
        assert!(loaded.get_edge(gone).is_none());
    }

    /// An edge whose visible version is a cold record that cannot be read
    /// fails the write, naming the edge, instead of being left out.
    #[cfg(feature = "tiered-storage")]
    #[test]
    fn an_edge_whose_cold_record_cannot_be_read_fails_the_write() {
        use grafeo_common::mvcc::{ColdVersionRef, OptionalEpochId};
        use grafeo_common::types::TransactionId;

        let store = LpgStore::new().unwrap();
        let epoch = store.current_epoch();
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        let knows = store.create_edge(alix, gus, "KNOWS");
        store.create_edge(gus, alix, "KNOWS");
        // The version of `knows` points into a cold block the cold store
        // does not hold.
        store
            .edge_versions
            .write()
            .get_mut(&knows)
            .unwrap()
            .freeze_epoch(
                epoch,
                std::iter::once(ColdVersionRef {
                    epoch,
                    block_offset: 19,
                    length: 88,
                    created_by: TransactionId::SYSTEM,
                    deleted_epoch: OptionalEpochId::NONE,
                    deleted_by: None,
                }),
            );

        let error = try_chunks(&store, caps(4, 1 << 20))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(&format!("the record of edge {} cannot be read", knows.0)),
            "the write fails naming the edge: {error}"
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
        // Ids far above 0 (a 0.5.x compacted base that kept no ids folds
        // under ids holding its table number in the high bits), here in the
        // middle of a row group: the groups stay aligned to max_rows.
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
            decode(&written, 0, LABELS, u64::MAX - 1).values,
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
        let city = key_column(&meta, 0, Table::Node, "city");
        let position = |kind: ChunkKind, column: Column| {
            written
                .iter()
                .position(|(chunk, _)| {
                    chunk.kind == kind && (chunk.namespace, chunk.column_id) == column
                })
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

        let since = key_column(&meta, 0, Table::Edge, "since");
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
        let column = key_column(&meta(&written), 0, Table::Node, "city");
        let city: Vec<(ChunkKind, u64, u32)> = written
            .iter()
            .filter(|(chunk, _)| (chunk.namespace, chunk.column_id) == column)
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
            decode_kind(&written, ChunkKind::History, 0, column, 0).values,
            [(
                0,
                Value::List(Arc::from(vec![
                    version(3, Value::from("Amsterdam")),
                    version(19, Value::Null)
                ]))
            )]
        );
        assert_eq!(
            decode(&written, 0, column, 1).values,
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
        let embedding = key_column(&meta(&written), 0, Table::Node, "embedding");
        assert_eq!(cells(&written, ChunkKind::Column, 0, embedding), expected);
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
        let trips = key_column(&meta(&written), 0, Table::Node, "trips");
        let history = decode_kind(&written, ChunkKind::History, 0, trips, 0);
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

    /// A named graph may have the empty name: the default graph is the one
    /// with id 0, so the name cannot collide with it.
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
        let graphs: Vec<(u32, String)> = meta(&written)
            .graphs
            .into_iter()
            .map(|g| (g.id, g.name))
            .collect();
        assert_eq!(
            graphs,
            [(0, String::new()), (1, String::new()), (2, "trips".into())]
        );
        assert_eq!(cells(&written, ChunkKind::Column, 0, LABELS).len(), 1);
        assert_eq!(cells(&written, ChunkKind::Column, 1, LABELS).len(), 2);
    }

    /// Ids are permanent from one write to the next: a label, edge type or
    /// key added between two writes gets the next id and the names before
    /// keep theirs, also when the new ones sort before them (before stable
    /// ids, the next write sorted every name into place and ids moved).
    #[test]
    fn a_name_added_between_two_writes_gets_the_next_id_and_the_others_keep_theirs() {
        let store = LpgStore::new().unwrap();
        let mia = store.create_node(&["Person"]);
        store.set_node_property(mia, "name", Value::from("Mia"));
        let first = meta(&chunks(&store, ChunkCaps::DEFAULT));
        assert_eq!(first.graphs[0].labels.names, [(0, "Person".into())]);
        assert_eq!(first.graphs[0].keys.names, [(0, "name".into())]);

        let gus = store.create_node(&["Artist"]);
        store.set_node_property(gus, "age", Value::Int64(19));
        store.create_edge(mia, gus, "ADMIRES");
        let written = chunks(&store, ChunkCaps::DEFAULT);
        assert_layout(&written);
        let second = meta(&written);
        let graph = &second.graphs[0];
        assert_eq!(
            graph.labels.names,
            [(0, "Person".into()), (1, "Artist".into())]
        );
        assert_eq!(graph.keys.names, [(0, "name".into()), (1, "age".into())]);
        assert_eq!(graph.edge_types.names, [(0, "ADMIRES".into())]);
        assert_eq!(
            decode(&written, 0, LABELS, 0).values,
            [(0, Value::from("0")), (1, Value::from("1"))]
        );
        assert_eq!(
            cells(
                &written,
                ChunkKind::Column,
                0,
                (ChunkNamespace::NodeProperties, 0)
            ),
            [(0, Value::from("Mia"), 0)],
            "name keeps column 0"
        );
        assert_eq!(
            cells(
                &written,
                ChunkKind::Column,
                0,
                (ChunkNamespace::NodeProperties, 1)
            ),
            [(1, Value::Int64(19), 0)],
            "age gets column 1"
        );
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
        write_lpg_chunks(&store, caps(2, 1 << 20), OpenChangeSource::None, &mut image)
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
        // The dictionaries, read after the rows, hold the label and the edge
        // type the transaction created (as they do after a rollback), with
        // the next ids. The key it set after the node table's columns were
        // read has no column in this write, so no id yet.
        let graph = &meta.graphs[0];
        assert_eq!(
            graph.labels.names,
            [(0, "Person".into()), (1, "Traveller".into())]
        );
        assert_eq!(graph.edge_types.names, [(0, "VISITED".into())]);
        assert_eq!(graph.keys.names, [(0, "name".into())]);
        // Without `temporal` the transaction's label set is written in place
        // (the store keeps no versions); with it, the set an open
        // transaction wrote is pending and not written.
        #[cfg(not(feature = "temporal"))]
        let node_5 = "0,1";
        #[cfg(feature = "temporal")]
        let node_5 = "0";
        assert_eq!(
            decode(&written, 0, LABELS, 4).values,
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
        let label = |name: &str| id_in(&meta.graphs[0].labels, name);
        let mut gus = [label("Employee"), label("Person")];
        gus.sort_unstable();
        assert_eq!(
            decode(&written, 0, LABELS, 0).values,
            [
                (0, Value::from(label("Person").to_string())),
                (1, Value::from(format!("{},{}", gus[0], gus[1]))),
            ],
            "the committed label sets, ids ascending"
        );
    }

    /// Alix -KNOWS-> Gus -KNOWS-> Mia, with values on the nodes and the
    /// edges, and Prague -ROUTE-> Berlin in graph "trips": the committed
    /// state the transactions below begin from.
    fn travellers() -> LpgStore {
        let store = LpgStore::new().unwrap();
        let alix = store.create_node_with_props(
            &["Person"],
            [
                ("name", Value::from("Alix")),
                ("age", Value::Int64(19)),
                ("city", Value::from("Amsterdam")),
            ],
        );
        let gus = store.create_node_with_props(
            &["Person", "Employee"],
            [
                ("name", Value::from("Gus")),
                ("age", Value::Int64(3)),
                ("city", Value::from("Paris")),
            ],
        );
        let mia = store.create_node_with_props(
            &["Person", "Employee"],
            [("name", Value::from("Mia")), ("age", Value::Int64(88))],
        );
        store.create_edge_with_props(alix, gus, "KNOWS", [("since", Value::Int64(1988))]);
        store.create_edge_with_props(gus, mia, "KNOWS", [("since", Value::Int64(319))]);
        store.create_graph("trips").unwrap();
        let trips = store.graph("trips").unwrap();
        let prague = trips.create_node_with_props(&["City"], [("name", Value::from("Prague"))]);
        let berlin = trips.create_node_with_props(&["City"], [("name", Value::from("Berlin"))]);
        trips.create_edge_with_props(prague, berlin, "ROUTE", [("hours", Value::Int64(3))]);
        store
    }

    /// The ids [`travellers`] gives Alix, Gus and Mia, and the edges from
    /// Alix to Gus and from Gus to Mia.
    const ALIX: NodeId = NodeId(0);
    const GUS: NodeId = NodeId(1);
    const MIA: NodeId = NodeId(2);
    const ALIX_GUS: EdgeId = EdgeId(0);
    const GUS_MIA: EdgeId = EdgeId(1);

    // ── The committed state from change sets ────────────────────────

    /// A change set's slot for `graph` (`None`: the default graph).
    fn lpg_slot(
        set: &mut grafeo_common::change::ChangeSet,
        graph: Option<&str>,
    ) -> grafeo_common::change::GraphSlot {
        use grafeo_common::change::{DataModel, GraphRef};

        set.slot(GraphRef {
            model: DataModel::Lpg,
            key: graph.map(arcstr::ArcStr::from),
        })
        .unwrap()
    }

    /// Changes [`travellers`] in a transaction left open, through each
    /// graph's change target, recording the changes in one change set, and
    /// returns the committed view of its entries: Alix's values (one set
    /// twice, one added, one removed), Mia's labels, Gus's values and labels
    /// and then Gus with his edges deleted (one edge changed first), nodes
    /// and edges created (some deleted again), and in graph "trips" a value
    /// changed, an edge and a node deleted.
    fn change_travellers_through_a_change_set(store: &LpgStore) -> OpenChangesByGraph {
        use crate::graph::apply::Writer;
        use crate::graph::lpg::store::testing::Recorder;
        use grafeo_common::change::ChangeSet;
        use grafeo_common::types::TransactionId;

        let writer = Writer::Transaction {
            id: TransactionId::new(19),
            snapshot: store.current_epoch(),
        };
        let mut set = ChangeSet::new();
        let default = lpg_slot(&mut set, None);
        let trips_slot = lpg_slot(&mut set, Some("trips"));
        let main = Recorder {
            store,
            slot: default,
            writer,
        };
        // Alix: a value set twice, one added, one removed.
        for age in [88, 3] {
            main.set_node(&mut set, ALIX, "age", Value::Int64(age));
        }
        main.set_node(&mut set, ALIX, "nickname", Value::from("Al"));
        main.remove_node(&mut set, ALIX, "city");
        // Mia: a label added then removed, one removed then added back, one
        // removed.
        main.add_label(&mut set, MIA, "Traveller");
        main.remove_label(&mut set, MIA, "Traveller");
        main.remove_label(&mut set, MIA, "Person");
        main.add_label(&mut set, MIA, "Person");
        main.remove_label(&mut set, MIA, "Employee");
        // Gus: values and labels changed, then deleted with his edges, one of
        // them changed first.
        main.set_node(&mut set, GUS, "age", Value::Int64(19));
        main.set_node(&mut set, GUS, "nickname", Value::from("G"));
        main.add_label(&mut set, GUS, "Traveller");
        main.remove_label(&mut set, GUS, "Employee");
        main.set_edge(&mut set, ALIX_GUS, "since", Value::Int64(3));
        main.set_edge(&mut set, ALIX_GUS, "note", Value::from("Paris"));
        main.delete_edge(&mut set, ALIX_GUS);
        main.delete_edge(&mut set, GUS_MIA);
        main.delete_node(&mut set, GUS);
        // Nodes and edges the transaction created, and some of them deleted.
        let vincent = main.create_node(&mut set, &["Person"], &[]);
        main.set_node(&mut set, vincent, "name", Value::from("Vincent"));
        main.create_edge(&mut set, vincent, MIA, "KNOWS", &[]);
        let jules = main.create_node(&mut set, &["Person"], &[("name", Value::from("Jules"))]);
        let visited = main.create_edge(&mut set, ALIX, MIA, "VISITED", &[]);
        main.set_edge(&mut set, visited, "since", Value::Int64(88));
        main.delete_edge(&mut set, visited);
        main.delete_node(&mut set, jules);
        // The named graph: a value changed, an edge and a node deleted.
        let trips = store.graph("trips").unwrap();
        let named = Recorder {
            store: &trips,
            slot: trips_slot,
            writer,
        };
        named.set_node(&mut set, NodeId(0), "name", Value::from("Paris"));
        named.delete_edge(&mut set, EdgeId(0));
        named.delete_node(&mut set, NodeId(1));
        OpenChangesByGraph::index([
            (None, set.in_graph(default)),
            (Some("trips"), set.in_graph(trips_slot)),
        ])
    }

    /// A write while a transaction is open holds the committed state of
    /// what it changed, from the committed view of its change set: a value
    /// the transaction set twice, added or removed, labels it added and
    /// removed, a node and its edges it deleted after changing them, and the
    /// node and edge it created; in the default graph and in a named graph.
    /// So does a committed copy.
    #[test]
    fn open_changes_from_entries_matches_the_committed_state() {
        let recorded = travellers();
        let view = change_travellers_through_a_change_set(&recorded);
        let open = OpenChangeSource::ChangeSets(&view);
        for caps in [caps(2, 1 << 20), caps(1, 64), ChunkCaps::DEFAULT] {
            assert_layout(&try_chunks_from(&recorded, caps, open).unwrap());
            assert_same_graph(&travellers(), &round_trip_from(&recorded, caps, open));
        }
        assert_same_graph(&travellers(), &recorded.committed_copy_with(&view).unwrap());
        assert!(
            recorded.get_node(GUS).is_none(),
            "the store holds the open transaction's delete"
        );
    }

    /// A rollback to a savepoint undoes the change set's tail: a write from
    /// what the set keeps holds the committed state of it, of what the
    /// rollback restored, and of what the transaction changed after it.
    #[test]
    fn a_write_after_a_savepoint_rollback_holds_the_committed_state_from_change_sets() {
        use crate::graph::apply::{ChangeTarget, Writer};
        use crate::graph::lpg::store::testing::Recorder;
        use grafeo_common::change::ChangeSet;
        use grafeo_common::types::TransactionId;

        let store = travellers();
        let transaction = TransactionId::new(88);
        let mut set = ChangeSet::new();
        let slot = lpg_slot(&mut set, None);
        let recorder = Recorder {
            store: &store,
            slot,
            writer: Writer::Transaction {
                id: transaction,
                snapshot: store.current_epoch(),
            },
        };
        recorder.set_node(&mut set, GUS, "age", Value::Int64(19));
        let savepoint = set.mark();
        recorder.set_node(&mut set, ALIX, "age", Value::Int64(88));
        recorder.delete_edge(&mut set, ALIX_GUS);
        recorder.delete_edge(&mut set, GUS_MIA);
        recorder.delete_node(&mut set, GUS);
        let tail = set.split_off(savepoint);
        store.undo(transaction, &mut tail.iter()).unwrap();
        assert!(store.get_node(GUS).is_some(), "the rollback restored Gus");
        recorder.add_label(&mut set, ALIX, "Traveller");
        recorder.set_node(&mut set, MIA, "age", Value::Int64(3));

        let view = OpenChangesByGraph::index([(None, set.in_graph(slot))]);
        let open = OpenChangeSource::ChangeSets(&view);
        for caps in [caps(2, 1 << 20), ChunkCaps::DEFAULT] {
            assert_layout(&try_chunks_from(&store, caps, open).unwrap());
            assert_same_graph(&travellers(), &round_trip_from(&store, caps, open));
        }
    }

    /// A key that only a committed value an open transaction replaced has
    /// (no column of the store holds it) joins the property columns with the
    /// next key id, and its value is written.
    #[cfg(not(feature = "temporal"))]
    #[test]
    fn a_key_only_a_replaced_value_has_joins_the_columns() {
        use super::{Place, TableWriter, column, describe_graph};

        let store = LpgStore::new().unwrap();
        let key = PropertyKey::new("nickname");
        let committed = BTreeMap::from([(key.clone(), vec![(GUS, Some(Value::from("G")))])]);
        let table = TableWriter {
            table: Table::Node,
            fixed: vec![column(Table::Node.structure(), COLUMN_LABELS)],
            properties: &store.node_properties,
            committed: &committed,
            graph: &store,
        };
        let place = Place {
            graph_id: 0,
            graph: describe_graph(0, ""),
            caps: ChunkCaps::DEFAULT,
        };
        let mut image = MemoryImage::new();
        image
            .begin_section(SectionType::LpgStore, LPG_SECTION_VERSION)
            .unwrap();
        let rows = [Ok((GUS, vec![Some((Value::from(""), 0))]))];
        assert_eq!(
            table.write(&place, rows.into_iter(), &mut image).unwrap(),
            2
        );
        let section = image.take_section(SectionType::LpgStore).unwrap();
        let written: Vec<(ChunkMeta, Bytes)> = (0..section.chunks().len())
            .map(|index| (section.chunks()[index], section.fetch(index).unwrap()))
            .collect();
        let column_id = store.property_key_id(key.as_str());
        assert_eq!(column_id, 0, "the first key's id");
        assert_eq!(
            cells(
                &written,
                ChunkKind::Column,
                0,
                (ChunkNamespace::NodeProperties, column_id)
            ),
            [(1, Value::from("G"), 0)],
            "the committed value of the key"
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

    // ── Reading (F4) ────────────────────────────────────────────────

    /// Writes `store` with `caps`, reading what open transactions changed
    /// from `open`, and loads the chunks into `into`.
    fn load_into_from(
        store: &LpgStore,
        caps: ChunkCaps,
        open: OpenChangeSource<'_>,
        into: &LpgStore,
    ) -> Result<()> {
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::LpgStore, LPG_SECTION_VERSION)?;
        write_lpg_chunks(store, caps, open, &mut image)?;
        let source = image
            .section_source(SectionType::LpgStore)
            .expect("the section holds its metadata chunk");
        read_lpg_chunks(into, &*source)
    }

    /// `store` written with `caps` and loaded into a new store.
    fn round_trip(store: &LpgStore, caps: ChunkCaps) -> LpgStore {
        round_trip_from(store, caps, OpenChangeSource::None)
    }

    /// `store` written with `caps`, reading what open transactions changed
    /// from `open`, and loaded into a new store.
    fn round_trip_from(store: &LpgStore, caps: ChunkCaps, open: OpenChangeSource<'_>) -> LpgStore {
        let back = LpgStore::new().unwrap();
        load_into_from(store, caps, open, &back).unwrap();
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

    // ── The committed copy ──────────────────────────────────────────

    /// Asserts that [`LpgStore::committed_copy`] of `store` holds what a
    /// write of `store` loads, and returns the copy: in every graph the same
    /// nodes, edges, labels and values, next ids and registered labels and
    /// edge types (with `temporal`, also the same epoch and versions).
    fn committed_copy_as_loaded(store: &LpgStore) -> LpgStore {
        committed_copy_as_loaded_from(store, OpenChangeSource::None)
    }

    /// [`committed_copy_as_loaded`] while transactions are open, whose
    /// changes `open` gives.
    fn committed_copy_as_loaded_from(store: &LpgStore, open: OpenChangeSource<'_>) -> LpgStore {
        let copy = match open {
            OpenChangeSource::None => store.committed_copy(),
            OpenChangeSource::ChangeSets(view) => store.committed_copy_with(view),
        }
        .unwrap();
        let loaded = round_trip_from(store, ChunkCaps::DEFAULT, open);
        assert_same_graph(&loaded, &copy);
        assert_same_registers(&loaded, &copy, "the default graph");
        let mut names = loaded.graph_names();
        names.sort();
        for name in names {
            assert_same_registers(
                &loaded.graph(&name).unwrap(),
                &copy.graph(&name).unwrap(),
                &format!("graph {name:?}"),
            );
        }
        copy
    }

    /// Asserts that `loaded` and `copy` have the same next ids and
    /// registered labels and edge types and, with `temporal`, the same epoch
    /// and versions of every value.
    fn assert_same_registers(loaded: &LpgStore, copy: &LpgStore, graph: &str) {
        assert_eq!(
            (loaded.next_node_id(), loaded.next_edge_id()),
            (copy.next_node_id(), copy.next_edge_id()),
            "{graph}: the next node and edge ids"
        );
        let sorted = |mut names: Vec<String>| {
            names.sort();
            names
        };
        assert_eq!(
            sorted(loaded.all_labels()),
            sorted(copy.all_labels()),
            "{graph}: the registered labels"
        );
        assert_eq!(
            sorted(loaded.all_edge_types()),
            sorted(copy.all_edge_types()),
            "{graph}: the registered edge types"
        );
        #[cfg(feature = "temporal")]
        {
            assert_eq!(loaded.current_epoch(), copy.current_epoch(), "{graph}");
            for id in loaded.node_ids() {
                assert_eq!(
                    versions(loaded.node_property_history(id)),
                    versions(copy.node_property_history(id)),
                    "{graph}: the versions of node {}",
                    id.0
                );
            }
            for id in loaded.try_edge_ids().unwrap() {
                assert_eq!(
                    versions(loaded.edge_property_history(id)),
                    versions(copy.edge_property_history(id)),
                    "{graph}: the versions of edge {}",
                    id.0
                );
            }
        }
    }

    /// The versions of a property log, sorted by key: each as its epoch and
    /// its codec bytes.
    #[cfg(feature = "temporal")]
    fn versions(
        log: Vec<(PropertyKey, Vec<(grafeo_common::types::EpochId, Value)>)>,
    ) -> Vec<(String, Vec<(u64, Vec<u8>)>)> {
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
    }

    /// The committed copy of a store holds every kind of value, nodes with
    /// 0, 1 and 3 labels and the named graphs, as a write and a load do.
    #[test]
    fn a_committed_copy_holds_what_a_write_loads() {
        let store = round_trip_store();
        let copy = committed_copy_as_loaded(&store);
        assert_same_graph(&store, &copy);
    }

    /// The committed copy of a store whose transaction is still open holds
    /// the committed state, as a write does: what the transaction deleted,
    /// with the values and labels it changed as they were committed, and
    /// nothing it created, nor a value set on an id without a node. The
    /// store keeps what the transaction wrote.
    #[test]
    fn a_committed_copy_leaves_out_what_open_transactions_changed() {
        let store = travellers();
        let view = change_travellers_through_a_change_set(&store);
        let name = PropertyKey::new("name");
        store.set_node_property(NodeId::new(88), "name", Value::from("Vincent"));

        let copy = committed_copy_as_loaded_from(&store, OpenChangeSource::ChangeSets(&view));
        assert_same_graph(&travellers(), &copy);
        assert!(
            !copy
                .node_properties
                .column_ids(&name)
                .contains(&NodeId::new(88)),
            "the value of an id without a node is not copied"
        );
        assert!(
            store.get_node(GUS).is_none(),
            "the store still holds the open transaction's delete"
        );
    }

    /// With `temporal`, the committed copy holds the committed versions of
    /// each value (the null version of a removed one too) and the label
    /// sets of the store's epoch: not the version nor the labels a
    /// transaction still open wrote.
    #[cfg(feature = "temporal")]
    #[test]
    fn a_committed_copy_holds_the_committed_versions() {
        use grafeo_common::types::{EpochId, TransactionId};

        let store = LpgStore::new().unwrap();
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person", "Employee"]);
        let at = EpochId::new;
        store.set_node_property_at_epoch(alix, "city", Value::from("Amsterdam"), at(3));
        store.set_node_property_at_epoch(alix, "city", Value::from("Berlin"), at(19));
        store.set_node_property_at_epoch(gus, "city", Value::from("Prague"), at(3));
        store.set_node_property_at_epoch(gus, "city", Value::Null, at(19));
        let knows = store.create_edge(alix, gus, "KNOWS");
        store.set_edge_property_at_epoch(knows, "since", Value::Int64(3), at(3));
        store.sync_epoch(at(88));
        let transaction = TransactionId::new(88);
        store
            .set_node_property_versioned(alix, "city", Value::from("Paris"), transaction)
            .unwrap();
        assert!(store.add_label_versioned(alix, "Traveller", transaction));
        assert!(store.remove_label_versioned(gus, "Employee", transaction));

        let copy = committed_copy_as_loaded(&store);
        let city = PropertyKey::new("city");
        assert_eq!(
            copy.get_node_property(alix, &city),
            Some(Value::from("Berlin")),
            "the committed value, not the open transaction's"
        );
        assert_eq!(
            copy.get_node_property_at_epoch(alix, &city, at(3)),
            Some(Value::from("Amsterdam")),
            "an older version"
        );
        assert_eq!(copy.get_node_property(gus, &city), None, "a removed value");
        let labels = |id: NodeId| {
            let mut labels: Vec<String> = copy
                .get_node(id)
                .unwrap()
                .labels
                .iter()
                .map(ToString::to_string)
                .collect();
            labels.sort();
            labels
        };
        assert_eq!(labels(alix), ["Person"], "the committed labels of Alix");
        assert_eq!(
            labels(gus),
            ["Employee", "Person"],
            "the committed labels of Gus"
        );
        assert_eq!(copy.current_epoch(), at(88));
    }

    /// The committed copy reads spilled values, and fails rather than copy a
    /// column without one it cannot read, as a write does (#594).
    #[cfg(not(feature = "temporal"))]
    #[test]
    fn a_committed_copy_reads_spilled_values_and_fails_on_an_unreadable_one() {
        use crate::graph::lpg::property::test_backing::MemoryBacking;

        let store = LpgStore::new().unwrap();
        let key = PropertyKey::new("embedding");
        let embedding = Value::Vector(vec![3.0, 19.0].into());
        let alix = store.create_node_with_props(&["Item"], [("embedding", embedding.clone())]);
        let snapshot = store.node_property_column_entries(&key).unwrap();
        let backing = MemoryBacking::of(&snapshot);
        assert!(store.spill_node_property_column(&key, backing.clone(), &snapshot));

        let copy = store.committed_copy().unwrap();
        assert_eq!(copy.get_node_property(alix, &key), Some(embedding));

        backing.fail_reads(true);
        assert!(store.committed_copy().is_err(), "a copy without the value");
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
        let visits = key_column(&meta(&written), 0, Table::Node, "visits");
        assert_eq!(
            cells(&written, ChunkKind::Column, 0, visits),
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
        // Every name keeps its id in its graph's dictionary, used or not.
        let (default, trips) = (&meta.graphs[0], &meta.graphs[1]);
        assert_eq!(
            default.labels.names,
            [
                (0, "Person".into()),
                (1, "Ghost".into()),
                (2, "City".into())
            ]
        );
        assert_eq!(
            default.edge_types.names,
            [(0, "OLD".into()), (1, "KNOWS".into())]
        );
        assert_eq!(
            trips.labels.names,
            [
                (0, "City".into()),
                (1, "Museum".into()),
                (2, "Person".into())
            ]
        );
        assert_eq!(
            trips.edge_types.names,
            [(0, "FERRY".into()), (1, "KNOWS".into())]
        );

        let back = round_trip(&store, ChunkCaps::DEFAULT);
        assert_eq!(registries(&back), before, "every graph's names come back");
        assert_eq!(
            chunks(&back, ChunkCaps::DEFAULT),
            written,
            "the loaded store writes the same chunks"
        );
    }

    /// A metadata chunk alone (no rows), loaded into a new store.
    fn load_meta(meta: &LpgMeta) -> LpgStore {
        let mut image = MemoryImage::new();
        image
            .begin_section(SectionType::LpgStore, LPG_SECTION_VERSION)
            .unwrap();
        image
            .write_chunk(ChunkMeta::meta(), &encode_lpg_meta(meta).unwrap())
            .unwrap();
        let source = image.section_source(SectionType::LpgStore).unwrap();
        let store = LpgStore::new().unwrap();
        read_lpg_chunks(&store, &*source).unwrap();
        store
    }

    /// Graph ids are permanent: a graph created later that sorts first by
    /// name takes the next id and the others keep theirs (before stable ids
    /// a graph's id was its position in name order); a dropped graph's id is
    /// never given again, also after a load, so a graph created again under
    /// its name gets a new one.
    #[test]
    fn graph_ids_are_never_reused_or_renumbered() {
        let ids = |meta: &LpgMeta| -> Vec<(u32, String)> {
            meta.graphs
                .iter()
                .map(|graph| (graph.id, graph.name.clone()))
                .collect()
        };
        let store = LpgStore::new().unwrap();
        store.create_graph("trips").unwrap();
        store.create_graph("travel").unwrap();
        store.graph("trips").unwrap().create_node(&["City"]);
        store.create_graph("archive").unwrap();
        let written = chunks(&store, ChunkCaps::DEFAULT);
        assert_layout(&written);
        assert_eq!(
            ids(&meta(&written)),
            [
                (0, String::new()),
                (1, "trips".into()),
                (2, "travel".into()),
                (3, "archive".into())
            ],
            "by id, not by name"
        );
        assert!(
            data(&written).iter().all(|(chunk, _)| chunk.graph_id == 1),
            "trips' rows keep graph id 1 although archive sorts first"
        );

        assert!(store.drop_graph("travel"));
        let back = round_trip(&store, ChunkCaps::DEFAULT);
        let reloaded = meta(&chunks(&back, ChunkCaps::DEFAULT));
        assert_eq!(
            ids(&reloaded),
            [
                (0, String::new()),
                (1, "trips".into()),
                (3, "archive".into())
            ],
            "a gap where travel was"
        );
        assert_eq!(reloaded.next_graph_id, 4);
        back.create_graph("travel").unwrap();
        assert_eq!(
            back.graph("travel").unwrap().graph_id(),
            4,
            "travel created again gets a new id"
        );
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

        let store = LpgStore::new().unwrap();
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
        assert_eq!(store.label_id("City"), Some(5));
        let alix = store.create_node(&["Person", "Station"]);
        assert_eq!(
            store.label_id("Station"),
            Some(10),
            "after the file's next id"
        );
        assert_eq!(store.property_key_id("age"), 12);
        store.set_node_property(alix, "name", Value::from("Alix"));
        let written = chunks(&store, ChunkCaps::DEFAULT);
        assert_layout(&written);
        let graph = &meta(&written).graphs[0];
        assert_eq!(
            graph.labels.names,
            [
                (0, "Person".into()),
                (5, "City".into()),
                (9, "Museum".into()),
                (10, "Station".into())
            ]
        );
        assert_eq!(graph.labels.next_id, 11);
        assert_eq!(
            graph.keys,
            DictionaryMeta {
                next_id: 13,
                names: vec![(3, "name".into()), (12, "age".into())],
            }
        );
        let row = alix.as_u64();
        assert_eq!(
            cells(&written, ChunkKind::Column, 0, LABELS),
            [(row, Value::from("0,10"), 0)]
        );
        assert_eq!(
            cells(
                &written,
                ChunkKind::Column,
                0,
                (ChunkNamespace::NodeProperties, 3)
            ),
            [(row, Value::from("Alix"), 0)],
            "the key's column is its id from the file"
        );
    }

    /// `clear()` keeps the ids: a cleared graph that meets a name again
    /// gives it its old id, and a new name the next one.
    #[test]
    fn clear_keeps_the_name_ids() {
        let store = LpgStore::new().unwrap();
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["City"]);
        store.create_edge(alix, gus, "LIVES_IN");
        store.clear();
        let mia = store.create_node(&["City"]);
        let vincent = store.create_node(&["Artist"]);
        store.create_edge(mia, vincent, "KNOWS");
        assert_eq!(
            (
                store.label_id("Person"),
                store.label_id("City"),
                store.label_id("Artist")
            ),
            (Some(0), Some(1), Some(2))
        );
        assert_eq!(
            (store.edge_type_id("LIVES_IN"), store.edge_type_id("KNOWS")),
            (Some(0), Some(1))
        );
        let written = chunks(&store, ChunkCaps::DEFAULT);
        assert_layout(&written);
        let graph = &meta(&written).graphs[0];
        assert_eq!(names_of(&graph.labels), ["Person", "City", "Artist"]);
        assert_eq!(names_of(&graph.edge_types), ["LIVES_IN", "KNOWS"]);
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
        // A store whose ids start far above 0.
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
        write_lpg_chunks(&store, caps(2, 1 << 20), OpenChangeSource::None, &mut image).unwrap();
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

        fn stored_length(&self, index: usize) -> Result<u64> {
            self.inner.stored_length(index)
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
        write_lpg_chunks(&store, caps(4, 512), OpenChangeSource::None, &mut image).unwrap();
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
