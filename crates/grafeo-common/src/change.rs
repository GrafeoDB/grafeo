//! Change sets: what a transaction changed, one entry per logical write.
//!
//! A transaction records each write as it applies it, in order, in its
//! [`ChangeSet`]. A data entry holds two halves:
//!
//! - its op, a [`DataOp`]: what the write did, with its after-image (a
//!   created entity's labels or type, endpoints and properties, a new value,
//!   a label). The op is the payload of the write's log record
//!   ([`LogRecord::Data`]), so the log holds exactly what was applied and
//!   replay applies the same op again;
//! - its before-image, a [`Before`]: what the write replaced (the old value,
//!   the node's labels before it, a deleted entity's labels, endpoints and
//!   properties). It stays in memory for undo and change data capture, and
//!   is never logged.
//!
//! With them a data entry keeps its [`PendingVersion`]: whether the write
//! created the transaction's pending version of what it wrote or changed
//! one an earlier write of the transaction created. Undo needs it in a store
//! that changes a pending version in place; it is never logged either.
//!
//! One set serves every consumer of a transaction's changes: the log group
//! written at commit ([`ChangeSet::log_records`]), the stamping of the commit
//! epoch, rollback and savepoints ([`ChangeSet::split_off`]), and change data
//! capture. None of them reads the store back to learn what changed. Entries
//! carry no timestamps: change data capture gives a commit one when it
//! publishes it.
//!
//! An entry names its graph by a [`GraphSlot`], an index into the set's table
//! of [`GraphRef`]s, so a transaction that writes several graphs keeps one
//! sequence of entries. Its op names the table and the entity
//! ([`DataOp::entity`]) and, for a property, the key
//! ([`DataOp::property_key`]). Labels, edge types and property keys travel as
//! names: a store maps them to ids of its own.
//!
//! A bulk write records a [`BulkRange`] instead of an entry per row: the ids
//! it reserved in one table, undone and stamped as a range. A bulk write that
//! holds commits off for its whole run (an import) logs its rows as it
//! applies them and keeps none. One inside a transaction (a batch call)
//! cannot log before its commit, so when the commit's log or change data
//! capture reads its rows it keeps them with the range
//! ([`ChangeSet::push_bulk_rows`]): each a create, without an entry, a
//! before-image or a pending version of its own.
//!
//! A [`StandaloneOp`] (a graph, catalog or RDF graph operation) is no entry
//! of a set: it is validated, logged in a group of its own
//! ([`LogRecord::Standalone`]) and applied after that group.
//!
//! [`LogRecord::Data`]: crate::storage::log_record::LogRecord::Data
//! [`LogRecord::Standalone`]: crate::storage::log_record::LogRecord::Standalone

use std::ops::Range;
use std::sync::Arc;

use smallvec::SmallVec;

use crate::collections::GrafeoMap;
use crate::storage::catalog_record::{CatalogKey, CatalogRecord};
use crate::storage::log_record::{LogRecordRef, RdfGraphTarget, TermRecord, TripleRecord};
use crate::types::{ArcStr, EdgeId, NodeId, PropertyKey, Value};
use crate::utils::error::{Error, Result};

/// The labels of a node, as an entry holds them: most nodes have one or two.
pub type Labels = SmallVec<[ArcStr; 2]>;

/// The properties of a node or an edge, as an entry holds them, in the order
/// they were written.
pub type Properties = Vec<(PropertyKey, Value)>;

/// The data model of a graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataModel {
    /// A labeled property graph: nodes, edges, labels and properties.
    Lpg,
    /// An RDF graph: triples.
    Rdf,
}

/// A graph a change set writes: its model and its storage key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphRef {
    /// The graph's model.
    pub model: DataModel,
    /// The graph's storage key; `None` for the model's default graph.
    pub key: Option<ArcStr>,
}

/// A graph of a change set, by its place in the set's graph table (see
/// [`ChangeSet::slot`]). A slot means something only in the set that gave it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphSlot(u32);

impl GraphSlot {
    /// The slot's place in the set's graph table ([`ChangeSet::graphs`]).
    #[must_use]
    pub const fn index(self) -> usize {
        // A u32 always fits a usize: the targets this crate builds for (it
        // needs std) have pointers of 32 bits or more.
        self.0 as usize
    }
}

/// The table of a labeled property graph an entry writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Table {
    /// The node table.
    Nodes,
    /// The edge table.
    Edges,
}

/// The node or edge an entry writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Entity {
    /// A node.
    Node(NodeId),
    /// An edge.
    Edge(EdgeId),
}

impl Entity {
    /// The table the entity lives in.
    #[must_use]
    pub const fn table(self) -> Table {
        match self {
            Self::Node(_) => Table::Nodes,
            Self::Edge(_) => Table::Edges,
        }
    }
}

/// What a write did: the redo half of an entry, with its after-image. It is
/// the payload of the write's log record (kinds 16 to 25, 64 and 65, see
/// [`log_record`](crate::storage::log_record)).
///
/// Every op names its entity by a real id: a live write reserves the id of
/// what it creates before it applies the create, and replay passes the
/// logged one. Matches over it are exhaustive on purpose: an op added later
/// must be handled by every consumer.
#[derive(Debug, Clone, PartialEq)]
pub enum DataOp {
    /// A node was created with its labels and properties.
    CreateNode {
        /// The node's id.
        id: NodeId,
        /// Its labels.
        labels: Labels,
        /// Its properties.
        properties: Properties,
    },
    /// A node was deleted. Its edges were deleted before it, each by an op
    /// of its own (`DETACH DELETE` included).
    DeleteNode {
        /// The node's id.
        id: NodeId,
    },
    /// An edge was created with its properties.
    CreateEdge {
        /// The edge's id.
        id: EdgeId,
        /// The node it leaves.
        src: NodeId,
        /// The node it enters.
        dst: NodeId,
        /// Its type.
        edge_type: ArcStr,
        /// Its properties.
        properties: Properties,
    },
    /// An edge was deleted.
    DeleteEdge {
        /// The edge's id.
        id: EdgeId,
    },
    /// A node property was set.
    SetNodeProperty {
        /// The node's id.
        id: NodeId,
        /// The property key.
        key: PropertyKey,
        /// The new value.
        value: Value,
    },
    /// A node property was removed.
    RemoveNodeProperty {
        /// The node's id.
        id: NodeId,
        /// The property key.
        key: PropertyKey,
    },
    /// An edge property was set.
    SetEdgeProperty {
        /// The edge's id.
        id: EdgeId,
        /// The property key.
        key: PropertyKey,
        /// The new value.
        value: Value,
    },
    /// An edge property was removed.
    RemoveEdgeProperty {
        /// The edge's id.
        id: EdgeId,
        /// The property key.
        key: PropertyKey,
    },
    /// A label was added to a node.
    AddNodeLabel {
        /// The node's id.
        id: NodeId,
        /// The label.
        label: ArcStr,
    },
    /// A label was removed from a node.
    RemoveNodeLabel {
        /// The node's id.
        id: NodeId,
        /// The label.
        label: ArcStr,
    },
    /// An RDF triple was inserted.
    InsertTriple {
        /// The triple.
        triple: Box<TripleRecord>,
    },
    /// An RDF triple was deleted.
    DeleteTriple {
        /// The triple.
        triple: Box<TripleRecord>,
    },
}

impl DataOp {
    /// The model of the graphs the op applies to: triples to RDF graphs,
    /// every other op to labeled property graphs.
    #[must_use]
    pub const fn model(&self) -> DataModel {
        match self {
            Self::InsertTriple { .. } | Self::DeleteTriple { .. } => DataModel::Rdf,
            Self::CreateNode { .. }
            | Self::DeleteNode { .. }
            | Self::CreateEdge { .. }
            | Self::DeleteEdge { .. }
            | Self::SetNodeProperty { .. }
            | Self::RemoveNodeProperty { .. }
            | Self::SetEdgeProperty { .. }
            | Self::RemoveEdgeProperty { .. }
            | Self::AddNodeLabel { .. }
            | Self::RemoveNodeLabel { .. } => DataModel::Lpg,
        }
    }

    /// The node or edge the op writes, which names its table; `None` for a
    /// triple.
    #[must_use]
    pub const fn entity(&self) -> Option<Entity> {
        match self {
            Self::CreateNode { id, .. }
            | Self::DeleteNode { id }
            | Self::SetNodeProperty { id, .. }
            | Self::RemoveNodeProperty { id, .. }
            | Self::AddNodeLabel { id, .. }
            | Self::RemoveNodeLabel { id, .. } => Some(Entity::Node(*id)),
            Self::CreateEdge { id, .. }
            | Self::DeleteEdge { id }
            | Self::SetEdgeProperty { id, .. }
            | Self::RemoveEdgeProperty { id, .. } => Some(Entity::Edge(*id)),
            Self::InsertTriple { .. } | Self::DeleteTriple { .. } => None,
        }
    }

    /// The key of the property the op sets or removes; `None` for the other
    /// ops.
    #[must_use]
    pub const fn property_key(&self) -> Option<&PropertyKey> {
        match self {
            Self::SetNodeProperty { key, .. }
            | Self::RemoveNodeProperty { key, .. }
            | Self::SetEdgeProperty { key, .. }
            | Self::RemoveEdgeProperty { key, .. } => Some(key),
            Self::CreateNode { .. }
            | Self::DeleteNode { .. }
            | Self::CreateEdge { .. }
            | Self::DeleteEdge { .. }
            | Self::AddNodeLabel { .. }
            | Self::RemoveNodeLabel { .. }
            | Self::InsertTriple { .. }
            | Self::DeleteTriple { .. } => None,
        }
    }

    /// Whether every id the op names (the entity, an edge's endpoints) is a
    /// real one, not [`NodeId::INVALID`] or [`EdgeId::INVALID`].
    const fn names_real_ids(&self) -> bool {
        match self {
            Self::CreateEdge { id, src, dst, .. } => {
                id.is_valid() && src.is_valid() && dst.is_valid()
            }
            Self::InsertTriple { .. } | Self::DeleteTriple { .. } => true,
            Self::CreateNode { .. }
            | Self::DeleteNode { .. }
            | Self::DeleteEdge { .. }
            | Self::SetNodeProperty { .. }
            | Self::RemoveNodeProperty { .. }
            | Self::SetEdgeProperty { .. }
            | Self::RemoveEdgeProperty { .. }
            | Self::AddNodeLabel { .. }
            | Self::RemoveNodeLabel { .. } => match self.entity() {
                Some(Entity::Node(id)) => id.is_valid(),
                Some(Entity::Edge(id)) => id.is_valid(),
                None => true,
            },
        }
    }
}

/// What a write replaced: the undo half of an entry, never logged. Undo
/// restores it and change data capture reports it, without reading the store.
#[derive(Debug, Clone, PartialEq)]
pub enum Before {
    /// Nothing: the write created what it wrote, or inserted or deleted a
    /// triple (an RDF store applies those when the transaction commits).
    Absent,
    /// The value a property set or removal replaced: `None` when the
    /// property was not there.
    Value(Option<Value>),
    /// The node's labels before a label was added or removed.
    Labels(Labels),
    /// The node a delete removed.
    Node(Box<NodeImage>),
    /// The edge a delete removed.
    Edge(Box<EdgeImage>),
}

impl Before {
    /// Whether this is the before-image `op` has, of a write that changed
    /// something: [`Absent`](Self::Absent) for a create or a triple,
    /// [`Value`](Self::Value) for a property set and a value for a removal,
    /// [`Labels`](Self::Labels) without an added label or with a removed
    /// one, [`Node`](Self::Node) or [`Edge`](Self::Edge) for a delete.
    fn fits(&self, op: &DataOp) -> bool {
        match op {
            DataOp::CreateNode { .. }
            | DataOp::CreateEdge { .. }
            | DataOp::InsertTriple { .. }
            | DataOp::DeleteTriple { .. } => matches!(self, Self::Absent),
            DataOp::SetNodeProperty { .. } | DataOp::SetEdgeProperty { .. } => {
                matches!(self, Self::Value(_))
            }
            DataOp::RemoveNodeProperty { .. } | DataOp::RemoveEdgeProperty { .. } => {
                matches!(self, Self::Value(Some(_)))
            }
            DataOp::AddNodeLabel { label, .. } => {
                matches!(self, Self::Labels(labels) if !labels.contains(label))
            }
            DataOp::RemoveNodeLabel { label, .. } => {
                matches!(self, Self::Labels(labels) if labels.contains(label))
            }
            DataOp::DeleteNode { .. } => matches!(self, Self::Node(_)),
            DataOp::DeleteEdge { .. } => matches!(self, Self::Edge(_)),
        }
    }

    /// The before-image's variant, for errors.
    const fn shape(&self) -> &'static str {
        match self {
            Self::Absent => "Absent",
            Self::Value(_) => "Value",
            Self::Labels(_) => "Labels",
            Self::Node(_) => "Node",
            Self::Edge(_) => "Edge",
        }
    }
}

/// What a write did to the transaction's pending version of what it wrote,
/// as the store that applied it reports (what one version covers, an
/// entity, a property or a node's labels, is the store's own): whether undo
/// drops a version or rewrites one. Never logged: replay commits each write
/// at once.
///
/// A store that keeps one pending version per transaction and changes it
/// in place on the transaction's next write (multi-version concurrency
/// control) needs it: a rollback to a savepoint must keep a version that
/// holds what the transaction wrote before the savepoint, and the
/// before-image alone does not say whether it is the committed state or the
/// transaction's own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PendingVersion {
    /// The write created its pending version, or the store keeps none for
    /// what it wrote and writes in place. Undo drops the version, or writes
    /// the before-image back. Every create and every triple is one.
    Created,
    /// The write changed, in place, the pending version an earlier write of
    /// the transaction created, so the before-image is that write's state.
    /// Undo writes the before-image back into the version and keeps it.
    Replaced,
}

/// A node as a delete found it.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeImage {
    /// Its labels.
    pub labels: Labels,
    /// Its properties.
    pub properties: Properties,
}

/// An edge as a delete found it.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeImage {
    /// The node it left.
    pub src: NodeId,
    /// The node it entered.
    pub dst: NodeId,
    /// Its type.
    pub edge_type: ArcStr,
    /// Its properties.
    pub properties: Properties,
}

/// The ids a bulk write reserved in one table of a graph: one entry however
/// many rows it creates there. Undo removes what exists in the range, and
/// the commit stamps it, as a range. The rows themselves are logged as the
/// bulk write applies them, or kept with the range for the commit's log
/// (see [`ChangeSet::push_bulk_rows`]).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BulkRange {
    /// The graph, a labeled property graph.
    pub graph: GraphSlot,
    /// The table the ids belong to.
    pub table: Table,
    /// The ids reserved, as raw node or edge ids.
    pub ids: Range<u64>,
}

impl BulkRange {
    /// Whether `row` is a create in this range: a node create for a node
    /// range, an edge create with real endpoints for an edge range, at one
    /// of the range's ids.
    fn holds(&self, row: &DataOp) -> bool {
        let id = match (self.table, row) {
            (Table::Nodes, DataOp::CreateNode { id, .. }) => id.as_u64(),
            (Table::Edges, DataOp::CreateEdge { id, .. }) if row.names_real_ids() => id.as_u64(),
            _ => return false,
        };
        self.ids.contains(&id)
    }
}

/// One entry of a change set. Matches over it are exhaustive on purpose: an
/// entry kind added later must be handled by every consumer.
#[derive(Debug, Clone, PartialEq)]
pub enum Change {
    /// One logical write: what it did, and what it replaced.
    Data {
        /// The graph it wrote.
        graph: GraphSlot,
        /// What it did; logged.
        op: DataOp,
        /// What it replaced; kept for undo and change data capture.
        before: Before,
        /// Whether it created the transaction's pending version of what it
        /// wrote or changed one; kept for undo.
        version: PendingVersion,
    },
    /// The ids a bulk write reserved.
    Bulk(BulkRange),
}

impl Change {
    /// The graph the entry writes.
    #[must_use]
    pub const fn graph(&self) -> GraphSlot {
        match self {
            Self::Data { graph, .. } => *graph,
            Self::Bulk(range) => range.graph,
        }
    }
}

/// A change applied on its own, never inside a transaction's change set: it
/// is validated, logged as a group of its own and applied after that group.
/// The payload of log record kinds 32, 33, 40, 41 and 66 to 71.
#[derive(Debug, Clone, PartialEq)]
pub enum StandaloneOp {
    /// A named labeled property graph was created.
    CreateGraph {
        /// The graph's storage key.
        name: String,
    },
    /// A named labeled property graph was dropped.
    DropGraph {
        /// The graph's storage key.
        name: String,
    },
    /// A catalog record was created or replaced.
    PutCatalog(CatalogRecord),
    /// A catalog record was dropped.
    DropCatalog(CatalogKey),
    /// RDF graphs were created, dropped, cleared, copied, moved or added to.
    RdfGraph(RdfGraphOp),
}

/// An operation on whole RDF graphs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RdfGraphOp {
    /// A named RDF graph was created.
    Create {
        /// The graph's name.
        name: String,
    },
    /// RDF graphs were dropped.
    Drop {
        /// The graphs.
        target: RdfGraphTarget,
    },
    /// RDF graphs were cleared.
    Clear {
        /// The graphs.
        target: RdfGraphTarget,
    },
    /// An RDF graph's triples were copied over another graph's.
    Copy {
        /// The graph copied; `None` for the default graph.
        source: Option<String>,
        /// The graph replaced; `None` for the default graph.
        target: Option<String>,
    },
    /// An RDF graph's triples were moved over another graph's.
    Move {
        /// The graph moved; `None` for the default graph.
        source: Option<String>,
        /// The graph replaced; `None` for the default graph.
        target: Option<String>,
    },
    /// An RDF graph's triples were added to another graph.
    Add {
        /// The graph added; `None` for the default graph.
        source: Option<String>,
        /// The graph added to; `None` for the default graph.
        target: Option<String>,
    },
}

/// A position in a change set, for a savepoint: the entries recorded after
/// it are the ones a rollback to the savepoint undoes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChangeMark(usize);

/// The changes of one transaction, in the order they were applied.
///
/// Undo walks entries in reverse: the order they were recorded is the order
/// they were applied, per entity. The set checks each entry as it takes it
/// (its graph, its model, its ids, the shape of its before-image, its
/// pending version), so a consumer can rely on them.
#[derive(Debug, Default)]
pub struct ChangeSet {
    /// The graphs the set writes; a [`GraphSlot`] indexes it.
    graphs: Vec<GraphRef>,
    /// The slot of each graph in `graphs`.
    slots: GrafeoMap<GraphRef, GraphSlot>,
    /// The entries, in recorded order.
    entries: Vec<Change>,
    /// The rows kept with bulk ranges (see [`push_bulk_rows`](Self::push_bulk_rows)):
    /// the index of each range's entry with its rows, in entry order.
    bulk_rows: Vec<(usize, Vec<DataOp>)>,
    /// What the entries and the kept rows hold on the heap.
    held: Held,
}

/// The before-image of a row kept with a bulk range: a create replaced
/// nothing.
static ABSENT: Before = Before::Absent;

impl ChangeSet {
    /// An empty change set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The slot of `graph`, added to the set's graph table the first time.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidValue`] when the set already writes
    /// [`u32::MAX`] graphs plus one.
    pub fn slot(&mut self, graph: GraphRef) -> Result<GraphSlot> {
        if let Some(&slot) = self.slots.get(&graph) {
            return Ok(slot);
        }
        let slot = u32::try_from(self.graphs.len())
            .map(GraphSlot)
            .map_err(|_| {
                Error::InvalidValue(format!(
                    "a transaction writes at most {} graphs",
                    u64::from(u32::MAX) + 1
                ))
            })?;
        self.slots.insert(graph.clone(), slot);
        self.graphs.push(graph);
        Ok(slot)
    }

    /// The graph of `slot`, or `None` for a slot this set did not give.
    #[must_use]
    pub fn graph(&self, slot: GraphSlot) -> Option<&GraphRef> {
        self.graphs.get(slot.index())
    }

    /// The graphs the set writes, by slot. A graph stays once it has a slot,
    /// also when a rollback to a savepoint removed its entries.
    #[must_use]
    pub fn graphs(&self) -> &[GraphRef] {
        &self.graphs
    }

    /// Records a write to `graph` that applied `op`, replaced `before`, and
    /// created or replaced the transaction's pending version of what it
    /// wrote (`version`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Internal`], and records nothing, when `graph` is not
    /// a slot of this set, `op` belongs to the other data model, `op` names
    /// an invalid id, `before` is not the before-image `op` has (see
    /// [`Before`]) or says the write changed nothing (a removal that replaced
    /// no value, a label added to labels that held it, or one removed from
    /// labels that did not), or a create or a triple says it replaced a
    /// pending version. A write that changes nothing is not recorded, so
    /// replay, which refuses one, never meets it.
    pub fn push(
        &mut self,
        graph: GraphSlot,
        op: DataOp,
        before: Before,
        version: PendingVersion,
    ) -> Result<()> {
        self.check_graph(graph, op.model(), || {
            format!("a change of kind {}", op.kind())
        })?;
        if !op.names_real_ids() {
            return Err(Error::Internal(format!(
                "a change of kind {} names an invalid id: {op:?}",
                op.kind()
            )));
        }
        if !before.fits(&op) {
            return Err(Error::Internal(format!(
                "a change of kind {} with a before-image of shape {} that does not fit it, or \
                 says it changed nothing",
                op.kind(),
                before.shape()
            )));
        }
        // `fits` gives a create and a triple `Absent`, and only them.
        if version == PendingVersion::Replaced && matches!(before, Before::Absent) {
            return Err(Error::Internal(format!(
                "a change of kind {} that replaced a pending version: a create or a triple \
                 creates what it writes",
                op.kind()
            )));
        }
        let change = Change::Data {
            graph,
            op,
            before,
            version,
        };
        self.held.change(&change, Count::Add);
        self.entries.push(change);
        Ok(())
    }

    /// Records the ids a bulk write reserved. An empty range reserves
    /// nothing and adds no entry.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Internal`], and records nothing, when the range's
    /// graph is not a slot of this set or not a labeled property graph.
    pub fn push_bulk(&mut self, range: BulkRange) -> Result<()> {
        self.check_graph(range.graph, DataModel::Lpg, || {
            format!("a bulk range of {:?} ids", range.table)
        })?;
        if !range.ids.is_empty() {
            self.entries.push(Change::Bulk(range));
        }
        Ok(())
    }

    /// Keeps `rows`, which a bulk write applied in the range it recorded
    /// last ([`push_bulk`](Self::push_bulk)), with that range for the
    /// commit's log and change data capture ([`ops`](Self::ops),
    /// [`log_records`](Self::log_records)), after the rows kept with it
    /// before. They add no entry: the range stays the one entry undo and
    /// stamp read. A rollback to a savepoint before the range drops them
    /// with it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Internal`], and keeps nothing, when the last entry is
    /// not a range of `graph`, or a row is not a create in the range's table
    /// at one of its ids (with real endpoints for an edge).
    pub fn push_bulk_rows(&mut self, graph: GraphSlot, rows: Vec<DataOp>) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let at = self.entries.len().checked_sub(1);
        let range = match at.map(|at| &self.entries[at]) {
            Some(Change::Bulk(range)) if range.graph == graph => range,
            last => {
                return Err(Error::Internal(format!(
                    "{} rows of a bulk write in graph slot {}, but the last entry is not a bulk \
                     range of that graph: {last:?}",
                    rows.len(),
                    graph.index()
                )));
            }
        };
        if let Some(row) = rows.iter().find(|row| !range.holds(row)) {
            return Err(Error::Internal(format!(
                "a row that is no create in the bulk range of {:?} ids {:?}: {row:?}",
                range.table, range.ids
            )));
        }
        for row in &rows {
            self.held.row(row, Count::Add);
        }
        // `at` is the last entry: the rows kept last are its or older.
        let at = self.entries.len() - 1;
        match self.bulk_rows.last_mut() {
            Some((index, kept)) if *index == at => kept.extend(rows),
            _ => self.bulk_rows.push((at, rows)),
        }
        Ok(())
    }

    /// The rows kept with the bulk range of entry `index`.
    fn rows_of(&self, index: usize) -> &[DataOp] {
        self.bulk_rows
            .binary_search_by_key(&index, |(at, _)| *at)
            .map_or(&[], |found| self.bulk_rows[found].1.as_slice())
    }

    /// What the set's writes did, in the order they were applied, with the
    /// graph of each and what it replaced: the op of each data entry, and at
    /// a bulk range's place the rows kept with it (see
    /// [`push_bulk_rows`](Self::push_bulk_rows)), creates that replaced
    /// nothing. What the commit logs and reports to change data capture.
    pub fn ops(&self) -> impl Iterator<Item = (GraphSlot, &DataOp, &Before)> {
        self.entries
            .iter()
            .enumerate()
            .flat_map(move |(index, change)| {
                let (data, rows) = match change {
                    Change::Data {
                        graph, op, before, ..
                    } => (Some((*graph, op, before)), &[][..]),
                    Change::Bulk(_) => (None, self.rows_of(index)),
                };
                let graph = change.graph();
                data.into_iter()
                    .chain(rows.iter().map(move |row| (graph, row, &ABSENT)))
            })
    }

    /// Checks that `graph` is a slot of this set, of `model`; `what` names
    /// the change in the error.
    fn check_graph(
        &self,
        graph: GraphSlot,
        model: DataModel,
        what: impl FnOnce() -> String,
    ) -> Result<()> {
        match self.graph(graph) {
            Some(known) if known.model == model => Ok(()),
            Some(known) => Err(Error::Internal(format!(
                "{}, of {model:?} data, in the {:?} graph {:?}",
                what(),
                known.model,
                known.key
            ))),
            None => Err(Error::Internal(format!(
                "{} in graph slot {}, of a change set with {} graphs",
                what(),
                graph.index(),
                self.graphs.len()
            ))),
        }
    }

    /// The number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the set has no entries: a commit then writes no log group.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entries, in recorded order.
    #[must_use]
    pub fn entries(&self) -> &[Change] {
        &self.entries
    }

    /// The entries of one graph, in recorded order (reverse it to undo).
    pub fn in_graph(&self, slot: GraphSlot) -> impl DoubleEndedIterator<Item = &Change> {
        self.entries
            .iter()
            .filter(move |change| change.graph() == slot)
    }

    /// The position after the last entry, for a savepoint.
    #[must_use]
    pub fn mark(&self) -> ChangeMark {
        ChangeMark(self.entries.len())
    }

    /// The entries recorded after `mark`, in recorded order: what
    /// [`split_off`](Self::split_off) would remove, left in place (a
    /// rollback to a savepoint checks them before it undoes anything).
    #[must_use]
    pub fn after(&self, mark: ChangeMark) -> &[Change] {
        self.entries.get(mark.0..).unwrap_or_default()
    }

    /// Removes the entries recorded after `mark` and returns them in
    /// recorded order, for the caller to undo in reverse. A mark at or past
    /// the end (one taken after entries a rollback already removed) returns
    /// nothing.
    pub fn split_off(&mut self, mark: ChangeMark) -> Vec<Change> {
        if mark.0 >= self.entries.len() {
            return Vec::new();
        }
        let tail = self.entries.split_off(mark.0);
        for change in &tail {
            self.held.change(change, Count::Remove);
        }
        let kept = self.bulk_rows.partition_point(|(index, _)| *index < mark.0);
        for (_, rows) in self.bulk_rows.drain(kept..) {
            for row in &rows {
                self.held.row(row, Count::Remove);
            }
        }
        tail
    }

    /// An estimate of the memory the set holds, in bytes: each entry's own
    /// size and what its op and before-image hold on the heap (lists,
    /// values, names, triples), and the graph table.
    ///
    /// A payload of at least 256 bytes that several entries share (one
    /// string, list or vector set on many nodes) counts once, as long as an
    /// entry holds it; a smaller one counts for each entry that holds it.
    /// Spare capacity of the set's own vectors does not count, so the
    /// estimate depends on the entries only: after [`split_off`](Self::split_off)
    /// it is what it was at the mark, when no graph got a slot meanwhile.
    #[must_use]
    pub fn approx_bytes(&self) -> usize {
        // A graph is held twice, in the table and as the key of its slot,
        // sharing its key's bytes.
        let graphs = self
            .graphs
            .iter()
            .map(|graph| {
                2 * size_of::<GraphRef>()
                    + size_of::<GraphSlot>()
                    + graph.key.as_ref().map_or(0, |key| key.len())
            })
            .fold(0, usize::saturating_add);
        let entries = self.entries.len().saturating_mul(size_of::<Change>());
        let shared = self
            .held
            .shared
            .len()
            .saturating_mul(size_of::<(usize, (usize, usize))>());
        size_of::<Self>()
            .saturating_add(graphs)
            .saturating_add(entries)
            .saturating_add(shared)
            .saturating_add(self.held.bytes)
    }

    /// The log records of the set's writes, in the order they were applied
    /// (see [`ops`](Self::ops)), each naming its graph by storage key: the
    /// op of each data entry and the rows kept with a bulk range. A range
    /// whose rows were logged as they were applied has none. Before-images
    /// and pending versions are never logged.
    pub fn log_records(&self) -> impl Iterator<Item = LogRecordRef<'_>> {
        self.ops().map(|(graph, op, _)| LogRecordRef::Data {
            // `push` and `push_bulk` took only slots of this set, and slots
            // stay.
            graph: self.graphs[graph.index()].key.as_deref(),
            op,
        })
    }
}

/// The smallest payload [`ChangeSet::approx_bytes`] counts once however many
/// entries share it.
const SHARED_FROM_BYTES: usize = 256;

/// What a set's entries hold on the heap (see [`ChangeSet::approx_bytes`]).
#[derive(Debug, Default)]
struct Held {
    /// The bytes counted.
    bytes: usize,
    /// The payloads of at least [`SHARED_FROM_BYTES`], by address: how many
    /// entries hold each, and its bytes, counted once.
    shared: GrafeoMap<usize, (usize, usize)>,
}

/// Whether an entry joins or leaves the set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Count {
    Add,
    Remove,
}

impl Held {
    /// Counts what `change` holds, as it joins or leaves the set. An entry
    /// is counted the same way both times, so what leaves is what joined.
    fn change(&mut self, change: &Change, count: Count) {
        match change {
            Change::Data { op, before, .. } => {
                self.op(op, count);
                self.before(before, count);
            }
            Change::Bulk(_) => {}
        }
    }

    /// Counts a row kept with a bulk range: its op, and its place in the
    /// range's list of rows.
    fn row(&mut self, row: &DataOp, count: Count) {
        self.own(size_of::<DataOp>(), count);
        self.op(row, count);
    }

    fn op(&mut self, op: &DataOp, count: Count) {
        match op {
            DataOp::CreateNode {
                labels, properties, ..
            } => {
                self.labels(labels, count);
                self.properties(properties, count);
            }
            DataOp::CreateEdge {
                edge_type,
                properties,
                ..
            } => {
                self.name(edge_type, count);
                self.properties(properties, count);
            }
            DataOp::SetNodeProperty { key, value, .. }
            | DataOp::SetEdgeProperty { key, value, .. } => {
                self.name(key.as_str(), count);
                self.value(value, count);
            }
            DataOp::RemoveNodeProperty { key, .. } | DataOp::RemoveEdgeProperty { key, .. } => {
                self.name(key.as_str(), count);
            }
            DataOp::AddNodeLabel { label, .. } | DataOp::RemoveNodeLabel { label, .. } => {
                self.name(label, count);
            }
            DataOp::InsertTriple { triple } | DataOp::DeleteTriple { triple } => {
                self.own(size_of::<TripleRecord>(), count);
                for term in [&triple.subject, &triple.predicate, &triple.object] {
                    self.own(term_bytes(term), count);
                }
            }
            DataOp::DeleteNode { .. } | DataOp::DeleteEdge { .. } => {}
        }
    }

    fn before(&mut self, before: &Before, count: Count) {
        match before {
            Before::Absent | Before::Value(None) => {}
            Before::Value(Some(value)) => self.value(value, count),
            Before::Labels(labels) => self.labels(labels, count),
            Before::Node(image) => {
                self.own(size_of::<NodeImage>(), count);
                self.labels(&image.labels, count);
                self.properties(&image.properties, count);
            }
            Before::Edge(image) => {
                self.own(size_of::<EdgeImage>(), count);
                self.name(&image.edge_type, count);
                self.properties(&image.properties, count);
            }
        }
    }

    fn labels(&mut self, labels: &Labels, count: Count) {
        if labels.spilled() {
            self.own(labels.capacity().saturating_mul(size_of::<ArcStr>()), count);
        }
        for label in labels {
            self.name(label, count);
        }
    }

    fn properties(&mut self, properties: &Properties, count: Count) {
        self.own(
            properties
                .capacity()
                .saturating_mul(size_of::<(PropertyKey, Value)>()),
            count,
        );
        for (key, value) in properties {
            self.name(key.as_str(), count);
            self.value(value, count);
        }
    }

    /// A name: a shared string, by the address of its bytes.
    fn name(&mut self, name: &str, count: Count) {
        self.payload(name.as_ptr().addr(), || name.len(), count);
    }

    fn value(&mut self, value: &Value, count: Count) {
        match payload_address(value) {
            Some(address) => self.payload(address, || value.estimated_size_bytes(), count),
            None => self.own(value.estimated_size_bytes(), count),
        }
    }

    /// Bytes the entry holds alone.
    fn own(&mut self, bytes: usize, count: Count) {
        self.bytes = match count {
            Count::Add => self.bytes.saturating_add(bytes),
            Count::Remove => self.bytes.saturating_sub(bytes),
        };
    }

    /// A shared payload at `address`, of the bytes `measure` gives: counted
    /// once while any entry holds it when it is large, else per entry. A
    /// payload's bytes never change while an entry holds it (it is shared,
    /// so immutable), and its address stays its own, so an entry that
    /// leaves finds what it joined.
    fn payload(&mut self, address: usize, measure: impl FnOnce() -> usize, count: Count) {
        if let Some((holders, bytes)) = self.shared.get_mut(&address) {
            match count {
                Count::Add => *holders += 1,
                Count::Remove => {
                    *holders -= 1;
                    if *holders == 0 {
                        let bytes = *bytes;
                        self.shared.remove(&address);
                        self.own(bytes, Count::Remove);
                    }
                }
            }
            return;
        }
        let bytes = measure();
        if bytes >= SHARED_FROM_BYTES && count == Count::Add {
            self.shared.insert(address, (1, bytes));
        }
        self.own(bytes, count);
    }
}

/// The address of the payload a value shares through its `Arc`, or `None`
/// for a value with no payload or with two (a path, an on-counter), which
/// counts for each entry that holds it.
fn payload_address(value: &Value) -> Option<usize> {
    match value {
        Value::String(text) => Some(text.as_ptr().addr()),
        Value::Bytes(bytes) => Some(bytes.as_ptr().addr()),
        Value::List(items) => Some(items.as_ptr().addr()),
        Value::Vector(items) => Some(items.as_ptr().addr()),
        Value::Map(map) => Some(Arc::as_ptr(map).addr()),
        Value::GCounter(counts) => Some(Arc::as_ptr(counts).addr()),
        Value::Path { .. }
        | Value::OnCounter { .. }
        | Value::Null
        | Value::Bool(_)
        | Value::Int64(_)
        | Value::Float64(_)
        | Value::Timestamp(_)
        | Value::Date(_)
        | Value::Time(_)
        | Value::Duration(_)
        | Value::ZonedDatetime(_) => None,
    }
}

/// The bytes an RDF term's strings hold.
fn term_bytes(term: &TermRecord) -> usize {
    match term {
        TermRecord::Iri(text) | TermRecord::Blank(text) => text.capacity(),
        TermRecord::Literal {
            value,
            datatype,
            language,
        } => value.capacity() + datatype.capacity() + language.as_ref().map_or(0, String::capacity),
    }
}

#[cfg(test)]
mod tests {
    use super::PendingVersion::{Created, Replaced};
    use super::*;
    use crate::storage::catalog_record::{IndexKeyRecord, SchemaRecord};
    use crate::storage::log_record::{LogRecord, read_log_records};

    fn lpg(key: Option<&str>) -> GraphRef {
        GraphRef {
            model: DataModel::Lpg,
            key: key.map(ArcStr::from),
        }
    }

    fn rdf(key: Option<&str>) -> GraphRef {
        GraphRef {
            model: DataModel::Rdf,
            key: key.map(ArcStr::from),
        }
    }

    fn labels(names: &[&str]) -> Labels {
        names.iter().map(|&name| ArcStr::from(name)).collect()
    }

    fn iri(value: &str) -> TermRecord {
        TermRecord::Iri(value.to_string())
    }

    fn triple(subject: TermRecord, predicate: TermRecord, object: TermRecord) -> Box<TripleRecord> {
        Box::new(TripleRecord {
            subject,
            predicate,
            object,
        })
    }

    fn node_image(names: &[&str], properties: Properties) -> Before {
        Before::Node(Box::new(NodeImage {
            labels: labels(names),
            properties,
        }))
    }

    fn edge_image(src: u64, dst: u64, edge_type: &str, properties: Properties) -> Before {
        Before::Edge(Box::new(EdgeImage {
            src: NodeId::new(src),
            dst: NodeId::new(dst),
            edge_type: edge_type.into(),
            properties,
        }))
    }

    fn create_node(id: u64, names: &[&str], properties: Properties) -> DataOp {
        DataOp::CreateNode {
            id: NodeId::new(id),
            labels: labels(names),
            properties,
        }
    }

    fn set_node_property(id: u64, key: &str, value: Value) -> DataOp {
        DataOp::SetNodeProperty {
            id: NodeId::new(id),
            key: key.into(),
            value,
        }
    }

    fn read_all(bytes: &[u8]) -> Vec<LogRecord> {
        let mut records = Vec::new();
        read_log_records(bytes, &mut |record| {
            records.push(record);
            Ok(())
        })
        .unwrap();
        records
    }

    fn encoded(set: &ChangeSet) -> Vec<Vec<u8>> {
        set.log_records()
            .map(|record| {
                let mut bytes = Vec::new();
                record.encode_framed(&mut bytes).unwrap();
                bytes
            })
            .collect()
    }

    fn kinds(changes: &[Change]) -> Vec<u8> {
        changes
            .iter()
            .map(|change| match change {
                Change::Data { op, .. } => op.kind(),
                Change::Bulk(_) => 0,
            })
            .collect()
    }

    fn versions(changes: &[Change]) -> Vec<PendingVersion> {
        changes
            .iter()
            .map(|change| match change {
                Change::Data { version, .. } => *version,
                Change::Bulk(_) => panic!("{change:?}"),
            })
            .collect()
    }

    /// The data entries of the log record layouts the first log record
    /// encoder pinned (one of every data kind, in kind order), and the
    /// standalone ops of the same records. A write to what an earlier entry
    /// wrote replaced that entry's pending version.
    fn pinned_entries() -> (ChangeSet, Vec<StandaloneOp>) {
        let mut set = ChangeSet::new();
        let default = set.slot(lpg(None)).unwrap();
        let trips = set.slot(lpg(Some("trips"))).unwrap();
        let rdf_default = set.slot(rdf(None)).unwrap();
        let rdf_named = set.slot(rdf(Some("ex:g"))).unwrap();
        let entries = [
            (
                default,
                create_node(3, &["Person"], vec![("name".into(), Value::from("Alix"))]),
                Before::Absent,
                Created,
            ),
            (
                trips,
                DataOp::DeleteNode {
                    id: NodeId::new(19),
                },
                node_image(&["City"], Vec::new()),
                Created,
            ),
            (
                default,
                DataOp::CreateEdge {
                    id: EdgeId::new(88),
                    src: NodeId::new(3),
                    dst: NodeId::new(19),
                    edge_type: "KNOWS".into(),
                    properties: vec![("since".into(), Value::Int64(1988))],
                },
                Before::Absent,
                Created,
            ),
            (
                default,
                DataOp::DeleteEdge {
                    id: EdgeId::new(88),
                },
                edge_image(3, 19, "KNOWS", vec![("since".into(), Value::Int64(1988))]),
                Replaced,
            ),
            (
                default,
                set_node_property(3, "city", Value::from("Paris")),
                Before::Value(None),
                Created,
            ),
            (
                default,
                DataOp::RemoveNodeProperty {
                    id: NodeId::new(3),
                    key: "city".into(),
                },
                Before::Value(Some(Value::from("Paris"))),
                Replaced,
            ),
            (
                trips,
                DataOp::SetEdgeProperty {
                    id: EdgeId::new(88),
                    key: "km".into(),
                    value: Value::Int64(319),
                },
                Before::Value(Some(Value::Int64(88))),
                Created,
            ),
            (
                trips,
                DataOp::RemoveEdgeProperty {
                    id: EdgeId::new(88),
                    key: "km".into(),
                },
                Before::Value(Some(Value::Int64(319))),
                Replaced,
            ),
            (
                default,
                DataOp::AddNodeLabel {
                    id: NodeId::new(19),
                    label: "City".into(),
                },
                Before::Labels(labels(&["Place"])),
                Created,
            ),
            (
                default,
                DataOp::RemoveNodeLabel {
                    id: NodeId::new(19),
                    label: "City".into(),
                },
                Before::Labels(labels(&["Place", "City"])),
                Replaced,
            ),
            (
                rdf_default,
                DataOp::InsertTriple {
                    triple: triple(
                        iri("ex:Gus"),
                        iri("ex:knows"),
                        TermRecord::Blank("b3".into()),
                    ),
                },
                Before::Absent,
                Created,
            ),
            (
                rdf_named,
                DataOp::DeleteTriple {
                    triple: triple(
                        TermRecord::Blank("b3".into()),
                        iri("ex:name"),
                        TermRecord::Literal {
                            value: "Mia".into(),
                            datatype: "xsd:string".into(),
                            language: Some("nl".into()),
                        },
                    ),
                },
                Before::Absent,
                Created,
            ),
        ];
        for (graph, op, before, version) in entries {
            set.push(graph, op, before, version).unwrap();
        }
        let standalone = vec![
            StandaloneOp::CreateGraph {
                name: "trips".into(),
            },
            StandaloneOp::DropGraph {
                name: "trips".into(),
            },
            StandaloneOp::PutCatalog(CatalogRecord::Schema(SchemaRecord {
                name: "travel".into(),
            })),
            StandaloneOp::DropCatalog(CatalogKey::Schema("travel".into())),
            StandaloneOp::RdfGraph(RdfGraphOp::Create {
                name: "ex:g".into(),
            }),
            StandaloneOp::RdfGraph(RdfGraphOp::Drop {
                target: RdfGraphTarget::Named("ex:g".into()),
            }),
            StandaloneOp::RdfGraph(RdfGraphOp::Clear {
                target: RdfGraphTarget::All,
            }),
            StandaloneOp::RdfGraph(RdfGraphOp::Copy {
                source: None,
                target: Some("ex:b".into()),
            }),
            StandaloneOp::RdfGraph(RdfGraphOp::Move {
                source: Some("ex:b".into()),
                target: Some("ex:g".into()),
            }),
            StandaloneOp::RdfGraph(RdfGraphOp::Add {
                source: Some("ex:g".into()),
                target: None,
            }),
        ];
        (set, standalone)
    }

    /// Every entry's op goes to the log as the record of its kind, naming
    /// its graph, and comes back as the same op in the same graph; every
    /// standalone op too. The before-images stay behind.
    #[test]
    fn every_entry_kind_round_trips_through_its_log_record() {
        let (set, standalone) = pinned_entries();
        let mut bytes = Vec::new();
        for record in set.log_records() {
            record.encode_framed(&mut bytes).unwrap();
        }
        let back = read_all(&bytes);
        assert_eq!(
            back.iter().map(LogRecord::kind).collect::<Vec<_>>(),
            [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 64, 65],
            "one record per entry, in recorded order"
        );
        assert_eq!(back.len(), set.len());
        for (record, change) in back.iter().zip(set.entries()) {
            let Change::Data { graph, op, .. } = change else {
                panic!("a bulk range among the pinned entries");
            };
            let LogRecord::Data {
                op: logged_op,
                graph: logged_graph,
            } = record
            else {
                panic!("{record:?} is no data record");
            };
            assert_eq!(logged_op, op, "the op is the record's payload");
            assert_eq!(
                record.graph().as_ref(),
                set.graph(*graph),
                "the record names the entry's graph: its key, its model by the op"
            );
            assert_eq!(
                logged_graph.as_deref(),
                set.graph(*graph).unwrap().key.as_deref()
            );
        }

        // From the log back into a set: the same records again.
        let mut rebuilt = ChangeSet::new();
        for (record, change) in back.into_iter().zip(set.entries()) {
            let Change::Data {
                before, version, ..
            } = change
            else {
                unreachable!("checked above");
            };
            let graph = rebuilt.slot(record.graph().unwrap()).unwrap();
            let LogRecord::Data { op, .. } = record else {
                unreachable!("checked above");
            };
            rebuilt.push(graph, op, before.clone(), *version).unwrap();
        }
        assert_eq!(encoded(&rebuilt), encoded(&set));

        let mut kinds = Vec::new();
        for op in standalone {
            let record = LogRecord::Standalone(op);
            let mut bytes = Vec::new();
            record.encode_framed(&mut bytes).unwrap();
            assert_eq!(
                read_all(&bytes),
                std::slice::from_ref(&record),
                "{record:?}"
            );
            assert_eq!(record.graph(), None, "a standalone op is in no graph");
            kinds.push(record.kind());
        }
        assert_eq!(kinds, [32, 33, 40, 41, 66, 67, 68, 69, 70, 71]);
    }

    /// The log record of every entry kind has the bytes the first log record
    /// encoder pinned for its kind, before the change set's types became the
    /// payloads (`log_record`'s `the_record_layouts_are_pinned`, whose
    /// records these entries and ops hold).
    #[test]
    fn every_entry_kind_encodes_to_the_bytes_pinned_before_the_change_set() {
        #[rustfmt::skip]
        const DATA: &[&[u8]] = &[
            &[16, 1, 26, 0, 0, 0, 0, 3, 1, 6, 80, 101, 114, 115, 111, 110, 1, 4, 110, 97, 109, 101,
              9, 4, 4, 0, 0, 0, 65, 108, 105, 120],
            &[17, 1, 8, 0, 0, 0, 1, 5, 116, 114, 105, 112, 115, 19],
            &[18, 1, 27, 0, 0, 0, 0, 88, 3, 19, 5, 75, 78, 79, 87, 83, 1, 5, 115, 105, 110, 99, 101,
              9, 2, 196, 7, 0, 0, 0, 0, 0, 0],
            &[19, 1, 2, 0, 0, 0, 0, 88],
            &[20, 1, 18, 0, 0, 0, 0, 3, 4, 99, 105, 116, 121, 10, 4, 5, 0, 0, 0, 80, 97, 114, 105,
              115],
            &[21, 1, 7, 0, 0, 0, 0, 3, 4, 99, 105, 116, 121],
            &[22, 1, 21, 0, 0, 0, 1, 5, 116, 114, 105, 112, 115, 88, 2, 107, 109, 9, 2, 63, 1, 0, 0,
              0, 0, 0, 0],
            &[23, 1, 11, 0, 0, 0, 1, 5, 116, 114, 105, 112, 115, 88, 2, 107, 109],
            &[24, 1, 7, 0, 0, 0, 0, 19, 4, 67, 105, 116, 121],
            &[25, 1, 7, 0, 0, 0, 0, 19, 4, 67, 105, 116, 121],
            &[64, 1, 23, 0, 0, 0, 0, 0, 6, 101, 120, 58, 71, 117, 115, 0, 8, 101, 120, 58, 107, 110,
              111, 119, 115, 1, 2, 98, 51],
            &[65, 1, 39, 0, 0, 0, 1, 4, 101, 120, 58, 103, 1, 2, 98, 51, 0, 7, 101, 120, 58, 110,
              97, 109, 101, 2, 3, 77, 105, 97, 10, 120, 115, 100, 58, 115, 116, 114, 105, 110, 103,
              1, 2, 110, 108],
        ];
        #[rustfmt::skip]
        const STANDALONE: &[&[u8]] = &[
            &[32, 1, 6, 0, 0, 0, 5, 116, 114, 105, 112, 115],
            &[33, 1, 6, 0, 0, 0, 5, 116, 114, 105, 112, 115],
            &[40, 1, 8, 0, 0, 0, 1, 6, 116, 114, 97, 118, 101, 108],
            &[41, 1, 8, 0, 0, 0, 1, 6, 116, 114, 97, 118, 101, 108],
            &[66, 1, 5, 0, 0, 0, 4, 101, 120, 58, 103],
            &[67, 1, 6, 0, 0, 0, 1, 4, 101, 120, 58, 103],
            &[68, 1, 1, 0, 0, 0, 3],
            &[69, 1, 7, 0, 0, 0, 0, 1, 4, 101, 120, 58, 98],
            &[70, 1, 12, 0, 0, 0, 1, 4, 101, 120, 58, 98, 1, 4, 101, 120, 58, 103],
            &[71, 1, 7, 0, 0, 0, 1, 4, 101, 120, 58, 103, 0],
        ];
        let (set, standalone) = pinned_entries();
        assert_eq!(encoded(&set), DATA);
        let standalone: Vec<Vec<u8>> = standalone
            .into_iter()
            .map(|op| {
                let mut bytes = Vec::new();
                LogRecord::Standalone(op).encode_framed(&mut bytes).unwrap();
                bytes
            })
            .collect();
        assert_eq!(standalone, STANDALONE);
    }

    /// A savepoint is a mark; rolling back to it hands back exactly the
    /// entries recorded after it, in recorded order (undo walks them in
    /// reverse), and leaves the earlier ones. Nested savepoints unwind in
    /// turn, and a mark that a rollback already passed hands back nothing.
    #[test]
    fn a_savepoint_rollback_returns_the_tail_in_recorded_order() {
        let mut set = ChangeSet::new();
        let default = set.slot(lpg(None)).unwrap();
        let trips = set.slot(lpg(Some("trips"))).unwrap();
        set.push(
            default,
            create_node(3, &["Person"], Vec::new()),
            Before::Absent,
            Created,
        )
        .unwrap();
        set.push(
            default,
            set_node_property(3, "name", Value::from("Alix")),
            Before::Value(None),
            Created,
        )
        .unwrap();
        let statement = set.mark();
        set.push(
            trips,
            create_node(19, &["City"], Vec::new()),
            Before::Absent,
            Created,
        )
        .unwrap();
        set.push(
            default,
            DataOp::AddNodeLabel {
                id: NodeId::new(3),
                label: "Traveler".into(),
            },
            Before::Labels(labels(&["Person"])),
            Replaced,
        )
        .unwrap();
        let inner = set.mark();
        set.push(
            default,
            set_node_property(3, "name", Value::from("Gus")),
            Before::Value(Some(Value::from("Alix"))),
            Replaced,
        )
        .unwrap();
        assert_eq!(set.len(), 5);
        assert_eq!(
            kinds(set.after(statement)),
            [16, 24, 20],
            "what a split would take, left in place"
        );
        assert_eq!(set.len(), 5);

        let tail = set.split_off(inner);
        assert_eq!(kinds(&tail), [20], "only what came after the inner mark");
        assert!(
            set.after(inner).is_empty(),
            "a mark past the end sees nothing"
        );
        assert_eq!(
            tail[0],
            Change::Data {
                graph: default,
                op: set_node_property(3, "name", Value::from("Gus")),
                before: Before::Value(Some(Value::from("Alix"))),
                version: Replaced,
            },
            "the entry comes back whole, its before-image and pending version with it"
        );
        let tail = set.split_off(statement);
        assert_eq!(
            kinds(&tail),
            [16, 24],
            "in recorded order, across both graphs"
        );
        assert_eq!(
            versions(&tail),
            [Created, Replaced],
            "each entry keeps its own pending version"
        );
        assert_eq!(tail[0].graph(), trips);
        assert!(
            set.split_off(inner).is_empty(),
            "a mark past the end hands back nothing"
        );
        assert_eq!(kinds(set.entries()), [16, 20], "the earlier entries stay");
        assert_eq!(
            set.mark(),
            statement,
            "the set is back at the statement's mark"
        );

        // Writing on after a rollback records after the mark again.
        set.push(
            trips,
            create_node(88, &["City"], Vec::new()),
            Before::Absent,
            Created,
        )
        .unwrap();
        assert_eq!(kinds(&set.split_off(statement)), [16]);
        assert_eq!(set.graphs(), [lpg(None), lpg(Some("trips"))], "slots stay");
    }

    /// The estimate grows with what entries hold, counts a large payload
    /// shared by several entries once (and a copy of it again), counts a
    /// delete's before-image, and comes back to its value at a mark when a
    /// rollback removes the entries after it.
    #[test]
    fn approx_bytes_counts_shared_values_once() {
        const NOTES_BYTES: usize = 65_540;
        let mut set = ChangeSet::new();
        let empty = set.approx_bytes();
        let default = set.slot(lpg(None)).unwrap();
        set.push(
            default,
            create_node(3, &["Person"], Vec::new()),
            Before::Absent,
            Created,
        )
        .unwrap();
        let first = set.approx_bytes();
        assert!(first > empty, "an entry takes room: {empty} then {first}");
        let mark = set.mark();

        let notes = Value::from("Amsterdam ".repeat(NOTES_BYTES / 10).as_str());
        for id in [3, 19, 88] {
            set.push(
                default,
                set_node_property(id, "notes", notes.clone()),
                Before::Value(None),
                Created,
            )
            .unwrap();
        }
        let shared = set.approx_bytes();
        assert!(
            shared - first >= NOTES_BYTES,
            "the notes count: {first} then {shared}"
        );
        assert!(
            shared - first < 2 * NOTES_BYTES,
            "three entries hold one string, counted once: {first} then {shared}"
        );

        // An equal string in an allocation of its own counts again; the old
        // value it replaced is the shared one, counted already.
        let copy = Value::from("Amsterdam ".repeat(NOTES_BYTES / 10).as_str());
        set.push(
            default,
            set_node_property(3, "notes", copy),
            Before::Value(Some(notes.clone())),
            Replaced,
        )
        .unwrap();
        let copied = set.approx_bytes();
        assert!(
            copied - shared >= NOTES_BYTES && copied - shared < 2 * NOTES_BYTES,
            "a copy counts once more: {shared} then {copied}"
        );

        // A delete keeps the node's properties in its before-image.
        let vector = Value::Vector(Arc::from(vec![0.19f32; 4096]));
        set.push(
            default,
            DataOp::DeleteNode {
                id: NodeId::new(19),
            },
            node_image(&["Person"], vec![("embedding".into(), vector)]),
            Created,
        )
        .unwrap();
        let deleted = set.approx_bytes();
        assert!(
            deleted - copied >= 4096 * 4,
            "the image's vector counts: {copied} then {deleted}"
        );

        let tail = set.split_off(mark);
        assert_eq!(tail.len(), 5);
        assert_eq!(
            set.approx_bytes(),
            first,
            "the removed entries, and the payloads only they held, no longer count"
        );
        drop(tail);
        assert_eq!(set.split_off(ChangeMark(0)).len(), 1);
        let mut fresh = ChangeSet::new();
        fresh.slot(lpg(None)).unwrap();
        assert_eq!(
            set.approx_bytes(),
            fresh.approx_bytes(),
            "an empty set with one graph"
        );
    }

    /// An entry stays small, so a transaction of a million changes holds
    /// about a hundred megabytes plus its values. The pending version takes
    /// a byte of the padding after the graph slot.
    #[test]
    fn an_entry_takes_at_most_128_bytes_inline() {
        assert!(
            size_of::<Change>() <= 128,
            "an entry takes {} bytes",
            size_of::<Change>()
        );
    }

    /// Whether a write created its pending version is kept for undo and
    /// never logged: the same writes with the other answers encode to the
    /// same bytes, and hold the same memory.
    #[test]
    fn the_pending_version_stays_in_memory_and_never_reaches_the_log() {
        let (set, _) = pinned_entries();
        let mut other = ChangeSet::new();
        for graph in set.graphs() {
            other.slot(graph.clone()).unwrap();
        }
        for change in set.entries() {
            let Change::Data {
                graph,
                op,
                before,
                version,
            } = change
            else {
                panic!("a bulk range among the pinned entries");
            };
            let version = match (before, version) {
                (Before::Absent, _) => Created,
                (_, Created) => Replaced,
                (_, Replaced) => Created,
            };
            other
                .push(*graph, op.clone(), before.clone(), version)
                .unwrap();
        }
        assert_ne!(
            versions(other.entries()),
            versions(set.entries()),
            "the sets differ in their pending versions"
        );
        assert_eq!(encoded(&other), encoded(&set), "the log never holds them");
        assert_eq!(other.approx_bytes(), set.approx_bytes());
    }

    /// One property of one node in a store that keeps one pending version
    /// per transaction and changes it in place on the transaction's next
    /// write, as the row-group store's version info will.
    struct OneProperty {
        committed: Option<Value>,
        /// The transaction's pending version, once it set the property: the
        /// value it set.
        pending: Option<Value>,
    }

    impl OneProperty {
        /// Sets the property as the transaction, and records the write.
        fn set(&mut self, set: &mut ChangeSet, graph: GraphSlot, city: &str) {
            let value = Value::from(city);
            let (before, version) = match self.pending.replace(value.clone()) {
                None => (self.committed.clone(), Created),
                Some(own) => (Some(own), Replaced),
            };
            set.push(
                graph,
                set_node_property(3, "city", value),
                Before::Value(before),
                version,
            )
            .unwrap();
        }

        /// Undoes one entry: drops the pending version the entry created,
        /// or writes its before-image back into the one it replaced.
        fn undo(&mut self, change: &Change) {
            let Change::Data {
                before: Before::Value(before),
                version,
                ..
            } = change
            else {
                panic!("{change:?}");
            };
            self.pending = match version {
                Created => None,
                Replaced => before.clone(),
            };
        }

        /// What the transaction reads.
        fn read(&self) -> Option<&Value> {
            self.pending.as_ref().or(self.committed.as_ref())
        }

        fn commit(&mut self) {
            if let Some(value) = self.pending.take() {
                self.committed = Some(value);
            }
        }
    }

    /// A savepoint rollback restores what the transaction had at the
    /// savepoint, its own pending value included, because each entry says
    /// whether it created the pending version or replaced the
    /// transaction's own: the before-image alone cannot tell.
    #[test]
    fn undoing_by_pending_version_restores_the_value_at_the_savepoint() {
        let mut set = ChangeSet::new();
        let default = set.slot(lpg(None)).unwrap();
        let mut store = OneProperty {
            committed: Some(Value::from("Amsterdam")),
            pending: None,
        };
        store.set(&mut set, default, "Berlin");
        let savepoint = set.mark();
        store.set(&mut set, default, "Paris");
        store.set(&mut set, default, "Prague");

        let tail = set.split_off(savepoint);
        for change in tail.iter().rev() {
            store.undo(change);
        }
        assert_eq!(
            store.read(),
            Some(&Value::from("Berlin")),
            "the transaction's own value at the savepoint, not the committed one"
        );
        assert_eq!(versions(&tail), [Replaced, Replaced], "kept by the split");
        assert_eq!(versions(set.entries()), [Created]);
        store.commit();
        assert_eq!(
            store.committed,
            Some(Value::from("Berlin")),
            "another session reads it after the commit"
        );

        // A full rollback of the next transaction drops its pending version.
        let mut set = ChangeSet::new();
        let default = set.slot(lpg(None)).unwrap();
        store.set(&mut set, default, "Paris");
        store.set(&mut set, default, "Prague");
        for change in set.split_off(ChangeMark(0)).iter().rev() {
            store.undo(change);
        }
        assert_eq!(store.pending, None, "nothing pending");
        assert_eq!(store.read(), Some(&Value::from("Berlin")));
    }

    /// A bulk write's reserved ids are one entry per range, whatever its
    /// size, in a labeled property graph; undone in reverse (edges before
    /// nodes); never logged by the set, since the rows were logged as they
    /// were applied.
    #[test]
    fn a_bulk_range_is_one_entry_undone_by_range_and_logged_by_its_rows() {
        let mut set = ChangeSet::new();
        let prague = set.slot(lpg(Some("Prague"))).unwrap();
        let before = set.approx_bytes();
        set.push_bulk(BulkRange {
            graph: prague,
            table: Table::Nodes,
            ids: 3..88,
        })
        .unwrap();
        let one = set.approx_bytes();
        set.push_bulk(BulkRange {
            graph: prague,
            table: Table::Edges,
            ids: 19..19 + 1_000_000,
        })
        .unwrap();
        assert_eq!(set.len(), 2, "one entry per range");
        assert_eq!(
            set.approx_bytes() - one,
            one - before,
            "a million ids cost what 85 do"
        );
        assert_eq!(
            set.log_records().count(),
            0,
            "the rows were logged as applied"
        );
        let undo: Vec<Table> = set
            .in_graph(prague)
            .rev()
            .map(|change| match change {
                Change::Bulk(range) => range.table,
                Change::Data { .. } => panic!("{change:?}"),
            })
            .collect();
        assert_eq!(undo, [Table::Edges, Table::Nodes], "edges undone first");

        // An empty range adds nothing; a range outside the set's graphs or in
        // an RDF graph is refused.
        set.push_bulk(BulkRange {
            graph: prague,
            table: Table::Nodes,
            ids: 88..88,
        })
        .unwrap();
        assert_eq!(set.len(), 2);
        let triples = set.slot(rdf(Some("Prague"))).unwrap();
        for graph in [triples, GraphSlot(19)] {
            let error = set
                .push_bulk(BulkRange {
                    graph,
                    table: Table::Nodes,
                    ids: 3..19,
                })
                .unwrap_err();
            assert!(matches!(error, Error::Internal(_)), "{error:?}");
        }
        assert_eq!(set.len(), 2, "a refused range records nothing");
    }

    /// The rows a bulk write inside a transaction applied (a batch call)
    /// stay with its range for the commit: the log and change data capture
    /// read them at the range's place among the entries, each a create that
    /// replaced nothing, while the range stays one entry for undo and stamp.
    /// A rollback to a savepoint before the range drops them with it. Rows
    /// that do not belong to the range are refused, and nothing is kept.
    #[test]
    fn a_bulk_range_keeps_the_rows_it_is_given_for_the_commit() {
        let mut set = ChangeSet::new();
        let default = set.slot(lpg(None)).unwrap();
        let trips = set.slot(lpg(Some("trips"))).unwrap();
        set.push(
            default,
            create_node(1, &["Person"], vec![("name".into(), Value::from("Alix"))]),
            Before::Absent,
            Created,
        )
        .unwrap();
        let mark = set.mark();
        let empty = set.approx_bytes();
        set.push_bulk(BulkRange {
            graph: default,
            table: Table::Nodes,
            ids: 3..6,
        })
        .unwrap();
        let range_only = set.approx_bytes();
        let city = |id: u64, name: &str| {
            create_node(id, &["City"], vec![("name".into(), Value::from(name))])
        };
        // Two chunks for one range.
        set.push_bulk_rows(default, vec![city(3, "Amsterdam"), city(4, "Berlin")])
            .unwrap();
        set.push_bulk_rows(default, vec![city(5, "Paris")]).unwrap();
        set.push(
            default,
            set_node_property(4, "visits", Value::Int64(19)),
            Before::Value(None),
            Created,
        )
        .unwrap();
        assert_eq!(set.len(), 3, "the rows are no entries of their own");
        assert!(
            set.approx_bytes() > range_only,
            "the kept rows count in the set's memory"
        );

        let ops: Vec<(GraphSlot, Option<Entity>, bool)> = set
            .ops()
            .map(|(graph, op, before)| (graph, op.entity(), *before == Before::Absent))
            .collect();
        let node = |id: u64| Some(Entity::Node(NodeId::new(id)));
        assert_eq!(
            ops,
            [
                (default, node(1), true),
                (default, node(3), true),
                (default, node(4), true),
                (default, node(5), true),
                (default, node(4), false),
            ],
            "every op in the order it was applied, the rows at their range's place"
        );
        let logged: Vec<Option<Entity>> = read_all(&encoded(&set).concat())
            .iter()
            .map(|record| match record {
                LogRecord::Data { graph: None, op } => op.entity(),
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(logged, [node(1), node(3), node(4), node(5), node(4)]);

        // Refused: rows outside the range or of the other table, rows of
        // another graph, and rows when the last entry is not a range.
        let edge = DataOp::CreateEdge {
            id: EdgeId::new(4),
            src: NodeId::new(3),
            dst: NodeId::new(5),
            edge_type: "ROUTE".into(),
            properties: Vec::new(),
        };
        let before_refusals = set.approx_bytes();
        set.push_bulk(BulkRange {
            graph: default,
            table: Table::Nodes,
            ids: 19..22,
        })
        .unwrap();
        for (graph, rows) in [
            (default, vec![city(19, "Prague"), city(88, "Barcelona")]),
            (default, vec![edge.clone()]),
            (trips, vec![city(19, "Prague")]),
        ] {
            let error = set.push_bulk_rows(graph, rows).unwrap_err();
            assert!(matches!(error, Error::Internal(_)), "{error:?}");
        }
        assert_eq!(set.split_off(set.mark()).len(), 0);
        let last = set.split_off(ChangeMark(3));
        assert_eq!(last.len(), 1);
        assert_eq!(
            set.approx_bytes(),
            before_refusals,
            "a refused row is not kept"
        );
        let error = set
            .push_bulk_rows(default, vec![city(19, "Prague")])
            .unwrap_err();
        assert!(matches!(error, Error::Internal(_)), "{error:?}");

        // A rollback to a savepoint before the range drops its rows.
        let tail = set.split_off(mark);
        assert_eq!(tail.len(), 2);
        assert_eq!(set.approx_bytes(), empty, "the rows left with their range");
        assert_eq!(set.ops().count(), 1);
        assert_eq!(set.log_records().count(), 1);
    }

    /// `DETACH DELETE` records each edge delete before the node delete: the
    /// log replays them in that order, and undo, walking back, restores the
    /// node before the edges that need it.
    #[test]
    fn a_detach_delete_records_its_edges_before_its_node() {
        let mut set = ChangeSet::new();
        let default = set.slot(lpg(None)).unwrap();
        // Alix (3) knows Gus (19) through edge 3, Vincent (88) knows Alix
        // through edge 19.
        set.push(
            default,
            DataOp::DeleteEdge { id: EdgeId::new(3) },
            edge_image(3, 19, "KNOWS", Vec::new()),
            Created,
        )
        .unwrap();
        set.push(
            default,
            DataOp::DeleteEdge {
                id: EdgeId::new(19),
            },
            edge_image(88, 3, "KNOWS", Vec::new()),
            Created,
        )
        .unwrap();
        set.push(
            default,
            DataOp::DeleteNode { id: NodeId::new(3) },
            node_image(&["Person"], vec![("name".into(), Value::from("Alix"))]),
            Created,
        )
        .unwrap();

        let logged: Vec<Option<Entity>> = read_all(&encoded(&set).concat())
            .iter()
            .map(|record| match record {
                LogRecord::Data { op, .. } => op.entity(),
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(
            logged,
            [
                Some(Entity::Edge(EdgeId::new(3))),
                Some(Entity::Edge(EdgeId::new(19))),
                Some(Entity::Node(NodeId::new(3))),
            ],
            "replay deletes the edges, then the node"
        );
        let undone: Vec<Option<Entity>> = set
            .in_graph(default)
            .rev()
            .map(|change| match change {
                Change::Data { op, .. } => op.entity(),
                Change::Bulk(_) => None,
            })
            .collect();
        assert_eq!(
            undone,
            [
                Some(Entity::Node(NodeId::new(3))),
                Some(Entity::Edge(EdgeId::new(19))),
                Some(Entity::Edge(EdgeId::new(3))),
            ],
            "undo restores the node, then its edges"
        );
    }

    /// A push the set refuses records nothing: an unknown slot, an op of
    /// the other model, an invalid id, a before-image of another shape, one
    /// that says the write changed nothing, or a create or a triple that
    /// says it replaced a pending version.
    #[test]
    fn a_refused_push_records_nothing() {
        let mut set = ChangeSet::new();
        let default = set.slot(lpg(None)).unwrap();
        let triples = set.slot(rdf(None)).unwrap();
        set.push(
            default,
            create_node(3, &["Person"], Vec::new()),
            Before::Absent,
            Created,
        )
        .unwrap();
        let bytes = set.approx_bytes();
        let insert = DataOp::InsertTriple {
            triple: triple(iri("ex:Mia"), iri("ex:knows"), iri("ex:Jules")),
        };
        let cases = [
            (
                GraphSlot(2),
                create_node(19, &[], Vec::new()),
                Before::Absent,
            ),
            (triples, create_node(19, &[], Vec::new()), Before::Absent),
            (default, insert.clone(), Before::Absent),
            (
                default,
                create_node(u64::MAX, &["Person"], Vec::new()),
                Before::Absent,
            ),
            (
                default,
                DataOp::CreateEdge {
                    id: EdgeId::new(19),
                    src: NodeId::new(3),
                    dst: NodeId::INVALID,
                    edge_type: "KNOWS".into(),
                    properties: Vec::new(),
                },
                Before::Absent,
            ),
            (
                default,
                DataOp::DeleteEdge {
                    id: EdgeId::INVALID,
                },
                edge_image(3, 19, "KNOWS", Vec::new()),
            ),
            (
                default,
                create_node(19, &[], Vec::new()),
                Before::Value(None),
            ),
            (
                default,
                set_node_property(3, "city", Value::from("Berlin")),
                Before::Absent,
            ),
            (
                default,
                DataOp::RemoveNodeLabel {
                    id: NodeId::new(3),
                    label: "Person".into(),
                },
                Before::Value(None),
            ),
            (
                default,
                DataOp::DeleteNode { id: NodeId::new(3) },
                edge_image(3, 19, "KNOWS", Vec::new()),
            ),
            (
                default,
                DataOp::DeleteEdge { id: EdgeId::new(3) },
                node_image(&["Person"], Vec::new()),
            ),
            (triples, insert.clone(), Before::Labels(labels(&["Person"]))),
            // Writes that changed nothing, which strict replay would refuse:
            // removing a property that was not there, adding a label the node
            // had, removing one it did not have.
            (
                default,
                DataOp::RemoveNodeProperty {
                    id: NodeId::new(3),
                    key: "city".into(),
                },
                Before::Value(None),
            ),
            (
                default,
                DataOp::RemoveEdgeProperty {
                    id: EdgeId::new(19),
                    key: "since".into(),
                },
                Before::Value(None),
            ),
            (
                default,
                DataOp::AddNodeLabel {
                    id: NodeId::new(3),
                    label: "Person".into(),
                },
                Before::Labels(labels(&["Person"])),
            ),
            (
                default,
                DataOp::RemoveNodeLabel {
                    id: NodeId::new(3),
                    label: "Traveler".into(),
                },
                Before::Labels(labels(&["Person"])),
            ),
        ];
        // A create or a triple created what it wrote: it replaced no pending
        // version of the transaction (ids are never reused).
        let replacing = [
            (default, create_node(19, &["City"], Vec::new())),
            (
                default,
                DataOp::CreateEdge {
                    id: EdgeId::new(19),
                    src: NodeId::new(3),
                    dst: NodeId::new(3),
                    edge_type: "KNOWS".into(),
                    properties: Vec::new(),
                },
            ),
            (triples, insert),
        ];
        let cases = cases
            .into_iter()
            .map(|(graph, op, before)| (graph, op, before, Created))
            .chain(
                replacing
                    .into_iter()
                    .map(|(graph, op)| (graph, op, Before::Absent, Replaced)),
            );
        for (graph, op, before, version) in cases {
            let what = format!("{graph:?} {op:?} {before:?} {version:?}");
            let error = set.push(graph, op, before, version).unwrap_err();
            assert!(matches!(error, Error::Internal(_)), "{what}: {error:?}");
        }
        assert_eq!(set.len(), 1, "nothing recorded");
        assert_eq!(set.approx_bytes(), bytes, "nothing counted");

        // The same ops with the before-images they have are taken, and a
        // write that replaced the transaction's own pending version.
        let fitting = [
            (
                DataOp::RemoveNodeProperty {
                    id: NodeId::new(3),
                    key: "city".into(),
                },
                Before::Value(Some(Value::from("Prague"))),
                Created,
            ),
            (
                DataOp::AddNodeLabel {
                    id: NodeId::new(3),
                    label: "Traveler".into(),
                },
                Before::Labels(labels(&["Person"])),
                Replaced,
            ),
            (
                DataOp::RemoveNodeLabel {
                    id: NodeId::new(3),
                    label: "Person".into(),
                },
                Before::Labels(labels(&["Person", "Traveler"])),
                Replaced,
            ),
            (
                DataOp::DeleteNode { id: NodeId::new(3) },
                node_image(&["Traveler"], Vec::new()),
                Replaced,
            ),
        ];
        for (op, before, version) in fitting {
            set.push(default, op, before, version).unwrap();
        }
        assert_eq!(set.len(), 5);
    }

    /// A graph gets one slot, by its model and key; the entries of a graph
    /// are listed in recorded order, either way.
    #[test]
    fn each_graph_has_one_slot_and_lists_its_own_entries() {
        let mut set = ChangeSet::new();
        let graphs = [lpg(None), rdf(None), lpg(Some("Paris")), rdf(Some("Paris"))];
        let slots: Vec<GraphSlot> = graphs
            .iter()
            .map(|graph| set.slot(graph.clone()).unwrap())
            .collect();
        assert_eq!(
            slots.iter().map(|slot| slot.index()).collect::<Vec<_>>(),
            [0, 1, 2, 3],
            "a model and a key make a graph"
        );
        for (graph, slot) in graphs.iter().zip(&slots) {
            assert_eq!(set.slot(graph.clone()).unwrap(), *slot, "{graph:?}");
            assert_eq!(set.graph(*slot), Some(graph));
        }
        assert_eq!(set.graphs(), graphs);
        assert_eq!(set.graph(GraphSlot(4)), None);

        let (default, paris) = (slots[0], slots[2]);
        for id in [3, 19, 88] {
            for graph in [default, paris] {
                set.push(
                    graph,
                    create_node(id, &["City"], Vec::new()),
                    Before::Absent,
                    Created,
                )
                .unwrap();
            }
        }
        let ids = |changes: Vec<&Change>| -> Vec<Option<Entity>> {
            changes
                .into_iter()
                .map(|change| match change {
                    Change::Data { op, .. } => op.entity(),
                    Change::Bulk(_) => None,
                })
                .collect()
        };
        let node = |id| Some(Entity::Node(NodeId::new(id)));
        assert_eq!(
            ids(set.in_graph(paris).collect()),
            [node(3), node(19), node(88)]
        );
        assert_eq!(
            ids(set.in_graph(default).rev().collect()),
            [node(88), node(19), node(3)]
        );
        assert_eq!(set.in_graph(slots[1]).count(), 0);
    }

    /// Every op names its model, its table and entity, and for a property
    /// its key, so a consumer (undo, dirty tracking, change data capture)
    /// needs nothing else.
    #[test]
    fn every_op_names_its_model_entity_and_key() {
        let (set, _) = pinned_entries();
        let named: Vec<(DataModel, Option<Entity>, Option<&str>)> = set
            .entries()
            .iter()
            .map(|change| match change {
                Change::Data { op, .. } => (
                    op.model(),
                    op.entity(),
                    op.property_key().map(PropertyKey::as_str),
                ),
                Change::Bulk(_) => panic!("{change:?}"),
            })
            .collect();
        let node = |id| Some(Entity::Node(NodeId::new(id)));
        let edge = |id| Some(Entity::Edge(EdgeId::new(id)));
        let lpg = DataModel::Lpg;
        assert_eq!(
            named,
            [
                (lpg, node(3), None),
                (lpg, node(19), None),
                (lpg, edge(88), None),
                (lpg, edge(88), None),
                (lpg, node(3), Some("city")),
                (lpg, node(3), Some("city")),
                (lpg, edge(88), Some("km")),
                (lpg, edge(88), Some("km")),
                (lpg, node(19), None),
                (lpg, node(19), None),
                (DataModel::Rdf, None, None),
                (DataModel::Rdf, None, None),
            ]
        );
        assert_eq!(Entity::Node(NodeId::new(3)).table(), Table::Nodes);
        assert_eq!(Entity::Edge(EdgeId::new(3)).table(), Table::Edges);
        // The catalog key of a dropped index names its graph too.
        let dropped = CatalogKey::Index {
            graph: Some("trips".into()),
            index: IndexKeyRecord::Property { key: "city".into() },
        };
        assert_eq!(dropped.kind(), 7);
    }
}
