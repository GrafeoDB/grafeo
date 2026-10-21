//! The conformance suite of the stores behind the change set.
//!
//! Every case writes a store through [`ChangeTarget`] as the engine does: a
//! transaction's writes recorded in a [`ChangeSet`], then stamped at commit
//! or undone at rollback and savepoints; a direct call as an immediate
//! write; recovery as replay. It checks what readers see through the read
//! traits only, so any store passes the same cases: `LpgStore` now, the
//! row-group store from its first cluster on (workstream H).
//!
//! A store lists the cases it is known to fail, each with its reason (a
//! [`KnownGap`]). Such a case must keep failing: once it passes, its test
//! fails too and asks to drop the gap, so the list says exactly where a
//! store stands.

mod cases;
mod epochs;
mod graphs;
mod indexes;
mod lpg;
mod rowgroup;
#[cfg(any(feature = "text-index", feature = "vector-index"))]
mod searches;

use std::collections::{BTreeMap, BTreeSet};
use std::panic::{AssertUnwindSafe, catch_unwind};

use grafeo_common::change::{ChangeMark, ChangeSet, DataModel, DataOp, GraphRef, GraphSlot};
use grafeo_common::types::{
    ArcStr, EdgeId, EpochId, NodeId, PropertyKey, ReadView, TransactionId, Value,
};
use grafeo_common::utils::hash::FxHashMap;

use crate::graph::Direction;
use crate::graph::apply::{Applied, ApplyError, ChangeTarget, Writer};
use crate::graph::traits::GraphStoreMut;

/// A store the suite can run: written through the change target, read
/// through the read traits.
pub(crate) trait Store: GraphStoreMut + ChangeTarget {}

impl<T: GraphStoreMut + ChangeTarget> Store for T {}

/// A case a store is known to fail, and why.
#[derive(Clone, Copy)]
pub(crate) struct KnownGap {
    /// The case's name.
    pub case: &'static str,
    /// Why the store fails it (an issue or an inbox note).
    pub reason: &'static str,
}

/// What a case finds: `Err` names the first guarantee that did not hold.
pub(crate) type Outcome = Result<(), String>;

/// Runs `case` on stores from `make` and checks the outcome against the
/// store's known gaps.
///
/// # Panics
///
/// Panics when a case fails that is not a known gap, and when a known gap
/// passes.
pub(crate) fn run<S: Store>(
    name: &str,
    case: fn(&dyn Fn() -> S) -> Outcome,
    make: fn() -> S,
    gaps: &[KnownGap],
) {
    let outcome = match catch_unwind(AssertUnwindSafe(|| case(&make))) {
        Ok(outcome) => outcome,
        Err(panic) => Err(panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|text| (*text).to_string()))
            .unwrap_or_else(|| "the case panicked".to_string())),
    };
    match (outcome, gaps.iter().find(|gap| gap.case == name)) {
        (Ok(()), None) | (Err(_), Some(_)) => {}
        (Err(failure), None) => panic!("{name}: {failure}"),
        (Ok(()), Some(gap)) => panic!(
            "{name} passes now: drop it from the store's known gaps ({})",
            gap.reason
        ),
    }
}

/// Instantiates every case for a store: `conformance_suite!(Store, make,
/// GAPS)` in a test module gives one test per case.
macro_rules! conformance_suite {
    ($store:ty, $make:expr, $gaps:expr) => {
        $crate::graph::conformance::conformance_suite!(@cases $store, $make, $gaps;
            each_op_is_seen_by_its_transaction_and_committed_by_stamp,
            undo_restores_the_committed_state,
            a_savepoint_rollback_restores_each_tail,
            a_savepoint_rollback_keeps_the_transactions_own_value,
            a_node_created_and_deleted_in_one_transaction_comes_back_at_a_savepoint,
            a_detach_delete_is_undone_with_its_edges,
            before_images_describe_what_each_write_replaced,
            writes_that_change_nothing_are_unchanged,
            replay_refuses_what_a_log_cannot_hold,
            replay_rebuilds_what_live_commits_wrote,
            immediate_writes_commit_at_once,
            immediate_writes_and_replay_agree,
            reserved_ids_are_never_given_out_again,
            bulk_ranges_are_stamped_and_undone_by_range,
            a_bulk_range_across_row_groups_commits_each_row,
            counts_move_at_commit_only,
            every_access_path_agrees,
            a_node_with_edges_cannot_be_deleted,
            names_first_used_by_a_rolled_back_transaction_stay_listed,
            an_open_transactions_creates_are_invisible_to_others,
            an_open_transactions_value_and_label_writes_are_invisible_to_others,
            an_open_transactions_deletes_are_invisible_to_others,
            a_reader_at_an_earlier_epoch_keeps_its_values_and_labels,
            a_reader_at_an_earlier_epoch_keeps_deleted_entities,
            a_delete_is_marked_at_its_commit_epoch,
            every_read_path_sees_the_transactions_own_writes,
            every_read_path_hides_an_open_transactions_creates_and_deletes,
            every_read_path_hides_an_open_transactions_values_and_labels,
            every_read_path_at_an_earlier_epoch_keeps_creates_and_deletes,
            every_read_path_at_an_earlier_epoch_keeps_values_and_labels,
            every_read_path_at_the_commit_epoch_sees_the_commit,
            an_immediate_write_is_seen_at_its_epoch_through_every_read_path,
            an_immediate_write_keeps_the_old_value_for_an_older_reader,
            a_property_set_to_null_is_removed,
        );
    };
    (@cases $store:ty, $make:expr, $gaps:expr; $($case:ident),* $(,)?) => {
        $(
            #[test]
            fn $case() {
                $crate::graph::conformance::run::<$store>(
                    stringify!($case),
                    $crate::graph::conformance::cases::$case::<$store>,
                    $make,
                    $gaps,
                );
            }
        )*
    };
}
pub(crate) use conformance_suite;

// ── Writing ─────────────────────────────────────────────────────────

/// The labels of `names`.
pub(crate) fn labels(names: &[&str]) -> grafeo_common::change::Labels {
    names.iter().map(|name| ArcStr::from(*name)).collect()
}

/// The properties of `pairs`.
pub(crate) fn properties(pairs: &[(&str, Value)]) -> grafeo_common::change::Properties {
    pairs
        .iter()
        .map(|(key, value)| (PropertyKey::new(*key), value.clone()))
        .collect()
}

/// The epoch after the store's current one: where the next commit lands.
pub(crate) fn next_epoch<S: Store + ?Sized>(store: &S) -> EpochId {
    EpochId::new(store.current_epoch().as_u64() + 1)
}

/// A change set's slot for the default graph.
fn lpg_slot(set: &mut ChangeSet) -> GraphSlot {
    set.slot(GraphRef {
        model: DataModel::Lpg,
        key: None,
    })
    .expect("a new change set takes a slot")
}

/// A transaction on one store, as the engine runs one: each write that
/// changed something recorded in its change set, which commit stamps and
/// rollback undoes.
pub(crate) struct Tx<'s, S: Store> {
    store: &'s S,
    pub id: TransactionId,
    pub snapshot: EpochId,
    set: ChangeSet,
    slot: GraphSlot,
}

impl<'s, S: Store> Tx<'s, S> {
    /// Begins transaction `id` at the store's current epoch.
    pub fn begin(store: &'s S, id: u64) -> Self {
        let mut set = ChangeSet::new();
        let slot = lpg_slot(&mut set);
        Self {
            store,
            id: TransactionId::new(id),
            snapshot: store.current_epoch(),
            set,
            slot,
        }
    }

    /// The transaction's writer.
    pub fn writer(&self) -> Writer {
        Writer::Transaction {
            id: self.id,
            snapshot: self.snapshot,
        }
    }

    /// Applies `op` and records it when it changed something.
    ///
    /// # Errors
    ///
    /// Returns what `apply` refused, and a change set refusing the entry as
    /// `Refused`.
    pub fn apply(&mut self, op: DataOp) -> Result<Applied, ApplyError> {
        let applied = self.store.apply(&op, self.writer())?;
        if let Applied::Changed { before, version } = &applied {
            self.set
                .push(self.slot, op, before.clone(), *version)
                .map_err(|error| ApplyError::Refused(error.to_string()))?;
        }
        Ok(applied)
    }

    /// Applies `op`, which must change something.
    pub fn change(&mut self, op: DataOp) -> Outcome {
        match self.apply(op.clone()) {
            Ok(Applied::Changed { .. }) => Ok(()),
            other => Err(format!("{op:?}: expected a change, got {other:?}")),
        }
    }

    /// Creates a node at a reserved id.
    pub fn create_node(
        &mut self,
        names: &[&str],
        values: &[(&str, Value)],
    ) -> Result<NodeId, String> {
        let id = NodeId::new(reserve_node(self.store)?);
        self.change(DataOp::CreateNode {
            id,
            labels: labels(names),
            properties: properties(values),
        })?;
        Ok(id)
    }

    /// Creates an edge at a reserved id.
    pub fn create_edge(
        &mut self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        values: &[(&str, Value)],
    ) -> Result<EdgeId, String> {
        let id = EdgeId::new(reserve_edge(self.store)?);
        self.change(DataOp::CreateEdge {
            id,
            src,
            dst,
            edge_type: ArcStr::from(edge_type),
            properties: properties(values),
        })?;
        Ok(id)
    }

    /// Sets a node's value.
    pub fn set_node(&mut self, id: NodeId, key: &str, value: Value) -> Outcome {
        self.change(DataOp::SetNodeProperty {
            id,
            key: PropertyKey::new(key),
            value,
        })
    }

    /// Removes a node's value, which it has.
    pub fn remove_node_value(&mut self, id: NodeId, key: &str) -> Outcome {
        self.change(DataOp::RemoveNodeProperty {
            id,
            key: PropertyKey::new(key),
        })
    }

    /// Sets an edge's value.
    pub fn set_edge(&mut self, id: EdgeId, key: &str, value: Value) -> Outcome {
        self.change(DataOp::SetEdgeProperty {
            id,
            key: PropertyKey::new(key),
            value,
        })
    }

    /// Removes an edge's value, which it has.
    pub fn remove_edge_value(&mut self, id: EdgeId, key: &str) -> Outcome {
        self.change(DataOp::RemoveEdgeProperty {
            id,
            key: PropertyKey::new(key),
        })
    }

    /// Adds a label the node lacks.
    pub fn add_label(&mut self, id: NodeId, label: &str) -> Outcome {
        self.change(DataOp::AddNodeLabel {
            id,
            label: ArcStr::from(label),
        })
    }

    /// Removes a label the node has.
    pub fn remove_label(&mut self, id: NodeId, label: &str) -> Outcome {
        self.change(DataOp::RemoveNodeLabel {
            id,
            label: ArcStr::from(label),
        })
    }

    /// Deletes an edge.
    pub fn delete_edge(&mut self, id: EdgeId) -> Outcome {
        self.change(DataOp::DeleteEdge { id })
    }

    /// Deletes a node without edges.
    pub fn delete_node(&mut self, id: NodeId) -> Outcome {
        self.change(DataOp::DeleteNode { id })
    }

    /// Deletes a node and, first, every edge of it the transaction sees, as
    /// a detach delete records them.
    pub fn detach_delete(&mut self, id: NodeId) -> Outcome {
        let view = self.view();
        let edges: Vec<u64> = view
            .edges
            .iter()
            .filter(|(_, (src, dst, _, _))| *src == id.as_u64() || *dst == id.as_u64())
            .map(|(edge, _)| *edge)
            .collect();
        for edge in edges {
            self.delete_edge(EdgeId::new(edge))?;
        }
        self.delete_node(id)
    }

    /// What the transaction sees, its own writes included.
    pub fn view(&self) -> Image {
        image(self.store, Reader::Transaction(self.id, self.snapshot))
    }

    /// A savepoint.
    pub fn mark(&self) -> ChangeMark {
        self.set.mark()
    }

    /// Rolls back to `mark`: the writes after it are undone, last first.
    pub fn roll_back_to(&mut self, mark: ChangeMark) -> Outcome {
        let tail = self.set.split_off(mark);
        self.store
            .undo(self.id, &mut tail.iter())
            .map_err(|error| format!("undo to a savepoint: {error:?}"))
    }

    /// The ops recorded so far, in write order: what the log holds.
    pub fn ops(&self) -> Vec<DataOp> {
        self.set
            .entries()
            .iter()
            .filter_map(|change| match change {
                grafeo_common::change::Change::Data { op, .. } => Some(op.clone()),
                grafeo_common::change::Change::Bulk(_) => None,
            })
            .collect()
    }

    /// Records a bulk write's reserved range.
    pub fn push_bulk(
        &mut self,
        table: grafeo_common::change::Table,
        ids: std::ops::Range<u64>,
    ) -> Outcome {
        self.set
            .push_bulk(grafeo_common::change::BulkRange {
                graph: self.slot,
                table,
                ids,
            })
            .map_err(|error| format!("a bulk range: {error}"))
    }

    /// Commits at the next epoch, which it returns.
    pub fn commit(self) -> Result<EpochId, String> {
        let epoch = next_epoch(self.store);
        self.store
            .stamp(self.id, &mut self.set.entries().iter(), epoch)
            .map_err(|error| format!("stamp: {error:?}"))?;
        Ok(epoch)
    }

    /// Rolls the whole transaction back.
    pub fn roll_back(self) -> Outcome {
        self.store
            .undo(self.id, &mut self.set.entries().iter())
            .map_err(|error| format!("undo: {error:?}"))
    }
}

/// Reserves one node id.
pub(crate) fn reserve_node<S: Store + ?Sized>(store: &S) -> Result<u64, String> {
    store
        .reserve_node_ids(1)
        .map(|ids| ids.start)
        .map_err(|error| format!("reserve a node id: {error:?}"))
}

/// Reserves one edge id.
pub(crate) fn reserve_edge<S: Store + ?Sized>(store: &S) -> Result<u64, String> {
    store
        .reserve_edge_ids(1)
        .map(|ids| ids.start)
        .map_err(|error| format!("reserve an edge id: {error:?}"))
}

/// Applies `ops` as `writer`, each of which must apply.
pub(crate) fn apply_all<S: Store + ?Sized>(store: &S, ops: &[DataOp], writer: Writer) -> Outcome {
    for op in ops {
        store
            .apply(op, writer)
            .map_err(|error| format!("{op:?} as {writer:?}: {error:?}"))?;
    }
    Ok(())
}

// ── Reading ─────────────────────────────────────────────────────────

/// A node as a reader sees it: its labels and values, sorted.
pub(crate) type NodeView = (Vec<String>, Vec<(String, Value)>);

/// An edge as a reader sees it: source, target, type and values.
pub(crate) type EdgeView = (u64, u64, String, Vec<(String, Value)>);

/// Who reads.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Reader {
    /// A transaction at its snapshot, its own writes included.
    Transaction(TransactionId, EpochId),
    /// A committed reader at an epoch.
    At(EpochId),
}

/// What a reader sees of a store, and every access path that disagrees
/// with it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Image {
    /// The nodes it sees, by id.
    pub nodes: BTreeMap<u64, NodeView>,
    /// The edges it sees from their source's outgoing adjacency, by id.
    pub edges: BTreeMap<u64, EdgeView>,
    /// Where another access path disagrees: incoming adjacency, an edge with
    /// an endpoint the reader does not see, the label scans, the property
    /// index on `name`.
    pub problems: Vec<String>,
}

impl Image {
    /// Whether the reader sees `id`, with `labels` and the value `value` of
    /// `key` (`None`: no value).
    pub fn has_node(
        &self,
        id: NodeId,
        labels: &[&str],
        key: &str,
        value: Option<&Value>,
    ) -> Outcome {
        let Some((node_labels, values)) = self.nodes.get(&id.as_u64()) else {
            return Err(format!("node {} is not seen", id.as_u64()));
        };
        let mut expected: Vec<String> = labels.iter().map(|label| (*label).to_string()).collect();
        expected.sort();
        if *node_labels != expected {
            return Err(format!(
                "node {} has the labels {node_labels:?}, expected {expected:?}",
                id.as_u64()
            ));
        }
        let found = values
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value);
        if found != value {
            return Err(format!(
                "node {}'s {key} is {found:?}, expected {value:?}",
                id.as_u64()
            ));
        }
        Ok(())
    }

    /// Fails with `what` when the image has problems.
    pub fn consistent(&self, what: &str) -> Outcome {
        if self.problems.is_empty() {
            Ok(())
        } else {
            Err(format!("{what}: {}", self.problems.join("; ")))
        }
    }
}

/// Fails with `what` unless `left` equals `right`.
pub(crate) fn same<T: PartialEq + std::fmt::Debug>(left: &T, right: &T, what: &str) -> Outcome {
    if left == right {
        Ok(())
    } else {
        Err(format!(
            "{what}:\n  found    {left:?}\n  expected {right:?}"
        ))
    }
}

/// Fails with `what` unless `condition` holds.
pub(crate) fn ensure(condition: bool, what: impl FnOnce() -> String) -> Outcome {
    if condition { Ok(()) } else { Err(what()) }
}

fn sorted_values(values: &grafeo_common::types::PropertyMap) -> Vec<(String, Value)> {
    values
        .to_btree_map()
        .into_iter()
        .map(|(key, value)| (key.as_str().to_string(), value))
        .collect()
}

fn node_view<S: Store + ?Sized>(store: &S, reader: Reader, id: NodeId) -> Option<NodeView> {
    let node = match reader {
        Reader::Transaction(transaction, snapshot) => {
            store.get_node_versioned(id, snapshot, transaction)
        }
        Reader::At(epoch) => store.get_node_at_epoch(id, epoch),
    }?;
    let mut names: Vec<String> = node.labels.iter().map(ToString::to_string).collect();
    names.sort();
    Some((names, sorted_values(&node.properties)))
}

fn edge_view<S: Store + ?Sized>(store: &S, reader: Reader, id: EdgeId) -> Option<EdgeView> {
    let edge = match reader {
        Reader::Transaction(transaction, snapshot) => {
            store.get_edge_versioned(id, snapshot, transaction)
        }
        Reader::At(epoch) => store.get_edge_at_epoch(id, epoch),
    }?;
    Some((
        edge.src.as_u64(),
        edge.dst.as_u64(),
        edge.edge_type.to_string(),
        sorted_values(&edge.properties),
    ))
}

/// What `reader` sees of `store`: the nodes the store lists, the edges of
/// their outgoing adjacency, each read at the reader's snapshot; then the
/// other access paths checked against them.
pub(crate) fn image<S: Store + ?Sized>(store: &S, reader: Reader) -> Image {
    let mut problems = Vec::new();
    let mut nodes = BTreeMap::new();
    for id in store.all_node_ids() {
        if let Some(view) = node_view(store, reader, id) {
            nodes.insert(id.as_u64(), view);
        }
    }

    let mut edges = BTreeMap::new();
    for &node in nodes.keys() {
        for (target, edge) in store.edges_from(NodeId::new(node), Direction::Outgoing) {
            let Some(view) = edge_view(store, reader, edge) else {
                continue;
            };
            if view.0 != node || view.1 != target.as_u64() {
                problems.push(format!(
                    "edge {} is in node {node}'s outgoing adjacency to {} but runs {} to {}",
                    edge.as_u64(),
                    target.as_u64(),
                    view.0,
                    view.1
                ));
            }
            if edges.insert(edge.as_u64(), view).is_some() {
                problems.push(format!("edge {} is listed twice", edge.as_u64()));
            }
        }
    }
    for (edge, (src, dst, _, _)) in &edges {
        if !nodes.contains_key(dst) || !nodes.contains_key(src) {
            problems.push(format!(
                "edge {edge} runs {src} to {dst}, an endpoint not seen"
            ));
        }
    }

    // Incoming edges: from the incoming adjacency, or by a scan of a store
    // that keeps none; either way every edge into the node.
    for &node in nodes.keys() {
        let incoming: BTreeSet<u64> = store
            .edges_from(NodeId::new(node), Direction::Incoming)
            .into_iter()
            .filter(|(_, edge)| edge_view(store, reader, *edge).is_some())
            .map(|(_, edge)| edge.as_u64())
            .collect();
        let expected: BTreeSet<u64> = edges
            .iter()
            .filter(|(_, (_, dst, _, _))| *dst == node)
            .map(|(edge, _)| *edge)
            .collect();
        if incoming != expected {
            problems.push(format!(
                "node {node}'s incoming adjacency lists {incoming:?}, its edges are {expected:?}"
            ));
        }
    }

    let mut label_names: BTreeSet<String> = store.all_labels().into_iter().collect();
    for (names, _) in nodes.values() {
        label_names.extend(names.iter().cloned());
    }
    for label in label_names {
        // At the reader's view: the labels a scan finds are those of the
        // view, as the point reads' are.
        let scanned: BTreeSet<u64> = store
            .nodes_by_label_at(&label, read_view(reader))
            .into_iter()
            .map(|id| id.as_u64())
            .filter(|id| nodes.contains_key(id))
            .collect();
        let expected: BTreeSet<u64> = nodes
            .iter()
            .filter(|(_, (names, _))| names.contains(&label))
            .map(|(id, _)| *id)
            .collect();
        let missed: Vec<&u64> = expected.difference(&scanned).collect();
        let extra: Vec<&u64> = scanned.difference(&expected).collect();
        if !missed.is_empty() || !extra.is_empty() {
            problems.push(format!(
                "the {label} scan misses {missed:?} and returns {extra:?} without the label"
            ));
        }
    }

    if store.has_property_index("name") {
        // Values have no order: group the nodes by name in a list.
        let mut by_name: Vec<(&Value, BTreeSet<u64>)> = Vec::new();
        for (id, (_, values)) in &nodes {
            if let Some((_, value)) = values.iter().find(|(key, _)| key == "name") {
                match by_name.iter_mut().find(|(name, _)| *name == value) {
                    Some((_, ids)) => {
                        ids.insert(*id);
                    }
                    None => by_name.push((value, BTreeSet::from([*id]))),
                }
            }
        }
        for (value, expected) in by_name {
            match store.find_nodes_maybe_equal("name", value) {
                None => problems.push(format!("the name index gives no answer for {value:?}")),
                Some(found) => {
                    let found: BTreeSet<u64> = found.into_iter().map(|id| id.as_u64()).collect();
                    let missed: Vec<&u64> = expected.difference(&found).collect();
                    if !missed.is_empty() {
                        problems.push(format!("the name index misses {missed:?} for {value:?}"));
                    }
                }
            }
        }
    }

    Image {
        nodes,
        edges,
        problems,
    }
}

/// The committed state now, as a reader at the store's epoch sees it.
pub(crate) fn committed<S: Store + ?Sized>(store: &S) -> Image {
    image(store, Reader::At(store.current_epoch()))
}

/// The counts the store keeps of its committed state: nodes, edges, and
/// nodes per label of `labels`.
pub(crate) fn counts<S: Store + ?Sized>(store: &S, labels: &[&str]) -> (usize, usize, Vec<usize>) {
    (
        store.node_count(),
        store.edge_count(),
        labels
            .iter()
            .map(|label| store.nodes_by_label_count(label))
            .collect(),
    )
}

/// The counts an image gives: nodes, edges, and nodes per label of
/// `labels`.
pub(crate) fn counts_of(image: &Image, labels: &[&str]) -> (usize, usize, Vec<usize>) {
    (
        image.nodes.len(),
        image.edges.len(),
        labels
            .iter()
            .map(|label| {
                image
                    .nodes
                    .values()
                    .filter(|(names, _)| names.iter().any(|name| name == label))
                    .count()
            })
            .collect(),
    )
}

// ── Reading at a view ───────────────────────────────────────────────

/// The keys the reads at a view probe.
const KEYS: [&str; 3] = ["name", "age", "city"];

/// The label sets the label unions read.
const UNIONS: [&[&str]; 3] = [&["Employee", "Traveller"], &["Person", "City"], &LABELS];

/// The ranges of `age` the range lookups read: (min, max, min inclusive,
/// max inclusive).
const AGE_RANGES: [(Option<i64>, Option<i64>, bool, bool); 4] = [
    (Some(10), Some(50), true, false),
    (None, Some(19), false, true),
    (Some(3), None, true, true),
    (Some(3), Some(88), false, false),
];

/// The view of `reader`.
pub(crate) fn read_view(reader: Reader) -> ReadView {
    match reader {
        Reader::Transaction(transaction, snapshot) => ReadView::transaction(transaction, snapshot),
        Reader::At(epoch) => ReadView::committed(epoch),
    }
}

impl Image {
    /// Gives node `id` the value `value` of `key` (`None`: no value).
    pub fn set_value(&mut self, id: NodeId, key: &str, value: Option<Value>) {
        if let Some((_, values)) = self.nodes.get_mut(&id.as_u64()) {
            set_sorted(values, key, value);
        }
    }

    /// Gives node `id` exactly the labels `names`.
    pub fn set_labels(&mut self, id: NodeId, names: &[&str]) {
        if let Some((labels, _)) = self.nodes.get_mut(&id.as_u64()) {
            *labels = names.iter().map(|name| (*name).to_string()).collect();
            labels.sort();
        }
    }

    /// Gives edge `id` the value `value` of `key` (`None`: no value).
    pub fn set_edge_value(&mut self, id: EdgeId, key: &str, value: Option<Value>) {
        if let Some((_, _, _, values)) = self.edges.get_mut(&id.as_u64()) {
            set_sorted(values, key, value);
        }
    }
}

/// Sets `key` to `value` in a list of values sorted by key.
fn set_sorted(values: &mut Vec<(String, Value)>, key: &str, value: Option<Value>) {
    values.retain(|(name, _)| name != key);
    if let Some(value) = value {
        values.push((key.to_string(), value));
        values.sort_by(|a, b| a.0.cmp(&b.0));
    }
}

/// The node ids of an answer, as a set, with a problem when one repeats.
fn id_set(ids: &[NodeId], what: &str, problems: &mut Vec<String>) -> BTreeSet<u64> {
    let set: BTreeSet<u64> = ids.iter().map(|id| id.as_u64()).collect();
    if set.len() != ids.len() {
        problems.push(format!("{what} lists a node twice: {ids:?}"));
    }
    set
}

/// The values of a batch read, sorted by key.
fn sorted_map(map: &FxHashMap<PropertyKey, Value>) -> Vec<(String, Value)> {
    let mut values: Vec<(String, Value)> = map
        .iter()
        .map(|(key, value)| (key.as_str().to_string(), value.clone()))
        .collect();
    values.sort_by(|a, b| a.0.cmp(&b.0));
    values
}

/// Every view-taking read of `store` at `reader`'s view, each checked
/// against `expected`, what the reader must see: point reads of nodes and
/// edges (whole, one value, visibility), labels and label checks, the
/// batch and selective batch reads, adjacency each way, label scans and
/// unions, equality lookups (one key, two keys, maybe-equal through an
/// index) and range lookups (eager and lazy). The lookups probe the values
/// of `expected` and of `others` (images the reader must not see), so a
/// value written by someone the reader does not see finds nothing.
///
/// # Errors
///
/// Every read that disagrees with `expected`, one per line.
pub(crate) fn paths_at<S: Store + ?Sized>(
    store: &S,
    reader: Reader,
    expected: &Image,
    others: &[&Image],
) -> Outcome {
    let view = read_view(reader);
    let mut problems = Vec::new();
    let mut node_ids: BTreeSet<u64> = store
        .all_node_ids()
        .into_iter()
        .map(|id| id.as_u64())
        .collect();
    let mut edge_ids: BTreeSet<u64> = BTreeSet::new();
    for image in std::iter::once(expected).chain(others.iter().copied()) {
        node_ids.extend(image.nodes.keys().copied());
        edge_ids.extend(image.edges.keys().copied());
    }
    let nodes: Vec<NodeId> = node_ids.iter().map(|id| NodeId::new(*id)).collect();
    let edges: Vec<EdgeId> = edge_ids.iter().map(|id| EdgeId::new(*id)).collect();

    // Nodes, one at a time.
    for &node in &nodes {
        let raw = node.as_u64();
        let want = expected.nodes.get(&raw);
        if store.is_node_visible_at(node, view) != want.is_some() {
            problems.push(format!(
                "is_node_visible_at({raw}) is {}, expected {}",
                want.is_none(),
                want.is_some()
            ));
        }
        let found = store.get_node_at(node, view).map(|node| {
            let mut names: Vec<String> = node.labels.iter().map(ToString::to_string).collect();
            names.sort();
            (names, sorted_values(&node.properties))
        });
        if found.as_ref() != want {
            problems.push(format!(
                "get_node_at({raw}) is {found:?}, expected {want:?}"
            ));
        }
        let labels = store.node_labels_at(node, view).map(|labels| {
            let mut names: Vec<String> = labels.iter().map(ToString::to_string).collect();
            names.sort();
            names
        });
        if labels.as_ref() != want.map(|(names, _)| names) {
            problems.push(format!(
                "node_labels_at({raw}) is {labels:?}, expected {:?}",
                want.map(|(names, _)| names)
            ));
        }
        for label in LABELS {
            let has = store.node_has_label_at(node, label, view);
            let expected_has = want.map(|(names, _)| names.iter().any(|name| name == label));
            if has != expected_has {
                problems.push(format!(
                    "node_has_label_at({raw}, {label}) is {has:?}, expected {expected_has:?}"
                ));
            }
        }
        for key in KEYS {
            let value = store.node_property_at(node, key, view);
            let expected_value = want.and_then(|(_, values)| {
                values
                    .iter()
                    .find(|(name, _)| name == key)
                    .map(|(_, value)| value.clone())
            });
            if value != expected_value {
                problems.push(format!(
                    "node_property_at({raw}, {key}) is {value:?}, expected {expected_value:?}"
                ));
            }
        }
    }

    // Nodes, in batches.
    let expected_values = |node: &NodeId, keys: Option<&[&str]>| -> Vec<(String, Value)> {
        expected
            .nodes
            .get(&node.as_u64())
            .map(|(_, values)| {
                values
                    .iter()
                    .filter(|(key, _)| keys.is_none_or(|keys| keys.contains(&key.as_str())))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    };
    for key in KEYS {
        let found = store.get_node_property_batch_at(&nodes, &PropertyKey::new(key), view);
        for (node, value) in nodes.iter().zip(&found) {
            let expected_value = expected_values(node, Some(&[key])).pop().map(|(_, v)| v);
            if *value != expected_value {
                problems.push(format!(
                    "get_node_property_batch_at: node {}'s {key} is {value:?}, expected {expected_value:?}",
                    node.as_u64()
                ));
            }
        }
    }
    let all = store.get_nodes_properties_batch_at(&nodes, view);
    let selected = store.get_nodes_properties_selective_batch_at(
        &nodes,
        &[PropertyKey::new("name"), PropertyKey::new("age")],
        view,
    );
    for (index, node) in nodes.iter().enumerate() {
        let (found, wanted) = (
            all.get(index).map(sorted_map),
            Some(expected_values(node, None)),
        );
        if found != wanted {
            problems.push(format!(
                "get_nodes_properties_batch_at: node {} has {found:?}, expected {wanted:?}",
                node.as_u64()
            ));
        }
        let (found, wanted) = (
            selected.get(index).map(sorted_map),
            Some(expected_values(node, Some(&["name", "age"]))),
        );
        if found != wanted {
            problems.push(format!(
                "get_nodes_properties_selective_batch_at: node {} has {found:?}, expected {wanted:?}",
                node.as_u64()
            ));
        }
    }

    // Edges.
    for &edge in &edges {
        let raw = edge.as_u64();
        let want = expected.edges.get(&raw);
        if store.is_edge_visible_at(edge, view) != want.is_some() {
            problems.push(format!(
                "is_edge_visible_at({raw}) is {}, expected {}",
                want.is_none(),
                want.is_some()
            ));
        }
        let found = store.get_edge_at(edge, view).map(|edge| {
            (
                edge.src.as_u64(),
                edge.dst.as_u64(),
                edge.edge_type.to_string(),
                sorted_values(&edge.properties),
            )
        });
        if found.as_ref() != want {
            problems.push(format!(
                "get_edge_at({raw}) is {found:?}, expected {want:?}"
            ));
        }
        let since = store.edge_property_at(edge, "since", view);
        let expected_since = want.and_then(|(_, _, _, values)| {
            values
                .iter()
                .find(|(key, _)| key == "since")
                .map(|(_, value)| value.clone())
        });
        if since != expected_since {
            problems.push(format!(
                "edge_property_at({raw}, since) is {since:?}, expected {expected_since:?}"
            ));
        }
    }
    let found =
        store.get_edges_properties_selective_batch_at(&edges, &[PropertyKey::new("since")], view);
    for (index, edge) in edges.iter().enumerate() {
        let wanted: Vec<(String, Value)> = expected
            .edges
            .get(&edge.as_u64())
            .map(|(_, _, _, values)| {
                values
                    .iter()
                    .filter(|(key, _)| key == "since")
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        let found = found.get(index).map(sorted_map);
        if found.as_ref() != Some(&wanted) {
            problems.push(format!(
                "get_edges_properties_selective_batch_at: edge {} has {found:?}, expected {wanted:?}",
                edge.as_u64()
            ));
        }
    }

    // Adjacency.
    // Every direction, also on a store that keeps no incoming edges.
    let directions = [Direction::Outgoing, Direction::Incoming, Direction::Both];
    for &raw in expected.nodes.keys() {
        for &direction in &directions {
            let mut wanted: Vec<(u64, u64)> = expected
                .edges
                .iter()
                .flat_map(|(edge, (src, dst, _, _))| {
                    let out =
                        (*src == raw && direction != Direction::Incoming).then_some((*dst, *edge));
                    let into =
                        (*dst == raw && direction != Direction::Outgoing).then_some((*src, *edge));
                    // `Both` lists a self-loop from each side.
                    out.into_iter().chain(into)
                })
                .collect();
            wanted.sort_unstable();
            let mut found: Vec<(u64, u64)> = store
                .edges_from_at(NodeId::new(raw), direction, view)
                .into_iter()
                .map(|(other, edge)| (other.as_u64(), edge.as_u64()))
                .collect();
            found.sort_unstable();
            if found != wanted {
                problems.push(format!(
                    "edges_from_at({raw}, {direction:?}) is {found:?}, expected {wanted:?}"
                ));
            }
            let mut neighbors: Vec<u64> = store
                .neighbors_at(NodeId::new(raw), direction, view)
                .into_iter()
                .map(|id| id.as_u64())
                .collect();
            neighbors.sort_unstable();
            let wanted: Vec<u64> = wanted.iter().map(|(other, _)| *other).collect();
            let mut wanted = wanted;
            wanted.sort_unstable();
            if neighbors != wanted {
                problems.push(format!(
                    "neighbors_at({raw}, {direction:?}) is {neighbors:?}, expected {wanted:?}"
                ));
            }
        }
    }

    // Label scans and unions.
    let with_any = |labels: &[&str]| -> BTreeSet<u64> {
        expected
            .nodes
            .iter()
            .filter(|(_, (names, _))| names.iter().any(|name| labels.contains(&name.as_str())))
            .map(|(id, _)| *id)
            .collect()
    };
    let mut label_names: BTreeSet<String> = store.all_labels().into_iter().collect();
    label_names.extend(LABELS.iter().map(|label| (*label).to_string()));
    for label in &label_names {
        let what = format!("nodes_by_label_at({label})");
        let found = id_set(&store.nodes_by_label_at(label, view), &what, &mut problems);
        let wanted = with_any(&[label.as_str()]);
        if found != wanted {
            problems.push(format!("{what} is {found:?}, expected {wanted:?}"));
        }
    }
    for labels in UNIONS {
        let what = format!("nodes_with_any_label_at({labels:?})");
        let found = id_set(
            &store.nodes_with_any_label_at(labels, view),
            &what,
            &mut problems,
        );
        let wanted = with_any(labels);
        if found != wanted {
            problems.push(format!("{what} is {found:?}, expected {wanted:?}"));
        }
    }

    // Equality lookups, probing every value of the images.
    let mut probes: Vec<(&str, Value)> = Vec::new();
    for image in std::iter::once(expected).chain(others.iter().copied()) {
        for (_, values) in image.nodes.values() {
            for (key, value) in values {
                if let Some(key) = KEYS.iter().find(|name| **name == key.as_str())
                    && !probes.iter().any(|(k, v)| k == key && v == value)
                {
                    probes.push((key, value.clone()));
                }
            }
        }
    }
    let equal = |key: &str, value: &Value| -> BTreeSet<u64> {
        expected
            .nodes
            .iter()
            .filter(|(_, (_, values))| values.iter().any(|(k, v)| k == key && v == value))
            .map(|(id, _)| *id)
            .collect()
    };
    for (key, value) in &probes {
        let what = format!("find_nodes_by_property_at({key}, {value:?})");
        let found = id_set(
            &store.find_nodes_by_property_at(key, value, view),
            &what,
            &mut problems,
        );
        let wanted = equal(key, value);
        if found != wanted {
            problems.push(format!("{what} is {found:?}, expected {wanted:?}"));
        }
        match store.find_nodes_maybe_equal_at(key, value, view) {
            None if store.has_property_index(key) => problems.push(format!(
                "find_nodes_maybe_equal_at({key}, {value:?}) gives no answer with an index"
            )),
            None => {}
            Some(found) => {
                let what = format!("find_nodes_maybe_equal_at({key}, {value:?})");
                let found = id_set(&found, &what, &mut problems);
                let missed: Vec<&u64> = wanted.difference(&found).collect();
                let unseen: Vec<&u64> = found
                    .iter()
                    .filter(|id| !expected.nodes.contains_key(id))
                    .collect();
                if !missed.is_empty() || !unseen.is_empty() {
                    problems.push(format!(
                        "{what} misses {missed:?} and lists {unseen:?}, which the view does not see"
                    ));
                }
            }
        }
    }
    let names: Vec<&Value> = probes
        .iter()
        .filter(|(key, _)| *key == "name")
        .map(|(_, value)| value)
        .collect();
    let ages: Vec<&Value> = probes
        .iter()
        .filter(|(key, _)| *key == "age")
        .map(|(_, value)| value)
        .collect();
    for name in &names {
        for age in &ages {
            let conditions = [("name", (*name).clone()), ("age", (*age).clone())];
            let what = format!("find_nodes_by_properties_at({conditions:?})");
            let found = id_set(
                &store.find_nodes_by_properties_at(&conditions, view),
                &what,
                &mut problems,
            );
            let wanted: BTreeSet<u64> = equal("name", name)
                .intersection(&equal("age", age))
                .copied()
                .collect();
            if found != wanted {
                problems.push(format!("{what} is {found:?}, expected {wanted:?}"));
            }
        }
    }

    // Range lookups of `age`.
    for (min, max, min_inclusive, max_inclusive) in AGE_RANGES {
        let (min, max) = (min.map(Value::Int64), max.map(Value::Int64));
        let wanted: BTreeSet<u64> = expected
            .nodes
            .iter()
            .filter(|(_, (_, values))| {
                values.iter().any(|(key, value)| {
                    key == "age"
                        && matches!(value, Value::Int64(age)
                        if min.as_ref().is_none_or(|min| match min {
                            Value::Int64(min) => if min_inclusive { age >= min } else { age > min },
                            _ => false,
                        })
                        && max.as_ref().is_none_or(|max| match max {
                            Value::Int64(max) => if max_inclusive { age <= max } else { age < max },
                            _ => false,
                        }))
                })
            })
            .map(|(id, _)| *id)
            .collect();
        let what = format!("find_nodes_in_range_at(age, {min:?}, {max:?})");
        let found = id_set(
            &store.find_nodes_in_range_at(
                "age",
                min.as_ref(),
                max.as_ref(),
                min_inclusive,
                max_inclusive,
                view,
            ),
            &what,
            &mut problems,
        );
        if found != wanted {
            problems.push(format!("{what} is {found:?}, expected {wanted:?}"));
        }
        let lazy: Vec<NodeId> = store
            .find_nodes_in_range_iter_at(
                "age",
                min.as_ref(),
                max.as_ref(),
                min_inclusive,
                max_inclusive,
                view,
            )
            .collect();
        let what = format!("find_nodes_in_range_iter_at(age, {min:?}, {max:?})");
        let found = id_set(&lazy, &what, &mut problems);
        if found != wanted {
            problems.push(format!("{what} is {found:?}, expected {wanted:?}"));
        }
    }

    if problems.is_empty() {
        Ok(())
    } else {
        Err(format!("at {reader:?}:\n  {}", problems.join("\n  ")))
    }
}

// ── Data ─────────────────────────────────────────────────────────────

/// The people every case starts from, replayed at the first epoch: Alix
/// knows Gus, who knows Mia.
pub(crate) struct People {
    pub alix: NodeId,
    pub gus: NodeId,
    pub mia: NodeId,
    pub alix_gus: EdgeId,
    pub gus_mia: EdgeId,
}

/// Writes [`People`] into `store` through replay.
pub(crate) fn people<S: Store + ?Sized>(store: &S) -> Result<People, String> {
    let alix = NodeId::new(reserve_node(store)?);
    let gus = NodeId::new(reserve_node(store)?);
    let mia = NodeId::new(reserve_node(store)?);
    let alix_gus = EdgeId::new(reserve_edge(store)?);
    let gus_mia = EdgeId::new(reserve_edge(store)?);
    let ops = [
        DataOp::CreateNode {
            id: alix,
            labels: labels(&["Person"]),
            properties: properties(&[
                ("name", Value::from("Alix")),
                ("age", Value::Int64(19)),
                ("city", Value::from("Amsterdam")),
            ]),
        },
        DataOp::CreateNode {
            id: gus,
            labels: labels(&["Person", "Employee"]),
            properties: properties(&[("name", Value::from("Gus")), ("age", Value::Int64(3))]),
        },
        DataOp::CreateNode {
            id: mia,
            labels: labels(&["Person"]),
            properties: properties(&[("name", Value::from("Mia"))]),
        },
        DataOp::CreateEdge {
            id: alix_gus,
            src: alix,
            dst: gus,
            edge_type: ArcStr::from("KNOWS"),
            properties: properties(&[("since", Value::Int64(1988))]),
        },
        DataOp::CreateEdge {
            id: gus_mia,
            src: gus,
            dst: mia,
            edge_type: ArcStr::from("KNOWS"),
            properties: properties(&[("since", Value::Int64(319))]),
        },
    ];
    apply_all(
        store,
        &ops,
        Writer::Replay {
            epoch: next_epoch(store),
        },
    )?;
    Ok(People {
        alix,
        gus,
        mia,
        alix_gus,
        gus_mia,
    })
}

/// A seeded generator (xorshift), so a failing workload can be replayed.
pub(crate) struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn below(&mut self, bound: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        usize::try_from(self.0 % bound.max(1) as u64).unwrap_or(0)
    }

    /// A number below `bound`, as a value.
    pub fn int(&mut self, bound: usize) -> i64 {
        i64::try_from(self.below(bound)).expect("a bound below i64::MAX")
    }

    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> Option<&'a T> {
        (!items.is_empty()).then(|| &items[self.below(items.len())])
    }
}

/// The labels the workloads use.
pub(crate) const LABELS: [&str; 4] = ["Person", "Employee", "Traveller", "City"];

/// One write of a random kind, valid for what `tx` sees.
pub(crate) fn random_write<S: Store>(tx: &mut Tx<'_, S>, rng: &mut Rng) -> Outcome {
    const NAMES: [&str; 6] = ["Alix", "Gus", "Mia", "Vincent", "Jules", "Django"];
    const CITIES: [&str; 4] = ["Amsterdam", "Berlin", "Paris", "Prague"];
    let view = tx.view();
    let nodes: Vec<u64> = view.nodes.keys().copied().collect();
    let edges: Vec<u64> = view.edges.keys().copied().collect();
    match rng.below(10) {
        0 | 1 => {
            let count = 1 + rng.below(2);
            let mut names: Vec<&str> = (0..count)
                .map(|_| LABELS[rng.below(LABELS.len())])
                .collect();
            names.sort_unstable();
            names.dedup();
            let name = NAMES[rng.below(NAMES.len())];
            tx.create_node(
                &names,
                &[
                    ("name", Value::from(name)),
                    ("age", Value::Int64(rng.int(88))),
                ],
            )
            .map(|_| ())
        }
        2 if nodes.len() >= 2 => {
            let src = NodeId::new(*rng.pick(&nodes).expect("nodes"));
            let dst = NodeId::new(*rng.pick(&nodes).expect("nodes"));
            let edge_type = ["KNOWS", "LIVES_IN"][rng.below(2)];
            tx.create_edge(
                src,
                dst,
                edge_type,
                &[("since", Value::Int64(1988 + rng.int(40)))],
            )
            .map(|_| ())
        }
        3 if !nodes.is_empty() => {
            let id = NodeId::new(*rng.pick(&nodes).expect("nodes"));
            match rng.below(3) {
                0 => tx.set_node(id, "name", Value::from(NAMES[rng.below(NAMES.len())])),
                1 => tx.set_node(id, "city", Value::from(CITIES[rng.below(CITIES.len())])),
                _ => tx.set_node(id, "age", Value::Int64(rng.int(88))),
            }
        }
        4 if !nodes.is_empty() => {
            let raw = *rng.pick(&nodes).expect("nodes");
            let keys: Vec<String> = view.nodes[&raw]
                .1
                .iter()
                .map(|(key, _)| key.clone())
                .collect();
            match rng.pick(&keys) {
                Some(key) => tx.remove_node_value(NodeId::new(raw), key),
                None => Ok(()),
            }
        }
        5 if !nodes.is_empty() => {
            let raw = *rng.pick(&nodes).expect("nodes");
            let has = &view.nodes[&raw].0;
            let label = LABELS[rng.below(LABELS.len())];
            if has.iter().any(|name| name == label) {
                tx.remove_label(NodeId::new(raw), label)
            } else {
                tx.add_label(NodeId::new(raw), label)
            }
        }
        6 if !edges.is_empty() => {
            let raw = *rng.pick(&edges).expect("edges");
            if rng.below(2) == 0 {
                tx.set_edge(EdgeId::new(raw), "since", Value::Int64(1988 + rng.int(40)))
            } else if view.edges[&raw].3.iter().any(|(key, _)| key == "since") {
                tx.remove_edge_value(EdgeId::new(raw), "since")
            } else {
                Ok(())
            }
        }
        7 if !edges.is_empty() => tx.delete_edge(EdgeId::new(*rng.pick(&edges).expect("edges"))),
        8 if !nodes.is_empty() => tx.detach_delete(NodeId::new(*rng.pick(&nodes).expect("nodes"))),
        _ => Ok(()),
    }
}
