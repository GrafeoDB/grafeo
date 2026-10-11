//! The writer with a recording: every write through the graph's change
//! target, each one that changed something recorded once in the
//! transaction's change set, after the claim of what it writes.

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use grafeo_common::change::{
    Before, BulkRange, Change, ChangeSet, DataModel, DataOp, GraphRef, GraphSlot, PendingVersion,
    Table,
};
use grafeo_common::storage::value_codec::MAX_PROPERTY_VALUE_DEPTH;
use grafeo_common::types::{ArcStr, EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use parking_lot::{Mutex, RwLock};

use super::tests::{BRIEFLY, CustomRules, every_write, labels, nested, pairs, people_store};
use super::{
    BulkRows, ChangeRecorder, GraphWriter, NewEdge, OperatorError, Recording, WriteClaim,
    WriteClaims, WriteInProgress, WriteTarget, node_labels, property_list,
};
use crate::graph::GraphStoreMut;
use crate::graph::apply::{Applied, ApplyError, ChangeTarget, ExternalTarget, Writer};
use crate::graph::lpg::{Edge, LpgStore, Node};

/// The transaction the tests write as.
const TRANSACTION: u64 = 19;

/// What a [`Recorder`] was asked, in order.
#[derive(Debug, Clone)]
enum Event {
    /// A claim, with the claimed node or edge as the transaction saw it
    /// then.
    Claim(WriteClaim, String),
    /// An entry recorded, with its entity as the transaction saw it right
    /// after the store changed.
    Record(DataOp, String),
}

/// A change recorder over one graph: a change set, the claims asked for,
/// and a write freeze that counts its requests.
struct Recorder {
    store: Arc<LpgStore>,
    writer: Writer,
    slot: GraphSlot,
    set: Mutex<ChangeSet>,
    events: Mutex<Vec<Event>>,
    freeze: RwLock<()>,
    requests: AtomicUsize,
    taken_already: AtomicBool,
    /// Refuses the claims of an edge's endpoints, as another transaction's
    /// delete of one would.
    refuse_endpoints: AtomicBool,
    /// Whether it takes bulk writes, and what it does with their rows.
    bulk: Option<BulkRows>,
}

impl Recorder {
    /// A recorder of transaction [`TRANSACTION`] in `store`'s default graph.
    fn new(store: &Arc<LpgStore>) -> Arc<Self> {
        Self::with_bulk(store, None)
    }

    /// A recorder as [`new`](Self::new) makes, which takes bulk writes as
    /// `bulk` says.
    fn with_bulk(store: &Arc<LpgStore>, bulk: Option<BulkRows>) -> Arc<Self> {
        let mut set = ChangeSet::new();
        let slot = set
            .slot(GraphRef {
                model: DataModel::Lpg,
                key: None,
            })
            .unwrap();
        Arc::new(Self {
            store: Arc::clone(store),
            writer: Writer::Transaction {
                id: TransactionId::new(TRANSACTION),
                snapshot: store.current_epoch(),
            },
            slot,
            set: Mutex::new(set),
            events: Mutex::new(Vec::new()),
            freeze: RwLock::new(()),
            requests: AtomicUsize::new(0),
            taken_already: AtomicBool::new(false),
            refuse_endpoints: AtomicBool::new(false),
            bulk,
        })
    }

    /// The ops the set holds for the commit's log, in order, with their
    /// before-images (rows kept with a bulk range included).
    fn ops(&self) -> Vec<(DataOp, Before)> {
        self.set
            .lock()
            .ops()
            .map(|(_, op, before)| (op.clone(), before.clone()))
            .collect()
    }

    /// The claims asked for, in order.
    fn claims(&self) -> Vec<WriteClaim> {
        self.events
            .lock()
            .iter()
            .filter_map(|event| match event {
                Event::Claim(claim, _) => Some(*claim),
                Event::Record(..) => None,
            })
            .collect()
    }

    /// The entries recorded, in order.
    fn entries(&self) -> Vec<Change> {
        self.set.lock().entries().to_vec()
    }

    /// The kinds of the entries recorded, in order.
    fn kinds(&self) -> Vec<&'static str> {
        self.entries().iter().map(kind).collect()
    }

    /// Node `id` as the transaction sees it.
    fn node_state(&self, id: NodeId) -> String {
        let Writer::Transaction { id: tx, snapshot } = self.writer else {
            unreachable!("a recorder writes as a transaction");
        };
        format!(
            "{:?}",
            self.store
                .get_node_versioned(id, snapshot, tx)
                .map(|node| describe_node(&node))
        )
    }

    /// Edge `id` as the transaction sees it.
    fn edge_state(&self, id: EdgeId) -> String {
        let Writer::Transaction { id: tx, snapshot } = self.writer else {
            unreachable!("a recorder writes as a transaction");
        };
        format!(
            "{:?}",
            self.store
                .get_edge_versioned(id, snapshot, tx)
                .map(|edge| describe_edge(&edge))
        )
    }
}

impl ChangeRecorder for Recorder {
    fn writer(&self) -> Writer {
        self.writer
    }

    fn record(
        &self,
        op: DataOp,
        before: Before,
        version: PendingVersion,
    ) -> Result<(), OperatorError> {
        let state = match op.entity() {
            Some(grafeo_common::change::Entity::Node(id)) => self.node_state(id),
            Some(grafeo_common::change::Entity::Edge(id)) => self.edge_state(id),
            None => String::new(),
        };
        self.events.lock().push(Event::Record(op.clone(), state));
        self.set
            .lock()
            .push(self.slot, op, before, version)
            .map_err(|error| OperatorError::Execution(error.to_string()))
    }

    fn bulk(&self) -> Option<BulkRows> {
        self.bulk
    }

    fn record_bulk(&self, table: Table, ids: Range<u64>) -> Result<(), OperatorError> {
        self.set
            .lock()
            .push_bulk(BulkRange {
                graph: self.slot,
                table,
                ids,
            })
            .map_err(|error| OperatorError::Execution(error.to_string()))
    }

    fn record_bulk_rows(&self, rows: Vec<DataOp>) -> Result<(), OperatorError> {
        self.set
            .lock()
            .push_bulk_rows(self.slot, rows)
            .map_err(|error| OperatorError::Execution(error.to_string()))
    }
}

impl WriteClaims for Recorder {
    fn claim(&self, claim: WriteClaim) -> Result<(), OperatorError> {
        if matches!(claim, WriteClaim::Endpoints(..))
            && self.refuse_endpoints.load(Ordering::SeqCst)
        {
            return Err(OperatorError::WriteConflict(
                "another transaction deletes an endpoint".to_string(),
            ));
        }
        let state = match claim {
            WriteClaim::Node(id) | WriteClaim::NodeDelete(id) => self.node_state(id),
            WriteClaim::Edge(id) => self.edge_state(id),
            WriteClaim::Endpoints(..) => String::new(),
        };
        self.events.lock().push(Event::Claim(claim, state));
        Ok(())
    }

    fn write_in_progress(&self) -> Option<WriteInProgress<'_>> {
        self.requests.fetch_add(1, Ordering::SeqCst);
        if self.freeze.is_locked() {
            self.taken_already.store(true, Ordering::SeqCst);
        }
        Some(self.freeze.read())
    }
}

/// A writer of transaction [`TRANSACTION`] in `store` through `target`,
/// recording in `recorder`.
fn writer_through(
    store: &Arc<LpgStore>,
    target: WriteTarget,
    recorder: &Arc<Recorder>,
) -> GraphWriter {
    GraphWriter::new(Arc::clone(store) as Arc<dyn GraphStoreMut>).with_recording(Recording {
        target,
        recorder: Arc::clone(recorder) as Arc<dyn ChangeRecorder>,
    })
}

/// A writer of transaction [`TRANSACTION`] in `store`, through the store's
/// own change target.
fn recording_writer(store: &Arc<LpgStore>, recorder: &Arc<Recorder>) -> GraphWriter {
    writer_through(
        store,
        WriteTarget::Store(Arc::clone(store) as Arc<dyn ChangeTarget>),
        recorder,
    )
}

/// A node's labels and properties, sorted.
fn describe_node(node: &Node) -> (Vec<String>, Vec<(String, Value)>) {
    let mut labels = node_labels(node);
    labels.sort();
    let mut properties = property_list(&node.properties);
    properties.sort_by(|a, b| a.0.cmp(&b.0));
    (labels, properties)
}

/// An edge's endpoints, type and properties, sorted.
fn describe_edge(edge: &Edge) -> (u64, u64, String, Vec<(String, Value)>) {
    let mut properties = property_list(&edge.properties);
    properties.sort_by(|a, b| a.0.cmp(&b.0));
    (
        edge.src.as_u64(),
        edge.dst.as_u64(),
        edge.edge_type.to_string(),
        properties,
    )
}

/// Every node and edge `reader` sees in `store`, with its labels, type,
/// endpoints and properties.
fn dump(store: &LpgStore, reader: &GraphWriter) -> Vec<String> {
    let mut lines = Vec::new();
    for raw in 0..store.next_node_id() {
        if let Some(node) = reader.node(NodeId::new(raw)) {
            lines.push(format!("node {raw} {:?}", describe_node(&node)));
        }
    }
    for raw in 0..store.next_edge_id() {
        if let Some(edge) = reader.edge(EdgeId::new(raw)) {
            lines.push(format!("edge {raw} {:?}", describe_edge(&edge)));
        }
    }
    lines
}

/// An entry's kind, by the op's name.
fn kind(change: &Change) -> &'static str {
    let Change::Data { op, .. } = change else {
        return "Bulk";
    };
    match op {
        DataOp::CreateNode { .. } => "CreateNode",
        DataOp::DeleteNode { .. } => "DeleteNode",
        DataOp::CreateEdge { .. } => "CreateEdge",
        DataOp::DeleteEdge { .. } => "DeleteEdge",
        DataOp::SetNodeProperty { .. } => "SetNodeProperty",
        DataOp::RemoveNodeProperty { .. } => "RemoveNodeProperty",
        DataOp::SetEdgeProperty { .. } => "SetEdgeProperty",
        DataOp::RemoveEdgeProperty { .. } => "RemoveEdgeProperty",
        DataOp::AddNodeLabel { .. } => "AddNodeLabel",
        DataOp::RemoveNodeLabel { .. } => "RemoveNodeLabel",
        DataOp::InsertTriple { .. } => "InsertTriple",
        DataOp::DeleteTriple { .. } => "DeleteTriple",
    }
}

/// The entries each write of [`every_write`] records, by kind.
fn expected_kinds(name: &str) -> &'static [&'static str] {
    match name {
        "create_node" => &["CreateNode"],
        "create_node_with" => &["CreateNode", "SetNodeProperty"],
        "set_node_properties" => &["SetNodeProperty"],
        "remove_node_property" => &["RemoveNodeProperty"],
        "add_labels" => &["AddNodeLabel"],
        "remove_labels" => &["RemoveNodeLabel"],
        "delete_node detaching" => &["DeleteEdge", "DeleteNode"],
        "delete_node" => &["DeleteNode"],
        "create_edge" => &["CreateEdge"],
        "create_edge_with" => &["CreateEdge", "SetEdgeProperty"],
        "set_edge_properties" => &["SetEdgeProperty"],
        "remove_edge_property" => &["RemoveEdgeProperty"],
        "delete_edge" => &["DeleteEdge"],
        other => panic!("no entries listed for {other}"),
    }
}

/// Every write method records one entry per store change, and the entries
/// hold exactly what it applied: replayed on the same graph they give what
/// the transaction sees after the write (nothing applied is missing from
/// them), and undone they give back what it saw before (nothing applied
/// escapes them).
#[test]
fn every_writer_method_records_exactly_what_it_applied() {
    for (name, write) in every_write() {
        let (store, people) = people_store();
        let recorder = Recorder::new(&store);
        let writer = recording_writer(&store, &recorder);
        let before = dump(&store, &writer);
        write(&writer, &people).unwrap_or_else(|error| panic!("{name}: {error}"));
        let after = dump(&store, &writer);
        assert_ne!(after, before, "{name} changes what the transaction sees");
        assert_eq!(recorder.kinds(), expected_kinds(name), "{name}");
        let entries = recorder.entries();

        let (replayed, _) = people_store();
        let epoch = EpochId::new(replayed.current_epoch().as_u64() + 1);
        for change in &entries {
            let Change::Data { op, .. } = change else {
                panic!("{name} recorded a bulk range");
            };
            let applied = replayed
                .apply(op, Writer::Replay { epoch })
                .unwrap_or_else(|error| panic!("{name}: replay of {op:?}: {error}"));
            assert_eq!(applied, Applied::Committed, "{name}");
        }
        let reader = GraphWriter::new(Arc::clone(&replayed) as Arc<dyn GraphStoreMut>);
        assert_eq!(
            dump(&replayed, &reader),
            after,
            "{name}: the entries redo exactly what it applied"
        );

        store
            .undo(TransactionId::new(TRANSACTION), &mut entries.iter())
            .unwrap_or_else(|error| panic!("{name}: undo: {error}"));
        assert_eq!(
            dump(&store, &writer),
            before,
            "{name}: undoing the entries undoes all it applied"
        );
    }
}

/// A store that refuses every op of one kind and is `LpgStore` otherwise.
struct Refusing {
    store: Arc<LpgStore>,
    kind: &'static str,
}

impl ChangeTarget for Refusing {
    fn reserve_node_ids(&self, count: u64) -> Result<std::ops::Range<u64>, ApplyError> {
        self.store.reserve_node_ids(count)
    }

    fn reserve_edge_ids(&self, count: u64) -> Result<std::ops::Range<u64>, ApplyError> {
        self.store.reserve_edge_ids(count)
    }

    fn apply(&self, op: &DataOp, writer: Writer) -> Result<Applied, ApplyError> {
        let change = Change::Data {
            graph: slot_zero(),
            op: op.clone(),
            before: Before::Absent,
            version: PendingVersion::Created,
        };
        if kind(&change) == self.kind {
            return Err(ApplyError::Refused(format!("no {} today", self.kind)));
        }
        self.store.apply(op, writer)
    }

    fn stamp(
        &self,
        transaction: TransactionId,
        entries: &mut dyn Iterator<Item = &Change>,
        epoch: EpochId,
    ) -> Result<(), ApplyError> {
        self.store.stamp(transaction, entries, epoch)
    }

    fn undo(
        &self,
        transaction: TransactionId,
        entries: &mut dyn DoubleEndedIterator<Item = &Change>,
    ) -> Result<(), ApplyError> {
        self.store.undo(transaction, entries)
    }
}

/// The slot of the default graph in a fresh change set.
fn slot_zero() -> GraphSlot {
    ChangeSet::new()
        .slot(GraphRef {
            model: DataModel::Lpg,
            key: None,
        })
        .unwrap()
}

/// A write the store refuses records nothing and changes nothing; one that
/// fails after earlier store changes of its own (a detach delete whose node
/// delete the store refuses) records exactly those, so undoing its entries
/// leaves nothing of it.
#[test]
fn a_refused_write_records_nothing() {
    for (name, write) in every_write() {
        let kinds = expected_kinds(name);
        for (refused, kept) in [(kinds[0], 0), (kinds[kinds.len() - 1], kinds.len() - 1)] {
            let (store, people) = people_store();
            let recorder = Recorder::new(&store);
            let target = Refusing {
                store: Arc::clone(&store),
                kind: refused,
            };
            let writer = writer_through(&store, WriteTarget::Store(Arc::new(target)), &recorder);
            let before = dump(&store, &writer);
            let error = write(&writer, &people).expect_err(name);
            assert!(
                error.to_string().contains(&format!("no {refused} today")),
                "{name}: the store's refusal: {error}"
            );
            assert_eq!(
                recorder.kinds(),
                kinds[..kept],
                "{name} refused at {refused}: only what applied is recorded"
            );
            store
                .undo(
                    TransactionId::new(TRANSACTION),
                    &mut recorder.entries().iter(),
                )
                .unwrap();
            assert_eq!(dump(&store, &writer), before, "{name} refused at {refused}");
        }
    }

    // What the checks refuse never reaches the store.
    let (store, people) = people_store();
    let recorder = Recorder::new(&store);
    let writer = recording_writer(&store, &recorder).with_validator(Arc::new(CustomRules));
    let before = dump(&store, &writer);
    let refused = [
        writer.create_edge(people.alix, people.vincent, "HATES", Vec::new()),
        writer.create_edge(
            people.alix,
            people.vincent,
            "KNOWS",
            pairs("grudge", &Value::from("Paris")),
        ),
        writer.create_edge(people.alix, people.vincent, "TRAVELS", Vec::new()),
    ];
    for (index, result) in refused.iter().enumerate() {
        assert!(
            matches!(result, Err(OperatorError::ConstraintViolation(_))),
            "edge {index}: {result:?}"
        );
    }
    let too_deep = nested(MAX_PROPERTY_VALUE_DEPTH + 1);
    assert!(
        writer
            .create_node(&labels(&["Person"]), Vec::new())
            .is_err(),
        "a default nested too deep"
    );
    assert!(
        writer
            .set_node_properties(people.alix, &pairs("trips", &too_deep), false)
            .is_err()
    );
    assert!(
        writer
            .create_edge(people.alix, NodeId::new(388), "KNOWS", Vec::new())
            .is_err(),
        "an endpoint that does not exist"
    );
    assert!(recorder.entries().is_empty(), "{:?}", recorder.kinds());
    assert_eq!(dump(&store, &writer), before);
}

/// A write that changes nothing records nothing: removing a value or a
/// label the entity lacks, adding a label it has, deleting what the
/// transaction deleted already. A set of the value a property holds is a
/// write, and is recorded.
#[test]
fn a_write_that_changes_nothing_records_nothing() {
    let (store, people) = people_store();
    let recorder = Recorder::new(&store);
    let writer = recording_writer(&store, &recorder);
    let unchanged = |what: &str, recorded: usize| {
        assert_eq!(recorder.entries().len(), recorded, "{what}");
    };

    assert!(!writer.remove_node_property(people.gus, "city").unwrap());
    unchanged("a removal of an absent value", 0);
    writer
        .set_node_properties(people.gus, &pairs("city", &Value::Null), false)
        .unwrap();
    unchanged("a null for an absent value", 0);
    assert!(!writer.remove_edge_property(people.knows, "until").unwrap());
    unchanged("an edge's absent value", 0);
    assert_eq!(
        writer
            .add_labels(people.alix, &labels(&["Person"]))
            .unwrap(),
        0
    );
    unchanged("a label the node has", 0);
    assert_eq!(
        writer
            .remove_labels(people.alix, &labels(&["Traveller"]))
            .unwrap(),
        0
    );
    unchanged("a label the node lacks", 0);

    assert!(writer.delete_edge(people.knows).unwrap());
    unchanged("the first delete of an edge", 1);
    assert!(!writer.delete_edge(people.knows).unwrap());
    unchanged("a delete of a deleted edge", 1);
    assert!(writer.delete_node(people.vincent, false).unwrap());
    unchanged("the first delete of a node", 2);
    assert!(!writer.delete_node(people.vincent, true).unwrap());
    unchanged("a delete of a deleted node", 2);

    writer
        .set_node_properties(
            people.alix,
            &pairs("city", &Value::from("Amsterdam")),
            false,
        )
        .unwrap();
    unchanged("a set of the value held", 3);
}

/// Every write claims what it changes before the store changes it: the
/// transaction saw the entity as it was before the write when it claimed
/// it. A node the transaction creates is claimed by nobody: no other
/// transaction knows its id.
#[test]
fn the_claim_precedes_the_store_change() {
    for (name, write) in every_write() {
        let (store, people) = people_store();
        let recorder = Recorder::new(&store);
        let writer = recording_writer(&store, &recorder);
        write(&writer, &people).unwrap_or_else(|error| panic!("{name}: {error}"));

        let events = recorder.events.lock().clone();
        let mut created = Vec::new();
        let mut checked = 0;
        for (at, event) in events.iter().enumerate() {
            let Event::Record(op, after) = event else {
                continue;
            };
            let entity = op.entity().expect("an LPG op");
            if let DataOp::CreateNode { id, .. } = op {
                created.push(*id);
                continue;
            }
            if matches!(entity, grafeo_common::change::Entity::Node(id) if created.contains(&id)) {
                continue;
            }
            let claimed = |claim: &WriteClaim| match (claim, entity) {
                (
                    WriteClaim::Node(id) | WriteClaim::NodeDelete(id),
                    grafeo_common::change::Entity::Node(node),
                ) => *id == node,
                (WriteClaim::Edge(id), grafeo_common::change::Entity::Edge(edge)) => *id == edge,
                _ => false,
            };
            let claim = events[..at]
                .iter()
                .rposition(|earlier| matches!(earlier, Event::Claim(claim, _) if claimed(claim)));
            let Some(claim) = claim else {
                panic!("{name}: {op:?} was not claimed");
            };
            let first_after_claim = !events[claim + 1..at].iter().any(|between| {
                matches!(between, Event::Record(other, _) if other.entity() == Some(entity))
            });
            if !first_after_claim {
                continue;
            }
            let Event::Claim(_, seen) = &events[claim] else {
                unreachable!("found as a claim");
            };
            assert_ne!(
                seen, after,
                "{name}: {op:?} changed the store before its claim"
            );
            checked += 1;
        }
        // A node create, and a value of it, claim nothing.
        if !matches!(name, "create_node" | "create_node_with") {
            assert!(checked > 0, "{name}: no write was checked");
        }
    }
}

/// An external store creates at the ids it gives itself: the writer records
/// the op with that id, as a create that replaced nothing, and claims the
/// new edge once its id is known. The transaction sees what it created.
#[test]
fn an_external_target_creates_at_the_ids_the_store_gives() {
    let (store, people) = people_store();
    let external = Arc::new(ExternalTarget::new(
        Arc::clone(&store) as Arc<dyn GraphStoreMut>
    ));
    let recorder = Recorder::new(&store);
    let writer = writer_through(
        &store,
        WriteTarget::External(Arc::clone(&external)),
        &recorder,
    );

    let (next_node, next_edge) = (store.next_node_id(), store.next_edge_id());
    let mia = writer
        .create_node(
            &labels(&["Person", "Person"]),
            pairs("name", &Value::from("Mia")),
        )
        .unwrap();
    let likes = writer
        .create_edge(people.alix, mia, "LIKES", pairs("since", &Value::Int64(88)))
        .unwrap();
    assert_eq!(
        (mia.as_u64(), likes.as_u64()),
        (next_node, next_edge),
        "the store's own next ids"
    );
    let person = || ArcStr::from("Person");
    assert_eq!(
        recorder.entries(),
        [
            Change::Data {
                graph: slot_zero(),
                op: DataOp::CreateNode {
                    id: mia,
                    labels: [person()].into_iter().collect(),
                    properties: vec![(PropertyKey::new("name"), Value::from("Mia"))],
                },
                before: Before::Absent,
                version: PendingVersion::Created,
            },
            Change::Data {
                graph: slot_zero(),
                op: DataOp::CreateEdge {
                    id: likes,
                    src: people.alix,
                    dst: mia,
                    edge_type: ArcStr::from("LIKES"),
                    properties: vec![(PropertyKey::new("since"), Value::Int64(88))],
                },
                before: Before::Absent,
                version: PendingVersion::Created,
            },
        ]
    );
    let claims: Vec<WriteClaim> = recorder
        .events
        .lock()
        .iter()
        .filter_map(|event| match event {
            Event::Claim(claim, _) => Some(*claim),
            Event::Record(..) => None,
        })
        .collect();
    assert_eq!(
        claims,
        [
            WriteClaim::Endpoints(people.alix, mia),
            WriteClaim::Edge(likes)
        ]
    );
    assert_eq!(
        writer.node(mia).map(|node| describe_node(&node)),
        Some((
            vec!["Person".to_string()],
            vec![("name".to_string(), Value::from("Mia"))]
        ))
    );
    assert!(store.get_node(mia).is_none(), "others do not see it yet");

    // An endpoint the writer does not see: nothing is created.
    let edges = store.next_edge_id();
    assert_eq!(
        external
            .create_edge(
                people.alix,
                NodeId::new(388),
                ArcStr::from("LIKES"),
                Vec::new(),
                recorder.writer(),
            )
            .map(|(op, _)| op),
        Err(ApplyError::Missing(grafeo_common::change::Entity::Node(
            NodeId::new(388)
        )))
    );
    assert_eq!(store.next_edge_id(), edges);
}

/// Every recorded write waits while the store is frozen (a checkpoint holds
/// the freeze) and changes nothing the transaction sees until it is
/// released; it asks for the freeze, and never again while it holds it.
#[test]
fn every_recorded_write_waits_for_the_freeze_and_asks_once() {
    for (name, write) in every_write() {
        let (store, people) = people_store();
        let recorder = Recorder::new(&store);
        let writer = recording_writer(&store, &recorder);
        let before = dump(&store, &writer);
        std::thread::scope(|scope| {
            let frozen = recorder.freeze.write();
            let (done, finished) = std::sync::mpsc::channel();
            let worker = {
                let (writer, people) = (&writer, &people);
                scope.spawn(move || {
                    let result = write(writer, people);
                    let _ = done.send(());
                    result
                })
            };
            assert!(
                finished.recv_timeout(BRIEFLY).is_err(),
                "{name} finished while the store was frozen"
            );
            assert_eq!(
                dump(&store, &writer),
                before,
                "{name} changed the store while it was frozen"
            );
            drop(frozen);
            worker
                .join()
                .unwrap()
                .unwrap_or_else(|error| panic!("{name}: {error}"));
        });
        assert!(recorder.requests.load(Ordering::SeqCst) > 0, "{name}");
        // Once frozen by the test, every request saw the freeze taken; ask
        // again on a fresh recorder for the reentrancy check.
        let (store, people) = people_store();
        let recorder = Recorder::new(&store);
        let writer = recording_writer(&store, &recorder);
        write(&writer, &people).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert!(
            !recorder.taken_already.load(Ordering::SeqCst),
            "{name} asked for the freeze while it held it"
        );
    }
}

/// The claims a recorder was asked for, in order.
fn claims(recorder: &Recorder) -> Vec<WriteClaim> {
    recorder
        .events
        .lock()
        .iter()
        .filter_map(|event| match event {
            Event::Claim(claim, _) => Some(*claim),
            Event::Record(..) => None,
        })
        .collect()
}

/// A new edge claims its endpoints and then itself before it is written,
/// also one MERGE creates, and a delete of a node is claimed as a delete,
/// then each edge a detach delete removes: the claims a concurrent delete
/// conflicts with.
#[test]
fn an_edge_claims_its_endpoints_and_a_delete_is_claimed_as_one() {
    let (store, people) = people_store();
    let recorder = Recorder::new(&store);
    let writer = recording_writer(&store, &recorder);
    let knows = writer
        .create_edge(people.vincent, people.gus, "KNOWS", Vec::new())
        .unwrap();
    let merged = writer
        .create_edge_with(people.gus, people.alix, "KNOWS", Vec::new(), |_| {
            Ok(Vec::new())
        })
        .unwrap();
    writer.delete_node(people.vincent, true).unwrap();

    assert_eq!(
        claims(&recorder),
        [
            WriteClaim::Endpoints(people.vincent, people.gus),
            WriteClaim::Edge(knows),
            WriteClaim::Endpoints(people.gus, people.alix),
            WriteClaim::Edge(merged),
            WriteClaim::NodeDelete(people.vincent),
            WriteClaim::Edge(knows),
        ]
    );
}

/// An edge the validator refuses claims no endpoint: a transaction that
/// deletes one of them later does not conflict with it.
#[test]
fn an_edge_the_validator_refuses_claims_no_endpoint() {
    let (store, people) = people_store();
    let recorder = Recorder::new(&store);
    let writer = recording_writer(&store, &recorder).with_validator(Arc::new(CustomRules));
    let grudge = || pairs("grudge", &Value::from("Paris"));

    let refused = [
        writer.create_edge(people.alix, people.vincent, "HATES", Vec::new()),
        writer.create_edge(people.alix, people.vincent, "KNOWS", grudge()),
        writer.create_edge_with(people.alix, people.vincent, "HATES", Vec::new(), |_| {
            Ok(Vec::new())
        }),
        writer.create_edge_with(people.alix, people.vincent, "KNOWS", grudge(), |_| {
            Ok(Vec::new())
        }),
    ];
    for (index, result) in refused.iter().enumerate() {
        assert!(
            matches!(result, Err(OperatorError::ConstraintViolation(_))),
            "edge {index}: {result:?}"
        );
    }
    assert!(recorder.entries().is_empty(), "no edge written");
    assert!(
        claims(&recorder).is_empty(),
        "a refused edge claims nothing: {:?}",
        claims(&recorder)
    );

    let knows = writer
        .create_edge(people.alix, people.vincent, "KNOWS", Vec::new())
        .unwrap();
    assert_eq!(
        claims(&recorder),
        [
            WriteClaim::Endpoints(people.alix, people.vincent),
            WriteClaim::Edge(knows),
        ],
        "an edge the validator accepts claims its endpoints"
    );
}

/// A refused claim, and an endpoint the transaction does not see, write
/// no edge.
#[test]
fn an_edge_whose_endpoints_cannot_be_claimed_is_not_written() {
    let (store, people) = people_store();
    let recorder = Recorder::new(&store);
    let writer = recording_writer(&store, &recorder);
    let before = dump(&store, &writer);
    recorder.refuse_endpoints.store(true, Ordering::SeqCst);
    let refused = writer.create_edge(people.alix, people.vincent, "KNOWS", Vec::new());
    assert!(
        matches!(refused, Err(OperatorError::WriteConflict(_))),
        "got {refused:?}"
    );
    assert_eq!(recorder.kinds(), Vec::<&str>::new());
    assert_eq!(dump(&store, &writer), before);
    recorder.refuse_endpoints.store(false, Ordering::SeqCst);

    writer.delete_node(people.vincent, false).unwrap();
    let deleted = writer.create_edge(people.alix, people.vincent, "KNOWS", Vec::new());
    assert!(deleted.is_err(), "an endpoint the transaction deleted");
    let missing = writer.create_edge(people.alix, NodeId::new(388), "KNOWS", Vec::new());
    assert!(missing.is_err(), "an endpoint that does not exist");
    assert_eq!(recorder.kinds(), ["DeleteNode"], "no edge written");
}

/// The expressions `derive` evaluates run with no write in progress: a
/// checkpoint waiting meanwhile does not wait for them.
#[test]
fn derive_runs_with_no_write_in_progress() {
    let (store, _) = people_store();
    let recorder = Recorder::new(&store);
    let writer = recording_writer(&store, &recorder);
    let frozen_in_derive = AtomicBool::new(true);
    writer
        .create_node_with(&labels(&["Person"]), Vec::new(), |_| {
            frozen_in_derive.store(recorder.freeze.is_locked(), Ordering::SeqCst);
            Ok(pairs("name", &Value::from("Butch")))
        })
        .unwrap();
    assert!(
        !frozen_in_derive.load(Ordering::SeqCst),
        "derive runs with no write in progress"
    );
    assert_eq!(recorder.kinds(), ["CreateNode", "SetNodeProperty"]);
}

// === Bulk creates ===

/// The rows of the bulk creates in [`bulk_writes`]: three cities, then
/// edges among the people and the cities, some sharing endpoints.
fn bulk_writes(writer: &GraphWriter, people: &super::tests::People) -> (Vec<NodeId>, Vec<EdgeId>) {
    let cities = writer
        .create_nodes(
            &labels(&["City", "Place"]),
            vec![
                pairs("name", &Value::from("Amsterdam")),
                pairs("name", &Value::from("Berlin")),
                Vec::new(),
            ],
        )
        .unwrap();
    let visit = |src: NodeId, dst: NodeId, year: i64| NewEdge {
        src,
        dst,
        edge_type: "VISITED".to_string(),
        properties: pairs("year", &Value::Int64(year)),
    };
    let edges = writer
        .create_edges(vec![
            visit(people.alix, cities[0], 1988),
            visit(people.alix, cities[1], 2019),
            visit(people.gus, cities[0], 2003),
            visit(people.alix, cities[0], 2019),
            NewEdge {
                src: cities[1],
                dst: cities[2],
                edge_type: "ROUTE".to_string(),
                properties: Vec::new(),
            },
        ])
        .unwrap();
    (cities, edges)
}

/// A bulk create writes what one create per row writes, the same nodes and
/// edges with the same ids and values, but records one entry per call: the
/// range of ids it reserved, which undo reads to take every row back. The
/// rows are kept with the range for the commit when the recorder asks for
/// them, as the very ops one create per row records; otherwise none.
#[test]
fn a_bulk_create_writes_what_one_create_per_row_writes_as_one_entry() {
    for keep in [BulkRows::Keep, BulkRows::Drop] {
        let (store, people) = people_store();
        let recorder = Recorder::with_bulk(&store, Some(keep));
        let writer = recording_writer(&store, &recorder);
        let before = dump(&store, &writer);
        let (row_store, _) = people_store();
        let rows = Recorder::new(&row_store);
        let row_writer = recording_writer(&row_store, &rows);

        let created = bulk_writes(&writer, &people);
        assert_eq!(bulk_writes(&row_writer, &people), created, "the same ids");
        assert_eq!(
            dump(&store, &writer),
            dump(&row_store, &row_writer),
            "{keep:?}: the same graph"
        );
        assert_eq!(recorder.kinds(), ["Bulk", "Bulk"], "{keep:?}");
        assert_eq!(
            rows.kinds(),
            [["CreateNode"; 3].as_slice(), ["CreateEdge"; 5].as_slice()].concat()
        );
        match keep {
            BulkRows::Keep => assert_eq!(recorder.ops(), rows.ops(), "the rows kept"),
            BulkRows::Drop => assert!(recorder.ops().is_empty(), "no row kept"),
        }

        store
            .undo(
                TransactionId::new(TRANSACTION),
                &mut recorder.entries().iter(),
            )
            .unwrap();
        assert_eq!(
            dump(&store, &writer),
            before,
            "{keep:?}: the ranges undo every row"
        );
    }
}

/// A bulk create of edges claims each endpoint once per call, before the
/// first edge that names it is written, and none of its new edges: their
/// ids are the call's reserved range, which no other transaction can name,
/// and a concurrent delete of an endpoint meets the endpoint's claim. Its
/// store changes run as writes in progress, the freeze asked for once per
/// run of rows and never while held.
#[test]
fn a_bulk_create_claims_each_endpoint_once_and_no_new_edge() {
    let (store, people) = people_store();
    let recorder = Recorder::with_bulk(&store, Some(BulkRows::Drop));
    let writer = recording_writer(&store, &recorder);
    let (cities, _) = bulk_writes(&writer, &people);
    assert_eq!(
        recorder.claims(),
        [
            WriteClaim::Endpoints(people.alix, cities[0]),
            WriteClaim::Endpoints(people.alix, cities[1]),
            WriteClaim::Endpoints(people.gus, cities[0]),
            WriteClaim::Endpoints(cities[1], cities[2]),
        ]
    );
    assert_eq!(
        recorder.requests.load(Ordering::SeqCst),
        2,
        "one write in progress per call of fewer rows than a run"
    );
    assert!(!recorder.taken_already.load(Ordering::SeqCst));
}

/// A bulk create stops at the first row a check refuses (an endpoint the
/// transaction does not see, a constraint, a claim another transaction
/// holds), as one create of that row would: the rows before it stay
/// written, and the range recorded before the first row undoes them.
#[test]
fn a_refused_row_stops_a_bulk_create_and_its_range_undoes_the_rows_before_it() {
    let (store, people) = people_store();
    let missing = NodeId::new(388);
    let visit = |src: NodeId, dst: NodeId, edge_type: &str| NewEdge {
        src,
        dst,
        edge_type: edge_type.to_string(),
        properties: Vec::new(),
    };
    type Refusal = (
        &'static str,
        fn(&Recorder),
        Vec<(NodeId, NodeId, &'static str)>,
    );
    let refusals: Vec<Refusal> = vec![
        (
            "an endpoint that does not exist",
            |_| {},
            vec![
                (people.alix, people.gus, "KNOWS"),
                (people.alix, missing, "KNOWS"),
            ],
        ),
        (
            "a constraint",
            |_| {},
            vec![
                (people.alix, people.gus, "KNOWS"),
                (people.gus, people.vincent, "HATES"),
            ],
        ),
        (
            "a claim another transaction holds",
            |recorder| recorder.refuse_endpoints.store(true, Ordering::SeqCst),
            vec![(people.alix, people.gus, "KNOWS")],
        ),
    ];
    for (what, prepare, edges) in refusals {
        let recorder = Recorder::with_bulk(&store, Some(BulkRows::Keep));
        let writer =
            recording_writer(&store, &recorder).with_validator(Arc::new(super::tests::CustomRules));
        let before = dump(&store, &writer);
        prepare(&recorder);
        let refused = writer.create_edges(
            edges
                .iter()
                .map(|(src, dst, edge_type)| visit(*src, *dst, edge_type))
                .collect(),
        );
        assert!(refused.is_err(), "{what}: {refused:?}");
        let entries = recorder.entries();
        assert_eq!(
            recorder.kinds(),
            ["Bulk"],
            "{what}: the range is recorded first"
        );
        assert!(
            recorder.ops().is_empty(),
            "{what}: a refused call keeps no row"
        );
        store
            .undo(TransactionId::new(TRANSACTION), &mut entries.iter())
            .unwrap();
        assert_eq!(dump(&store, &writer), before, "{what}: the range undoes it");
    }
}

/// A recorder that takes no bulk writes (an immediate write) gets one
/// create per row, each recorded as one would be.
#[test]
fn a_recorder_without_bulk_writes_gets_one_create_per_row() {
    let (store, people) = people_store();
    let recorder = Recorder::new(&store);
    let writer = recording_writer(&store, &recorder);
    bulk_writes(&writer, &people);
    assert!(
        recorder
            .kinds()
            .iter()
            .all(|kind| kind.starts_with("Create"))
    );
    assert_eq!(recorder.kinds().len(), 8);
}
