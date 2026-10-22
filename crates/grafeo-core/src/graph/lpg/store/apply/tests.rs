//! The store side of the change set: apply, stamp and undo, against what
//! readers see, in every build (default, `temporal`, `tiered-storage`).

use std::collections::BTreeMap;

use grafeo_common::change::{
    Before, BulkRange, Change, ChangeSet, DataModel, DataOp, Entity, GraphRef, GraphSlot,
    PendingVersion, Table,
};
use grafeo_common::storage::log_record::LogRecordRef;
use grafeo_common::types::{ArcStr, EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};

use super::testing::{Recorder, labels, properties, transaction};
use crate::graph::apply::{Applied, ApplyError, ChangeTarget, Writer};
use crate::graph::lpg::LpgStore;

/// A change set's slot for the default labeled property graph.
fn default_slot(set: &mut ChangeSet) -> GraphSlot {
    set.slot(GraphRef {
        model: DataModel::Lpg,
        key: None,
    })
    .unwrap()
}

/// A recorder writing `store` as transaction `id` at the store's epoch,
/// with a new change set.
fn begin(store: &LpgStore, id: u64) -> (Recorder<'_>, ChangeSet) {
    let mut set = ChangeSet::new();
    let slot = default_slot(&mut set);
    (
        Recorder {
            store,
            slot,
            writer: transaction(id, store.current_epoch()),
        },
        set,
    )
}

/// Commits `set` of `transaction` at the store's next epoch.
fn commit(store: &LpgStore, transaction: TransactionId, set: &ChangeSet) -> EpochId {
    let epoch = EpochId::new(store.current_epoch().as_u64() + 1);
    store
        .stamp(transaction, &mut set.entries().iter(), epoch)
        .unwrap();
    epoch
}

/// Undoes the entries of `set` after `mark`, as a rollback to a savepoint.
fn roll_back_to(
    store: &LpgStore,
    transaction: TransactionId,
    set: &mut ChangeSet,
    mark: grafeo_common::change::ChangeMark,
) {
    let tail = set.split_off(mark);
    store.undo(transaction, &mut tail.iter()).unwrap();
}

/// A node as a reader sees it: its labels and values, sorted.
type NodeView = (Vec<String>, Vec<(String, Value)>);

/// What a reader sees of a store: as a transaction (its own writes
/// included), or as a committed reader at an epoch; with the label index
/// and the property index on "name".
#[derive(Debug, PartialEq)]
struct Image {
    nodes: BTreeMap<u64, NodeView>,
    edges: BTreeMap<u64, (u64, u64, String, Vec<(String, Value)>)>,
    by_label: Vec<(String, Vec<u64>)>,
    by_name: Vec<(String, Option<Vec<u64>>)>,
}

/// A reader's view: `Some((transaction, snapshot))`, or the committed state
/// at the store's epoch.
type Reader = Option<(TransactionId, EpochId)>;

fn sorted_values(values: &grafeo_common::types::PropertyMap) -> Vec<(String, Value)> {
    values
        .to_btree_map()
        .into_iter()
        .map(|(key, value)| (key.as_str().to_string(), value))
        .collect()
}

fn image(store: &LpgStore, reader: Reader) -> Image {
    let mut nodes = BTreeMap::new();
    for raw in 0..store.next_node_id() {
        let id = NodeId::new(raw);
        let node = match reader {
            Some((transaction, snapshot)) => store.get_node_versioned(id, snapshot, transaction),
            None => store.get_node_at_epoch(id, store.current_epoch()),
        };
        if let Some(node) = node {
            let mut names: Vec<String> = node.labels.iter().map(ToString::to_string).collect();
            names.sort();
            nodes.insert(raw, (names, sorted_values(&node.properties)));
        }
    }
    let mut edges = BTreeMap::new();
    for raw in 0..store.next_edge_id() {
        let id = EdgeId::new(raw);
        let edge = match reader {
            Some((transaction, snapshot)) => store.get_edge_versioned(id, snapshot, transaction),
            None => store.get_edge_at_epoch(id, store.current_epoch()),
        };
        if let Some(edge) = edge {
            edges.insert(
                raw,
                (
                    edge.src.as_u64(),
                    edge.dst.as_u64(),
                    edge.edge_type.to_string(),
                    sorted_values(&edge.properties),
                ),
            );
        }
    }
    let by_label = ["Person", "Employee", "Traveller", "City"]
        .iter()
        .map(|label| {
            let ids = store
                .nodes_by_label(label)
                .into_iter()
                .map(|id| id.as_u64())
                .collect();
            ((*label).to_string(), ids)
        })
        .collect();
    let by_name = ["Alix", "Gus", "Mia", "Vincent", "Jules"]
        .iter()
        .map(|name| {
            let ids = store
                .find_nodes_maybe_equal("name", &Value::from(*name))
                .map(|ids| {
                    let mut ids: Vec<u64> = ids.into_iter().map(|id| id.as_u64()).collect();
                    ids.sort_unstable();
                    ids
                });
            ((*name).to_string(), ids)
        })
        .collect();
    Image {
        nodes,
        edges,
        by_label,
        by_name,
    }
}

/// The statistics counters: live nodes, live edges, live edges per type.
fn counters(store: &LpgStore) -> (i64, i64, BTreeMap<String, i64>) {
    use std::sync::atomic::Ordering;
    let counts = store.edge_type_live_counts.read().clone();
    let per_type = store
        .all_edge_types()
        .into_iter()
        .filter_map(|name| {
            let id = store.edge_type_id(&name)? as usize;
            let count = counts.get(id).copied().unwrap_or(0);
            (count != 0).then_some((name, count))
        })
        .collect();
    (
        store.live_node_count.load(Ordering::Relaxed),
        store.live_edge_count.load(Ordering::Relaxed),
        per_type,
    )
}

/// Alix and Gus (and Mia) know each other, committed through replay at
/// epoch 1, with a property index on "name".
struct Base {
    store: LpgStore,
    alix: NodeId,
    gus: NodeId,
    mia: NodeId,
    alix_gus: EdgeId,
    gus_mia: EdgeId,
}

fn base() -> Base {
    let store = LpgStore::new().unwrap();
    store.create_property_index("name");
    let replay = Writer::Replay {
        epoch: EpochId::new(1),
    };
    let mut set = ChangeSet::new();
    let slot = default_slot(&mut set);
    let recorder = Recorder {
        store: &store,
        slot,
        writer: replay,
    };
    let alix = recorder.create_node(
        &mut set,
        &["Person"],
        &[
            ("name", Value::from("Alix")),
            ("age", Value::Int64(19)),
            ("city", Value::from("Amsterdam")),
        ],
    );
    let gus = recorder.create_node(
        &mut set,
        &["Person", "Employee"],
        &[("name", Value::from("Gus")), ("age", Value::Int64(3))],
    );
    let mia = recorder.create_node(&mut set, &["Person"], &[("name", Value::from("Mia"))]);
    let alix_gus = recorder.create_edge(
        &mut set,
        alix,
        gus,
        "KNOWS",
        &[("since", Value::Int64(1988))],
    );
    let gus_mia =
        recorder.create_edge(&mut set, gus, mia, "KNOWS", &[("since", Value::Int64(319))]);
    assert!(
        set.is_empty(),
        "replay records nothing: {:?}",
        set.entries()
    );
    Base {
        store,
        alix,
        gus,
        mia,
        alix_gus,
        gus_mia,
    }
}

/// Undo restores what each kind of op changed: a rollback to each savepoint
/// gives back what the transaction saw there (its own writes included, the
/// label and property indexes too), and a full rollback the committed state.
/// A delete of a node the transaction created reports a replaced version and
/// is undone too.
#[test]
fn undo_restores_every_op_kind_and_savepoint_tails() {
    let base = base();
    let store = &base.store;
    let committed = image(store, None);
    let counts = counters(store);
    let id = TransactionId::new(19);
    let reader = Some((id, store.current_epoch()));
    let (recorder, mut set) = begin(store, 19);

    let at_start = set.mark();
    let seen_at_start = image(store, reader);
    let vincent = recorder.create_node(
        &mut set,
        &["Person"],
        &[
            ("name", Value::from("Vincent")),
            ("city", Value::from("Paris")),
        ],
    );
    let vincent_mia = recorder.create_edge(
        &mut set,
        vincent,
        base.mia,
        "KNOWS",
        &[("since", Value::Int64(3))],
    );
    recorder.set_node(&mut set, base.alix, "age", Value::Int64(88));
    recorder.remove_node(&mut set, base.alix, "city");
    recorder.set_node(&mut set, base.mia, "name", Value::from("Jules"));
    recorder.add_label(&mut set, base.mia, "Traveller");
    recorder.remove_label(&mut set, base.gus, "Employee");

    let first = set.mark();
    let seen_at_first = image(store, reader);
    recorder.set_edge(&mut set, base.alix_gus, "since", Value::Int64(19));
    recorder.remove_edge(&mut set, base.gus_mia, "since");
    recorder.delete_edge(&mut set, base.alix_gus);
    recorder.delete_edge(&mut set, base.gus_mia);
    recorder.delete_node(&mut set, base.gus);
    recorder.delete_edge(&mut set, vincent_mia);
    recorder.delete_node(&mut set, vincent);

    let second = set.mark();
    let seen_at_second = image(store, reader);
    recorder.set_node(&mut set, base.alix, "name", Value::from("Mia"));
    recorder.add_label(&mut set, base.alix, "Traveller");
    let visited = recorder.create_edge(&mut set, base.alix, base.mia, "VISITED", &[]);
    recorder.set_edge(&mut set, visited, "since", Value::Int64(88));

    let kinds: std::collections::BTreeSet<u8> = set
        .entries()
        .iter()
        .filter_map(|change| match change {
            Change::Data { op, .. } => Some(op.kind()),
            Change::Bulk(_) => None,
        })
        .collect();
    assert_eq!(kinds.len(), 10, "every op kind of a labeled property graph");
    assert!(
        set.entries().iter().any(|change| matches!(
            change,
            Change::Data {
                op: DataOp::DeleteNode { .. },
                version: PendingVersion::Replaced,
                ..
            }
        )),
        "the delete of a node the transaction created replaced its version"
    );
    assert_eq!(
        counters(store),
        counts,
        "a transaction's writes count at commit"
    );

    roll_back_to(store, id, &mut set, second);
    assert_eq!(image(store, reader), seen_at_second, "the second savepoint");
    roll_back_to(store, id, &mut set, first);
    assert_eq!(image(store, reader), seen_at_first, "the first savepoint");
    roll_back_to(store, id, &mut set, at_start);
    assert_eq!(image(store, reader), seen_at_start, "the start");
    assert_eq!(image(store, None), committed, "the committed state");
    assert_eq!(counters(store), counts, "the counters");
}

/// A transaction sets a value, takes a savepoint, sets it twice more and
/// rolls back to the savepoint: it reads its own value from before the
/// savepoint, and after the commit every reader does.
#[test]
fn a_savepoint_rollback_keeps_the_transactions_own_value() {
    let base = base();
    let store = &base.store;
    let x = PropertyKey::new("x");
    let id = TransactionId::new(3);
    let snapshot = store.current_epoch();
    let (recorder, mut set) = begin(store, 3);
    recorder.set_node(&mut set, base.alix, "x", Value::Int64(3));
    let savepoint = set.mark();
    recorder.set_node(&mut set, base.alix, "x", Value::Int64(19));
    recorder.set_node(&mut set, base.alix, "x", Value::Int64(88));
    roll_back_to(store, id, &mut set, savepoint);

    let own = store.get_node_versioned(base.alix, snapshot, id).unwrap();
    assert_eq!(
        own.get_property("x"),
        Some(&Value::Int64(3)),
        "its own value"
    );
    let epoch = commit(store, id, &set);
    let committed = store.get_node_at_epoch(base.alix, epoch).unwrap();
    assert_eq!(
        committed.get_property("x"),
        Some(&Value::Int64(3)),
        "the committed value"
    );
    #[cfg(feature = "temporal")]
    {
        assert_eq!(
            store.get_node_property_at_epoch(base.alix, &x, epoch),
            Some(Value::Int64(3))
        );
        assert_eq!(
            store.get_node_property_at_epoch(base.alix, &x, snapshot),
            None,
            "no value before the commit"
        );
    }
    #[cfg(not(feature = "temporal"))]
    let _ = x;
}

/// A commit stamps the entities its entries name and no other: another open
/// transaction's create, values and labels stay its own; its rollback then
/// leaves the committed transaction's writes alone.
#[test]
fn stamp_and_undo_touch_only_listed_entities() {
    let base = base();
    let store = &base.store;
    let snapshot = store.current_epoch();
    let (first, mut first_set) = begin(store, 3);
    let (second, mut second_set) = begin(store, 19);
    let vincent = first.create_node(
        &mut first_set,
        &["Person"],
        &[("name", Value::from("Vincent"))],
    );
    first.set_node(&mut first_set, base.alix, "age", Value::Int64(88));
    first.add_label(&mut first_set, base.alix, "Traveller");
    let jules = second.create_node(
        &mut second_set,
        &["Person"],
        &[("name", Value::from("Jules"))],
    );
    second.set_node(&mut second_set, base.gus, "age", Value::Int64(19));
    second.add_label(&mut second_set, base.gus, "Traveller");
    second.delete_edge(&mut second_set, base.gus_mia);

    let epoch = commit(store, TransactionId::new(3), &first_set);
    assert!(
        store.is_node_visible_at_epoch(vincent, epoch),
        "the commit's create"
    );
    assert!(
        !store.is_node_visible_at_epoch(jules, epoch),
        "the open transaction's create stays its own"
    );
    assert!(
        store.is_node_visible_versioned(jules, snapshot, TransactionId::new(19)),
        "and it still sees it"
    );
    #[cfg(feature = "temporal")]
    {
        let age = PropertyKey::new("age");
        assert_eq!(
            store.get_node_property_at_epoch(base.alix, &age, epoch),
            Some(Value::Int64(88)),
            "the committed value"
        );
        assert_eq!(
            store.get_node_property_at_epoch(base.gus, &age, epoch),
            Some(Value::Int64(3)),
            "the open transaction's value stays pending"
        );
        let labels_at = |id: NodeId| {
            let mut names: Vec<String> = store
                .node_with_labels_at(id, epoch)
                .labels
                .iter()
                .map(ToString::to_string)
                .collect();
            names.sort();
            names
        };
        assert_eq!(labels_at(base.alix), ["Person", "Traveller"]);
        assert_eq!(labels_at(base.gus), ["Employee", "Person"]);
    }

    store
        .undo(TransactionId::new(19), &mut second_set.entries().iter())
        .unwrap();
    assert!(
        store.get_node(jules).is_none(),
        "the rolled back create is gone"
    );
    let alix = store.get_node(base.alix).unwrap();
    assert_eq!(alix.get_property("age"), Some(&Value::Int64(88)));
    assert!(alix.has_label("Traveller"), "the committed label stays");
    let gus = store.get_node(base.gus).unwrap();
    assert_eq!(gus.get_property("age"), Some(&Value::Int64(3)));
    assert!(!gus.has_label("Traveller"));
    assert!(
        store.get_edge(base.gus_mia).is_some(),
        "the rolled back delete"
    );
    assert!(
        store.get_node(vincent).is_some(),
        "the committed create stays"
    );
}

/// A transaction's creates and deletes move the counters when it commits,
/// not when it writes, and a rollback leaves them as they were.
#[test]
fn counters_move_at_commit_only() {
    let base = base();
    let store = &base.store;
    let (nodes, edges, per_type) = counters(store);
    assert_eq!((nodes, edges), (3, 2), "the base, counted at replay");
    assert_eq!(per_type.get("KNOWS"), Some(&2));

    let (recorder, mut set) = begin(store, 3);
    let vincent = recorder.create_node(&mut set, &["Person"], &[]);
    let jules = recorder.create_node(&mut set, &["Person"], &[]);
    recorder.create_edge(&mut set, vincent, jules, "VISITED", &[]);
    recorder.delete_edge(&mut set, base.gus_mia);
    recorder.delete_node(&mut set, base.mia);
    assert_eq!(
        counters(store),
        (nodes, edges, per_type.clone()),
        "nothing moves at write time"
    );
    store.ensure_statistics_fresh();
    assert_eq!(
        store.statistics().total_nodes,
        3,
        "statistics count commits"
    );

    commit(store, TransactionId::new(3), &set);
    let (after_nodes, after_edges, after_types) = counters(store);
    assert_eq!(
        (after_nodes, after_edges),
        (4, 2),
        "two creates, a delete; one of each"
    );
    assert_eq!(after_types.get("KNOWS"), Some(&1));
    assert_eq!(after_types.get("VISITED"), Some(&1));
    store.ensure_statistics_fresh();
    assert_eq!(store.statistics().total_nodes, 4);
    assert_eq!(store.statistics().total_edges, 2);

    let before_rollback = counters(store);
    let (recorder, mut set) = begin(store, 19);
    let butch = recorder.create_node(&mut set, &["Person"], &[]);
    recorder.create_edge(&mut set, butch, base.alix, "KNOWS", &[]);
    recorder.delete_edge(&mut set, base.alix_gus);
    store
        .undo(TransactionId::new(19), &mut set.entries().iter())
        .unwrap();
    assert_eq!(counters(store), before_rollback, "a rollback moves nothing");
}

/// Replay counts what it applies, as the live commit did: the counters
/// after replaying a commit's log records equal the live ones.
#[test]
fn replay_counts_as_the_live_commit_did() {
    let live = LpgStore::new().unwrap();
    let (recorder, mut set) = begin(&live, 3);
    let alix = recorder.create_node(&mut set, &["Person"], &[("name", Value::from("Alix"))]);
    let gus = recorder.create_node(&mut set, &["Person"], &[("name", Value::from("Gus"))]);
    let prague = recorder.create_node(&mut set, &["City"], &[("name", Value::from("Prague"))]);
    let knows = recorder.create_edge(&mut set, alix, gus, "KNOWS", &[]);
    recorder.create_edge(&mut set, alix, prague, "VISITED", &[]);
    let gus_prague = recorder.create_edge(&mut set, gus, prague, "VISITED", &[]);
    recorder.delete_edge(&mut set, knows);
    recorder.delete_edge(&mut set, gus_prague);
    recorder.delete_node(&mut set, gus);
    let epoch = commit(&live, TransactionId::new(3), &set);

    let replayed = LpgStore::new().unwrap();
    for record in set.log_records() {
        let LogRecordRef::Data { op, .. } = record else {
            panic!("a change set's records are data records");
        };
        assert_eq!(
            replayed.apply(op, Writer::Replay { epoch }),
            Ok(Applied::Committed)
        );
    }
    assert_eq!(counters(&replayed), counters(&live));
    let (nodes, edges, per_type) = counters(&live);
    assert_eq!((nodes, edges), (2, 1));
    assert_eq!(per_type.get("VISITED"), Some(&1));
    assert_eq!(per_type.get("KNOWS"), None);
    live.ensure_statistics_fresh();
    replayed.ensure_statistics_fresh();
    assert_eq!(
        (
            replayed.statistics().total_nodes,
            replayed.statistics().total_edges
        ),
        (live.statistics().total_nodes, live.statistics().total_edges)
    );
    assert_eq!(replayed.current_epoch(), epoch, "replay stamps its epoch");
}

/// Replay is strict: a create at an id in use, a missing entity, a node
/// delete with edges, removing an absent value or label and adding a present
/// label are errors, and nothing changes. A transaction's write that would
/// change nothing changes nothing and is not recorded.
#[test]
fn strict_apply_refuses_a_used_id_and_a_missing_entity() {
    let base = base();
    let store = &base.store;
    let replay = Writer::Replay {
        epoch: EpochId::new(2),
    };
    let missing = NodeId::new(88);
    let key = PropertyKey::new("nickname");
    let refused: Vec<(DataOp, ApplyError)> = vec![
        (
            DataOp::CreateNode {
                id: base.alix,
                labels: labels(&["Person"]),
                properties: properties(&[]),
            },
            ApplyError::Exists(Entity::Node(base.alix)),
        ),
        (
            DataOp::CreateEdge {
                id: base.alix_gus,
                src: base.gus,
                dst: base.mia,
                edge_type: ArcStr::from("KNOWS"),
                properties: properties(&[]),
            },
            ApplyError::Exists(Entity::Edge(base.alix_gus)),
        ),
        (
            DataOp::CreateEdge {
                id: EdgeId::new(88),
                src: base.alix,
                dst: missing,
                edge_type: ArcStr::from("KNOWS"),
                properties: properties(&[]),
            },
            ApplyError::Missing(Entity::Node(missing)),
        ),
        (
            DataOp::SetNodeProperty {
                id: missing,
                key: key.clone(),
                value: Value::Int64(3),
            },
            ApplyError::Missing(Entity::Node(missing)),
        ),
        (
            DataOp::DeleteEdge {
                id: EdgeId::new(88),
            },
            ApplyError::Missing(Entity::Edge(EdgeId::new(88))),
        ),
        (
            DataOp::DeleteNode { id: base.gus },
            ApplyError::HasEdges(base.gus),
        ),
        (
            DataOp::DeleteNode { id: base.mia },
            ApplyError::HasEdges(base.mia),
        ),
        (
            DataOp::RemoveNodeProperty {
                id: base.alix,
                key: key.clone(),
            },
            ApplyError::Missing(Entity::Node(base.alix)),
        ),
        (
            DataOp::RemoveEdgeProperty {
                id: base.alix_gus,
                key: key.clone(),
            },
            ApplyError::Missing(Entity::Edge(base.alix_gus)),
        ),
        (
            DataOp::AddNodeLabel {
                id: base.alix,
                label: ArcStr::from("Person"),
            },
            ApplyError::Exists(Entity::Node(base.alix)),
        ),
        (
            DataOp::RemoveNodeLabel {
                id: base.alix,
                label: ArcStr::from("Employee"),
            },
            ApplyError::Missing(Entity::Node(base.alix)),
        ),
    ];
    let before = image(store, None);
    let counts = counters(store);
    for (op, error) in &refused {
        assert_eq!(
            store.apply(op, replay).as_ref(),
            Err(error),
            "replay of {op:?}"
        );
        assert_eq!(image(store, None), before, "nothing changed after {op:?}");
        assert_eq!(counters(store), counts, "no counter moved after {op:?}");
    }

    // A transaction: a used id and a missing endpoint are errors too.
    let (recorder, mut set) = begin(store, 3);
    for (op, error) in refused.iter().take(3) {
        assert_eq!(recorder.write(&mut set, op.clone()).as_ref(), Err(error));
    }
    // The writes that change nothing are not recorded.
    let unchanged = [
        DataOp::DeleteNode { id: missing },
        DataOp::SetNodeProperty {
            id: missing,
            key: key.clone(),
            value: Value::Int64(3),
        },
        DataOp::RemoveNodeProperty {
            id: base.alix,
            key: key.clone(),
        },
        DataOp::AddNodeLabel {
            id: base.alix,
            label: ArcStr::from("Person"),
        },
        DataOp::RemoveNodeLabel {
            id: base.alix,
            label: ArcStr::from("Employee"),
        },
    ];
    for op in unchanged {
        assert_eq!(
            recorder.write(&mut set, op.clone()),
            Ok(Applied::Unchanged),
            "{op:?}"
        );
    }
    assert!(set.is_empty(), "{:?}", set.entries());
    assert_eq!(image(store, None), before);

    // Writers and ops the store does not take.
    let system = Writer::Transaction {
        id: TransactionId::SYSTEM,
        snapshot: EpochId::new(1),
    };
    assert!(matches!(
        store.apply(&DataOp::DeleteNode { id: base.alix }, system),
        Err(ApplyError::Refused(_))
    ));
    assert!(matches!(
        store.apply(
            &DataOp::CreateNode {
                id: NodeId::INVALID,
                labels: labels(&[]),
                properties: properties(&[]),
            },
            replay
        ),
        Err(ApplyError::Refused(_))
    ));
}

/// A detach delete records the deletes of its edges before the delete of
/// its node, which a node with edges refuses; undo, last to first, restores
/// the node before its edges, which refuse to come back to a deleted node.
#[test]
fn a_detach_delete_records_its_edges_before_its_node() {
    let base = base();
    let store = &base.store;
    let committed = image(store, None);
    let id = TransactionId::new(3);
    let (recorder, mut set) = begin(store, 3);
    let gus_gus = recorder.create_edge(&mut set, base.gus, base.gus, "KNOWS", &[]);
    assert_eq!(
        recorder.write(&mut set, DataOp::DeleteNode { id: base.gus }),
        Err(ApplyError::HasEdges(base.gus)),
        "a node with edges is refused"
    );
    let mark = set.mark();
    for edge in [base.alix_gus, base.gus_mia, gus_gus] {
        recorder.delete_edge(&mut set, edge);
    }
    recorder.delete_node(&mut set, base.gus);
    let tail = set.split_off(mark);
    let kinds: Vec<u8> = tail
        .iter()
        .map(|change| match change {
            Change::Data { op, .. } => op.kind(),
            Change::Bulk(_) => 0,
        })
        .collect();
    let delete_edge = DataOp::DeleteEdge { id: gus_gus }.kind();
    let delete_node = DataOp::DeleteNode { id: base.gus }.kind();
    assert_eq!(kinds, [delete_edge, delete_edge, delete_edge, delete_node]);

    // Undone first to last, an edge would come back to a deleted node.
    let other = self::base();
    let (wrong, mut wrong_set) = begin(&other.store, 3);
    for edge in [other.alix_gus, other.gus_mia] {
        wrong.delete_edge(&mut wrong_set, edge);
    }
    wrong.delete_node(&mut wrong_set, other.gus);
    assert_eq!(
        other.store.undo(id, &mut wrong_set.entries().iter().rev()),
        Err(ApplyError::Missing(Entity::Node(other.gus)))
    );

    store.undo(id, &mut tail.iter()).unwrap();
    store.undo(id, &mut set.entries().iter()).unwrap();
    assert_eq!(
        image(store, None),
        committed,
        "the node and its edges are back"
    );
}

/// With `temporal`, a committed delete keeps the node's label history: at an
/// epoch before the delete, the node has its labels and its values. (They
/// are read directly: a delete marks the version deleted at the
/// transaction's start epoch, as the store's versioned writes do.)
#[cfg(feature = "temporal")]
#[test]
fn a_delete_keeps_the_history_of_labels_and_values() {
    let base = base();
    let store = &base.store;
    let before = store.current_epoch();
    let (recorder, mut set) = begin(store, 3);
    recorder.delete_edge(&mut set, base.gus_mia);
    recorder.delete_node(&mut set, base.mia);
    let epoch = commit(store, TransactionId::new(3), &set);
    assert!(
        store.get_node_at_epoch(base.mia, epoch).is_none(),
        "deleted"
    );
    let labels = store.node_with_labels_at(base.mia, before).labels;
    assert_eq!(
        labels.as_slice(),
        ["Person"],
        "the labels before the delete"
    );
    assert!(
        store.node_with_labels_at(base.mia, epoch).labels.is_empty(),
        "none after it"
    );
    assert_eq!(
        store.get_node_property_at_epoch(base.mia, &PropertyKey::new("name"), before),
        Some(Value::from("Mia"))
    );
}

/// Without backward adjacency a node delete still refuses a node with an
/// outgoing edge, and in a debug build one with an incoming edge too (found
/// by a scan); an edge the writer does not see (another transaction's
/// pending one) does not count.
#[test]
fn a_store_without_backward_adjacency_refuses_a_node_with_edges() {
    use crate::graph::lpg::store::LpgStoreConfig;

    let store = LpgStore::with_config(LpgStoreConfig {
        backward_edges: false,
        ..LpgStoreConfig::default()
    })
    .unwrap();
    let replay = Writer::Replay {
        epoch: EpochId::new(1),
    };
    let mut set = ChangeSet::new();
    let slot = default_slot(&mut set);
    let committed = Recorder {
        store: &store,
        slot,
        writer: replay,
    };
    let alix = committed.create_node(&mut set, &["Person"], &[]);
    let gus = committed.create_node(&mut set, &["Person"], &[]);
    let mia = committed.create_node(&mut set, &["Person"], &[]);
    committed.create_edge(&mut set, alix, gus, "KNOWS", &[]);

    let delete =
        |id: NodeId| store.apply(&DataOp::DeleteNode { id }, transaction(3, EpochId::new(1)));
    assert_eq!(
        delete(alix),
        Err(ApplyError::HasEdges(alix)),
        "an outgoing edge"
    );
    if cfg!(debug_assertions) {
        assert_eq!(
            delete(gus),
            Err(ApplyError::HasEdges(gus)),
            "an incoming edge"
        );
    }
    let (other, mut other_set) = begin(&store, 19);
    other.create_edge(&mut other_set, mia, alix, "KNOWS", &[]);
    assert!(
        matches!(delete(mia), Ok(Applied::Changed { .. })),
        "another transaction's pending edge is not the writer's to see"
    );
}

/// A bulk write's reserved range is one entry: its commit stamps and counts
/// the rows the transaction created in it (absent ids skipped), and its
/// rollback removes them.
#[test]
fn bulk_ranges_are_stamped_and_undone_by_range() {
    let base = base();
    let store = &base.store;
    let id = TransactionId::new(3);
    let writer = transaction(3, store.current_epoch());
    let mut set = ChangeSet::new();
    let slot = default_slot(&mut set);
    let ids = store.reserve_node_ids(5).unwrap();
    let mut rows = Vec::new();
    for (row, raw) in ids.clone().enumerate() {
        if row == 2 || row == 4 {
            continue;
        }
        let node = NodeId::new(raw);
        let op = DataOp::CreateNode {
            id: node,
            labels: labels(&["City"]),
            properties: properties(&[("name", Value::from("Prague"))]),
        };
        assert!(matches!(
            store.apply(&op, writer),
            Ok(Applied::Changed { .. })
        ));
        rows.push(node);
    }
    set.push_bulk(BulkRange {
        graph: slot,
        table: Table::Nodes,
        ids: ids.clone(),
    })
    .unwrap();
    let (nodes, edges, _) = counters(store);
    let epoch = commit(store, id, &set);
    for node in &rows {
        assert!(store.is_node_visible_at_epoch(*node, epoch), "row {node:?}");
        #[cfg(feature = "temporal")]
        assert_eq!(
            store.get_node_property_at_epoch(*node, &PropertyKey::new("name"), epoch),
            Some(Value::from("Prague")),
            "its value is stamped too"
        );
    }
    assert_eq!(counters(store).0, nodes + 3, "three rows created");

    let rollback = TransactionId::new(19);
    let writer = transaction(19, store.current_epoch());
    let mut set = ChangeSet::new();
    let slot = default_slot(&mut set);
    let edge_ids = store.reserve_edge_ids(3).unwrap();
    for (raw, (src, dst)) in edge_ids
        .clone()
        .zip([(rows[0], rows[1]), (rows[1], rows[2])])
    {
        let op = DataOp::CreateEdge {
            id: EdgeId::new(raw),
            src,
            dst,
            edge_type: ArcStr::from("ROUTE"),
            properties: properties(&[]),
        };
        assert!(matches!(
            store.apply(&op, writer),
            Ok(Applied::Changed { .. })
        ));
    }
    set.push_bulk(BulkRange {
        graph: slot,
        table: Table::Edges,
        ids: edge_ids.clone(),
    })
    .unwrap();
    store.undo(rollback, &mut set.entries().iter()).unwrap();
    for raw in edge_ids {
        assert!(
            store
                .get_edge_versioned(EdgeId::new(raw), EpochId::new(88), rollback)
                .is_none(),
            "edge {raw} is gone"
        );
    }
    assert_eq!(counters(store).1, edges, "no edge counted");
    assert!(
        store
            .edges_from(rows[0], crate::graph::Direction::Outgoing)
            .next()
            .is_none()
    );
}

/// A bulk write applies each row as the create it is, at an id of its
/// range, as the transaction's pending version: committed by the range's
/// stamp, gone with its undo. The checks of the row are the bulk write's:
/// the store does not look up an edge's endpoints again (the bulk write's
/// writer saw them), but a row at an id in use still changes nothing, and
/// only a transaction's creates are bulk rows.
#[test]
fn a_bulk_row_is_a_create_whose_endpoints_the_bulk_write_checked() {
    let base = base();
    let store = &base.store;
    let id = TransactionId::new(3);
    let writer = transaction(3, store.current_epoch());
    let (nodes, edges, _) = counters(store);
    let ids = store.reserve_edge_ids(2).unwrap();
    let mut set = ChangeSet::new();
    let slot = default_slot(&mut set);
    set.push_bulk(BulkRange {
        graph: slot,
        table: Table::Edges,
        ids: ids.clone(),
    })
    .unwrap();
    let edge = |raw: u64, src: NodeId, dst: NodeId| DataOp::CreateEdge {
        id: EdgeId::new(raw),
        src,
        dst,
        edge_type: ArcStr::from("VISITED"),
        properties: properties(&[("year", Value::Int64(1988))]),
    };
    let before = image(store, None);
    store
        .apply_bulk_row(&edge(ids.start, base.alix, base.gus), writer)
        .unwrap();
    assert_eq!(
        store.apply_bulk_row(&edge(base.alix_gus.as_u64(), base.alix, base.mia), writer),
        Err(ApplyError::Exists(Entity::Edge(base.alix_gus))),
        "a row at an id in use is refused"
    );
    let vincent = NodeId::new(store.reserve_node_ids(1).unwrap().start);
    for refused in [
        (
            DataOp::SetNodeProperty {
                id: base.alix,
                key: PropertyKey::new("age"),
                value: Value::Int64(88),
            },
            writer,
        ),
        (
            DataOp::CreateNode {
                id: vincent,
                labels: labels(&["Person"]),
                properties: Vec::new(),
            },
            Writer::Replay {
                epoch: EpochId::new(88),
            },
        ),
    ] {
        assert!(
            matches!(
                store.apply_bulk_row(&refused.0, refused.1),
                Err(ApplyError::Refused(_))
            ),
            "{refused:?}"
        );
    }
    assert!(
        store.get_node(vincent).is_none(),
        "a refused row is nothing"
    );
    assert_eq!(image(store, None), before, "readers see no pending row");
    assert_eq!(
        counters(store),
        (nodes, edges, counters(store).2),
        "rows count at commit"
    );

    // The endpoints are the bulk write's to check: an edge to a node the
    // writer does not see is applied as given (`apply` would refuse it).
    let unseen = NodeId::new(store.reserve_node_ids(1).unwrap().start);
    assert_eq!(
        store.apply(&edge(ids.start + 1, base.gus, unseen), writer),
        Err(ApplyError::Missing(Entity::Node(unseen)))
    );
    store
        .apply_bulk_row(&edge(ids.start + 1, base.gus, base.mia), writer)
        .unwrap();

    let epoch = commit(store, id, &set);
    for raw in ids {
        let edge = store
            .get_edge_at_epoch(EdgeId::new(raw), epoch)
            .unwrap_or_else(|| panic!("row {raw} is committed"));
        assert_eq!(edge.get_property("year"), Some(&Value::Int64(1988)));
    }
    assert_eq!(counters(store).1, edges + 2, "the rows counted at commit");
}

/// The before-images a transaction's writes report are what they replaced,
/// and replay builds none.
#[test]
fn apply_reports_what_each_write_replaced() {
    let base = base();
    let store = &base.store;
    let (recorder, mut set) = begin(store, 3);
    let write = |set: &mut ChangeSet, op: DataOp| match recorder.write(set, op) {
        Ok(Applied::Changed { before, .. }) => before,
        other => panic!("{other:?}"),
    };
    let key = PropertyKey::new;
    assert_eq!(
        write(
            &mut set,
            DataOp::SetNodeProperty {
                id: base.alix,
                key: key("age"),
                value: Value::Int64(88)
            }
        ),
        Before::Value(Some(Value::Int64(19)))
    );
    assert_eq!(
        write(
            &mut set,
            DataOp::SetNodeProperty {
                id: base.alix,
                key: key("nickname"),
                value: Value::from("Al")
            }
        ),
        Before::Value(None)
    );
    assert_eq!(
        write(
            &mut set,
            DataOp::AddNodeLabel {
                id: base.gus,
                label: ArcStr::from("Traveller")
            }
        ),
        Before::Labels(labels(&["Person", "Employee"]))
    );
    let Before::Edge(edge) = write(&mut set, DataOp::DeleteEdge { id: base.gus_mia }) else {
        panic!("an edge image");
    };
    assert_eq!(
        (edge.src, edge.dst, edge.edge_type.as_str()),
        (base.gus, base.mia, "KNOWS")
    );
    assert_eq!(edge.properties, properties(&[("since", Value::Int64(319))]));
    write(&mut set, DataOp::DeleteEdge { id: base.alix_gus });
    let Before::Node(node) = write(&mut set, DataOp::DeleteNode { id: base.gus }) else {
        panic!("a node image");
    };
    assert_eq!(node.labels, labels(&["Person", "Employee", "Traveller"]));
    assert_eq!(
        node.properties,
        properties(&[("age", Value::Int64(3)), ("name", Value::from("Gus"))])
    );
    assert_eq!(
        store.apply(
            &DataOp::SetNodeProperty {
                id: base.mia,
                key: key("age"),
                value: Value::Int64(3)
            },
            Writer::Replay {
                epoch: EpochId::new(2)
            }
        ),
        Ok(Applied::Committed),
        "replay builds no image"
    );
}

// ── Live writes and replay build the same store ─────────────────────

/// A small deterministic generator (a linear congruential one): each seed
/// gives the same writes on every run and platform.
struct Dice(u64);

impl Dice {
    fn roll(&mut self, sides: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) % sides
    }

    fn pick<'a>(&mut self, names: &[&'a str]) -> &'a str {
        let at = usize::try_from(self.roll(names.len() as u64)).unwrap();
        names[at]
    }

    fn value(&mut self) -> Value {
        match self.roll(4) {
            0 => Value::Int64(3),
            1 => Value::Int64(19),
            2 => Value::from("Berlin"),
            _ => Value::Float64(19.88),
        }
    }
}

const LABELS: [&str; 3] = ["Person", "City", "Employee"];
const KEYS: [&str; 3] = ["name", "age", "city"];
const EDGE_TYPES: [&str; 2] = ["KNOWS", "VISITED"];

/// One transaction of random writes on `store`, with savepoints and their
/// rollbacks, ending in a commit (returned with its epoch) or a rollback.
fn random_transaction(
    store: &LpgStore,
    dice: &mut Dice,
    id: u64,
    savepoint_rollbacks: &mut usize,
) -> Option<(ChangeSet, EpochId)> {
    let transaction = TransactionId::new(id);
    let (recorder, mut set) = begin(store, id);
    let mut marks = Vec::new();
    for _ in 0..24 {
        let nodes = store.next_node_id().max(1);
        let edges = store.next_edge_id().max(1);
        let node = NodeId::new(dice.roll(nodes));
        let edge = EdgeId::new(dice.roll(edges));
        let op = match dice.roll(13) {
            0 | 1 => {
                let id = NodeId::new(store.reserve_node_ids(1).unwrap().start);
                let mut names = vec![dice.pick(&LABELS)];
                if dice.roll(2) == 0 {
                    names.push(dice.pick(&LABELS));
                }
                names.dedup();
                DataOp::CreateNode {
                    id,
                    labels: labels(&names),
                    properties: properties(&[(dice.pick(&KEYS), dice.value())]),
                }
            }
            2 | 3 => DataOp::CreateEdge {
                id: EdgeId::new(store.reserve_edge_ids(1).unwrap().start),
                src: node,
                dst: NodeId::new(dice.roll(nodes)),
                edge_type: ArcStr::from(dice.pick(&EDGE_TYPES)),
                properties: properties(&[(dice.pick(&KEYS), dice.value())]),
            },
            4 => {
                // A detach delete: the edges the transaction sees first.
                let seen: Vec<EdgeId> = store
                    .edges_from(node, crate::graph::Direction::Both)
                    .map(|(_, edge)| edge)
                    .collect();
                for edge in seen {
                    recorder
                        .write(&mut set, DataOp::DeleteEdge { id: edge })
                        .unwrap();
                }
                DataOp::DeleteNode { id: node }
            }
            5 => DataOp::DeleteEdge { id: edge },
            6 => DataOp::SetNodeProperty {
                id: node,
                key: PropertyKey::new(dice.pick(&KEYS)),
                value: dice.value(),
            },
            7 => DataOp::RemoveNodeProperty {
                id: node,
                key: PropertyKey::new(dice.pick(&KEYS)),
            },
            8 => DataOp::SetEdgeProperty {
                id: edge,
                key: PropertyKey::new(dice.pick(&KEYS)),
                value: dice.value(),
            },
            9 => DataOp::RemoveEdgeProperty {
                id: edge,
                key: PropertyKey::new(dice.pick(&KEYS)),
            },
            10 => DataOp::AddNodeLabel {
                id: node,
                label: ArcStr::from(dice.pick(&LABELS)),
            },
            11 => DataOp::RemoveNodeLabel {
                id: node,
                label: ArcStr::from(dice.pick(&LABELS)),
            },
            _ => {
                if dice.roll(2) == 0 || marks.is_empty() {
                    marks.push(set.mark());
                } else if let Some(mark) = marks.pop() {
                    roll_back_to(store, transaction, &mut set, mark);
                    *savepoint_rollbacks += 1;
                }
                continue;
            }
        };
        match recorder.write(&mut set, op.clone()) {
            Ok(_) | Err(ApplyError::Missing(Entity::Node(_))) => {}
            Err(error) => panic!("{op:?}: {error}"),
        }
    }
    if dice.roll(4) == 0 {
        store.undo(transaction, &mut set.entries().iter()).unwrap();
        return None;
    }
    let epoch = commit(store, transaction, &set);
    Some((set, epoch))
}

/// The labels of every node at every epoch up to `last` (with `temporal`,
/// the label history; without it, the current labels).
fn label_history(store: &LpgStore, last: EpochId, nodes: u64) -> Vec<(u64, u64, Vec<String>)> {
    let mut history = Vec::new();
    for raw in 0..nodes {
        for epoch in 0..=last.as_u64() {
            let mut names: Vec<String> = store
                .node_with_labels_at(NodeId::new(raw), EpochId::new(epoch))
                .labels
                .iter()
                .map(ToString::to_string)
                .collect();
            names.sort();
            history.push((raw, epoch, names));
        }
    }
    history
}

/// Live writes as transactions, stamped at their commits (with savepoint
/// rollbacks and whole rollbacks among them), build the same store as
/// replay of the commits' log records, applied with their ids on an empty
/// store: the same nodes, edges, labels and values, the same counters and,
/// with `temporal`, the same versions.
#[test]
fn apply_with_ids_rebuilds_the_same_store() {
    let mut replayed_kinds: BTreeMap<u8, usize> = BTreeMap::new();
    let (mut savepoint_rollbacks, mut rollbacks) = (0, 0);
    for seed in 0..160_u64 {
        let live = LpgStore::new().unwrap();
        let replayed = LpgStore::new().unwrap();
        let mut dice = Dice(seed);
        for transaction in 0..5 {
            let Some((set, epoch)) = random_transaction(
                &live,
                &mut dice,
                100 + transaction,
                &mut savepoint_rollbacks,
            ) else {
                rollbacks += 1;
                continue;
            };
            for record in set.log_records() {
                let LogRecordRef::Data { op, .. } = record else {
                    panic!("seed {seed}: a change set's records are data records");
                };
                *replayed_kinds.entry(op.kind()).or_default() += 1;
                assert_eq!(
                    replayed.apply(op, Writer::Replay { epoch }),
                    Ok(Applied::Committed),
                    "seed {seed}: replay of {op:?}"
                );
            }
        }
        replayed.sync_epoch(live.current_epoch());
        assert_eq!(
            image(&replayed, None),
            image(&live, None),
            "seed {seed}: the stores"
        );
        assert_eq!(
            counters(&replayed),
            counters(&live),
            "seed {seed}: counters"
        );
        assert!(replayed.next_node_id() <= live.next_node_id());
        let nodes = live.next_node_id();
        assert_eq!(
            label_history(&replayed, live.current_epoch(), nodes),
            label_history(&live, live.current_epoch(), nodes),
            "seed {seed}: the labels"
        );
        #[cfg(feature = "temporal")]
        for raw in 0..nodes {
            assert_eq!(
                by_key(replayed.node_property_history(NodeId::new(raw))),
                by_key(live.node_property_history(NodeId::new(raw))),
                "seed {seed}: the versions of node {raw}"
            );
        }
        #[cfg(feature = "temporal")]
        for raw in 0..live.next_edge_id() {
            assert_eq!(
                by_key(replayed.edge_property_history(EdgeId::new(raw))),
                by_key(live.edge_property_history(EdgeId::new(raw))),
                "seed {seed}: the versions of edge {raw}"
            );
        }
    }
    // The writes reach every kind of op, often: the generator is not
    // degenerate.
    assert_eq!(replayed_kinds.len(), 10, "{replayed_kinds:?}");
    assert!(
        replayed_kinds.values().all(|count| *count >= 40),
        "{replayed_kinds:?}"
    );
    assert!(
        savepoint_rollbacks >= 40 && rollbacks >= 40,
        "{savepoint_rollbacks} savepoint rollbacks, {rollbacks} rollbacks"
    );
}

/// A value history in key order (a store lists its columns in any order).
#[cfg(feature = "temporal")]
fn by_key(
    mut history: Vec<(PropertyKey, Vec<(EpochId, Value)>)>,
) -> Vec<(PropertyKey, Vec<(EpochId, Value)>)> {
    history.sort_by(|(a, _), (b, _)| a.cmp(b));
    history
}

/// An immediate write (a direct call while no transaction is open) is
/// committed as it is applied, through replay's path: readers see it at
/// once, the counters move at once, and the store's epoch follows. Unlike
/// replay it is lenient (a write that changes nothing is `Unchanged`) and
/// reports what each write replaced, so its entries are what a transaction
/// would record; replayed at the same epoch they rebuild the same store.
#[test]
fn an_immediate_write_is_committed_at_once_with_its_before_image() {
    let Base {
        store,
        alix,
        gus,
        mia,
        gus_mia,
        ..
    } = base();
    let epoch = EpochId::new(2);
    let mut set = ChangeSet::new();
    let slot = default_slot(&mut set);
    let recorder = Recorder {
        store: &store,
        slot,
        writer: Writer::Immediate {
            epoch,
            before_images: true,
        },
    };
    recorder.set_node(&mut set, alix, "city", Value::from("Paris"));
    recorder.remove_node(&mut set, gus, "age");
    recorder.add_label(&mut set, alix, "Traveller");
    recorder.remove_label(&mut set, gus, "Employee");
    recorder.delete_edge(&mut set, gus_mia);
    recorder.delete_node(&mut set, mia);
    let vincent = recorder.create_node(&mut set, &["Person"], &[("name", Value::from("Vincent"))]);
    recorder.create_edge(
        &mut set,
        vincent,
        alix,
        "KNOWS",
        &[("since", Value::Int64(88))],
    );
    assert_eq!(
        recorder.write(
            &mut set,
            DataOp::RemoveNodeProperty {
                id: gus,
                key: PropertyKey::new("age"),
            }
        ),
        Ok(Applied::Unchanged),
        "lenient: a removal of an absent value changes nothing"
    );

    let before: Vec<&Before> = set
        .entries()
        .iter()
        .map(|change| match change {
            Change::Data {
                before, version, ..
            } => {
                assert_eq!(*version, PendingVersion::Created);
                before
            }
            Change::Bulk(_) => panic!("no bulk range"),
        })
        .collect();
    assert_eq!(before.len(), 8);
    assert_eq!(*before[0], Before::Value(Some(Value::from("Amsterdam"))));
    assert_eq!(*before[1], Before::Value(Some(Value::Int64(3))));
    assert!(matches!(before[4], Before::Edge(_)), "{:?}", before[4]);
    assert!(matches!(before[5], Before::Node(_)), "{:?}", before[5]);
    assert_eq!(store.current_epoch(), epoch, "the store's epoch follows");

    let committed = image(&store, None);
    assert_eq!(
        committed
            .nodes
            .get(&alix.as_u64())
            .map(|(labels, _)| labels.clone()),
        Some(vec!["Person".to_string(), "Traveller".to_string()]),
        "a committed reader sees it at once"
    );
    assert!(!committed.nodes.contains_key(&mia.as_u64()));

    let replayed = base();
    for record in set.log_records() {
        let LogRecordRef::Data { op, .. } = record else {
            panic!("a change set's records are data records");
        };
        assert_eq!(
            replayed.store.apply(op, Writer::Replay { epoch }),
            Ok(Applied::Committed)
        );
    }
    assert_eq!(image(&replayed.store, None), committed);
    assert_eq!(counters(&replayed.store), counters(&store));

    // Without before-images (nothing reads them), a write is committed and
    // reports so; one that changes nothing is still `Unchanged`.
    let quiet = Writer::Immediate {
        epoch: EpochId::new(3),
        before_images: false,
    };
    assert_eq!(
        store.apply(
            &DataOp::SetNodeProperty {
                id: alix,
                key: PropertyKey::new("city"),
                value: Value::from("Prague"),
            },
            quiet
        ),
        Ok(Applied::Committed)
    );
    assert_eq!(
        store.apply(
            &DataOp::RemoveNodeProperty {
                id: gus,
                key: PropertyKey::new("age"),
            },
            quiet
        ),
        Ok(Applied::Unchanged)
    );
    assert_eq!(store.current_epoch(), EpochId::new(3));
}

/// A dropped named graph takes no write from any writer, also one that
/// resolved the graph before the drop, and the refusal changes nothing; the
/// graph created again under its name is a new store that takes writes.
#[test]
fn a_dropped_graph_takes_no_more_writes() {
    let root = LpgStore::new().unwrap();
    assert!(root.create_graph("trips").unwrap());
    let trips = root.graph("trips").unwrap();
    let (recorder, mut set) = begin(&trips, 3);
    let alix = recorder.create_node(&mut set, &["Person"], &[]);
    commit(&trips, TransactionId::new(3), &set);
    assert!(!trips.is_dropped());

    assert!(root.drop_graph("trips"));
    assert!(trips.is_dropped(), "the handle resolved before the drop");
    let epoch = EpochId::new(trips.current_epoch().as_u64() + 1);
    let writers = [
        transaction(19, trips.current_epoch()),
        Writer::Immediate {
            epoch,
            before_images: true,
        },
        Writer::Replay { epoch },
    ];
    let gus = NodeId::new(trips.reserve_node_ids(1).unwrap().start);
    for writer in writers {
        for op in [
            DataOp::SetNodeProperty {
                id: alix,
                key: PropertyKey::new("city"),
                value: Value::from("Amsterdam"),
            },
            DataOp::CreateNode {
                id: gus,
                labels: labels(&["Person"]),
                properties: properties(&[]),
            },
        ] {
            assert!(
                matches!(trips.apply(&op, writer), Err(ApplyError::Refused(_))),
                "{writer:?} wrote {op:?} into a dropped graph"
            );
        }
    }
    assert_eq!(
        image(&trips, None)
            .nodes
            .keys()
            .copied()
            .collect::<Vec<_>>(),
        [alix.as_u64()],
        "the refused writes changed nothing"
    );
    assert!(
        image(&trips, None).nodes[&alix.as_u64()].1.is_empty(),
        "no value was set"
    );

    assert!(root.create_graph("trips").unwrap());
    let again = root.graph("trips").unwrap();
    assert!(!again.is_dropped(), "a graph created again is a new store");
    let (recorder, mut set) = begin(&again, 88);
    recorder.create_node(&mut set, &["Person"], &[]);
    assert_eq!(set.len(), 1);
}
