//! The cases: each checks one guarantee of a store behind the change set,
//! through `ChangeTarget` and the read traits only.

use grafeo_common::change::{Before, DataOp, Table};
use grafeo_common::types::{ArcStr, EdgeId, EpochId, NodeId, PropertyKey, Value};

use super::{
    Image, LABELS, Outcome, Reader, Rng, Store, Tx, apply_all, committed, counts, counts_of,
    ensure, image, labels, next_epoch, people, properties, random_write, reserve_edge,
    reserve_node, same,
};
use crate::graph::apply::{Applied, ApplyError, Writer};

/// The nodes and edges two images hold, without their problems.
fn data(
    image: &Image,
) -> (
    &std::collections::BTreeMap<u64, super::NodeView>,
    &std::collections::BTreeMap<u64, super::EdgeView>,
) {
    (&image.nodes, &image.edges)
}

/// Fails unless `found` holds the same nodes and edges as `expected`, and
/// every access path of `found` agrees.
fn same_data(found: &Image, expected: &Image, what: &str) -> Outcome {
    found.consistent(what)?;
    same(&data(found), &data(expected), what)
}

/// One write of every kind on [`People`](super::People): Jules is created
/// with an edge to Alix, Alix moves to Berlin, Gus loses his age and the
/// label `Employee`, Mia becomes a `Traveller` and then goes with her edge,
/// and the edge from Alix to Gus gets a new `since`.
fn every_op_kind<S: Store>(
    tx: &mut Tx<'_, S>,
    people: &super::People,
) -> Result<(NodeId, EdgeId), String> {
    let jules = tx.create_node(&["Person"], &[("name", Value::from("Jules"))])?;
    let jules_alix = tx.create_edge(
        jules,
        people.alix,
        "KNOWS",
        &[("since", Value::Int64(2088))],
    )?;
    tx.set_node(people.alix, "city", Value::from("Berlin"))?;
    tx.remove_node_value(people.gus, "age")?;
    tx.remove_label(people.gus, "Employee")?;
    tx.add_label(people.mia, "Traveller")?;
    tx.set_edge(people.alix_gus, "since", Value::Int64(2019))?;
    tx.remove_edge_value(people.gus_mia, "since")?;
    tx.delete_edge(people.gus_mia)?;
    tx.delete_node(people.mia)?;
    Ok((jules, jules_alix))
}

/// Checks that `image` shows what [`every_op_kind`] wrote.
fn shows_every_op_kind(
    image: &Image,
    people: &super::People,
    jules: NodeId,
    jules_alix: EdgeId,
) -> Outcome {
    image.has_node(jules, &["Person"], "name", Some(&Value::from("Jules")))?;
    image.has_node(
        people.alix,
        &["Person"],
        "city",
        Some(&Value::from("Berlin")),
    )?;
    image.has_node(people.gus, &["Person"], "age", None)?;
    ensure(!image.nodes.contains_key(&people.mia.as_u64()), || {
        "Mia is deleted".to_string()
    })?;
    ensure(!image.edges.contains_key(&people.gus_mia.as_u64()), || {
        "the edge to Mia is deleted".to_string()
    })?;
    same(
        &image
            .edges
            .get(&jules_alix.as_u64())
            .map(|edge| (edge.0, edge.1, edge.3.clone())),
        &Some((
            jules.as_u64(),
            people.alix.as_u64(),
            vec![("since".to_string(), Value::Int64(2088))],
        )),
        "Jules knows Alix",
    )?;
    same(
        &image
            .edges
            .get(&people.alix_gus.as_u64())
            .map(|edge| edge.3.clone()),
        &Some(vec![("since".to_string(), Value::Int64(2019))]),
        "the edge from Alix to Gus has its new since",
    )
}

/// A transaction sees each of its writes as it makes them; its commit makes
/// them what every later reader sees, through every access path.
pub(crate) fn each_op_is_seen_by_its_transaction_and_committed_by_stamp<S: Store>(
    make: &dyn Fn() -> S,
) -> Outcome {
    let store = make();
    let people = people(&store)?;
    let mut tx = Tx::begin(&store, 10);
    let (jules, jules_alix) = every_op_kind(&mut tx, &people)?;
    let own = tx.view();
    own.consistent("the transaction's own view")?;
    shows_every_op_kind(&own, &people, jules, jules_alix)?;

    let epoch = tx.commit()?;
    let after = image(&store, Reader::At(epoch));
    same_data(
        &after,
        &own,
        "a reader at the commit epoch sees what the transaction saw",
    )?;
    let later = Tx::begin(&store, 11);
    same_data(&later.view(), &own, "a later transaction sees the commit")
}

/// A full rollback leaves the committed state as it was, for the
/// transaction's reads and everyone else's, through every access path.
pub(crate) fn undo_restores_the_committed_state<S: Store>(make: &dyn Fn() -> S) -> Outcome {
    let store = make();
    let people = people(&store)?;
    let before = committed(&store);
    let mut tx = Tx::begin(&store, 10);
    every_op_kind(&mut tx, &people)?;
    tx.roll_back()?;
    same_data(
        &committed(&store),
        &before,
        "the committed state after a rollback",
    )?;
    same_data(
        &Tx::begin(&store, 11).view(),
        &before,
        "a later transaction after a rollback",
    )
}

/// Rolling back to each savepoint, latest first, gives back what the
/// transaction saw when it took it; back at the first, the committed state.
pub(crate) fn a_savepoint_rollback_restores_each_tail<S: Store>(make: &dyn Fn() -> S) -> Outcome {
    let store = make();
    let people = people(&store)?;
    let before = committed(&store);
    let mut tx = Tx::begin(&store, 10);
    let mut marks = vec![(tx.mark(), tx.view())];

    let jules = tx.create_node(&["Person", "Traveller"], &[("name", Value::from("Jules"))])?;
    tx.create_edge(jules, people.alix, "KNOWS", &[])?;
    tx.set_node(people.alix, "city", Value::from("Berlin"))?;
    marks.push((tx.mark(), tx.view()));

    tx.add_label(people.mia, "City")?;
    tx.remove_label(people.gus, "Employee")?;
    tx.remove_node_value(people.alix, "age")?;
    tx.set_edge(people.alix_gus, "since", Value::Int64(2019))?;
    marks.push((tx.mark(), tx.view()));

    tx.delete_edge(people.gus_mia)?;
    tx.delete_node(people.mia)?;
    tx.set_node(jules, "city", Value::from("Paris"))?;

    for (stage, (mark, view)) in marks.into_iter().enumerate().rev() {
        tx.roll_back_to(mark)?;
        same_data(&tx.view(), &view, &format!("back at savepoint {stage}"))?;
    }
    tx.commit()?;
    same_data(
        &committed(&store),
        &before,
        "after rolling back everything and committing",
    )
}

/// A savepoint rollback keeps what the transaction wrote before the
/// savepoint, also for a value and a label it wrote again after it.
pub(crate) fn a_savepoint_rollback_keeps_the_transactions_own_value<S: Store>(
    make: &dyn Fn() -> S,
) -> Outcome {
    let store = make();
    let people = people(&store)?;
    let mut tx = Tx::begin(&store, 10);
    tx.set_node(people.alix, "age", Value::Int64(3))?;
    tx.add_label(people.alix, "Traveller")?;
    tx.set_edge(people.alix_gus, "since", Value::Int64(2019))?;
    let mark = tx.mark();
    tx.set_node(people.alix, "age", Value::Int64(19))?;
    tx.set_node(people.alix, "age", Value::Int64(88))?;
    tx.remove_label(people.alix, "Traveller")?;
    tx.add_label(people.alix, "City")?;
    tx.set_edge(people.alix_gus, "since", Value::Int64(1988))?;
    tx.roll_back_to(mark)?;

    let own = tx.view();
    own.has_node(
        people.alix,
        &["Person", "Traveller"],
        "age",
        Some(&Value::Int64(3)),
    )?;
    same(
        &own.edges
            .get(&people.alix_gus.as_u64())
            .map(|edge| edge.3.clone()),
        &Some(vec![("since".to_string(), Value::Int64(2019))]),
        "the edge keeps the transaction's value",
    )?;
    let epoch = tx.commit()?;
    image(&store, Reader::At(epoch)).has_node(
        people.alix,
        &["Person", "Traveller"],
        "age",
        Some(&Value::Int64(3)),
    )
}

/// A node the transaction created and then deleted comes back, with its
/// labels and values, when it rolls back to a savepoint between the two.
pub(crate) fn a_node_created_and_deleted_in_one_transaction_comes_back_at_a_savepoint<S: Store>(
    make: &dyn Fn() -> S,
) -> Outcome {
    let store = make();
    people(&store)?;
    let mut tx = Tx::begin(&store, 10);
    let jules = tx.create_node(&["Person", "Traveller"], &[("name", Value::from("Jules"))])?;
    let mark = tx.mark();
    tx.delete_node(jules)?;
    ensure(!tx.view().nodes.contains_key(&jules.as_u64()), || {
        "Jules is deleted".to_string()
    })?;
    tx.roll_back_to(mark)?;
    tx.view().has_node(
        jules,
        &["Person", "Traveller"],
        "name",
        Some(&Value::from("Jules")),
    )?;
    let epoch = tx.commit()?;
    image(&store, Reader::At(epoch)).has_node(
        jules,
        &["Person", "Traveller"],
        "name",
        Some(&Value::from("Jules")),
    )
}

/// A detach delete (the node's edges first, then the node) is undone whole:
/// the node, its edges, both adjacencies and the label scans come back.
pub(crate) fn a_detach_delete_is_undone_with_its_edges<S: Store>(make: &dyn Fn() -> S) -> Outcome {
    let store = make();
    let people = people(&store)?;
    let before = committed(&store);
    let mut tx = Tx::begin(&store, 10);
    tx.detach_delete(people.gus)?;
    let own = tx.view();
    own.consistent("after the detach delete")?;
    ensure(
        own.edges.is_empty() && !own.nodes.contains_key(&people.gus.as_u64()),
        || format!("Gus and his edges are gone: {own:?}"),
    )?;
    tx.roll_back()?;
    same_data(
        &committed(&store),
        &before,
        "after undoing the detach delete",
    )
}

/// Each write reports what it replaced: nothing for a create, the old value
/// (or none) for a value, the label set before for a label op, the whole
/// entity for a delete.
pub(crate) fn before_images_describe_what_each_write_replaced<S: Store>(
    make: &dyn Fn() -> S,
) -> Outcome {
    let store = make();
    let people = people(&store)?;
    let mut tx = Tx::begin(&store, 10);
    let before = |applied: Result<Applied, ApplyError>, what: &str| match applied {
        Ok(Applied::Changed { before, .. }) => Ok(before),
        other => Err(format!("{what}: expected a change, got {other:?}")),
    };
    let sorted_labels = |labels: &grafeo_common::change::Labels| {
        let mut names: Vec<String> = labels.iter().map(ToString::to_string).collect();
        names.sort();
        names
    };
    let sorted_properties = |values: &grafeo_common::change::Properties| {
        let mut pairs: Vec<(String, Value)> = values
            .iter()
            .map(|(key, value)| (key.as_str().to_string(), value.clone()))
            .collect();
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        pairs
    };

    let jules = NodeId::new(reserve_node(&store)?);
    let created = before(
        tx.apply(DataOp::CreateNode {
            id: jules,
            labels: labels(&["Person"]),
            properties: properties(&[("name", Value::from("Jules"))]),
        }),
        "create Jules",
    )?;
    same(&created, &Before::Absent, "a create replaced nothing")?;

    let city = before(
        tx.apply(DataOp::SetNodeProperty {
            id: people.alix,
            key: PropertyKey::new("city"),
            value: Value::from("Berlin"),
        }),
        "set Alix's city",
    )?;
    same(
        &city,
        &Before::Value(Some(Value::from("Amsterdam"))),
        "a set replaced the old value",
    )?;
    let new_key = before(
        tx.apply(DataOp::SetNodeProperty {
            id: people.mia,
            key: PropertyKey::new("city"),
            value: Value::from("Prague"),
        }),
        "set Mia's city",
    )?;
    same(
        &new_key,
        &Before::Value(None),
        "a set of a new key replaced no value",
    )?;
    let removed = before(
        tx.apply(DataOp::RemoveNodeProperty {
            id: people.gus,
            key: PropertyKey::new("age"),
        }),
        "remove Gus's age",
    )?;
    same(
        &removed,
        &Before::Value(Some(Value::Int64(3))),
        "a removal replaced the old value",
    )?;

    let added = before(
        tx.apply(DataOp::AddNodeLabel {
            id: people.alix,
            label: ArcStr::from("Traveller"),
        }),
        "add Traveller to Alix",
    )?;
    match &added {
        Before::Labels(labels) => same(
            &sorted_labels(labels),
            &vec!["Person".to_string()],
            "the labels before the add",
        )?,
        other => return Err(format!("a label add replaced {other:?}")),
    }
    let taken = before(
        tx.apply(DataOp::RemoveNodeLabel {
            id: people.gus,
            label: ArcStr::from("Employee"),
        }),
        "remove Employee from Gus",
    )?;
    match &taken {
        Before::Labels(labels) => same(
            &sorted_labels(labels),
            &vec!["Employee".to_string(), "Person".to_string()],
            "the labels before the removal",
        )?,
        other => return Err(format!("a label removal replaced {other:?}")),
    }

    let since = before(
        tx.apply(DataOp::SetEdgeProperty {
            id: people.alix_gus,
            key: PropertyKey::new("since"),
            value: Value::Int64(2019),
        }),
        "set the edge's since",
    )?;
    same(
        &since,
        &Before::Value(Some(Value::Int64(1988))),
        "an edge set replaced the old value",
    )?;
    let edge = before(
        tx.apply(DataOp::DeleteEdge { id: people.gus_mia }),
        "delete the edge to Mia",
    )?;
    match &edge {
        Before::Edge(image) => same(
            &(
                image.src,
                image.dst,
                image.edge_type.to_string(),
                sorted_properties(&image.properties),
            ),
            &(
                people.gus,
                people.mia,
                "KNOWS".to_string(),
                vec![("since".to_string(), Value::Int64(319))],
            ),
            "an edge delete replaced the whole edge",
        )?,
        other => return Err(format!("an edge delete replaced {other:?}")),
    }
    let node = before(
        tx.apply(DataOp::DeleteNode { id: people.mia }),
        "delete Mia",
    )?;
    match &node {
        Before::Node(image) => same(
            &(
                sorted_labels(&image.labels),
                sorted_properties(&image.properties),
            ),
            &(
                vec!["Person".to_string()],
                vec![
                    ("city".to_string(), Value::from("Prague")),
                    ("name".to_string(), Value::from("Mia")),
                ],
            ),
            "a node delete replaced the whole node, the transaction's own value included",
        ),
        other => Err(format!("a node delete replaced {other:?}")),
    }
}

/// A write that changes nothing (removing a value or label the entity does
/// not have, adding a label it has) is `Unchanged` for a transaction and an
/// immediate write, and changes nothing a reader sees.
pub(crate) fn writes_that_change_nothing_are_unchanged<S: Store>(make: &dyn Fn() -> S) -> Outcome {
    let store = make();
    let people = people(&store)?;
    let before = committed(&store);
    let no_ops = [
        DataOp::RemoveNodeProperty {
            id: people.mia,
            key: PropertyKey::new("age"),
        },
        DataOp::RemoveNodeLabel {
            id: people.alix,
            label: ArcStr::from("Employee"),
        },
        DataOp::AddNodeLabel {
            id: people.gus,
            label: ArcStr::from("Employee"),
        },
        DataOp::RemoveEdgeProperty {
            id: people.alix_gus,
            key: PropertyKey::new("weight"),
        },
    ];
    let mut tx = Tx::begin(&store, 10);
    for op in &no_ops {
        same(
            &tx.apply(op.clone()),
            &Ok(Applied::Unchanged),
            &format!("{op:?} in a transaction"),
        )?;
    }
    tx.commit()?;
    for op in &no_ops {
        for before_images in [true, false] {
            let writer = Writer::Immediate {
                epoch: next_epoch(&store),
                before_images,
            };
            same(
                &store.apply(op, writer),
                &Ok(Applied::Unchanged),
                &format!("{op:?} as {writer:?}"),
            )?;
        }
    }
    same_data(&committed(&store), &before, "nothing changed")
}

/// Replay is strict: a create at an id in use, a write to a missing entity,
/// removing what is not there, adding a label that is, and deleting a node
/// with edges are errors, and each leaves the store as it was.
pub(crate) fn replay_refuses_what_a_log_cannot_hold<S: Store>(make: &dyn Fn() -> S) -> Outcome {
    let store = make();
    let people = people(&store)?;
    let before = committed(&store);
    let missing = NodeId::new(people.mia.as_u64() + 1_000);
    let cases: Vec<(DataOp, fn(&ApplyError) -> bool)> = vec![
        (
            DataOp::CreateNode {
                id: people.gus,
                labels: labels(&["Person"]),
                properties: properties(&[]),
            },
            |error| matches!(error, ApplyError::Exists(_)),
        ),
        (
            DataOp::SetNodeProperty {
                id: missing,
                key: PropertyKey::new("name"),
                value: Value::from("Vincent"),
            },
            |error| matches!(error, ApplyError::Missing(_)),
        ),
        (
            DataOp::RemoveNodeProperty {
                id: people.mia,
                key: PropertyKey::new("age"),
            },
            |error| matches!(error, ApplyError::Missing(_)),
        ),
        (
            DataOp::RemoveNodeLabel {
                id: people.alix,
                label: ArcStr::from("Employee"),
            },
            |error| matches!(error, ApplyError::Missing(_)),
        ),
        (
            DataOp::AddNodeLabel {
                id: people.gus,
                label: ArcStr::from("Employee"),
            },
            |error| matches!(error, ApplyError::Exists(_)),
        ),
        (DataOp::DeleteNode { id: people.gus }, |error| {
            matches!(error, ApplyError::HasEdges(_))
        }),
        (
            DataOp::CreateEdge {
                id: EdgeId::new(reserve_edge(&store)?),
                src: people.alix,
                dst: missing,
                edge_type: ArcStr::from("KNOWS"),
                properties: properties(&[]),
            },
            |error| matches!(error, ApplyError::Missing(_)),
        ),
    ];
    for (op, expected) in cases {
        let applied = store.apply(
            &op,
            Writer::Replay {
                epoch: next_epoch(&store),
            },
        );
        ensure(matches!(&applied, Err(error) if expected(error)), || {
            format!("replay of {op:?} gave {applied:?}")
        })?;
        same_data(
            &committed(&store),
            &before,
            &format!("after the refused {op:?}"),
        )?;
    }
    Ok(())
}

/// The committed transactions of a seeded workload, replayed into a new
/// store at their commit epochs, give the same data and counts as the live
/// store; the rolled-back ones and the undone savepoint tails leave no trace.
pub(crate) fn replay_rebuilds_what_live_commits_wrote<S: Store>(make: &dyn Fn() -> S) -> Outcome {
    for seed in 1..=24 {
        let live = make();
        let mut rng = Rng::new(seed);
        let mut log: Vec<(EpochId, Vec<DataOp>)> = Vec::new();
        for transaction in 0..12_u64 {
            let mut tx = Tx::begin(&live, 100 + transaction);
            for step in 0..10 {
                if step == 5 && rng.below(3) == 0 {
                    let mark = tx.mark();
                    random_write(&mut tx, &mut rng)?;
                    random_write(&mut tx, &mut rng)?;
                    tx.roll_back_to(mark)?;
                }
                random_write(&mut tx, &mut rng)?;
            }
            if rng.below(4) == 0 {
                tx.roll_back()?;
            } else {
                let ops = tx.ops();
                let epoch = tx.commit()?;
                log.push((epoch, ops));
            }
        }

        let replayed = make();
        for (epoch, ops) in &log {
            apply_all(&replayed, ops, Writer::Replay { epoch: *epoch })?;
        }
        let what = |part: &str| format!("seed {seed}: {part}");
        same_data(
            &committed(&replayed),
            &committed(&live),
            &what("the replayed data"),
        )?;
        same(
            &replayed.current_epoch(),
            &live.current_epoch(),
            &what("the epoch"),
        )?;
        same(
            &counts(&replayed, &LABELS),
            &counts(&live, &LABELS),
            &what("the counts"),
        )?;
    }
    Ok(())
}

/// An immediate write commits as it applies: readers see it at its epoch,
/// it reports the before-image only when asked to, and the store's epoch
/// moves to it.
pub(crate) fn immediate_writes_commit_at_once<S: Store>(make: &dyn Fn() -> S) -> Outcome {
    let store = make();
    let people = people(&store)?;
    let epoch = next_epoch(&store);
    let applied = store.apply(
        &DataOp::SetNodeProperty {
            id: people.alix,
            key: PropertyKey::new("city"),
            value: Value::from("Berlin"),
        },
        Writer::Immediate {
            epoch,
            before_images: true,
        },
    );
    ensure(
        matches!(&applied, Ok(Applied::Changed { before: Before::Value(Some(value)), .. }) if *value == Value::from("Amsterdam")),
        || format!("an immediate write with images gave {applied:?}"),
    )?;
    ensure(store.current_epoch() >= epoch, || {
        "the store's epoch moved to the write".to_string()
    })?;
    image(&store, Reader::At(epoch)).has_node(
        people.alix,
        &["Person"],
        "city",
        Some(&Value::from("Berlin")),
    )?;

    let epoch = next_epoch(&store);
    let jules = NodeId::new(reserve_node(&store)?);
    let applied = store.apply(
        &DataOp::CreateNode {
            id: jules,
            labels: labels(&["Person"]),
            properties: properties(&[("name", Value::from("Jules"))]),
        },
        Writer::Immediate {
            epoch,
            before_images: false,
        },
    );
    same(
        &applied,
        &Ok(Applied::Committed),
        "an immediate write without images",
    )?;
    let after = image(&store, Reader::At(epoch));
    after.has_node(jules, &["Person"], "name", Some(&Value::from("Jules")))?;
    after.consistent("after the immediate writes")?;
    same(
        &counts(&store, &LABELS),
        &counts_of(&after, &LABELS),
        "the counts include the immediate writes",
    )
}

/// The same writes applied as immediate writes and replayed give the same
/// store.
pub(crate) fn immediate_writes_and_replay_agree<S: Store>(make: &dyn Fn() -> S) -> Outcome {
    let immediate = make();
    let replayed = make();
    let people_now = people(&immediate)?;
    people(&replayed)?;
    let jules = NodeId::new(reserve_node(&immediate)?);
    reserve_node(&replayed)?;
    let edge = EdgeId::new(reserve_edge(&immediate)?);
    reserve_edge(&replayed)?;
    let ops = vec![
        DataOp::CreateNode {
            id: jules,
            labels: labels(&["Person", "Traveller"]),
            properties: properties(&[("name", Value::from("Jules"))]),
        },
        DataOp::CreateEdge {
            id: edge,
            src: jules,
            dst: people_now.mia,
            edge_type: ArcStr::from("KNOWS"),
            properties: properties(&[("since", Value::Int64(2088))]),
        },
        DataOp::SetNodeProperty {
            id: people_now.alix,
            key: PropertyKey::new("city"),
            value: Value::from("Paris"),
        },
        DataOp::RemoveNodeLabel {
            id: people_now.gus,
            label: ArcStr::from("Employee"),
        },
        DataOp::DeleteEdge {
            id: people_now.alix_gus,
        },
        DataOp::DeleteNode {
            id: people_now.alix,
        },
    ];
    for op in &ops {
        let epoch = next_epoch(&immediate);
        immediate
            .apply(
                op,
                Writer::Immediate {
                    epoch,
                    before_images: true,
                },
            )
            .map_err(|error| format!("{op:?} immediate: {error:?}"))?;
        replayed
            .apply(op, Writer::Replay { epoch })
            .map_err(|error| format!("{op:?} replayed: {error:?}"))?;
    }
    same_data(
        &committed(&immediate),
        &committed(&replayed),
        "immediate writes and replay",
    )?;
    same(
        &counts(&immediate, &LABELS),
        &counts(&replayed, &LABELS),
        "their counts",
    )
}

/// Reserved ids are never given out again, also after the transaction that
/// used them rolled back, and a replayed create keeps the allocators above
/// its id.
pub(crate) fn reserved_ids_are_never_given_out_again<S: Store>(make: &dyn Fn() -> S) -> Outcome {
    let store = make();
    people(&store)?;
    let nodes = store
        .reserve_node_ids(3)
        .map_err(|error| format!("{error:?}"))?;
    let edges = store
        .reserve_edge_ids(2)
        .map_err(|error| format!("{error:?}"))?;
    let mut tx = Tx::begin(&store, 10);
    let node = NodeId::new(nodes.start);
    tx.change(DataOp::CreateNode {
        id: node,
        labels: labels(&["Person"]),
        properties: properties(&[]),
    })?;
    tx.change(DataOp::CreateEdge {
        id: EdgeId::new(edges.start),
        src: node,
        dst: node,
        edge_type: ArcStr::from("KNOWS"),
        properties: properties(&[]),
    })?;
    tx.roll_back()?;
    let next_node = reserve_node(&store)?;
    let next_edge = reserve_edge(&store)?;
    ensure(next_node >= nodes.end && next_edge >= edges.end, || {
        format!(
            "after reserving {nodes:?} and {edges:?}, the next ids are {next_node} and {next_edge}"
        )
    })?;

    let far = NodeId::new(next_node + 1_000);
    let far_edge = EdgeId::new(next_edge + 1_000);
    let epoch = next_epoch(&store);
    apply_all(
        &store,
        &[
            DataOp::CreateNode {
                id: far,
                labels: labels(&["City"]),
                properties: properties(&[]),
            },
            DataOp::CreateEdge {
                id: far_edge,
                src: far,
                dst: far,
                edge_type: ArcStr::from("LIVES_IN"),
                properties: properties(&[]),
            },
        ],
        Writer::Replay { epoch },
    )?;
    let after_node = reserve_node(&store)?;
    let after_edge = reserve_edge(&store)?;
    ensure(
        after_node > far.as_u64() && after_edge > far_edge.as_u64(),
        || {
            format!(
                "after replayed creates at {far:?} and {far_edge:?}, the next ids are {after_node} and {after_edge}"
            )
        },
    )
}

/// A bulk write's reserved range is one entry: its commit makes the rows it
/// created visible and counted (absent ids skipped), its rollback removes
/// them.
pub(crate) fn bulk_ranges_are_stamped_and_undone_by_range<S: Store>(
    make: &dyn Fn() -> S,
) -> Outcome {
    let store = make();
    let people = people(&store)?;
    let (nodes_before, edges_before, _) = counts(&store, &LABELS);
    let mut tx = Tx::begin(&store, 10);
    let ids = store
        .reserve_node_ids(5)
        .map_err(|error| format!("{error:?}"))?;
    let mut rows = Vec::new();
    for (row, raw) in ids.clone().enumerate() {
        if row == 2 || row == 4 {
            continue;
        }
        let op = DataOp::CreateNode {
            id: NodeId::new(raw),
            labels: labels(&["City"]),
            properties: properties(&[("name", Value::from("Prague"))]),
        };
        let applied = store.apply(&op, tx.writer());
        ensure(matches!(applied, Ok(Applied::Changed { .. })), || {
            format!("bulk row {raw}: {applied:?}")
        })?;
        rows.push(raw);
    }
    tx.push_bulk(Table::Nodes, ids)?;
    let epoch = tx.commit()?;
    let after = image(&store, Reader::At(epoch));
    for raw in &rows {
        after.has_node(
            NodeId::new(*raw),
            &["City"],
            "name",
            Some(&Value::from("Prague")),
        )?;
    }
    same(
        &counts(&store, &LABELS).0,
        &(nodes_before + rows.len()),
        "the bulk rows are counted",
    )?;

    let before = committed(&store);
    let mut tx = Tx::begin(&store, 11);
    let edge_ids = store
        .reserve_edge_ids(3)
        .map_err(|error| format!("{error:?}"))?;
    for (raw, (src, dst)) in edge_ids
        .clone()
        .zip([(rows[0], rows[1]), (rows[1], people.alix.as_u64())])
    {
        let op = DataOp::CreateEdge {
            id: EdgeId::new(raw),
            src: NodeId::new(src),
            dst: NodeId::new(dst),
            edge_type: ArcStr::from("ROUTE"),
            properties: properties(&[]),
        };
        let applied = store.apply(&op, tx.writer());
        ensure(matches!(applied, Ok(Applied::Changed { .. })), || {
            format!("bulk edge {raw}: {applied:?}")
        })?;
    }
    tx.push_bulk(Table::Edges, edge_ids)?;
    tx.roll_back()?;
    same_data(&committed(&store), &before, "after the bulk rollback")?;
    same(
        &counts(&store, &LABELS).1,
        &edges_before,
        "no bulk edge is counted",
    )
}

/// The counts move when a transaction commits, by what it wrote, and not
/// while it is open or when it rolls back.
pub(crate) fn counts_move_at_commit_only<S: Store>(make: &dyn Fn() -> S) -> Outcome {
    let store = make();
    let people = people(&store)?;
    let before = counts(&store, &LABELS);
    same(
        &before,
        &counts_of(&committed(&store), &LABELS),
        "the counts start right",
    )?;

    let mut tx = Tx::begin(&store, 10);
    let jules = tx.create_node(&["Person", "Traveller"], &[])?;
    tx.create_edge(jules, people.alix, "KNOWS", &[])?;
    tx.remove_label(people.gus, "Employee")?;
    tx.detach_delete(people.mia)?;
    same(
        &counts(&store, &LABELS),
        &before,
        "the counts while the transaction is open",
    )?;
    let own = tx.view();
    tx.commit()?;
    same(
        &counts(&store, &LABELS),
        &counts_of(&own, &LABELS),
        "the counts after the commit",
    )?;

    let after = counts(&store, &LABELS);
    let mut tx = Tx::begin(&store, 11);
    tx.create_node(&["City"], &[])?;
    tx.detach_delete(people.gus)?;
    tx.roll_back()?;
    same(
        &counts(&store, &LABELS),
        &after,
        "the counts after a rollback",
    )
}

/// After a seeded workload of commits, rollbacks and savepoint rollbacks,
/// every access path agrees with the nodes and edges: both adjacencies, the
/// label scans and counts, the property index, the point reads and the
/// degrees.
pub(crate) fn every_access_path_agrees<S: Store>(make: &dyn Fn() -> S) -> Outcome {
    for seed in 1..=12 {
        let store = make();
        let mut rng = Rng::new(seed);
        for transaction in 0..10_u64 {
            let mut tx = Tx::begin(&store, 100 + transaction);
            for step in 0..12 {
                if step == 6 && rng.below(2) == 0 {
                    let mark = tx.mark();
                    random_write(&mut tx, &mut rng)?;
                    tx.roll_back_to(mark)?;
                }
                random_write(&mut tx, &mut rng)?;
            }
            tx.view().consistent(&format!(
                "seed {seed}, transaction {transaction}, its own view"
            ))?;
            if rng.below(3) == 0 {
                tx.roll_back()?;
            } else {
                tx.commit()?;
            }
        }
        let now = committed(&store);
        now.consistent(&format!("seed {seed}"))?;
        same(
            &counts(&store, &LABELS),
            &counts_of(&now, &LABELS),
            &format!("seed {seed}: the counts"),
        )?;
        for (raw, (_, values)) in &now.nodes {
            let id = NodeId::new(*raw);
            for (key, value) in values {
                same(
                    &store.get_node_property(id, &PropertyKey::new(key.as_str())),
                    &Some(value.clone()),
                    &format!("seed {seed}: node {raw}'s {key} read alone"),
                )?;
            }
            let out = now.edges.values().filter(|edge| edge.0 == *raw).count();
            let incoming = now.edges.values().filter(|edge| edge.1 == *raw).count();
            same(
                &store.out_degree(id),
                &out,
                &format!("seed {seed}: node {raw}'s out-degree"),
            )?;
            if store.has_backward_adjacency() {
                same(
                    &store.in_degree(id),
                    &incoming,
                    &format!("seed {seed}: node {raw}'s in-degree"),
                )?;
            }
        }
        for (raw, (_, _, _, values)) in &now.edges {
            for (key, value) in values {
                same(
                    &store.get_edge_property(EdgeId::new(*raw), &PropertyKey::new(key.as_str())),
                    &Some(value.clone()),
                    &format!("seed {seed}: edge {raw}'s {key} read alone"),
                )?;
            }
        }
    }
    Ok(())
}

/// A node delete while the writer sees an edge of the node is refused
/// (`HasEdges`), for outgoing edges and, with backward adjacency, incoming
/// ones; detach deletes remove the edges first.
pub(crate) fn a_node_with_edges_cannot_be_deleted<S: Store>(make: &dyn Fn() -> S) -> Outcome {
    let store = make();
    let people = people(&store)?;
    let before = committed(&store);
    let mut tx = Tx::begin(&store, 10);
    let mut refused = vec![people.alix, people.gus];
    if store.has_backward_adjacency() {
        refused.push(people.mia);
    }
    for node in refused {
        let applied = tx.apply(DataOp::DeleteNode { id: node });
        ensure(
            matches!(applied, Err(ApplyError::HasEdges(id)) if id == node),
            || {
                format!(
                    "deleting node {} with edges gave {applied:?}",
                    node.as_u64()
                )
            },
        )?;
    }
    tx.roll_back()?;
    same_data(&committed(&store), &before, "after the refused deletes")
}

/// A label or edge type first used by a transaction that rolled back stays
/// listed: name dictionaries are not transactional (change target R12).
pub(crate) fn names_first_used_by_a_rolled_back_transaction_stay_listed<S: Store>(
    make: &dyn Fn() -> S,
) -> Outcome {
    let store = make();
    let people = people(&store)?;
    let mut tx = Tx::begin(&store, 10);
    tx.add_label(people.alix, "Traveller")?;
    tx.create_edge(people.alix, people.mia, "VISITED", &[])?;
    tx.roll_back()?;
    ensure(
        store.all_labels().iter().any(|label| label == "Traveller"),
        || format!("the labels {:?} list Traveller", store.all_labels()),
    )?;
    ensure(
        store
            .all_edge_types()
            .iter()
            .any(|edge_type| edge_type == "VISITED"),
        || format!("the edge types {:?} list VISITED", store.all_edge_types()),
    )
}

/// The nodes and edges an open transaction creates are seen by no one else.
pub(crate) fn an_open_transactions_creates_are_invisible_to_others<S: Store>(
    make: &dyn Fn() -> S,
) -> Outcome {
    let store = make();
    let people = people(&store)?;
    let before = committed(&store);
    let other = Tx::begin(&store, 11);
    let mut tx = Tx::begin(&store, 10);
    let jules = tx.create_node(&["Person"], &[("name", Value::from("Jules"))])?;
    tx.create_edge(jules, people.alix, "KNOWS", &[])?;
    tx.create_edge(people.mia, jules, "KNOWS", &[])?;
    same_data(&other.view(), &before, "another transaction")?;
    same_data(&committed(&store), &before, "a committed reader")
}

/// The values and labels an open transaction writes are seen by no one
/// else (#412).
pub(crate) fn an_open_transactions_value_and_label_writes_are_invisible_to_others<S: Store>(
    make: &dyn Fn() -> S,
) -> Outcome {
    let store = make();
    let people = people(&store)?;
    let before = committed(&store);
    let other = Tx::begin(&store, 11);
    let mut tx = Tx::begin(&store, 10);
    tx.set_node(people.alix, "city", Value::from("Berlin"))?;
    tx.remove_node_value(people.gus, "age")?;
    tx.add_label(people.mia, "Traveller")?;
    tx.remove_label(people.gus, "Employee")?;
    tx.set_edge(people.alix_gus, "since", Value::Int64(2019))?;
    same_data(&other.view(), &before, "another transaction")?;
    same_data(&committed(&store), &before, "a committed reader")
}

/// The nodes and edges an open transaction deletes are still seen by
/// everyone else (#412).
pub(crate) fn an_open_transactions_deletes_are_invisible_to_others<S: Store>(
    make: &dyn Fn() -> S,
) -> Outcome {
    let store = make();
    let people = people(&store)?;
    let before = committed(&store);
    let other = Tx::begin(&store, 11);
    let mut tx = Tx::begin(&store, 10);
    tx.delete_edge(people.gus_mia)?;
    tx.delete_node(people.mia)?;
    same_data(&other.view(), &before, "another transaction")?;
    same_data(&committed(&store), &before, "a committed reader")
}

/// A reader at an earlier epoch keeps the values and labels of that epoch
/// after later commits change them (snapshot reads, and history).
pub(crate) fn a_reader_at_an_earlier_epoch_keeps_its_values_and_labels<S: Store>(
    make: &dyn Fn() -> S,
) -> Outcome {
    let store = make();
    let people = people(&store)?;
    let epoch = store.current_epoch();
    let before = committed(&store);
    let reader = Tx::begin(&store, 11);
    let mut tx = Tx::begin(&store, 10);
    tx.set_node(people.alix, "city", Value::from("Berlin"))?;
    tx.remove_node_value(people.gus, "age")?;
    tx.add_label(people.mia, "Traveller")?;
    tx.remove_label(people.gus, "Employee")?;
    tx.set_edge(people.alix_gus, "since", Value::Int64(2019))?;
    tx.commit()?;
    same_data(
        &reader.view(),
        &before,
        "a transaction that began before the commit",
    )?;
    same_data(
        &image(&store, Reader::At(epoch)),
        &before,
        "a reader at the earlier epoch",
    )
}

/// A reader at an earlier epoch keeps the nodes and edges a later commit
/// deleted, with their adjacency.
pub(crate) fn a_reader_at_an_earlier_epoch_keeps_deleted_entities<S: Store>(
    make: &dyn Fn() -> S,
) -> Outcome {
    let store = make();
    let people = people(&store)?;
    let epoch = store.current_epoch();
    let before = committed(&store);
    let reader = Tx::begin(&store, 11);
    let mut tx = Tx::begin(&store, 10);
    tx.delete_edge(people.gus_mia)?;
    tx.delete_node(people.mia)?;
    tx.delete_edge(people.alix_gus)?;
    tx.commit()?;
    same_data(
        &reader.view(),
        &before,
        "a transaction that began before the commit",
    )?;
    same_data(
        &image(&store, Reader::At(epoch)),
        &before,
        "a reader at the earlier epoch",
    )
}

/// A delete takes effect at its commit epoch: a reader at an epoch after
/// the deleting transaction began but before it committed still sees the
/// node, and replay of the log agrees with the live store at every epoch
/// (inbox: a transactional delete is marked at the start epoch).
pub(crate) fn a_delete_is_marked_at_its_commit_epoch<S: Store>(make: &dyn Fn() -> S) -> Outcome {
    let store = make();
    let people = people(&store)?;
    let mut deleting = Tx::begin(&store, 10);
    deleting.delete_edge(people.gus_mia)?;
    deleting.delete_node(people.mia)?;

    let mut other = Tx::begin(&store, 11);
    other.set_node(people.alix, "city", Value::from("Paris"))?;
    let between = other.commit()?;
    let ops = deleting.ops();
    let commit = deleting.commit()?;

    let at_between = image(&store, Reader::At(between));
    ensure(at_between.nodes.contains_key(&people.mia.as_u64()), || {
        format!(
            "a reader at epoch {between:?}, before the delete committed at {commit:?}, sees Mia"
        )
    })?;
    ensure(
        at_between.edges.contains_key(&people.gus_mia.as_u64()),
        || "it sees the edge to Mia too".to_string(),
    )?;
    ensure(
        !image(&store, Reader::At(commit))
            .nodes
            .contains_key(&people.mia.as_u64()),
        || "a reader at the commit epoch does not see Mia".to_string(),
    )?;

    let replayed = make();
    let replayed_people = super::people(&replayed)?;
    apply_all(
        &replayed,
        &[DataOp::SetNodeProperty {
            id: replayed_people.alix,
            key: PropertyKey::new("city"),
            value: Value::from("Paris"),
        }],
        Writer::Replay { epoch: between },
    )?;
    apply_all(&replayed, &ops, Writer::Replay { epoch: commit })?;
    for epoch in [between, commit] {
        same_data(
            &image(&replayed, Reader::At(epoch)),
            &image(&store, Reader::At(epoch)),
            &format!("replay and the live store at epoch {epoch:?}"),
        )?;
    }
    Ok(())
}
