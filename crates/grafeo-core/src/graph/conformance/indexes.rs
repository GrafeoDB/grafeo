//! The property index at a view, on every store: its equality and
//! maybe-equal lookups, each answering
//! what the reader's view holds (a transaction's own writes found, another
//! open one's not, a reader at an earlier epoch finding the values of its
//! epoch), the entries of an undone write gone, and each named graph's
//! indexes its own (#412 S7). A store takes part through [`Indexed`], which
//! forwards to its own methods of the same names; the text and vector
//! searches are in `searches`.

use std::collections::BTreeSet;

use grafeo_common::change::DataOp;
use grafeo_common::types::{NodeId, ReadView, Value};

use super::graphs::Registry;
use super::{
    KnownGap, Outcome, Store, Tx, apply_all, ensure, labels, next_epoch, people, properties,
    reserve_node, same,
};
use crate::graph::apply::Writer;
use crate::graph::rowgroup::RowGroupStore;

/// A store with a property index.
pub(crate) trait Indexed: Store {
    fn create_property_index(&self, property: &str);
    fn drop_property_index(&self, property: &str) -> bool;
    fn property_index_keys(&self) -> Vec<String>;
}

/// Implements [`Indexed`] for a store by its own methods.
macro_rules! indexed {
    ($store:ty) => {
        impl Indexed for $store {
            fn create_property_index(&self, property: &str) {
                <$store>::create_property_index(self, property);
            }
            fn drop_property_index(&self, property: &str) -> bool {
                <$store>::drop_property_index(self, property)
            }
            fn property_index_keys(&self) -> Vec<String> {
                <$store>::property_index_keys(self)
            }
        }
    };
}

indexed!(RowGroupStore);

/// Instantiates the cases `$case` of the module at path `$cases` for a
/// store.
macro_rules! index_cases {
    ($store:ty, $make:expr, $gaps:expr, $cases:ident; $($case:ident),* $(,)?) => {
        $(
            #[test]
            fn $case() {
                $crate::graph::conformance::run::<$store>(
                    stringify!($case),
                    $cases::$case::<$store>,
                    $make,
                    $gaps,
                );
            }
        )*
    };
}
#[cfg(any(feature = "text-index", feature = "vector-index"))]
pub(crate) use index_cases;

/// Instantiates every property index case for a store.
macro_rules! index_suite {
    ($store:ty, $make:expr, $gaps:expr) => {
        index_cases!($store, $make, $gaps, super;
            an_indexed_lookup_answers_at_its_view,
            an_undone_write_leaves_no_index_entry,
            maybe_equal_finds_every_spelling_at_a_view,
            indexes_are_created_listed_and_dropped,
            each_graph_keeps_its_own_indexes,
        );
    };
}

/// The ids of `ids`, as a set.
pub(crate) fn set(ids: &[NodeId]) -> BTreeSet<u64> {
    ids.iter().map(|id| id.as_u64()).collect()
}

/// The nodes `view` finds with `key` equal to `value`, through every
/// equality lookup: one key, two keys (with a condition every node meets
/// at every view, so the answer is the same), and maybe-equal (whose extra
/// answers a filter would drop: here only nodes equal by `=` qualify).
fn found<S: Indexed>(
    store: &S,
    key: &str,
    value: &Value,
    view: ReadView,
) -> Result<BTreeSet<u64>, String> {
    let one = set(&store.find_nodes_by_property_at(key, value, view));
    let two = set(&store.find_nodes_by_properties_at(
        &[(key, value.clone()), ("kind", Value::from("person"))],
        view,
    ));
    same(
        &two,
        &one,
        &format!("the two-key lookup of {key} = {value:?} at {view:?}"),
    )?;
    let maybe = store
        .find_nodes_maybe_equal_at(key, value, view)
        .ok_or_else(|| format!("no maybe-equal answer for {key} with an index"))?;
    let maybe: BTreeSet<u64> = maybe
        .into_iter()
        .filter(|id| store.node_property_at(*id, key, view).as_ref() == Some(value))
        .map(|id| id.as_u64())
        .collect();
    same(
        &maybe,
        &one,
        &format!("the maybe-equal lookup of {key} = {value:?} at {view:?}"),
    )?;
    Ok(one)
}

/// Alix, Gus and Mia with a `kind` every one has, indexed on `city` and
/// `name`.
fn indexed_people<S: Indexed>(store: &S) -> Result<super::People, String> {
    store.create_property_index("city");
    let people = people(store)?;
    store.create_property_index("name");
    let epoch = next_epoch(store);
    let ops: Vec<DataOp> = [people.alix, people.gus, people.mia]
        .into_iter()
        .map(|id| DataOp::SetNodeProperty {
            id,
            key: grafeo_common::types::PropertyKey::new("kind"),
            value: Value::from("person"),
        })
        .collect();
    apply_all(store, &ops, Writer::Replay { epoch })?;
    Ok(people)
}

/// Through the index: a transaction finds its own indexed write and not the
/// value it replaced, another open transaction and a committed reader find
/// the committed value and not the pending one, its own create is found by
/// it alone; after the commit a reader at an earlier epoch still finds the
/// old value and one at the commit the new one.
pub(crate) fn an_indexed_lookup_answers_at_its_view<S: Indexed>(make: &dyn Fn() -> S) -> Outcome {
    let store = make();
    let people = indexed_people(&store)?;
    let before = ReadView::committed(store.current_epoch());
    let (amsterdam, berlin, paris) = (
        Value::from("Amsterdam"),
        Value::from("Berlin"),
        Value::from("Paris"),
    );
    let mut writer = Tx::begin(&store, 10);
    let mut other = Tx::begin(&store, 11);
    writer.set_node(people.alix, "city", berlin.clone())?;
    let jules = writer.create_node(
        &["Person"],
        &[
            ("name", Value::from("Jules")),
            ("city", paris.clone()),
            ("kind", Value::from("person")),
        ],
    )?;
    other.set_node(people.gus, "city", paris.clone())?;
    let own = ReadView::transaction(writer.id, writer.snapshot);
    let others = ReadView::transaction(other.id, other.snapshot);

    same(
        &found(&store, "city", &berlin, own)?,
        &set(&[people.alix]),
        "the writer's Berlin",
    )?;
    same(
        &found(&store, "city", &amsterdam, own)?,
        &BTreeSet::new(),
        "the writer's Amsterdam",
    )?;
    same(
        &found(&store, "city", &paris, own)?,
        &set(&[jules]),
        "the writer's Paris",
    )?;
    same(
        &found(&store, "name", &Value::from("Jules"), own)?,
        &set(&[jules]),
        "the writer's Jules",
    )?;
    for (view, what) in [
        (others, "the other transaction"),
        (before, "a committed reader"),
    ] {
        same(
            &found(&store, "city", &berlin, view)?,
            &BTreeSet::new(),
            &format!("{what}'s Berlin"),
        )?;
        same(
            &found(&store, "city", &amsterdam, view)?,
            &set(&[people.alix]),
            &format!("{what}'s Amsterdam"),
        )?;
        same(
            &found(&store, "name", &Value::from("Jules"), view)?,
            &BTreeSet::new(),
            &format!("{what}'s Jules"),
        )?;
    }
    same(
        &found(&store, "city", &paris, others)?,
        &set(&[people.gus]),
        "the other's own Paris",
    )?;
    same(
        &found(&store, "city", &paris, before)?,
        &BTreeSet::new(),
        "a committed reader's Paris",
    )?;

    let committed = ReadView::committed(writer.commit()?);
    other.roll_back()?;
    same(
        &found(&store, "city", &amsterdam, before)?,
        &set(&[people.alix]),
        "Amsterdam before the commit",
    )?;
    same(
        &found(&store, "city", &berlin, before)?,
        &BTreeSet::new(),
        "Berlin before the commit",
    )?;
    same(
        &found(&store, "city", &berlin, committed)?,
        &set(&[people.alix]),
        "Berlin at the commit",
    )?;
    same(
        &found(&store, "city", &paris, committed)?,
        &set(&[jules]),
        "Paris at the commit",
    )?;
    same(
        &found(&store, "city", &amsterdam, committed)?,
        &BTreeSet::new(),
        "Amsterdam at the commit",
    )
}

/// A rolled-back write, and a write rolled back to a savepoint, leave no
/// entry: the index holds no node for the value they wrote.
pub(crate) fn an_undone_write_leaves_no_index_entry<S: Indexed>(make: &dyn Fn() -> S) -> Outcome {
    let store = make();
    let people = indexed_people(&store)?;
    let held = |city: &str| -> Result<BTreeSet<u64>, String> {
        store
            .find_nodes_maybe_equal("city", &Value::from(city))
            .map(|found| set(&found))
            .ok_or_else(|| "no answer from the index on city".to_string())
    };
    let mut tx = Tx::begin(&store, 10);
    tx.set_node(people.gus, "city", Value::from("Prague"))?;
    let mark = tx.mark();
    tx.set_node(people.mia, "city", Value::from("Paris"))?;
    let vincent = tx.create_node(&["Person"], &[("city", Value::from("Paris"))])?;
    ensure(held("Paris")?.contains(&vincent.as_u64()), || {
        "the pending create is in the index".to_string()
    })?;
    tx.roll_back_to(mark)?;
    same(
        &held("Paris")?,
        &BTreeSet::new(),
        "Paris after the savepoint rollback",
    )?;
    same(
        &held("Prague")?,
        &set(&[people.gus]),
        "Prague after the savepoint rollback",
    )?;
    tx.roll_back()?;
    same(
        &held("Prague")?,
        &BTreeSet::new(),
        "Prague after the rollback",
    )?;
    same(
        &held("Amsterdam")?,
        &set(&[people.alix]),
        "Amsterdam after the rollback",
    )
}

/// `=` finds a number in every spelling (an integer, a float, a numeric
/// string): the maybe-equal lookup at a view finds each node whose value
/// at the view `=` finds equal, and not one whose value changed away from
/// it, nor one a pending write moved to it, for the readers that do not
/// see the write.
pub(crate) fn maybe_equal_finds_every_spelling_at_a_view<S: Indexed>(
    make: &dyn Fn() -> S,
) -> Outcome {
    let store = make();
    store.create_property_index("code");
    let epoch = next_epoch(&store);
    let spellings = [
        Value::Int64(42),
        Value::Float64(42.0),
        Value::from("042"),
        Value::from("abc"),
    ];
    let mut nodes = Vec::new();
    let mut ops = Vec::new();
    for spelling in &spellings {
        let id = NodeId::new(reserve_node(&store)?);
        nodes.push(id);
        ops.push(DataOp::CreateNode {
            id,
            labels: labels(&["Person"]),
            properties: properties(&[("code", spelling.clone())]),
        });
    }
    apply_all(&store, &ops, Writer::Replay { epoch })?;
    let before = ReadView::committed(store.current_epoch());
    let mut tx = Tx::begin(&store, 10);
    tx.set_node(nodes[0], "code", Value::from("xyz"))?;
    tx.set_node(nodes[3], "code", Value::from("42"))?;
    let own = ReadView::transaction(tx.id, tx.snapshot);
    let maybe = |view: ReadView, key: &Value| -> Result<BTreeSet<u64>, String> {
        store
            .find_nodes_maybe_equal_at("code", key, view)
            .map(|found| set(&found))
            .ok_or_else(|| "no answer from the index on code".to_string())
    };
    for key in [Value::Int64(42), Value::Float64(42.0), Value::from("42.0")] {
        let at_before = maybe(before, &key)?;
        ensure(
            [nodes[0], nodes[1], nodes[2]]
                .iter()
                .all(|id| at_before.contains(&id.as_u64()))
                && !at_before.contains(&nodes[3].as_u64()),
            || {
                format!(
                    "{key:?} before the write finds {at_before:?}, expected the three spellings of 42"
                )
            },
        )?;
        let at_own = maybe(own, &key)?;
        ensure(
            [nodes[1], nodes[2], nodes[3]]
                .iter()
                .all(|id| at_own.contains(&id.as_u64()))
                && !at_own.contains(&nodes[0].as_u64()),
            || format!("{key:?} in the transaction finds {at_own:?}, expected the 42s it sees"),
        )?;
    }
    same(
        &maybe(before, &Value::from("xyz"))?,
        &BTreeSet::new(),
        "xyz before the write",
    )?;
    same(
        &maybe(own, &Value::from("xyz"))?,
        &set(&[nodes[0]]),
        "xyz in the transaction",
    )?;
    same(
        &maybe(own, &Value::Null)?,
        &BTreeSet::new(),
        "NULL equals nothing",
    )
}

/// Property, text and vector indexes are listed once created and gone once
/// dropped; a second create of a property index changes nothing.
pub(crate) fn indexes_are_created_listed_and_dropped<S: Indexed>(make: &dyn Fn() -> S) -> Outcome {
    let store = make();
    let people = people(&store)?;
    store.create_property_index("city");
    store.create_property_index("city");
    same(
        &store.property_index_keys(),
        &vec!["city".to_string()],
        "the indexed keys",
    )?;
    ensure(
        store.has_property_index("city") && !store.has_property_index("name"),
        || "has_property_index".to_string(),
    )?;
    same(
        &store
            .find_nodes_maybe_equal("city", &Value::from("Amsterdam"))
            .map(|found| set(&found)),
        &Some(set(&[people.alix])),
        "a lookup through the index",
    )?;
    same(
        &store.find_nodes_maybe_equal("name", &Value::from("Alix")),
        &None,
        "a key without an index",
    )?;
    ensure(
        store.drop_property_index("city") && !store.drop_property_index("city"),
        || "drop_property_index returns whether it dropped".to_string(),
    )?;
    ensure(!store.has_property_index("city"), || {
        "the dropped index is gone".to_string()
    })?;
    Ok(())
}

/// A named graph's indexes are its own: an index of the root is not one of
/// the graph's, a lookup through the graph's index finds the graph's nodes
/// alone, and a write to the root never reaches the graph's index.
pub(crate) fn each_graph_keeps_its_own_indexes<S: Indexed + Registry>(
    make: &dyn Fn() -> S,
) -> Outcome {
    let root = make();
    let people = people(&root)?;
    root.create_property_index("name");
    let graph = root
        .graph_or_create("travel")
        .map_err(|error| format!("graph_or_create: {error}"))?;
    ensure(!graph.has_property_index("name"), || {
        "the root's index is not the graph's".to_string()
    })?;
    graph.create_property_index("city");
    ensure(!root.has_property_index("city"), || {
        "the graph's index is not the root's".to_string()
    })?;
    let mut tx = Tx::begin(&*graph, 10);
    let berlin = tx.create_node(&["City"], &[("city", Value::from("Amsterdam"))])?;
    let view = ReadView::committed(tx.commit()?);
    same(
        &set(&graph.find_nodes_by_property_at("city", &Value::from("Amsterdam"), view)),
        &set(&[berlin]),
        "the graph's Amsterdam",
    )?;
    let root_view = ReadView::committed(root.current_epoch());
    same(
        &set(&root.find_nodes_by_property_at("city", &Value::from("Amsterdam"), root_view)),
        &set(&[people.alix]),
        "the root's Amsterdam",
    )?;
    same(
        &graph
            .find_nodes_maybe_equal("city", &Value::from("Amsterdam"))
            .map(|found| set(&found)),
        &Some(set(&[berlin])),
        "the graph's index holds the graph's node alone",
    )
}

/// The row-group store's index cases: it answers every one at its view.
mod rowgroup {
    use super::{KnownGap, RowGroupStore};

    const GAPS: &[KnownGap] = &[];

    index_suite!(RowGroupStore, RowGroupStore::new, GAPS);
}
