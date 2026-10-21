//! Named graphs, on every store: create, list, count, look up, drop,
//! `graph_or_create`, graph ids never given twice, a named graph's data its
//! own, and the restore a load runs. A store takes part through
//! [`Registry`], which forwards to its own methods of the same names.

use std::collections::BTreeSet;
use std::sync::Arc;

use grafeo_common::change::DataOp;
use grafeo_common::memory::arena::AllocError;
use grafeo_common::types::{EpochId, NodeId, PropertyKey, Value};

use super::{KnownGap, Outcome, Store, Tx, committed, ensure, labels, properties, same};
use crate::graph::apply::{ApplyError, Writer};
use crate::graph::rowgroup::RowGroupStore;

/// A store that is a root of named graphs, each a store of the same kind.
pub(crate) trait Registry: Store + Sized {
    fn graph(&self, name: &str) -> Option<Arc<Self>>;
    fn graph_or_create(&self, name: &str) -> Result<Arc<Self>, AllocError>;
    fn create_graph(&self, name: &str) -> Result<bool, AllocError>;
    fn drop_graph(&self, name: &str) -> bool;
    fn is_dropped(&self) -> bool;
    fn graph_names(&self) -> Vec<String>;
    fn graph_count(&self) -> usize;
    fn graph_id(&self) -> u32;
    fn next_graph_id(&self) -> u32;
    fn restore_graphs(
        &self,
        graphs: &[(u32, &str)],
        next_graph_id: u32,
    ) -> Result<Vec<Arc<Self>>, String>;
}

/// Implements [`Registry`] for a store by its own methods.
macro_rules! registry {
    ($store:ty) => {
        impl Registry for $store {
            fn graph(&self, name: &str) -> Option<Arc<Self>> {
                <$store>::graph(self, name)
            }
            fn graph_or_create(&self, name: &str) -> Result<Arc<Self>, AllocError> {
                <$store>::graph_or_create(self, name)
            }
            fn create_graph(&self, name: &str) -> Result<bool, AllocError> {
                <$store>::create_graph(self, name)
            }
            fn drop_graph(&self, name: &str) -> bool {
                <$store>::drop_graph(self, name)
            }
            fn is_dropped(&self) -> bool {
                <$store>::is_dropped(self)
            }
            fn graph_names(&self) -> Vec<String> {
                <$store>::graph_names(self)
            }
            fn graph_count(&self) -> usize {
                <$store>::graph_count(self)
            }
            fn graph_id(&self) -> u32 {
                <$store>::graph_id(self)
            }
            fn next_graph_id(&self) -> u32 {
                <$store>::next_graph_id(self)
            }
            fn restore_graphs(
                &self,
                graphs: &[(u32, &str)],
                next_graph_id: u32,
            ) -> Result<Vec<Arc<Self>>, String> {
                <$store>::restore_graphs(self, graphs, next_graph_id)
            }
        }
    };
}

registry!(RowGroupStore);

/// Instantiates every named-graph case for a store.
macro_rules! graph_suite {
    ($store:ty, $make:expr, $gaps:expr) => {
        graph_suite!(@cases $store, $make, $gaps;
            named_graphs_are_created_listed_and_counted,
            graph_or_create_returns_the_graph_or_creates_it,
            a_dropped_graph_refuses_writes_and_leaves_the_list,
            graph_ids_are_never_given_twice,
            a_named_graphs_data_is_its_own,
            restore_graphs_gives_the_ids_and_the_next_id,
            restore_graphs_refuses_ids_outside_the_named_range,
        );
    };
    (@cases $store:ty, $make:expr, $gaps:expr; $($case:ident),* $(,)?) => {
        $(
            #[test]
            fn $case() {
                $crate::graph::conformance::run::<$store>(
                    stringify!($case),
                    super::$case::<$store>,
                    $make,
                    $gaps,
                );
            }
        )*
    };
}

/// The error of a registry call that failed.
fn created(result: Result<bool, AllocError>, name: &str) -> Result<bool, String> {
    result.map_err(|error| format!("create_graph({name:?}): {error}"))
}

/// The names, sorted: `graph_names` lists in no promised order (each
/// store's map has a hash seed of its own).
fn names<S: Registry>(root: &S) -> Vec<String> {
    let mut names = root.graph_names();
    names.sort();
    names
}

/// The ids of `names`' graphs, in their order (`None` for a name not held).
fn ids<S: Registry>(root: &S, names: &[&str]) -> Vec<Option<u32>> {
    names
        .iter()
        .map(|name| root.graph(name).map(|graph| graph.graph_id()))
        .collect()
}

/// The ids the committed image of `store` holds: its nodes and edges.
fn data<S: Store>(store: &S) -> (Vec<u64>, Vec<u64>) {
    let image = committed(store);
    (
        image.nodes.keys().copied().collect(),
        image.edges.keys().copied().collect(),
    )
}

/// A new root has no named graphs and is graph 0; each created graph is
/// listed, counted and found by name, a second create of a name changes
/// nothing, and the ids follow the creates from 1.
pub(crate) fn named_graphs_are_created_listed_and_counted<S: Registry>(
    make: &dyn Fn() -> S,
) -> Outcome {
    let root = make();
    same(&root.graph_count(), &0, "a new root's graphs")?;
    same(&root.graph_names(), &Vec::new(), "a new root's names")?;
    same(
        &(root.graph_id(), root.next_graph_id()),
        &(0, 1),
        "a new root's ids",
    )?;
    ensure(!root.is_dropped(), || "a new root is dropped".to_string())?;

    for name in ["trips", "travel", "archive"] {
        ensure(created(root.create_graph(name), name)?, || {
            format!("{name} was not created")
        })?;
    }
    ensure(!created(root.create_graph("trips"), "trips")?, || {
        "a second create of trips reports a create".to_string()
    })?;
    same(&root.graph_count(), &3, "the graphs counted")?;
    same(
        &names(&root),
        &vec!["archive".to_string(), "travel".into(), "trips".into()],
        "the names listed",
    )?;
    same(
        &ids(&root, &["trips", "travel", "archive", "Amsterdam"]),
        &vec![Some(1), Some(2), Some(3), None],
        "the ids, by create",
    )?;
    same(&root.next_graph_id(), &4, "the next graph id")?;
    let (first, again) = (root.graph("trips"), root.graph("trips"));
    ensure(
        matches!((&first, &again), (Some(a), Some(b)) if Arc::ptr_eq(a, b)),
        || "two lookups of trips give two stores".to_string(),
    )?;
    let trips = first.expect("trips");
    same(&trips.graph_count(), &0, "a named graph's own graphs")?;
    same(&root.graph_id(), &0, "the root after the creates")
}

/// `graph_or_create` gives the graph a name holds, created by either call,
/// and creates one the first time, with the next id.
pub(crate) fn graph_or_create_returns_the_graph_or_creates_it<S: Registry>(
    make: &dyn Fn() -> S,
) -> Outcome {
    let root = make();
    let error = |error: AllocError| format!("graph_or_create: {error}");
    let trips = root.graph_or_create("trips").map_err(error)?;
    same(
        &(root.graph_count(), trips.graph_id()),
        &(1, 1),
        "trips created",
    )?;
    let again = root.graph_or_create("trips").map_err(error)?;
    ensure(Arc::ptr_eq(&trips, &again), || {
        "a second graph_or_create gave another store".to_string()
    })?;
    created(root.create_graph("travel"), "travel")?;
    let travel = root.graph_or_create("travel").map_err(error)?;
    ensure(
        root.graph("travel")
            .is_some_and(|graph| Arc::ptr_eq(&graph, &travel)),
        || "graph_or_create gave another store than create_graph made".to_string(),
    )?;
    same(
        &(root.graph_count(), travel.graph_id(), root.next_graph_id()),
        &(2, 2, 3),
        "after travel",
    )
}

/// A dropped graph leaves the list and the count, a second drop reports
/// none, and its store, also a handle resolved before the drop, refuses
/// every writer's write as `Refused`, changing nothing. The name created
/// again is a new store, empty, with a new id, that takes writes.
pub(crate) fn a_dropped_graph_refuses_writes_and_leaves_the_list<S: Registry>(
    make: &dyn Fn() -> S,
) -> Outcome {
    let root = make();
    created(root.create_graph("trips"), "trips")?;
    created(root.create_graph("travel"), "travel")?;
    let trips = root.graph("trips").ok_or("trips is not found")?;
    let mut tx = Tx::begin(&*trips, 3);
    let alix = tx.create_node(&["Person"], &[("name", Value::from("Alix"))])?;
    tx.commit()?;

    ensure(root.drop_graph("trips"), || {
        "trips was not dropped".to_string()
    })?;
    ensure(!root.drop_graph("trips"), || {
        "a second drop of trips reports a drop".to_string()
    })?;
    ensure(!root.drop_graph("Berlin"), || {
        "a drop of a name never created reports a drop".to_string()
    })?;
    ensure(trips.is_dropped(), || {
        "the handle resolved before the drop is not dropped".to_string()
    })?;
    ensure(!root.is_dropped(), || "the root is dropped".to_string())?;
    ensure(root.graph("trips").is_none(), || {
        "trips is found after its drop".to_string()
    })?;
    same(
        &names(&root),
        &vec!["travel".to_string()],
        "the names after the drop",
    )?;
    same(&root.graph_count(), &1, "the count after the drop")?;

    let before = data(&*trips);
    let epoch = EpochId::new(trips.current_epoch().as_u64() + 1);
    let gus = NodeId::new(super::reserve_node(&*trips)?);
    let writers = [
        Writer::Transaction {
            id: grafeo_common::types::TransactionId::new(19),
            snapshot: trips.current_epoch(),
        },
        Writer::Immediate {
            epoch,
            before_images: true,
        },
        Writer::Replay { epoch },
    ];
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
            let outcome = trips.apply(&op, writer);
            ensure(matches!(outcome, Err(ApplyError::Refused(_))), || {
                format!("{writer:?} wrote {op:?} into a dropped graph: {outcome:?}")
            })?;
        }
    }
    same(&data(&*trips), &before, "the refused writes")?;
    committed(&*trips).has_node(alix, &["Person"], "city", None)?;

    ensure(created(root.create_graph("trips"), "trips")?, || {
        "trips was not created again".to_string()
    })?;
    let again = root.graph("trips").ok_or("trips is not found again")?;
    ensure(!Arc::ptr_eq(&trips, &again) && !again.is_dropped(), || {
        "trips created again is the dropped store".to_string()
    })?;
    same(&again.graph_id(), &3, "the id of trips created again")?;
    same(
        &data(&*again),
        &(Vec::new(), Vec::new()),
        "trips created again",
    )?;
    let mut tx = Tx::begin(&*again, 4);
    tx.create_node(&["Person"], &[("name", Value::from("Gus"))])?;
    tx.commit()?;
    same(&data(&*again).0.len(), &1, "a write to trips created again")
}

/// Graph ids only grow: a drop frees no id, the name created again gets
/// the next one, and dropping every graph keeps the next id.
pub(crate) fn graph_ids_are_never_given_twice<S: Registry>(make: &dyn Fn() -> S) -> Outcome {
    let root = make();
    created(root.create_graph("trips"), "trips")?;
    created(root.create_graph("travel"), "travel")?;
    root.drop_graph("travel");
    created(root.create_graph("archive"), "archive")?;
    created(root.create_graph("travel"), "travel")?;
    same(
        &ids(&root, &["trips", "archive", "travel"]),
        &vec![Some(1), Some(3), Some(4)],
        "the ids after a drop",
    )?;
    for name in root.graph_names() {
        root.drop_graph(&name);
    }
    same(
        &(root.graph_count(), root.next_graph_id()),
        &(0, 5),
        "every graph dropped",
    )?;
    root.graph_or_create("trips")
        .map_err(|error| format!("graph_or_create: {error}"))?;
    same(
        &ids(&root, &["trips"]),
        &vec![Some(5)],
        "trips created last",
    )
}

/// A named graph is a store of its own: writes through its change target
/// change neither the root nor a sibling, each graph gives its own ids from
/// 0, and each commits at its own epochs.
pub(crate) fn a_named_graphs_data_is_its_own<S: Registry>(make: &dyn Fn() -> S) -> Outcome {
    let root = make();
    let people = super::people(&root)?;
    let before = committed(&root);
    created(root.create_graph("trips"), "trips")?;
    created(root.create_graph("travel"), "travel")?;
    let (trips, travel) = (
        root.graph("trips").ok_or("trips is not found")?,
        root.graph("travel").ok_or("travel is not found")?,
    );

    let mut tx = Tx::begin(&*trips, 10);
    let vincent = tx.create_node(&["Person"], &[("name", Value::from("Vincent"))])?;
    let paris = tx.create_node(&["City"], &[("name", Value::from("Paris"))])?;
    let visit = tx.create_edge(vincent, paris, "VISITED", &[])?;
    tx.commit()?;
    let mut tx = Tx::begin(&*travel, 11);
    let jules = tx.create_node(&["Person"], &[("name", Value::from("Jules"))])?;
    tx.commit()?;
    let mut tx = Tx::begin(&*trips, 12);
    tx.set_node(vincent, "city", Value::from("Berlin"))?;
    tx.commit()?;

    let now = committed(&root);
    same(&now.nodes, &before.nodes, "the root's nodes")?;
    same(&now.edges, &before.edges, "the root's edges")?;
    same(&root.node_count(), &3, "the root's node count")?;
    same(
        &data(&*trips),
        &(vec![vincent.as_u64(), paris.as_u64()], vec![visit.as_u64()]),
        "trips",
    )?;
    same(
        &(vincent.as_u64(), visit.as_u64(), people.alix.as_u64()),
        &(0, 0, 0),
        "each graph gives ids from 0",
    )?;
    committed(&*trips).has_node(vincent, &["Person"], "city", Some(&Value::from("Berlin")))?;
    same(
        &data(&*travel),
        &(vec![jules.as_u64()], Vec::new()),
        "travel",
    )?;
    same(
        &(trips.node_count(), trips.edge_count(), travel.node_count()),
        &(2, 1, 1),
        "the named graphs' counts",
    )?;
    same(
        &(trips.current_epoch(), travel.current_epoch()),
        &(EpochId::new(2), EpochId::new(1)),
        "each graph's epoch",
    )?;
    let labels: BTreeSet<String> = trips.all_labels().into_iter().collect();
    same(
        &labels,
        &BTreeSet::from(["City".to_string(), "Person".into()]),
        "trips' labels",
    )
}

/// The restore a load runs: a listed graph takes the given id, also one
/// created before it (the same store stays), a new one is created at it, a
/// graph not listed moves past the given ids, the stores come back in the
/// order given, and the next id is the given one at least; a graph created
/// after the restore gets the next id, never a gap's.
pub(crate) fn restore_graphs_gives_the_ids_and_the_next_id<S: Registry>(
    make: &dyn Fn() -> S,
) -> Outcome {
    let root = make();
    created(root.create_graph("reporting"), "reporting")?;
    created(root.create_graph("scratch"), "scratch")?;
    let reporting = root.graph("reporting").ok_or("reporting is not found")?;
    let restored = root.restore_graphs(&[(2, "reporting"), (1, "social"), (5, "archive")], 7)?;
    same(&restored.len(), &3, "the stores restored")?;
    ensure(Arc::ptr_eq(&restored[0], &reporting), || {
        "the graph created before the restore is not the one restored".to_string()
    })?;
    ensure(
        root.graph("social")
            .is_some_and(|graph| Arc::ptr_eq(&graph, &restored[1])),
        || "social is not the store restored for it".to_string(),
    )?;
    same(
        &restored
            .iter()
            .map(|graph| graph.graph_id())
            .collect::<Vec<_>>(),
        &vec![2, 1, 5],
        "the ids given",
    )?;
    same(
        &(ids(&root, &["scratch"]), root.next_graph_id()),
        &(vec![Some(7)], 8),
        "the unlisted graph past the ids, and the next id",
    )?;
    same(&root.graph_count(), &4, "the graphs after the restore")?;
    created(root.create_graph("travel"), "travel")?;
    same(
        &ids(&root, &["travel"]),
        &vec![Some(8)],
        "travel created after",
    )?;

    // A next id below the store's own does not move it back: the four
    // graphs not listed move past the store's own.
    let earlier = root.restore_graphs(&[(1, "social")], 2)?;
    same(
        &(earlier[0].graph_id(), root.next_graph_id()),
        &(1, 13),
        "a restore with an earlier next id",
    )?;
    let all: BTreeSet<Option<u32>> = ids(&root, &["reporting", "scratch", "archive", "travel"])
        .into_iter()
        .collect();
    same(
        &all,
        &(9..13).map(Some).collect(),
        "the graphs not listed, past the store's next id",
    )?;
    let mut tx = Tx::begin(&*restored[1], 3);
    tx.create_node(&["Person"], &[("name", Value::from("Mia"))])?;
    tx.commit()?;
    same(
        &data(&*restored[1]).0.len(),
        &1,
        "a write to a restored graph",
    )
}

/// A restore refuses id 0 (the default graph's) and an id at or above the
/// next id, and changes nothing then.
pub(crate) fn restore_graphs_refuses_ids_outside_the_named_range<S: Registry>(
    make: &dyn Fn() -> S,
) -> Outcome {
    let root = make();
    created(root.create_graph("trips"), "trips")?;
    for (graphs, next) in [
        (vec![(1, "social"), (0, "travel")], 4),
        (vec![(1, "social"), (4, "travel")], 4),
        (vec![(1, "social")], 1),
    ] {
        ensure(root.restore_graphs(&graphs, next).is_err(), || {
            format!("the restore of {graphs:?} below {next} is not refused")
        })?;
    }
    same(
        &names(&root),
        &vec!["trips".to_string()],
        "the names after the refusals",
    )?;
    same(
        &(ids(&root, &["trips"]), root.next_graph_id()),
        &(vec![Some(1)], 2),
        "the ids after the refusals",
    )
}

/// The row-group store's named graphs.
mod rowgroup {
    use super::{KnownGap, RowGroupStore};

    const GAPS: &[KnownGap] = &[];

    graph_suite!(RowGroupStore, RowGroupStore::new, GAPS);
}

/// The row-group store's named graphs on a root that keeps no incoming
/// edges (10b), whose named graphs keep none either.
mod rowgroup_without_backward_edges {
    use super::{KnownGap, RowGroupStore};

    const GAPS: &[KnownGap] = &[];

    fn store() -> RowGroupStore {
        RowGroupStore::with_backward_edges(false)
    }

    graph_suite!(RowGroupStore, store, GAPS);
}
