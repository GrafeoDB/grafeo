//! A store's epoch and its garbage collection, on every store: `sync_epoch`
//! moves the epoch forward only, `new_epoch` moves it past every write, and
//! `gc_versions` at a horizon leaves every reader at the horizon or later,
//! an open transaction included, what it saw (#412 S6). A store takes part
//! through [`Epochs`], which forwards to its own methods of the same names.

use grafeo_common::types::{EpochId, Value};

use super::cases::{create_and_delete_writes, value_and_label_writes};
use super::{KnownGap, Outcome, Reader, Store, Tx, committed, image, people, same};
use crate::graph::rowgroup::RowGroupStore;

/// A store with an epoch it moves and versions it collects.
pub(crate) trait Epochs: Store {
    fn sync_epoch(&self, epoch: EpochId);
    fn new_epoch(&self) -> EpochId;
    fn gc_versions(&self, min_epoch: EpochId) -> usize;
}

/// Implements [`Epochs`] for a store by its own methods.
macro_rules! epochs {
    ($store:ty) => {
        impl Epochs for $store {
            fn sync_epoch(&self, epoch: EpochId) {
                <$store>::sync_epoch(self, epoch);
            }
            fn new_epoch(&self) -> EpochId {
                <$store>::new_epoch(self)
            }
            fn gc_versions(&self, min_epoch: EpochId) -> usize {
                <$store>::gc_versions(self, min_epoch)
            }
        }
    };
}

epochs!(RowGroupStore);

/// Instantiates every epoch case for a store.
macro_rules! epoch_suite {
    ($store:ty, $make:expr, $gaps:expr) => {
        epoch_suite!(@cases $store, $make, $gaps;
            sync_epoch_only_moves_the_epoch_forward,
            new_epoch_moves_past_every_write,
            gc_versions_leaves_readers_at_the_horizon_what_they_saw,
            gc_versions_leaves_an_open_transactions_writes,
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

/// `sync_epoch` moves the store's epoch to a later epoch and leaves it at
/// an earlier one; the next commit lands after it.
pub(crate) fn sync_epoch_only_moves_the_epoch_forward<S: Epochs>(make: &dyn Fn() -> S) -> Outcome {
    let store = make();
    let people = people(&store)?;
    let start = store.current_epoch().as_u64();
    let later = EpochId::new(start + 18);
    store.sync_epoch(later);
    same(
        &store.current_epoch(),
        &later,
        "after a sync to a later epoch",
    )?;
    store.sync_epoch(EpochId::new(start));
    same(
        &store.current_epoch(),
        &later,
        "after a sync to an earlier epoch",
    )?;
    let mut tx = Tx::begin(&store, 10);
    tx.set_node(people.alix, "city", Value::from("Paris"))?;
    let epoch = tx.commit()?;
    same(
        &epoch,
        &EpochId::new(later.as_u64() + 1),
        "the commit after the sync",
    )?;
    same(&store.current_epoch(), &epoch, "the epoch after the commit")
}

/// `new_epoch` returns the epoch after the current one and moves the store
/// to it: a reader there sees every write so far, and a commit after it is
/// seen from its own epoch on.
pub(crate) fn new_epoch_moves_past_every_write<S: Epochs>(make: &dyn Fn() -> S) -> Outcome {
    let store = make();
    people(&store)?;
    let before = committed(&store);
    let start = store.current_epoch().as_u64();
    let fresh = store.new_epoch();
    same(&fresh, &EpochId::new(start + 1), "the new epoch")?;
    same(&store.current_epoch(), &fresh, "the store's epoch")?;
    same(
        &image(&store, Reader::At(fresh)),
        &before,
        "a reader at the new epoch",
    )?;
    let mut tx = Tx::begin(&store, 10);
    let jules = tx.create_node(&["Person"], &[("name", Value::from("Jules"))])?;
    let epoch = tx.commit()?;
    same(
        &epoch,
        &EpochId::new(fresh.as_u64() + 1),
        "the commit after it",
    )?;
    same(
        &image(&store, Reader::At(fresh)).nodes.get(&jules.as_u64()),
        &None,
        "Jules at the new epoch",
    )?;
    committed(&store).has_node(jules, &["Person"], "name", Some(&Value::from("Jules")))
}

/// After `gc_versions(horizon)`, a reader at the horizon, a transaction
/// that began there and a reader at a later commit see what they saw
/// before it.
pub(crate) fn gc_versions_leaves_readers_at_the_horizon_what_they_saw<S: Epochs>(
    make: &dyn Fn() -> S,
) -> Outcome {
    let store = make();
    let people = people(&store)?;
    let mut tx = Tx::begin(&store, 10);
    tx.set_node(people.mia, "city", Value::from("Berlin"))?;
    tx.add_label(people.gus, "Traveller")?;
    tx.set_edge(people.gus_mia, "since", Value::Int64(2019))?;
    tx.commit()?;
    let mut tx = Tx::begin(&store, 11);
    value_and_label_writes(&mut tx, &people)?;
    create_and_delete_writes(&mut tx, &people)?;
    let horizon = tx.commit()?;
    let reader = Tx::begin(&store, 12);
    let mut tx = Tx::begin(&store, 13);
    tx.set_node(people.mia, "city", Value::from("Prague"))?;
    tx.remove_label(people.gus, "Person")?;
    let later = tx.commit()?;

    let at_horizon = image(&store, Reader::At(horizon));
    let at_later = image(&store, Reader::At(later));
    let seen = reader.view();
    store.gc_versions(horizon);
    same(
        &image(&store, Reader::At(horizon)),
        &at_horizon,
        "a reader at the horizon",
    )?;
    same(&reader.view(), &seen, "the transaction at the horizon")?;
    same(
        &image(&store, Reader::At(later)),
        &at_later,
        "a reader at the later commit",
    )
}

/// `gc_versions` at the store's epoch while a transaction is open leaves it
/// its writes and every other reader what it saw, and the transaction
/// commits them afterwards.
pub(crate) fn gc_versions_leaves_an_open_transactions_writes<S: Epochs>(
    make: &dyn Fn() -> S,
) -> Outcome {
    let store = make();
    let people = people(&store)?;
    let mut tx = Tx::begin(&store, 10);
    value_and_label_writes(&mut tx, &people)?;
    create_and_delete_writes(&mut tx, &people)?;
    let own = tx.view();
    let others = committed(&store);
    store.gc_versions(store.current_epoch());
    same(&tx.view(), &own, "the open transaction")?;
    same(&committed(&store), &others, "the other readers")?;
    tx.commit()?;
    let now = committed(&store);
    same(
        &(&now.nodes, &now.edges),
        &(&own.nodes, &own.edges),
        "the committed state after its commit",
    )
}

/// The row-group store's epochs.
mod rowgroup {
    use super::{KnownGap, RowGroupStore};

    const GAPS: &[KnownGap] = &[];

    epoch_suite!(RowGroupStore, RowGroupStore::new, GAPS);
}
