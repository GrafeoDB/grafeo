//! The store's epoch and its garbage collection, with the names and
//! behavior of `LpgStore`'s (#412 S6), so the engine can switch stores
//! without renaming a call.
//!
//! The store's epoch is the latest one a write was stamped at: a commit's
//! `stamp`, an immediate write and replay move it as they stamp, and
//! [`sync_epoch`](RowGroupStore::sync_epoch) moves it to an epoch the engine
//! decided (a direct call's, a load's). [`gc_versions`](RowGroupStore::gc_versions)
//! is [`collect`](RowGroupStore::collect) at the readers' horizon, over the
//! store and its named graphs.
//!
//! An immediate write keeps what it replaces, as a transaction's does: the
//! store does not know its readers, and a reader at an earlier epoch must
//! not see the write. The engine asks for a collect after each immediate
//! call and each commit, at its read horizon
//! ([`gc_versions_after_write`](RowGroupStore::gc_versions_after_write)), so
//! what such a write kept lasts as long as an older reader needs it, and no
//! longer than a bounded number of further writes once none does (decision
//! B, amortized: `collect`); the collect visits only the vectors that hold
//! versions.

use std::sync::atomic::Ordering;

use grafeo_common::types::EpochId;

use super::RowGroupStore;

impl RowGroupStore {
    /// Returns the current epoch: the latest one a write was stamped at, or
    /// the store was moved to.
    #[must_use]
    pub fn current_epoch(&self) -> EpochId {
        EpochId::new(self.epoch.load(Ordering::Acquire))
    }

    /// Moves the store's epoch one past the current one and returns it, as
    /// `LpgStore::new_epoch` does: a reader at the returned epoch sees every
    /// write stamped so far, and the next immediate write stamps after it.
    ///
    /// Commits, immediate writes and replay move the epoch as they stamp, so
    /// the engine needs no call of this; it is kept for the callers that read
    /// at a fresh epoch between raw writes, as tests do.
    #[doc(hidden)]
    pub fn new_epoch(&self) -> EpochId {
        EpochId::new(self.epoch.fetch_add(1, Ordering::AcqRel) + 1)
    }

    /// Moves the store's epoch to `epoch` if it is later; an earlier one
    /// changes nothing.
    ///
    /// The engine calls it as it calls `LpgStore::sync_epoch`: to move a
    /// store to a direct call's epoch before the call writes, to follow a
    /// commit that wrote another graph (the database has one epoch), and
    /// after a load or a recovery.
    #[doc(hidden)]
    pub fn sync_epoch(&self, epoch: EpochId) {
        self.epoch.fetch_max(epoch.as_u64(), Ordering::AcqRel);
    }

    /// Whether the engine runs
    /// [`gc_versions_after_write`](Self::gc_versions_after_write) after every
    /// immediate call and commit, at its read horizon: this store does, as
    /// an immediate write keeps what it replaces; the call collects when a
    /// collect is due and returns at once otherwise (`LpgStore` is collected
    /// every `gc_interval` commits).
    #[doc(hidden)]
    pub const COLLECTS_AFTER_EVERY_WRITE: bool = true;

    /// What the engine runs after each immediate call and commit, at its
    /// read horizon `min_epoch`: [`collect_after_write`](Self::collect_after_write)
    /// of this store and of every named graph that lists a vector, each
    /// collecting only when a collect is due for it (once a bounded number
    /// of writes came since its last collect, or a reader that held its last
    /// collect back ended). Returns how many versions it dropped, as
    /// [`gc_versions`](Self::gc_versions) does; a call on a named graph
    /// covers that graph alone.
    #[doc(hidden)]
    pub fn gc_versions_after_write(&self, min_epoch: EpochId) -> usize {
        let mut dropped = self.collect_after_write(min_epoch);
        for graph in self.listing_graphs() {
            dropped += graph.gc_versions_after_write(min_epoch);
        }
        dropped
    }

    /// Garbage collects what no reader at `min_epoch` or later can see any
    /// more, in this store and every named graph it holds: a
    /// [`collect`](Self::collect) of each at `min_epoch`, the oldest epoch a
    /// reader still reads at (the engine's read horizon).
    ///
    /// Returns how many versions it dropped: the rows taken out, and the
    /// values and label sets kept for older readers. Unlike `LpgStore`'s,
    /// which collects its own versions only, it covers the named graphs: a
    /// call on a named graph collects that graph alone.
    ///
    /// Each store visits only the vectors it lists, and one that lists none
    /// costs a flag test. It collects whether or not a collect is due: the
    /// engine's explicit collect; after a write the engine runs
    /// [`gc_versions_after_write`](Self::gc_versions_after_write).
    #[doc(hidden)]
    pub fn gc_versions(&self, min_epoch: EpochId) -> usize {
        let collected = self.collect_versions(min_epoch);
        let mut dropped = collected.rows + collected.values;
        for graph in self.listing_graphs() {
            dropped += graph.gc_versions(min_epoch);
        }
        dropped
    }
}

#[cfg(test)]
mod tests {
    use grafeo_common::change::{DataOp, Table};
    use grafeo_common::types::{ArcStr, EpochId, NodeId, PropertyKey, ReadView, Value};

    use crate::graph::apply::Writer;
    use crate::graph::conformance::{Tx, apply_all, next_epoch, people};
    use crate::graph::rowgroup::RowGroupStore;
    use crate::graph::traits::GraphStore;

    /// No vector, as `listed_vectors` lists them.
    const NOTHING: [(Table, u64); 0] = [];

    /// Gus turns 40 and becomes a `Traveller`, as immediate writes at
    /// `epoch`.
    fn gus_travels(store: &RowGroupStore, gus: NodeId, epoch: EpochId) {
        let ops = [
            DataOp::SetNodeProperty {
                id: gus,
                key: PropertyKey::new("age"),
                value: Value::Int64(40),
            },
            DataOp::AddNodeLabel {
                id: gus,
                label: ArcStr::from("Traveller"),
            },
        ];
        apply_all(
            store,
            &ops,
            Writer::Immediate {
                epoch,
                before_images: true,
            },
        )
        .unwrap();
    }

    /// Gus's age and whether he is a `Traveller`, at `view`.
    fn gus_at(store: &RowGroupStore, gus: NodeId, view: ReadView) -> (Option<Value>, bool) {
        let traveller = store.get_node_at(gus, view).is_some_and(|node| {
            node.labels
                .iter()
                .any(|label| label.as_str() == "Traveller")
        });
        (store.node_property_at(gus, "age", view), traveller)
    }

    /// An immediate write keeps what it replaces, whatever readers there
    /// are; a collect right after it at its epoch, with no reader older,
    /// leaves the version info as flat as before, with nothing listed.
    #[test]
    fn an_immediate_write_collected_at_its_epoch_leaves_the_versions_flat() {
        let store = RowGroupStore::new();
        let people = people(&store).unwrap();
        store.collect(store.current_epoch());
        let flat = store.version_bytes();
        assert_eq!(store.listed_vectors(), NOTHING, "nothing to collect");
        let epoch = next_epoch(&store);
        gus_travels(&store, people.gus, epoch);
        assert_eq!(store.update_nodes(), 2, "the age and the labels");
        assert_eq!(store.listed_vectors(), [(Table::Nodes, 0)], "Gus's vector");
        assert_eq!(store.collect(epoch), 0, "no row goes");
        assert_eq!(store.update_nodes(), 0);
        assert_eq!(store.listed_vectors(), NOTHING);
        assert_eq!(store.version_bytes(), flat, "a row costs what it did");
        assert_eq!(
            gus_at(&store, people.gus, ReadView::committed(epoch)),
            (Some(Value::Int64(40)), true)
        );
    }

    /// A reader before an immediate write's epoch, pinned at the horizon,
    /// keeps what it read through every collect until it is gone and the
    /// next collect runs at a later horizon.
    #[test]
    fn an_older_reader_keeps_its_values_until_it_unpins_and_the_next_collect() {
        let store = RowGroupStore::new();
        let people = people(&store).unwrap();
        store.collect(store.current_epoch());
        let flat = store.version_bytes();
        let pinned = store.current_epoch();
        let epoch = next_epoch(&store);
        gus_travels(&store, people.gus, epoch);
        // The collect right after the write, while the reader is pinned.
        assert_eq!(store.collect(pinned), 0);
        assert_eq!(store.update_nodes(), 2, "the reader needs them");
        assert_eq!(store.listed_vectors(), [(Table::Nodes, 0)], "still listed");
        assert_eq!(
            gus_at(&store, people.gus, ReadView::committed(pinned)),
            (Some(Value::Int64(3)), false),
            "the pinned reader"
        );
        assert_eq!(
            gus_at(&store, people.gus, ReadView::committed(epoch)),
            (Some(Value::Int64(40)), true)
        );
        // The reader unpins; the next collect runs at the write's epoch.
        store.collect(epoch);
        assert_eq!(store.update_nodes(), 0);
        assert_eq!(store.listed_vectors(), NOTHING);
        assert_eq!(store.version_bytes(), flat);
    }

    /// A transaction's write keeps what it replaces until its commit is
    /// collected: a collect while it is open leaves it, and its vector
    /// listed.
    #[test]
    fn a_transactions_write_keeps_the_old_value_until_its_commit_is_collected() {
        let store = RowGroupStore::new();
        let people = people(&store).unwrap();
        store.collect(store.current_epoch());
        let mut tx = Tx::begin(&store, 10);
        tx.set_node(people.gus, "age", Value::Int64(40)).unwrap();
        tx.add_label(people.gus, "Traveller").unwrap();
        store.collect(EpochId::new(1_000));
        assert_eq!(store.update_nodes(), 2, "the open transaction's");
        assert_eq!(store.listed_vectors(), [(Table::Nodes, 0)]);
        assert_eq!(
            gus_at(
                &store,
                people.gus,
                ReadView::committed(store.current_epoch())
            ),
            (Some(Value::Int64(3)), false),
            "another reader"
        );
        let epoch = tx.commit().unwrap();
        store.collect(epoch);
        assert_eq!(store.update_nodes(), 0);
        assert_eq!(store.listed_vectors(), NOTHING);
    }

    /// A named graph whose collect an older reader held off is collected by
    /// the root's next `gc_versions`, though only the root was written
    /// since; a root with nothing listed anywhere returns at once.
    #[test]
    fn gc_versions_on_the_root_collects_a_named_graph_an_older_reader_held_off() {
        let root = RowGroupStore::new();
        let people = people(&root).unwrap();
        let trips = root.graph_or_create("trips").unwrap();
        let mut tx = Tx::begin(&*trips, 10);
        let vincent = tx
            .create_node(&["Person"], &[("city", Value::from("Paris"))])
            .unwrap();
        let created = tx.commit().unwrap();
        assert_eq!(root.gc_versions(created), 0, "a create drops no version");
        assert_eq!(trips.listed_vectors(), NOTHING);

        let pinned = created;
        let mut tx = Tx::begin(&*trips, 11);
        tx.set_node(vincent, "city", Value::from("Berlin")).unwrap();
        let moved = tx.commit().unwrap();
        root.sync_epoch(moved);
        assert_eq!(root.gc_versions(pinned), 0, "the reader at the create");
        assert_eq!(trips.update_nodes(), 1);

        let epoch = next_epoch(&root);
        gus_travels(&root, people.gus, epoch);
        assert_eq!(
            root.gc_versions(epoch),
            3,
            "Gus's age and labels, and Vincent's city"
        );
        assert_eq!((root.update_nodes(), trips.update_nodes()), (0, 0));
        assert_eq!(root.listed_vectors(), NOTHING);
        assert_eq!(trips.listed_vectors(), NOTHING);
        assert_eq!(root.gc_versions(epoch), 0, "nothing is listed");
    }

    /// `gc_versions` on a root collects it and every named graph, and counts
    /// the rows it took out and the kept values it dropped; on a named graph
    /// it collects that graph alone.
    #[test]
    fn gc_versions_collects_every_named_graph_and_counts_what_it_dropped() {
        let root = RowGroupStore::new();
        let people = people(&root).unwrap();
        let mut tx = Tx::begin(&root, 10);
        tx.set_node(people.alix, "age", Value::Int64(20)).unwrap();
        tx.commit().unwrap();

        let trips = root.graph_or_create("trips").unwrap();
        let mut tx = Tx::begin(&*trips, 11);
        let vincent = tx
            .create_node(&["Person"], &[("name", Value::from("Vincent"))])
            .unwrap();
        let paris = tx
            .create_node(&["City"], &[("name", Value::from("Paris"))])
            .unwrap();
        let visit = tx.create_edge(vincent, paris, "VISITED", &[]).unwrap();
        tx.commit().unwrap();
        let mut tx = Tx::begin(&*trips, 12);
        tx.set_node(vincent, "city", Value::from("Berlin")).unwrap();
        tx.delete_edge(visit).unwrap();
        tx.delete_node(paris).unwrap();
        tx.commit().unwrap();
        assert_eq!((root.update_nodes(), trips.update_nodes()), (1, 1));

        let horizon = root.current_epoch().max(trips.current_epoch());
        assert_eq!(
            root.gc_versions(horizon),
            4,
            "Alix's age, Vincent's city, Paris and the visit"
        );
        assert_eq!((root.update_nodes(), trips.update_nodes()), (0, 0));
        assert_eq!((root.stamped_vectors(), trips.stamped_vectors()), (0, 0));
        assert_eq!(root.gc_versions(horizon), 0, "nothing is left");

        let mut tx = Tx::begin(&root, 13);
        tx.set_node(people.alix, "age", Value::Int64(21)).unwrap();
        tx.commit().unwrap();
        let mut tx = Tx::begin(&*trips, 14);
        tx.set_node(vincent, "city", Value::from("Paris")).unwrap();
        tx.commit().unwrap();
        let horizon = root.current_epoch().max(trips.current_epoch());
        assert_eq!(trips.gc_versions(horizon), 1, "trips alone");
        assert_eq!(root.update_nodes(), 1, "the root keeps Alix's age");
    }
}
