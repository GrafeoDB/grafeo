//! Update chains: what a row held before the writes some readers must not
//! see yet, per column (or per row group's labels) and vector of 2,048 rows
//! (masterplan 6.6, after DuckDB's `UpdateSegment`).
//!
//! The column holds the newest value of each row, an open transaction's
//! included. A write that some reader may not see (a transaction's, or an
//! immediate write, which a reader at an earlier epoch must not see) first
//! copies the value it replaces into an undo node of its vector: one node
//! per writer, holding the rows it wrote, each with what the row held before
//! the writer's first write of it. A node is stamped with the transaction
//! while it is open and with the commit epoch from `stamp` on. The nodes of
//! a vector are kept in the order they were made, newest last: a writer adds
//! to the newest node when it is its own and makes a new one otherwise, so
//! the nodes that hold a row are in the order of the row's writes (the
//! claims of the transaction manager let one writer at a time write a row).
//!
//! A reader takes the column's value unless a node it does not see holds
//! the row: then the oldest such node's value is what the row held before
//! every write the reader does not see. A vector without nodes costs a
//! reader one test. `collect` drops the nodes every reader sees, so a row
//! costs nothing more once no reader is older than its last write. The
//! store lists the vectors that have a chain, so its collect visits those
//! alone (`collect`).

use super::bitset::Bitset;
use super::versions::VECTOR_ROWS;
use super::{PENDING, Read};

/// A row of a vector, as a node holds it.
fn offset(row: usize) -> u16 {
    u16::try_from(row % VECTOR_ROWS).expect("a row of a vector is below 2,048")
}

/// One writer's undo node: what the rows it wrote held before.
#[derive(Debug)]
struct UndoNode<T> {
    /// The writer's transaction (with [`PENDING`]) or its commit epoch.
    stamp: u64,
    /// The rows of the vector, ascending, each with what it held before
    /// the writer's first write of it.
    rows: Vec<(u16, T)>,
}

impl<T> UndoNode<T> {
    fn find(&self, row: u16) -> Result<usize, usize> {
        self.rows.binary_search_by_key(&row, |(at, _)| *at)
    }

    fn get(&self, row: u16) -> Option<&T> {
        self.find(row).ok().map(|index| &self.rows[index].1)
    }
}

/// The undo nodes of one vector, oldest first.
#[derive(Debug)]
struct Chain<T> {
    nodes: Vec<UndoNode<T>>,
}

impl<T> Chain<T> {
    /// The index of the newest node that holds `row`.
    fn newest_with(&self, row: u16) -> Option<usize> {
        self.nodes.iter().rposition(|node| node.find(row).is_ok())
    }
}

/// The vectors of a row group (65,536 rows).
pub(super) const VECTORS: usize = 32;

const _: () = assert!(VECTORS * VECTOR_ROWS == 65_536);

/// The chain of each vector of a row group, `None` where it has none.
type Chains<T> = [Option<Box<Chain<T>>>; VECTORS];

/// The update chains of one column, or of one row group's labels: `T` is
/// what a row held (`Option<Value>` for a value, no value being `None`; the
/// label ids for labels). One pointer while no vector has a chain; the
/// first chain allocates a pointer per vector, freed with the last chain.
#[derive(Debug)]
pub(super) struct Updates<T> {
    chains: Option<Box<Chains<T>>>,
}

impl<T> Default for Updates<T> {
    fn default() -> Self {
        Self { chains: None }
    }
}

impl<T> Updates<T> {
    /// The chain of `row`'s vector.
    fn chain(&self, row: usize) -> Option<&Chain<T>> {
        self.chains.as_ref()?[row / VECTOR_ROWS].as_deref()
    }

    /// Whether no vector has a chain.
    pub(super) fn is_empty(&self) -> bool {
        self.chains.is_none()
    }

    /// What `row` held for `read`, when a node `read` does not see holds
    /// it; `None` when the column's value is what it reads.
    pub(super) fn saved(&self, row: usize, read: Read) -> Option<&T> {
        let chain = self.chain(row)?;
        let row = offset(row);
        let mut saved = None;
        for node in chain.nodes.iter().rev() {
            if !read.sees(node.stamp) {
                if let Some(value) = node.get(row) {
                    // An older node of the row may hold what came before.
                    saved = Some(value);
                }
            } else if node.stamp & PENDING != 0 && node.find(row).is_ok() {
                // The reader's own write: the newest of the row.
                return None;
            }
            // A committed node the reader sees: the older writes of its
            // rows are committed before it, so the reader sees them too.
        }
        saved
    }

    /// Keeps what `row` holds (`current`) before writer `stamp` writes it,
    /// unless a node of the writer holds the row already. `snapshot` is
    /// the epoch the writer reads at: every node of the writer is newer
    /// than the nodes committed by then, so the search stops at the first.
    /// Returns whether it kept the value (the write created the writer's
    /// version of the row) or a node of the writer had it (the write
    /// replaced its own version).
    pub(super) fn keep(
        &mut self,
        row: usize,
        stamp: u64,
        snapshot: u64,
        current: impl FnOnce() -> T,
    ) -> bool {
        let at = offset(row);
        if let Some(chain) = self.chain(row) {
            for node in chain.nodes.iter().rev() {
                if node.stamp == stamp && node.find(at).is_ok() {
                    return false;
                }
                if node.stamp & PENDING == 0 && node.stamp <= snapshot {
                    break;
                }
            }
        }
        let chains = self
            .chains
            .get_or_insert_with(|| Box::new(std::array::from_fn(|_| None)));
        let chain = chains[row / VECTOR_ROWS].get_or_insert_with(|| {
            Box::new(Chain {
                nodes: Vec::with_capacity(1),
            })
        });
        match chain.nodes.last_mut() {
            Some(node) if node.stamp == stamp => {
                let position = node.find(at).unwrap_err();
                node.rows.insert(position, (at, current()));
            }
            _ => chain.nodes.push(UndoNode {
                stamp,
                rows: vec![(at, current())],
            }),
        }
        true
    }

    /// Commits the newest node that holds `row` at `at`, if transaction
    /// stamp `pending` made it; returns that node's stamp before (`None`:
    /// no node holds the row). A node another entry of the same commit
    /// stamped already is found at `at`.
    pub(super) fn stamp(&mut self, row: usize, pending: u64, at: u64) -> Option<u64> {
        let chain = self.chains.as_mut()?[row / VECTOR_ROWS].as_deref_mut()?;
        let index = chain.newest_with(offset(row))?;
        let node = &mut chain.nodes[index];
        let before = node.stamp;
        if before == pending {
            node.stamp = at;
        }
        Some(before)
    }

    /// Takes `row` out of the newest node that holds it, when transaction
    /// stamp `pending` made that node, and returns what the row held before
    /// the transaction wrote it: a rollback of the transaction's first
    /// write of the row. A node left empty goes, and a chain left empty.
    pub(super) fn undo(&mut self, row: usize, pending: u64) -> Option<T> {
        let slot = &mut self.chains.as_mut()?[row / VECTOR_ROWS];
        let chain = slot.as_deref_mut()?;
        let at = offset(row);
        let index = chain.newest_with(at)?;
        if chain.nodes[index].stamp != pending {
            return None;
        }
        let node = &mut chain.nodes[index];
        let position = node.find(at).ok()?;
        let (_, saved) = node.rows.remove(position);
        if node.rows.is_empty() {
            chain.nodes.remove(index);
        }
        if chain.nodes.is_empty() {
            *slot = None;
            self.trim();
        }
        Some(saved)
    }

    /// Drops every node committed at or before `horizon`: every reader at
    /// the horizon or later sees it, and so the newer values it replaced.
    /// Returns the kept values (or label sets) it dropped.
    pub(super) fn collect(&mut self, horizon: u64) -> usize {
        let Some(chains) = self.chains.as_mut() else {
            return 0;
        };
        let dropped = chains
            .iter_mut()
            .map(|slot| Self::collect_slot(slot, horizon))
            .sum();
        self.trim();
        dropped
    }

    /// [`collect`](Self::collect) of vector `vector` alone.
    pub(super) fn collect_vector(&mut self, vector: usize, horizon: u64) -> usize {
        let Some(chains) = self.chains.as_mut() else {
            return 0;
        };
        let dropped = Self::collect_slot(&mut chains[vector], horizon);
        if dropped > 0 {
            self.trim();
        }
        dropped
    }

    /// Drops the nodes of one vector's chain committed at or before
    /// `horizon`, and the chain once it is empty; returns the kept values
    /// it dropped.
    fn collect_slot(slot: &mut Option<Box<Chain<T>>>, horizon: u64) -> usize {
        let Some(chain) = slot else {
            return 0;
        };
        let mut dropped = 0;
        chain.nodes.retain(|node| {
            let keep = node.stamp & PENDING != 0 || node.stamp > horizon;
            if !keep {
                dropped += node.rows.len();
            }
            keep
        });
        if chain.nodes.is_empty() {
            *slot = None;
        }
        dropped
    }

    /// Whether vector `vector` has a chain.
    pub(super) fn has_chain(&self, vector: usize) -> bool {
        self.chains
            .as_ref()
            .is_some_and(|chains| chains[vector].is_some())
    }

    /// Takes `row` out of every node, for a row that is gone.
    pub(super) fn remove_row(&mut self, row: usize) {
        let Some(chains) = self.chains.as_mut() else {
            return;
        };
        let slot = &mut chains[row / VECTOR_ROWS];
        let Some(chain) = slot.as_deref_mut() else {
            return;
        };
        let at = offset(row);
        for node in &mut chain.nodes {
            if let Ok(position) = node.find(at) {
                node.rows.remove(position);
            }
        }
        chain.nodes.retain(|node| !node.rows.is_empty());
        if chain.nodes.is_empty() {
            *slot = None;
            self.trim();
        }
    }

    /// Frees the pointers per vector once no vector has a chain.
    fn trim(&mut self) {
        if self
            .chains
            .as_ref()
            .is_some_and(|chains| chains.iter().all(Option::is_none))
        {
            self.chains = None;
        }
    }

    /// The chains of the vectors that have one, with their vector index.
    fn each(&self) -> impl Iterator<Item = (usize, &Chain<T>)> + '_ {
        self.chains.iter().flat_map(|chains| {
            chains
                .iter()
                .enumerate()
                .filter_map(|(index, chain)| Some((index, chain.as_deref()?)))
        })
    }

    /// The rows some node holds, `None` when no vector has a chain.
    pub(super) fn rows(&self) -> Option<Bitset> {
        self.chains.as_ref()?;
        let mut rows = Bitset::default();
        for (index, chain) in self.each() {
            for node in &chain.nodes {
                for (row, _) in &node.rows {
                    rows.set(index * VECTOR_ROWS + usize::from(*row));
                }
            }
        }
        Some(rows)
    }

    /// Everything a node of any vector holds for `row`, oldest first: what
    /// the row held before each write some reader may not see.
    pub(super) fn kept(&self, row: usize) -> impl Iterator<Item = &T> + '_ {
        let at = offset(row);
        self.chain(row)
            .into_iter()
            .flat_map(|chain| &chain.nodes)
            .filter_map(move |node| node.get(at))
    }

    /// As [`kept`](Self::kept), each with the stamp of its node: who wrote
    /// over it, a transaction (with [`PENDING`]) or a commit epoch.
    pub(super) fn kept_stamped(&self, row: usize) -> impl Iterator<Item = (u64, &T)> + '_ {
        let at = offset(row);
        self.chain(row)
            .into_iter()
            .flat_map(|chain| &chain.nodes)
            .filter_map(move |node| Some((node.stamp, node.get(at)?)))
    }

    /// The heap bytes the chains take, with what `held` says each kept
    /// value holds on the heap.
    pub(super) fn heap_bytes(&self, held: impl Fn(&T) -> usize) -> usize {
        if self.chains.is_none() {
            return 0;
        }
        std::mem::size_of::<Chains<T>>()
            + self
                .each()
                .map(|(_, chain)| {
                    std::mem::size_of::<Chain<T>>()
                        + chain.nodes.capacity() * std::mem::size_of::<UndoNode<T>>()
                        + chain
                            .nodes
                            .iter()
                            .map(|node| {
                                node.rows.capacity() * std::mem::size_of::<(u16, T)>()
                                    + node.rows.iter().map(|(_, kept)| held(kept)).sum::<usize>()
                            })
                            .sum::<usize>()
                })
                .sum::<usize>()
    }

    /// The nodes of every chain.
    #[cfg(test)]
    pub(super) fn nodes(&self) -> usize {
        self.each().map(|(_, chain)| chain.nodes.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use grafeo_common::change::{ChangeMark, DataOp};
    use grafeo_common::types::{ArcStr, EpochId, NodeId, PropertyKey, ReadView, Value};

    use super::{Updates, VECTOR_ROWS};
    use crate::graph::apply::{Applied, ChangeTarget, Writer};
    use crate::graph::conformance::{Rng, Tx, apply_all, labels, next_epoch, properties};
    use crate::graph::rowgroup::{PENDING, Read, RowGroupStore};
    use crate::graph::traits::GraphStore;

    /// A column of one row's values, newest in place, with its chain.
    struct Column {
        value: Vec<Option<u64>>,
        updates: Updates<Option<u64>>,
    }

    impl Column {
        fn new(rows: usize) -> Self {
            Self {
                value: vec![Some(0); rows],
                updates: Updates::default(),
            }
        }

        fn write(&mut self, row: usize, value: u64, stamp: u64, snapshot: u64) -> bool {
            let current = self.value[row];
            let kept = self.updates.keep(row, stamp, snapshot, || current);
            self.value[row] = Some(value);
            kept
        }

        fn read(&self, row: usize, read: Read) -> Option<u64> {
            match self.updates.saved(row, read) {
                Some(kept) => *kept,
                None => self.value[row],
            }
        }
    }

    /// A vector without a chain reads the column; a write keeps the value
    /// it replaced for the readers that do not see it, the writer's next
    /// write of the row keeps nothing more, and the commit makes the kept
    /// value the one readers before the commit epoch read.
    #[test]
    fn a_write_keeps_the_value_it_replaced_for_readers_that_do_not_see_it() {
        let id = 19;
        let writer = PENDING | id;
        let other = Read::Transaction { id: 7, snapshot: 3 };
        let own = Read::Transaction { id, snapshot: 3 };
        let mut column = Column::new(3 * VECTOR_ROWS);
        assert!(column.updates.is_empty());
        assert_eq!(column.read(5, Read::At(3)), Some(0));

        assert!(
            column.write(VECTOR_ROWS + 5, 88, writer, 3),
            "the first write keeps"
        );
        assert!(
            !column.write(VECTOR_ROWS + 5, 89, writer, 3),
            "the second keeps nothing"
        );
        assert_eq!(column.read(VECTOR_ROWS + 5, own), Some(89));
        assert_eq!(column.read(VECTOR_ROWS + 5, other), Some(0));
        assert_eq!(column.read(VECTOR_ROWS + 5, Read::At(1_000)), Some(0));
        assert_eq!(
            column.read(VECTOR_ROWS + 6, other),
            Some(0),
            "a row not written"
        );
        assert_eq!(column.read(5, other), Some(0), "a vector without a chain");

        assert_eq!(
            column.updates.stamp(VECTOR_ROWS + 5, writer, 4),
            Some(writer)
        );
        assert_eq!(
            column.updates.stamp(VECTOR_ROWS + 5, writer, 4),
            Some(4),
            "an entry stamped twice finds the commit's epoch"
        );
        assert_eq!(column.read(VECTOR_ROWS + 5, Read::At(3)), Some(0));
        assert_eq!(column.read(VECTOR_ROWS + 5, Read::At(4)), Some(89));
        column.updates.collect(3);
        assert_eq!(column.updates.nodes(), 1, "a reader at 3 still needs it");
        column.updates.collect(4);
        assert!(column.updates.is_empty());
        assert_eq!(column.updates.heap_bytes(|_| 0), 0, "nothing is left");
    }

    /// Two transactions write rows of one vector, interleaved: the second
    /// writer's nodes come after the first's, so a row two of them wrote
    /// reads in the order of its writes, for every reader.
    #[test]
    fn interleaved_writers_keep_the_order_of_each_rows_writes() {
        let (first_id, second_id) = (1, 2);
        let (first, second) = (PENDING | first_id, PENDING | second_id);
        let mut column = Column::new(VECTOR_ROWS);
        // The second opens its node first, on another row.
        assert!(column.write(9, 30, second, 3));
        assert!(column.write(5, 10, first, 3));
        assert_eq!(column.updates.stamp(5, first, 4), Some(first));
        // Its next write of row 5 comes after the first's commit.
        assert!(column.write(5, 20, second, 3));
        assert_eq!(column.updates.nodes(), 3, "a new node of the second writer");
        let second_reader = Read::Transaction {
            id: second_id,
            snapshot: 3,
        };
        assert_eq!(column.read(5, Read::At(3)), Some(0));
        assert_eq!(column.read(5, Read::At(4)), Some(10), "the first's commit");
        assert_eq!(column.read(5, second_reader), Some(20));
        assert_eq!(column.read(9, second_reader), Some(30));
        assert_eq!(column.read(9, Read::At(4)), Some(0));
        // A write of row 9 again finds the second writer's older node.
        assert!(!column.write(9, 31, second, 3));

        assert_eq!(column.updates.undo(5, second), Some(Some(10)));
        assert_eq!(
            column.updates.undo(5, second),
            None,
            "the first writer's node is next"
        );
        assert_eq!(column.updates.undo(9, second), Some(Some(0)));
        assert_eq!(column.updates.nodes(), 1);
    }

    /// One node's state as readers see it: its age, its city and its labels.
    #[derive(Debug, Clone, PartialEq, Eq, Default)]
    struct State {
        age: Option<i64>,
        city: Option<String>,
        labels: BTreeSet<&'static str>,
    }

    /// Whether an open transaction wrote a field, and what.
    #[derive(Debug, Clone, Default)]
    enum Written<T> {
        #[default]
        Not,
        To(T),
    }

    impl<T: Clone> Written<T> {
        fn over(&self, base: &T) -> T {
            match self {
                Self::Not => base.clone(),
                Self::To(value) => value.clone(),
            }
        }
    }

    /// What an open transaction wrote of one node: each field it wrote.
    #[derive(Debug, Clone, Default)]
    struct Writes {
        age: Written<Option<i64>>,
        city: Written<Option<String>>,
        labels: Written<BTreeSet<&'static str>>,
    }

    impl Writes {
        /// `base` with these writes.
        fn over(&self, base: &State) -> State {
            State {
                age: self.age.over(&base.age),
                city: self.city.over(&base.city),
                labels: self.labels.over(&base.labels),
            }
        }
    }

    /// An open transaction and what it wrote, with its savepoints.
    struct Open<'s> {
        tx: Tx<'s, RowGroupStore>,
        writes: BTreeMap<usize, Writes>,
        marks: Vec<(ChangeMark, BTreeMap<usize, Writes>)>,
    }

    /// The committed history of each node, ascending by epoch.
    struct Model {
        ids: Vec<NodeId>,
        history: Vec<Vec<(u64, State)>>,
    }

    impl Model {
        fn at(&self, node: usize, epoch: u64) -> &State {
            &self.history[node]
                .iter()
                .rev()
                .find(|(at, _)| *at <= epoch)
                .expect("a state from the load on")
                .1
        }

        fn newest(&self, node: usize) -> &State {
            &self.history[node].last().expect("a state").1
        }

        /// The state of `node` as `open` sees it.
        fn seen_by(&self, open: &Open<'_>, node: usize) -> State {
            let base = self.at(node, open.tx.snapshot.as_u64());
            open.writes
                .get(&node)
                .map_or_else(|| base.clone(), |writes| writes.over(base))
        }

        /// The state of `node` a write of `open` changes: its own writes,
        /// over the newest committed state (the store writes in place).
        fn written_by(&self, open: &Open<'_>, node: usize) -> State {
            let base = self.newest(node);
            open.writes
                .get(&node)
                .map_or_else(|| base.clone(), |writes| writes.over(base))
        }
    }

    const LABELS: [&str; 3] = ["Person", "Traveller", "Employee"];
    const CITIES: [&str; 3] = ["Amsterdam", "Berlin", "Paris"];

    /// One reader's view.
    #[derive(Clone, Copy)]
    enum Reader {
        At(u64),
        Open(usize),
    }

    /// Checks every read of `store` at `reader`'s view against the model.
    fn check(store: &RowGroupStore, model: &Model, open: &[Open<'_>], reader: Reader, what: &str) {
        let view = match reader {
            Reader::At(epoch) => ReadView::committed(EpochId::new(epoch)),
            Reader::Open(index) => {
                ReadView::transaction(open[index].tx.id, open[index].tx.snapshot)
            }
        };
        let expected: Vec<State> = (0..model.ids.len())
            .map(|node| match reader {
                Reader::At(epoch) => model.at(node, epoch).clone(),
                Reader::Open(index) => model.seen_by(&open[index], node),
            })
            .collect();
        for (node, (id, state)) in model.ids.iter().zip(&expected).enumerate() {
            let what = format!("{what}, node {node}");
            assert_eq!(
                store.node_property_at(*id, "age", view),
                state.age.map(Value::Int64),
                "{what}: age"
            );
            assert_eq!(
                store.node_property_at(*id, "city", view),
                state.city.clone().map(Value::from),
                "{what}: city"
            );
            let found = store.get_node_at(*id, view).expect("the node exists");
            let mut names: Vec<String> = found.labels.iter().map(ToString::to_string).collect();
            names.sort();
            let mut wanted: Vec<String> = state.labels.iter().map(ToString::to_string).collect();
            wanted.sort();
            assert_eq!(names, wanted, "{what}: labels");
            assert_eq!(
                found.properties.get(&PropertyKey::new("age")).cloned(),
                state.age.map(Value::Int64),
                "{what}: the whole node's age"
            );
            for label in LABELS {
                assert_eq!(
                    store.node_has_label_at(*id, label, view),
                    Some(state.labels.contains(label)),
                    "{what}: has {label}"
                );
            }
        }
        for label in LABELS {
            let found: BTreeSet<NodeId> =
                store.nodes_by_label_at(label, view).into_iter().collect();
            for (id, state) in model.ids.iter().zip(&expected) {
                assert_eq!(
                    found.contains(id),
                    state.labels.contains(label),
                    "{what}: the {label} scan and node {id:?}"
                );
            }
        }
        for city in CITIES {
            let found: BTreeSet<NodeId> = store
                .find_nodes_by_property_at("city", &Value::from(city), view)
                .into_iter()
                .collect();
            for (id, state) in model.ids.iter().zip(&expected) {
                assert_eq!(
                    found.contains(id),
                    state.city.as_deref() == Some(city),
                    "{what}: the lookup of {city} and node {id:?}"
                );
            }
        }
    }

    /// Random value and label writes by up to three open transactions at
    /// once, each writing nodes no other open one holds (as the claims
    /// allow, also one a commit after its snapshot wrote), with savepoints,
    /// rollbacks to them, commits and rollbacks; immediate writes of nodes
    /// no one holds; collects at a horizon no reader is older than; reads
    /// of every kind at random views: all agree with a model of each node's
    /// committed history and each transaction's own writes. Once every
    /// transaction has ended and a collect at the store's epoch ran, no
    /// undo node is left.
    #[test]
    fn values_and_labels_agree_with_a_model_of_versioned_states() {
        // Rows across three vectors, at their ends and inside.
        let rows: Vec<u64> = vec![
            0, 1, 2, 100, 2_046, 2_047, 2_048, 2_049, 3_000, 4_095, 4_096, 4_100,
        ];
        let count = rows.iter().max().expect("rows") + 1;
        for seed in 1..=40 {
            let store = RowGroupStore::new();
            let first = store.reserve_node_ids(count).expect("ids").start;
            let mut model = Model {
                ids: Vec::new(),
                history: Vec::new(),
            };
            let epoch = next_epoch(&store);
            let mut ops = Vec::new();
            for raw in first..first + count {
                let offset = raw - first;
                let hot = rows.contains(&offset);
                let state = State {
                    age: (!offset.is_multiple_of(3))
                        .then(|| i64::try_from(offset % 88).expect("age")),
                    city: Some(CITIES[usize::try_from(offset % 3).expect("an index")].to_string()),
                    labels: if offset.is_multiple_of(2) {
                        BTreeSet::from(["Person"])
                    } else {
                        BTreeSet::from(["Person", "Employee"])
                    },
                };
                let mut values = vec![("city", Value::from(state.city.clone().expect("city")))];
                if let Some(age) = state.age {
                    values.push(("age", Value::Int64(age)));
                }
                let names: Vec<&str> = state.labels.iter().copied().collect();
                ops.push(DataOp::CreateNode {
                    id: NodeId::new(raw),
                    labels: labels(&names),
                    properties: properties(&values),
                });
                if hot {
                    model.ids.push(NodeId::new(raw));
                    model.history.push(vec![(epoch.as_u64(), state)]);
                }
            }
            apply_all(&store, &ops, Writer::Replay { epoch }).expect("the load");
            // Half the seeds look cities up through an index, which holds
            // old and pending values the lookups at a view must recheck.
            let indexed = seed % 2 == 0;
            if indexed {
                store.create_property_index("city");
            }

            let mut rng = Rng::new(seed);
            let mut open: Vec<Open<'_>> = Vec::new();
            let mut next_transaction = 10;
            // Readers older than the last collect's horizon are not served.
            let mut floor = epoch.as_u64();
            for step in 0..600 {
                let what = format!("seed {seed}, step {step}");
                let held = |open: &[Open<'_>], node: usize, except: Option<usize>| {
                    open.iter().enumerate().any(|(index, other)| {
                        Some(index) != except && other.writes.contains_key(&node)
                    })
                };
                match rng.below(20) {
                    0 | 1 if open.len() < 3 => {
                        next_transaction += 1;
                        open.push(Open {
                            tx: Tx::begin(&store, next_transaction),
                            writes: BTreeMap::new(),
                            marks: Vec::new(),
                        });
                    }
                    2..=10 if !open.is_empty() => {
                        let index = rng.below(open.len());
                        let node = rng.below(model.ids.len());
                        if held(&open, node, Some(index)) {
                            continue;
                        }
                        let id = model.ids[node];
                        let base = model.written_by(&open[index], node);
                        let writes = open[index].writes.get(&node).cloned().unwrap_or_default();
                        let mut writes = writes;
                        let tx = &mut open[index].tx;
                        match rng.below(6) {
                            0 => {
                                let age = rng.int(88);
                                tx.set_node(id, "age", Value::Int64(age)).expect("set age");
                                writes.age = Written::To(Some(age));
                            }
                            1 => {
                                let city = CITIES[rng.below(CITIES.len())];
                                tx.set_node(id, "city", Value::from(city))
                                    .expect("set city");
                                writes.city = Written::To(Some(city.to_string()));
                            }
                            2 => {
                                // A set to null removes the value (Q8).
                                let applied = tx
                                    .apply(DataOp::SetNodeProperty {
                                        id,
                                        key: PropertyKey::new("age"),
                                        value: Value::Null,
                                    })
                                    .expect("set to null");
                                if base.age.is_some() {
                                    assert!(matches!(applied, Applied::Changed { .. }), "{what}");
                                    writes.age = Written::To(None);
                                } else {
                                    assert_eq!(applied, Applied::Unchanged, "{what}");
                                }
                            }
                            3 if base.city.is_some() => {
                                tx.remove_node_value(id, "city").expect("remove city");
                                writes.city = Written::To(None);
                            }
                            _ => {
                                let label = LABELS[1 + rng.below(2)];
                                let mut set = base.labels.clone();
                                if set.contains(label) {
                                    tx.remove_label(id, label).expect("remove a label");
                                    set.remove(label);
                                } else {
                                    tx.add_label(id, label).expect("add a label");
                                    set.insert(label);
                                }
                                writes.labels = Written::To(set);
                            }
                        }
                        open[index].writes.insert(node, writes);
                    }
                    11 if !open.is_empty() => {
                        let index = rng.below(open.len());
                        let mark = open[index].tx.mark();
                        let writes = open[index].writes.clone();
                        open[index].marks.push((mark, writes));
                    }
                    12 if !open.is_empty() => {
                        let index = rng.below(open.len());
                        if let Some((mark, writes)) = open[index].marks.pop() {
                            open[index].tx.roll_back_to(mark).expect("to the savepoint");
                            // The claims stay until the transaction ends.
                            let claimed: Vec<usize> = open[index].writes.keys().copied().collect();
                            open[index].writes = writes;
                            for node in claimed {
                                open[index].writes.entry(node).or_default();
                            }
                        }
                    }
                    13 | 14 if !open.is_empty() => {
                        let ended = open.swap_remove(rng.below(open.len()));
                        if rng.below(3) == 0 {
                            ended.tx.roll_back().expect("the rollback");
                        } else {
                            let epoch = ended.tx.commit().expect("the commit").as_u64();
                            for (node, writes) in &ended.writes {
                                let state = writes.over(model.newest(*node));
                                model.history[*node].push((epoch, state));
                            }
                        }
                    }
                    15 => {
                        let node = rng.below(model.ids.len());
                        if held(&open, node, None) {
                            continue;
                        }
                        let id = model.ids[node];
                        let mut state = model.newest(node).clone();
                        let op = if rng.below(2) == 0 {
                            let age = rng.int(88);
                            state.age = Some(age);
                            DataOp::SetNodeProperty {
                                id,
                                key: PropertyKey::new("age"),
                                value: Value::Int64(age),
                            }
                        } else {
                            let label = LABELS[1 + rng.below(2)];
                            if state.labels.remove(label) {
                                DataOp::RemoveNodeLabel {
                                    id,
                                    label: ArcStr::from(label),
                                }
                            } else {
                                state.labels.insert(label);
                                DataOp::AddNodeLabel {
                                    id,
                                    label: ArcStr::from(label),
                                }
                            }
                        };
                        let epoch = next_epoch(&store);
                        store
                            .apply(
                                &op,
                                Writer::Immediate {
                                    epoch,
                                    before_images: rng.below(2) == 0,
                                },
                            )
                            .expect("the immediate write");
                        model.history[node].push((epoch.as_u64(), state));
                    }
                    16 => {
                        let now = store.current_epoch().as_u64();
                        let oldest = open
                            .iter()
                            .map(|open| open.tx.snapshot.as_u64())
                            .min()
                            .unwrap_or(now);
                        let top = oldest.max(floor);
                        let horizon = floor
                            + rng.below(usize::try_from(top - floor).expect("epochs") + 1) as u64;
                        store.collect(EpochId::new(horizon));
                        floor = horizon;
                    }
                    _ => {
                        let now = store.current_epoch().as_u64();
                        let reader = if !open.is_empty() && rng.below(2) == 0 {
                            Reader::Open(rng.below(open.len()))
                        } else {
                            Reader::At(
                                floor
                                    + rng.below(usize::try_from(now - floor).expect("epochs") + 1)
                                        as u64,
                            )
                        };
                        check(&store, &model, &open, reader, &what);
                    }
                }
            }
            for ended in open.drain(..) {
                let epoch = ended.tx.commit().expect("the last commits").as_u64();
                for (node, writes) in &ended.writes {
                    let state = writes.over(model.newest(*node));
                    model.history[*node].push((epoch, state));
                }
            }
            let now = store.current_epoch();
            check(
                &store,
                &model,
                &[],
                Reader::At(now.as_u64()),
                &format!("seed {seed}, at the end"),
            );
            store.collect(now);
            assert_eq!(store.update_nodes(), 0, "seed {seed}: no undo node is left");
            if indexed {
                // No reader needs an old value any more: the index holds
                // each node under its newest city alone.
                for (node, id) in model.ids.iter().enumerate() {
                    for city in CITIES {
                        let held = store
                            .find_nodes_maybe_equal("city", &Value::from(city))
                            .expect("the index")
                            .contains(id);
                        assert_eq!(
                            held,
                            model.newest(node).city.as_deref() == Some(city),
                            "seed {seed}: node {node} under {city} after the collect"
                        );
                    }
                }
            }
            check(
                &store,
                &model,
                &[],
                Reader::At(now.as_u64()),
                &format!("seed {seed}, collected"),
            );
        }
    }
}
