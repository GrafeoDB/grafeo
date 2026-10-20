//! Garbage collection over the list of vectors with versions (#412 S6).
//!
//! Every vector of 2,048 rows, of either table, that gains a version stamp
//! (a create or a delete) or an undo node of a value or label chain is
//! listed once in its store: a bit per vector in its row group, and the row
//! group once in the store's list. An immediate write keeps what it
//! replaces whatever readers there are, and the engine collects right after
//! each immediate call and each commit, so the list holds what the last
//! writes left and what an older reader still needs:
//! [`collect`](RowGroupStore::collect) visits those vectors alone, and
//! returns at once, without the lock, while nothing is listed. A vector stays
//! listed while something in it is above the horizon (a stamp or an undo
//! node not committed by then), and leaves the list once it is flat with no
//! chain.
//!
//! The list may hold more than that (a rollback emptied a vector since),
//! never less: every vector with a stamp or a chain is listed.
//! [`collect_all`](RowGroupStore::collect_all) walks every vector, for tests
//! and as a fallback, and lists exactly the vectors with versions left.

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;

use grafeo_common::change::Table;
use grafeo_common::types::EpochId;

use super::updates::VECTORS;
use super::versions::{VECTOR_ROWS, Versions};
use super::{Collected, EdgeGroup, Inner, NodeGroup, ROWS_PER_GROUP, RowGroupStore, locate};

const _: () = assert!(
    VECTORS == u32::BITS as usize,
    "a bit per vector of a row group"
);

/// The row groups with a listed vector, each once, per table.
#[derive(Debug, Default)]
pub(super) struct Listed {
    nodes: Vec<u64>,
    edges: Vec<u64>,
}

impl Listed {
    /// Whether no row group has a listed vector.
    pub(super) fn is_empty(&self) -> bool {
        self.nodes.is_empty() && self.edges.is_empty()
    }
}

/// The bit of `row`'s vector in its row group's mask of listed vectors.
fn bit(row: usize) -> u32 {
    1 << (row / VECTOR_ROWS)
}

/// The vectors of `mask`, ascending.
fn vectors(mut mask: u32) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        if mask == 0 {
            return None;
        }
        let vector = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        Some(vector)
    })
}

/// The mask of the vectors of a row group `unsettled` says hold versions.
fn mask_of(unsettled: impl Fn(usize) -> bool) -> u32 {
    (0..VECTORS)
        .filter(|vector| unsettled(*vector))
        .fold(0, |mask, vector| mask | 1 << vector)
}

/// Keeps in `listed` the row groups that still list a vector, dropping from
/// `groups` the ones left without rows, as a load would not make them.
/// `parts` gives a group's versions and its mask. In place, so the list
/// keeps its buffer from one collect to the next.
fn keep_listed<G>(
    groups: &mut BTreeMap<u64, G>,
    listed: &mut Vec<u64>,
    parts: impl Fn(&G) -> (&Versions, u32),
) {
    listed.retain(|index| {
        let Some(group) = groups.get(index) else {
            return false;
        };
        let (versions, mask) = parts(group);
        if versions.is_empty() {
            groups.remove(index);
            false
        } else {
            mask != 0
        }
    });
}

impl NodeGroup {
    /// Whether vector `vector` holds versions: stamps, or a value or label
    /// chain.
    fn unsettled(&self, vector: usize) -> bool {
        self.versions.is_stamped(vector)
            || self.columns.has_chain(vector)
            || self.label_updates.has_chain(vector)
    }
}

impl EdgeGroup {
    /// Whether vector `vector` holds versions: stamps, or a value chain.
    fn unsettled(&self, vector: usize) -> bool {
        self.versions.is_stamped(vector) || self.columns.has_chain(vector)
    }
}

impl Inner {
    /// Lists the vector of row `raw` of `table`, a row whose group exists;
    /// returns whether the store listed nothing before.
    pub(super) fn list(&mut self, table: Table, raw: u64) -> bool {
        let was_empty = self.listed.is_empty();
        let (index, row) = locate(raw);
        let (mask, groups) = match table {
            Table::Nodes => (
                self.nodes.get_mut(&index).map(|group| &mut group.listed),
                &mut self.listed.nodes,
            ),
            Table::Edges => (
                self.edges.get_mut(&index).map(|group| &mut group.listed),
                &mut self.listed.edges,
            ),
        };
        let Some(mask) = mask else {
            return false;
        };
        if *mask == 0 {
            groups.push(index);
        }
        *mask |= bit(row);
        was_empty && !self.listed.is_empty()
    }

    /// Collects the listed vectors at `horizon` (see
    /// [`RowGroupStore::collect`]) and keeps listed the ones that still hold
    /// versions.
    fn collect_listed(&mut self, horizon: u64) -> Collected {
        let mut nodes = std::mem::take(&mut self.listed.nodes);
        let mut edges = std::mem::take(&mut self.listed.edges);
        for groups in [&mut nodes, &mut edges] {
            groups.sort_unstable();
            groups.dedup();
        }
        // The index entries no reader at the horizon or later finds go
        // first, while the rows still hold what they were made from.
        self.unindex_collected(horizon, &nodes);
        let mut values = 0;
        let mut gone_edges = Vec::new();
        for index in &edges {
            let Some(group) = self.edges.get_mut(index) else {
                continue;
            };
            let base = index * ROWS_PER_GROUP;
            let mut mask = group.listed;
            for vector in vectors(group.listed) {
                for row in group.versions.collapse_vector(vector, horizon) {
                    group.columns.clear_row(row);
                    gone_edges.push(base + row as u64);
                }
                values += group.columns.collect_vector(vector, horizon);
                if !group.unsettled(vector) {
                    mask &= !(1 << vector);
                }
            }
            group.listed = mask;
        }
        let mut rows = gone_edges.len();
        self.unlink_edges(&gone_edges);
        for index in &nodes {
            let Some(group) = self.nodes.get_mut(index) else {
                continue;
            };
            let mut mask = group.listed;
            for vector in vectors(group.listed) {
                for row in group.versions.collapse_vector(vector, horizon) {
                    group.clear(row);
                    rows += 1;
                }
                values += group.columns.collect_vector(vector, horizon);
                values += group.label_updates.collect_vector(vector, horizon);
                if !group.unsettled(vector) {
                    mask &= !(1 << vector);
                }
            }
            group.listed = mask;
        }
        keep_listed(&mut self.edges, &mut edges, |group| {
            (&group.versions, group.listed)
        });
        keep_listed(&mut self.nodes, &mut nodes, |group| {
            (&group.versions, group.listed)
        });
        // Nothing was listed meanwhile: the collect holds the lock.
        self.listed.edges = edges;
        self.listed.nodes = nodes;
        Collected { rows, values }
    }

    /// What [`RowGroupStore::collect_all`] does: the collect of every vector
    /// at `horizon`, then the list built anew.
    fn collect_all(&mut self, horizon: u64) -> Collected {
        let every: Vec<u64> = self.nodes.keys().copied().collect();
        self.unindex_collected(horizon, &every);
        let mut values = 0;
        let mut gone_edges: Vec<u64> = Vec::new();
        for (index, group) in &mut self.edges {
            for row in group.versions.collapse(horizon) {
                group.columns.clear_row(row);
                gone_edges.push(index * ROWS_PER_GROUP + row as u64);
            }
            values += group.columns.collect(horizon);
        }
        let mut rows = gone_edges.len();
        self.unlink_edges(&gone_edges);
        for group in self.nodes.values_mut() {
            for row in group.versions.collapse(horizon) {
                group.clear(row);
                rows += 1;
            }
            values += group.columns.collect(horizon);
            if !group.label_updates.is_empty() {
                values += group.label_updates.collect(horizon);
            }
        }
        // A row group without rows goes whole, as a load would not make it.
        self.edges.retain(|_, group| !group.versions.is_empty());
        self.nodes.retain(|_, group| !group.versions.is_empty());
        self.relist();
        Collected { rows, values }
    }

    /// Lists exactly the vectors that hold versions.
    fn relist(&mut self) {
        self.listed = Listed::default();
        for (index, group) in &mut self.nodes {
            group.listed = mask_of(|vector| group.unsettled(vector));
            if group.listed != 0 {
                self.listed.nodes.push(*index);
            }
        }
        for (index, group) in &mut self.edges {
            group.listed = mask_of(|vector| group.unsettled(vector));
            if group.listed != 0 {
                self.listed.edges.push(*index);
            }
        }
    }

    /// Takes the adjacency entries of the collected edge rows `edges` out
    /// of their nodes' rows; the edge rows still hold their ends.
    fn unlink_edges(&mut self, edges: &[u64]) {
        for edge in edges {
            let (group, row) = locate(*edge);
            let (src, dst, _) = self.edges[&group].ends(row);
            for (node, outgoing) in [(src, true), (dst, false)] {
                let (group, row) = locate(node);
                if let Some(group) = self.nodes.get_mut(&group) {
                    group.adjacency_mut(outgoing).remove(row, *edge);
                }
            }
        }
    }

    /// The node row groups with a listed vector, each with its index: the
    /// only ones a stamp or a chain can be in.
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    pub(super) fn listed_node_groups(&self) -> impl Iterator<Item = (u64, &NodeGroup)> + '_ {
        self.listed
            .nodes
            .iter()
            .filter_map(|index| Some((*index, self.nodes.get(index)?)))
    }
}

impl RowGroupStore {
    /// Collects what no reader at `horizon` or later can see any more, in
    /// the vectors the store lists (see the module docs): the version info
    /// of each vector whose creates and deletes are all committed at or
    /// before `horizon` collapses to one bit per row, the rows deleted by
    /// then are taken out with their values, labels, adjacency and index
    /// entries, and the undo nodes of value and label writes committed by
    /// then go, with the chains they leave empty; a row group left without
    /// rows goes whole. A vector still holding versions stays listed for
    /// the next collect. With nothing listed, it returns at once.
    ///
    /// `horizon` is the oldest epoch a reader still reads at (the engine's
    /// read horizon: open and committing snapshots, pinned reads, the
    /// published epoch; H2c adds the retention window): a reader before it
    /// may lose rows it saw. Returns the rows taken out.
    ///
    /// This store alone: [`gc_versions`](Self::gc_versions) collects its
    /// named graphs too.
    pub fn collect(&self, horizon: EpochId) -> usize {
        self.collect_versions(horizon).rows
    }

    /// What [`collect`](Self::collect) does, walking every vector of the
    /// store instead of the listed ones, and listing anew the vectors that
    /// still hold versions: for tests, and as a fallback should the list
    /// ever miss a vector. Returns the rows taken out.
    #[doc(hidden)]
    pub fn collect_all(&self, horizon: EpochId) -> usize {
        let mut inner = self.inner.write();
        let collected = inner.collect_all(horizon.as_u64());
        self.has_listed
            .store(!inner.listed.is_empty(), Ordering::Release);
        collected.rows
    }

    /// What [`collect`](Self::collect) does; returns what it took out.
    pub(super) fn collect_versions(&self, horizon: EpochId) -> Collected {
        if !self.has_listed() {
            return Collected::default();
        }
        let mut inner = self.inner.write();
        let collected = inner.collect_listed(horizon.as_u64());
        self.has_listed
            .store(!inner.listed.is_empty(), Ordering::Release);
        collected
    }

    /// Whether the store lists a vector: a collect has work for it.
    pub(super) fn has_listed(&self) -> bool {
        self.has_listed.load(Ordering::Acquire)
    }

    /// Notes that the store lists a vector, after a write listed the first.
    pub(super) fn note_listed(&self) {
        self.has_listed.store(true, Ordering::Release);
    }

    /// The vectors the store lists, by table and first row id, ascending.
    #[cfg(test)]
    pub(crate) fn listed_vectors(&self) -> Vec<(Table, u64)> {
        let inner = self.inner.read();
        let mut listed = Vec::new();
        for (table, groups) in [
            (Table::Nodes, &inner.listed.nodes),
            (Table::Edges, &inner.listed.edges),
        ] {
            for index in groups {
                let mask = match table {
                    Table::Nodes => inner.nodes.get(index).map(|group| group.listed),
                    Table::Edges => inner.edges.get(index).map(|group| group.listed),
                };
                for vector in vectors(mask.unwrap_or(0)) {
                    listed.push((
                        table,
                        index * ROWS_PER_GROUP + (vector * VECTOR_ROWS) as u64,
                    ));
                }
            }
        }
        listed.sort_unstable_by_key(|(table, first)| (matches!(table, Table::Edges), *first));
        listed.dedup();
        listed
    }

    /// The vectors that hold versions (stamps or a chain), by table and
    /// first row id, ascending: found by walking every vector.
    #[cfg(test)]
    pub(crate) fn vectors_with_versions(&self) -> Vec<(Table, u64)> {
        let inner = self.inner.read();
        let mut found = Vec::new();
        for (index, group) in &inner.nodes {
            for vector in vectors(mask_of(|vector| group.unsettled(vector))) {
                found.push((
                    Table::Nodes,
                    index * ROWS_PER_GROUP + (vector * VECTOR_ROWS) as u64,
                ));
            }
        }
        for (index, group) in &inner.edges {
            for vector in vectors(mask_of(|vector| group.unsettled(vector))) {
                found.push((
                    Table::Edges,
                    index * ROWS_PER_GROUP + (vector * VECTOR_ROWS) as u64,
                ));
            }
        }
        found
    }

    /// What the store holds, written out in an order of its own (the hash
    /// maps of two stores list alike entries in different orders): per row
    /// group the versions, labels, chains, columns, the adjacency of its
    /// rows and its listed vectors, then the counts and the list. Two
    /// stores that hold the same write the same.
    #[cfg(test)]
    pub(crate) fn state(&self) -> String {
        use std::fmt::Write;

        let inner = self.inner.read();
        let mut out = String::new();
        for (index, group) in &inner.nodes {
            let mut labels: Vec<_> = group.labels.iter().collect();
            labels.sort_unstable_by_key(|(id, _)| **id);
            let adjacency: Vec<_> = group
                .versions
                .rows()
                .map(|(row, _)| {
                    let mut out: Vec<_> = group.outgoing.of(row).collect();
                    let mut into: Vec<_> = group.incoming.of(row).collect();
                    out.sort_unstable();
                    into.sort_unstable();
                    (row, out, into)
                })
                .collect();
            writeln!(
                out,
                "node group {index}: {:?} {labels:?} {:?} {:?} {adjacency:?} {:b}",
                group.versions,
                group.label_updates,
                group.columns.canonical(),
                group.listed
            )
            .expect("a string takes it");
        }
        for (index, group) in &inner.edges {
            writeln!(
                out,
                "edge group {index}: {:?} {:?} {:?} {:?} {:?} {:b}",
                group.versions,
                group.src,
                group.dst,
                group.edge_type,
                group.columns.canonical(),
                group.listed
            )
            .expect("a string takes it");
        }
        writeln!(out, "{:?} {:?}", inner.counts, inner.listed).expect("a string takes it");
        out
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use grafeo_common::change::{ChangeMark, DataOp, Table};
    use grafeo_common::types::{ArcStr, EdgeId, EpochId, NodeId, PropertyKey, Value};

    use crate::graph::apply::{Applied, ChangeTarget, Writer};
    use crate::graph::conformance::{
        Image, Reader, Rng, Tx, apply_all, image, labels, next_epoch, properties,
    };
    use crate::graph::rowgroup::{ROWS_PER_GROUP, RowGroupStore, VECTOR_ROWS};

    /// No vector, as `listed_vectors` lists them.
    const NOTHING: [(Table, u64); 0] = [];

    /// The first row id of the vector of row `raw`.
    fn vector_of(raw: u64) -> u64 {
        raw - raw % VECTOR_ROWS as u64
    }

    /// An immediate write of `op` at the store's next epoch.
    fn immediate(store: &RowGroupStore, op: &DataOp) -> Result<Applied, String> {
        store
            .apply(
                op,
                Writer::Immediate {
                    epoch: next_epoch(store),
                    before_images: true,
                },
            )
            .map_err(|error| error.to_string())
    }

    /// People at node rows in two vectors of row group 0 and one of row
    /// group 1, replayed: Alix, Gus and Vincent.
    fn spread_people(store: &RowGroupStore) -> [NodeId; 3] {
        let reserved = store
            .reserve_node_ids(ROWS_PER_GROUP + 3 * VECTOR_ROWS as u64)
            .unwrap();
        assert_eq!(reserved.start, 0);
        let ids = [
            NodeId::new(5),
            NodeId::new(VECTOR_ROWS as u64 + 1),
            NodeId::new(ROWS_PER_GROUP + 2 * VECTOR_ROWS as u64 + 7),
        ];
        let ops: Vec<DataOp> = ids
            .iter()
            .zip(["Alix", "Gus", "Vincent"])
            .map(|(id, name)| DataOp::CreateNode {
                id: *id,
                labels: labels(&["Person"]),
                properties: properties(&[("name", Value::from(name)), ("age", Value::Int64(30))]),
            })
            .collect();
        apply_all(
            store,
            &ops,
            Writer::Replay {
                epoch: next_epoch(store),
            },
        )
        .unwrap();
        ids
    }

    /// Each kind of write lists the vector of the row it writes, once; a
    /// replayed value write keeps nothing and lists nothing; a collect
    /// leaves listed exactly the vectors that still hold versions, and a
    /// rollback's vector leaves the list at the next collect.
    #[test]
    fn the_list_holds_exactly_the_vectors_the_writes_touched() {
        let store = RowGroupStore::new();
        let [alix, gus, vincent] = spread_people(&store);
        assert_eq!(
            store.listed_vectors(),
            [
                (Table::Nodes, 0),
                (Table::Nodes, VECTOR_ROWS as u64),
                (Table::Nodes, vector_of(vincent.as_u64())),
            ],
            "replay stamps the vectors it creates rows in"
        );
        store.collect(store.current_epoch());
        assert_eq!(store.listed_vectors(), NOTHING);

        // A replayed value write keeps nothing.
        let replayed = DataOp::SetNodeProperty {
            id: alix,
            key: PropertyKey::new("age"),
            value: Value::Int64(31),
        };
        apply_all(
            &store,
            &[replayed],
            Writer::Replay {
                epoch: next_epoch(&store),
            },
        )
        .unwrap();
        assert_eq!(store.listed_vectors(), NOTHING, "replay keeps nothing");

        // An immediate value write, twice on the same vector: listed once.
        for age in [32, 33] {
            immediate(
                &store,
                &DataOp::SetNodeProperty {
                    id: alix,
                    key: PropertyKey::new("age"),
                    value: Value::Int64(age),
                },
            )
            .unwrap();
        }
        // A transaction's label write, open.
        let mut tx = Tx::begin(&store, 10);
        tx.add_label(gus, "Traveller").unwrap();
        // An immediate edge create.
        let edge = EdgeId::new(store.reserve_edge_ids(1).unwrap().start);
        immediate(
            &store,
            &DataOp::CreateEdge {
                id: edge,
                src: alix,
                dst: vincent,
                edge_type: ArcStr::from("KNOWS"),
                properties: Vec::new(),
            },
        )
        .unwrap();
        let touched = vec![
            (Table::Nodes, 0),
            (Table::Nodes, VECTOR_ROWS as u64),
            (Table::Edges, 0),
        ];
        assert_eq!(store.listed_vectors(), touched);
        assert_eq!(store.vectors_with_versions(), touched);

        // The collect while the transaction is open keeps its vector.
        store.collect(store.current_epoch());
        assert_eq!(
            store.listed_vectors(),
            [(Table::Nodes, VECTOR_ROWS as u64)],
            "the open transaction's"
        );
        assert_eq!(store.vectors_with_versions(), store.listed_vectors());

        // A rollback leaves the vector listed until the next collect.
        tx.roll_back().unwrap();
        let mut tx = Tx::begin(&store, 11);
        tx.set_node(vincent, "age", Value::Int64(40)).unwrap();
        tx.roll_back().unwrap();
        assert_eq!(
            store.listed_vectors(),
            [
                (Table::Nodes, VECTOR_ROWS as u64),
                (Table::Nodes, vector_of(vincent.as_u64())),
            ]
        );
        assert_eq!(store.vectors_with_versions(), NOTHING, "both undone");
        assert_eq!(store.collect(store.current_epoch()), 0);
        assert_eq!(store.listed_vectors(), NOTHING);
    }

    /// With no reader older than each write, immediate writes of an
    /// indexed value, each collected right after, leave the store's memory
    /// and its index as they were: nothing is listed, no chain is left.
    #[test]
    fn memory_returns_to_flat_after_each_write_when_no_reader_is_older() {
        let store = RowGroupStore::new();
        store.create_property_index("age");
        let people = spread_people(&store);
        store.collect(store.current_epoch());
        let bytes = store.heap_bytes();
        for round in 0..1_000_i64 {
            let id = people[usize::try_from(round).unwrap() % people.len()];
            let op = DataOp::SetNodeProperty {
                id,
                key: PropertyKey::new("age"),
                value: Value::Int64(30 + round % 2),
            };
            immediate(&store, &op).unwrap();
            store.collect(store.current_epoch());
            assert_eq!(store.update_nodes(), 0, "round {round}");
        }
        // Every person holds 30 or 31 now: the index holds those two.
        assert_eq!(store.property_index_len("age"), Some(2));
        assert_eq!(store.listed_vectors(), NOTHING);
        assert_eq!(store.heap_bytes(), bytes, "the memory is flat");
    }

    /// One writer of the lockstep test: the same transaction on both
    /// stores, holding the rows of one partition (ids modulo 3), as the
    /// claims let one writer at a time write a row.
    struct Open<'s> {
        a: Tx<'s, RowGroupStore>,
        b: Tx<'s, RowGroupStore>,
        part: u64,
        marks: Vec<(ChangeMark, ChangeMark)>,
    }

    /// The partition of row `raw`.
    fn part(raw: u64) -> u64 {
        raw % 3
    }

    /// A fresh id, reserved alike on both stores; now and then after a
    /// jump, so the rows spread over vectors and row groups.
    fn fresh(a: &RowGroupStore, b: &RowGroupStore, rng: &mut Rng, nodes: bool) -> u64 {
        let jump = match rng.below(12) {
            0 => VECTOR_ROWS as u64,
            1 if rng.below(8) == 0 => ROWS_PER_GROUP,
            _ => 1,
        };
        let reserve = |store: &RowGroupStore| {
            let ids = if nodes {
                store.reserve_node_ids(jump)
            } else {
                store.reserve_edge_ids(jump)
            };
            ids.unwrap().end - 1
        };
        let id = reserve(a);
        assert_eq!(reserve(b), id, "both stores give the same ids");
        id
    }

    /// A random write of the rows of partition `part_of` that `seen` shows,
    /// or of fresh rows; `None` when there is nothing to write.
    fn random_op(
        a: &RowGroupStore,
        b: &RowGroupStore,
        rng: &mut Rng,
        seen: &Image,
        part_of: u64,
    ) -> Option<DataOp> {
        let nodes: Vec<u64> = seen
            .nodes
            .keys()
            .copied()
            .filter(|raw| part(*raw) == part_of)
            .collect();
        let edges: Vec<u64> = seen
            .edges
            .keys()
            .copied()
            .filter(|raw| part(*raw) == part_of)
            .collect();
        let node = rng.pick(&nodes).copied().map(NodeId::new);
        let edge = rng.pick(&edges).copied().map(EdgeId::new);
        Some(match rng.below(9) {
            0 => {
                let mut names = vec!["Person"];
                if rng.below(2) == 0 {
                    names.push("Employee");
                }
                DataOp::CreateNode {
                    id: NodeId::new(fresh(a, b, rng, true)),
                    labels: labels(&names),
                    properties: properties(&[("age", Value::Int64(rng.int(88)))]),
                }
            }
            1 => {
                let src = node?;
                let dst = NodeId::new(*rng.pick(&nodes)?);
                DataOp::CreateEdge {
                    id: EdgeId::new(fresh(a, b, rng, false)),
                    src,
                    dst,
                    edge_type: ArcStr::from("KNOWS"),
                    properties: properties(&[("since", Value::Int64(1988 + rng.int(40)))]),
                }
            }
            2 | 3 => DataOp::SetNodeProperty {
                id: node?,
                key: PropertyKey::new("age"),
                value: Value::Int64(rng.int(88)),
            },
            4 => DataOp::RemoveNodeProperty {
                id: node?,
                key: PropertyKey::new("age"),
            },
            5 => {
                let id = node?;
                let has = seen.nodes[&id.as_u64()]
                    .0
                    .iter()
                    .any(|label| label == "Traveller");
                let label = ArcStr::from("Traveller");
                if has {
                    DataOp::RemoveNodeLabel { id, label }
                } else {
                    DataOp::AddNodeLabel { id, label }
                }
            }
            6 => DataOp::SetEdgeProperty {
                id: edge?,
                key: PropertyKey::new("since"),
                value: Value::Int64(1988 + rng.int(40)),
            },
            7 => DataOp::DeleteEdge { id: edge? },
            _ => DataOp::DeleteNode { id: node? },
        })
    }

    /// The vectors `store` lists, as a set.
    fn listed(store: &RowGroupStore) -> BTreeSet<(bool, u64)> {
        store
            .listed_vectors()
            .into_iter()
            .map(|(table, first)| (table == Table::Edges, first))
            .collect()
    }

    /// The vectors that hold versions in `store`, as a set.
    fn with_versions(store: &RowGroupStore) -> BTreeSet<(bool, u64)> {
        store
            .vectors_with_versions()
            .into_iter()
            .map(|(table, first)| (table == Table::Edges, first))
            .collect()
    }

    /// Two stores see the same random writes: creates, deletes, value and
    /// label writes of nodes and edges by up to three open transactions at
    /// once (with savepoints, rollbacks to them, commits and rollbacks) and
    /// immediate writes, rows spread over vectors and row groups, an index
    /// on `age` in half the seeds. At the same random horizons no reader is
    /// older than, one store collects the vectors it lists, the other walks
    /// every vector. They take out the same, hold the same state (rows,
    /// versions, chains, index entries, list) and read alike at every view;
    /// the list always holds every vector with versions, and right after a
    /// collect those alone. In the end both are flat.
    #[test]
    fn a_collect_of_the_listed_vectors_equals_a_collect_of_every_vector() {
        for seed in 1..=30 {
            let (a, b) = (RowGroupStore::new(), RowGroupStore::new());
            if seed % 2 == 0 {
                a.create_property_index("age");
                b.create_property_index("age");
            }
            let people = spread_people(&a);
            assert_eq!(spread_people(&b), people);
            let mut rng = Rng::new(seed);
            let mut open: Vec<Open<'_>> = Vec::new();
            let mut next_transaction = 10;
            let mut floor = a.current_epoch().as_u64();
            for step in 0..500 {
                let what = format!("seed {seed}, step {step}");
                let held: BTreeSet<u64> = open.iter().map(|open| open.part).collect();
                match rng.below(20) {
                    0 | 1 if open.len() < 3 => {
                        let Some(part) = (0..3).find(|part| !held.contains(part)) else {
                            continue;
                        };
                        next_transaction += 1;
                        open.push(Open {
                            a: Tx::begin(&a, next_transaction),
                            b: Tx::begin(&b, next_transaction),
                            part,
                            marks: Vec::new(),
                        });
                    }
                    2..=8 if !open.is_empty() => {
                        let index = rng.below(open.len());
                        let seen = open[index].a.view();
                        let Some(op) = random_op(&a, &b, &mut rng, &seen, open[index].part) else {
                            continue;
                        };
                        let applied_a = open[index].a.apply(op.clone()).map_err(|e| e.to_string());
                        let applied_b = open[index].b.apply(op).map_err(|e| e.to_string());
                        assert_eq!(applied_a, applied_b, "{what}");
                    }
                    9 if !open.is_empty() => {
                        let index = rng.below(open.len());
                        let marks = (open[index].a.mark(), open[index].b.mark());
                        open[index].marks.push(marks);
                    }
                    10 if !open.is_empty() => {
                        let index = rng.below(open.len());
                        if let Some((mark_a, mark_b)) = open[index].marks.pop() {
                            open[index].a.roll_back_to(mark_a).unwrap();
                            open[index].b.roll_back_to(mark_b).unwrap();
                        }
                    }
                    11 | 12 if !open.is_empty() => {
                        let ended = open.swap_remove(rng.below(open.len()));
                        if rng.below(3) == 0 {
                            ended.a.roll_back().unwrap();
                            ended.b.roll_back().unwrap();
                        } else {
                            assert_eq!(ended.a.commit(), ended.b.commit(), "{what}");
                        }
                    }
                    13..=15 => {
                        let free: Vec<u64> = (0..3).filter(|part| !held.contains(part)).collect();
                        let Some(part) = rng.pick(&free).copied() else {
                            continue;
                        };
                        let seen = image(&a, Reader::At(a.current_epoch()));
                        let Some(op) = random_op(&a, &b, &mut rng, &seen, part) else {
                            continue;
                        };
                        assert_eq!(immediate(&a, &op), immediate(&b, &op), "{what}");
                    }
                    16 | 17 => {
                        let now = a.current_epoch().as_u64();
                        let top = open
                            .iter()
                            .map(|open| open.a.snapshot.as_u64())
                            .min()
                            .unwrap_or(now)
                            .max(floor);
                        let horizon =
                            floor + rng.below(usize::try_from(top - floor).unwrap() + 1) as u64;
                        let horizon = EpochId::new(horizon);
                        assert_eq!(a.collect(horizon), b.collect_all(horizon), "{what}");
                        floor = horizon.as_u64();
                        assert_eq!(a.state(), b.state(), "{what}: the state after the collect");
                        assert_eq!(listed(&a), with_versions(&a), "{what}: exactly listed");
                        assert_eq!(
                            a.property_index_len("age"),
                            b.property_index_len("age"),
                            "{what}: the index"
                        );
                        for age in 0..88 {
                            let age = Value::Int64(age);
                            assert_eq!(
                                a.find_nodes_maybe_equal("age", &age),
                                b.find_nodes_maybe_equal("age", &age),
                                "{what}: the index entries of {age:?}"
                            );
                        }
                    }
                    _ => {
                        let now = a.current_epoch().as_u64();
                        let reader = if !open.is_empty() && rng.below(2) == 0 {
                            let open = &open[rng.below(open.len())];
                            Reader::Transaction(open.a.id, open.a.snapshot)
                        } else {
                            let epoch =
                                floor + rng.below(usize::try_from(now - floor).unwrap() + 1) as u64;
                            Reader::At(EpochId::new(epoch))
                        };
                        let seen = image(&a, reader);
                        assert!(seen.problems.is_empty(), "{what}: {:?}", seen.problems);
                        assert_eq!(seen, image(&b, reader), "{what}: the reads");
                    }
                }
                assert!(
                    listed(&a).is_superset(&with_versions(&a)),
                    "{what}: a vector with versions is not listed"
                );
            }
            for ended in open.drain(..) {
                assert_eq!(ended.a.commit(), ended.b.commit(), "seed {seed}");
            }
            let now = a.current_epoch();
            assert_eq!(a.collect(now), b.collect_all(now), "seed {seed}");
            assert_eq!(a.state(), b.state(), "seed {seed}: in the end");
            assert_eq!(
                listed(&a),
                BTreeSet::new(),
                "seed {seed}: nothing is listed"
            );
            assert_eq!(
                (a.update_nodes(), a.stamped_vectors()),
                (0, 0),
                "seed {seed}"
            );
        }
    }
}
