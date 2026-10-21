//! Named graphs: the row-group stores a root store holds by name, as
//! `LpgStore` holds them (P1).
//!
//! The root store is the default graph, id 0; each named graph is a store of
//! its own with the id it was created with. Ids only grow, so a dropped
//! graph's id is never given again, and a graph created again under a
//! dropped name is a new store with a new id. A dropped graph's store
//! refuses every write through its [`ChangeTarget`](crate::graph::apply::ChangeTarget),
//! so a writer that resolved it before the drop changes nothing a commit
//! would log for it.
//!
//! A section holds the root and its named graphs together:
//! [`RowGroupStore::write_section`] writes them in id order with the next
//! graph id, [`RowGroupStore::read_section`] restores them with their ids.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use grafeo_common::memory::arena::AllocError;
use grafeo_common::storage::{ChunkCaps, SectionSink, SectionSource};
use grafeo_common::types::EpochId;
use grafeo_common::utils::error::{Error, Result};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use parking_lot::RwLock;

use super::RowGroupStore;
use super::section::{
    GraphToWrite, LoadedGraph, read_lpg_section_with_backward_edges, write_lpg_section_with_history,
};
use crate::graph::apply::ApplyError;

/// A store's place among the graphs: the named graphs it holds (a root's),
/// its own graph id, the next id, and whether it was dropped.
#[derive(Debug)]
pub(super) struct Graphs {
    /// The named graphs by name, in a hash map as `LpgStore`'s: its seed is
    /// its own, so the names list in no promised order.
    named: RwLock<FxHashMap<String, Arc<RowGroupStore>>>,
    /// This graph's id in the file: 0 for the default graph, the id it was
    /// created with for a named graph. Atomic only so a load can give a
    /// graph an earlier step of the same open created the id the file holds.
    id: AtomicU32,
    /// The id the next named graph created in this store gets (1 for a new
    /// store); it only grows.
    next_id: AtomicU32,
    /// Whether this store is a named graph that was dropped.
    dropped: AtomicBool,
}

impl Default for Graphs {
    fn default() -> Self {
        Self {
            named: RwLock::new(FxHashMap::default()),
            id: AtomicU32::new(0),
            next_id: AtomicU32::new(1),
            dropped: AtomicBool::new(false),
        }
    }
}

impl Graphs {
    /// Takes the next graph id. The caller holds `named` for writing, so ids
    /// are given out one at a time.
    fn take_id(&self) -> std::result::Result<u32, AllocError> {
        let id = self.next_id.load(Ordering::Acquire);
        let next = id
            .checked_add(1)
            .ok_or(AllocError::IdsExhausted { what: "graph" })?;
        self.next_id.store(next, Ordering::Release);
        Ok(id)
    }

    /// A new store for a named graph of `root`, with the next graph id,
    /// keeping history and backward edges as the root does.
    fn new_graph(
        &self,
        root: &RowGroupStore,
    ) -> std::result::Result<Arc<RowGroupStore>, AllocError> {
        let store = RowGroupStore::with_backward_edges(root.keeps_backward_edges());
        store.graphs.id.store(self.take_id()?, Ordering::Release);
        store.set_keeps_history(root.keeps_history());
        Ok(Arc::new(store))
    }
}

impl RowGroupStore {
    /// Returns a named graph by name, or `None` if it does not exist.
    #[must_use]
    pub fn graph(&self, name: &str) -> Option<Arc<RowGroupStore>> {
        self.graphs.named.read().get(name).cloned()
    }

    /// Returns a named graph, creating it if it does not exist.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError::IdsExhausted`] once every graph id was given
    /// out.
    pub fn graph_or_create(
        &self,
        name: &str,
    ) -> std::result::Result<Arc<RowGroupStore>, AllocError> {
        if let Some(graph) = self.graph(name) {
            return Ok(graph);
        }
        let mut named = self.graphs.named.write();
        // Created meanwhile, between the two locks.
        if let Some(graph) = named.get(name) {
            return Ok(Arc::clone(graph));
        }
        let graph = self.graphs.new_graph(self)?;
        named.insert(name.to_string(), Arc::clone(&graph));
        Ok(graph)
    }

    /// Creates a named graph. Returns `true` on success, `false` if it
    /// already exists.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError::IdsExhausted`] once every graph id was given
    /// out.
    pub fn create_graph(&self, name: &str) -> std::result::Result<bool, AllocError> {
        let mut named = self.graphs.named.write();
        if named.contains_key(name) {
            return Ok(false);
        }
        let graph = self.graphs.new_graph(self)?;
        named.insert(name.to_string(), graph);
        Ok(true)
    }

    /// Drops a named graph. Returns `false` if it did not exist. Its id is
    /// not given out again: a graph created later under the same name gets
    /// a new one.
    ///
    /// The dropped graph's store refuses every write through its
    /// [`ChangeTarget`](crate::graph::apply::ChangeTarget) from now on (see
    /// [`is_dropped`](Self::is_dropped)).
    pub fn drop_graph(&self, name: &str) -> bool {
        let removed = self.graphs.named.write().remove(name);
        if let Some(graph) = &removed {
            graph.graphs.dropped.store(true, Ordering::Release);
        }
        removed.is_some()
    }

    /// Whether this store is a named graph that was dropped (see
    /// [`drop_graph`](Self::drop_graph)).
    #[must_use]
    pub fn is_dropped(&self) -> bool {
        self.graphs.dropped.load(Ordering::Acquire)
    }

    /// Returns all named graph names, in no promised order (as `LpgStore`).
    #[must_use]
    pub fn graph_names(&self) -> Vec<String> {
        self.graphs.named.read().keys().cloned().collect()
    }

    /// The named graphs that list a vector, the ones a collect has work in
    /// (see [`gc_versions`](Self::gc_versions)).
    pub(super) fn listing_graphs(&self) -> Vec<Arc<RowGroupStore>> {
        self.graphs
            .named
            .read()
            .values()
            .filter(|graph| graph.has_listed())
            .cloned()
            .collect()
    }

    /// Returns the number of named graphs.
    #[must_use]
    pub fn graph_count(&self) -> usize {
        self.graphs.named.read().len()
    }

    /// This graph's id: 0 for the default graph, the id it was created with
    /// for a named graph.
    #[must_use]
    pub(crate) fn graph_id(&self) -> u32 {
        self.graphs.id.load(Ordering::Acquire)
    }

    /// The id the next named graph created in this store gets.
    #[must_use]
    pub(crate) fn next_graph_id(&self) -> u32 {
        self.graphs.next_id.load(Ordering::Acquire)
    }

    /// Restores the named graphs `graphs` (`(id, name)`, each id from 1 and
    /// below `next_graph_id`, each id and name once) with their ids, as a
    /// load does, and makes `next_graph_id` the next graph id at least.
    /// Returns the graphs' stores, in the order of `graphs`.
    ///
    /// A graph this store holds already was created in this process (an
    /// earlier step of the open creates some, as the catalog does for a
    /// schema's graphs), so its id was never written: a listed one takes its
    /// id from `graphs`, and one not listed moves to an id at or above
    /// `next_graph_id`, so no two graphs share an id.
    ///
    /// # Errors
    ///
    /// Returns what is wrong when an id is 0 or not below `next_graph_id`,
    /// or when every graph id is given out; nothing changed then.
    pub(crate) fn restore_graphs(
        &self,
        graphs: &[(u32, &str)],
        next_graph_id: u32,
    ) -> std::result::Result<Vec<Arc<RowGroupStore>>, String> {
        let mut named = self.graphs.named.write();
        if let Some((id, name)) = graphs
            .iter()
            .find(|(id, _)| *id == 0 || *id >= next_graph_id)
        {
            return Err(format!(
                "graph {name:?} has id {id}, where named graphs have ids from 1 below \
                 {next_graph_id}"
            ));
        }
        let listed: FxHashSet<&str> = graphs.iter().map(|(_, name)| *name).collect();
        let unlisted = named
            .keys()
            .filter(|name| !listed.contains(name.as_str()))
            .count();
        let next = self.next_graph_id().max(next_graph_id);
        if u32::try_from(unlisted)
            .ok()
            .and_then(|count| next.checked_add(count))
            .is_none()
        {
            return Err(AllocError::IdsExhausted { what: "graph" }.to_string());
        }
        self.graphs.next_id.store(next, Ordering::Release);
        for (name, graph) in named.iter() {
            if !listed.contains(name.as_str()) {
                let id = self.graphs.take_id().map_err(|error| error.to_string())?;
                graph.graphs.id.store(id, Ordering::Release);
            }
        }
        let backward_edges = self.keeps_backward_edges();
        let mut restored = Vec::with_capacity(graphs.len());
        for &(id, name) in graphs {
            let graph =
                Arc::clone(named.entry(name.to_string()).or_insert_with(|| {
                    Arc::new(RowGroupStore::with_backward_edges(backward_edges))
                }));
            graph.graphs.id.store(id, Ordering::Release);
            graph.set_keeps_history(self.keeps_history());
            restored.push(graph);
        }
        Ok(restored)
    }

    /// Refuses every write once the graph is dropped: a writer that resolved
    /// the graph before would log its write under the graph's name, and
    /// replay would create the graph again for it.
    pub(super) fn refuse_if_dropped(&self) -> std::result::Result<(), ApplyError> {
        if self.is_dropped() {
            return Err(ApplyError::Refused(
                "the graph was dropped: it takes no more writes".to_string(),
            ));
        }
        Ok(())
    }

    /// Writes the `LPG_STORE` section (version 4) of this store and its
    /// named graphs at the latest epoch any of them was stamped or moved at
    /// (see [`write_section_at`](Self::write_section_at)).
    ///
    /// # Errors
    ///
    /// As [`write_lpg_section`](super::section::write_lpg_section).
    pub fn write_section(&self, caps: ChunkCaps, sink: &mut dyn SectionSink) -> Result<()> {
        let epoch = self
            .graph_names()
            .iter()
            .filter_map(|name| self.graph(name))
            .map(|graph| graph.current_epoch())
            .fold(self.current_epoch(), std::cmp::max);
        self.write_section_at(epoch, caps, sink)
    }

    /// Writes the `LPG_STORE` section (version 4) of this store and its
    /// named graphs: the default graph (id 0) first, then the named graphs
    /// by id, each graph's committed state at `epoch` (the rows, values and
    /// labels a reader at `epoch` sees: nothing of a transaction still
    /// open, nothing committed later), and the next graph id. A graph
    /// dropped while the names are read is left out.
    ///
    /// `epoch` is at or after the horizon the graphs were last collected at;
    /// a checkpoint passes the epoch it records, which every commit it holds
    /// is stamped at or before.
    ///
    /// # Errors
    ///
    /// As [`write_lpg_section`](super::section::write_lpg_section).
    pub fn write_section_at(
        &self,
        epoch: EpochId,
        caps: ChunkCaps,
        sink: &mut dyn SectionSink,
    ) -> Result<()> {
        self.write_section_with_history(epoch, None, caps, sink)
    }

    /// Writes the `LPG_STORE` section as
    /// [`write_section_at`](Self::write_section_at) does, and with
    /// `history_from` (an epoch H before `epoch`, at or after the horizon
    /// the graphs were last collected at) each graph's history from H: what
    /// a reader at any epoch from H to `epoch` reads (see
    /// [`write_lpg_section_with_history`]). A checkpoint of a database that
    /// keeps history passes the start of its retention window.
    ///
    /// # Errors
    ///
    /// As [`write_lpg_section_with_history`].
    pub fn write_section_with_history(
        &self,
        epoch: EpochId,
        history_from: Option<EpochId>,
        caps: ChunkCaps,
        sink: &mut dyn SectionSink,
    ) -> Result<()> {
        let mut named: Vec<(String, Arc<RowGroupStore>)> = self
            .graphs
            .named
            .read()
            .iter()
            .map(|(name, graph)| (name.clone(), Arc::clone(graph)))
            .collect();
        named.sort_unstable_by_key(|(_, graph)| graph.graph_id());
        let graphs: Vec<GraphToWrite<'_>> = std::iter::once(GraphToWrite {
            id: 0,
            name: "",
            store: self,
        })
        .chain(named.iter().map(|(name, graph)| GraphToWrite {
            id: graph.graph_id(),
            name: name.as_str(),
            store: graph,
        }))
        .collect();
        // Read after the graphs: above every graph id written.
        write_lpg_section_with_history(
            &graphs,
            self.next_graph_id(),
            epoch,
            history_from,
            caps,
            sink,
        )
    }

    /// Reads an `LPG_STORE` section of version 4 into this store, a root:
    /// the default graph's data into the store itself, each named graph's
    /// into the store of its name (one created before the load is kept,
    /// see `restore_graphs`), with the file's ids
    /// and next graph id. What a graph held before is replaced; a graph
    /// the section does not name keeps its data. The graphs keep the
    /// adjacency this store keeps, whatever the section holds (see
    /// [`read_lpg_section_with_backward_edges`]).
    ///
    /// # Errors
    ///
    /// As [`read_lpg_section_with_backward_edges`], and
    /// [`Error::Corruption`] for a default graph with a name or a name two
    /// graphs share. Nothing changed then.
    pub fn read_section(&self, source: &dyn SectionSource) -> Result<()> {
        let (next_graph_id, loaded) =
            read_lpg_section_with_backward_edges(source, self.keeps_backward_edges())?;
        let mut default: Option<RowGroupStore> = None;
        let mut named: Vec<LoadedGraph> = Vec::with_capacity(loaded.len());
        let mut names: FxHashSet<String> = FxHashSet::default();
        for graph in loaded {
            if graph.id == 0 {
                if !graph.name.is_empty() {
                    return Err(Error::corruption(format!(
                        "LPG metadata chunk: graph 0, the default graph, is named {:?}",
                        graph.name
                    )));
                }
                default = Some(graph.store);
            } else if !names.insert(graph.name.clone()) {
                return Err(Error::corruption(format!(
                    "LPG metadata chunk: two graphs are named {:?}",
                    graph.name
                )));
            } else {
                named.push(graph);
            }
        }
        let listed: Vec<(u32, &str)> = named
            .iter()
            .map(|graph| (graph.id, graph.name.as_str()))
            .collect();
        let targets = self
            .restore_graphs(&listed, next_graph_id)
            .map_err(|error| Error::corruption(format!("LPG section: {error}")))?;
        if let Some(store) = default {
            self.install(store);
        }
        for (target, graph) in targets.iter().zip(named) {
            target.install(graph.store);
        }
        Ok(())
    }

    /// Replaces this store's data with `loaded`'s: its rows, dictionaries
    /// and counts, and its indexes, which a loaded store has none of (the
    /// loader builds them from the rows, as the catalog defines them); the
    /// next ids and the epoch move to the loaded ones when those are later.
    /// The named graphs and the graph id stay.
    fn install(&self, loaded: RowGroupStore) {
        let listed = loaded.has_listed();
        *self.inner.write() = loaded.inner.into_inner();
        if listed {
            // The history it loaded: the next collect has work.
            self.note_listed();
        }
        self.next_node_id
            .fetch_max(loaded.next_node_id.into_inner(), Ordering::AcqRel);
        self.next_edge_id
            .fetch_max(loaded.next_edge_id.into_inner(), Ordering::AcqRel);
        self.epoch
            .fetch_max(loaded.epoch.into_inner(), Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use grafeo_common::storage::{
        ChunkCaps, ImageSource, MemoryImage, SectionSink, SectionSource, SectionType,
    };
    use grafeo_common::types::{EpochId, Value};
    use grafeo_common::utils::error::Error;

    use crate::graph::apply::ChangeTarget;
    use crate::graph::conformance::{Tx, committed, people};
    use crate::graph::rowgroup::RowGroupStore;
    use crate::graph::rowgroup::section::{GraphToWrite, LPG_SECTION_VERSION, write_lpg_section};
    use crate::graph::traits::GraphStore;

    /// An image holding one LPG section, written by `write`.
    fn image(write: impl FnOnce(&mut dyn SectionSink)) -> MemoryImage {
        let mut image = MemoryImage::new();
        image
            .begin_section(SectionType::LpgStore, LPG_SECTION_VERSION)
            .unwrap();
        write(&mut image);
        image
    }

    fn source(image: &MemoryImage) -> Box<dyn SectionSource + '_> {
        image.section_source(SectionType::LpgStore).unwrap()
    }

    /// A root with people, trips (Vincent visited Paris), an empty archive,
    /// and a gap where travel was: its id is not given again.
    fn travels() -> RowGroupStore {
        let root = RowGroupStore::new();
        people(&root).unwrap();
        root.create_graph("trips").unwrap();
        root.create_graph("travel").unwrap();
        root.create_graph("archive").unwrap();
        let trips = root.graph("trips").unwrap();
        let mut tx = Tx::begin(&*trips, 10);
        let vincent = tx
            .create_node(&["Person"], &[("name", Value::from("Vincent"))])
            .unwrap();
        let paris = tx
            .create_node(&["City"], &[("name", Value::from("Paris"))])
            .unwrap();
        tx.create_edge(vincent, paris, "VISITED", &[("since", Value::Int64(1994))])
            .unwrap();
        tx.commit().unwrap();
        let travel = root.graph("travel").unwrap();
        let mut tx = Tx::begin(&*travel, 11);
        tx.create_node(&["Person"], &[]).unwrap();
        tx.commit().unwrap();
        assert!(root.drop_graph("travel"));
        root
    }

    /// Checks that `loaded` holds what `store` had committed, and its ids.
    fn assert_same(loaded: &RowGroupStore, store: &RowGroupStore, what: &str) {
        let (found, expected) = (committed(loaded), committed(store));
        found.consistent(what).unwrap();
        assert_eq!(
            (&found.nodes, &found.edges),
            (&expected.nodes, &expected.edges),
            "{what}"
        );
        assert_eq!(loaded.all_labels(), store.all_labels(), "{what}: labels");
        assert_eq!(loaded.node_count(), store.node_count(), "{what}: count");
        assert_eq!(loaded.graph_id(), store.graph_id(), "{what}: the graph id");
        let next = |store: &RowGroupStore| {
            (
                store.reserve_node_ids(1).unwrap().start,
                store.reserve_edge_ids(1).unwrap().start,
            )
        };
        assert_eq!(next(loaded), next(store), "{what}: the next ids");
    }

    /// A root and its named graphs round trip through one section: the
    /// names, ids, data and next ids of each, and the next graph id past a
    /// dropped graph's; the loaded root writes the same section again.
    #[test]
    fn a_root_with_named_graphs_round_trips() {
        let root = travels();
        let written = image(|sink| root.write_section(ChunkCaps::DEFAULT, sink).unwrap());
        let loaded = RowGroupStore::new();
        loaded.read_section(&*source(&written)).unwrap();

        let again = image(|sink| loaded.write_section(ChunkCaps::DEFAULT, sink).unwrap());
        let (first, second) = (source(&written), source(&again));
        assert_eq!(first.chunks(), second.chunks(), "the same chunks again");
        for index in 0..first.chunks().len() {
            assert_eq!(first.fetch(index).unwrap(), second.fetch(index).unwrap());
        }

        let mut names = loaded.graph_names();
        names.sort();
        assert_eq!(names, ["archive", "trips"]);
        assert_eq!(loaded.next_graph_id(), 4, "past travel's id");
        assert_same(&loaded, &root, "the root");
        for name in ["trips", "archive"] {
            let graph = loaded.graph(name).unwrap();
            assert!(!graph.is_dropped());
            assert_same(&graph, &root.graph(name).unwrap(), name);
        }
        assert_eq!(loaded.graph("trips").unwrap().node_count(), 2);
        loaded.create_graph("travel").unwrap();
        assert_eq!(
            loaded.graph("travel").unwrap().graph_id(),
            4,
            "travel created again gets a new id"
        );
        assert_eq!(loaded.graph("travel").unwrap().node_count(), 0);
    }

    /// A checkpoint during open writes in the root and in a named graph
    /// writes each graph's committed state: no value, label, row or edge an
    /// open transaction wrote, and every one it deleted or changed as it was
    /// committed.
    #[test]
    fn open_writes_in_named_graphs_are_not_in_the_section() {
        use grafeo_common::types::{EdgeId, NodeId};

        let root = travels();
        let trips = root.graph("trips").unwrap();
        let (root_before, trips_before) = (committed(&root), committed(&*trips));
        let (vincent, paris, visit) = (NodeId::new(0), NodeId::new(1), EdgeId::new(0));
        assert_eq!(
            trips.get_node(vincent).map(|node| node.labels.len()),
            Some(1),
            "Vincent of trips"
        );

        let mut in_trips = Tx::begin(&*trips, 20);
        in_trips
            .set_node(vincent, "name", Value::from("Vince"))
            .unwrap();
        in_trips.add_label(vincent, "Traveller").unwrap();
        in_trips.remove_label(paris, "City").unwrap();
        in_trips
            .set_edge(visit, "since", Value::Int64(2030))
            .unwrap();
        let berlin = in_trips
            .create_node(&["City"], &[("name", Value::from("Berlin"))])
            .unwrap();
        in_trips
            .create_edge(vincent, berlin, "VISITED", &[])
            .unwrap();
        let mut in_root = Tx::begin(&root, 21);
        in_root
            .set_node(NodeId::new(0), "age", Value::Int64(88))
            .unwrap();
        in_root.delete_edge(EdgeId::new(1)).unwrap();

        let written = image(|sink| root.write_section(ChunkCaps::DEFAULT, sink).unwrap());
        let loaded = RowGroupStore::new();
        loaded.read_section(&*source(&written)).unwrap();
        let loaded_trips = loaded.graph("trips").unwrap();
        for (graph, before, what) in [
            (&loaded, root_before, "the root"),
            (&*loaded_trips, trips_before, "trips"),
        ] {
            let found = committed(graph);
            found.consistent(what).unwrap();
            assert_eq!(
                (&found.nodes, &found.edges),
                (&before.nodes, &before.edges),
                "{what}"
            );
        }
    }

    /// A load into a root that holds graphs already (an earlier step of the
    /// open creates some) loads into the store a name holds, which keeps its
    /// handles, gives it the file's id, and moves a graph the file does not
    /// name past the file's ids, with its data.
    #[test]
    fn a_load_keeps_the_graphs_created_before_it() {
        let root = travels();
        let written = image(|sink| root.write_section(ChunkCaps::DEFAULT, sink).unwrap());
        let loaded = RowGroupStore::new();
        loaded.create_graph("archive").unwrap();
        loaded.create_graph("scratch").unwrap();
        let archive = loaded.graph("archive").unwrap();
        let scratch = loaded.graph("scratch").unwrap();
        let mut tx = Tx::begin(&*scratch, 3);
        tx.create_node(&["Person"], &[("name", Value::from("Mia"))])
            .unwrap();
        tx.commit().unwrap();
        assert_eq!((archive.graph_id(), scratch.graph_id()), (1, 2));

        loaded.read_section(&*source(&written)).unwrap();
        assert!(Arc::ptr_eq(&archive, &loaded.graph("archive").unwrap()));
        assert_eq!(
            (
                loaded.graph("trips").unwrap().graph_id(),
                archive.graph_id(),
                scratch.graph_id(),
                loaded.next_graph_id()
            ),
            (1, 3, 4, 5),
            "the file's ids, and the unlisted graph past them"
        );
        assert_eq!(
            scratch.node_count(),
            1,
            "a graph the file does not name keeps its data"
        );
        assert_same(
            &loaded.graph("trips").unwrap(),
            &root.graph("trips").unwrap(),
            "trips",
        );
    }

    /// A section whose default graph has a name, or whose named graphs
    /// share a name, is refused as corruption, and the root is unchanged.
    #[test]
    fn graphs_a_writer_does_not_write_are_refused() {
        let (alix, gus) = (RowGroupStore::new(), RowGroupStore::new());
        people(&alix).unwrap();
        for graphs in [
            vec![GraphToWrite {
                id: 0,
                name: "trips",
                store: &alix,
            }],
            vec![
                GraphToWrite {
                    id: 0,
                    name: "",
                    store: &alix,
                },
                GraphToWrite {
                    id: 1,
                    name: "trips",
                    store: &gus,
                },
                GraphToWrite {
                    id: 2,
                    name: "trips",
                    store: &gus,
                },
            ],
        ] {
            let written = image(|sink| {
                write_lpg_section(&graphs, 3, EpochId::new(1), ChunkCaps::DEFAULT, sink).unwrap();
            });
            let root = RowGroupStore::new();
            root.create_graph("travel").unwrap();
            let refused = root.read_section(&*source(&written)).unwrap_err();
            assert!(matches!(refused, Error::Corruption(_)), "{refused:?}");
            assert_eq!(
                (root.node_count(), root.graph_names(), root.next_graph_id()),
                (0, vec!["travel".to_string()], 2),
                "nothing changed"
            );
        }
    }
}
