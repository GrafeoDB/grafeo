//! A copy of a store's committed state, made store to store.
//!
//! [`LpgStore::committed_copy`] copies what a checkpoint writes while
//! transactions are open (see the `chunked` module, whose selection of the
//! committed nodes, edges, labels and values it shares), without writing
//! and reading the section's chunks: a copy in memory needs no codec, and a
//! build that never writes a file (the WASM build) links none for it.

#[cfg(not(feature = "temporal"))]
use std::collections::BTreeMap;

#[cfg(feature = "temporal")]
use grafeo_common::types::EpochId;
#[cfg(not(feature = "temporal"))]
use grafeo_common::types::{PropertyKey, Value};
use grafeo_common::utils::error::{Error, Result};

#[cfg(feature = "temporal")]
use super::chunked::committed_versions;
#[cfg(not(feature = "temporal"))]
use super::chunked::{column_rows, for_each_committed_value};
use super::chunked::{committed_edges, committed_nodes, section_epoch};
#[cfg(not(feature = "temporal"))]
use super::property::{EntityId, PropertyStorage};
use super::store::{OpenChangeSource, OpenChanges};
use super::{LpgStore, OpenChangesByGraph};

impl LpgStore {
    /// A new store holding the committed state of this store and its named
    /// graphs, as a checkpoint writes it while transactions are open and a
    /// load reads it back: the nodes and edges visible now and those an
    /// open transaction deleted, with the labels and values they were
    /// committed with (with `temporal`, the committed versions of each value
    /// and the store's epoch), nothing an open transaction created, and the
    /// next ids. Every graph keeps its id and its label, edge type and
    /// property key ids, so the copy writes the same ids the store would.
    /// Indexes are not copied.
    ///
    /// This copies the stores as they are, which is their committed state
    /// while no transaction is open; [`committed_copy_with`](Self::committed_copy_with)
    /// takes what open transactions changed. The copy costs as much memory
    /// as the committed data.
    ///
    /// # Errors
    ///
    /// Returns the error of reading a node or edge record or a spilled
    /// property value, and an allocation error of the copy.
    pub fn committed_copy(&self) -> Result<Self> {
        self.committed_copy_from(OpenChangeSource::None)
    }

    /// [`committed_copy`](Self::committed_copy) while transactions are open,
    /// reading what they changed from `changes`, indexed from their change
    /// sets: the stores and the change sets are read together, so the caller
    /// holds their writes, rollbacks and commits for the whole copy, as a
    /// checkpoint does (see [`OpenChangesByGraph`]); a change made meanwhile
    /// may be copied in part.
    ///
    /// # Errors
    ///
    /// As [`committed_copy`](Self::committed_copy).
    pub fn committed_copy_with(&self, changes: &OpenChangesByGraph) -> Result<Self> {
        self.committed_copy_from(OpenChangeSource::ChangeSets(changes))
    }

    fn committed_copy_from(&self, open: OpenChangeSource<'_>) -> Result<Self> {
        let copy = Self::new()?;
        let epoch = section_epoch(self);
        copy_graph(self, &copy, epoch, &open.changes_of(None))?;
        // A graph dropped since the names were read is left out.
        let graphs: Vec<(String, std::sync::Arc<LpgStore>)> = self
            .graph_names()
            .into_iter()
            .filter_map(|name| self.graph(&name).map(|graph| (name, graph)))
            .collect();
        let ids: Vec<(u32, &str)> = graphs
            .iter()
            .map(|(name, graph)| (graph.graph_id(), name.as_str()))
            .collect();
        // Read after the graphs: above every graph id copied.
        let targets = copy
            .restore_graphs(&ids, self.next_graph_id())
            .map_err(|error| Error::Internal(format!("the committed copy: {error}")))?;
        // In any order: each graph is copied on its own.
        for ((name, graph), target) in graphs.iter().zip(&targets) {
            copy_graph(graph, target, epoch, &open.changes_of(Some(name.as_str())))?;
        }
        Ok(copy)
    }
}

/// Copies the committed state of `graph` (one store, whose open
/// transactions changed `changes`) into `target`, a new store: its name
/// dictionaries (ids included, so the nodes and edges find their names'
/// ids), its nodes, then its edges, then their values, then its next ids
/// and, with `temporal`, `epoch` (the root store's).
fn copy_graph(
    graph: &LpgStore,
    target: &LpgStore,
    epoch: u64,
    changes: &OpenChanges,
) -> Result<()> {
    target.copy_name_dictionaries(graph);
    let mut nodes = Vec::new();
    for (id, labels) in committed_nodes(graph, changes, epoch)? {
        let labels: Vec<&str> = labels.iter().map(|label| label.as_str()).collect();
        target.create_node_with_id(id, &labels)?;
        nodes.push(id);
    }
    let mut edges = Vec::new();
    for edge in committed_edges(graph, changes) {
        let edge = edge?;
        target.create_edge_with_id(edge.id, edge.src, edge.dst, &edge.edge_type)?;
        edges.push(edge.id);
    }

    #[cfg(not(feature = "temporal"))]
    {
        copy_values(
            &graph.node_properties,
            changes.node_values(),
            &nodes,
            |id, key, value| target.set_node_property(id, key.as_str(), value),
        )?;
        copy_values(
            &graph.edge_properties,
            changes.edge_values(),
            &edges,
            |id, key, value| target.set_edge_property(id, key.as_str(), value),
        )?;
    }
    #[cfg(feature = "temporal")]
    {
        for &id in &nodes {
            for (key, versions) in committed_versions(&graph.node_properties, id) {
                for (at, value) in versions {
                    target.set_node_property_at_epoch(id, key.as_str(), value, at);
                }
            }
        }
        for &id in &edges {
            for (key, versions) in committed_versions(&graph.edge_properties, id) {
                for (at, value) in versions {
                    target.set_edge_property_at_epoch(id, key.as_str(), value, at);
                }
            }
        }
        target.sync_epoch(EpochId::new(epoch));
    }

    target.set_next_node_id(target.next_node_id().max(graph.next_node_id()));
    target.set_next_edge_id(target.next_edge_id().max(graph.next_edge_id()));
    Ok(())
}

/// Copies the committed values of the property columns of `properties` (and
/// of the keys only a value an open transaction replaced has, `committed`)
/// of the nodes or edges in `present` (ascending), through `set`.
///
/// # Errors
///
/// Returns the error of reading a spilled value.
#[cfg(not(feature = "temporal"))]
fn copy_values<Id: EntityId>(
    properties: &PropertyStorage<Id>,
    committed: &BTreeMap<PropertyKey, Vec<(Id, Option<Value>)>>,
    present: &[Id],
    set: impl Fn(Id, &PropertyKey, Value),
) -> Result<()> {
    // Each key once, in any order: the copy holds the same values whatever
    // order they are set in, and a sort would link one more sort into the
    // WASM build.
    let mut keys = properties.keys();
    let replaced_only: Vec<PropertyKey> = committed
        .keys()
        .filter(|key| !keys.contains(key))
        .cloned()
        .collect();
    keys.extend(replaced_only);
    for key in keys {
        let stored = properties.column_ids(&key);
        let replaced = committed.get(&key).map_or(&[][..], Vec::as_slice);
        let rows = column_rows(&stored, replaced, present);
        for_each_committed_value(properties, &key, &rows, |id, value| {
            set(id, &key, value);
            Ok(())
        })?;
    }
    Ok(())
}
