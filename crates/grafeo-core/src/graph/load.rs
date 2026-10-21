//! Rows loaded under the ids they were written with, outside any
//! transaction: what the readers of the formats of earlier releases write
//! into a store (the LPG sections of 0.5.x and of 0.6.0-beta, version 3; the
//! compacted base of a 0.5.x file; the snapshots; the data records of the WAL
//! of the first log format), and the copies of graph data.
//!
//! A reader writes through [`LoadTarget`], which both stores implement, so
//! one reader fills `LpgStore` (the engine's store until 0.6.0 switched to
//! the row-group store, #412 H2d) and the row-group store, which takes every row
//! through its change target's replay writer at its current epoch: no
//! reader reads `LpgStore` to fill the row-group store. A load runs while
//! nothing else reads or writes the store, so it builds no before-image and
//! keeps nothing for other readers.
//!
//! Versions: a reader passes each property's current value
//! ([`LoadTarget::load_value`]) and the older versions a file or snapshot
//! keeps ([`LoadTarget::load_old_value`]). `LpgStore` keeps those with the
//! `temporal` feature, at their epochs; the row-group store keeps the
//! current value only, as `LpgStore` without `temporal` does (its retained
//! history comes with time travel, H2c).

use std::sync::Arc;

use grafeo_common::change::Entity;
use grafeo_common::types::{EdgeId, EpochId, NodeId, Value};
use grafeo_common::utils::error::Result;

use super::traits::GraphStore;

/// The names of one graph's dictionaries as a file holds them, id for id:
/// the labels, the edge types and the property keys, with the next id of
/// each (the ids below it are never given out again, also those that name
/// nothing).
#[derive(Debug, Clone, Copy)]
pub struct LoadedNames<'n> {
    /// The labels, by id.
    pub labels: &'n [(u32, String)],
    /// The edge types, by id.
    pub edge_types: &'n [(u32, String)],
    /// The property keys, by id.
    pub keys: &'n [(u32, String)],
    /// The next label, edge type and property key ids.
    pub next_ids: [u32; 3],
}

/// A store that loads rows under the ids it is given (see the module docs).
///
/// Every method changes the store at once, as recovery does; none may run
/// while a transaction is open in the store or a reader reads it.
pub trait LoadTarget: GraphStore + Send + Sync + Sized {
    /// Creates node `id` with `labels` and no values.
    ///
    /// # Errors
    ///
    /// When the store holds a node `id` already, or cannot allocate it.
    fn load_node(&self, id: NodeId, labels: &[&str]) -> Result<()>;

    /// Creates edge `id` from `src` to `dst` of type `edge_type`, with no
    /// values; returns whether the store holds it now. An edge whose end
    /// the store does not hold (only a damaged file or log holds one) is
    /// kept by `LpgStore`, which shows it to no query, and left out by the
    /// row-group store (`false`), whose caller says so.
    ///
    /// # Errors
    ///
    /// When the store holds an edge `id` already, or cannot allocate it.
    fn load_edge(&self, id: EdgeId, src: NodeId, dst: NodeId, edge_type: &str) -> Result<bool>;

    /// Sets the current value of `key` of `entity`, written at `epoch`
    /// (`None`: at the store's epoch). A value of an entity the store does
    /// not hold changes nothing, as a property set of a recovery does.
    ///
    /// # Errors
    ///
    /// When the store refuses the value.
    fn load_value(
        &self,
        entity: Entity,
        key: &str,
        value: Value,
        epoch: Option<EpochId>,
    ) -> Result<()>;

    /// An older version of `key` of `entity`, written at `epoch` and replaced
    /// since; the versions of a key come oldest first, before its current
    /// value. A store that keeps no history drops it.
    fn load_old_value(&self, entity: Entity, key: &str, value: Value, epoch: EpochId);

    /// Removes the value of `key` of `entity`; nothing when it has none.
    ///
    /// # Errors
    ///
    /// When the value cannot be read (a spilled one of `LpgStore`).
    fn load_remove_value(&self, entity: Entity, key: &str) -> Result<()>;

    /// Deletes `entity`; returns whether the store held it.
    ///
    /// # Errors
    ///
    /// When the store refuses the delete (a node whose edges it still holds,
    /// in the row-group store).
    fn load_delete(&self, entity: Entity) -> Result<bool>;

    /// Gives node `id` the label `label` (`add`) or takes it away; returns
    /// whether that changed the node.
    ///
    /// # Errors
    ///
    /// When the store refuses the write.
    fn load_label(&self, id: NodeId, label: &str, add: bool) -> Result<bool>;

    /// Forgets the label `name` when no node has it: its id is never given
    /// out again, and the next checkpoint writes the dictionary without it.
    /// Returns whether it was forgotten.
    fn load_forget_label(&self, name: &str) -> bool;

    /// Gives the store the names of a file's dictionaries, id for id, before
    /// any row names them.
    ///
    /// # Errors
    ///
    /// Returns what is wrong when an id or a name is taken.
    fn load_names(&self, names: LoadedNames<'_>) -> std::result::Result<(), String>;

    /// Makes the next node and edge ids at least `node` and `edge`.
    fn load_next_ids(&self, node: u64, edge: u64);

    /// The epoch the section or snapshot was written at, once its rows are
    /// in: the store continues at it (`LpgStore` only with `temporal`, which
    /// keeps the epochs of the versions it loaded).
    fn load_section_epoch(&self, epoch: EpochId);

    /// Restores the named graphs `graphs` (`(id, name)`) with their ids, and
    /// makes `next_graph_id` the next graph id at least. Returns their
    /// stores, in the order of `graphs`.
    ///
    /// # Errors
    ///
    /// Returns what is wrong when an id is 0 or not below `next_graph_id`,
    /// or the graph ids are exhausted.
    fn load_graphs(
        &self,
        graphs: &[(u32, &str)],
        next_graph_id: u32,
    ) -> std::result::Result<Vec<Arc<Self>>, String>;

    /// The named graph `name`, created when the store has none of that name.
    ///
    /// # Errors
    ///
    /// When the graph ids are exhausted.
    fn load_named_graph(&self, name: &str) -> Result<Arc<Self>>;

    /// Ends the load of this graph's rows: the row-group store collapses the
    /// version info of the rows it loaded, so a loaded graph costs what one
    /// read from a section of this release does.
    fn load_finish(&self);
}

/// What the tests of the loaders compare: two stores holding the same rows.
#[cfg(test)]
pub(crate) mod testing {
    use std::collections::BTreeMap;

    use grafeo_common::types::{EdgeId, NodeId, Value};

    use crate::graph::Direction;
    use crate::graph::traits::GraphStore;

    /// A node's sorted labels and its values by key.
    type NodeRow = (Vec<String>, BTreeMap<String, Value>);
    /// An edge's ends, type and values by key.
    type EdgeRow = (NodeId, NodeId, String, BTreeMap<String, Value>);

    /// Every node and edge of `store` as a reader now sees it, by id.
    pub(crate) fn rows(
        store: &dyn GraphStore,
    ) -> (BTreeMap<NodeId, NodeRow>, BTreeMap<EdgeId, EdgeRow>) {
        let mut nodes = BTreeMap::new();
        let mut edges = BTreeMap::new();
        for id in store.node_ids() {
            let node = store.get_node(id).expect("a node it lists");
            let mut labels: Vec<String> = node.labels.iter().map(ToString::to_string).collect();
            labels.sort();
            let values = node
                .properties
                .iter()
                .map(|(key, value)| (key.to_string(), value.clone()))
                .collect();
            nodes.insert(id, (labels, values));
            for (_, edge) in store.edges_from(id, Direction::Outgoing) {
                let edge = store.get_edge(edge).expect("an edge it lists");
                let values = edge
                    .properties
                    .iter()
                    .map(|(key, value)| (key.to_string(), value.clone()))
                    .collect();
                edges.insert(
                    edge.id,
                    (edge.src, edge.dst, edge.edge_type.to_string(), values),
                );
            }
        }
        (nodes, edges)
    }

    /// Asserts `a` and `b` hold the same nodes, labels, edges and values,
    /// and count them alike.
    pub(crate) fn assert_same_rows(a: &dyn GraphStore, b: &dyn GraphStore, what: &str) {
        assert_eq!(rows(a), rows(b), "{what}");
        assert_eq!(
            (a.node_count(), a.edge_count()),
            (b.node_count(), b.edge_count()),
            "{what}: the counts"
        );
    }
}
