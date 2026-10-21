//! The row-group store as a [`LoadTarget`]: the readers of earlier formats,
//! the snapshots and the copies of graph data load their rows through it
//! (P5).
//!
//! Every row comes in through the change target's replay writer
//! ([`Writer::Replay`]) at the store's current epoch, the one function a
//! live write and a WAL replay apply through, so a loaded row is the row a
//! replay of its create would make. No reader runs during a load, so replay
//! keeps nothing for one. Replay is strict where a recovery of `LpgStore` is
//! lenient (a value of a node it does not hold, the removal of a value it
//! does not have): those are checked first and change nothing, as they do
//! in `LpgStore`. [`load_finish`](LoadTarget::load_finish) collapses the
//! version info of the rows the load stamped, so a migrated graph costs what
//! one read from a section of version 4 does.
//!
//! The dictionaries of a section of version 3 are restored id for id, as the
//! section of version 4 restores them; the store keeps no history, so the
//! older versions a file or a snapshot holds are dropped (time travel keeps
//! retained versions from H2c on).

use std::sync::Arc;
use std::sync::atomic::Ordering;

use grafeo_common::change::{DataOp, Entity};
use grafeo_common::types::{ArcStr, EdgeId, EpochId, NodeId, PropertyKey, Value};
use grafeo_common::utils::error::{Error, Result};

use super::RowGroupStore;
use crate::graph::apply::{ApplyError, ChangeTarget, Writer};
use crate::graph::load::{LoadTarget, LoadedNames};
use crate::graph::traits::GraphStore;

/// The crate's error for a load the store refused.
fn refused(error: &ApplyError) -> Error {
    Error::Internal(format!("the row-group store refused a loaded row: {error}"))
}

impl RowGroupStore {
    /// Applies `op` through the replay writer at the store's epoch.
    fn replay(&self, op: &DataOp) -> Result<()> {
        self.apply(
            op,
            Writer::Replay {
                epoch: self.current_epoch(),
            },
        )
        .map(|_| ())
        .map_err(|error| refused(&error))
    }

    /// Removes every row of this store, as `LpgStore::clear` does for a
    /// snapshot restore: the rows, their counts and the indexes go, the next
    /// node and edge ids start over, and the dictionaries keep every id they
    /// gave out (a name used again gets its old id), and the store keeps
    /// backward edges as it did. The epoch stays: it
    /// never goes back, so a reader's epoch keeps its meaning. The named
    /// graphs stay too (a restore drops them first). No transaction may be
    /// open in the store.
    pub fn clear(&self) {
        let mut inner = self.inner.write();
        let labels = std::mem::take(&mut inner.labels);
        let edge_types = std::mem::take(&mut inner.edge_types);
        let keys = std::mem::take(&mut inner.keys);
        let incoming_edges = inner.incoming_edges;
        *inner = super::Inner {
            labels,
            edge_types,
            keys,
            incoming_edges,
            ..super::Inner::default()
        };
        self.next_node_id.store(0, Ordering::Release);
        self.next_edge_id.store(0, Ordering::Release);
    }

    /// Fills values of `key` from a spill file an older build left behind
    /// (#594), as `LpgStore::fill_missing_node_values` does: a load step, so
    /// each value goes in through the replay writer at the store's epoch. A
    /// value fills only a node that exists, carries `label` and has no
    /// value for `key` (a value written later wins), so a second run changes
    /// nothing; the indexes take the value as a write's. Returns how many
    /// values were filled.
    ///
    /// # Errors
    ///
    /// The error of a value the store refuses; the values filled before it
    /// stay. (The store holds no value it cannot read, unlike a spilled
    /// column of `LpgStore`.)
    pub fn fill_missing_node_values(
        &self,
        label: &str,
        key: &PropertyKey,
        values: impl IntoIterator<Item = (NodeId, Value)>,
    ) -> Result<usize> {
        let mut filled = 0;
        for (id, value) in values {
            let fills = {
                let now = self.now();
                let inner = self.inner.read();
                let Some(label) = inner.label_id(label) else {
                    return Ok(filled);
                };
                inner.node(id.as_u64()).is_some_and(|(group, row)| {
                    group.version(row).visible(now)
                        && group.has_label_at(row, label, now)
                        && inner
                            .keys
                            .get_id(key.as_str())
                            .and_then(|key| group.columns.get(key, row))
                            .is_none()
                })
            };
            if fills {
                self.replay(&DataOp::SetNodeProperty {
                    id,
                    key: key.clone(),
                    value,
                })?;
                filled += 1;
            }
        }
        Ok(filled)
    }

    /// Whether the store holds `entity` now.
    fn holds(&self, entity: Entity) -> bool {
        let now = self.now();
        let inner = self.inner.read();
        match entity {
            Entity::Node(id) => inner.node_visible(id.as_u64(), now),
            Entity::Edge(id) => inner.edge_visible(id.as_u64(), now),
        }
    }
}

impl LoadTarget for RowGroupStore {
    fn load_node(&self, id: NodeId, labels: &[&str]) -> Result<()> {
        self.replay(&DataOp::CreateNode {
            id,
            labels: labels.iter().map(|label| ArcStr::from(*label)).collect(),
            properties: Vec::new(),
        })
    }

    /// An edge whose end the store does not hold is left out (`false`):
    /// the store keeps an edge in the adjacency of its ends. Only a damaged
    /// file or log holds one (a 0.5.43 WAL directory replays the changes
    /// after its first switch to a named graph into that graph, edges
    /// between nodes of the default graph included), and `LpgStore`, which
    /// keeps it, shows it to no query either.
    fn load_edge(&self, id: EdgeId, src: NodeId, dst: NodeId, edge_type: &str) -> Result<bool> {
        if [src, dst]
            .into_iter()
            .any(|end| !self.holds(Entity::Node(end)))
        {
            return Ok(false);
        }
        self.replay(&DataOp::CreateEdge {
            id,
            src,
            dst,
            edge_type: ArcStr::from(edge_type),
            properties: Vec::new(),
        })?;
        Ok(true)
    }

    fn load_value(
        &self,
        entity: Entity,
        key: &str,
        value: Value,
        _epoch: Option<EpochId>,
    ) -> Result<()> {
        if !self.holds(entity) {
            return Ok(());
        }
        let key = PropertyKey::new(key);
        self.replay(&match entity {
            Entity::Node(id) => DataOp::SetNodeProperty { id, key, value },
            Entity::Edge(id) => DataOp::SetEdgeProperty { id, key, value },
        })
    }

    fn load_old_value(&self, _entity: Entity, _key: &str, _value: Value, _epoch: EpochId) {}

    fn load_remove_value(&self, entity: Entity, key: &str) -> Result<()> {
        let property = PropertyKey::new(key);
        let has = match entity {
            Entity::Node(id) => self.get_node_property(id, &property).is_some(),
            Entity::Edge(id) => self.get_edge_property(id, &property).is_some(),
        };
        if !has {
            return Ok(());
        }
        self.replay(&match entity {
            Entity::Node(id) => DataOp::RemoveNodeProperty { id, key: property },
            Entity::Edge(id) => DataOp::RemoveEdgeProperty { id, key: property },
        })
    }

    fn load_delete(&self, entity: Entity) -> Result<bool> {
        if !self.holds(entity) {
            return Ok(false);
        }
        self.replay(&match entity {
            Entity::Node(id) => DataOp::DeleteNode { id },
            Entity::Edge(id) => DataOp::DeleteEdge { id },
        })?;
        Ok(true)
    }

    fn load_label(&self, id: NodeId, label: &str, add: bool) -> Result<bool> {
        let has = {
            let now = self.now();
            let inner = self.inner.read();
            let Some((group, row)) = inner.node(id.as_u64()) else {
                return Ok(false);
            };
            if !group.version(row).visible(now) {
                return Ok(false);
            }
            inner
                .label_id(label)
                .is_some_and(|label| group.has_label_at(row, label, now))
        };
        if has == add {
            return Ok(false);
        }
        let label = ArcStr::from(label);
        self.replay(&if add {
            DataOp::AddNodeLabel { id, label }
        } else {
            DataOp::RemoveNodeLabel { id, label }
        })?;
        Ok(true)
    }

    fn load_forget_label(&self, name: &str) -> bool {
        let mut inner = self.inner.write();
        let Some(id) = inner.label_id(name) else {
            return false;
        };
        let used = inner.nodes.values().any(|group| {
            group.labels.get(&id).is_some_and(|rows| !rows.is_empty())
                // Kept label sets of older readers may name it: none
                // after a load, which keeps nothing.
                || !group.label_updates.is_empty()
        });
        if used {
            return false;
        }
        for group in inner.nodes.values_mut() {
            group.labels.remove(&id);
        }
        inner.labels.remove(name).is_some()
    }

    fn load_names(&self, names: LoadedNames<'_>) -> std::result::Result<(), String> {
        let mut inner = self.inner.write();
        let super::Inner {
            labels,
            edge_types,
            keys,
            ..
        } = &mut *inner;
        let [next_label, next_edge_type, next_key] = names.next_ids;
        for (dictionary, listed, next, what) in [
            (labels, names.labels, next_label, "label"),
            (edge_types, names.edge_types, next_edge_type, "edge type"),
            (keys, names.keys, next_key, "property key"),
        ] {
            for (id, name) in listed {
                dictionary
                    .insert_at(*id, name)
                    .map_err(|error| format!("{what}: {error}"))?;
            }
            dictionary.reserve_below(next);
        }
        Ok(())
    }

    fn load_next_ids(&self, node: u64, edge: u64) {
        self.next_node_id.fetch_max(node, Ordering::AcqRel);
        self.next_edge_id.fetch_max(edge, Ordering::AcqRel);
    }

    fn load_section_epoch(&self, epoch: EpochId) {
        self.sync_epoch(epoch);
    }

    fn load_graphs(
        &self,
        graphs: &[(u32, &str)],
        next_graph_id: u32,
    ) -> std::result::Result<Vec<Arc<Self>>, String> {
        self.restore_graphs(graphs, next_graph_id)
    }

    fn load_named_graph(&self, name: &str) -> Result<Arc<Self>> {
        Ok(self.graph_or_create(name)?)
    }

    fn load_finish(&self) {
        self.collect(self.current_epoch());
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use grafeo_common::change::Entity;
    use grafeo_common::storage::{ImageSource, MemoryImage, Section, SectionSink, SectionType};
    use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, Value};

    use super::super::RowGroupStore;
    use super::super::section::RowGroupStoreSection;
    use crate::graph::load::testing::{assert_same_rows, fixture_bytes, version_3_fixture};
    use crate::graph::load::{LoadTarget, LoadedNames};
    use crate::graph::traits::{GraphStore, GraphStoreMut};

    /// What the `beta` fixtures hold (written by `LpgStore` with the
    /// writes below, see `tests/fixtures/lpg-v3/README.md`): nodes of 0, 1
    /// and 3 labels, values of several kinds, two edge types, a deleted node
    /// (a gap in the ids), a named graph, and Alix's age changed after it
    /// was set. Returns the store and Alix.
    fn beta_store() -> (RowGroupStore, NodeId) {
        let store = RowGroupStore::new();
        let alix = store.create_node(&[]);
        let gus = store.create_node(&["Person"]);
        let vincent = store.create_node(&["Person"]);
        let mia = store.create_node(&["Person", "Employee", "Traveller"]);
        store.set_node_property(alix, "name", Value::from("Alix"));
        store.set_node_property(alix, "age", Value::Int64(19));
        store.set_node_property(alix, "age", Value::Int64(88));
        store.set_node_property(gus, "score", Value::Float64(-0.5));
        store.set_node_property(mia, "active", Value::Bool(true));
        store.set_node_property(
            mia,
            "stops",
            Value::List(Arc::from(vec![Value::from("Paris"), Value::Int64(3)])),
        );
        store.set_node_property(
            mia,
            "embedding",
            Value::Vector(Arc::from(vec![3.0f32, 19.0])),
        );
        let knows = store.create_edge(alix, gus, "KNOWS");
        store.set_edge_property(knows, "since", Value::Int64(2019));
        store.create_edge(gus, mia, "VISITED");
        assert!(store.delete_node(vincent));
        store.create_graph("trips").unwrap();
        let trips = store.graph("trips").unwrap();
        let berlin = trips.create_node(&["City"]);
        trips.set_node_property(berlin, "name", Value::from("Berlin"));
        let paris = trips.create_node(&["City"]);
        trips.create_edge(berlin, paris, "TRAIN");
        (store, alix)
    }

    /// The section `section` writes, in an image.
    fn image_of(section: &dyn Section) -> MemoryImage {
        let mut image = MemoryImage::new();
        image
            .begin_section(section.section_type(), section.version())
            .unwrap();
        section.write_to(&mut image).unwrap();
        image
    }

    /// Reads the LPG section of `image` into a new root.
    fn migrate(image: &MemoryImage) -> Arc<RowGroupStore> {
        let root = Arc::new(RowGroupStore::new());
        RowGroupStoreSection::new(Arc::clone(&root))
            .read_from(&*image.section_source(SectionType::LpgStore).unwrap())
            .unwrap();
        root
    }

    /// Asserts `root` holds what `beta` holds, every graph with its id, and
    /// costs what a loaded root does.
    fn assert_migrated(beta: &RowGroupStore, alix: NodeId, root: &RowGroupStore) {
        assert_same_rows(beta, root, "the default graph");
        let trips = root.graph("trips").expect("the named graph");
        assert_same_rows(&*beta.graph("trips").unwrap(), &*trips, "graph trips");
        assert_eq!(trips.graph_id(), beta.graph("trips").unwrap().graph_id());
        assert_eq!(root.next_graph_id(), beta.next_graph_id());
        assert_eq!(
            root.get_node_property(alix, &PropertyKey::new("age")),
            Some(Value::Int64(88)),
            "Alix's current age, not the older version"
        );
        assert_eq!(root.stamped_vectors() + trips.stamped_vectors(), 0);
        assert_eq!(root.update_nodes(), 0);
    }

    /// A section of version 3 (0.6.0-beta.1) migrates into a root: the rows,
    /// the graphs with their ids, the next ids (Vincent's id is never given
    /// again) and the dictionaries' ids, which a checkpoint of the root
    /// writes and reads back.
    #[test]
    fn a_section_of_version_3_migrates_into_a_root() {
        let (beta, alix) = beta_store();
        let image = version_3_fixture("beta_caps_2_64.chunks");
        let root = migrate(&image);
        assert_migrated(&beta, alix, &root);
        assert_eq!(root.next_node_id(), beta.next_node_id());
        assert_eq!(root.next_edge_id(), beta.next_edge_id());
        {
            let inner = root.inner.read();
            assert_eq!(inner.labels.get_id("Person"), Some(0));
            assert_eq!(inner.labels.get_id("Traveller"), Some(2));
            assert_eq!(inner.edge_types.get_id("VISITED"), Some(1));
        }

        // The checkpoint of the migrated root (version 4) holds the same.
        let checkpoint = image_of(&RowGroupStoreSection::new(Arc::clone(&root)));
        let reopened = migrate(&checkpoint);
        assert_migrated(&beta, alix, &reopened);
        assert_eq!(reopened.next_node_id(), beta.next_node_id());
        assert_eq!(
            reopened.inner.read().labels.get_id("Traveller"),
            Some(2),
            "the ids stay, checkpoint after checkpoint"
        );
    }

    /// An edge whose end the graph does not hold, which `LpgStore` kept (a
    /// replay of a 0.5.43 WAL directory made some), is left out of the
    /// migrated root, and the rest loads.
    #[test]
    fn an_edge_without_its_end_is_left_out() {
        let (beta, alix) = beta_store();
        let dangling = EdgeId::new(beta.next_edge_id());
        let root = migrate(&version_3_fixture("beta_dangling_edge.chunks"));
        assert_migrated(&beta, alix, &root);
        assert!(root.get_edge(dangling).is_none());
        assert_eq!(root.edge_count(), beta.edge_count());
        assert_eq!(
            root.edges_from(alix, crate::graph::Direction::Outgoing)
                .len(),
            1,
            "Alix knows Gus"
        );
        assert!(
            root.next_edge_id() > dangling.as_u64(),
            "its id is not given again"
        );
    }

    /// The LPG section of 0.5.x (the block layout, one raw chunk) migrates
    /// into a root.
    #[test]
    fn a_0_5_section_migrates_into_a_root() {
        let (beta, alix) = beta_store();
        let image =
            MemoryImage::from_raw(vec![(SectionType::LpgStore, fixture_bytes("beta_0_5.bin"))])
                .unwrap();
        let root = migrate(&image);
        assert_migrated(&beta, alix, &root);
        let highest = beta.node_ids().into_iter().max().unwrap();
        assert!(
            root.next_node_id() > highest.as_u64(),
            "no loaded id is given again"
        );
    }

    /// A section of version 3 its reader refuses (here, one whose metadata
    /// chunk is lost) is refused, naming the LPG section.
    #[test]
    fn a_damaged_section_of_version_3_is_refused() {
        let image = version_3_fixture("beta_caps_2_64.chunks");
        let source = image.section_source(SectionType::LpgStore).unwrap();
        let (_, data) = source.chunks().split_last().unwrap();
        let mut damaged = MemoryImage::new();
        damaged.begin_section(SectionType::LpgStore, 3).unwrap();
        for (index, chunk) in data.iter().enumerate() {
            damaged
                .write_chunk(*chunk, &source.fetch(index).unwrap())
                .unwrap();
        }
        let error = RowGroupStoreSection::new(Arc::new(RowGroupStore::new()))
            .read_from(&*damaged.section_source(SectionType::LpgStore).unwrap())
            .unwrap_err();
        assert!(error.to_string().contains("LPG section"), "{error}");
    }

    /// Rows loaded at the store's epoch are seen by every reader from it,
    /// and cost one bit per row once the load is finished.
    #[test]
    fn loaded_rows_are_committed_and_collapsed() {
        let store = RowGroupStore::new();
        store.sync_epoch(EpochId::new(19));
        store.load_node(NodeId::new(3), &["Person"]).unwrap();
        store.load_node(NodeId::new(88), &["City"]).unwrap();
        store
            .load_value(
                Entity::Node(NodeId::new(3)),
                "name",
                Value::from("Alix"),
                None,
            )
            .unwrap();
        store
            .load_edge(
                grafeo_common::types::EdgeId::new(7),
                NodeId::new(3),
                NodeId::new(88),
                "LIVES_IN",
            )
            .unwrap();
        assert!(store.stamped_vectors() > 0, "replay stamps what it loads");
        store.load_finish();
        assert_eq!(store.stamped_vectors(), 0);
        assert_eq!(store.node_count(), 2);
        assert_eq!(store.edge_count(), 1);
        assert_eq!(
            store.get_node_property(NodeId::new(3), &PropertyKey::new("name")),
            Some(Value::from("Alix"))
        );
        assert_eq!(store.current_epoch(), EpochId::new(19));
        // The next ids are above the loaded ones.
        assert_eq!(store.next_node_id(), 89);
        assert_eq!(store.next_edge_id(), 8);
    }

    /// A value of a node the store does not hold, and the removal of one a
    /// node lacks, change nothing, as a recovery of `LpgStore` does.
    #[test]
    fn writes_of_what_the_store_lacks_change_nothing() {
        let store = RowGroupStore::new();
        let gus = NodeId::new(1);
        store
            .load_value(Entity::Node(gus), "age", Value::Int64(26), None)
            .unwrap();
        assert_eq!(store.node_count(), 0);
        store.load_node(gus, &[]).unwrap();
        store.load_remove_value(Entity::Node(gus), "age").unwrap();
        assert!(!store.load_delete(Entity::Node(NodeId::new(2))).unwrap());
        assert!(store.load_delete(Entity::Node(gus)).unwrap());
        assert_eq!(store.node_count(), 0);
    }

    /// Labels are added and taken away once; a name no node has is
    /// forgotten, one a node has is kept.
    #[test]
    fn labels_load_and_unused_names_are_forgotten() {
        let store = RowGroupStore::new();
        let mia = NodeId::new(0);
        store.load_node(mia, &["Employee|Person"]).unwrap();
        assert!(!store.load_forget_label("Employee|Person"), "Mia has it");
        assert!(store.load_label(mia, "Employee|Person", false).unwrap());
        assert!(!store.load_label(mia, "Employee|Person", false).unwrap());
        assert!(store.load_label(mia, "Person", true).unwrap());
        assert!(!store.load_label(mia, "Person", true).unwrap());
        assert!(store.load_forget_label("Employee|Person"));
        assert_eq!(store.all_labels(), vec!["Person".to_string()]);
        assert_eq!(store.nodes_by_label("Person"), vec![mia]);
    }

    /// A clear takes every row and index out, keeps the names' ids and the
    /// epoch, and starts the next ids over.
    #[test]
    fn a_clear_keeps_the_names_and_the_epoch() {
        let store = RowGroupStore::new();
        store.sync_epoch(EpochId::new(19));
        store
            .load_node(NodeId::new(3), &["Person", "City"])
            .unwrap();
        store
            .load_value(
                Entity::Node(NodeId::new(3)),
                "name",
                Value::from("Jules"),
                None,
            )
            .unwrap();
        store.create_property_index("name");
        store.clear();
        assert_eq!((store.node_count(), store.next_node_id()), (0, 0));
        assert_eq!(store.property_index_keys(), Vec::<String>::new());
        assert_eq!(store.current_epoch(), EpochId::new(19));
        store.load_node(NodeId::new(0), &["City"]).unwrap();
        assert_eq!(
            store.inner.read().labels.get_id("City"),
            Some(1),
            "a name keeps its id"
        );
        assert_eq!(store.nodes_by_label("Person"), Vec::<NodeId>::new());
    }

    /// Values an older build spilled fill only the nodes of the label that
    /// lack one, and a second fill changes nothing.
    #[test]
    fn spilled_values_fill_only_what_lacks_one() {
        let store = RowGroupStore::new();
        let (alix, gus, mia) = (NodeId::new(0), NodeId::new(1), NodeId::new(2));
        store.load_node(alix, &["Item"]).unwrap();
        store.load_node(gus, &["Item"]).unwrap();
        store.load_node(mia, &["Person"]).unwrap();
        let key = PropertyKey::new("embedding");
        let later = Value::Vector(Arc::from(vec![1.0f32, 2.0]));
        store
            .load_value(Entity::Node(gus), "embedding", later.clone(), None)
            .unwrap();
        let spilled = |id| (id, Value::Vector(Arc::from(vec![3.0f32, 19.0])));
        let values = [
            spilled(alix),
            spilled(gus),
            spilled(mia),
            spilled(NodeId::new(88)),
        ];
        assert_eq!(
            store
                .fill_missing_node_values("Item", &key, values.clone())
                .unwrap(),
            1
        );
        assert_eq!(store.get_node_property(alix, &key), Some(spilled(alix).1));
        assert_eq!(
            store.get_node_property(gus, &key),
            Some(later),
            "written later"
        );
        assert_eq!(store.get_node_property(mia, &key), None, "not an Item");
        assert_eq!(
            store
                .fill_missing_node_values("Item", &key, values)
                .unwrap(),
            0
        );
    }

    /// A file's names keep their ids, gaps included, and the next ids.
    #[test]
    fn names_load_id_for_id() {
        let store = RowGroupStore::new();
        let labels = [(0, "Person".to_string()), (3, "City".to_string())];
        let keys = [(1, "name".to_string())];
        store
            .load_names(LoadedNames {
                labels: &labels,
                edge_types: &[],
                keys: &keys,
                next_ids: [5, 2, 4],
            })
            .unwrap();
        let inner = store.inner.read();
        assert_eq!(inner.labels.get_id("City"), Some(3));
        assert_eq!(inner.labels.next_id(), 5);
        assert_eq!(inner.edge_types.next_id(), 2);
        assert_eq!(inner.keys.get_id("name"), Some(1));
        drop(inner);
        let error = store
            .load_names(LoadedNames {
                labels: &[(3, "Town".to_string())],
                edge_types: &[],
                keys: &[],
                next_ids: [0; 3],
            })
            .unwrap_err();
        assert!(error.starts_with("label:"), "{error}");
    }
}
