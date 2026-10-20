//! `LpgStore` as a [`LoadTarget`]: the recovery writes the readers of
//! earlier formats made before the row-group store took the same rows.

use std::sync::Arc;

use grafeo_common::change::Entity;
use grafeo_common::types::{EdgeId, EpochId, NodeId, Value};
use grafeo_common::utils::error::{Error, Result};

use super::LpgStore;
use crate::graph::load::{LoadTarget, LoadedNames};

impl LoadTarget for LpgStore {
    fn load_node(&self, id: NodeId, labels: &[&str]) -> Result<()> {
        Ok(self.create_node_with_id(id, labels)?)
    }

    fn load_edge(&self, id: EdgeId, src: NodeId, dst: NodeId, edge_type: &str) -> Result<bool> {
        self.create_edge_with_id(id, src, dst, edge_type)?;
        Ok(true)
    }

    fn load_value(
        &self,
        entity: Entity,
        key: &str,
        value: Value,
        epoch: Option<EpochId>,
    ) -> Result<()> {
        match (entity, epoch) {
            #[cfg(feature = "temporal")]
            (Entity::Node(id), Some(epoch)) => {
                self.set_node_property_at_epoch(id, key, value, epoch);
            }
            #[cfg(feature = "temporal")]
            (Entity::Edge(id), Some(epoch)) => {
                self.set_edge_property_at_epoch(id, key, value, epoch);
            }
            (Entity::Node(id), _) => self.set_node_property(id, key, value),
            (Entity::Edge(id), _) => self.set_edge_property(id, key, value),
        }
        Ok(())
    }

    #[cfg_attr(
        not(feature = "temporal"),
        expect(
            unused_variables,
            reason = "only `temporal` keeps the older versions of a value"
        )
    )]
    fn load_old_value(&self, entity: Entity, key: &str, value: Value, epoch: EpochId) {
        #[cfg(feature = "temporal")]
        match entity {
            Entity::Node(id) => self.set_node_property_at_epoch(id, key, value, epoch),
            Entity::Edge(id) => self.set_edge_property_at_epoch(id, key, value, epoch),
        }
    }

    fn load_remove_value(&self, entity: Entity, key: &str) -> Result<()> {
        match entity {
            Entity::Node(id) => self.remove_node_property(id, key)?,
            Entity::Edge(id) => self.remove_edge_property(id, key)?,
        };
        Ok(())
    }

    fn load_delete(&self, entity: Entity) -> Result<bool> {
        Ok(match entity {
            Entity::Node(id) => self.delete_node(id),
            Entity::Edge(id) => self.delete_edge(id),
        })
    }

    fn load_label(&self, id: NodeId, label: &str, add: bool) -> Result<bool> {
        Ok(if add {
            self.add_label(id, label)
        } else {
            self.remove_label(id, label)
        })
    }

    fn load_forget_label(&self, name: &str) -> bool {
        self.drop_unused_label(name)
    }

    fn load_names(&self, names: LoadedNames<'_>) -> std::result::Result<(), String> {
        let refuse = |what: &str, error: String| format!("{what}: {error}");
        for (id, name) in names.labels {
            self.restore_label(*id, name)
                .map_err(|error| refuse("label", error))?;
        }
        for (id, name) in names.edge_types {
            self.restore_edge_type(*id, name)
                .map_err(|error| refuse("edge type", error))?;
        }
        for (id, name) in names.keys {
            self.restore_property_key(*id, name)
                .map_err(|error| refuse("property key", error))?;
        }
        self.reserve_name_ids_below(names.next_ids);
        Ok(())
    }

    fn load_next_ids(&self, node: u64, edge: u64) {
        self.set_next_node_id(self.next_node_id().max(node));
        self.set_next_edge_id(self.next_edge_id().max(edge));
    }

    #[cfg_attr(
        not(feature = "temporal"),
        expect(
            unused_variables,
            reason = "only `temporal` continues at the epoch of the versions it loaded"
        )
    )]
    fn load_section_epoch(&self, epoch: EpochId) {
        #[cfg(feature = "temporal")]
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
        self.create_graph(name)?;
        self.graph(name)
            .ok_or_else(|| Error::Internal(format!("graph {name:?} was created, then not found")))
    }

    fn load_finish(&self) {}
}
