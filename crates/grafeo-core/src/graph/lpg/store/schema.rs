//! Schema, label, edge-type, and property-key methods for [`LpgStore`].

use super::{LpgStore, PropertyUndoEntry};
#[cfg(feature = "temporal")]
use grafeo_common::types::EpochId;
use grafeo_common::types::{NodeId, TransactionId};
use grafeo_common::utils::hash::FxHashMap;

impl LpgStore {
    /// Adds a label to a node.
    ///
    /// Returns true if the label was added, false if the node doesn't exist
    /// or already has the label.
    pub fn add_label(&self, node_id: NodeId, label: &str) -> bool {
        self.is_node_visible_at_epoch(node_id, self.current_epoch())
            && self.add_label_to_existing(node_id, label)
    }

    /// Removes a label from a node.
    ///
    /// Returns true if the label was removed, false if the node doesn't exist
    /// or doesn't have the label.
    pub fn remove_label(&self, node_id: NodeId, label: &str) -> bool {
        self.is_node_visible_at_epoch(node_id, self.current_epoch())
            && self.remove_label_from_existing(node_id, label)
    }

    /// Adds a label to a node the caller found to exist: a committed node for
    /// [`add_label`](Self::add_label) and, without `temporal`, a node the
    /// transaction sees (one it created included) for
    /// [`add_label_versioned`](Self::add_label_versioned). With `temporal`,
    /// the change is recorded at the current epoch (the versioned variant
    /// records its own at `EpochId::PENDING`).
    ///
    /// Returns false if the node already has the label.
    fn add_label_to_existing(&self, node_id: NodeId, label: &str) -> bool {
        let label_id = self.get_or_create_label_id(label);

        // Add to node_labels map
        let mut node_labels = self.node_labels.write();

        #[cfg(not(feature = "temporal"))]
        {
            let label_set = node_labels.entry(node_id).or_default();
            if label_set.contains(&label_id) {
                return false;
            }
            label_set.insert(label_id);
        }

        #[cfg(feature = "temporal")]
        {
            let current = node_labels
                .get(&node_id)
                .and_then(|log| log.latest())
                .cloned()
                .unwrap_or_default();
            if current.contains(&label_id) {
                return false;
            }
            let mut new_set = current;
            new_set.insert(label_id);
            self.append_labels(&mut node_labels, node_id, self.current_epoch(), new_set);
        }

        drop(node_labels);

        // Add to label_index
        let mut index = self.label_index.write();
        if (label_id as usize) >= index.len() {
            index.resize(label_id as usize + 1, FxHashMap::default());
        }
        index[label_id as usize].insert(node_id, ());
        drop(index);
        self.index_node_under_label(node_id, label);

        #[cfg(not(any(feature = "temporal", feature = "tiered-storage")))]
        self.update_label_count(node_id);

        true
    }

    /// Removes a label from a node the caller found to exist, like
    /// [`add_label_to_existing`](Self::add_label_to_existing).
    ///
    /// Returns false if the node doesn't have the label.
    fn remove_label_from_existing(&self, node_id: NodeId, label: &str) -> bool {
        // Get label ID
        let label_id = {
            let reg = self.label_registry.read();
            match reg.get_id(label) {
                Some(id) => id,
                None => return false, // Label doesn't exist
            }
        };

        // Remove from node_labels map
        let mut node_labels = self.node_labels.write();

        #[cfg(not(feature = "temporal"))]
        {
            if let Some(label_set) = node_labels.get_mut(&node_id) {
                if !label_set.remove(&label_id) {
                    return false;
                }
            } else {
                return false;
            }
        }

        #[cfg(feature = "temporal")]
        {
            let current = node_labels
                .get(&node_id)
                .and_then(|log| log.latest())
                .cloned()
                .unwrap_or_default();
            if !current.contains(&label_id) {
                return false;
            }
            let mut new_set = current;
            new_set.remove(&label_id);
            self.append_labels(&mut node_labels, node_id, self.current_epoch(), new_set);
        }

        drop(node_labels);

        // Remove from label_index
        let mut index = self.label_index.write();
        if (label_id as usize) < index.len() {
            index[label_id as usize].remove(&node_id);
        }
        drop(index);
        self.unindex_node_under_label(node_id, label);

        #[cfg(not(any(feature = "temporal", feature = "tiered-storage")))]
        self.update_label_count(node_id);

        true
    }

    /// Stores the node's label count in its newest record.
    #[cfg(not(any(feature = "temporal", feature = "tiered-storage")))]
    fn update_label_count(&self, node_id: NodeId) {
        if let Some(chain) = self.nodes.write().get_mut(&node_id)
            && let Some(record) = chain.latest_mut()
        {
            let count = self.node_labels.read().get(&node_id).map_or(0, |s| s.len());
            record.set_label_count(u16::try_from(count).unwrap_or(u16::MAX));
        }
    }

    /// Returns all nodes with a specific label.
    ///
    /// Uses the label index for O(1) lookup per label. Returns a snapshot -
    /// concurrent modifications won't affect the returned vector. Results are
    /// sorted by NodeId for deterministic iteration order.
    pub fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        let reg = self.label_registry.read();
        if let Some(label_id) = reg.get_id(label) {
            let index = self.label_index.read();
            if let Some(set) = index.get(label_id as usize) {
                let mut ids: Vec<NodeId> = set.keys().copied().collect();
                ids.sort_unstable();
                return ids;
            }
        }
        Vec::new()
    }

    /// Returns the number of nodes with a specific label without allocating
    /// the full ID list. O(1) via the label index.
    #[must_use]
    pub fn nodes_by_label_count(&self, label: &str) -> usize {
        let reg = self.label_registry.read();
        let Some(label_id) = reg.get_id(label) else {
            return 0;
        };
        self.label_index
            .read()
            .get(label_id as usize)
            .map_or(0, |set| set.len())
    }

    /// Returns the number of distinct labels in the store.
    #[must_use]
    pub fn label_count(&self) -> usize {
        self.label_registry.read().len()
    }

    /// Returns the number of distinct property keys in the store.
    ///
    /// This counts unique property keys across both nodes and edges.
    #[must_use]
    pub fn property_key_count(&self) -> usize {
        let node_keys = self.node_properties.column_count();
        let edge_keys = self.edge_properties.column_count();
        // Note: This may count some keys twice if the same key is used
        // for both nodes and edges. A more precise count would require
        // tracking unique keys across both storages.
        node_keys + edge_keys
    }

    /// Returns the number of distinct edge types in the store.
    #[must_use]
    pub fn edge_type_count(&self) -> usize {
        self.id_to_edge_type.read().len()
    }

    /// Returns all label names in the database.
    pub fn all_labels(&self) -> Vec<String> {
        self.label_registry
            .read()
            .names()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Returns all edge type names in the database.
    pub fn all_edge_types(&self) -> Vec<String> {
        self.id_to_edge_type
            .read()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Returns all property keys used in the database.
    pub fn all_property_keys(&self) -> Vec<String> {
        let mut keys = std::collections::HashSet::new();
        for key in self.node_properties.keys() {
            keys.insert(key.to_string());
        }
        for key in self.edge_properties.keys() {
            keys.insert(key.to_string());
        }
        keys.into_iter().collect()
    }

    /// Returns the next node ID that will be allocated.
    #[must_use]
    pub fn peek_next_node_id(&self) -> u64 {
        self.next_node_id.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Returns the next edge ID that will be allocated.
    #[must_use]
    pub fn peek_next_edge_id(&self) -> u64 {
        self.next_edge_id.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Adds a label to a node within a transaction, recording the change
    /// in the undo log so it can be reversed on rollback.
    ///
    /// Returns false if the transaction does not see the node (it sees the
    /// nodes it created itself) or the node already has the label.
    #[cfg(not(feature = "temporal"))]
    pub fn add_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        let added = self.is_node_visible_versioned(node_id, self.current_epoch(), transaction_id)
            && self.add_label_to_existing(node_id, label);
        if added {
            self.property_undo_log
                .write()
                .entry(transaction_id)
                .or_default()
                .push(PropertyUndoEntry::LabelAdded {
                    node_id,
                    label: label.to_string(),
                });
        }
        added
    }

    /// Adds a label to a node within a transaction (temporal version).
    ///
    /// Uses `EpochId::PENDING` for the version log entry, finalized on commit.
    /// Returns false if the transaction does not see the node (it sees the
    /// nodes it created itself) or the node already has the label.
    #[cfg(feature = "temporal")]
    pub fn add_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        if !self.is_node_visible_versioned(node_id, self.current_epoch(), transaction_id) {
            return false;
        }
        let label_id = self.get_or_create_label_id(label);

        let mut node_labels = self.node_labels.write();
        let current = node_labels
            .get(&node_id)
            .and_then(|log| log.latest())
            .cloned()
            .unwrap_or_default();
        if current.contains(&label_id) {
            return false;
        }
        let mut new_set = current;
        new_set.insert(label_id);
        self.append_labels(&mut node_labels, node_id, EpochId::PENDING, new_set);
        drop(node_labels);

        // Update label_index
        let mut index = self.label_index.write();
        if (label_id as usize) >= index.len() {
            index.resize(label_id as usize + 1, FxHashMap::default());
        }
        index[label_id as usize].insert(node_id, ());
        drop(index);
        self.index_node_under_label(node_id, label);

        // Record in undo log
        self.property_undo_log
            .write()
            .entry(transaction_id)
            .or_default()
            .push(PropertyUndoEntry::LabelAdded {
                node_id,
                label: label.to_string(),
            });

        true
    }

    /// Removes a label from a node within a transaction, recording the change
    /// in the undo log so it can be restored on rollback.
    ///
    /// Returns false if the transaction does not see the node (it sees the
    /// nodes it created itself) or the node doesn't have the label.
    #[cfg(not(feature = "temporal"))]
    pub fn remove_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        let removed = self.is_node_visible_versioned(node_id, self.current_epoch(), transaction_id)
            && self.remove_label_from_existing(node_id, label);
        if removed {
            self.property_undo_log
                .write()
                .entry(transaction_id)
                .or_default()
                .push(PropertyUndoEntry::LabelRemoved {
                    node_id,
                    label: label.to_string(),
                });
        }
        removed
    }

    /// Removes a label from a node within a transaction (temporal version).
    ///
    /// Returns false if the transaction does not see the node (it sees the
    /// nodes it created itself) or the node doesn't have the label.
    #[cfg(feature = "temporal")]
    pub fn remove_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        if !self.is_node_visible_versioned(node_id, self.current_epoch(), transaction_id) {
            return false;
        }
        let label_id = {
            let reg = self.label_registry.read();
            match reg.get_id(label) {
                Some(id) => id,
                None => return false,
            }
        };

        let mut node_labels = self.node_labels.write();
        let current = node_labels
            .get(&node_id)
            .and_then(|log| log.latest())
            .cloned()
            .unwrap_or_default();
        if !current.contains(&label_id) {
            return false;
        }
        let mut new_set = current;
        new_set.remove(&label_id);
        self.append_labels(&mut node_labels, node_id, EpochId::PENDING, new_set);
        drop(node_labels);

        // Update label_index
        let mut index = self.label_index.write();
        if (label_id as usize) < index.len() {
            index[label_id as usize].remove(&node_id);
        }
        drop(index);
        self.unindex_node_under_label(node_id, label);

        // Record in undo log
        self.property_undo_log
            .write()
            .entry(transaction_id)
            .or_default()
            .push(PropertyUndoEntry::LabelRemoved {
                node_id,
                label: label.to_string(),
            });

        true
    }
}
