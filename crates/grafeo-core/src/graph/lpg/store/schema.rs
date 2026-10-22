//! Schema, label, edge-type, and property-key methods for [`LpgStore`].

use super::super::dictionary::NameDictionary;
use super::LpgStore;
use grafeo_common::types::{EpochId, NodeId, PropertyKey, TransactionId};
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
        self.add_label_at(node_id, label, self.current_epoch())
    }

    /// Adds a label to a node the caller found to exist, with `temporal` as
    /// a new label set at `at` (`EpochId::PENDING` for a transaction's
    /// write); without it the set changes in place and `at` is not used.
    ///
    /// Returns false if the node already has the label.
    pub(super) fn add_label_at(&self, node_id: NodeId, label: &str, at: EpochId) -> bool {
        #[cfg(not(feature = "temporal"))]
        let _ = at;
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
            self.append_labels(&mut node_labels, node_id, at, new_set);
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
        self.remove_label_at(node_id, label, self.current_epoch())
    }

    /// Removes a label from a node the caller found to exist, as
    /// [`add_label_at`](Self::add_label_at) adds one.
    ///
    /// Returns false if the node doesn't have the label.
    pub(super) fn remove_label_at(&self, node_id: NodeId, label: &str, at: EpochId) -> bool {
        #[cfg(not(feature = "temporal"))]
        let _ = at;
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
            self.append_labels(&mut node_labels, node_id, at, new_set);
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
    pub(super) fn update_label_count(&self, node_id: NodeId) {
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
        self.edge_types.read().len()
    }

    /// Returns all label names in the database.
    pub fn all_labels(&self) -> Vec<String> {
        self.label_registry
            .read()
            .iter()
            .map(|(_, name)| name.to_string())
            .collect()
    }

    /// Returns all edge type names in the database.
    pub fn all_edge_types(&self) -> Vec<String> {
        self.edge_types
            .read()
            .iter()
            .map(|(_, name)| name.to_string())
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

    /// The id of `label`, `None` when no node of this graph ever had it.
    pub(crate) fn label_id(&self, label: &str) -> Option<u32> {
        self.label_registry.read().get_id(label)
    }

    /// The id of `edge_type`, `None` when this graph never had it.
    pub(crate) fn edge_type_id(&self, edge_type: &str) -> Option<u32> {
        self.edge_types.read().get_id(edge_type)
    }

    /// The id of property key `key`, given the next id when it has none: a
    /// checkpoint gives a key its id when it first writes the key's column.
    pub(crate) fn property_key_id(&self, key: &str) -> u32 {
        if let Some(id) = self.property_keys.read().get_id(key) {
            return id;
        }
        self.property_keys.write().get_or_create(key)
    }

    /// This graph's label, edge type and property key dictionaries, as they
    /// are now.
    pub(crate) fn name_dictionaries(&self) -> [NameDictionary; 3] {
        [
            self.label_registry.read().clone(),
            self.edge_types.read().clone(),
            self.property_keys.read().clone(),
        ]
    }

    /// Gives this graph `source`'s dictionaries, ids included, as a copy of
    /// a store does before it creates any node or edge.
    pub(crate) fn copy_name_dictionaries(&self, source: &LpgStore) {
        let [labels, edge_types, keys] = source.name_dictionaries();
        let types = edge_types.next_id() as usize;
        *self.label_registry.write() = labels;
        *self.edge_types.write() = edge_types;
        *self.property_keys.write() = keys;
        let mut counts = self.edge_type_live_counts.write();
        if counts.len() < types {
            counts.resize(types, 0);
        }
    }

    /// Restores label `name` with the id `id`, as a load does before any
    /// node has it.
    ///
    /// # Errors
    ///
    /// Returns what is wrong when `id` or `name` is taken.
    pub(crate) fn restore_label(&self, id: u32, name: &str) -> Result<(), String> {
        self.label_registry.write().insert_at(id, name)
    }

    /// Drops label `name` from the label dictionary when no node has it, as
    /// the fold of a 0.5.x compacted base does for a joined name it split
    /// (see `compact::fold`): its id becomes a gap that is never given out
    /// again, and the next checkpoint writes the dictionary without it.
    /// Every other name keeps its id for good, also one whose nodes all
    /// lost it.
    ///
    /// Returns whether `name` was dropped: false when it has no id, or when
    /// the label index holds a node with it (a node of an open transaction
    /// included).
    pub(crate) fn drop_unused_label(&self, name: &str) -> bool {
        let mut registry = self.label_registry.write();
        let Some(id) = registry.get_id(name) else {
            return false;
        };
        if self
            .label_index
            .read()
            .get(id as usize)
            .is_some_and(|nodes| !nodes.is_empty())
        {
            return false;
        }
        registry.remove(name).is_some()
    }

    /// Restores edge type `name` with the id `id`, as a load does before any
    /// edge has it.
    ///
    /// # Errors
    ///
    /// Returns what is wrong when `id` or `name` is taken.
    pub(crate) fn restore_edge_type(&self, id: u32, name: &str) -> Result<(), String> {
        self.edge_types.write().insert_at(id, name)?;
        let mut counts = self.edge_type_live_counts.write();
        if counts.len() <= id as usize {
            counts.resize(id as usize + 1, 0);
        }
        Ok(())
    }

    /// Restores property key `name` with the id `id`, as a load does.
    ///
    /// # Errors
    ///
    /// Returns what is wrong when `id` or `name` is taken.
    pub(crate) fn restore_property_key(&self, id: u32, name: &str) -> Result<(), String> {
        self.property_keys.write().insert_at(id, name)
    }

    /// Makes the next ids of the label, edge type and property key
    /// dictionaries at least `next`, as a load restores them: the ids below
    /// are never given out again.
    pub(crate) fn reserve_name_ids_below(&self, [labels, edge_types, keys]: [u32; 3]) {
        self.label_registry.write().reserve_below(labels);
        self.edge_types.write().reserve_below(edge_types);
        self.property_keys.write().reserve_below(keys);
    }

    /// Returns the keys of the node property columns, in key order. A column
    /// is listed once it was created, also when no node has a value for it
    /// any more.
    #[must_use]
    pub fn node_property_keys(&self) -> Vec<PropertyKey> {
        let mut keys = self.node_properties.keys();
        keys.sort_unstable();
        keys
    }

    /// Returns the keys of the edge property columns, in key order, as
    /// [`node_property_keys`](Self::node_property_keys) does for nodes.
    #[must_use]
    pub fn edge_property_keys(&self) -> Vec<PropertyKey> {
        let mut keys = self.edge_properties.keys();
        keys.sort_unstable();
        keys
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

    /// Adds a label to a node as `transaction_id`. Nothing records the
    /// change (a transaction the engine runs writes through the store's
    /// change target, see `ChangeTarget::apply`).
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
        self.is_node_visible_versioned(node_id, self.current_epoch(), transaction_id)
            && self.add_label_to_existing(node_id, label)
    }

    /// Adds a label to a node as `transaction_id` (temporal version): a
    /// PENDING label set, which nothing records.
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
        true
    }

    /// Removes a label from a node as `transaction_id`. Nothing records the
    /// change.
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
        self.is_node_visible_versioned(node_id, self.current_epoch(), transaction_id)
            && self.remove_label_from_existing(node_id, label)
    }

    /// Removes a label from a node as `transaction_id` (temporal version): a
    /// PENDING label set, which nothing records.
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
        true
    }
}
