//! Merge operator for MERGE clause execution.
//!
//! The MERGE operator implements the Cypher MERGE semantics:
//! 1. Try to match the pattern in the graph
//! 2. If found, return existing element (optionally apply ON MATCH SET)
//! 3. If not found, create the element (optionally apply ON CREATE SET)

use super::{Operator, OperatorResult};
use crate::execution::chunk::DataChunkBuilder;
use crate::graph::lpg::LpgStore;
use grafeo_common::types::{LogicalType, NodeId, PropertyKey, Value};
use std::sync::Arc;

/// Merge operator for MERGE clause.
///
/// Tries to match a node with the given labels and properties.
/// If found, returns the existing node. If not found, creates a new node.
pub struct MergeOperator {
    /// The graph store.
    store: Arc<LpgStore>,
    /// Variable name for the merged node.
    variable: String,
    /// Labels to match/create.
    labels: Vec<String>,
    /// Properties that must match (also used for creation).
    match_properties: Vec<(String, Value)>,
    /// Properties to set on CREATE.
    on_create_properties: Vec<(String, Value)>,
    /// Properties to set on MATCH.
    on_match_properties: Vec<(String, Value)>,
    /// Whether we've already executed.
    executed: bool,
}

impl MergeOperator {
    /// Creates a new merge operator.
    pub fn new(
        store: Arc<LpgStore>,
        variable: String,
        labels: Vec<String>,
        match_properties: Vec<(String, Value)>,
        on_create_properties: Vec<(String, Value)>,
        on_match_properties: Vec<(String, Value)>,
    ) -> Self {
        Self {
            store,
            variable,
            labels,
            match_properties,
            on_create_properties,
            on_match_properties,
            executed: false,
        }
    }

    /// Returns the variable name for the merged node.
    #[must_use]
    pub fn variable(&self) -> &str {
        &self.variable
    }

    /// Tries to find a matching node.
    fn find_matching_node(&self) -> Option<NodeId> {
        // Get all nodes with the first label (or all nodes if no labels)
        let candidates: Vec<NodeId> = if let Some(first_label) = self.labels.first() {
            self.store.nodes_by_label(first_label)
        } else {
            self.store.node_ids()
        };

        // Filter by all labels and properties
        for node_id in candidates {
            if let Some(node) = self.store.get_node(node_id) {
                // Check all labels
                let has_all_labels = self.labels.iter().all(|label| node.has_label(label));
                if !has_all_labels {
                    continue;
                }

                // Check all match properties
                let has_all_props = self.match_properties.iter().all(|(key, expected_value)| {
                    node.properties
                        .get(&PropertyKey::new(key.as_str()))
                        .is_some_and(|v| v == expected_value)
                });

                if has_all_props {
                    return Some(node_id);
                }
            }
        }

        None
    }

    /// Creates a new node with the specified labels and properties.
    fn create_node(&self) -> NodeId {
        // Combine match properties with on_create properties
        let mut all_props: Vec<(PropertyKey, Value)> = self
            .match_properties
            .iter()
            .map(|(k, v)| (PropertyKey::new(k.as_str()), v.clone()))
            .collect();

        // Add on_create properties (may override match properties)
        for (k, v) in &self.on_create_properties {
            // Check if property already exists, if so update it
            if let Some(existing) = all_props.iter_mut().find(|(key, _)| key.as_str() == k) {
                existing.1 = v.clone();
            } else {
                all_props.push((PropertyKey::new(k.as_str()), v.clone()));
            }
        }

        let labels: Vec<&str> = self.labels.iter().map(String::as_str).collect();
        self.store.create_node_with_props(&labels, all_props)
    }

    /// Applies ON MATCH properties to an existing node.
    fn apply_on_match(&self, node_id: NodeId) {
        for (key, value) in &self.on_match_properties {
            self.store
                .set_node_property(node_id, key.as_str(), value.clone());
        }
    }
}

impl Operator for MergeOperator {
    fn next(&mut self) -> OperatorResult {
        if self.executed {
            return Ok(None);
        }
        self.executed = true;

        // Try to find matching node
        let (node_id, was_created) = if let Some(existing_id) = self.find_matching_node() {
            // Node exists - apply ON MATCH properties
            self.apply_on_match(existing_id);
            (existing_id, false)
        } else {
            // Node doesn't exist - create it
            let new_id = self.create_node();
            (new_id, true)
        };

        // Build output chunk with the node ID
        let mut builder = DataChunkBuilder::new(&[LogicalType::Node]);
        builder.column_mut(0).unwrap().push_node_id(node_id);
        builder.advance_row();

        // Log for debugging (in real code, this would be removed)
        let _ = was_created; // Suppress unused variable warning

        Ok(Some(builder.finish()))
    }

    fn reset(&mut self) {
        self.executed = false;
    }

    fn name(&self) -> &'static str {
        "Merge"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_creates_new_node() {
        let store = Arc::new(LpgStore::new());

        // MERGE should create a new node since none exists
        let mut merge = MergeOperator::new(
            Arc::clone(&store),
            "n".to_string(),
            vec!["Person".to_string()],
            vec![("name".to_string(), Value::String("Alice".into()))],
            vec![], // no on_create
            vec![], // no on_match
        );

        let result = merge.next().unwrap();
        assert!(result.is_some());

        // Verify node was created
        let nodes = store.nodes_by_label("Person");
        assert_eq!(nodes.len(), 1);

        let node = store.get_node(nodes[0]).unwrap();
        assert!(node.has_label("Person"));
        assert_eq!(
            node.properties.get(&PropertyKey::new("name")),
            Some(&Value::String("Alice".into()))
        );
    }

    #[test]
    fn test_merge_matches_existing_node() {
        let store = Arc::new(LpgStore::new());

        // Create an existing node
        store.create_node_with_props(
            &["Person"],
            vec![(PropertyKey::new("name"), Value::String("Bob".into()))],
        );

        // MERGE should find the existing node
        let mut merge = MergeOperator::new(
            Arc::clone(&store),
            "n".to_string(),
            vec!["Person".to_string()],
            vec![("name".to_string(), Value::String("Bob".into()))],
            vec![], // no on_create
            vec![], // no on_match
        );

        let result = merge.next().unwrap();
        assert!(result.is_some());

        // Verify only one node exists (no new node created)
        let nodes = store.nodes_by_label("Person");
        assert_eq!(nodes.len(), 1);
    }

    #[test]
    fn test_merge_with_on_create() {
        let store = Arc::new(LpgStore::new());

        // MERGE with ON CREATE SET
        let mut merge = MergeOperator::new(
            Arc::clone(&store),
            "n".to_string(),
            vec!["Person".to_string()],
            vec![("name".to_string(), Value::String("Charlie".into()))],
            vec![("created".to_string(), Value::Bool(true))], // on_create
            vec![],                                           // no on_match
        );

        let _ = merge.next().unwrap();

        // Verify node has both match properties and on_create properties
        let nodes = store.nodes_by_label("Person");
        let node = store.get_node(nodes[0]).unwrap();
        assert_eq!(
            node.properties.get(&PropertyKey::new("name")),
            Some(&Value::String("Charlie".into()))
        );
        assert_eq!(
            node.properties.get(&PropertyKey::new("created")),
            Some(&Value::Bool(true))
        );
    }

    #[test]
    fn test_merge_with_on_match() {
        let store = Arc::new(LpgStore::new());

        // Create an existing node
        let node_id = store.create_node_with_props(
            &["Person"],
            vec![(PropertyKey::new("name"), Value::String("Diana".into()))],
        );

        // MERGE with ON MATCH SET
        let mut merge = MergeOperator::new(
            Arc::clone(&store),
            "n".to_string(),
            vec!["Person".to_string()],
            vec![("name".to_string(), Value::String("Diana".into()))],
            vec![],                                           // no on_create
            vec![("updated".to_string(), Value::Bool(true))], // on_match
        );

        let _ = merge.next().unwrap();

        // Verify node has the on_match property added
        let node = store.get_node(node_id).unwrap();
        assert_eq!(
            node.properties.get(&PropertyKey::new("updated")),
            Some(&Value::Bool(true))
        );
    }
}
