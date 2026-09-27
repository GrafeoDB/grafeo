//! Graph projections: read-only, filtered views of a graph store.
//!
//! A [`GraphProjection`] wraps an existing [`GraphStore`] and presents a
//! subgraph defined by a [`ProjectionSpec`]. Only nodes with matching labels
//! and edges with matching types (whose endpoints are both in the projection)
//! are visible. Everything else is filtered out transparently.
//!
//! Projections are read-only: they implement [`GraphStore`] but not
//! [`super::GraphStoreMut`].
//!
//! # Example
//!
//! ```ignore
//! let spec = ProjectionSpec::new()
//!     .with_node_labels(["Person", "City"])
//!     .with_edge_types(["LIVES_IN"]);
//! let projected = GraphProjection::new(store, spec);
//! // Only Person/City nodes and LIVES_IN edges are visible
//! ```

use std::collections::HashSet;
use std::sync::Arc;

use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::hash::FxHashMap;

use super::Direction;
use super::lpg::{CompareOp, Edge, Node};
use super::traits::{GraphStore, GraphStoreSearch};
use crate::statistics::Statistics;

/// Defines which nodes and edges are included in a projection.
#[derive(Debug, Clone, Default)]
pub struct ProjectionSpec {
    /// Node labels to include. Empty means all nodes.
    node_labels: HashSet<String>,
    /// Edge types to include. Empty means all edges.
    edge_types: HashSet<String>,
}

impl ProjectionSpec {
    /// Creates an empty spec (all nodes, all edges).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Restricts the projection to nodes with any of these labels.
    #[must_use]
    pub fn with_node_labels(mut self, labels: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.node_labels = labels.into_iter().map(Into::into).collect();
        self
    }

    /// Restricts the projection to edges with any of these types.
    #[must_use]
    pub fn with_edge_types(mut self, types: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.edge_types = types.into_iter().map(Into::into).collect();
        self
    }

    /// Returns true if node labels are filtered.
    fn filters_labels(&self) -> bool {
        !self.node_labels.is_empty()
    }

    /// Returns true if edge types are filtered.
    fn filters_edge_types(&self) -> bool {
        !self.edge_types.is_empty()
    }
}

/// A read-only, filtered view of a graph store.
///
/// Delegates all reads to the inner store, filtering results by the
/// [`ProjectionSpec`]. Nodes without matching labels and edges without
/// matching types are invisible.
pub struct GraphProjection {
    inner: Arc<dyn GraphStoreSearch>,
    spec: ProjectionSpec,
}

impl GraphProjection {
    /// Creates a new projection over the given store.
    pub fn new(inner: Arc<dyn GraphStoreSearch>, spec: ProjectionSpec) -> Self {
        Self { inner, spec }
    }

    /// Returns true if a node passes the label filter.
    fn node_matches(&self, node: &Node) -> bool {
        if !self.spec.filters_labels() {
            return true;
        }
        node.labels
            .iter()
            .any(|l| self.spec.node_labels.contains(l.as_str()))
    }

    /// Returns true if a node ID passes the label filter.
    fn node_id_matches(&self, id: NodeId) -> bool {
        if !self.spec.filters_labels() {
            return true;
        }
        self.inner
            .get_node(id)
            .is_some_and(|n| self.node_matches(&n))
    }

    /// Returns true if an edge type passes the type filter.
    fn edge_type_matches(&self, edge_type: &str) -> bool {
        if !self.spec.filters_edge_types() {
            return true;
        }
        self.spec.edge_types.contains(edge_type)
    }

    /// Returns true if an edge passes both endpoint and type filters.
    fn edge_matches(&self, edge: &Edge) -> bool {
        if !self.edge_type_matches(&edge.edge_type) {
            return false;
        }
        self.node_id_matches(edge.src) && self.node_id_matches(edge.dst)
    }
}

impl GraphStore for GraphProjection {
    // --- Point lookups ---

    fn get_node(&self, id: NodeId) -> Option<Node> {
        self.inner.get_node(id).filter(|n| self.node_matches(n))
    }

    fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        self.inner.get_edge(id).filter(|e| self.edge_matches(e))
    }

    fn get_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Node> {
        self.inner
            .get_node_versioned(id, epoch, transaction_id)
            .filter(|n| self.node_matches(n))
    }

    /// Returns a versioned edge if it passes projection filters.
    ///
    /// **Limitation**: `edge_matches` checks endpoint visibility via `get_node`
    /// (current snapshot), not `get_node_versioned`, because `GraphProjection`
    /// does not store epoch/transaction context. This means endpoint filtering
    /// may reflect the current state rather than the requested version.
    fn get_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Edge> {
        self.inner
            .get_edge_versioned(id, epoch, transaction_id)
            .filter(|e| self.edge_matches(e))
    }

    fn get_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> Option<Node> {
        self.inner
            .get_node_at_epoch(id, epoch)
            .filter(|n| self.node_matches(n))
    }

    fn get_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> Option<Edge> {
        self.inner
            .get_edge_at_epoch(id, epoch)
            .filter(|e| self.edge_matches(e))
    }

    // --- Property access ---

    fn get_node_property(&self, id: NodeId, key: &PropertyKey) -> Option<Value> {
        if !self.node_id_matches(id) {
            return None;
        }
        self.inner.get_node_property(id, key)
    }

    fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value> {
        self.inner
            .get_edge(id)
            .filter(|e| self.edge_matches(e))
            .and_then(|_| self.inner.get_edge_property(id, key))
    }

    fn get_node_property_batch(&self, ids: &[NodeId], key: &PropertyKey) -> Vec<Option<Value>> {
        let filtered: Vec<_> = ids
            .iter()
            .map(|&id| {
                if self.node_id_matches(id) {
                    self.inner.get_node_property(id, key)
                } else {
                    None
                }
            })
            .collect();
        filtered
    }

    fn get_nodes_properties_batch(&self, ids: &[NodeId]) -> Vec<FxHashMap<PropertyKey, Value>> {
        ids.iter()
            .map(|&id| {
                if self.node_id_matches(id) {
                    self.inner
                        .get_nodes_properties_batch(std::slice::from_ref(&id))
                        .into_iter()
                        .next()
                        .unwrap_or_default()
                } else {
                    FxHashMap::default()
                }
            })
            .collect()
    }

    fn get_nodes_properties_selective_batch(
        &self,
        ids: &[NodeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        ids.iter()
            .map(|&id| {
                if self.node_id_matches(id) {
                    self.inner
                        .get_nodes_properties_selective_batch(std::slice::from_ref(&id), keys)
                        .into_iter()
                        .next()
                        .unwrap_or_default()
                } else {
                    FxHashMap::default()
                }
            })
            .collect()
    }

    fn get_edges_properties_selective_batch(
        &self,
        ids: &[EdgeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        ids.iter()
            .map(|&id| {
                if self.get_edge(id).is_some() {
                    self.inner
                        .get_edges_properties_selective_batch(std::slice::from_ref(&id), keys)
                        .into_iter()
                        .next()
                        .unwrap_or_default()
                } else {
                    FxHashMap::default()
                }
            })
            .collect()
    }

    // --- Traversal ---

    fn neighbors(&self, node: NodeId, direction: Direction) -> Vec<NodeId> {
        if !self.node_id_matches(node) {
            return Vec::new();
        }
        // Use edges_from (which filters by edge type and endpoint visibility)
        // and extract the target node IDs, so neighbors connected only via
        // excluded edge types are not returned.
        self.edges_from(node, direction)
            .into_iter()
            .map(|(target, _)| target)
            .collect()
    }

    fn edges_from(&self, node: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)> {
        if !self.node_id_matches(node) {
            return Vec::new();
        }
        self.inner
            .edges_from(node, direction)
            .into_iter()
            .filter(|&(target, edge_id)| {
                self.node_id_matches(target)
                    && self
                        .inner
                        .edge_type(edge_id)
                        .is_some_and(|t| self.edge_type_matches(&t))
            })
            .collect()
    }

    fn out_degree(&self, node: NodeId) -> usize {
        self.edges_from(node, Direction::Outgoing).len()
    }

    fn in_degree(&self, node: NodeId) -> usize {
        self.edges_from(node, Direction::Incoming).len()
    }

    fn has_backward_adjacency(&self) -> bool {
        self.inner.has_backward_adjacency()
    }

    // --- Scans ---

    fn node_ids(&self) -> Vec<NodeId> {
        if !self.spec.filters_labels() {
            return self.inner.node_ids();
        }
        self.inner
            .node_ids()
            .into_iter()
            .filter(|&id| self.node_id_matches(id))
            .collect()
    }

    fn all_node_ids(&self) -> Vec<NodeId> {
        if !self.spec.filters_labels() {
            return self.inner.all_node_ids();
        }
        self.inner
            .all_node_ids()
            .into_iter()
            .filter(|&id| self.node_id_matches(id))
            .collect()
    }

    fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        if self.spec.filters_labels() && !self.spec.node_labels.contains(label) {
            return Vec::new();
        }
        self.inner.nodes_by_label(label)
    }

    fn nodes_by_label_count(&self, label: &str) -> usize {
        if self.spec.filters_labels() && !self.spec.node_labels.contains(label) {
            return 0;
        }
        self.inner.nodes_by_label_count(label)
    }

    fn node_count(&self) -> usize {
        self.node_ids().len()
    }

    fn edge_count(&self) -> usize {
        // Approximate: count edges whose type is in the spec
        if !self.spec.filters_edge_types() && !self.spec.filters_labels() {
            return self.inner.edge_count();
        }
        // Fallback: scan all nodes and count projected edges
        self.node_ids().iter().map(|&id| self.out_degree(id)).sum()
    }

    // --- Entity metadata ---

    fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
        // Check type filter first (cheap: no property loading)
        let et = self.inner.edge_type(id)?;
        if !self.edge_type_matches(&et) {
            return None;
        }
        // Check endpoint visibility only if labels are filtered
        if self.spec.filters_labels() {
            let edge = self.inner.get_edge(id)?;
            if !self.node_id_matches(edge.src) || !self.node_id_matches(edge.dst) {
                return None;
            }
        }
        Some(et)
    }

    /// Returns the type of a versioned edge if it passes projection filters.
    ///
    /// **Limitation**: endpoint visibility is checked via `get_node` (current
    /// snapshot), not `get_node_versioned`. See `get_edge_versioned` for details.
    fn edge_type_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<ArcStr> {
        let et = self.inner.edge_type_versioned(id, epoch, transaction_id)?;
        if !self.edge_type_matches(&et) {
            return None;
        }
        if self.spec.filters_labels() {
            let edge = self.inner.get_edge_versioned(id, epoch, transaction_id)?;
            if !self.node_id_matches(edge.src) || !self.node_id_matches(edge.dst) {
                return None;
            }
        }
        Some(et)
    }

    // --- Index introspection ---

    fn has_property_index(&self, property: &str) -> bool {
        self.inner.has_property_index(property)
    }

    // --- Filtered search ---

    fn find_nodes_by_property(&self, property: &str, value: &Value) -> Vec<NodeId> {
        self.inner
            .find_nodes_by_property(property, value)
            .into_iter()
            .filter(|&id| self.node_id_matches(id))
            .collect()
    }

    fn find_nodes_by_properties(&self, conditions: &[(&str, Value)]) -> Vec<NodeId> {
        self.inner
            .find_nodes_by_properties(conditions)
            .into_iter()
            .filter(|&id| self.node_id_matches(id))
            .collect()
    }

    fn find_nodes_in_range(
        &self,
        property: &str,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Vec<NodeId> {
        self.inner
            .find_nodes_in_range(property, min, max, min_inclusive, max_inclusive)
            .into_iter()
            .filter(|&id| self.node_id_matches(id))
            .collect()
    }

    // --- Zone maps ---

    fn node_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        self.inner.node_property_might_match(property, op, value)
    }

    fn edge_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        self.inner.edge_property_might_match(property, op, value)
    }

    // --- Statistics ---

    fn statistics(&self) -> Arc<Statistics> {
        self.inner.statistics()
    }

    fn estimate_label_cardinality(&self, label: &str) -> f64 {
        if self.spec.filters_labels() && !self.spec.node_labels.contains(label) {
            return 0.0;
        }
        self.inner.estimate_label_cardinality(label)
    }

    fn estimate_avg_degree(&self, edge_type: &str, outgoing: bool) -> f64 {
        if self.spec.filters_edge_types() && !self.spec.edge_types.contains(edge_type) {
            return 0.0;
        }
        self.inner.estimate_avg_degree(edge_type, outgoing)
    }

    // --- Epoch ---

    fn current_epoch(&self) -> EpochId {
        self.inner.current_epoch()
    }

    // --- Schema introspection ---

    fn all_labels(&self) -> Vec<String> {
        if self.spec.filters_labels() {
            self.spec.node_labels.iter().cloned().collect()
        } else {
            self.inner.all_labels()
        }
    }

    fn all_edge_types(&self) -> Vec<String> {
        if self.spec.filters_edge_types() {
            self.spec.edge_types.iter().cloned().collect()
        } else {
            self.inner.all_edge_types()
        }
    }

    fn all_property_keys(&self) -> Vec<String> {
        self.inner.all_property_keys()
    }
}

impl GraphStoreSearch for GraphProjection {}

#[cfg(test)]
#[cfg(feature = "lpg")]
mod tests {
    use super::*;
    use crate::graph::lpg::LpgStore;

    fn setup_social_graph() -> Arc<LpgStore> {
        let store = Arc::new(LpgStore::new().unwrap());
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        let amsterdam = store.create_node(&["City"]);
        let grafeo = store.create_node(&["Software"]);

        store.set_node_property(alix, "name", Value::from("Alix"));
        store.set_node_property(gus, "name", Value::from("Gus"));
        store.set_node_property(amsterdam, "name", Value::from("Amsterdam"));
        store.set_node_property(grafeo, "name", Value::from("Grafeo"));

        store.create_edge(alix, gus, "KNOWS");
        store.create_edge(alix, amsterdam, "LIVES_IN");
        store.create_edge(gus, amsterdam, "LIVES_IN");
        store.create_edge(alix, grafeo, "CONTRIBUTES_TO");

        store
    }

    #[test]
    fn unfiltered_projection_sees_everything() {
        let store = setup_social_graph();
        let proj = GraphProjection::new(store.clone(), ProjectionSpec::new());
        assert_eq!(proj.node_count(), store.node_count());
        assert_eq!(proj.edge_count(), store.edge_count());
    }

    #[test]
    fn filter_by_label() {
        let store = setup_social_graph();
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let proj = GraphProjection::new(store, spec);

        assert_eq!(proj.node_count(), 2);
        assert_eq!(proj.nodes_by_label("Person").len(), 2);
        assert!(proj.nodes_by_label("City").is_empty());
        assert!(proj.nodes_by_label("Software").is_empty());
    }

    #[test]
    fn filter_by_edge_type() {
        let store = setup_social_graph();
        let spec = ProjectionSpec::new().with_edge_types(["KNOWS"]);
        let proj = GraphProjection::new(store, spec);

        // All nodes visible (no label filter), but only KNOWS edges
        assert_eq!(proj.node_count(), 4);
        assert_eq!(proj.edge_count(), 1);
    }

    #[test]
    fn combined_label_and_edge_filter() {
        let store = setup_social_graph();
        let spec = ProjectionSpec::new()
            .with_node_labels(["Person", "City"])
            .with_edge_types(["LIVES_IN"]);
        let proj = GraphProjection::new(store, spec);

        assert_eq!(proj.node_count(), 3); // 2 Person + 1 City
        assert_eq!(proj.edge_count(), 2); // 2 LIVES_IN edges
    }

    #[test]
    fn edge_excluded_when_endpoint_excluded() {
        let store = setup_social_graph();
        // Only Person nodes, but LIVES_IN edge type
        // LIVES_IN goes Person -> City, but City is excluded
        let spec = ProjectionSpec::new()
            .with_node_labels(["Person"])
            .with_edge_types(["LIVES_IN"]);
        let proj = GraphProjection::new(store, spec);

        assert_eq!(proj.node_count(), 2);
        // LIVES_IN edges should be excluded because City endpoints are filtered out
        assert_eq!(proj.edge_count(), 0);
    }

    #[test]
    fn get_node_filtered() {
        let store = setup_social_graph();
        let all_ids = store.node_ids();
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let proj = GraphProjection::new(store.clone(), spec);

        // Person nodes visible
        assert!(proj.get_node(all_ids[0]).is_some()); // Alix (Person)
        assert!(proj.get_node(all_ids[1]).is_some()); // Gus (Person)
        // City and Software nodes hidden
        assert!(proj.get_node(all_ids[2]).is_none()); // Amsterdam (City)
        assert!(proj.get_node(all_ids[3]).is_none()); // Grafeo (Software)
    }

    #[test]
    fn neighbors_filtered() {
        let store = setup_social_graph();
        let alix_id = store.node_ids()[0];

        // Without projection: Alix has 3 outgoing neighbors (Gus, Amsterdam, Grafeo)
        let all_neighbors: Vec<_> = store.neighbors(alix_id, Direction::Outgoing).collect();
        assert_eq!(all_neighbors.len(), 3);

        // With Person-only projection: Alix -> Gus only
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let proj = GraphProjection::new(store, spec);
        let neighbors = proj.neighbors(alix_id, Direction::Outgoing);
        assert_eq!(neighbors.len(), 1);
    }

    #[test]
    fn neighbors_filtered_by_edge_type() {
        let store = setup_social_graph();
        let alix_id = store.node_ids()[0];

        // With edge-type filter: only KNOWS edges visible
        // Alix KNOWS Gus, but LIVES_IN Amsterdam and CONTRIBUTES_TO Grafeo are excluded
        let spec = ProjectionSpec::new().with_edge_types(["KNOWS"]);
        let proj = GraphProjection::new(store, spec);
        let neighbors = proj.neighbors(alix_id, Direction::Outgoing);
        assert_eq!(neighbors.len(), 1);
    }

    #[test]
    fn property_access_respects_filter() {
        let store = setup_social_graph();
        let city_id = store.node_ids()[2]; // Amsterdam
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let proj = GraphProjection::new(store, spec);

        // City node properties are inaccessible
        assert!(
            proj.get_node_property(city_id, &PropertyKey::from("name"))
                .is_none()
        );
    }

    #[test]
    fn cardinality_estimation_respects_filter() {
        let store = setup_social_graph();
        let spec = ProjectionSpec::new()
            .with_node_labels(["Person"])
            .with_edge_types(["KNOWS"]);
        let proj = GraphProjection::new(store, spec);

        assert_eq!(proj.estimate_label_cardinality("City"), 0.0);
        assert_eq!(proj.estimate_avg_degree("LIVES_IN", true), 0.0);
    }

    #[test]
    fn schema_introspection_reflects_filter() {
        let store = setup_social_graph();
        let spec = ProjectionSpec::new()
            .with_node_labels(["Person"])
            .with_edge_types(["KNOWS"]);
        let proj = GraphProjection::new(store, spec);

        let labels = proj.all_labels();
        assert_eq!(labels.len(), 1);
        assert!(labels.contains(&"Person".to_string()));

        let edge_types = proj.all_edge_types();
        assert_eq!(edge_types.len(), 1);
        assert!(edge_types.contains(&"KNOWS".to_string()));
    }

    /// Helper: returns (node_ids, edge_ids) from the social graph.
    fn setup_social_graph_with_ids() -> (Arc<LpgStore>, Vec<NodeId>, Vec<EdgeId>) {
        let store = Arc::new(LpgStore::new().unwrap());
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        let amsterdam = store.create_node(&["City"]);
        let grafeo = store.create_node(&["Software"]);

        store.set_node_property(alix, "name", Value::from("Alix"));
        store.set_node_property(gus, "name", Value::from("Gus"));
        store.set_node_property(amsterdam, "name", Value::from("Amsterdam"));
        store.set_node_property(grafeo, "name", Value::from("Grafeo"));
        store.set_node_property(alix, "age", Value::from(30));
        store.set_node_property(gus, "age", Value::from(25));

        let e_knows = store.create_edge(alix, gus, "KNOWS");
        let e_alix_lives = store.create_edge(alix, amsterdam, "LIVES_IN");
        let e_gus_lives = store.create_edge(gus, amsterdam, "LIVES_IN");
        let e_contrib = store.create_edge(alix, grafeo, "CONTRIBUTES_TO");

        store.set_edge_property(e_knows, "since", Value::from(2020));
        store.set_edge_property(e_alix_lives, "since", Value::from(2018));

        let nodes = vec![alix, gus, amsterdam, grafeo];
        let edges = vec![e_knows, e_alix_lives, e_gus_lives, e_contrib];
        (store, nodes, edges)
    }

    // 1. get_edge with edge that passes/fails type filter

    #[test]
    fn get_edge_passes_type_filter() {
        let (store, _, edges) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_edge_types(["KNOWS"]);
        let proj = GraphProjection::new(store, spec);

        // KNOWS edge passes filter
        assert!(proj.get_edge(edges[0]).is_some());
        // LIVES_IN edge does not pass filter
        assert!(proj.get_edge(edges[1]).is_none());
        // CONTRIBUTES_TO edge does not pass filter
        assert!(proj.get_edge(edges[3]).is_none());
    }

    #[test]
    fn get_edge_excluded_by_endpoint_label_filter() {
        let (store, _, edges) = setup_social_graph_with_ids();
        // Only Person nodes, LIVES_IN goes Person->City, so excluded
        let spec = ProjectionSpec::new()
            .with_node_labels(["Person"])
            .with_edge_types(["LIVES_IN"]);
        let proj = GraphProjection::new(store, spec);

        assert!(proj.get_edge(edges[1]).is_none()); // Alix->Amsterdam
        assert!(proj.get_edge(edges[2]).is_none()); // Gus->Amsterdam
    }

    // 2. get_node_versioned and get_edge_versioned

    #[test]
    fn get_node_versioned_respects_filter() {
        let (store, nodes, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let proj = GraphProjection::new(store, spec);

        let epoch = EpochId(0);
        let txn = TransactionId(0);

        // Person node visible
        assert!(proj.get_node_versioned(nodes[0], epoch, txn).is_some());
        // City node filtered out
        assert!(proj.get_node_versioned(nodes[2], epoch, txn).is_none());
    }

    #[test]
    fn get_edge_versioned_respects_filter() {
        let (store, _, edges) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_edge_types(["KNOWS"]);
        let proj = GraphProjection::new(store, spec);

        let epoch = EpochId(0);
        let txn = TransactionId(0);

        // KNOWS edge visible
        assert!(proj.get_edge_versioned(edges[0], epoch, txn).is_some());
        // LIVES_IN edge filtered out
        assert!(proj.get_edge_versioned(edges[1], epoch, txn).is_none());
    }

    // 3. get_node_at_epoch and get_edge_at_epoch

    #[test]
    fn get_node_at_epoch_respects_filter() {
        let (store, nodes, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_node_labels(["City"]);
        let proj = GraphProjection::new(store, spec);

        let epoch = EpochId(0);

        // Amsterdam (City) visible
        assert!(proj.get_node_at_epoch(nodes[2], epoch).is_some());
        // Alix (Person) filtered out
        assert!(proj.get_node_at_epoch(nodes[0], epoch).is_none());
    }

    #[test]
    fn get_edge_at_epoch_respects_filter() {
        let (store, _, edges) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_edge_types(["LIVES_IN"]);
        let proj = GraphProjection::new(store, spec);

        let epoch = EpochId(0);

        // LIVES_IN edge visible
        assert!(proj.get_edge_at_epoch(edges[1], epoch).is_some());
        // KNOWS edge filtered out
        assert!(proj.get_edge_at_epoch(edges[0], epoch).is_none());
    }

    // 4. get_edge_property for edges in/out of projection

    #[test]
    fn get_edge_property_in_projection() {
        let (store, _, edges) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_edge_types(["KNOWS"]);
        let proj = GraphProjection::new(store, spec);

        let key = PropertyKey::from("since");
        // KNOWS edge has "since" property and passes filter
        assert_eq!(
            proj.get_edge_property(edges[0], &key),
            Some(Value::from(2020))
        );
    }

    #[test]
    fn get_edge_property_outside_projection() {
        let (store, _, edges) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_edge_types(["KNOWS"]);
        let proj = GraphProjection::new(store, spec);

        let key = PropertyKey::from("since");
        // LIVES_IN edge has "since" but is filtered out
        assert!(proj.get_edge_property(edges[1], &key).is_none());
    }

    // 5. get_node_property_batch for mixed in/out of projection nodes

    #[test]
    fn get_node_property_batch_mixed() {
        let (store, nodes, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let proj = GraphProjection::new(store, spec);

        let key = PropertyKey::from("name");
        // Alix (Person), Amsterdam (City), Gus (Person)
        let ids = vec![nodes[0], nodes[2], nodes[1]];
        let results = proj.get_node_property_batch(&ids, &key);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0], Some(Value::from("Alix"))); // Person: visible
        assert_eq!(results[1], None); // City: filtered out
        assert_eq!(results[2], Some(Value::from("Gus"))); // Person: visible
    }

    // 6. get_nodes_properties_batch and selective batch

    #[test]
    fn get_nodes_properties_batch_filters() {
        let (store, nodes, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let proj = GraphProjection::new(store, spec);

        let ids = vec![nodes[0], nodes[2]]; // Alix (Person), Amsterdam (City)
        let results = proj.get_nodes_properties_batch(&ids);

        assert_eq!(results.len(), 2);
        // Alix has properties
        assert!(results[0].contains_key(&PropertyKey::from("name")));
        // Amsterdam filtered out, empty map
        assert!(results[1].is_empty());
    }

    #[test]
    fn get_nodes_properties_selective_batch_filters() {
        let (store, nodes, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let proj = GraphProjection::new(store, spec);

        let ids = vec![nodes[0], nodes[2]]; // Alix (Person), Amsterdam (City)
        let keys = vec![PropertyKey::from("name")];
        let results = proj.get_nodes_properties_selective_batch(&ids, &keys);

        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].get(&PropertyKey::from("name")),
            Some(&Value::from("Alix"))
        );
        assert!(results[1].is_empty());
    }

    // 7. get_edges_properties_selective_batch

    #[test]
    fn get_edges_properties_selective_batch_filters() {
        let (store, _, edges) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_edge_types(["KNOWS"]);
        let proj = GraphProjection::new(store, spec);

        let ids = vec![edges[0], edges[1]]; // KNOWS, LIVES_IN
        let keys = vec![PropertyKey::from("since")];
        let results = proj.get_edges_properties_selective_batch(&ids, &keys);

        assert_eq!(results.len(), 2);
        // KNOWS edge has "since" and passes filter
        assert_eq!(
            results[0].get(&PropertyKey::from("since")),
            Some(&Value::from(2020))
        );
        // LIVES_IN edge filtered out
        assert!(results[1].is_empty());
    }

    // 8. edges_from with edge type filter

    #[test]
    fn edges_from_with_edge_type_filter() {
        let (store, nodes, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_edge_types(["LIVES_IN"]);
        let proj = GraphProjection::new(store, spec);

        // Alix has outgoing KNOWS, LIVES_IN, CONTRIBUTES_TO
        // Only LIVES_IN should be visible
        let alix_edges = proj.edges_from(nodes[0], Direction::Outgoing);
        assert_eq!(alix_edges.len(), 1);
        assert_eq!(alix_edges[0].0, nodes[2]); // target is Amsterdam
    }

    #[test]
    fn edges_from_filtered_node_returns_empty() {
        let (store, nodes, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let proj = GraphProjection::new(store, spec);

        // Amsterdam (City) is filtered out, so edges_from returns empty
        let amsterdam_edges = proj.edges_from(nodes[2], Direction::Outgoing);
        assert!(amsterdam_edges.is_empty());
    }

    // 9. out_degree and in_degree with filtered projection

    #[test]
    fn out_degree_with_filter() {
        let (store, nodes, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new()
            .with_node_labels(["Person", "City"])
            .with_edge_types(["LIVES_IN"]);
        let proj = GraphProjection::new(store, spec);

        // Alix has 1 outgoing LIVES_IN to Amsterdam
        assert_eq!(proj.out_degree(nodes[0]), 1);
        // Gus has 1 outgoing LIVES_IN to Amsterdam
        assert_eq!(proj.out_degree(nodes[1]), 1);
        // Amsterdam has no outgoing LIVES_IN
        assert_eq!(proj.out_degree(nodes[2]), 0);
    }

    #[test]
    fn in_degree_with_filter() {
        let (store, nodes, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new()
            .with_node_labels(["Person", "City"])
            .with_edge_types(["LIVES_IN"]);
        let proj = GraphProjection::new(store, spec);

        // Amsterdam has 2 incoming LIVES_IN edges
        assert_eq!(proj.in_degree(nodes[2]), 2);
        // Alix has 0 incoming LIVES_IN
        assert_eq!(proj.in_degree(nodes[0]), 0);
    }

    // 10. all_node_ids

    #[test]
    fn all_node_ids_with_label_filter() {
        let (store, nodes, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let proj = GraphProjection::new(store, spec);

        let ids = proj.all_node_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&nodes[0])); // Alix
        assert!(ids.contains(&nodes[1])); // Gus
        assert!(!ids.contains(&nodes[2])); // Amsterdam excluded
    }

    #[test]
    fn all_node_ids_unfiltered() {
        let (store, _, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new();
        let proj = GraphProjection::new(store.clone(), spec);

        assert_eq!(proj.all_node_ids().len(), store.all_node_ids().len());
    }

    // 11. node_count and edge_count with various filters

    #[test]
    fn node_count_with_city_filter() {
        let (store, _, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_node_labels(["City"]);
        let proj = GraphProjection::new(store, spec);

        assert_eq!(proj.node_count(), 1);
    }

    #[test]
    fn edge_count_with_combined_filter() {
        let (store, _, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new()
            .with_node_labels(["Person"])
            .with_edge_types(["KNOWS"]);
        let proj = GraphProjection::new(store, spec);

        // Only KNOWS between Person nodes: Alix->Gus
        assert_eq!(proj.edge_count(), 1);
    }

    #[test]
    fn edge_count_unfiltered_delegates() {
        let (store, _, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new();
        let proj = GraphProjection::new(store.clone(), spec);

        assert_eq!(proj.edge_count(), store.edge_count());
    }

    // 12. find_nodes_by_property with label filter

    #[test]
    fn find_nodes_by_property_with_label_filter() {
        let (store, nodes, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let proj = GraphProjection::new(store, spec);

        // "name" = "Alix" exists on a Person node
        let found = proj.find_nodes_by_property("name", &Value::from("Alix"));
        assert_eq!(found.len(), 1);
        assert_eq!(found[0], nodes[0]);

        // "name" = "Amsterdam" exists but on a City node, which is filtered
        let found = proj.find_nodes_by_property("name", &Value::from("Amsterdam"));
        assert!(found.is_empty());
    }

    // 13. find_nodes_by_properties with label filter

    #[test]
    fn find_nodes_by_properties_with_label_filter() {
        let (store, nodes, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let proj = GraphProjection::new(store, spec);

        let conditions = vec![("name", Value::from("Gus"))];
        let found = proj.find_nodes_by_properties(&conditions);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0], nodes[1]);

        // Search for city name, filtered out
        let conditions = vec![("name", Value::from("Amsterdam"))];
        let found = proj.find_nodes_by_properties(&conditions);
        assert!(found.is_empty());
    }

    // 14. find_nodes_in_range with label filter

    #[test]
    fn find_nodes_in_range_with_label_filter() {
        let (store, nodes, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let proj = GraphProjection::new(store, spec);

        // Age range 20..=30 should find both Alix (30) and Gus (25)
        let min = Value::from(20);
        let max = Value::from(30);
        let found = proj.find_nodes_in_range("age", Some(&min), Some(&max), true, true);
        assert_eq!(found.len(), 2);
        assert!(found.contains(&nodes[0])); // Alix
        assert!(found.contains(&nodes[1])); // Gus
    }

    #[test]
    fn find_nodes_in_range_excludes_filtered_labels() {
        let (store, _, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_node_labels(["City"]);
        let proj = GraphProjection::new(store, spec);

        // City nodes don't have "age" property
        let min = Value::from(20);
        let max = Value::from(30);
        let found = proj.find_nodes_in_range("age", Some(&min), Some(&max), true, true);
        assert!(found.is_empty());
    }

    // 15. node_property_might_match and edge_property_might_match

    #[test]
    fn node_property_might_match_delegates() {
        let (store, _, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let proj = GraphProjection::new(store.clone(), spec);

        let key = PropertyKey::from("name");
        let val = Value::from("Alix");
        // Delegates to inner store, so result should match
        let inner_result = store.node_property_might_match(&key, CompareOp::Eq, &val);
        assert_eq!(
            proj.node_property_might_match(&key, CompareOp::Eq, &val),
            inner_result
        );
    }

    #[test]
    fn edge_property_might_match_delegates() {
        let (store, _, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_edge_types(["KNOWS"]);
        let proj = GraphProjection::new(store.clone(), spec);

        let key = PropertyKey::from("since");
        let val = Value::from(2020);
        let inner_result = store.edge_property_might_match(&key, CompareOp::Eq, &val);
        assert_eq!(
            proj.edge_property_might_match(&key, CompareOp::Eq, &val),
            inner_result
        );
    }

    // 16. current_epoch

    #[test]
    fn current_epoch_delegates() {
        let (store, _, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new();
        let proj = GraphProjection::new(store.clone(), spec);

        assert_eq!(proj.current_epoch(), store.current_epoch());
    }

    // 17. all_property_keys

    #[test]
    fn all_property_keys_delegates() {
        let (store, _, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let proj = GraphProjection::new(store.clone(), spec);

        let proj_keys = proj.all_property_keys();
        let store_keys = store.all_property_keys();
        // Delegates to inner, so same result
        assert_eq!(proj_keys.len(), store_keys.len());
    }

    // 18. statistics returns non-null

    #[test]
    fn statistics_returns_value() {
        let (store, _, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new();
        let proj = GraphProjection::new(store, spec);

        let stats = proj.statistics();
        // Just verify it returns without panicking and is non-null (Arc)
        let _ = stats;
    }

    // 19. has_backward_adjacency delegates to inner

    #[test]
    fn has_backward_adjacency_delegates() {
        let (store, _, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new();
        let proj = GraphProjection::new(store.clone(), spec);

        assert_eq!(
            proj.has_backward_adjacency(),
            store.has_backward_adjacency()
        );
    }

    // 20. has_property_index delegates to inner

    #[test]
    fn has_property_index_delegates() {
        let (store, _, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new();
        let proj = GraphProjection::new(store.clone(), spec);

        // LpgStore default has no property indexes
        assert_eq!(
            proj.has_property_index("name"),
            store.has_property_index("name")
        );
    }
}
