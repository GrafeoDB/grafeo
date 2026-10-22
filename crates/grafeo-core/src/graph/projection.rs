//! Graph projections: read-only, filtered views of a graph store.
//!
//! A [`GraphProjection`] wraps an existing [`GraphStore`] and presents a
//! subgraph defined by a [`ProjectionSpec`]. Only nodes with matching labels
//! and edges with matching types (whose endpoints are both in the projection)
//! are visible. Everything else is filtered out transparently.
//!
//! Projections are read-only: they implement [`GraphStore`] but not
//! [`super::GraphBackend`]. A projection keeps no text or vector index of
//! its own: its text reads answer as a store without one, and its vector
//! search measures the vectors of the nodes in it.
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
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, ReadView, Value};
use grafeo_common::utils::error::Result;
use grafeo_common::utils::hash::FxHashMap;

use super::Direction;
use super::lpg::{CompareOp, Edge, Node};
use super::traits::{GraphStore, Labels};
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
    inner: Arc<dyn GraphStore>,
    spec: ProjectionSpec,
}

impl GraphProjection {
    /// Creates a new projection over the given store.
    pub fn new(inner: Arc<dyn GraphStore>, spec: ProjectionSpec) -> Self {
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

    /// Whether node `id` passes the label filter with the labels `view`
    /// sees: every node passes without a filter.
    fn node_id_matches(&self, id: NodeId, view: ReadView) -> Result<bool> {
        if !self.spec.filters_labels() {
            return Ok(true);
        }
        Ok(self.inner.node_labels(id, view)?.is_some_and(|labels| {
            labels
                .iter()
                .any(|label| self.spec.node_labels.contains(label.as_str()))
        }))
    }

    /// Returns true if every node with `label` is in the projection: no
    /// label filter, or a label of the spec.
    fn label_is_projected(&self, label: &str) -> bool {
        !self.spec.filters_labels() || self.spec.node_labels.contains(label)
    }

    /// Returns true if an edge type passes the type filter.
    fn edge_type_matches(&self, edge_type: &str) -> bool {
        if !self.spec.filters_edge_types() {
            return true;
        }
        self.spec.edge_types.contains(edge_type)
    }

    /// The nodes of `ids` in the projection at `view`, in their order.
    fn projected(&self, ids: Vec<NodeId>, view: ReadView) -> Result<Vec<NodeId>> {
        if !self.spec.filters_labels() {
            return Ok(ids);
        }
        let mut kept = Vec::with_capacity(ids.len());
        for id in ids {
            if self.node_id_matches(id, view)? {
                kept.push(id);
            }
        }
        Ok(kept)
    }

    /// The edge types an adjacency read of `types` asks the inner store
    /// for: `types` within the spec's types, the spec's types for none;
    /// `None` when no type is left.
    fn inner_types(&self, types: &[ArcStr]) -> Option<Vec<ArcStr>> {
        if !self.spec.filters_edge_types() {
            return Some(types.to_vec());
        }
        let wanted: Vec<ArcStr> = if types.is_empty() {
            self.spec
                .edge_types
                .iter()
                .map(|name| ArcStr::from(name.as_str()))
                .collect()
        } else {
            types
                .iter()
                .filter(|name| self.spec.edge_types.contains(name.as_str()))
                .cloned()
                .collect()
        };
        (!wanted.is_empty()).then_some(wanted)
    }
}

impl GraphStore for GraphProjection {
    // --- Point reads ---

    fn node(&self, id: NodeId, view: ReadView) -> Result<Option<Node>> {
        Ok(self
            .inner
            .node(id, view)?
            .filter(|node| self.node_matches(node)))
    }

    fn edge(&self, id: EdgeId, view: ReadView) -> Result<Option<Edge>> {
        let Some(edge) = self.inner.edge(id, view)? else {
            return Ok(None);
        };
        let kept = self.edge_type_matches(&edge.edge_type)
            && self.node_id_matches(edge.src, view)?
            && self.node_id_matches(edge.dst, view)?;
        Ok(kept.then_some(edge))
    }

    fn is_node_visible(&self, id: NodeId, view: ReadView) -> Result<bool> {
        Ok(self.inner.is_node_visible(id, view)? && self.node_id_matches(id, view)?)
    }

    fn is_edge_visible(&self, id: EdgeId, view: ReadView) -> Result<bool> {
        // `edge_type` applies the type filter and, under a label filter, the
        // endpoint filter.
        Ok(self.edge_type(id, view)?.is_some())
    }

    fn node_property(&self, id: NodeId, key: &str, view: ReadView) -> Result<Option<Value>> {
        if !self.node_id_matches(id, view)? {
            return Ok(None);
        }
        self.inner.node_property(id, key, view)
    }

    fn edge_property(&self, id: EdgeId, key: &str, view: ReadView) -> Result<Option<Value>> {
        if self.edge_type(id, view)?.is_none() {
            return Ok(None);
        }
        self.inner.edge_property(id, key, view)
    }

    fn node_labels(&self, id: NodeId, view: ReadView) -> Result<Option<Labels>> {
        Ok(self.inner.node_labels(id, view)?.filter(|labels| {
            !self.spec.filters_labels()
                || labels
                    .iter()
                    .any(|label| self.spec.node_labels.contains(label.as_str()))
        }))
    }

    fn node_has_label(&self, id: NodeId, label: &str, view: ReadView) -> Result<Option<bool>> {
        if !self.node_id_matches(id, view)? {
            return Ok(None);
        }
        self.inner.node_has_label(id, label, view)
    }

    fn edge_type(&self, id: EdgeId, view: ReadView) -> Result<Option<ArcStr>> {
        // Check type filter first (cheap: no property loading)
        let Some(edge_type) = self.inner.edge_type(id, view)? else {
            return Ok(None);
        };
        if !self.edge_type_matches(&edge_type) {
            return Ok(None);
        }
        // Check endpoint visibility only if labels are filtered
        if self.spec.filters_labels() {
            let Some(edge) = self.inner.edge(id, view)? else {
                return Ok(None);
            };
            if !self.node_id_matches(edge.src, view)? || !self.node_id_matches(edge.dst, view)? {
                return Ok(None);
            }
        }
        Ok(Some(edge_type))
    }

    // --- Batch reads ---

    fn node_property_batch(
        &self,
        ids: &[NodeId],
        key: &PropertyKey,
        view: ReadView,
    ) -> Result<Vec<Option<Value>>> {
        // The nodes in the projection read through the inner store's batch;
        // the others have no value here.
        let mut included = Vec::with_capacity(ids.len());
        for &id in ids {
            included.push(self.node_id_matches(id, view)?);
        }
        let inside: Vec<NodeId> = ids
            .iter()
            .zip(&included)
            .filter_map(|(&id, &included)| included.then_some(id))
            .collect();
        let mut values = self
            .inner
            .node_property_batch(&inside, key, view)?
            .into_iter();
        Ok(included
            .iter()
            .map(|&included| {
                if included {
                    values.next().flatten()
                } else {
                    None
                }
            })
            .collect())
    }

    fn node_properties_batch(
        &self,
        ids: &[NodeId],
        view: ReadView,
    ) -> Result<Vec<FxHashMap<PropertyKey, Value>>> {
        ids.iter()
            .map(|&id| {
                if self.node_id_matches(id, view)? {
                    Ok(self
                        .inner
                        .node_properties_batch(std::slice::from_ref(&id), view)?
                        .into_iter()
                        .next()
                        .unwrap_or_default())
                } else {
                    Ok(FxHashMap::default())
                }
            })
            .collect()
    }

    fn node_properties_selective_batch(
        &self,
        ids: &[NodeId],
        keys: &[PropertyKey],
        view: ReadView,
    ) -> Result<Vec<FxHashMap<PropertyKey, Value>>> {
        ids.iter()
            .map(|&id| {
                if self.node_id_matches(id, view)? {
                    Ok(self
                        .inner
                        .node_properties_selective_batch(std::slice::from_ref(&id), keys, view)?
                        .into_iter()
                        .next()
                        .unwrap_or_default())
                } else {
                    Ok(FxHashMap::default())
                }
            })
            .collect()
    }

    fn edge_properties_selective_batch(
        &self,
        ids: &[EdgeId],
        keys: &[PropertyKey],
        view: ReadView,
    ) -> Result<Vec<FxHashMap<PropertyKey, Value>>> {
        ids.iter()
            .map(|&id| {
                if self.edge_type(id, view)?.is_some() {
                    Ok(self
                        .inner
                        .edge_properties_selective_batch(std::slice::from_ref(&id), keys, view)?
                        .into_iter()
                        .next()
                        .unwrap_or_default())
                } else {
                    Ok(FxHashMap::default())
                }
            })
            .collect()
    }

    // --- Traversal ---

    /// The inner store's edges of the types left after the spec's, whose
    /// other end is in the projection.
    fn edges(
        &self,
        node: NodeId,
        direction: Direction,
        types: &[ArcStr],
        view: ReadView,
    ) -> Result<Vec<(NodeId, EdgeId)>> {
        if !self.node_id_matches(node, view)? {
            return Ok(Vec::new());
        }
        let Some(types) = self.inner_types(types) else {
            return Ok(Vec::new());
        };
        let mut edges = Vec::new();
        for (other, edge) in self.inner.edges(node, direction, &types, view)? {
            if self.node_id_matches(other, view)? {
                edges.push((other, edge));
            }
        }
        Ok(edges)
    }

    fn has_backward_adjacency(&self) -> bool {
        self.inner.has_backward_adjacency()
    }

    // --- Scans ---

    fn node_ids(&self, view: ReadView) -> Result<Vec<NodeId>> {
        self.projected(self.inner.node_ids(view)?, view)
    }

    // A node with a label outside the spec is in the projection when it has
    // a label of the spec too, and `node` shows it with both: the inner
    // store's nodes with the label are filtered, in their order.
    fn nodes_by_label(&self, label: &str, view: ReadView) -> Result<Vec<NodeId>> {
        let ids = self.inner.nodes_by_label(label, view)?;
        if self.label_is_projected(label) {
            return Ok(ids);
        }
        self.projected(ids, view)
    }

    fn nodes_with_any_label(&self, labels: &[&str], view: ReadView) -> Result<Vec<NodeId>> {
        let ids = self.inner.nodes_with_any_label(labels, view)?;
        if labels.iter().all(|label| self.label_is_projected(label)) {
            return Ok(ids);
        }
        self.projected(ids, view)
    }

    // Exact for a label outside the spec as well: `count(n)` reads it, not
    // only the planner.
    fn nodes_by_label_count(&self, label: &str, view: ReadView) -> Result<usize> {
        if self.label_is_projected(label) {
            return self.inner.nodes_by_label_count(label, view);
        }
        Ok(self.nodes_by_label(label, view)?.len())
    }

    fn node_count(&self, view: ReadView) -> Result<usize> {
        if !self.spec.filters_labels() {
            return self.inner.node_count(view);
        }
        Ok(self.node_ids(view)?.len())
    }

    fn edge_count(&self, view: ReadView) -> Result<usize> {
        if !self.spec.filters_edge_types() && !self.spec.filters_labels() {
            return self.inner.edge_count(view);
        }
        let mut count = 0;
        for node in self.node_ids(view)? {
            count += self.degree(node, Direction::Outgoing, view)?;
        }
        Ok(count)
    }

    // --- Names ---

    fn all_labels(&self) -> Result<Vec<String>> {
        if self.spec.filters_labels() {
            Ok(self.spec.node_labels.iter().cloned().collect())
        } else {
            self.inner.all_labels()
        }
    }

    fn all_edge_types(&self) -> Result<Vec<String>> {
        if self.spec.filters_edge_types() {
            Ok(self.spec.edge_types.iter().cloned().collect())
        } else {
            self.inner.all_edge_types()
        }
    }

    fn all_property_keys(&self) -> Result<Vec<String>> {
        self.inner.all_property_keys()
    }

    // --- Lookups: the inner store's, filtered by the spec ---

    fn has_property_index(&self, property: &str) -> bool {
        self.inner.has_property_index(property)
    }

    fn find_nodes_by_property(
        &self,
        property: &str,
        value: &Value,
        view: ReadView,
    ) -> Result<Vec<NodeId>> {
        self.projected(
            self.inner.find_nodes_by_property(property, value, view)?,
            view,
        )
    }

    fn find_nodes_by_properties(
        &self,
        conditions: &[(&str, Value)],
        view: ReadView,
    ) -> Result<Vec<NodeId>> {
        self.projected(self.inner.find_nodes_by_properties(conditions, view)?, view)
    }

    fn find_nodes_maybe_equal(
        &self,
        property: &str,
        value: &Value,
        view: ReadView,
    ) -> Result<Option<Vec<NodeId>>> {
        match self.inner.find_nodes_maybe_equal(property, value, view)? {
            Some(nodes) => Ok(Some(self.projected(nodes, view)?)),
            None => Ok(None),
        }
    }

    fn find_nodes_in_range(
        &self,
        property: &str,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
        view: ReadView,
    ) -> Result<Vec<NodeId>> {
        self.projected(
            self.inner.find_nodes_in_range(
                property,
                min,
                max,
                min_inclusive,
                max_inclusive,
                view,
            )?,
            view,
        )
    }

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

    /// The inner store's statistics, bounded by the spec: an edge type
    /// outside it has no edges, and a label outside a label filter at most
    /// as many nodes as the labels of the spec have. Estimates for the
    /// planner, not answers.
    fn committed_epoch(&self) -> EpochId {
        self.inner.committed_epoch()
    }

    fn statistics(&self) -> Arc<Statistics> {
        let inner = self.inner.statistics();
        if !self.spec.filters_labels() && !self.spec.filters_edge_types() {
            return inner;
        }
        let mut statistics = (*inner).clone();
        if self.spec.filters_edge_types() {
            for (name, edge_type) in &mut statistics.edge_types {
                if !self.spec.edge_types.contains(name) {
                    edge_type.edge_count = 0;
                    edge_type.avg_out_degree = 0.0;
                    edge_type.avg_in_degree = 0.0;
                }
            }
            statistics.total_edges = statistics
                .edge_types
                .values()
                .map(|edge_type| edge_type.edge_count)
                .sum();
        }
        if self.spec.filters_labels() {
            let projected: u64 = statistics
                .labels
                .iter()
                .filter(|(name, _)| self.spec.node_labels.contains(*name))
                .map(|(_, label)| label.node_count)
                .sum();
            // A label outside the spec counts the projection's nodes with it,
            // as a committed reader of the inner store's last commit finds
            // them (the planner scans the label with the fewest); when that
            // read fails, the inner count bounded by the projection's nodes
            // (statistics are estimates: a plan, not a result, depends on
            // them).
            let now = ReadView::committed(self.inner.committed_epoch());
            for (name, label) in &mut statistics.labels {
                if !self.spec.node_labels.contains(name) {
                    label.node_count = self
                        .nodes_by_label_count(name, now)
                        .ok()
                        .and_then(|count| u64::try_from(count).ok())
                        .unwrap_or_else(|| label.node_count.min(projected));
                }
            }
            statistics.total_nodes = statistics.total_nodes.min(projected);
        }
        Arc::new(statistics)
    }

    // --- History: the inner store's versions that are in the projection ---

    fn node_history(&self, id: NodeId) -> Result<Vec<(EpochId, Option<EpochId>, Node)>> {
        let mut versions = self.inner.node_history(id)?;
        versions.retain(|(_, _, node)| self.node_matches(node));
        Ok(versions)
    }

    fn edge_history(&self, id: EdgeId) -> Result<Vec<(EpochId, Option<EpochId>, Edge)>> {
        let mut versions = self.inner.edge_history(id)?;
        versions.retain(|(_, _, edge)| self.edge_type_matches(&edge.edge_type));
        Ok(versions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::rowgroup::RowGroupStore;
    use grafeo_common::types::TransactionId;

    fn setup_social_graph() -> Arc<RowGroupStore> {
        let store = Arc::new(RowGroupStore::new());
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
        assert_eq!(
            proj.node_count(ReadView::latest()).unwrap(),
            store.node_count(ReadView::latest()).unwrap()
        );
        assert_eq!(
            proj.edge_count(ReadView::latest()).unwrap(),
            store.edge_count(ReadView::latest()).unwrap()
        );
    }

    #[test]
    fn filter_by_label() {
        let store = setup_social_graph();
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let proj = GraphProjection::new(store, spec);

        assert_eq!(proj.node_count(ReadView::latest()).unwrap(), 2);
        assert_eq!(
            proj.nodes_by_label("Person", ReadView::latest())
                .unwrap()
                .len(),
            2
        );
        assert!(
            proj.nodes_by_label("City", ReadView::latest())
                .unwrap()
                .is_empty(),
            "expected empty"
        );
        assert!(
            proj.nodes_by_label("Software", ReadView::latest())
                .unwrap()
                .is_empty(),
            "expected empty"
        );
    }

    /// A label scan of a projection holds every node in it that has the
    /// label, as `get_node` reports its labels, in ID order, and the count is
    /// its length: for a label of the spec and for one outside it (`Admin`).
    /// Vincent is an `Admin` only, outside the projection of `Person`.
    #[test]
    fn nodes_by_label_holds_the_projected_nodes_with_any_label() {
        let store = Arc::new(RowGroupStore::new());
        let alix = store.create_node(&["Person", "Admin"]);
        let gus = store.create_node(&["Person"]);
        let vincent = store.create_node(&["Admin"]);
        let mia = store.create_node(&["Admin", "Person"]);
        let jules = store.create_node(&["City", "Admin"]);
        let spec = ProjectionSpec::new().with_node_labels(["Person", "City"]);
        let proj = GraphProjection::new(store, spec);

        assert_eq!(
            proj.nodes_by_label("Admin", ReadView::latest()).unwrap(),
            [alix, mia, jules]
        );
        assert_eq!(
            proj.nodes_by_label_count("Admin", ReadView::latest())
                .unwrap(),
            3
        );
        assert_eq!(
            proj.nodes_by_label("Person", ReadView::latest()).unwrap(),
            [alix, gus, mia]
        );
        assert_eq!(
            proj.nodes_by_label("Missing", ReadView::latest()).unwrap(),
            []
        );
        assert_eq!(
            proj.nodes_by_label_count("Missing", ReadView::latest())
                .unwrap(),
            0
        );
        assert!(
            proj.node(vincent, ReadView::latest()).unwrap().is_none(),
            "Vincent is outside"
        );
        for label in ["Admin", "Person", "City", "Missing"] {
            let with_label: Vec<NodeId> = proj
                .node_ids(ReadView::latest())
                .unwrap()
                .into_iter()
                .filter(|&id| {
                    proj.node(id, ReadView::latest())
                        .unwrap()
                        .is_some_and(|node| node.labels.iter().any(|l| l.as_str() == label))
                })
                .collect();
            assert_eq!(
                proj.nodes_by_label(label, ReadView::latest()).unwrap(),
                with_label,
                "{label}"
            );
            assert_eq!(
                proj.nodes_by_label_count(label, ReadView::latest())
                    .unwrap(),
                with_label.len(),
                "{label}"
            );
        }
        // The estimate of a label outside the spec is bounded by the nodes
        // of the spec's labels; the count reads the projection exactly.
        let statistics = proj.statistics();
        let projected = proj
            .nodes_by_label_count("Admin", ReadView::latest())
            .unwrap();
        assert!(statistics.estimate_label_cardinality("Admin") >= projected as f64);
    }

    #[test]
    fn filter_by_edge_type() {
        let store = setup_social_graph();
        let spec = ProjectionSpec::new().with_edge_types(["KNOWS"]);
        let proj = GraphProjection::new(store, spec);

        // All nodes visible (no label filter), but only KNOWS edges
        assert_eq!(proj.node_count(ReadView::latest()).unwrap(), 4);
        assert_eq!(proj.edge_count(ReadView::latest()).unwrap(), 1);
    }

    #[test]
    fn combined_label_and_edge_filter() {
        let store = setup_social_graph();
        let spec = ProjectionSpec::new()
            .with_node_labels(["Person", "City"])
            .with_edge_types(["LIVES_IN"]);
        let proj = GraphProjection::new(store, spec);

        assert_eq!(proj.node_count(ReadView::latest()).unwrap(), 3); // 2 Person + 1 City
        assert_eq!(proj.edge_count(ReadView::latest()).unwrap(), 2); // 2 LIVES_IN edges
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

        assert_eq!(proj.node_count(ReadView::latest()).unwrap(), 2);
        // LIVES_IN edges should be excluded because City endpoints are filtered out
        assert_eq!(proj.edge_count(ReadView::latest()).unwrap(), 0);
    }

    #[test]
    fn get_node_filtered() {
        let store = setup_social_graph();
        let all_ids = store.node_ids(ReadView::latest()).unwrap();
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let proj = GraphProjection::new(store.clone(), spec);

        // Person nodes visible
        assert!(proj.node(all_ids[0], ReadView::latest()).unwrap().is_some()); // Alix (Person)
        assert!(proj.node(all_ids[1], ReadView::latest()).unwrap().is_some()); // Gus (Person)
        // City and Software nodes hidden
        assert!(proj.node(all_ids[2], ReadView::latest()).unwrap().is_none()); // Amsterdam (City)
        assert!(proj.node(all_ids[3], ReadView::latest()).unwrap().is_none()); // Grafeo (Software)
    }

    #[test]
    fn neighbors_filtered() {
        let store = setup_social_graph();
        let alix_id = store.node_ids(ReadView::latest()).unwrap()[0];

        // Without projection: Alix has 3 outgoing neighbors (Gus, Amsterdam, Grafeo)
        let all_neighbors: Vec<_> = store
            .neighbors(alix_id, Direction::Outgoing, ReadView::latest())
            .unwrap();
        assert_eq!(all_neighbors.len(), 3);

        // With Person-only projection: Alix -> Gus only
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let proj = GraphProjection::new(store, spec);
        let neighbors = proj
            .neighbors(alix_id, Direction::Outgoing, ReadView::latest())
            .unwrap();
        assert_eq!(neighbors.len(), 1);
    }

    #[test]
    fn neighbors_filtered_by_edge_type() {
        let store = setup_social_graph();
        let alix_id = store.node_ids(ReadView::latest()).unwrap()[0];

        // With edge-type filter: only KNOWS edges visible
        // Alix KNOWS Gus, but LIVES_IN Amsterdam and CONTRIBUTES_TO Grafeo are excluded
        let spec = ProjectionSpec::new().with_edge_types(["KNOWS"]);
        let proj = GraphProjection::new(store, spec);
        let neighbors = proj
            .neighbors(alix_id, Direction::Outgoing, ReadView::latest())
            .unwrap();
        assert_eq!(neighbors.len(), 1);
    }

    #[test]
    fn property_access_respects_filter() {
        let store = setup_social_graph();
        let city_id = store.node_ids(ReadView::latest()).unwrap()[2]; // Amsterdam
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let proj = GraphProjection::new(store, spec);

        // City node properties are inaccessible
        assert!(
            proj.node_property(city_id, "name", ReadView::latest())
                .unwrap()
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

        let statistics = proj.statistics();
        assert_eq!(statistics.estimate_avg_degree("LIVES_IN", true), 0.0);
        assert!(
            statistics.estimate_label_cardinality("City")
                <= statistics.estimate_label_cardinality("Person")
        );
    }

    #[test]
    fn schema_introspection_reflects_filter() {
        let store = setup_social_graph();
        let spec = ProjectionSpec::new()
            .with_node_labels(["Person"])
            .with_edge_types(["KNOWS"]);
        let proj = GraphProjection::new(store, spec);

        let labels = proj.all_labels().unwrap();
        assert_eq!(labels.len(), 1);
        assert!(labels.contains(&"Person".to_string()));

        let edge_types = proj.all_edge_types().unwrap();
        assert_eq!(edge_types.len(), 1);
        assert!(edge_types.contains(&"KNOWS".to_string()));
    }

    /// Helper: returns (node_ids, edge_ids) from the social graph.
    fn setup_social_graph_with_ids() -> (Arc<RowGroupStore>, Vec<NodeId>, Vec<EdgeId>) {
        let store = Arc::new(RowGroupStore::new());
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
        assert!(proj.edge(edges[0], ReadView::latest()).unwrap().is_some());
        // LIVES_IN edge does not pass filter
        assert!(proj.edge(edges[1], ReadView::latest()).unwrap().is_none());
        // CONTRIBUTES_TO edge does not pass filter
        assert!(proj.edge(edges[3], ReadView::latest()).unwrap().is_none());
    }

    #[test]
    fn get_edge_excluded_by_endpoint_label_filter() {
        let (store, _, edges) = setup_social_graph_with_ids();
        // Only Person nodes, LIVES_IN goes Person->City, so excluded
        let spec = ProjectionSpec::new()
            .with_node_labels(["Person"])
            .with_edge_types(["LIVES_IN"]);
        let proj = GraphProjection::new(store, spec);

        assert!(proj.edge(edges[1], ReadView::latest()).unwrap().is_none()); // Alix->Amsterdam
        assert!(proj.edge(edges[2], ReadView::latest()).unwrap().is_none()); // Gus->Amsterdam
    }

    // 2. get_node_versioned and get_edge_versioned

    #[test]
    fn a_transaction_view_of_a_node_respects_the_filter() {
        let (store, nodes, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let epoch = store.current_epoch();
        let proj = GraphProjection::new(store, spec);
        let txn = TransactionId(0);

        // Person node visible
        assert!(
            proj.node(nodes[0], ReadView::transaction(txn, epoch))
                .unwrap()
                .is_some()
        );
        // City node filtered out
        assert!(
            proj.node(nodes[2], ReadView::transaction(txn, epoch))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn a_transaction_view_of_an_edge_respects_the_filter() {
        let (store, _, edges) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_edge_types(["KNOWS"]);
        let epoch = store.current_epoch();
        let proj = GraphProjection::new(store, spec);
        let txn = TransactionId(0);

        // KNOWS edge visible
        assert!(
            proj.edge(edges[0], ReadView::transaction(txn, epoch))
                .unwrap()
                .is_some()
        );
        // LIVES_IN edge filtered out
        assert!(
            proj.edge(edges[1], ReadView::transaction(txn, epoch))
                .unwrap()
                .is_none()
        );
    }

    // 3. get_node_at_epoch and get_edge_at_epoch

    #[test]
    fn get_node_at_epoch_respects_filter() {
        let (store, nodes, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_node_labels(["City"]);
        let epoch = store.current_epoch();
        let proj = GraphProjection::new(store, spec);

        // Amsterdam (City) visible
        assert!(
            proj.node(nodes[2], ReadView::committed(epoch))
                .unwrap()
                .is_some()
        );
        // Alix (Person) filtered out
        assert!(
            proj.node(nodes[0], ReadView::committed(epoch))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn get_edge_at_epoch_respects_filter() {
        let (store, _, edges) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_edge_types(["LIVES_IN"]);
        let epoch = store.current_epoch();
        let proj = GraphProjection::new(store, spec);

        // LIVES_IN edge visible
        assert!(
            proj.edge(edges[1], ReadView::committed(epoch))
                .unwrap()
                .is_some()
        );
        // KNOWS edge filtered out
        assert!(
            proj.edge(edges[0], ReadView::committed(epoch))
                .unwrap()
                .is_none()
        );
    }

    /// The reads at a view go to the inner store's, filtered by the spec: a
    /// transaction finds its own new node through every lookup, another
    /// reader finds none of it; the label filter reads the labels at the
    /// view, so the projection answers exactly when the inner store does.
    #[test]
    fn reads_at_a_view_forward_to_the_inner_store() {
        use crate::graph::GraphStore as _;
        use crate::graph::conformance::{Tx, people};
        use crate::graph::rowgroup::RowGroupStore;

        let store = Arc::new(RowGroupStore::new());
        let people = people(store.as_ref()).unwrap();
        let mut tx = Tx::begin(store.as_ref(), 10);
        let jules = tx
            .create_node(
                &["Person"],
                &[("name", Value::from("Jules")), ("age", Value::Int64(40))],
            )
            .unwrap();
        tx.create_edge(jules, people.alix, "KNOWS", &[]).unwrap();
        let own = ReadView::transaction(tx.id, tx.snapshot);
        let committed = ReadView::committed(store.current_epoch());
        let all = GraphProjection::new(
            Arc::clone(&store) as Arc<dyn GraphStore>,
            ProjectionSpec::new(),
        );
        let persons = GraphProjection::new(
            Arc::clone(&store) as Arc<dyn GraphStore>,
            ProjectionSpec::new().with_node_labels(["Person"]),
        );

        let name = Value::from("Jules");
        let age = Value::Int64(40);
        for proj in [&all, &persons] {
            assert_eq!(
                proj.nodes_by_label("Person", own).unwrap().last(),
                Some(&jules)
            );
            assert!(
                !proj
                    .nodes_by_label("Person", committed)
                    .unwrap()
                    .contains(&jules)
            );
            assert_eq!(
                proj.nodes_with_any_label(&["Person", "City"], own)
                    .unwrap()
                    .last(),
                Some(&jules)
            );
            assert_eq!(
                proj.find_nodes_by_property("name", &name, own).unwrap(),
                [jules]
            );
            assert!(
                proj.find_nodes_by_property("name", &name, committed)
                    .unwrap()
                    .is_empty(),
                "another reader does not find Jules"
            );
            assert_eq!(
                proj.find_nodes_by_properties(&[("name", name.clone()), ("age", age.clone())], own)
                    .unwrap(),
                [jules]
            );
            assert_eq!(
                proj.find_nodes_in_range("age", Some(&age), None, true, false, own)
                    .unwrap(),
                [jules]
            );
            assert_eq!(
                proj.edges(jules, Direction::Outgoing, &[], own)
                    .unwrap()
                    .len(),
                1
            );
            assert_eq!(
                proj.neighbors(people.alix, Direction::Incoming, own)
                    .unwrap(),
                [jules]
            );
            assert!(
                proj.neighbors(people.alix, Direction::Incoming, committed)
                    .unwrap()
                    .is_empty(),
                "another reader sees no edge to Alix"
            );
        }
    }

    #[test]
    fn visibility_at_epoch_respects_the_filters_and_deletes() {
        let (store, nodes, edges) = setup_social_graph_with_ids();
        let people = GraphProjection::new(
            Arc::clone(&store) as Arc<dyn GraphStore>,
            ProjectionSpec::new().with_node_labels(["Person"]),
        );
        let knows = GraphProjection::new(
            Arc::clone(&store) as Arc<dyn GraphStore>,
            ProjectionSpec::new().with_edge_types(["KNOWS"]),
        );
        let epoch = store.current_epoch();

        assert!(
            people
                .is_node_visible(nodes[0], ReadView::committed(epoch))
                .unwrap()
        );
        assert!(
            !people
                .is_node_visible(nodes[2], ReadView::committed(epoch))
                .unwrap(),
            "a City"
        );
        assert!(
            people
                .is_edge_visible(edges[0], ReadView::committed(epoch))
                .unwrap(),
            "KNOWS"
        );
        assert!(
            !people
                .is_edge_visible(edges[1], ReadView::committed(epoch))
                .unwrap(),
            "to a City"
        );
        assert!(
            knows
                .is_edge_visible(edges[0], ReadView::committed(epoch))
                .unwrap()
        );
        assert!(
            !knows
                .is_edge_visible(edges[1], ReadView::committed(epoch))
                .unwrap(),
            "LIVES_IN"
        );

        assert!(store.delete_edge(edges[0]));
        store.delete_node_edges(nodes[1]);
        assert!(store.delete_node(nodes[1]));
        let epoch = store.current_epoch();
        assert!(
            !people
                .is_node_visible(nodes[1], ReadView::committed(epoch))
                .unwrap(),
            "deleted"
        );
        assert!(
            !knows
                .is_edge_visible(edges[0], ReadView::committed(epoch))
                .unwrap(),
            "deleted"
        );
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
            proj.edge_property(edges[0], key.as_str(), ReadView::latest())
                .unwrap(),
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
        assert!(
            proj.edge_property(edges[1], key.as_str(), ReadView::latest())
                .unwrap()
                .is_none()
        );
    }

    // 5. node_property_batch for mixed in/out of projection nodes

    /// An inner store whose label answer for Alix changes between calls (a
    /// concurrent label write): `City` on the first read, `Person` after.
    /// Gus is always a `Person`. Every node's `name` reads as its name.
    struct RelabelingStore {
        alix: NodeId,
        gus: NodeId,
        alix_reads: std::sync::atomic::AtomicUsize,
    }

    impl GraphStore for RelabelingStore {
        fn node(&self, id: NodeId, _: ReadView) -> Result<Option<Node>> {
            let label = if id == self.alix {
                let reads = self
                    .alix_reads
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if reads == 0 { "City" } else { "Person" }
            } else if id == self.gus {
                "Person"
            } else {
                return Ok(None);
            };
            Ok(Some(Node::with_labels(id, [label])))
        }
        fn edge(&self, _: EdgeId, _: ReadView) -> Result<Option<Edge>> {
            Ok(None)
        }
        fn node_property(&self, id: NodeId, _: &str, _: ReadView) -> Result<Option<Value>> {
            Ok(if id == self.alix {
                Some(Value::from("Alix"))
            } else if id == self.gus {
                Some(Value::from("Gus"))
            } else {
                None
            })
        }
        fn edges(
            &self,
            _: NodeId,
            _: Direction,
            _: &[ArcStr],
            _: ReadView,
        ) -> Result<Vec<(NodeId, EdgeId)>> {
            Ok(Vec::new())
        }
        fn has_backward_adjacency(&self) -> bool {
            false
        }
        fn node_ids(&self, _: ReadView) -> Result<Vec<NodeId>> {
            Ok(vec![self.alix, self.gus])
        }
        fn all_labels(&self) -> Result<Vec<String>> {
            Ok(vec!["City".into(), "Person".into()])
        }
        fn all_edge_types(&self) -> Result<Vec<String>> {
            Ok(Vec::new())
        }
        fn all_property_keys(&self) -> Result<Vec<String>> {
            Ok(vec!["name".into()])
        }
        fn statistics(&self) -> Arc<Statistics> {
            Arc::new(Statistics::default())
        }
        fn committed_epoch(&self) -> EpochId {
            EpochId::INITIAL
        }
    }

    /// The fallible batch decides once which nodes are in the projection: a
    /// label that changes between two reads of it never hands one node's
    /// value to another. Alix is outside when the batch looks, so she has no
    /// value, and Gus keeps his own.
    #[test]
    fn node_property_batch_never_hands_a_value_to_another_node() {
        let (alix, gus) = (NodeId(3), NodeId(19));
        let inner = Arc::new(RelabelingStore {
            alix,
            gus,
            alix_reads: std::sync::atomic::AtomicUsize::new(0),
        });
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let proj = GraphProjection::new(inner, spec);

        let values = proj
            .node_property_batch(&[alix, gus], &PropertyKey::from("name"), ReadView::latest())
            .unwrap();
        assert_eq!(
            values,
            vec![None, Some(Value::from("Gus"))],
            "Alix was a City when the batch looked, and Gus's name is his"
        );
    }

    #[test]
    fn node_property_batch_mixed() {
        let (store, nodes, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let proj = GraphProjection::new(store, spec);

        let key = PropertyKey::from("name");
        // Alix (Person), Amsterdam (City), Gus (Person)
        let ids = vec![nodes[0], nodes[2], nodes[1]];
        let results = proj
            .node_property_batch(&ids, &key, ReadView::latest())
            .unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0], Some(Value::from("Alix"))); // Person: visible
        assert_eq!(results[1], None); // City: filtered out
        assert_eq!(results[2], Some(Value::from("Gus"))); // Person: visible
    }

    // 6. node_properties_batch and selective batch

    #[test]
    fn node_properties_batch_filters() {
        let (store, nodes, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let proj = GraphProjection::new(store, spec);

        let ids = vec![nodes[0], nodes[2]]; // Alix (Person), Amsterdam (City)
        let results = proj
            .node_properties_batch(&ids, ReadView::latest())
            .unwrap();

        assert_eq!(results.len(), 2);
        // Alix has properties
        assert!(results[0].contains_key(&PropertyKey::from("name")));
        // Amsterdam filtered out, empty map
        assert!(results[1].is_empty());
    }

    #[test]
    fn node_properties_selective_batch_filters() {
        let (store, nodes, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let proj = GraphProjection::new(store, spec);

        let ids = vec![nodes[0], nodes[2]]; // Alix (Person), Amsterdam (City)
        let keys = vec![PropertyKey::from("name")];
        let results = proj
            .node_properties_selective_batch(&ids, &keys, ReadView::latest())
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].get(&PropertyKey::from("name")),
            Some(&Value::from("Alix"))
        );
        assert!(results[1].is_empty());
    }

    // 7. edge_properties_selective_batch

    #[test]
    fn edge_properties_selective_batch_filters() {
        let (store, _, edges) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_edge_types(["KNOWS"]);
        let proj = GraphProjection::new(store, spec);

        let ids = vec![edges[0], edges[1]]; // KNOWS, LIVES_IN
        let keys = vec![PropertyKey::from("since")];
        let results = proj
            .edge_properties_selective_batch(&ids, &keys, ReadView::latest())
            .unwrap();

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
        let alix_edges = proj
            .edges(nodes[0], Direction::Outgoing, &[], ReadView::latest())
            .unwrap();
        assert_eq!(alix_edges.len(), 1);
        assert_eq!(alix_edges[0].0, nodes[2]); // target is Amsterdam
    }

    #[test]
    fn edges_from_filtered_node_returns_empty() {
        let (store, nodes, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let proj = GraphProjection::new(store, spec);

        // Amsterdam (City) is filtered out, so edges_from returns empty
        let amsterdam_edges = proj
            .edges(nodes[2], Direction::Outgoing, &[], ReadView::latest())
            .unwrap();
        assert!(amsterdam_edges.is_empty(), "{amsterdam_edges:?}");
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
        assert_eq!(
            proj.degree(
                nodes[0],
                crate::graph::Direction::Outgoing,
                ReadView::latest()
            )
            .unwrap(),
            1
        );
        // Gus has 1 outgoing LIVES_IN to Amsterdam
        assert_eq!(
            proj.degree(
                nodes[1],
                crate::graph::Direction::Outgoing,
                ReadView::latest()
            )
            .unwrap(),
            1
        );
        // Amsterdam has no outgoing LIVES_IN
        assert_eq!(
            proj.degree(
                nodes[2],
                crate::graph::Direction::Outgoing,
                ReadView::latest()
            )
            .unwrap(),
            0
        );
    }

    #[test]
    fn in_degree_with_filter() {
        let (store, nodes, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new()
            .with_node_labels(["Person", "City"])
            .with_edge_types(["LIVES_IN"]);
        let proj = GraphProjection::new(store, spec);

        // Amsterdam has 2 incoming LIVES_IN edges
        assert_eq!(
            proj.degree(
                nodes[2],
                crate::graph::Direction::Incoming,
                ReadView::latest()
            )
            .unwrap(),
            2
        );
        // Alix has 0 incoming LIVES_IN
        assert_eq!(
            proj.degree(
                nodes[0],
                crate::graph::Direction::Incoming,
                ReadView::latest()
            )
            .unwrap(),
            0
        );
    }

    // 10. all_node_ids

    #[test]
    fn all_node_ids_with_label_filter() {
        let (store, nodes, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let proj = GraphProjection::new(store, spec);

        let ids = proj.node_ids(ReadView::latest()).unwrap();
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

        assert_eq!(
            proj.node_ids(ReadView::latest()).unwrap().len(),
            store.node_ids(ReadView::latest()).unwrap().len()
        );
    }

    // 11. node_count and edge_count with various filters

    #[test]
    fn node_count_with_city_filter() {
        let (store, _, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_node_labels(["City"]);
        let proj = GraphProjection::new(store, spec);

        assert_eq!(proj.node_count(ReadView::latest()).unwrap(), 1);
    }

    #[test]
    fn edge_count_with_combined_filter() {
        let (store, _, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new()
            .with_node_labels(["Person"])
            .with_edge_types(["KNOWS"]);
        let proj = GraphProjection::new(store, spec);

        // Only KNOWS between Person nodes: Alix->Gus
        assert_eq!(proj.edge_count(ReadView::latest()).unwrap(), 1);
    }

    #[test]
    fn edge_count_unfiltered_delegates() {
        let (store, _, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new();
        let proj = GraphProjection::new(store.clone(), spec);

        assert_eq!(
            proj.edge_count(ReadView::latest()).unwrap(),
            store.edge_count(ReadView::latest()).unwrap()
        );
    }

    // 12. find_nodes_by_property with label filter

    #[test]
    fn find_nodes_by_property_with_label_filter() {
        let (store, nodes, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let proj = GraphProjection::new(store, spec);

        // "name" = "Alix" exists on a Person node
        let found = proj
            .find_nodes_by_property("name", &Value::from("Alix"), ReadView::latest())
            .unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0], nodes[0]);

        // "name" = "Amsterdam" exists but on a City node, which is filtered
        let found = proj
            .find_nodes_by_property("name", &Value::from("Amsterdam"), ReadView::latest())
            .unwrap();
        assert!(found.is_empty(), "{found:?}");
    }

    // 13. find_nodes_by_properties with label filter

    #[test]
    fn find_nodes_by_properties_with_label_filter() {
        let (store, nodes, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let proj = GraphProjection::new(store, spec);

        let conditions = vec![("name", Value::from("Gus"))];
        let found = proj
            .find_nodes_by_properties(&conditions, ReadView::latest())
            .unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0], nodes[1]);

        // Search for city name, filtered out
        let conditions = vec![("name", Value::from("Amsterdam"))];
        let found = proj
            .find_nodes_by_properties(&conditions, ReadView::latest())
            .unwrap();
        assert!(found.is_empty(), "{found:?}");
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
        let found = proj
            .find_nodes_in_range(
                "age",
                Some(&min),
                Some(&max),
                true,
                true,
                ReadView::latest(),
            )
            .unwrap();
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
        let found = proj
            .find_nodes_in_range(
                "age",
                Some(&min),
                Some(&max),
                true,
                true,
                ReadView::latest(),
            )
            .unwrap();
        assert!(found.is_empty(), "{found:?}");
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

    // 17. all_property_keys

    #[test]
    fn all_property_keys_delegates() {
        let (store, _, _) = setup_social_graph_with_ids();
        let spec = ProjectionSpec::new().with_node_labels(["Person"]);
        let proj = GraphProjection::new(store.clone(), spec);

        let proj_keys = proj.all_property_keys().unwrap();
        let store_keys = store.all_property_keys().unwrap();
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

        // RowGroupStore default has no property indexes
        assert_eq!(
            proj.has_property_index("name"),
            store.has_property_index("name")
        );
    }
}
