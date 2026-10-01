//! Adapter that wraps an [`RdfStore`] and implements [`GraphStore`].
//!
//! This bridges the gap between the RDF triple model and the LPG-oriented
//! algorithm infrastructure. Once constructed, the adapter can be passed to
//! any function that accepts `&dyn GraphStore`, giving RDF graphs access to
//! all 20+ graph algorithms (PageRank, BFS, k-core, triangle counting, etc.).
//!
//! # Mapping
//!
//! | RDF concept | LPG concept |
//! | ----------- | ----------- |
//! | IRI / blank node (as subject or object) | Node |
//! | Triple (subject, predicate, object) | Edge (src, dst, type = predicate local name) |
//! | `rdf:type` triple | Label on the subject node |
//! | Literal-valued triple | Property on the subject node |
//!
//! Only IRI and blank node terms become graph nodes. Literal objects are
//! stored as properties on the subject node, keyed by the predicate's local
//! name.
//!
//! # Example
//!
//! ```
//! use grafeo_core::graph::rdf::{RdfStore, Term, Triple, RdfGraphStoreAdapter};
//! use grafeo_core::graph::GraphStore;
//!
//! let store = RdfStore::new();
//! store.insert(Triple::new(
//!     Term::iri("http://example.org/alix"),
//!     Term::iri("http://xmlns.com/foaf/0.1/knows"),
//!     Term::iri("http://example.org/gus"),
//! ));
//!
//! let adapter = RdfGraphStoreAdapter::new(&store);
//! assert_eq!(adapter.node_count(), 2);
//! assert_eq!(adapter.edge_count(), 1);
//! ```

use super::store::RdfStore;
use super::term::{Literal, Term};
use super::triple::Triple;
use crate::graph::Direction;
use crate::graph::lpg::CompareOp;
use crate::graph::lpg::{Edge, Node};
use crate::graph::traits::{GraphStore, GraphStoreSearch};
use crate::statistics::{EdgeTypeStatistics, LabelStatistics, Statistics};
use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::hash::FxHashMap;
use smallvec::SmallVec;
use std::sync::Arc;

/// Well-known `rdf:type` predicate IRI.
const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

/// An adapter that presents an [`RdfStore`] as a [`GraphStore`].
///
/// Built by scanning all triples once and constructing pre-built adjacency
/// lists, node/edge mappings, and property tables. The adapter is immutable
/// after construction: it represents a snapshot of the `RdfStore` at the
/// time `new()` was called.
///
/// For algorithms that only use traversal methods (`neighbors`, `edges_from`,
/// `node_ids`, `out_degree`), the adapter has the same asymptotic performance
/// as [`LpgStore`](crate::graph::lpg::LpgStore): adjacency lookups are O(1)
/// indexing into pre-built Vecs.
pub struct RdfGraphStoreAdapter {
    /// Term -> NodeId mapping (only IRIs and blank nodes).
    term_to_node: FxHashMap<Term, NodeId>,
    /// NodeId -> Term mapping (indexed by NodeId.as_u64()).
    node_to_term: Vec<Term>,
    /// Labels per node (from rdf:type triples).
    node_labels: Vec<SmallVec<[ArcStr; 2]>>,
    /// Properties per node (from literal-object triples, keyed by predicate local name).
    node_properties: Vec<FxHashMap<PropertyKey, Value>>,
    /// Edge data: (src NodeId, dst NodeId, edge type ArcStr).
    edge_data: Vec<(NodeId, NodeId, ArcStr)>,
    /// Pre-built outgoing adjacency: node index -> Vec<(target NodeId, EdgeId)>.
    outgoing: Vec<Vec<(NodeId, EdgeId)>>,
    /// Pre-built incoming adjacency: node index -> Vec<(target NodeId, EdgeId)>.
    incoming: Vec<Vec<(NodeId, EdgeId)>>,
    /// Cached statistics.
    statistics: Arc<Statistics>,
}

impl RdfGraphStoreAdapter {
    /// Creates a new adapter by scanning all triples in the store.
    ///
    /// This performs a single pass over all triples to build node/edge
    /// mappings, adjacency lists, labels, and properties.
    pub fn new(store: &RdfStore) -> Self {
        let triples = store.triples();
        Self::from_triples(&triples)
    }

    /// Creates an adapter from a pre-collected slice of triples.
    ///
    /// Useful when you already hold the triples and want to avoid re-acquiring
    /// the lock on `RdfStore`.
    ///
    /// # Panics
    ///
    /// Panics if a triple's predicate is not an IRI.
    pub fn from_triples(triples: &[Arc<Triple>]) -> Self {
        let mut term_to_node: FxHashMap<Term, NodeId> = FxHashMap::default();
        let mut node_to_term: Vec<Term> = Vec::new();

        // Helper: get or create a NodeId for a non-literal term.
        let get_or_create_node =
            |term: &Term, map: &mut FxHashMap<Term, NodeId>, terms: &mut Vec<Term>| -> NodeId {
                if let Some(&id) = map.get(term) {
                    return id;
                }
                let id = NodeId::new(terms.len() as u64);
                map.insert(term.clone(), id);
                terms.push(term.clone());
                id
            };

        // Phase 1: Assign NodeIds to all non-literal subjects and objects.
        // Also separate triples into structural edges, rdf:type (labels),
        // and literal-object (properties).
        struct TripleClassification {
            src: NodeId,
            dst: NodeId,
            predicate_local: ArcStr,
        }

        let mut edges: Vec<TripleClassification> = Vec::new();
        let mut rdf_type_triples: Vec<(NodeId, ArcStr)> = Vec::new();
        let mut literal_triples: Vec<(NodeId, ArcStr, Value)> = Vec::new();

        for triple in triples {
            let subject = triple.subject();
            let predicate = triple.predicate();
            let object = triple.object();

            // Subject is always IRI or blank node.
            let src = get_or_create_node(subject, &mut term_to_node, &mut node_to_term);

            let predicate_iri = predicate.as_iri().expect("predicate must be an IRI");
            let predicate_str: ArcStr = ArcStr::from(predicate_iri.local_name());

            match object {
                Term::Literal(lit) => {
                    // Literal object -> property on the subject node.
                    let value = literal_to_value(lit);
                    literal_triples.push((src, predicate_str, value));
                }
                _ => {
                    // Check if this is an rdf:type triple.
                    if predicate_iri.as_str() == RDF_TYPE {
                        if let Some(type_iri) = object.as_iri() {
                            rdf_type_triples.push((src, ArcStr::from(type_iri.local_name())));
                        } else {
                            // rdf:type with blank node object: treat as label with blank node id.
                            if let Some(bn) = object.as_blank_node() {
                                rdf_type_triples.push((src, ArcStr::from(bn.id())));
                            }
                        }
                    } else {
                        let dst = get_or_create_node(object, &mut term_to_node, &mut node_to_term);
                        edges.push(TripleClassification {
                            src,
                            dst,
                            predicate_local: predicate_str,
                        });
                    }
                }
            }
        }

        let node_count = node_to_term.len();

        // Phase 2: Build labels per node.
        let mut node_labels: Vec<SmallVec<[ArcStr; 2]>> = vec![SmallVec::new(); node_count];
        for (node_id, label) in &rdf_type_triples {
            // reason: node IDs are sequential indices into Vec, fit usize
            #[allow(clippy::cast_possible_truncation)]
            let idx = node_id.as_u64() as usize;
            if !node_labels[idx].contains(label) {
                node_labels[idx].push(label.clone());
            }
        }

        // Phase 3: Build properties per node.
        let mut node_properties: Vec<FxHashMap<PropertyKey, Value>> =
            vec![FxHashMap::default(); node_count];
        for (node_id, key, value) in literal_triples {
            // reason: node IDs are sequential indices into Vec, fit usize
            #[allow(clippy::cast_possible_truncation)]
            let idx = node_id.as_u64() as usize;
            // If multiple triples with same predicate, last wins.
            node_properties[idx].insert(PropertyKey::from(key.as_str()), value);
        }

        // Phase 4: Build edge data and adjacency lists.
        let mut edge_data: Vec<(NodeId, NodeId, ArcStr)> = Vec::with_capacity(edges.len());
        let mut outgoing: Vec<Vec<(NodeId, EdgeId)>> = vec![Vec::new(); node_count];
        let mut incoming: Vec<Vec<(NodeId, EdgeId)>> = vec![Vec::new(); node_count];

        for classified in edges {
            let edge_id = EdgeId::new(edge_data.len() as u64);
            edge_data.push((classified.src, classified.dst, classified.predicate_local));
            // reason: node IDs are sequential indices into Vec, fit usize
            #[allow(clippy::cast_possible_truncation)]
            let src_idx = classified.src.as_u64() as usize;
            // reason: node IDs are sequential indices into Vec, fit usize
            #[allow(clippy::cast_possible_truncation)]
            let dst_idx = classified.dst.as_u64() as usize;
            outgoing[src_idx].push((classified.dst, edge_id));
            incoming[dst_idx].push((classified.src, edge_id));
        }

        // Phase 5: Build statistics.
        let statistics = build_statistics(node_count, &edge_data, &node_labels);

        Self {
            term_to_node,
            node_to_term,
            node_labels,
            node_properties,
            edge_data,
            outgoing,
            incoming,
            statistics: Arc::new(statistics),
        }
    }

    /// Returns the number of nodes in the adapted graph.
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.node_to_term.len()
    }

    /// Returns the number of edges in the adapted graph.
    #[must_use]
    pub fn num_edges(&self) -> usize {
        self.edge_data.len()
    }

    /// Looks up the `NodeId` for an RDF term, if it exists in the mapping.
    #[must_use]
    pub fn node_id_for_term(&self, term: &Term) -> Option<NodeId> {
        self.term_to_node.get(term).copied()
    }

    /// Returns the RDF term for a given `NodeId`.
    #[must_use]
    pub fn term_for_node_id(&self, id: NodeId) -> Option<&Term> {
        // reason: node IDs are sequential indices into Vec, fit usize
        #[allow(clippy::cast_possible_truncation)]
        // reason: entity IDs are sequential indices into Vec, fit usize
        #[allow(clippy::cast_possible_truncation)]
        let idx = id.as_u64() as usize;
        self.node_to_term.get(idx)
    }
}

impl GraphStore for RdfGraphStoreAdapter {
    // --- Point lookups ---

    fn get_node(&self, id: NodeId) -> Option<Node> {
        // reason: entity IDs are sequential indices into Vec, fit usize
        #[allow(clippy::cast_possible_truncation)]
        let idx = id.as_u64() as usize;
        if idx >= self.node_to_term.len() {
            return None;
        }
        let mut node = Node::new(id);
        node.labels.clone_from(&self.node_labels[idx]);
        for (key, value) in &self.node_properties[idx] {
            node.properties.insert(key.clone(), value.clone());
        }
        Some(node)
    }

    fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        // reason: entity IDs are sequential indices into Vec, fit usize
        #[allow(clippy::cast_possible_truncation)]
        let idx = id.as_u64() as usize;
        let (src, dst, edge_type) = self.edge_data.get(idx)?;
        Some(Edge {
            id,
            src: *src,
            dst: *dst,
            edge_type: edge_type.clone(),
            properties: grafeo_common::types::PropertyMap::new(),
        })
    }

    fn get_node_versioned(
        &self,
        id: NodeId,
        _epoch: EpochId,
        _transaction_id: TransactionId,
    ) -> Option<Node> {
        self.get_node(id)
    }

    fn get_edge_versioned(
        &self,
        id: EdgeId,
        _epoch: EpochId,
        _transaction_id: TransactionId,
    ) -> Option<Edge> {
        self.get_edge(id)
    }

    fn get_node_at_epoch(&self, id: NodeId, _epoch: EpochId) -> Option<Node> {
        self.get_node(id)
    }

    fn get_edge_at_epoch(&self, id: EdgeId, _epoch: EpochId) -> Option<Edge> {
        self.get_edge(id)
    }

    // --- Property access ---

    fn get_node_property(&self, id: NodeId, key: &PropertyKey) -> Option<Value> {
        // reason: entity IDs are sequential indices into Vec, fit usize
        #[allow(clippy::cast_possible_truncation)]
        let idx = id.as_u64() as usize;
        self.node_properties.get(idx)?.get(key).cloned()
    }

    fn get_edge_property(&self, _id: EdgeId, _key: &PropertyKey) -> Option<Value> {
        // RDF edges (triples) don't carry properties in this mapping.
        None
    }

    fn get_node_property_batch(&self, ids: &[NodeId], key: &PropertyKey) -> Vec<Option<Value>> {
        ids.iter()
            .map(|id| self.get_node_property(*id, key))
            .collect()
    }

    fn get_nodes_properties_batch(&self, ids: &[NodeId]) -> Vec<FxHashMap<PropertyKey, Value>> {
        ids.iter()
            .map(|id| {
                // reason: entity IDs are sequential indices into Vec, fit usize
                #[allow(clippy::cast_possible_truncation)]
                let idx = id.as_u64() as usize;
                self.node_properties.get(idx).cloned().unwrap_or_default()
            })
            .collect()
    }

    fn get_nodes_properties_selective_batch(
        &self,
        ids: &[NodeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        ids.iter()
            .map(|id| {
                // reason: entity IDs are sequential indices into Vec, fit usize
                #[allow(clippy::cast_possible_truncation)]
                let idx = id.as_u64() as usize;
                let mut result = FxHashMap::default();
                if let Some(props) = self.node_properties.get(idx) {
                    for key in keys {
                        if let Some(value) = props.get(key) {
                            result.insert(key.clone(), value.clone());
                        }
                    }
                }
                result
            })
            .collect()
    }

    fn get_edges_properties_selective_batch(
        &self,
        ids: &[EdgeId],
        _keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        vec![FxHashMap::default(); ids.len()]
    }

    // --- Traversal (hot path for algorithms) ---

    fn neighbors(&self, node: NodeId, direction: Direction) -> Vec<NodeId> {
        // reason: node IDs are sequential indices into Vec, fit usize
        #[allow(clippy::cast_possible_truncation)]
        let idx = node.as_u64() as usize;
        match direction {
            Direction::Outgoing => self
                .outgoing
                .get(idx)
                .map(|adj| adj.iter().map(|(target, _)| *target).collect())
                .unwrap_or_default(),
            Direction::Incoming => self
                .incoming
                .get(idx)
                .map(|adj| adj.iter().map(|(target, _)| *target).collect())
                .unwrap_or_default(),
            Direction::Both => {
                let mut result: Vec<NodeId> = Vec::new();
                if let Some(out) = self.outgoing.get(idx) {
                    result.extend(out.iter().map(|(t, _)| *t));
                }
                if let Some(inc) = self.incoming.get(idx) {
                    result.extend(inc.iter().map(|(t, _)| *t));
                }
                result
            }
        }
    }

    fn edges_from(&self, node: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)> {
        // reason: node IDs are sequential indices into Vec, fit usize
        #[allow(clippy::cast_possible_truncation)]
        let idx = node.as_u64() as usize;
        match direction {
            Direction::Outgoing => self.outgoing.get(idx).cloned().unwrap_or_default(),
            Direction::Incoming => self.incoming.get(idx).cloned().unwrap_or_default(),
            Direction::Both => {
                let mut result = Vec::new();
                if let Some(out) = self.outgoing.get(idx) {
                    result.extend_from_slice(out);
                }
                if let Some(inc) = self.incoming.get(idx) {
                    result.extend_from_slice(inc);
                }
                result
            }
        }
    }

    fn out_degree(&self, node: NodeId) -> usize {
        // reason: node IDs are sequential indices into Vec, fit usize
        #[allow(clippy::cast_possible_truncation)]
        let idx = node.as_u64() as usize;
        self.outgoing.get(idx).map_or(0, Vec::len)
    }

    fn in_degree(&self, node: NodeId) -> usize {
        // reason: node IDs are sequential indices into Vec, fit usize
        #[allow(clippy::cast_possible_truncation)]
        let idx = node.as_u64() as usize;
        self.incoming.get(idx).map_or(0, Vec::len)
    }

    fn has_backward_adjacency(&self) -> bool {
        true
    }

    // --- Scans ---

    fn node_ids(&self) -> Vec<NodeId> {
        (0..self.node_to_term.len() as u64)
            .map(NodeId::new)
            .collect()
    }

    fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        let label_arc = ArcStr::from(label);
        self.node_labels
            .iter()
            .enumerate()
            .filter(|(_, labels)| labels.contains(&label_arc))
            .map(|(idx, _)| NodeId::new(idx as u64))
            .collect()
    }

    fn nodes_by_label_count(&self, label: &str) -> usize {
        let label_arc = ArcStr::from(label);
        self.node_labels
            .iter()
            .filter(|labels| labels.contains(&label_arc))
            .count()
    }

    fn node_count(&self) -> usize {
        self.node_to_term.len()
    }

    fn edge_count(&self) -> usize {
        self.edge_data.len()
    }

    // --- Entity metadata ---

    fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
        self.edge_data
            // reason: edge IDs are sequential indices into Vec, fit usize
            // reason: edge IDs are sequential indices into Vec, fit usize
            .get({
                // reason: edge IDs are sequential indices into Vec, fit usize
                #[allow(clippy::cast_possible_truncation)]
                let idx = id.as_u64() as usize;
                idx
            })
            .map(|(_, _, t)| t.clone())
    }

    /// A read-only view without versions: every edge is the committed one.
    fn edge_type_versioned(&self, id: EdgeId, _: EpochId, _: TransactionId) -> Option<ArcStr> {
        self.edge_type(id)
    }

    // --- Filtered search ---

    fn find_nodes_by_property(&self, property: &str, value: &Value) -> Vec<NodeId> {
        let key = PropertyKey::from(property);
        self.node_properties
            .iter()
            .enumerate()
            .filter(|(_, props)| props.get(&key) == Some(value))
            .map(|(idx, _)| NodeId::new(idx as u64))
            .collect()
    }

    fn find_nodes_by_properties(&self, conditions: &[(&str, Value)]) -> Vec<NodeId> {
        self.node_properties
            .iter()
            .enumerate()
            .filter(|(_, props)| {
                conditions.iter().all(|(key, value)| {
                    let pk = PropertyKey::from(*key);
                    props.get(&pk) == Some(value)
                })
            })
            .map(|(idx, _)| NodeId::new(idx as u64))
            .collect()
    }

    fn find_nodes_in_range(
        &self,
        _property: &str,
        _min: Option<&Value>,
        _max: Option<&Value>,
        _min_inclusive: bool,
        _max_inclusive: bool,
    ) -> Vec<NodeId> {
        // Range queries not supported on the adapter; return empty.
        Vec::new()
    }

    // --- Zone maps ---

    fn node_property_might_match(
        &self,
        _property: &PropertyKey,
        _op: CompareOp,
        _value: &Value,
    ) -> bool {
        true // Conservative: always might match.
    }

    fn edge_property_might_match(
        &self,
        _property: &PropertyKey,
        _op: CompareOp,
        _value: &Value,
    ) -> bool {
        true
    }

    // --- Statistics ---

    fn statistics(&self) -> Arc<Statistics> {
        Arc::clone(&self.statistics)
    }

    fn estimate_label_cardinality(&self, label: &str) -> f64 {
        self.statistics.estimate_label_cardinality(label)
    }

    fn estimate_avg_degree(&self, edge_type: &str, outgoing: bool) -> f64 {
        self.statistics.estimate_avg_degree(edge_type, outgoing)
    }

    // --- Epoch ---

    fn current_epoch(&self) -> EpochId {
        EpochId::INITIAL
    }

    // --- Schema introspection ---

    fn all_labels(&self) -> Vec<String> {
        let mut labels = grafeo_common::utils::hash::FxHashSet::default();
        for node_labels in &self.node_labels {
            for label in node_labels {
                labels.insert(label.to_string());
            }
        }
        labels.into_iter().collect()
    }

    fn all_edge_types(&self) -> Vec<String> {
        let mut types = grafeo_common::utils::hash::FxHashSet::default();
        for (_, _, edge_type) in &self.edge_data {
            types.insert(edge_type.to_string());
        }
        types.into_iter().collect()
    }

    fn all_property_keys(&self) -> Vec<String> {
        let mut keys = grafeo_common::utils::hash::FxHashSet::default();
        for props in &self.node_properties {
            for key in props.keys() {
                keys.insert(key.to_string());
            }
        }
        keys.into_iter().collect()
    }
}

impl GraphStoreSearch for RdfGraphStoreAdapter {}

/// Converts an RDF literal to a Grafeo [`Value`].
fn literal_to_value(lit: &Literal) -> Value {
    match lit.datatype() {
        Literal::XSD_INTEGER => lit
            .as_integer()
            .map_or_else(|| Value::String(lit.value().into()), Value::Int64),
        Literal::XSD_DOUBLE | Literal::XSD_DECIMAL => lit
            .as_double()
            .map_or_else(|| Value::String(lit.value().into()), Value::Float64),
        Literal::XSD_BOOLEAN => lit
            .as_boolean()
            .map_or_else(|| Value::String(lit.value().into()), Value::Bool),
        _ => Value::String(lit.value().into()),
    }
}

/// Builds [`Statistics`] from the adapter's node/edge data.
fn build_statistics(
    node_count: usize,
    edge_data: &[(NodeId, NodeId, ArcStr)],
    node_labels: &[SmallVec<[ArcStr; 2]>],
) -> Statistics {
    let mut stats = Statistics::new();
    stats.total_nodes = node_count as u64;
    stats.total_edges = edge_data.len() as u64;

    // Per-label stats.
    let mut label_counts: FxHashMap<ArcStr, u64> = FxHashMap::default();
    for labels in node_labels {
        for label in labels {
            *label_counts.entry(label.clone()).or_default() += 1;
        }
    }
    for (label, count) in &label_counts {
        stats.update_label(label, LabelStatistics::new(*count));
    }

    // Per-edge-type stats.
    let mut edge_type_counts: FxHashMap<ArcStr, u64> = FxHashMap::default();
    let mut edge_type_sources: FxHashMap<ArcStr, grafeo_common::utils::hash::FxHashSet<u64>> =
        FxHashMap::default();
    let mut edge_type_targets: FxHashMap<ArcStr, grafeo_common::utils::hash::FxHashSet<u64>> =
        FxHashMap::default();

    for (src, dst, edge_type) in edge_data {
        *edge_type_counts.entry(edge_type.clone()).or_default() += 1;
        edge_type_sources
            .entry(edge_type.clone())
            .or_default()
            .insert(src.as_u64());
        edge_type_targets
            .entry(edge_type.clone())
            .or_default()
            .insert(dst.as_u64());
    }

    for (edge_type, count) in &edge_type_counts {
        let source_count = edge_type_sources
            .get(edge_type)
            .map_or(1, |s| s.len().max(1));
        let target_count = edge_type_targets
            .get(edge_type)
            .map_or(1, |s| s.len().max(1));
        let avg_out = *count as f64 / source_count as f64;
        let avg_in = *count as f64 / target_count as f64;
        stats.update_edge_type(edge_type, EdgeTypeStatistics::new(*count, avg_out, avg_in));
    }

    stats
}

// Compile-time assertion: adapter must be Send + Sync (required by GraphStore).
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<RdfGraphStoreAdapter>();
};

#[cfg(test)]
mod tests {
    use super::*;

    fn foaf(local: &str) -> Term {
        Term::iri(format!("http://xmlns.com/foaf/0.1/{local}"))
    }

    fn ex(local: &str) -> Term {
        Term::iri(format!("http://example.org/{local}"))
    }

    fn rdf_type() -> Term {
        Term::iri(RDF_TYPE)
    }

    #[test]
    fn test_empty_store() {
        let store = RdfStore::new();
        let adapter = RdfGraphStoreAdapter::new(&store);
        assert_eq!(adapter.node_count(), 0);
        assert_eq!(adapter.edge_count(), 0);
        assert!(adapter.node_ids().is_empty());
    }

    #[test]
    fn test_simple_edge() {
        let store = RdfStore::new();
        store.insert(Triple::new(ex("alix"), foaf("knows"), ex("gus")));

        let adapter = RdfGraphStoreAdapter::new(&store);
        assert_eq!(adapter.node_count(), 2);
        assert_eq!(adapter.edge_count(), 1);

        let alix = adapter.node_id_for_term(&ex("alix")).unwrap();
        let gus = adapter.node_id_for_term(&ex("gus")).unwrap();

        // Outgoing from Alix
        let out = adapter.neighbors(alix, Direction::Outgoing);
        assert_eq!(out, vec![gus]);

        // Incoming to Gus
        let inc = adapter.neighbors(gus, Direction::Incoming);
        assert_eq!(inc, vec![alix]);

        // No outgoing from Gus
        assert!(adapter.neighbors(gus, Direction::Outgoing).is_empty());
    }

    #[test]
    fn test_rdf_type_becomes_label() {
        let store = RdfStore::new();
        store.insert(Triple::new(ex("alix"), rdf_type(), foaf("Person")));
        store.insert(Triple::new(ex("alix"), foaf("knows"), ex("gus")));

        let adapter = RdfGraphStoreAdapter::new(&store);
        let alix = adapter.node_id_for_term(&ex("alix")).unwrap();
        let node = adapter.get_node(alix).unwrap();

        assert!(node.labels.iter().any(|l| l.as_str() == "Person"));
        // rdf:type triple should NOT create an edge
        assert_eq!(adapter.edge_count(), 1);
    }

    #[test]
    fn test_literal_becomes_property() {
        let store = RdfStore::new();
        store.insert(Triple::new(ex("alix"), foaf("name"), Term::literal("Alix")));
        store.insert(Triple::new(
            ex("alix"),
            Term::iri("http://example.org/age"),
            Term::typed_literal("19", Literal::XSD_INTEGER),
        ));
        store.insert(Triple::new(ex("alix"), foaf("knows"), ex("gus")));

        let adapter = RdfGraphStoreAdapter::new(&store);
        let alix = adapter.node_id_for_term(&ex("alix")).unwrap();

        // Literal triples become properties, not edges or nodes
        assert_eq!(adapter.node_count(), 2); // alix and gus only
        assert_eq!(adapter.edge_count(), 1); // knows only

        let name = adapter.get_node_property(alix, &PropertyKey::from("name"));
        assert_eq!(name, Some(Value::String("Alix".into())));

        let age = adapter.get_node_property(alix, &PropertyKey::from("age"));
        assert_eq!(age, Some(Value::Int64(19)));
    }

    #[test]
    fn test_triangle_graph() {
        let store = RdfStore::new();
        store.insert(Triple::new(ex("a"), ex("knows"), ex("b")));
        store.insert(Triple::new(ex("b"), ex("knows"), ex("c")));
        store.insert(Triple::new(ex("c"), ex("knows"), ex("a")));

        let adapter = RdfGraphStoreAdapter::new(&store);
        assert_eq!(adapter.node_count(), 3);
        assert_eq!(adapter.edge_count(), 3);

        // Each node has out-degree 1 and in-degree 1
        for node_id in adapter.node_ids() {
            assert_eq!(adapter.out_degree(node_id), 1);
            assert_eq!(adapter.in_degree(node_id), 1);
        }
    }

    #[test]
    fn test_nodes_by_label() {
        let store = RdfStore::new();
        store.insert(Triple::new(ex("alix"), rdf_type(), foaf("Person")));
        store.insert(Triple::new(ex("gus"), rdf_type(), foaf("Person")));
        store.insert(Triple::new(ex("amsterdam"), rdf_type(), ex("City")));
        // Add structural edges so all nodes appear
        store.insert(Triple::new(ex("alix"), ex("livesIn"), ex("amsterdam")));
        store.insert(Triple::new(ex("gus"), ex("livesIn"), ex("amsterdam")));

        let adapter = RdfGraphStoreAdapter::new(&store);

        let people = adapter.nodes_by_label("Person");
        assert_eq!(people.len(), 2);

        let cities = adapter.nodes_by_label("City");
        assert_eq!(cities.len(), 1);
    }

    #[test]
    fn test_get_edge_returns_correct_data() {
        let store = RdfStore::new();
        store.insert(Triple::new(ex("alix"), foaf("knows"), ex("gus")));

        let adapter = RdfGraphStoreAdapter::new(&store);
        let edge = adapter.get_edge(EdgeId::new(0)).unwrap();

        let alix = adapter.node_id_for_term(&ex("alix")).unwrap();
        let gus = adapter.node_id_for_term(&ex("gus")).unwrap();

        assert_eq!(edge.src, alix);
        assert_eq!(edge.dst, gus);
        assert_eq!(edge.edge_type.as_str(), "knows");
    }

    #[test]
    fn test_edge_type_query() {
        let store = RdfStore::new();
        store.insert(Triple::new(ex("alix"), foaf("knows"), ex("gus")));

        let adapter = RdfGraphStoreAdapter::new(&store);
        let edge_type = adapter.edge_type(EdgeId::new(0));
        assert_eq!(edge_type.as_deref(), Some("knows"));
    }

    #[test]
    fn test_backward_adjacency() {
        let store = RdfStore::new();
        store.insert(Triple::new(ex("alix"), foaf("knows"), ex("gus")));

        let adapter = RdfGraphStoreAdapter::new(&store);
        assert!(adapter.has_backward_adjacency());

        let gus = adapter.node_id_for_term(&ex("gus")).unwrap();
        let incoming = adapter.edges_from(gus, Direction::Incoming);
        assert_eq!(incoming.len(), 1);
    }

    #[test]
    fn test_both_direction() {
        let store = RdfStore::new();
        store.insert(Triple::new(ex("a"), ex("knows"), ex("b")));
        store.insert(Triple::new(ex("c"), ex("knows"), ex("a")));

        let adapter = RdfGraphStoreAdapter::new(&store);
        let a = adapter.node_id_for_term(&ex("a")).unwrap();

        let both = adapter.neighbors(a, Direction::Both);
        assert_eq!(both.len(), 2); // b (outgoing) + c (incoming)
    }

    #[test]
    fn test_statistics() {
        let store = RdfStore::new();
        store.insert(Triple::new(ex("a"), ex("knows"), ex("b")));
        store.insert(Triple::new(ex("b"), ex("knows"), ex("c")));

        let adapter = RdfGraphStoreAdapter::new(&store);
        let stats = adapter.statistics();
        assert_eq!(stats.total_nodes, 3);
        assert_eq!(stats.total_edges, 2);
        assert!(stats.edge_types.contains_key("knows"));
    }

    #[test]
    fn test_all_schema_introspection() {
        let store = RdfStore::new();
        store.insert(Triple::new(ex("alix"), rdf_type(), foaf("Person")));
        store.insert(Triple::new(ex("alix"), foaf("knows"), ex("gus")));
        store.insert(Triple::new(ex("alix"), foaf("name"), Term::literal("Alix")));

        let adapter = RdfGraphStoreAdapter::new(&store);

        let labels = adapter.all_labels();
        assert!(labels.contains(&"Person".to_string()));

        let edge_types = adapter.all_edge_types();
        assert!(edge_types.contains(&"knows".to_string()));

        let prop_keys = adapter.all_property_keys();
        assert!(prop_keys.contains(&"name".to_string()));
    }
}
