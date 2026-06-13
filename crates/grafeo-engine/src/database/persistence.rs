//! Persistence, snapshots, and data export for GrafeoDB.

#[cfg(any(feature = "wal", feature = "grafeo-file"))]
use std::path::Path;

#[cfg(any(feature = "vector-index", feature = "text-index"))]
use grafeo_common::grafeo_warn;
use grafeo_common::types::{EdgeId, EpochId, NodeId, Value};
use grafeo_common::utils::error::{Error, Result};
use hashbrown::HashSet;

use crate::config::Config;

#[cfg(feature = "wal")]
use grafeo_storage::wal::WalRecord;

use crate::catalog::{
    EdgeTypeDefinition, GraphTypeDefinition, NodeTypeDefinition, ProcedureDefinition,
};

/// Current snapshot version.
const SNAPSHOT_VERSION: u8 = 4;

/// Binary snapshot format (v4: graph data, named graphs, RDF, schema, index metadata,
/// and property version history for temporal support).
#[derive(serde::Serialize, serde::Deserialize)]
struct Snapshot {
    version: u8,
    nodes: Vec<SnapshotNode>,
    edges: Vec<SnapshotEdge>,
    named_graphs: Vec<NamedGraphSnapshot>,
    rdf_triples: Vec<SnapshotTriple>,
    rdf_named_graphs: Vec<RdfNamedGraphSnapshot>,
    schema: SnapshotSchema,
    indexes: SnapshotIndexes,
    /// Current store epoch at snapshot time (0 when temporal is disabled).
    epoch: u64,
}

/// Schema metadata within a snapshot.
#[derive(serde::Serialize, serde::Deserialize, Default)]
struct SnapshotSchema {
    node_types: Vec<NodeTypeDefinition>,
    edge_types: Vec<EdgeTypeDefinition>,
    graph_types: Vec<GraphTypeDefinition>,
    procedures: Vec<ProcedureDefinition>,
    schemas: Vec<String>,
    graph_type_bindings: Vec<(String, String)>,
}

/// Index metadata within a snapshot (definitions only, not index data).
#[derive(serde::Serialize, serde::Deserialize, Default)]
struct SnapshotIndexes {
    property_indexes: Vec<String>,
    vector_indexes: Vec<SnapshotVectorIndex>,
    text_indexes: Vec<SnapshotTextIndex>,
}

/// Vector index definition for snapshot persistence.
#[derive(serde::Serialize, serde::Deserialize)]
struct SnapshotVectorIndex {
    label: String,
    property: String,
    dimensions: usize,
    metric: grafeo_core::index::vector::DistanceMetric,
    m: usize,
    ef_construction: usize,
}

/// Text index definition for snapshot persistence.
#[derive(serde::Serialize, serde::Deserialize)]
struct SnapshotTextIndex {
    label: String,
    property: String,
}

/// A named graph partition within a v2 snapshot.
#[derive(serde::Serialize, serde::Deserialize)]
struct NamedGraphSnapshot {
    name: String,
    nodes: Vec<SnapshotNode>,
    edges: Vec<SnapshotEdge>,
}

/// An RDF triple in snapshot format (N-Triples encoded terms).
#[derive(serde::Serialize, serde::Deserialize)]
struct SnapshotTriple {
    subject: String,
    predicate: String,
    object: String,
}

/// An RDF named graph in snapshot format.
#[derive(serde::Serialize, serde::Deserialize)]
struct RdfNamedGraphSnapshot {
    name: String,
    triples: Vec<SnapshotTriple>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SnapshotNode {
    id: NodeId,
    labels: Vec<String>,
    /// Each property has a list of `(epoch, value)` entries (ascending epoch order).
    properties: Vec<(String, Vec<(EpochId, Value)>)>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SnapshotEdge {
    id: EdgeId,
    src: NodeId,
    dst: NodeId,
    edge_type: String,
    /// Each property has a list of `(epoch, value)` entries (ascending epoch order).
    properties: Vec<(String, Vec<(EpochId, Value)>)>,
}

/// Collects all nodes from a store into snapshot format.
///
/// With `temporal`: stores full property version history.
/// Without: wraps each current value as a single-entry version list at epoch 0.
fn collect_snapshot_nodes(store: &grafeo_core::graph::lpg::LpgStore) -> Vec<SnapshotNode> {
    let mut nodes: Vec<SnapshotNode> = store
        .all_nodes()
        .map(|n| {
            #[cfg(feature = "temporal")]
            let mut properties: Vec<(String, Vec<(EpochId, Value)>)> = store
                .node_property_history(n.id)
                .into_iter()
                .map(|(k, entries)| (k.to_string(), entries))
                .collect();

            #[cfg(not(feature = "temporal"))]
            let mut properties: Vec<(String, Vec<(EpochId, Value)>)> = n
                .properties
                .into_iter()
                .map(|(k, v)| (k.to_string(), vec![(EpochId::new(0), v)]))
                .collect();

            properties.sort_by(|(a, _), (b, _)| a.cmp(b));

            let mut labels: Vec<String> = n.labels.iter().map(|l| l.to_string()).collect();
            labels.sort();

            SnapshotNode {
                id: n.id,
                labels,
                properties,
            }
        })
        .collect();
    nodes.sort_by_key(|n| n.id);
    nodes
}

/// Collects all edges from a store into snapshot format.
///
/// With `temporal`: stores full property version history.
/// Without: wraps each current value as a single-entry version list at epoch 0.
fn collect_snapshot_edges(store: &grafeo_core::graph::lpg::LpgStore) -> Vec<SnapshotEdge> {
    let mut edges: Vec<SnapshotEdge> = store
        .all_edges()
        .map(|e| {
            #[cfg(feature = "temporal")]
            let mut properties: Vec<(String, Vec<(EpochId, Value)>)> = store
                .edge_property_history(e.id)
                .into_iter()
                .map(|(k, entries)| (k.to_string(), entries))
                .collect();

            #[cfg(not(feature = "temporal"))]
            let mut properties: Vec<(String, Vec<(EpochId, Value)>)> = e
                .properties
                .into_iter()
                .map(|(k, v)| (k.to_string(), vec![(EpochId::new(0), v)]))
                .collect();

            properties.sort_by(|(a, _), (b, _)| a.cmp(b));

            SnapshotEdge {
                id: e.id,
                src: e.src,
                dst: e.dst,
                edge_type: e.edge_type.to_string(),
                properties,
            }
        })
        .collect();
    edges.sort_by_key(|e| e.id);
    edges
}

/// Populates a store from snapshot node/edge data.
///
/// With `temporal`: replays all `(epoch, value)` entries into version logs.
/// Without: reads the latest value from each property's version list.
fn populate_store_from_snapshot(
    store: &grafeo_core::graph::lpg::LpgStore,
    nodes: Vec<SnapshotNode>,
    edges: Vec<SnapshotEdge>,
) -> Result<()> {
    for node in nodes {
        let label_refs: Vec<&str> = node.labels.iter().map(|s| s.as_str()).collect();
        store.create_node_with_id(node.id, &label_refs)?;
        for (key, entries) in node.properties {
            #[cfg(feature = "temporal")]
            for (epoch, value) in entries {
                store.set_node_property_at_epoch(node.id, &key, value, epoch);
            }
            #[cfg(not(feature = "temporal"))]
            if let Some((_, value)) = entries.into_iter().last() {
                store.set_node_property(node.id, &key, value);
            }
        }
    }
    for edge in edges {
        store.create_edge_with_id(edge.id, edge.src, edge.dst, &edge.edge_type)?;
        for (key, entries) in edge.properties {
            #[cfg(feature = "temporal")]
            for (epoch, value) in entries {
                store.set_edge_property_at_epoch(edge.id, &key, value, epoch);
            }
            #[cfg(not(feature = "temporal"))]
            if let Some((_, value)) = entries.into_iter().last() {
                store.set_edge_property(edge.id, &key, value);
            }
        }
    }
    Ok(())
}

/// Validates snapshot nodes/edges for duplicates and dangling references.
fn validate_snapshot_data(nodes: &[SnapshotNode], edges: &[SnapshotEdge]) -> Result<()> {
    let mut node_ids = HashSet::with_capacity(nodes.len());
    for node in nodes {
        if !node_ids.insert(node.id) {
            return Err(Error::Internal(format!(
                "snapshot contains duplicate node ID {}",
                node.id
            )));
        }
    }
    let mut edge_ids = HashSet::with_capacity(edges.len());
    for edge in edges {
        if !edge_ids.insert(edge.id) {
            return Err(Error::Internal(format!(
                "snapshot contains duplicate edge ID {}",
                edge.id
            )));
        }
        if !node_ids.contains(&edge.src) {
            return Err(Error::Internal(format!(
                "snapshot edge {} references non-existent source node {}",
                edge.id, edge.src
            )));
        }
        if !node_ids.contains(&edge.dst) {
            return Err(Error::Internal(format!(
                "snapshot edge {} references non-existent destination node {}",
                edge.id, edge.dst
            )));
        }
    }
    Ok(())
}

/// Collects all triples from an RDF store into snapshot format.
#[cfg(feature = "triple-store")]
fn collect_rdf_triples(store: &grafeo_core::graph::rdf::RdfStore) -> Vec<SnapshotTriple> {
    store
        .triples()
        .into_iter()
        .map(|t| SnapshotTriple {
            subject: t.subject().to_string(),
            predicate: t.predicate().to_string(),
            object: t.object().to_string(),
        })
        .collect()
}

/// Populates an RDF store from snapshot triples.
#[cfg(feature = "triple-store")]
fn populate_rdf_store(store: &grafeo_core::graph::rdf::RdfStore, triples: &[SnapshotTriple]) {
    use grafeo_core::graph::rdf::{Term, Triple};
    for triple in triples {
        if let (Some(s), Some(p), Some(o)) = (
            Term::from_ntriples(&triple.subject),
            Term::from_ntriples(&triple.predicate),
            Term::from_ntriples(&triple.object),
        ) {
            store.insert(Triple::new(s, p, o));
        }
    }
}

// =========================================================================
// Snapshot deserialization helpers (used by single-file format)
// =========================================================================

/// Decodes snapshot bytes and populates a store and catalog.
#[cfg(feature = "grafeo-file")]
pub(super) fn load_snapshot_into_store(
    store: &std::sync::Arc<grafeo_core::graph::lpg::LpgStore>,
    catalog: &std::sync::Arc<crate::catalog::Catalog>,
    #[cfg(feature = "triple-store")] rdf_store: &std::sync::Arc<grafeo_core::graph::rdf::RdfStore>,
    data: &[u8],
) -> grafeo_common::utils::error::Result<()> {
    use grafeo_common::utils::error::Error;

    let config = bincode::config::standard();
    let (snapshot, _) =
        bincode::serde::decode_from_slice::<Snapshot, _>(data, config).map_err(|e| {
            Error::Serialization(format!("failed to decode snapshot from .grafeo file: {e}"))
        })?;

    populate_store_from_snapshot_ref(store, &snapshot.nodes, &snapshot.edges)?;

    // Restore epoch from snapshot (store-level only; TransactionManager
    // sync is handled in with_config() after all recovery completes).
    #[cfg(feature = "temporal")]
    store.sync_epoch(EpochId::new(snapshot.epoch));

    for graph in &snapshot.named_graphs {
        store
            .create_graph(&graph.name)
            .map_err(|e| Error::Internal(e.to_string()))?;
        if let Some(graph_store) = store.graph(&graph.name) {
            populate_store_from_snapshot_ref(&graph_store, &graph.nodes, &graph.edges)?;
            #[cfg(feature = "temporal")]
            graph_store.sync_epoch(EpochId::new(snapshot.epoch));
        }
    }
    restore_schema_from_snapshot(store, catalog, &snapshot.schema);

    // Restore RDF triples
    #[cfg(feature = "triple-store")]
    {
        populate_rdf_store(rdf_store, &snapshot.rdf_triples);
        for rdf_graph in &snapshot.rdf_named_graphs {
            rdf_store.create_graph(&rdf_graph.name);
            if let Some(graph_store) = rdf_store.graph(&rdf_graph.name) {
                populate_rdf_store(&graph_store, &rdf_graph.triples);
            }
        }
    }

    Ok(())
}

/// Populates a store from snapshot refs (borrowed, for single-file loading).
#[cfg(feature = "grafeo-file")]
fn populate_store_from_snapshot_ref(
    store: &grafeo_core::graph::lpg::LpgStore,
    nodes: &[SnapshotNode],
    edges: &[SnapshotEdge],
) -> grafeo_common::utils::error::Result<()> {
    for node in nodes {
        let label_refs: Vec<&str> = node.labels.iter().map(|s| s.as_str()).collect();
        store.create_node_with_id(node.id, &label_refs)?;
        for (key, entries) in &node.properties {
            #[cfg(feature = "temporal")]
            for (epoch, value) in entries {
                store.set_node_property_at_epoch(node.id, key, value.clone(), *epoch);
            }
            #[cfg(not(feature = "temporal"))]
            if let Some((_, value)) = entries.last() {
                store.set_node_property(node.id, key, value.clone());
            }
        }
    }
    for edge in edges {
        store.create_edge_with_id(edge.id, edge.src, edge.dst, &edge.edge_type)?;
        for (key, entries) in &edge.properties {
            #[cfg(feature = "temporal")]
            for (epoch, value) in entries {
                store.set_edge_property_at_epoch(edge.id, key, value.clone(), *epoch);
            }
            #[cfg(not(feature = "temporal"))]
            if let Some((_, value)) = entries.last() {
                store.set_edge_property(edge.id, key, value.clone());
            }
        }
    }
    Ok(())
}

/// Restores schema definitions from a snapshot into the catalog.
///
/// Also ensures each schema has its `__default__` graph partition, which
/// may be missing in snapshots created before the schema hierarchy feature.
fn restore_schema_from_snapshot(
    store: &std::sync::Arc<grafeo_core::graph::lpg::LpgStore>,
    catalog: &std::sync::Arc<crate::catalog::Catalog>,
    schema: &SnapshotSchema,
) {
    for def in &schema.node_types {
        catalog.register_or_replace_node_type(def.clone());
    }
    for def in &schema.edge_types {
        catalog.register_or_replace_edge_type_def(def.clone());
    }
    for def in &schema.graph_types {
        let _ = catalog.register_graph_type(def.clone());
    }
    for def in &schema.procedures {
        catalog.replace_procedure(def.clone()).ok();
    }
    for name in &schema.schemas {
        let _ = catalog.register_schema_namespace(name.clone());
        // Ensure the schema's default graph partition exists
        let default_key = format!("{name}/__default__");
        let _ = store.create_graph(&default_key);
    }
    for (graph_name, type_name) in &schema.graph_type_bindings {
        let _ = catalog.bind_graph_type(graph_name, type_name.clone());
    }
}

/// Collects schema definitions from the catalog into snapshot format.
fn collect_schema(catalog: &std::sync::Arc<crate::catalog::Catalog>) -> SnapshotSchema {
    SnapshotSchema {
        node_types: catalog.all_node_type_defs(),
        edge_types: catalog.all_edge_type_defs(),
        graph_types: catalog.all_graph_type_defs(),
        procedures: catalog.all_procedure_defs(),
        schemas: catalog.schema_names(),
        graph_type_bindings: catalog.all_graph_type_bindings(),
    }
}

/// Restores indexes from snapshot metadata by rebuilding them from existing data.
///
/// Must be called after all nodes/edges have been populated, since index
/// creation scans existing data.
fn restore_indexes_from_snapshot(db: &super::GrafeoDB, indexes: &SnapshotIndexes) {
    for name in &indexes.property_indexes {
        db.lpg_store().create_property_index(name);
    }

    #[cfg(feature = "vector-index")]
    for vi in &indexes.vector_indexes {
        if let Err(err) = db.create_vector_index(
            &vi.label,
            &vi.property,
            Some(vi.dimensions),
            Some(vi.metric.name()),
            Some(vi.m),
            Some(vi.ef_construction),
            None,
        ) {
            grafeo_warn!(
                "Failed to restore vector index :{label}({property}): {err}",
                label = vi.label,
                property = vi.property,
            );
        }
    }

    #[cfg(feature = "text-index")]
    for ti in &indexes.text_indexes {
        if let Err(err) = db.create_text_index(&ti.label, &ti.property) {
            grafeo_warn!(
                "Failed to restore text index :{label}({property}): {err}",
                label = ti.label,
                property = ti.property,
            );
        }
    }
}

/// Collects index metadata from a store into snapshot format.
fn collect_index_metadata(store: &grafeo_core::graph::lpg::LpgStore) -> SnapshotIndexes {
    let property_indexes = store.property_index_keys();

    #[cfg(feature = "vector-index")]
    let vector_indexes: Vec<SnapshotVectorIndex> = store
        .vector_index_entries()
        .into_iter()
        .filter_map(|(key, index)| {
            let (label, property) = key.split_once(':')?;
            let config = index.config();
            Some(SnapshotVectorIndex {
                label: label.to_string(),
                property: property.to_string(),
                dimensions: config.dimensions,
                metric: config.metric,
                m: config.m,
                ef_construction: config.ef_construction,
            })
        })
        .collect();
    #[cfg(not(feature = "vector-index"))]
    let vector_indexes = Vec::new();

    #[cfg(feature = "text-index")]
    let text_indexes: Vec<SnapshotTextIndex> = store
        .text_index_entries()
        .into_iter()
        .filter_map(|(key, _)| {
            let (label, property) = key.split_once(':')?;
            Some(SnapshotTextIndex {
                label: label.to_string(),
                property: property.to_string(),
            })
        })
        .collect();
    #[cfg(not(feature = "text-index"))]
    let text_indexes = Vec::new();

    SnapshotIndexes {
        property_indexes,
        vector_indexes,
        text_indexes,
    }
}

impl super::GrafeoDB {
    // =========================================================================
    // ADMIN API: Persistence Control
    // =========================================================================

    /// Saves the database to a file path.
    ///
    /// - If the path ends in `.grafeo`: creates a single-file database
    /// - Otherwise: creates a WAL directory-backed database at the path
    /// - If in-memory: creates a new persistent database at path
    /// - If file-backed: creates a copy at the new path
    ///
    /// The original database remains unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error if the save operation fails.
    ///
    /// Requires the `wal` feature for persistence support.
    #[cfg(feature = "wal")]
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();

        // Single-file format: export snapshot directly to a .grafeo file
        #[cfg(feature = "grafeo-file")]
        if path.extension().is_some_and(|ext| ext == "grafeo") {
            return self.save_as_grafeo_file(path);
        }

        // Create target database with WAL enabled
        let target_config = Config::persistent(path);
        let target = Self::with_config(target_config)?;

        // Copy all nodes using WAL-enabled methods
        for node in self.lpg_store().all_nodes() {
            let label_refs: Vec<&str> = node.labels.iter().map(|s| &**s).collect();
            target
                .lpg_store()
                .create_node_with_id(node.id, &label_refs)?;

            // Log to WAL
            target.log_wal(&WalRecord::CreateNode {
                id: node.id,
                labels: node.labels.iter().map(|s| s.to_string()).collect(),
            })?;

            // Copy properties
            for (key, value) in node.properties {
                target
                    .lpg_store()
                    .set_node_property(node.id, key.as_str(), value.clone());
                target.log_wal(&WalRecord::SetNodeProperty {
                    id: node.id,
                    key: key.to_string(),
                    value,
                })?;
            }
        }

        // Copy all edges using WAL-enabled methods
        for edge in self.lpg_store().all_edges() {
            target
                .lpg_store()
                .create_edge_with_id(edge.id, edge.src, edge.dst, &edge.edge_type)?;

            // Log to WAL
            target.log_wal(&WalRecord::CreateEdge {
                id: edge.id,
                src: edge.src,
                dst: edge.dst,
                edge_type: edge.edge_type.to_string(),
            })?;

            // Copy properties
            for (key, value) in edge.properties {
                target
                    .lpg_store()
                    .set_edge_property(edge.id, key.as_str(), value.clone());
                target.log_wal(&WalRecord::SetEdgeProperty {
                    id: edge.id,
                    key: key.to_string(),
                    value,
                })?;
            }
        }

        // Copy named graphs
        for graph_name in self.lpg_store().graph_names() {
            if let Some(src_graph) = self.lpg_store().graph(&graph_name) {
                target.log_wal(&WalRecord::CreateNamedGraph {
                    name: graph_name.clone(),
                })?;
                target
                    .lpg_store()
                    .create_graph(&graph_name)
                    .map_err(|e| Error::Internal(e.to_string()))?;

                if let Some(dst_graph) = target.lpg_store().graph(&graph_name) {
                    // Switch WAL context to this named graph
                    target.log_wal(&WalRecord::SwitchGraph {
                        name: Some(graph_name.clone()),
                    })?;

                    for node in src_graph.all_nodes() {
                        let label_refs: Vec<&str> = node.labels.iter().map(|s| &**s).collect();
                        dst_graph.create_node_with_id(node.id, &label_refs)?;
                        target.log_wal(&WalRecord::CreateNode {
                            id: node.id,
                            labels: node.labels.iter().map(|s| s.to_string()).collect(),
                        })?;
                        for (key, value) in node.properties {
                            dst_graph.set_node_property(node.id, key.as_str(), value.clone());
                            target.log_wal(&WalRecord::SetNodeProperty {
                                id: node.id,
                                key: key.to_string(),
                                value,
                            })?;
                        }
                    }
                    for edge in src_graph.all_edges() {
                        dst_graph.create_edge_with_id(
                            edge.id,
                            edge.src,
                            edge.dst,
                            &edge.edge_type,
                        )?;
                        target.log_wal(&WalRecord::CreateEdge {
                            id: edge.id,
                            src: edge.src,
                            dst: edge.dst,
                            edge_type: edge.edge_type.to_string(),
                        })?;
                        for (key, value) in edge.properties {
                            dst_graph.set_edge_property(edge.id, key.as_str(), value.clone());
                            target.log_wal(&WalRecord::SetEdgeProperty {
                                id: edge.id,
                                key: key.to_string(),
                                value,
                            })?;
                        }
                    }
                }
            }
        }

        // Switch WAL context back to default graph
        if !self.lpg_store().graph_names().is_empty() {
            target.log_wal(&WalRecord::SwitchGraph { name: None })?;
        }

        // Copy RDF data with WAL logging
        #[cfg(feature = "triple-store")]
        {
            for triple in self.rdf_store.triples() {
                let record = WalRecord::InsertRdfTriple {
                    subject: triple.subject().to_string(),
                    predicate: triple.predicate().to_string(),
                    object: triple.object().to_string(),
                    graph: None,
                };
                target.rdf_store.insert((*triple).clone());
                target.log_wal(&record)?;
            }
            for name in self.rdf_store.graph_names() {
                target.log_wal(&WalRecord::CreateRdfGraph { name: name.clone() })?;
                if let Some(src_graph) = self.rdf_store.graph(&name) {
                    let dst_graph = target.rdf_store.graph_or_create(&name);
                    for triple in src_graph.triples() {
                        let record = WalRecord::InsertRdfTriple {
                            subject: triple.subject().to_string(),
                            predicate: triple.predicate().to_string(),
                            object: triple.object().to_string(),
                            graph: Some(name.clone()),
                        };
                        dst_graph.insert((*triple).clone());
                        target.log_wal(&record)?;
                    }
                }
            }
        }

        // Checkpoint and close the target database
        target.close()?;

        Ok(())
    }

    /// Creates an in-memory copy of this database.
    ///
    /// Returns a new database that is completely independent, including
    /// all named graph data.
    /// Useful for:
    /// Saves the database to a single `.grafeo` file.
    #[cfg(feature = "grafeo-file")]
    fn save_as_grafeo_file(&self, path: &Path) -> Result<()> {
        use grafeo_storage::file::GrafeoFileManager;

        let snapshot_data = self.export_snapshot()?;
        let epoch = self.lpg_store().current_epoch();
        let transaction_id = self
            .transaction_manager
            .last_assigned_transaction_id()
            .map_or(0, |t| t.0);
        let node_count = self.lpg_store().node_count() as u64;
        let edge_count = self.lpg_store().edge_count() as u64;

        let fm = GrafeoFileManager::create(path)?;
        fm.write_snapshot(
            &snapshot_data,
            epoch.0,
            transaction_id,
            node_count,
            edge_count,
        )?;
        Ok(())
    }

    /// - Testing modifications without affecting the original
    /// - Faster operations when persistence isn't needed
    ///
    /// # Errors
    ///
    /// Returns an error if the copy operation fails.
    pub fn to_memory(&self) -> Result<Self> {
        let config = Config::in_memory();
        let target = Self::with_config(config)?;

        // Copy default graph nodes
        for node in self.lpg_store().all_nodes() {
            let label_refs: Vec<&str> = node.labels.iter().map(|s| &**s).collect();
            target
                .lpg_store()
                .create_node_with_id(node.id, &label_refs)?;
            for (key, value) in node.properties {
                target
                    .lpg_store()
                    .set_node_property(node.id, key.as_str(), value);
            }
        }

        // Copy default graph edges
        for edge in self.lpg_store().all_edges() {
            target
                .lpg_store()
                .create_edge_with_id(edge.id, edge.src, edge.dst, &edge.edge_type)?;
            for (key, value) in edge.properties {
                target
                    .lpg_store()
                    .set_edge_property(edge.id, key.as_str(), value);
            }
        }

        // Copy named graphs
        for graph_name in self.lpg_store().graph_names() {
            if let Some(src_graph) = self.lpg_store().graph(&graph_name) {
                target
                    .lpg_store()
                    .create_graph(&graph_name)
                    .map_err(|e| Error::Internal(e.to_string()))?;
                if let Some(dst_graph) = target.lpg_store().graph(&graph_name) {
                    for node in src_graph.all_nodes() {
                        let label_refs: Vec<&str> = node.labels.iter().map(|s| &**s).collect();
                        dst_graph.create_node_with_id(node.id, &label_refs)?;
                        for (key, value) in node.properties {
                            dst_graph.set_node_property(node.id, key.as_str(), value);
                        }
                    }
                    for edge in src_graph.all_edges() {
                        dst_graph.create_edge_with_id(
                            edge.id,
                            edge.src,
                            edge.dst,
                            &edge.edge_type,
                        )?;
                        for (key, value) in edge.properties {
                            dst_graph.set_edge_property(edge.id, key.as_str(), value);
                        }
                    }
                }
            }
        }

        // Copy RDF data
        #[cfg(feature = "triple-store")]
        {
            for triple in self.rdf_store.triples() {
                target.rdf_store.insert((*triple).clone());
            }
            for name in self.rdf_store.graph_names() {
                if let Some(src_graph) = self.rdf_store.graph(&name) {
                    let dst_graph = target.rdf_store.graph_or_create(&name);
                    for triple in src_graph.triples() {
                        dst_graph.insert((*triple).clone());
                    }
                }
            }
        }

        Ok(target)
    }

    /// Opens a database file and loads it entirely into memory.
    ///
    /// The returned database has no connection to the original file.
    /// Changes will NOT be written back to the file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file can't be opened or loaded.
    #[cfg(feature = "wal")]
    pub fn open_in_memory(path: impl AsRef<Path>) -> Result<Self> {
        // Open the source database (triggers WAL recovery)
        let source = Self::open(path)?;

        // Create in-memory copy
        let target = source.to_memory()?;

        // Close the source (releases file handles)
        source.close()?;

        Ok(target)
    }

    // =========================================================================
    // ADMIN API: Snapshot Export/Import
    // =========================================================================

    /// Exports the entire database to a binary snapshot.
    ///
    /// The returned bytes can be stored (e.g. in IndexedDB) and later
    /// restored with [`import_snapshot()`](Self::import_snapshot).
    /// Includes all named graph data.
    ///
    /// Properties are stored as version-history lists. When `temporal` is
    /// enabled, the full history is captured. Otherwise, each property is
    /// wrapped as a single-entry list at epoch 0.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn export_snapshot(&self) -> Result<Vec<u8>> {
        let nodes = collect_snapshot_nodes(self.lpg_store());
        let edges = collect_snapshot_edges(self.lpg_store());

        // Collect named graphs
        let named_graphs: Vec<NamedGraphSnapshot> = self
            .lpg_store()
            .graph_names()
            .into_iter()
            .filter_map(|name| {
                self.lpg_store()
                    .graph(&name)
                    .map(|graph_store| NamedGraphSnapshot {
                        name,
                        nodes: collect_snapshot_nodes(&graph_store),
                        edges: collect_snapshot_edges(&graph_store),
                    })
            })
            .collect();

        // Collect RDF triples
        #[cfg(feature = "triple-store")]
        let rdf_triples = collect_rdf_triples(&self.rdf_store);
        #[cfg(not(feature = "triple-store"))]
        let rdf_triples = Vec::new();

        #[cfg(feature = "triple-store")]
        let rdf_named_graphs: Vec<RdfNamedGraphSnapshot> = self
            .rdf_store
            .graph_names()
            .into_iter()
            .filter_map(|name| {
                self.rdf_store
                    .graph(&name)
                    .map(|graph| RdfNamedGraphSnapshot {
                        name,
                        triples: collect_rdf_triples(&graph),
                    })
            })
            .collect();
        #[cfg(not(feature = "triple-store"))]
        let rdf_named_graphs = Vec::new();

        let schema = collect_schema(&self.catalog);
        let indexes = collect_index_metadata(self.lpg_store());

        let snapshot = Snapshot {
            version: SNAPSHOT_VERSION,
            nodes,
            edges,
            named_graphs,
            rdf_triples,
            rdf_named_graphs,
            schema,
            indexes,
            #[cfg(feature = "temporal")]
            epoch: self.transaction_manager.current_epoch().as_u64(),
            #[cfg(not(feature = "temporal"))]
            epoch: 0,
        };

        let config = bincode::config::standard();
        bincode::serde::encode_to_vec(&snapshot, config)
            .map_err(|e| Error::Internal(format!("snapshot export failed: {e}")))
    }

    /// Creates a new in-memory database from a binary snapshot.
    ///
    /// The `data` must have been produced by [`export_snapshot()`](Self::export_snapshot).
    ///
    /// All edge references are validated before any data is inserted: every
    /// edge's source and destination must reference a node present in the
    /// snapshot, and duplicate node/edge IDs are rejected. If validation
    /// fails, no database is created.
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot is invalid, contains dangling edge
    /// references, has duplicate IDs, or deserialization fails.
    pub fn import_snapshot(data: &[u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::Internal("empty snapshot data".to_string()));
        }

        let version = data[0];
        if version != 4 {
            return Err(Error::Internal(format!(
                "unsupported snapshot version: {version} (expected 4)"
            )));
        }

        let config = bincode::config::standard();
        let (snapshot, _): (Snapshot, _) = bincode::serde::decode_from_slice(data, config)
            .map_err(|e| Error::Internal(format!("snapshot import failed: {e}")))?;

        // Validate default graph data
        validate_snapshot_data(&snapshot.nodes, &snapshot.edges)?;

        // Validate each named graph
        for ng in &snapshot.named_graphs {
            validate_snapshot_data(&ng.nodes, &ng.edges)?;
        }

        let db = Self::new_in_memory();
        populate_store_from_snapshot(db.lpg_store(), snapshot.nodes, snapshot.edges)?;

        // Restore epoch from snapshot
        #[cfg(feature = "temporal")]
        {
            let epoch = EpochId::new(snapshot.epoch);
            db.lpg_store().sync_epoch(epoch);
            db.transaction_manager.sync_epoch(epoch);
        }

        // Capture epoch before moving snapshot fields
        #[cfg(feature = "temporal")]
        let snapshot_epoch = EpochId::new(snapshot.epoch);

        // Restore named graphs
        for ng in snapshot.named_graphs {
            db.lpg_store()
                .create_graph(&ng.name)
                .map_err(|e| Error::Internal(e.to_string()))?;
            if let Some(graph_store) = db.lpg_store().graph(&ng.name) {
                populate_store_from_snapshot(&graph_store, ng.nodes, ng.edges)?;
                // Named graph stores need the same epoch so temporal property
                // lookups via current_epoch() return the correct values.
                #[cfg(feature = "temporal")]
                graph_store.sync_epoch(snapshot_epoch);
            }
        }

        // Restore RDF triples
        #[cfg(feature = "triple-store")]
        {
            populate_rdf_store(&db.rdf_store, &snapshot.rdf_triples);
            for rng in &snapshot.rdf_named_graphs {
                let graph = db.rdf_store.graph_or_create(&rng.name);
                populate_rdf_store(&graph, &rng.triples);
            }
        }

        // Restore schema
        restore_schema_from_snapshot(db.lpg_store(), &db.catalog, &snapshot.schema);

        // Restore indexes (must come after data population)
        restore_indexes_from_snapshot(&db, &snapshot.indexes);

        Ok(db)
    }

    /// Replaces the current database contents with data from a binary snapshot.
    ///
    /// The `data` must have been produced by
    /// [`export_snapshot()`](Self::export_snapshot).
    ///
    /// All validation (duplicate IDs, dangling edge references) is performed
    /// before any data is modified. If validation fails, the current database
    /// is left unchanged. If validation passes, the store is cleared and
    /// rebuilt from the snapshot atomically (from the perspective of
    /// subsequent queries).
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot is invalid, contains dangling edge
    /// references, has duplicate IDs, or deserialization fails.
    pub fn restore_snapshot(&self, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Err(Error::Internal("empty snapshot data".to_string()));
        }

        let version = data[0];
        if version != 4 {
            return Err(Error::Internal(format!(
                "unsupported snapshot version: {version} (expected 4)"
            )));
        }

        let config = bincode::config::standard();
        let (snapshot, _): (Snapshot, _) = bincode::serde::decode_from_slice(data, config)
            .map_err(|e| Error::Internal(format!("snapshot restore failed: {e}")))?;

        // Validate all data before making any changes
        validate_snapshot_data(&snapshot.nodes, &snapshot.edges)?;
        for ng in &snapshot.named_graphs {
            validate_snapshot_data(&ng.nodes, &ng.edges)?;
        }

        // Drop all existing named graphs, then clear default store
        for name in self.lpg_store().graph_names() {
            self.lpg_store().drop_graph(&name);
        }
        self.lpg_store().clear();

        populate_store_from_snapshot(self.lpg_store(), snapshot.nodes, snapshot.edges)?;

        // Restore epoch from temporal snapshot
        #[cfg(feature = "temporal")]
        let snapshot_epoch = {
            let epoch = EpochId::new(snapshot.epoch);
            self.lpg_store().sync_epoch(epoch);
            self.transaction_manager.sync_epoch(epoch);
            epoch
        };

        // Restore named graphs
        for ng in snapshot.named_graphs {
            self.lpg_store()
                .create_graph(&ng.name)
                .map_err(|e| Error::Internal(e.to_string()))?;
            if let Some(graph_store) = self.lpg_store().graph(&ng.name) {
                populate_store_from_snapshot(&graph_store, ng.nodes, ng.edges)?;
                #[cfg(feature = "temporal")]
                graph_store.sync_epoch(snapshot_epoch);
            }
        }

        // Restore RDF data
        #[cfg(feature = "triple-store")]
        {
            // Clear existing RDF data
            self.rdf_store.clear();
            for name in self.rdf_store.graph_names() {
                self.rdf_store.drop_graph(&name);
            }
            populate_rdf_store(&self.rdf_store, &snapshot.rdf_triples);
            for rng in &snapshot.rdf_named_graphs {
                let graph = self.rdf_store.graph_or_create(&rng.name);
                populate_rdf_store(&graph, &rng.triples);
            }
        }

        // Restore schema
        restore_schema_from_snapshot(self.lpg_store(), &self.catalog, &snapshot.schema);

        // Restore indexes (must come after data population)
        restore_indexes_from_snapshot(self, &snapshot.indexes);

        Ok(())
    }

    // =========================================================================
    // ADMIN API: Iteration
    // =========================================================================

    /// Returns an iterator over all nodes in the database.
    ///
    /// Useful for dump/export operations.
    pub fn iter_nodes(&self) -> impl Iterator<Item = grafeo_core::graph::lpg::Node> + '_ {
        self.lpg_store().all_nodes()
    }

    /// Returns an iterator over all edges in the database.
    ///
    /// Useful for dump/export operations.
    pub fn iter_edges(&self) -> impl Iterator<Item = grafeo_core::graph::lpg::Edge> + '_ {
        self.lpg_store().all_edges()
    }
}

#[cfg(test)]
mod tests {
    use grafeo_common::types::{EdgeId, NodeId, Value};

    use super::super::GrafeoDB;
    use super::{
        SNAPSHOT_VERSION, Snapshot, SnapshotEdge, SnapshotIndexes, SnapshotNode, SnapshotSchema,
    };

    #[test]
    fn test_restore_snapshot_basic() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        // Populate
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

        let snapshot = db.export_snapshot().unwrap();

        // Modify
        session
            .execute("INSERT (:Person {name: 'Vincent'})")
            .unwrap();
        assert_eq!(db.lpg_store().node_count(), 3);

        // Restore original
        db.restore_snapshot(&snapshot).unwrap();

        assert_eq!(db.lpg_store().node_count(), 2);
        let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    fn test_restore_snapshot_validation_failure() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        // Corrupt snapshot: just garbage bytes
        let result = db.restore_snapshot(b"garbage");
        assert!(result.is_err());

        // DB should be unchanged
        assert_eq!(db.lpg_store().node_count(), 1);
    }

    #[test]
    fn test_restore_snapshot_empty_db() {
        let db = GrafeoDB::new_in_memory();

        // Export empty snapshot, then populate, then restore to empty
        let empty_snapshot = db.export_snapshot().unwrap();

        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        assert_eq!(db.lpg_store().node_count(), 1);

        db.restore_snapshot(&empty_snapshot).unwrap();
        assert_eq!(db.lpg_store().node_count(), 0);
    }

    #[test]
    fn test_restore_snapshot_with_edges() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        session
            .execute(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) INSERT (a)-[:KNOWS]->(b)",
            )
            .unwrap();

        let snapshot = db.export_snapshot().unwrap();
        assert_eq!(db.lpg_store().edge_count(), 1);

        // Modify: add more data
        session
            .execute("INSERT (:Person {name: 'Vincent'})")
            .unwrap();

        // Restore
        db.restore_snapshot(&snapshot).unwrap();
        assert_eq!(db.lpg_store().node_count(), 2);
        assert_eq!(db.lpg_store().edge_count(), 1);
    }

    #[test]
    fn test_restore_snapshot_preserves_sessions() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        let snapshot = db.export_snapshot().unwrap();

        // Modify
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

        // Restore
        db.restore_snapshot(&snapshot).unwrap();

        // Session should still work and see restored data
        let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn test_export_import_roundtrip() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session
            .execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();

        let snapshot = db.export_snapshot().unwrap();
        let db2 = GrafeoDB::import_snapshot(&snapshot).unwrap();
        let session2 = db2.session();

        let result = session2.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    // --- to_memory() ---

    #[test]
    fn test_to_memory_empty() {
        let db = GrafeoDB::new_in_memory();
        let copy = db.to_memory().unwrap();
        assert_eq!(copy.lpg_store().node_count(), 0);
        assert_eq!(copy.lpg_store().edge_count(), 0);
    }

    #[test]
    fn test_to_memory_copies_nodes_and_properties() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();
        session
            .execute("INSERT (:Person {name: 'Gus', age: 25})")
            .unwrap();

        let copy = db.to_memory().unwrap();
        assert_eq!(copy.lpg_store().node_count(), 2);

        let s2 = copy.session();
        let result = s2
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0][0], Value::String("Alix".into()));
        assert_eq!(result.rows[1][0], Value::String("Gus".into()));
    }

    #[test]
    fn test_to_memory_copies_edges_and_properties() {
        let db = GrafeoDB::new_in_memory();
        let a = db.create_node(&["Person"]).unwrap();
        db.set_node_property(a, "name", "Alix".into()).unwrap();
        let b = db.create_node(&["Person"]).unwrap();
        db.set_node_property(b, "name", "Gus".into()).unwrap();
        let edge = db.create_edge(a, b, "KNOWS");
        db.set_edge_property(edge, "since", Value::Int64(2020));

        let copy = db.to_memory().unwrap();
        assert_eq!(copy.lpg_store().node_count(), 2);
        assert_eq!(copy.lpg_store().edge_count(), 1);

        let s2 = copy.session();
        let result = s2.execute("MATCH ()-[e:KNOWS]->() RETURN e.since").unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(2020));
    }

    #[test]
    fn test_to_memory_is_independent() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        let copy = db.to_memory().unwrap();

        // Mutating original should not affect copy
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        assert_eq!(db.lpg_store().node_count(), 2);
        assert_eq!(copy.lpg_store().node_count(), 1);
    }

    // --- iter_nodes() / iter_edges() ---

    #[test]
    fn test_iter_nodes_empty() {
        let db = GrafeoDB::new_in_memory();
        assert_eq!(db.iter_nodes().count(), 0);
    }

    #[test]
    fn test_iter_nodes_returns_all() {
        let db = GrafeoDB::new_in_memory();
        let id1 = db.create_node(&["Person"]).unwrap();
        db.set_node_property(id1, "name", "Alix".into()).unwrap();
        let id2 = db.create_node(&["Animal"]).unwrap();
        db.set_node_property(id2, "name", "Fido".into()).unwrap();

        let nodes: Vec<_> = db.iter_nodes().collect();
        assert_eq!(nodes.len(), 2);

        let names: Vec<_> = nodes
            .iter()
            .filter_map(|n| n.properties.iter().find(|(k, _)| k.as_str() == "name"))
            .map(|(_, v)| v.clone())
            .collect();
        assert!(names.contains(&Value::String("Alix".into())));
        assert!(names.contains(&Value::String("Fido".into())));
    }

    #[test]
    fn test_iter_edges_empty() {
        let db = GrafeoDB::new_in_memory();
        assert_eq!(db.iter_edges().count(), 0);
    }

    #[test]
    fn test_iter_edges_returns_all() {
        let db = GrafeoDB::new_in_memory();
        let a = db.create_node(&["A"]).unwrap();
        let b = db.create_node(&["B"]).unwrap();
        let c = db.create_node(&["C"]).unwrap();
        db.create_edge(a, b, "R1");
        db.create_edge(b, c, "R2");

        let edges: Vec<_> = db.iter_edges().collect();
        assert_eq!(edges.len(), 2);

        let types: Vec<_> = edges.iter().map(|e| e.edge_type.as_ref()).collect();
        assert!(types.contains(&"R1"));
        assert!(types.contains(&"R2"));
    }

    // --- restore_snapshot() validation ---

    fn make_snapshot(version: u8, nodes: Vec<SnapshotNode>, edges: Vec<SnapshotEdge>) -> Vec<u8> {
        let snap = Snapshot {
            version,
            nodes,
            edges,
            named_graphs: vec![],
            rdf_triples: vec![],
            rdf_named_graphs: vec![],
            schema: SnapshotSchema::default(),
            indexes: SnapshotIndexes::default(),
            epoch: 0,
        };
        bincode::serde::encode_to_vec(&snap, bincode::config::standard()).unwrap()
    }

    #[test]
    fn test_restore_rejects_unsupported_version() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        let bytes = make_snapshot(99, vec![], vec![]);

        let result = db.restore_snapshot(&bytes);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("unsupported snapshot version"), "got: {err}");

        // DB unchanged
        assert_eq!(db.lpg_store().node_count(), 1);
    }

    #[test]
    fn test_restore_rejects_duplicate_node_ids() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        let bytes = make_snapshot(
            SNAPSHOT_VERSION,
            vec![
                SnapshotNode {
                    id: NodeId::new(0),
                    labels: vec!["A".into()],
                    properties: vec![],
                },
                SnapshotNode {
                    id: NodeId::new(0),
                    labels: vec!["B".into()],
                    properties: vec![],
                },
            ],
            vec![],
        );

        let result = db.restore_snapshot(&bytes);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("duplicate node ID"), "got: {err}");
        assert_eq!(db.lpg_store().node_count(), 1);
    }

    #[test]
    fn test_restore_rejects_duplicate_edge_ids() {
        let db = GrafeoDB::new_in_memory();

        let bytes = make_snapshot(
            SNAPSHOT_VERSION,
            vec![
                SnapshotNode {
                    id: NodeId::new(0),
                    labels: vec![],
                    properties: vec![],
                },
                SnapshotNode {
                    id: NodeId::new(1),
                    labels: vec![],
                    properties: vec![],
                },
            ],
            vec![
                SnapshotEdge {
                    id: EdgeId::new(0),
                    src: NodeId::new(0),
                    dst: NodeId::new(1),
                    edge_type: "REL".into(),
                    properties: vec![],
                },
                SnapshotEdge {
                    id: EdgeId::new(0),
                    src: NodeId::new(0),
                    dst: NodeId::new(1),
                    edge_type: "REL".into(),
                    properties: vec![],
                },
            ],
        );

        let result = db.restore_snapshot(&bytes);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("duplicate edge ID"), "got: {err}");
    }

    #[test]
    fn test_restore_rejects_dangling_source() {
        let db = GrafeoDB::new_in_memory();

        let bytes = make_snapshot(
            SNAPSHOT_VERSION,
            vec![SnapshotNode {
                id: NodeId::new(0),
                labels: vec![],
                properties: vec![],
            }],
            vec![SnapshotEdge {
                id: EdgeId::new(0),
                src: NodeId::new(999),
                dst: NodeId::new(0),
                edge_type: "REL".into(),
                properties: vec![],
            }],
        );

        let result = db.restore_snapshot(&bytes);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("non-existent source node"), "got: {err}");
    }

    #[test]
    fn test_restore_rejects_dangling_destination() {
        let db = GrafeoDB::new_in_memory();

        let bytes = make_snapshot(
            SNAPSHOT_VERSION,
            vec![SnapshotNode {
                id: NodeId::new(0),
                labels: vec![],
                properties: vec![],
            }],
            vec![SnapshotEdge {
                id: EdgeId::new(0),
                src: NodeId::new(0),
                dst: NodeId::new(999),
                edge_type: "REL".into(),
                properties: vec![],
            }],
        );

        let result = db.restore_snapshot(&bytes);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("non-existent destination node"), "got: {err}");
    }

    // --- index metadata roundtrip ---

    #[test]
    fn test_snapshot_roundtrip_property_index() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session
            .execute("INSERT (:Person {name: 'Alix', email: 'alix@example.com'})")
            .unwrap();
        db.create_property_index("email");
        assert!(db.has_property_index("email"));

        let snapshot = db.export_snapshot().unwrap();
        let db2 = GrafeoDB::import_snapshot(&snapshot).unwrap();

        assert!(db2.has_property_index("email"));

        // Verify the index actually works for O(1) lookups
        let found = db2.find_nodes_by_property("email", &Value::String("alix@example.com".into()));
        assert_eq!(found.len(), 1);
    }

    #[cfg(feature = "vector-index")]
    #[test]
    fn test_snapshot_roundtrip_vector_index() {
        use std::sync::Arc;

        let db = GrafeoDB::new_in_memory();

        let n1 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(
            n1,
            "embedding",
            Value::Vector(Arc::from([1.0_f32, 0.0, 0.0])),
        )
        .unwrap();
        let n2 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(
            n2,
            "embedding",
            Value::Vector(Arc::from([0.0_f32, 1.0, 0.0])),
        )
        .unwrap();

        db.create_vector_index(
            "Doc",
            "embedding",
            None,
            Some("cosine"),
            Some(4),
            Some(32),
            None,
        )
        .unwrap();

        let snapshot = db.export_snapshot().unwrap();
        let db2 = GrafeoDB::import_snapshot(&snapshot).unwrap();

        // Vector search should work on the restored database
        let results = db2
            .vector_search("Doc", "embedding", &[1.0, 0.0, 0.0], 2, None, None)
            .unwrap();
        assert_eq!(results.len(), 2);
        // Closest to [1,0,0] should be n1
        assert_eq!(results[0].0, n1);
    }

    #[cfg(feature = "text-index")]
    #[test]
    fn test_snapshot_roundtrip_text_index() {
        let db = GrafeoDB::new_in_memory();

        let n1 = db.create_node(&["Article"]).unwrap();
        db.set_node_property(n1, "body", Value::String("rust graph database".into()))
            .unwrap();
        let n2 = db.create_node(&["Article"]).unwrap();
        db.set_node_property(n2, "body", Value::String("python web framework".into()))
            .unwrap();

        db.create_text_index("Article", "body").unwrap();

        let snapshot = db.export_snapshot().unwrap();
        let db2 = GrafeoDB::import_snapshot(&snapshot).unwrap();

        // Text search should work on the restored database
        let results = db2
            .text_search("Article", "body", "graph database", 10)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, n1);
    }

    #[test]
    fn test_snapshot_roundtrip_property_index_via_restore() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session
            .execute("INSERT (:Person {name: 'Alix', email: 'alix@example.com'})")
            .unwrap();
        db.create_property_index("email");

        let snapshot = db.export_snapshot().unwrap();

        // Mutate the database
        session
            .execute("INSERT (:Person {name: 'Gus', email: 'gus@example.com'})")
            .unwrap();
        db.drop_property_index("email");
        assert!(!db.has_property_index("email"));

        // Restore should bring back the index
        db.restore_snapshot(&snapshot).unwrap();
        assert!(db.has_property_index("email"));
    }
}
