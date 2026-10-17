//! Persistence, snapshots, and data export for GrafeoDB.

#[cfg(any(feature = "wal", feature = "grafeo-file"))]
use std::path::Path;

#[cfg(any(feature = "vector-index", feature = "text-index"))]
use grafeo_common::grafeo_warn;
use grafeo_common::types::{EdgeId, EpochId, NodeId, Value};
use grafeo_common::utils::error::{Error, Result};
use hashbrown::HashSet;

use crate::config::Config;

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
/// The nodes of `store` as a snapshot holds them, read one at a time. A node
/// record or a spilled property value that cannot be read fails the export
/// instead of leaving the copy without it.
fn collect_snapshot_nodes(store: &grafeo_core::graph::lpg::LpgStore) -> Result<Vec<SnapshotNode>> {
    // With `temporal` the snapshot holds each property's history, read
    // below: the current values are not read.
    #[cfg(feature = "temporal")]
    let source = store
        .try_nodes_without_properties()?
        .map(Ok::<_, grafeo_common::utils::error::Error>);
    #[cfg(not(feature = "temporal"))]
    let source = store.try_nodes()?;
    let mut nodes: Vec<SnapshotNode> = Vec::new();
    for n in source {
        let n = n?;
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

        nodes.push(SnapshotNode {
            id: n.id,
            labels,
            properties,
        });
    }
    nodes.sort_by_key(|n| n.id);
    Ok(nodes)
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
            return Err(Error::InvalidValue(format!(
                "snapshot contains duplicate node ID {}",
                node.id
            )));
        }
        refuse_too_deep("node", node.id.as_u64(), &node.properties)?;
    }
    let mut edge_ids = HashSet::with_capacity(edges.len());
    for edge in edges {
        if !edge_ids.insert(edge.id) {
            return Err(Error::InvalidValue(format!(
                "snapshot contains duplicate edge ID {}",
                edge.id
            )));
        }
        refuse_too_deep("edge", edge.id.as_u64(), &edge.properties)?;
        if !node_ids.contains(&edge.src) {
            return Err(Error::InvalidValue(format!(
                "snapshot edge {} references non-existent source node {}",
                edge.id, edge.src
            )));
        }
        if !node_ids.contains(&edge.dst) {
            return Err(Error::InvalidValue(format!(
                "snapshot edge {} references non-existent destination node {}",
                edge.id, edge.dst
            )));
        }
    }
    Ok(())
}

/// Refuses a property version of snapshot `entity` `id` nested deeper than a
/// database can store (see [`nests_too_deep`]), naming the entity and the
/// property, as a write of the value would be refused.
fn refuse_too_deep(
    entity: &str,
    id: u64,
    properties: &[(String, Vec<(EpochId, Value)>)],
) -> Result<()> {
    use grafeo_common::storage::value_codec::{MAX_PROPERTY_VALUE_DEPTH, nests_too_deep};

    for (key, versions) in properties {
        if versions.iter().any(|(_, value)| nests_too_deep(value)) {
            return Err(Error::InvalidValue(format!(
                "snapshot {entity} {id}, property {key:?}: the value nests lists, maps and \
                 paths more than {MAX_PROPERTY_VALUE_DEPTH} levels deep, deeper than a database \
                 can store"
            )));
        }
    }
    Ok(())
}

/// Refuses `snapshot` when it holds data this build cannot read (see
/// [`FeatureData`](super::sections::FeatureData)): RDF triples or graphs
/// without the `triple-store` feature, vector or text index definitions
/// without `vector-index` or `text-index`. Such a build would `action`
/// (`import`, `restore`) the snapshot without that data.
///
/// # Errors
///
/// Returns an error that says what the snapshot holds (`1 in the default
/// graph, 3 in graph trips`, `on :Document(embedding)`) and the features, in
/// that case.
fn refuse_unreadable_snapshot(snapshot: &Snapshot, action: &str) -> Result<()> {
    use super::sections::{FeatureData, RDF_TRIPLES, TEXT_INDEXES, VECTOR_INDEXES, refusal_of};

    // Plain loops that append to one string per kind of data: this runs in
    // the WebAssembly packages, whose size is gated.
    let mut found: Vec<(&FeatureData, String)> = Vec::new();
    if !RDF_TRIPLES.in_build() {
        let mut held = String::new();
        if !snapshot.rdf_triples.is_empty() {
            held.push_str(&snapshot.rdf_triples.len().to_string());
            held.push_str(" in the default graph");
        }
        for graph in &snapshot.rdf_named_graphs {
            if !held.is_empty() {
                held.push_str(", ");
            }
            held.push_str(&graph.triples.len().to_string());
            held.push_str(" in graph ");
            held.push_str(&graph.name);
        }
        if !held.is_empty() {
            found.push((&RDF_TRIPLES, held));
        }
    }
    if !VECTOR_INDEXES.in_build() {
        let mut held = String::new();
        for def in &snapshot.indexes.vector_indexes {
            push_index_on(&mut held, &def.label, &def.property);
        }
        if !held.is_empty() {
            found.push((&VECTOR_INDEXES, held));
        }
    }
    if !TEXT_INDEXES.in_build() {
        let mut held = String::new();
        for def in &snapshot.indexes.text_indexes {
            push_index_on(&mut held, &def.label, &def.property);
        }
        if !held.is_empty() {
            found.push((&TEXT_INDEXES, held));
        }
    }
    if found.is_empty() {
        Ok(())
    } else {
        Err(refusal_of("the snapshot", action, &found))
    }
}

/// Appends the index on `property` of the nodes with `label` to `held`, the
/// list a refusal names: `on :Document(embedding), :Document(title)`.
fn push_index_on(held: &mut String, label: &str, property: &str) {
    held.push_str(if held.is_empty() { "on :" } else { ", :" });
    held.push_str(label);
    held.push('(');
    held.push_str(property);
    held.push(')');
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

/// The RDF graphs of a snapshot with their terms read: the default graph's
/// triples, then each named graph's name and triples.
#[cfg(feature = "triple-store")]
struct RdfSnapshotGraphs {
    default: Vec<grafeo_core::graph::rdf::Triple>,
    named: Vec<(String, Vec<grafeo_core::graph::rdf::Triple>)>,
}

/// Reads the terms of every RDF triple of a snapshot, before anything is
/// changed, so a snapshot that holds a term that does not parse changes
/// nothing.
///
/// # Errors
///
/// Returns [`Error::Serialization`] naming the graph, the triple and the term
/// when a term is not an N-Triples term: a snapshot never loses a triple.
#[cfg(feature = "triple-store")]
fn read_rdf_snapshot(
    triples: &[SnapshotTriple],
    named: &[RdfNamedGraphSnapshot],
) -> Result<RdfSnapshotGraphs> {
    Ok(RdfSnapshotGraphs {
        default: read_rdf_triples(None, triples)?,
        named: named
            .iter()
            .map(|graph| {
                Ok((
                    graph.name.clone(),
                    read_rdf_triples(Some(&graph.name), &graph.triples)?,
                ))
            })
            .collect::<Result<_>>()?,
    })
}

/// The triples of one graph of a snapshot (`None`: the default graph), their
/// terms read from their N-Triples strings.
#[cfg(feature = "triple-store")]
fn read_rdf_triples(
    graph: Option<&str>,
    triples: &[SnapshotTriple],
) -> Result<Vec<grafeo_core::graph::rdf::Triple>> {
    use grafeo_core::graph::rdf::{Term, Triple};

    triples
        .iter()
        .enumerate()
        .map(|(index, triple)| {
            let term = |text: &str, role: &str| {
                Term::from_ntriples(text).map_err(|error| {
                    let graph = graph.map_or_else(
                        || "the default graph".to_string(),
                        |name| format!("graph {name:?}"),
                    );
                    Error::Serialization(format!(
                        "snapshot RDF triple {index} of {graph}, {role}: {error}"
                    ))
                })
            };
            // Unchecked: the store holds what it was given, as it was
            // exported.
            Ok(Triple::new_unchecked(
                term(&triple.subject, "subject")?,
                term(&triple.predicate, "predicate")?,
                term(&triple.object, "object")?,
            ))
        })
        .collect()
}

/// Populates an RDF store from the graphs of a snapshot, creating its named
/// graphs.
#[cfg(feature = "triple-store")]
fn populate_rdf_store(store: &grafeo_core::graph::rdf::RdfStore, graphs: RdfSnapshotGraphs) {
    store.batch_insert(graphs.default);
    for (name, triples) in graphs.named {
        store.graph_or_create(&name).batch_insert(triples);
    }
}

// =========================================================================
// Snapshot deserialization helpers (used by single-file format)
// =========================================================================

/// Decodes snapshot bytes, the snapshot of the 0.5.x container v1 file
/// `path`, and populates a store and catalog.
///
/// # Errors
///
/// Returns an error if the snapshot does not decode or a term of its triples
/// does not parse; in a build without the `triple-store` feature, if it holds
/// RDF triples or graphs, which the database would be loaded (and migrated)
/// without.
#[cfg(feature = "grafeo-file")]
#[cfg_attr(
    feature = "triple-store",
    expect(
        unused_variables,
        reason = "only a build without `triple-store` refuses the snapshot, naming the file"
    )
)]
pub(super) fn load_snapshot_into_store(
    path: &Path,
    store: &std::sync::Arc<grafeo_core::graph::lpg::LpgStore>,
    catalog: &std::sync::Arc<crate::catalog::Catalog>,
    #[cfg(feature = "triple-store")] rdf_store: &std::sync::Arc<grafeo_core::graph::rdf::RdfStore>,
    data: &[u8],
) -> grafeo_common::utils::error::Result<()> {
    use grafeo_common::utils::error::Error;

    let config = bincode::config::standard();
    let (snapshot, _) =
        bincode::serde::decode_from_slice::<Snapshot, _>(data, config).map_err(|e| {
            Error::corruption(format!("failed to decode snapshot from .grafeo file: {e}"))
        })?;
    #[cfg(feature = "triple-store")]
    let rdf_graphs = read_rdf_snapshot(&snapshot.rdf_triples, &snapshot.rdf_named_graphs)?;
    #[cfg(not(feature = "triple-store"))]
    if !snapshot.rdf_triples.is_empty() || !snapshot.rdf_named_graphs.is_empty() {
        return Err(super::sections::refusal(
            path,
            &[(
                &super::sections::RDF_TRIPLES,
                "the snapshot of a 0.5.x file".to_string(),
            )],
        ));
    }

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
        populate_rdf_store(rdf_store, rdf_graphs);
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

/// Copies the nodes and edges of `source` into `target` with their IDs, and
/// those of each named graph, as the LPG section would load them: with
/// `temporal`, every property keeps its history and the stores their epoch.
fn copy_graph_data(
    source: &grafeo_core::graph::lpg::LpgStore,
    target: &grafeo_core::graph::lpg::LpgStore,
) -> Result<()> {
    // A node record or a spilled value that cannot be read fails the copy
    // instead of leaving it out. The nodes are read one at a time, each just
    // before it is written, so the copy never holds them all. With
    // `temporal` each property's history is copied: the current values are
    // not read.
    #[cfg(feature = "temporal")]
    let nodes = source
        .try_nodes_without_properties()?
        .map(Ok::<_, grafeo_common::utils::error::Error>);
    #[cfg(not(feature = "temporal"))]
    let nodes = source.try_nodes()?;
    for node in nodes {
        let node = node?;
        let labels: Vec<&str> = node.labels.iter().map(|label| &**label).collect();
        target.create_node_with_id(node.id, &labels)?;
        #[cfg(feature = "temporal")]
        for (key, history) in source.node_property_history(node.id) {
            for (epoch, value) in history {
                target.set_node_property_at_epoch(node.id, key.as_str(), value, epoch);
            }
        }
        #[cfg(not(feature = "temporal"))]
        for (key, value) in node.properties {
            target.set_node_property(node.id, key.as_str(), value);
        }
    }
    for edge in source.all_edges() {
        target.create_edge_with_id(edge.id, edge.src, edge.dst, &edge.edge_type)?;
        #[cfg(feature = "temporal")]
        for (key, history) in source.edge_property_history(edge.id) {
            for (epoch, value) in history {
                target.set_edge_property_at_epoch(edge.id, key.as_str(), value, epoch);
            }
        }
        #[cfg(not(feature = "temporal"))]
        for (key, value) in edge.properties {
            target.set_edge_property(edge.id, key.as_str(), value);
        }
    }
    #[cfg(feature = "temporal")]
    target.sync_epoch(source.current_epoch());

    for name in source.graph_names() {
        if let Some(source_graph) = source.graph(&name) {
            target
                .create_graph(&name)
                .map_err(|e| Error::Internal(e.to_string()))?;
            if let Some(target_graph) = target.graph(&name) {
                copy_graph_data(&source_graph, &target_graph)?;
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
/// creation scans existing data. A build without `vector-index` or
/// `text-index` has refused a snapshot with such definitions before (see
/// [`refuse_unreadable_snapshot`]).
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

    /// Saves a copy of the database to a new single file at `path`, whatever
    /// its extension (`.grafeo`, `.db` or none): a database of its own, with
    /// every graph, the schema and the indexes, and no sidecar WAL. Works the
    /// same for an in-memory and a persistent database; the original stays
    /// as it is. Like a checkpoint, the copy holds the committed state: a
    /// transaction still open is left out of it (see [`close`](Self::close)).
    ///
    /// The copy of an encrypted database (`Config::encryption`) is encrypted
    /// with the same key chain: it has a new database id and so its own
    /// keys.
    ///
    /// # Errors
    ///
    /// Returns an error if `path` already exists, if the file cannot be
    /// written, after a commit that did not complete (see
    /// [`TransactionManager`](crate::transaction::TransactionManager)), and
    /// the database-closed error after `close()` of a persistent database.
    ///
    /// Requires the `wal` feature for persistence support.
    #[cfg(feature = "wal")]
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        // `close()` waits for the save, and none runs after it.
        let _open = self.hold_open()?;
        // The spelling every open uses (see `normalize_path`): `copy/` names
        // the file `copy`, as an open of `copy/` does.
        let path = super::normalize_path(path.as_ref())?;
        self.write_image(&path)
    }

    /// Writes the database's complete state to a new `.grafeo` file at
    /// `path` (see [`write_image_with`](Self::write_image_with)), encrypted
    /// when this database is.
    ///
    /// # Errors
    ///
    /// The same as [`write_image_with`](Self::write_image_with).
    #[cfg(feature = "wal")]
    pub(crate) fn write_image(&self, path: &Path) -> Result<()> {
        self.write_image_with(
            path,
            &super::encryption::DatabaseKeys::from_config(&self.config),
        )
    }

    /// Writes the database's complete state to a new `.grafeo` file at
    /// `path`: every checkpoint section, as one image, with the header
    /// values of a checkpoint. The file has no sidecar WAL; it holds
    /// everything.
    ///
    /// The file is a database of its own, with a new database id, encrypted
    /// with the keys `keys` derive for that id (not encrypted when `keys`
    /// has no key chain).
    ///
    /// If writing the image fails, the new file is removed again: it would
    /// open as an empty database, and a retry would find `path` taken.
    ///
    /// # Errors
    ///
    /// Returns an error if `path` already exists, or a section fails to
    /// serialize, or the file cannot be written.
    #[cfg(feature = "grafeo-file")]
    pub(crate) fn write_image_with(
        &self,
        path: &Path,
        keys: &super::encryption::DatabaseKeys,
    ) -> Result<()> {
        use grafeo_storage::file::GrafeoFileManager;
        use grafeo_storage::file::v3::header::new_database_id;

        // Commits held off until the image is written: it holds every commit
        // whole, and none that did not complete.
        let commits = self.transaction_manager.hold_commits()?;
        let sources = self.checkpoint_sources();
        let sections = sources.sections(&commits);
        let section_refs: Vec<&dyn grafeo_common::storage::Section> =
            sections.iter().map(AsRef::as_ref).collect();
        let database_id = new_database_id();
        let fm = GrafeoFileManager::create_with_id(
            path,
            database_id,
            keys.container_cipher(database_id),
        )?;
        let written = fm.write_checkpoint(&section_refs, &sources.context().checkpoint_header());
        drop(commits);
        let written = written.and_then(|()| fm.close());
        if written.is_err() {
            drop(fm);
            // Best effort: the error that matters is the one returned.
            if let Err(error) = std::fs::remove_file(path) {
                grafeo_common::grafeo_warn!(
                    "cannot remove the incomplete image {}: {error}",
                    path.display()
                );
            }
        }
        written
    }

    /// Creates an in-memory copy of this database.
    ///
    /// The copy is independent of this database and holds what reopening
    /// it from a checkpoint would: every graph with its data, the schema and
    /// constraints, and the property, vector and text indexes. It is built
    /// from the checkpoint sections and loaded as a `.grafeo` file is, except
    /// that nodes and edges are copied from store to store, which is much
    /// faster than encoding them (while no transaction is open).
    ///
    /// Like a checkpoint, the copy holds the committed state: what a
    /// transaction still open deleted is in it, with the values and labels
    /// it changed as they were committed, and nothing it created is.
    ///
    /// Useful for:
    /// - Testing modifications without affecting the original
    /// - Faster operations when persistence isn't needed
    ///
    /// The copy of an encrypted database has no key (an in-memory database
    /// cannot carry `Config::encryption`), so a copy saved from it with
    /// [`save`](Self::save) is not encrypted. For an encrypted copy, call
    /// `save` on the encrypted database itself.
    ///
    /// # Errors
    ///
    /// Returns an error if the copy operation fails, or after a commit that
    /// did not complete.
    pub fn to_memory(&self) -> Result<Self> {
        let target = Self::with_config(Config::in_memory())?;
        // Each section is served once and freed as soon as it is loaded.
        let image = grafeo_common::storage::ServedOnce::new(self.copy_into(&target)?);
        let loaded = super::sections::load_sections(
            &image,
            None,
            &target.lpg_store(),
            &target.catalog,
            #[cfg(feature = "triple-store")]
            &target.rdf_store,
        )?;
        // The copy continues at the epoch of what it holds, which
        // `copy_into` gave its transaction manager, as a reopen of a
        // checkpoint continues at the checkpoint's epoch.
        target
            .lpg_store()
            .sync_epoch(target.transaction_manager.current_epoch());
        Self::continue_epochs(&target.lpg_store(), &target.transaction_manager);
        target.finish_load(loaded);
        Ok(target)
    }

    /// Copies this database's nodes and edges into `target`, and returns an
    /// image of every other section of a checkpoint, for
    /// [`to_memory`](Self::to_memory) to load. `target`'s transaction
    /// manager takes the epoch of the commits they hold.
    ///
    /// Both are taken under one commit hold, so they hold the same commits:
    /// every commit whole, none that did not complete, and nothing of a
    /// transaction still open (the hold also holds its writes). The LPG
    /// section is left out of the image: its nodes and edges are copied from store to store instead, which is
    /// much faster than encoding them. While a transaction is open the stores
    /// hold what it wrote, so the image holds the LPG section, which writes
    /// the committed state, and the copy loads it as a reopen does.
    ///
    /// # Errors
    ///
    /// Returns an error if commits cannot be held, a section fails to write,
    /// or the nodes and edges cannot be copied.
    pub(super) fn copy_into(&self, target: &Self) -> Result<grafeo_common::storage::MemoryImage> {
        use grafeo_common::storage::{MemoryImage, SectionType};

        let mut image = MemoryImage::new();
        let commits = self.transaction_manager.hold_commits()?;
        let sources = self.checkpoint_sources();
        // The epoch of the commits the copy holds, for `to_memory` to
        // continue from once its stores are loaded.
        target
            .transaction_manager
            .sync_epoch(EpochId::new(sources.epoch()));
        let open = sources.has_open_changes(&commits);
        for section in sources.sections(&commits) {
            if section.section_type() == SectionType::LpgStore && !open {
                copy_graph_data(&self.lpg_store(), &target.lpg_store())?;
            } else {
                image.begin_section(section.section_type(), section.version())?;
                section.write_to(&mut image)?;
            }
        }
        Ok(image)
    }

    /// Opens a database file and loads it entirely into memory.
    ///
    /// The returned database has no connection to the original file.
    /// Changes will NOT be written back to the file.
    ///
    /// This takes no key, so an encrypted database fails to open here: open
    /// it with its key ([`with_config`](Self::with_config)) and call
    /// [`to_memory`](Self::to_memory).
    ///
    /// An existing database is read as [`open_read_only`](Self::open_read_only)
    /// reads it, and nothing on disk changes: a 0.6 file under a shared lock
    /// (other readers may hold it too) with its sidecar WAL replayed, and no
    /// checkpoint or WAL removal when it is closed; a database written by 0.5.x
    /// (a `.grafeo` file or a WAL directory) with its WAL replayed, and not
    /// migrated. A path whose migration was cut off fails as a read-only open
    /// does. At a missing path a new, empty database file is created, as
    /// [`open`](Self::open) creates one.
    ///
    /// # Errors
    ///
    /// Returns an error if the file can't be opened or loaded.
    #[cfg(feature = "wal")]
    pub fn open_in_memory(path: impl AsRef<Path>) -> Result<Self> {
        use grafeo_storage::file::detect::{OnDisk, detect};

        // The spelling every open uses (see `normalize_path`).
        let path = &super::normalize_path(path.as_ref())?;
        match detect(path)? {
            // A missing file next to a cut-off migration or a kept copy
            // fails as a read-only open does, before anything is created.
            OnDisk::Missing => super::migration::check_read_only(path)?,
            // Read as a read-only open reads it: a read-write open would take
            // the exclusive lock, checkpoint the file and remove its WAL when
            // it closes, and migrate a 0.5.x database.
            _ => {
                let source = Self::with_config(Config::read_only(path))?;
                let target = source.to_memory()?;
                source.close()?;
                return Ok(target);
            }
        }

        // A missing path becomes a new, empty database, as `open` creates.
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
    /// The bytes are never encrypted, also for a database with
    /// `Config::encryption`: they are the plaintext data, to be stored as
    /// safely as the data itself. For an encrypted copy, use
    /// [`save`](Self::save) with a `.grafeo` path.
    ///
    /// Properties are stored as version-history lists. When `temporal` is
    /// enabled, the full history is captured. Otherwise, each property is
    /// wrapped as a single-entry list at epoch 0.
    ///
    /// The snapshot holds the committed state: what a transaction still open
    /// deleted is in it, with the values and labels it changed as they were
    /// committed, and nothing it created is (while one is open, the data is
    /// read from a committed copy, which takes as much memory again).
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails, or after a commit that did
    /// not complete.
    pub fn export_snapshot(&self) -> Result<Vec<u8>> {
        // The snapshot holds every commit whole, none that did not complete
        // (whose stamped part the store holds), and nothing of a transaction
        // still open (the hold also holds its writes).
        let commits = self.transaction_manager.hold_commits()?;
        let committed = self
            .checkpoint_sources()
            .committed(&commits)?
            .ok_or_else(|| {
                Error::Query(grafeo_common::utils::error::QueryError::unsupported(
                    "a snapshot export needs the built-in LPG store",
                ))
            })?;
        let store = &committed.store;
        let (nodes, edges) = (
            collect_snapshot_nodes(store)?,
            collect_snapshot_edges(store),
        );

        // Collect named graphs
        let mut named_graphs: Vec<NamedGraphSnapshot> = Vec::new();
        for name in store.graph_names() {
            if let Some(graph_store) = store.graph(&name) {
                named_graphs.push(NamedGraphSnapshot {
                    name,
                    nodes: collect_snapshot_nodes(&graph_store)?,
                    edges: collect_snapshot_edges(&graph_store),
                });
            }
        }

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
        let indexes = collect_index_metadata(&self.lpg_store());

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
    /// The `data` must have been produced by [`export_snapshot()`](Self::export_snapshot),
    /// so it is plaintext (snapshots are never encrypted).
    ///
    /// All edge references are validated before any data is inserted: every
    /// edge's source and destination must reference a node present in the
    /// snapshot, and duplicate node/edge IDs are rejected. If validation
    /// fails, no database is created.
    ///
    /// A build without the `triple-store`, `vector-index` or `text-index`
    /// feature refuses a snapshot that holds RDF triples, or vector or text
    /// index definitions: it would create the database without them.
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot is invalid, contains dangling edge
    /// references, has duplicate IDs, holds an RDF term that is not an
    /// N-Triples term, or deserialization fails; and, naming the data and the
    /// feature, if it holds data this build cannot read.
    pub fn import_snapshot(data: &[u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::InvalidValue("empty snapshot data".to_string()));
        }

        let version = data[0];
        if version != 4 {
            return Err(Error::InvalidValue(format!(
                "unsupported snapshot version: {version} (expected 4)"
            )));
        }

        let config = bincode::config::standard();
        let (snapshot, _): (Snapshot, _) = bincode::serde::decode_from_slice(data, config)
            .map_err(|e| Error::Serialization(format!("snapshot import failed: {e}")))?;
        // Before the database is created: this build would import the
        // snapshot without the data it cannot read.
        refuse_unreadable_snapshot(&snapshot, "import")?;

        // Validate default graph data
        validate_snapshot_data(&snapshot.nodes, &snapshot.edges)?;

        // Validate each named graph
        for ng in &snapshot.named_graphs {
            validate_snapshot_data(&ng.nodes, &ng.edges)?;
        }
        #[cfg(feature = "triple-store")]
        let rdf_graphs = read_rdf_snapshot(&snapshot.rdf_triples, &snapshot.rdf_named_graphs)?;

        let db = Self::new_in_memory();
        populate_store_from_snapshot(&db.lpg_store(), snapshot.nodes, snapshot.edges)?;

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
            populate_rdf_store(&db.rdf_store, rdf_graphs);
        }

        // Restore schema
        restore_schema_from_snapshot(&db.lpg_store(), &db.catalog, &snapshot.schema);

        // Restore indexes (must come after data population)
        restore_indexes_from_snapshot(&db, &snapshot.indexes);

        Ok(db)
    }

    /// Replaces the current database contents with data from a binary snapshot.
    ///
    /// The `data` must have been produced by
    /// [`export_snapshot()`](Self::export_snapshot), so it is plaintext
    /// (snapshots are never encrypted).
    ///
    /// All validation (duplicate IDs, dangling edge references, RDF terms,
    /// data this build cannot read) is performed before any data is
    /// modified. If validation fails, the current database
    /// is left unchanged. If validation passes, the store is cleared and
    /// rebuilt from the snapshot atomically (from the perspective of
    /// subsequent queries).
    ///
    /// A build without the `triple-store`, `vector-index` or `text-index`
    /// feature refuses a snapshot that holds RDF triples, or vector or text
    /// index definitions, and leaves the database as it was: it would restore
    /// the database without them.
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot is invalid, contains dangling edge
    /// references, has duplicate IDs, holds an RDF term that is not an
    /// N-Triples term or data this build cannot read (naming the data and the
    /// feature), or deserialization fails, after a commit that did not
    /// complete (the restored database could never be checkpointed, see
    /// [`TransactionManager`](crate::transaction::TransactionManager)),
    /// on a read-only database, and the database-closed error after `close()`
    /// of a persistent database (read-only or not).
    pub fn restore_snapshot(&self, data: &[u8]) -> Result<()> {
        // A restore writes no WAL record: after `close()` (which releases the
        // file) nothing would persist it, so it fails then, and `close()`
        // waits for one in progress.
        let _open = self.hold_open()?;
        if self.read_only {
            return Err(Error::Transaction(
                grafeo_common::utils::error::TransactionError::ReadOnly,
            ));
        }
        self.transaction_manager.check_no_incomplete_commit()?;
        self.transaction_manager.check_open()?;
        if data.is_empty() {
            return Err(Error::InvalidValue("empty snapshot data".to_string()));
        }

        let version = data[0];
        if version != 4 {
            return Err(Error::InvalidValue(format!(
                "unsupported snapshot version: {version} (expected 4)"
            )));
        }

        let config = bincode::config::standard();
        let (snapshot, _): (Snapshot, _) = bincode::serde::decode_from_slice(data, config)
            .map_err(|e| Error::Serialization(format!("snapshot restore failed: {e}")))?;

        // Validate all data before making any changes, and refuse data this
        // build cannot read, which it would restore the database without.
        refuse_unreadable_snapshot(&snapshot, "restore")?;
        validate_snapshot_data(&snapshot.nodes, &snapshot.edges)?;
        for ng in &snapshot.named_graphs {
            validate_snapshot_data(&ng.nodes, &ng.edges)?;
        }
        #[cfg(feature = "triple-store")]
        let rdf_graphs = read_rdf_snapshot(&snapshot.rdf_triples, &snapshot.rdf_named_graphs)?;

        // Drop all existing named graphs, then clear default store
        for name in self.lpg_store().graph_names() {
            self.lpg_store().drop_graph(&name);
        }
        self.lpg_store().clear();

        populate_store_from_snapshot(&self.lpg_store(), snapshot.nodes, snapshot.edges)?;

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
            populate_rdf_store(&self.rdf_store, rdf_graphs);
        }

        // Restore schema
        restore_schema_from_snapshot(&self.lpg_store(), &self.catalog, &snapshot.schema);

        // Restore indexes (must come after data population)
        restore_indexes_from_snapshot(self, &snapshot.indexes);

        Ok(())
    }

    // =========================================================================
    // ADMIN API: Iteration
    // =========================================================================

    /// Returns an iterator over all nodes in the database, as of the current
    /// epoch (see [`current_epoch`](Self::current_epoch)), in id order.
    ///
    /// Useful for dump/export operations.
    pub fn iter_nodes(&self) -> impl Iterator<Item = grafeo_core::graph::lpg::Node> + '_ {
        let epoch = self.read_epoch();
        let store = self.lpg_store();
        store
            .all_node_ids()
            .into_iter()
            .filter_map(move |id| store.get_node_at_epoch(id, epoch))
    }

    /// Returns an iterator over all edges in the database, as of the current
    /// epoch (see [`current_epoch`](Self::current_epoch)), in id order.
    ///
    /// Useful for dump/export operations.
    pub fn iter_edges(&self) -> impl Iterator<Item = grafeo_core::graph::lpg::Edge> + '_ {
        let epoch = self.read_epoch();
        let store = self.lpg_store();
        // The store numbers its edges densely.
        (0..store.next_edge_id()).filter_map(move |id| {
            store.get_edge_at_epoch(grafeo_common::types::EdgeId::new(id), epoch)
        })
    }
}

#[cfg(test)]
mod tests {
    use grafeo_common::types::{EdgeId, NodeId, Value};

    use super::super::GrafeoDB;
    use super::{
        SNAPSHOT_VERSION, Snapshot, SnapshotEdge, SnapshotIndexes, SnapshotNode, SnapshotSchema,
    };

    /// A database with Alix, whose embedding is spilled into a backing that
    /// cannot be read, and Gus, who has no embedding.
    #[cfg(not(feature = "temporal"))]
    fn with_an_unreadable_embedding() -> (GrafeoDB, NodeId) {
        use grafeo_common::types::PropertyKey;

        let db = GrafeoDB::new_in_memory();
        let alix = db
            .create_node_with_props(
                &["Item"],
                [
                    ("name", Value::from("Alix")),
                    ("embedding", Value::Vector(vec![3.0, 19.0].into())),
                ],
            )
            .unwrap();
        db.create_node_with_props(&["Item"], [("name", Value::from("Gus"))])
            .unwrap();
        assert!(db.export_snapshot().is_ok());
        let key = PropertyKey::new("embedding");
        let store = db.lpg_store();
        let snapshot = store.node_property_column_entries(&key).unwrap();
        assert!(store.spill_node_property_column(
            &key,
            std::sync::Arc::new(super::super::test_backing::Unreadable(alix)),
            &snapshot
        ));
        (db, alix)
    }

    /// An export reads spilled values, and fails rather than leave out one it
    /// cannot read (#594).
    #[cfg(not(feature = "temporal"))]
    #[test]
    fn an_export_fails_on_a_spilled_value_it_cannot_read() {
        let (db, _) = with_an_unreadable_embedding();
        assert!(
            db.export_snapshot().is_err(),
            "a snapshot without the embedding"
        );
        assert!(db.to_memory().is_err(), "a copy without the embedding");
    }

    /// A statement that would change a spilled value it cannot read errors and
    /// changes nothing: its rollback could not restore the value (#594).
    #[cfg(not(feature = "temporal"))]
    #[test]
    fn a_statement_on_a_spilled_value_it_cannot_read_errors() {
        let (db, alix) = with_an_unreadable_embedding();
        for statement in [
            "MATCH (n:Item {name: 'Alix'}) SET n.embedding = vector([88.0, 3.19])",
            "MATCH (n:Item {name: 'Alix'}) REMOVE n.embedding",
            "MATCH (n:Item {name: 'Alix'}) DELETE n",
        ] {
            let error = db.execute(statement).expect_err(statement);
            assert!(
                error.to_string().contains("cannot be read"),
                "{statement}: {error}"
            );
        }
        assert!(db.get_node(alix).is_some(), "Alix was not deleted");
        db.execute("MATCH (n:Item {name: 'Gus'}) SET n.embedding = vector([88.0, 3.19])")
            .unwrap();
    }

    /// A copy takes the nodes in one at a time: it reads a node's values just
    /// before it writes the node, so it never holds every node of the source
    /// at once (#594).
    #[cfg(not(feature = "temporal"))]
    #[test]
    fn a_copy_reads_each_node_just_before_it_writes_it() {
        use std::sync::Arc;

        use grafeo_common::types::PropertyKey;
        use grafeo_core::graph::lpg::{ColumnBacking, LpgStore};
        use parking_lot::Mutex;

        /// A backing that notes how many nodes the copy's target holds at
        /// each read.
        struct Watching {
            values: std::collections::HashMap<NodeId, Value>,
            target: Arc<LpgStore>,
            seen: Mutex<Vec<usize>>,
        }
        impl ColumnBacking<NodeId> for Watching {
            fn get(&self, id: NodeId) -> std::io::Result<Option<Value>> {
                self.seen.lock().push(self.target.node_count());
                Ok(self.values.get(&id).cloned())
            }
            fn contains(&self, id: NodeId) -> bool {
                self.values.contains_key(&id)
            }
            fn ids(&self) -> Vec<NodeId> {
                self.values.keys().copied().collect()
            }
            fn len(&self) -> usize {
                self.values.len()
            }
            fn heap_bytes(&self) -> usize {
                0
            }
        }

        let source = LpgStore::new().unwrap();
        for x in [3.0, 19.0, 88.0] {
            source
                .create_node_with_props(&["Item"], [("embedding", Value::Vector(vec![x].into()))]);
        }
        let key = PropertyKey::new("embedding");
        let snapshot = source.node_property_column_entries(&key).unwrap();
        let target = Arc::new(LpgStore::new().unwrap());
        let backing = Arc::new(Watching {
            values: snapshot.iter().cloned().collect(),
            target: Arc::clone(&target),
            seen: Mutex::new(Vec::new()),
        });
        assert!(source.spill_node_property_column(&key, backing.clone(), &snapshot));

        super::copy_graph_data(&source, &target).unwrap();
        assert_eq!(
            *backing.seen.lock(),
            vec![0, 1, 2],
            "nodes the target held at each read"
        );
        assert_eq!(target.node_count(), 3);
    }

    /// A direct call that removes a spilled value it cannot read (setting it
    /// to null, outside a transaction) errors and changes nothing: it hid the
    /// value and reported success while its WAL record and change event were
    /// left out (#594).
    #[cfg(not(feature = "temporal"))]
    #[test]
    fn a_direct_removal_of_a_spilled_value_it_cannot_read_errors() {
        use grafeo_common::types::PropertyKey;

        let (db, alix) = with_an_unreadable_embedding();
        let result = db.set_node_property(alix, "embedding", Value::Null);
        assert!(
            db.lpg_store()
                .node_property_column_ids(&PropertyKey::new("embedding"))
                .contains(&alix),
            "the call hid the value"
        );
        let error = result.expect_err("a removal of a value it cannot read");
        assert!(error.to_string().contains("cannot be read"), "{error}");
    }

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

    /// With `temporal`, copies taken while a transaction is open hold none of
    /// its versions: no property version of the copy is pending (it would
    /// never be finalized there), and no label it added is there (#412).
    #[cfg(feature = "temporal")]
    #[test]
    fn a_temporal_copy_holds_no_version_of_an_open_transaction() {
        use grafeo_common::types::EpochId;

        let db = GrafeoDB::new_in_memory();
        db.execute("INSERT (:Person {name: 'Alix', age: 19})")
            .unwrap();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute(
                "MATCH (p:Person {name: 'Alix'}) SET p.age = 88, p.city = 'Berlin', p:Traveller",
            )
            .unwrap();

        let copy = db.to_memory().unwrap();
        let imported = GrafeoDB::import_snapshot(&db.export_snapshot().unwrap()).unwrap();
        for (what, copy) in [("to_memory", &copy), ("export_snapshot", &imported)] {
            let store = copy.lpg_store();
            let ids = store.all_node_ids();
            assert_eq!(ids.len(), 1, "{what}: Alix");
            let mut history = store.node_property_history(ids[0]);
            history.sort_by(|(a, _), (b, _)| a.cmp(b));
            let keys: Vec<&str> = history.iter().map(|(key, _)| key.as_str()).collect();
            assert_eq!(keys, ["age", "name"], "{what}: the committed keys only");
            assert!(
                history.iter().all(|(_, versions)| versions
                    .iter()
                    .all(|(epoch, _)| *epoch != EpochId::PENDING)),
                "{what}: no pending version: {history:?}"
            );
            let labels = copy
                .execute("MATCH (p:Person) RETURN labels(p)")
                .unwrap()
                .rows()[0][0]
                .clone();
            assert_eq!(
                labels,
                Value::List(vec![Value::from("Person")].into()),
                "{what}: the committed labels"
            );
        }
        session.rollback().unwrap();
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
        let edge = db.create_edge(a, b, "KNOWS").unwrap();
        db.set_edge_property(edge, "since", Value::Int64(2020))
            .unwrap();

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
        db.create_edge(a, b, "R1").unwrap();
        db.create_edge(b, c, "R2").unwrap();

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

    /// A snapshot holding a value nested deeper than a database can store is
    /// refused before anything is imported, naming the node or edge and the
    /// property, as a write of the value is.
    #[test]
    fn a_snapshot_with_a_value_nested_too_deep_is_refused() {
        use grafeo_common::storage::value_codec::MAX_PROPERTY_VALUE_DEPTH;
        use grafeo_common::types::EpochId;

        let mut deep = Value::from("Berlin");
        for _ in 0..=MAX_PROPERTY_VALUE_DEPTH {
            deep = Value::List(vec![deep].into());
        }
        let node = |id: u64, value: &Value| SnapshotNode {
            id: NodeId::new(id),
            labels: vec!["City".into()],
            properties: vec![(
                "trips".into(),
                vec![
                    (EpochId::new(3), value.clone()),
                    (EpochId::new(19), Value::Int64(88)),
                ],
            )],
        };
        let deep_node = make_snapshot(
            SNAPSHOT_VERSION,
            vec![node(0, &Value::Null), node(3, &deep)],
            vec![],
        );
        let deep_edge = make_snapshot(
            SNAPSHOT_VERSION,
            vec![node(0, &Value::Null)],
            vec![SnapshotEdge {
                id: EdgeId::new(19),
                src: NodeId::new(0),
                dst: NodeId::new(0),
                edge_type: "ROUTE".into(),
                properties: vec![("stops".into(), vec![(EpochId::new(3), deep.clone())])],
            }],
        );
        for (bytes, words) in [
            (&deep_node, "node 3, property \"trips\""),
            (&deep_edge, "edge 19, property \"stops\""),
        ] {
            let error = GrafeoDB::import_snapshot(bytes).err().expect(words);
            assert!(
                matches!(error, grafeo_common::utils::error::Error::InvalidValue(_)),
                "an invalid value: {error:?}"
            );
            let error = error.to_string();
            assert!(error.contains(words), "{words}: {error}");
            let db = GrafeoDB::new_in_memory();
            db.execute("INSERT (:City {name: 'Amsterdam'})").unwrap();
            let error = db.restore_snapshot(bytes).unwrap_err().to_string();
            assert!(error.contains(words), "{words}: {error}");
            assert_eq!(
                db.lpg_store().node_count(),
                1,
                "a refused restore changes nothing"
            );
        }
    }

    /// The direct API and statements (through a parameter, the way a value
    /// deeper than any literal reaches them) refuse a value nested deeper
    /// than a database can store, and write nothing; the deepest one passes.
    #[test]
    fn writes_refuse_a_value_nested_too_deep() {
        use grafeo_common::storage::value_codec::MAX_PROPERTY_VALUE_DEPTH;

        let nested = |depth: usize| {
            let mut value = Value::from("Paris");
            for _ in 0..depth {
                value = Value::List(vec![value].into());
            }
            value
        };
        // A limit on input: an invalid value, never an internal error.
        let invalid = |error: grafeo_common::utils::error::Error, what: &str| {
            assert!(
                matches!(error, grafeo_common::utils::error::Error::InvalidValue(_)),
                "{what}: {error:?}"
            );
            let message = error.to_string();
            assert!(message.contains("\"trips\""), "{what}: {message}");
            assert!(message.contains("GRAFEO-V"), "{what}: {message}");
        };
        let db = GrafeoDB::new_in_memory();
        let alix = db.create_node(&["Person"]).unwrap();
        let too_deep = nested(MAX_PROPERTY_VALUE_DEPTH + 1);
        invalid(
            db.set_node_property(alix, "trips", too_deep.clone())
                .unwrap_err(),
            "set_node_property",
        );
        invalid(
            db.create_node_with_props(&["Person"], [("trips", too_deep.clone())])
                .unwrap_err(),
            "create_node_with_props",
        );
        let params = [("trips".to_string(), too_deep)].into_iter().collect();
        invalid(
            db.execute_with_params("INSERT (:Person {trips: $trips})", params)
                .unwrap_err(),
            "an INSERT parameter",
        );
        assert_eq!(db.node_count(), 1, "nothing was written");

        db.set_node_property(alix, "trips", nested(MAX_PROPERTY_VALUE_DEPTH))
            .unwrap();
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
        db.create_property_index("email").unwrap();
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
            .text_search("Article", "body", "graph database", 10, None)
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
        db.create_property_index("email").unwrap();

        let snapshot = db.export_snapshot().unwrap();

        // Mutate the database
        session
            .execute("INSERT (:Person {name: 'Gus', email: 'gus@example.com'})")
            .unwrap();
        db.drop_property_index("email").unwrap();
        assert!(!db.has_property_index("email"));

        // Restore should bring back the index
        db.restore_snapshot(&snapshot).unwrap();
        assert!(db.has_property_index("email"));
    }

    /// A `write_image` that fails, before or after the new header is
    /// written, leaves no file behind: the file `create` made would open as
    /// an empty database, and a retry would find the path taken. The retry
    /// then succeeds.
    #[cfg(all(
        feature = "wal",
        feature = "grafeo-file",
        feature = "testing-crash-injection"
    ))]
    #[test]
    fn a_failed_image_write_leaves_no_file_behind() {
        use grafeo_common::testing::crash::with_failure_at;

        let db = GrafeoDB::new_in_memory();
        db.create_node(&["City"]).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prague.grafeo");
        let points = [
            "checkpoint:after_chunks",
            "checkpoint:after_data_sync",
            "checkpoint:after_header",
            "checkpoint:before_trim",
        ];
        for (count, point) in (1..).zip(points) {
            let error = with_failure_at(count, || db.write_image(&path))
                .unwrap_err()
                .to_string();
            assert!(error.contains(point), "the original error: {error}");
            assert!(!path.exists(), "{point}: the failed image is removed");
        }

        db.write_image(&path).unwrap();
        let written = GrafeoDB::open(&path).unwrap();
        assert_eq!(written.node_count(), 1);
        written.close().unwrap();
    }
}
