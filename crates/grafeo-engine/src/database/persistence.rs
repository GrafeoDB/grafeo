//! Persistence, snapshots, and data export for GrafeoDB.

use grafeo_common::types::ReadView;
#[cfg(any(feature = "wal", feature = "grafeo-file"))]
use std::path::Path;

use grafeo_common::types::{EdgeId, EpochId, NodeId, Value};
use grafeo_common::utils::error::{Error, Result};
use hashbrown::HashSet;

use crate::config::Config;

use crate::catalog::{GraphTypeDefinition, ProcedureDefinition};

/// Current snapshot version: 5 holds the catalog as its records (the
/// catalog section's own, see `catalog_record`); 4, which 0.5.x and
/// 0.6.0-beta.1 wrote, its schema and index definitions, which an import
/// still reads.
const SNAPSHOT_VERSION: u8 = 5;

/// The snapshot version of 0.5.x and 0.6.0-beta.1.
const SNAPSHOT_V4: u8 = 4;

/// Binary snapshot format v5: the graph data of v4 (nodes and edges with
/// their property histories, named graphs, RDF), and the catalog as the
/// framed records the catalog section writes, so a snapshot holds every
/// definition a file holds (named constraints, index names, the options of
/// text and vector indexes, the indexes of every graph), except the stored
/// configuration, which belongs to the database file.
#[derive(serde::Serialize, serde::Deserialize)]
struct SnapshotV5 {
    version: u8,
    nodes: Vec<SnapshotNode>,
    edges: Vec<SnapshotEdge>,
    named_graphs: Vec<NamedGraphSnapshot>,
    rdf_triples: Vec<SnapshotTriple>,
    rdf_named_graphs: Vec<RdfNamedGraphSnapshot>,
    /// The catalog's records, framed.
    catalog: Vec<u8>,
    /// The epoch the snapshot holds the database at, which an import
    /// continues at (0 in the snapshots builds without `temporal` wrote
    /// before 0.6.0).
    epoch: u64,
}

/// A snapshot as read: its data in the v4 layout (a v5 snapshot's schema
/// and index definitions are empty there) and, for v5, its catalog records.
struct DecodedSnapshot {
    snapshot: Snapshot,
    records: Vec<grafeo_common::storage::CatalogRecord>,
}

/// Reads a snapshot of version 4 or 5, for `action` (`import`, `restore`).
///
/// # Errors
///
/// [`Error::InvalidValue`] for empty data or another version,
/// [`Error::Serialization`] when the data or the catalog records do not
/// decode.
fn decode_snapshot(data: &[u8], action: &str) -> Result<DecodedSnapshot> {
    let Some(&version) = data.first() else {
        return Err(Error::InvalidValue("empty snapshot data".to_string()));
    };
    let config = bincode::config::standard();
    let failed = |e: bincode::error::DecodeError| {
        Error::Serialization(format!("snapshot {action} failed: {e}"))
    };
    match version {
        SNAPSHOT_V4 => {
            let (snapshot, _): (Snapshot, _) =
                bincode::serde::decode_from_slice(data, config).map_err(failed)?;
            Ok(DecodedSnapshot {
                snapshot,
                records: Vec::new(),
            })
        }
        SNAPSHOT_VERSION => {
            let (snapshot, _): (SnapshotV5, _) =
                bincode::serde::decode_from_slice(data, config).map_err(failed)?;
            let mut records = Vec::new();
            grafeo_common::storage::read_catalog_records(
                &mut &snapshot.catalog[..],
                &mut |record| {
                    records.push(record);
                    Ok(())
                },
            )
            .map_err(|e| {
                Error::Serialization(format!("snapshot {action} failed: its catalog: {e}"))
            })?;
            Ok(DecodedSnapshot {
                snapshot: Snapshot {
                    version,
                    nodes: snapshot.nodes,
                    edges: snapshot.edges,
                    named_graphs: snapshot.named_graphs,
                    rdf_triples: snapshot.rdf_triples,
                    rdf_named_graphs: snapshot.rdf_named_graphs,
                    schema: SnapshotSchema::default(),
                    indexes: SnapshotIndexes::default(),
                    epoch: snapshot.epoch,
                },
                records,
            })
        }
        _ => Err(Error::InvalidValue(format!(
            "unsupported snapshot version: {version} (expected 4 or 5)"
        ))),
    }
}

/// Applies the catalog records of a v5 snapshot to `catalog` and to
/// `store`, the root store whose data is restored already, as the replay of
/// a WAL applies logged catalog puts (a definition the catalog holds already
/// is replaced or kept). As in the load of a catalog section, a constraint
/// record restores only the name: the node type records hold the type
/// constraints that enforce it. A stored configuration in the records is
/// left out: it belongs to the database file. Returns the vector and text
/// indexes they define, for the caller to build from the data.
///
/// # Errors
///
/// The error of a record the catalog refuses.
pub(super) fn apply_catalog_records<S: super::root_store::RootStore>(
    store: &std::sync::Arc<S>,
    catalog: &crate::catalog::Catalog,
    records: Vec<grafeo_common::storage::CatalogRecord>,
) -> Result<Vec<super::catalog_section::GraphIndexes>> {
    use super::catalog_records::constraint_definition;
    use super::standalone::{Applying, apply};
    use grafeo_common::change::StandaloneOp;
    use grafeo_common::storage::CatalogRecord;

    let mut unbuilt = Vec::new();
    let mut constraints = Vec::new();
    for record in records {
        let record = match record {
            CatalogRecord::StoredConfig(_) => continue,
            CatalogRecord::Constraint(constraint) => {
                constraints.push(constraint_definition(constraint));
                continue;
            }
            record => record,
        };
        apply(
            &StandaloneOp::PutCatalog(record),
            None,
            Some(store),
            catalog,
            &mut Applying::Snapshot {
                unbuilt: &mut unbuilt,
            },
        )?;
    }
    catalog.restore_constraint_names(constraints);
    Ok(unbuilt)
}

/// Binary snapshot format v4 (graph data, named graphs, RDF, schema, index
/// metadata, and property version history for temporal support), which an
/// import still reads. Removed in 0.7.0.
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
    node_types: Vec<super::catalog_records::LegacyNodeType>,
    edge_types: Vec<super::catalog_records::LegacyEdgeType>,
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

/// Populates a store from snapshot node/edge data, each property's versions
/// oldest first (see [`load_versions`](grafeo_core::graph::lpg::load_versions)):
/// the row-group store keeps the latest as the value.
fn populate_store_from_snapshot<S: grafeo_core::graph::load::LoadTarget>(
    store: &S,
    nodes: &[SnapshotNode],
    edges: &[SnapshotEdge],
) -> Result<()> {
    use grafeo_common::change::Entity;
    use grafeo_core::graph::lpg::load_versions;

    for node in nodes {
        let label_refs: Vec<&str> = node.labels.iter().map(|s| s.as_str()).collect();
        store.load_node(node.id, &label_refs)?;
        load_versions(store, Entity::Node(node.id), &node.properties)?;
    }
    for edge in edges {
        store.load_edge(edge.id, edge.src, edge.dst, &edge.edge_type)?;
        load_versions(store, Entity::Edge(edge.id), &edge.properties)?;
    }
    store.load_finish();
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
fn refuse_unreadable_snapshot(decoded: &DecodedSnapshot, action: &str) -> Result<()> {
    use super::sections::{FeatureData, RDF_TRIPLES, TEXT_INDEXES, VECTOR_INDEXES, refusal_of};
    use grafeo_common::storage::CatalogRecord;
    use grafeo_common::storage::catalog_record::{IndexKindRecord, IndexRecord};

    let snapshot = &decoded.snapshot;
    // The vector (`true`) and text index definitions of the v5 records.
    let index_records = |vector: bool| {
        decoded
            .records
            .iter()
            .filter_map(move |record| match record {
                CatalogRecord::Index(IndexRecord {
                    index:
                        IndexKindRecord::Vector {
                            label, property, ..
                        },
                    ..
                }) if vector => Some((label.as_str(), property.as_str())),
                CatalogRecord::Index(IndexRecord {
                    index:
                        IndexKindRecord::Text {
                            label, property, ..
                        },
                    ..
                }) if !vector => Some((label.as_str(), property.as_str())),
                _ => None,
            })
    };

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
        for (label, property) in index_records(true) {
            push_index_on(&mut held, label, property);
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
        for (label, property) in index_records(false) {
            push_index_on(&mut held, label, property);
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
    let mut triples: Vec<SnapshotTriple> = store
        .triples()
        .into_iter()
        .map(|t| SnapshotTriple {
            subject: t.subject().to_string(),
            predicate: t.predicate().to_string(),
            object: t.object().to_string(),
        })
        .collect();
    // In a fixed order, as the export of the same data gives the same bytes.
    triples.sort_unstable_by(|a, b| {
        (&a.subject, &a.predicate, &a.object).cmp(&(&b.subject, &b.predicate, &b.object))
    });
    triples
}

/// The RDF data of a snapshot of `store`: the default graph's triples, and
/// each named graph's, by name.
#[cfg(feature = "triple-store")]
fn rdf_snapshot(
    store: &grafeo_core::graph::rdf::RdfStore,
) -> (Vec<SnapshotTriple>, Vec<RdfNamedGraphSnapshot>) {
    let mut names = store.graph_names();
    names.sort_unstable();
    let named = names
        .into_iter()
        .filter_map(|name| {
            store.graph(&name).map(|graph| RdfNamedGraphSnapshot {
                name,
                triples: collect_rdf_triples(&graph),
            })
        })
        .collect();
    (collect_rdf_triples(store), named)
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
pub(super) fn load_snapshot_into_store<S: super::root_store::RootStore>(
    path: &Path,
    store: &std::sync::Arc<S>,
    catalog: &std::sync::Arc<crate::catalog::Catalog>,
    #[cfg(feature = "triple-store")] rdf_store: &std::sync::Arc<grafeo_core::graph::rdf::RdfStore>,
    data: &[u8],
) -> grafeo_common::utils::error::Result<super::catalog_section::GraphIndexes> {
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
    // Its indexes are built once its data is in, as a 0.6 file's are: a
    // build that cannot build them does not open it, as with other data of
    // a feature it lacks.
    let missing: Vec<_> = [
        (
            cfg!(not(feature = "vector-index")) && !snapshot.indexes.vector_indexes.is_empty(),
            &super::sections::VECTOR_INDEXES,
        ),
        (
            cfg!(not(feature = "text-index")) && !snapshot.indexes.text_indexes.is_empty(),
            &super::sections::TEXT_INDEXES,
        ),
    ]
    .into_iter()
    .filter(|(lacking, _)| *lacking)
    .map(|(_, feature)| (feature, "the snapshot of a 0.5.x file".to_string()))
    .collect();
    if !missing.is_empty() {
        return Err(super::sections::refusal(path, &missing));
    }

    populate_store_from_snapshot(&**store, &snapshot.nodes, &snapshot.edges)?;

    // Restore epoch from snapshot (store-level only; TransactionManager
    // sync is handled in with_config() after all recovery completes).
    store.load_section_epoch(EpochId::new(snapshot.epoch));

    for graph in &snapshot.named_graphs {
        let graph_store = store
            .load_named_graph(&graph.name)
            .map_err(|e| Error::Internal(e.to_string()))?;
        populate_store_from_snapshot(&*graph_store, &graph.nodes, &graph.edges)?;
        graph_store.load_section_epoch(EpochId::new(snapshot.epoch));
    }
    restore_schema_from_snapshot(store, catalog, &snapshot.schema)?;

    // Restore RDF triples
    #[cfg(feature = "triple-store")]
    {
        populate_rdf_store(rdf_store, rdf_graphs);
    }

    // A property index is built at once, as replay builds one; the vector
    // and text indexes once the data is in (see `LoadedSections::unbuilt_mut`).
    for key in &snapshot.indexes.property_indexes {
        store.create_property_index(key);
    }
    Ok(row_groups::indexes_of_v4(snapshot.indexes))
}

/// Restores schema definitions from a snapshot into the catalog.
///
/// Also ensures each schema has its `__default__` graph partition, which
/// may be missing in snapshots created before the schema hierarchy feature.
///
/// # Errors
///
/// A corruption error for a definition that does not apply: a graph type
/// defined twice, a procedure or schema the catalog refuses, a default
/// graph the store cannot make. A schema listed twice is no damage, and a
/// binding to a graph type the snapshot lacks is left out (0.5.x left one
/// behind when its graph type was dropped).
fn restore_schema_from_snapshot<S: super::catalog_section::CatalogStore>(
    store: &std::sync::Arc<S>,
    catalog: &std::sync::Arc<crate::catalog::Catalog>,
    schema: &SnapshotSchema,
) -> grafeo_common::utils::error::Result<()> {
    use crate::catalog::CatalogError;
    use grafeo_common::utils::error::Error;

    let damaged = |what: String| Error::corruption(format!("the snapshot's schema: {what}"));
    for def in &schema.node_types {
        let def = def.definition()?;
        catalog.register_or_replace_node_type(def);
    }
    for def in &schema.edge_types {
        let def = def.definition()?;
        catalog.register_or_replace_edge_type_def(def);
    }
    for def in &schema.graph_types {
        let name = def.name.clone();
        catalog
            .register_graph_type(def.clone())
            .map_err(|error| damaged(format!("graph type '{name}': {error}")))?;
    }
    for def in &schema.procedures {
        let name = def.name.clone();
        catalog
            .replace_procedure(def.clone())
            .map_err(|error| damaged(format!("procedure '{name}': {error}")))?;
    }
    for name in &schema.schemas {
        match catalog.register_schema_namespace(name.clone()) {
            Ok(()) | Err(CatalogError::SchemaAlreadyExists(_)) => {}
            Err(error) => return Err(damaged(format!("schema '{name}': {error}"))),
        }
        // Ensure the schema's default graph partition exists
        let default_key = format!("{name}/__default__");
        store
            .create_graph(&default_key)
            .map_err(|error| damaged(format!("the default graph of schema '{name}': {error}")))?;
    }
    for (graph_name, type_name) in &schema.graph_type_bindings {
        match catalog.bind_graph_type(graph_name, type_name.clone()) {
            Ok(()) | Err(CatalogError::TypeNotFound(_)) => {}
            Err(error) => {
                return Err(damaged(format!(
                    "the binding of graph '{graph_name}' to '{type_name}': {error}"
                )));
            }
        }
    }
    Ok(())
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
    /// Commits wait only while the copy takes its snapshot and the catalog
    /// and RDF data, and go on while it writes the graphs. The file is
    /// written as `<path>.partial` and renamed to `path` once complete.
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
    /// the database-closed error after `close()` of a persistent database;
    /// the unsupported error on a database on a backend of its own
    /// ([`with_store`](Self::with_store)), whose rows the backend keeps.
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
    /// The image holds the state at an epoch E, and commits go on while it
    /// is written (see `flush::write_copy`): it is written under a temporary
    /// name and renamed to `path` once complete, so `path` never opens as less
    /// than E.
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
        use grafeo_storage::file::v3::header::new_database_id;

        // An image of the state at an epoch, written with commits going on
        // (#362): they are held off only while the snapshot is taken. A new
        // database: its WAL starts at log position 0.
        super::flush::write_copy(
            &self.checkpoint_sources()?,
            path,
            new_database_id(),
            keys,
            || Ok(0),
        )
        .map(drop)
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
    /// did not complete; the unsupported error on a database on a backend of
    /// its own ([`with_store`](Self::with_store)).
    pub fn to_memory(&self) -> Result<Self> {
        // The copy keeps the backward edges this database keeps (10b), and
        // its history retention, with the history its sections hold.
        let mut config = Config::in_memory()
            .with_history_retention(self.transaction_manager.history_retention());
        if self
            .root_store()
            .is_some_and(|store| !store.keeps_backward_edges())
        {
            config = config.without_backward_edges();
        }
        let target = Self::with_config(config)?;
        // Each section is served once and freed as soon as it is loaded.
        let image = grafeo_common::storage::ServedOnce::new(self.copy_into(&target)?);
        let target_root = target.lpg_store()?;
        let mut loaded = super::sections::load_sections(
            &image,
            None,
            &target_root,
            &target.catalog,
            #[cfg(feature = "triple-store")]
            &target.rdf_store,
        )?;
        // The copy continues at the epoch of what it holds, which
        // `copy_into` gave its transaction manager, as a reopen of a
        // checkpoint continues at the checkpoint's epoch.
        target_root.sync_epoch(target.transaction_manager.current_epoch());
        super::row_group_sections::restore_row_group_history(
            &target_root,
            &target.transaction_manager,
            loaded.take_history(),
        );
        Self::continue_epochs(&target_root, &target.transaction_manager);
        target.finish_load(loaded);
        Ok(target)
    }

    /// Returns an image of every section of a checkpoint of this database,
    /// for [`to_memory`](Self::to_memory) to load; `target`'s transaction
    /// manager takes the epoch of the commits they hold.
    ///
    /// The sections are taken from one snapshot at an epoch E, as a
    /// checkpoint takes it, so they hold every commit up to E whole, none
    /// that did not complete, and nothing of a transaction still open: the
    /// LPG section writes every graph as a reader at E sees it. Commits are
    /// held off only while the snapshot is taken and the sections without
    /// versions are written, and go on while the LPG section is (#362).
    ///
    /// # Errors
    ///
    /// Returns an error if commits cannot be held or a section fails to
    /// write.
    pub(super) fn copy_into(&self, target: &Self) -> Result<grafeo_common::storage::MemoryImage> {
        let sources = self.checkpoint_sources()?;
        let commits = self.transaction_manager.hold_commits()?;
        #[cfg(feature = "grafeo-file")]
        {
            let mut image = grafeo_common::storage::MemoryImage::new();
            // The read pinned at E lives with `snapshot` until the copy is
            // written.
            let snapshot = sources.copy_snapshot(&commits)?;
            target
                .transaction_manager
                .sync_epoch(EpochId::new(snapshot.context.epoch));
            let mut commits = Some(commits);
            for (index, section) in snapshot.sections.iter().enumerate() {
                if index == snapshot.held {
                    // The LPG section is read at E with commits going on.
                    drop(commits.take());
                }
                image.begin_section(section.section_type(), section.version())?;
                section.write_to(&mut image)?;
            }
            Ok(image)
        }
        #[cfg(not(feature = "grafeo-file"))]
        {
            self.copy_held(target, &sources, &commits)
        }
    }

    /// [`copy_into`](Self::copy_into) without the container format, whose
    /// snapshot a copy takes: every section under the hold.
    #[cfg(not(feature = "grafeo-file"))]
    fn copy_held(
        &self,
        target: &Self,
        sources: &super::sections::CheckpointSources,
        commits: &crate::transaction::CommitsHeld<'_>,
    ) -> Result<grafeo_common::storage::MemoryImage> {
        let mut image = grafeo_common::storage::MemoryImage::new();
        // The epoch of the commits the copy holds, for `to_memory` to
        // continue from once its stores are loaded.
        let epoch = sources.epoch(commits);
        target.transaction_manager.sync_epoch(epoch);
        for section in sources.sections_at(commits, epoch) {
            image.begin_section(section.section_type(), section.version())?;
            section.write_to(&mut image)?;
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
    /// Properties are stored as version-history lists, each holding the
    /// value a reader at the export's epoch sees, at epoch 0. The snapshot
    /// holds the export's epoch, which an import or restore continues at.
    ///
    /// The snapshot holds the committed state: what a transaction still open
    /// deleted is in it, with the values and labels it changed as they were
    /// committed, and nothing it created is. Every graph is read as a
    /// committed reader at the export's epoch sees it, so no copy is made.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails, or after a commit that did
    /// not complete; the unsupported error on a database on a backend of its
    /// own ([`with_store`](Self::with_store)).
    pub fn export_snapshot(&self) -> Result<Vec<u8>> {
        // The snapshot holds every commit whole, none that did not complete
        // (whose stamped part the store holds), and nothing of a transaction
        // still open (the hold also holds its writes).
        let sources = self.checkpoint_sources()?;
        let commits = self.transaction_manager.hold_commits()?;
        let root = std::sync::Arc::clone(&sources.store);
        let epoch = sources.epoch(&commits);
        // The graphs are read at the epoch with commits going on, while a
        // read pinned at it keeps their versions (#362); the rest is taken
        // under the hold.
        let pin = self.transaction_manager.pin_read_at(epoch)?;
        let held = row_groups::export_held(&self.snapshot_parts(&root));
        drop(commits);
        #[cfg(feature = "testing-statement-injection")]
        grafeo_common::testing::commit_hook::run_during_checkpoint_write();
        let exported = row_groups::export_rows(held?, epoch);
        drop(pin);
        exported
    }

    /// The parts of this database a snapshot reads and writes.
    fn snapshot_parts<'a>(
        &'a self,
        root: &'a std::sync::Arc<grafeo_core::graph::rowgroup::RowGroupStore>,
    ) -> row_groups::Parts<'a> {
        row_groups::Parts {
            root,
            catalog: &self.catalog,
            #[cfg(feature = "triple-store")]
            rdf: &self.rdf_store,
        }
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
        // Decoded, validated and refused (data this build cannot read)
        // before the database is created.
        let decoded = decode_snapshot(data, "import")?;
        refuse_unreadable_snapshot(&decoded, "import")?;
        let db = Self::new_in_memory();
        let root = db.lpg_store()?;
        // Restored from the snapshot as decoded: decoding it again would
        // cost as much as the rest of a small import.
        row_groups::restore_decoded(&db.snapshot_parts(&root), decoded)?;
        // The database continues at the snapshot's epoch.
        Self::continue_epochs(&root, &db.transaction_manager);
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
        // A restore writes no WAL record but a checkpoint: after `close()`
        // (which releases the file) nothing would persist it, so it fails
        // then, and `close()` waits for one in progress.
        let _open = self.hold_open()?;
        if self.read_only {
            return Err(Error::Transaction(
                grafeo_common::utils::error::TransactionError::ReadOnly,
            ));
        }
        // A checkpoint reads the stores while it writes its image, with
        // commits going on: a restore replaces them whole, so it waits for
        // one in progress, and the next one waits for it. Commits wait for
        // the restore too, from the first store it replaces to the
        // checkpoint that makes it durable (#677).
        #[cfg(feature = "grafeo-file")]
        let _checkpoint = self.file_manager.as_ref().map(|fm| fm.checkpoint_guard());
        let commits = self.transaction_manager.hold_commits()?;
        self.transaction_manager.check_open()?;
        // Validates the snapshot and refuses data this build cannot read
        // before it changes anything (see `row_groups::restore`), and a
        // snapshot that does not apply whole (a catalog record that fails) is
        // refused before anything changes too: a restore into an emptied
        // database that stopped halfway would be checkpointed so (#677).
        row_groups::trial_restore(data)?;
        let root = self.lpg_store()?;
        row_groups::restore(&self.snapshot_parts(&root), data, "restore")?;
        // The database continues at the snapshot's epoch, or above it when
        // it had committed further (epochs never go back).
        Self::continue_epochs(&root, &self.transaction_manager);
        // Its graphs, indexes and types are the snapshot's now: no cached
        // plan was made for them.
        self.query_cache.clear();
        // Durable before it returns: the file holds the restored state and
        // the WAL starts after it, with no commit between, so a crash from
        // here reopens what the restore put in place, not what it replaced.
        #[cfg(feature = "grafeo-file")]
        if let Some(fm) = &self.file_manager {
            super::flush::flush_held(fm, &self.checkpoint_sources()?, None, commits)?;
        }
        #[cfg(not(feature = "grafeo-file"))]
        drop(commits);
        Ok(())
    }

    // =========================================================================
    // ADMIN API: Iteration
    // =========================================================================

    /// Returns an iterator over all nodes in the database, as of the current
    /// epoch (see [`current_epoch`](Self::current_epoch)), in id order.
    ///
    /// Useful for dump/export operations.
    ///
    /// The read is pinned at its epoch while the iterator lives, so a
    /// collect between two steps keeps every node it still reads.
    ///
    /// # Errors
    ///
    /// The unsupported error on a database on a store of its own: it walks
    /// the built-in store.
    pub fn iter_nodes(&self) -> Result<impl Iterator<Item = grafeo_core::graph::lpg::Node> + '_> {
        use grafeo_core::graph::GraphStore;

        let store = self.lpg_store()?;
        let pin = self.transaction_manager.pin_read();
        let view = ReadView::committed(self.read_epoch());
        // The database's own row-group store: its reads do not fail.
        let ids = store.node_ids(view).unwrap_or_default();
        Ok(ids.into_iter().filter_map(move |id| {
            let _pinned = &pin;
            store.node(id, view).ok().flatten()
        }))
    }

    /// Returns an iterator over all edges in the database, as of the current
    /// epoch (see [`current_epoch`](Self::current_epoch)), in id order.
    ///
    /// Useful for dump/export operations.
    ///
    /// The read is pinned at its epoch while the iterator lives (see
    /// [`iter_nodes`](Self::iter_nodes)).
    ///
    /// # Errors
    ///
    /// As [`iter_nodes`](Self::iter_nodes).
    pub fn iter_edges(&self) -> Result<impl Iterator<Item = grafeo_core::graph::lpg::Edge> + '_> {
        use grafeo_core::graph::GraphStore;

        let store = self.lpg_store()?;
        let pin = self.transaction_manager.pin_read();
        let epoch = self.read_epoch();
        // The store numbers its edges densely, and its reads do not fail.
        Ok((0..store.next_edge_id()).filter_map(move |id| {
            let _pinned = &pin;
            store
                .edge(
                    grafeo_common::types::EdgeId::new(id),
                    ReadView::committed(epoch),
                )
                .ok()
                .flatten()
        }))
    }
}

/// The snapshot paths on the row-group store (#412 P5): export, import and
/// restore of a database's root store and its named graphs, with the
/// snapshot layout, validation and refusals of [`GrafeoDB::export_snapshot`],
/// [`GrafeoDB::import_snapshot`] and [`GrafeoDB::restore_snapshot`].
///
/// An export reads each graph as a committed reader at the export's epoch
/// sees it, so it needs no committed copy while a transaction is open; it
/// holds each value once, and the export's epoch, which an import continues
/// at. An import or restore loads the rows through the replay writer
/// ([`LoadTarget`](grafeo_core::graph::load::LoadTarget)), keeping the
/// latest of each property's versions.
///
/// [`GrafeoDB::export_snapshot`]: super::GrafeoDB::export_snapshot
/// [`GrafeoDB::import_snapshot`]: super::GrafeoDB::import_snapshot
/// [`GrafeoDB::restore_snapshot`]: super::GrafeoDB::restore_snapshot
pub(super) mod row_groups {
    use std::sync::Arc;

    use grafeo_common::types::EpochId;
    use grafeo_common::utils::error::{Error, Result};
    use grafeo_core::graph::load::LoadTarget;
    use grafeo_core::graph::lpg::{Edge, Node};
    use grafeo_core::graph::rowgroup::RowGroupStore;

    use super::super::catalog_section::{
        CatalogSection, GraphIndexes, TextIndexDefinition, VectorIndexDefinition,
    };
    use super::{
        DecodedSnapshot, NamedGraphSnapshot, SNAPSHOT_VERSION, SnapshotEdge, SnapshotIndexes,
        SnapshotNode, SnapshotV5, apply_catalog_records, decode_snapshot,
        populate_store_from_snapshot, refuse_unreadable_snapshot, restore_schema_from_snapshot,
        validate_snapshot_data,
    };
    use crate::catalog::Catalog;

    /// The parts of a database a snapshot reads and writes.
    pub(in crate::database) struct Parts<'a> {
        /// The root store.
        pub root: &'a Arc<RowGroupStore>,
        /// The catalog.
        pub catalog: &'a Arc<Catalog>,
        /// The RDF store.
        #[cfg(feature = "triple-store")]
        pub rdf: &'a Arc<grafeo_core::graph::rdf::RdfStore>,
    }

    /// A node as a snapshot holds it: labels sorted, each value once at
    /// epoch 0, as a build without `temporal` writes it.
    fn snapshot_node(node: Node) -> SnapshotNode {
        let mut labels: Vec<String> = node.labels.iter().map(ToString::to_string).collect();
        labels.sort();
        let mut properties: Vec<(String, Vec<(EpochId, grafeo_common::types::Value)>)> = node
            .properties
            .into_iter()
            .map(|(key, value)| (key.to_string(), vec![(EpochId::new(0), value)]))
            .collect();
        properties.sort_by(|(a, _), (b, _)| a.cmp(b));
        SnapshotNode {
            id: node.id,
            labels,
            properties,
        }
    }

    /// An edge as a snapshot holds it (see [`snapshot_node`]).
    fn snapshot_edge(edge: Edge) -> SnapshotEdge {
        let mut properties: Vec<(String, Vec<(EpochId, grafeo_common::types::Value)>)> = edge
            .properties
            .into_iter()
            .map(|(key, value)| (key.to_string(), vec![(EpochId::new(0), value)]))
            .collect();
        properties.sort_by(|(a, _), (b, _)| a.cmp(b));
        SnapshotEdge {
            id: edge.id,
            src: edge.src,
            dst: edge.dst,
            edge_type: edge.edge_type.to_string(),
            properties,
        }
    }

    /// What an export takes while the caller holds commits off: the graphs,
    /// the RDF triples and the catalog's records, which have no versions to
    /// be read at the epoch later. [`export_rows`] reads the graphs' rows at
    /// the epoch afterwards, with commits going on, while a read pinned at
    /// it keeps their versions (#362).
    pub(in crate::database) struct ExportHeld {
        root: Arc<RowGroupStore>,
        /// The named graphs, by name.
        named: Vec<(String, Arc<RowGroupStore>)>,
        rdf_triples: Vec<super::SnapshotTriple>,
        rdf_named_graphs: Vec<super::RdfNamedGraphSnapshot>,
        catalog: Vec<u8>,
    }

    /// Takes the [`ExportHeld`] of `parts`, while the caller holds commits
    /// off.
    ///
    /// # Errors
    ///
    /// The error of a catalog record that does not encode.
    pub(in crate::database) fn export_held(parts: &Parts<'_>) -> Result<ExportHeld> {
        let root = parts.root;
        let mut names = root.graph_names();
        names.sort_unstable();
        let named = names
            .into_iter()
            .filter_map(|name| {
                let graph = root.graph(&name)?;
                Some((name, graph))
            })
            .collect();
        #[cfg(feature = "triple-store")]
        let (rdf_triples, rdf_named_graphs) = super::rdf_snapshot(parts.rdf);
        #[cfg(not(feature = "triple-store"))]
        let (rdf_triples, rdf_named_graphs) = (Vec::new(), Vec::new());
        let catalog = CatalogSection::new(Arc::clone(parts.catalog), Arc::clone(root), || 0)
            .snapshot_records()?;
        Ok(ExportHeld {
            root: Arc::clone(root),
            named,
            rdf_triples,
            rdf_named_graphs,
            catalog,
        })
    }

    /// The snapshot of version 5 of `held` as a committed reader at `epoch`
    /// sees it: every graph's rows at `epoch`, the RDF triples and the
    /// catalog's records with every graph's indexes. The caller pins a read
    /// at `epoch`, so its versions stay until the rows are read.
    ///
    /// # Errors
    ///
    /// The error of the encoding.
    pub(in crate::database) fn export_rows(held: ExportHeld, epoch: EpochId) -> Result<Vec<u8>> {
        let ExportHeld {
            root,
            named,
            rdf_triples,
            rdf_named_graphs,
            catalog,
        } = held;
        let nodes = root.nodes_at(epoch).map(snapshot_node).collect();
        let edges = root.edges_at(epoch).map(snapshot_edge).collect();
        let named_graphs = named
            .into_iter()
            .map(|(name, graph)| NamedGraphSnapshot {
                name,
                nodes: graph.nodes_at(epoch).map(snapshot_node).collect(),
                edges: graph.edges_at(epoch).map(snapshot_edge).collect(),
            })
            .collect();
        let snapshot = SnapshotV5 {
            version: SNAPSHOT_VERSION,
            nodes,
            edges,
            named_graphs,
            rdf_triples,
            rdf_named_graphs,
            catalog,
            epoch: epoch.as_u64(),
        };
        bincode::serde::encode_to_vec(&snapshot, bincode::config::standard())
            .map_err(|e| Error::Internal(format!("snapshot export failed: {e}")))
    }

    /// The default graph's indexes a snapshot of version 4 (and a 0.5.x
    /// container v1 file) defines.
    pub(super) fn indexes_of_v4(indexes: SnapshotIndexes) -> GraphIndexes {
        GraphIndexes {
            graph: None,
            property: indexes.property_indexes,
            vector: indexes
                .vector_indexes
                .into_iter()
                .map(|def| VectorIndexDefinition {
                    label: def.label,
                    property: def.property,
                    dimensions: def.dimensions,
                    metric: def.metric,
                    m: def.m,
                    ef_construction: def.ef_construction,
                    quantization: None,
                })
                .collect(),
            text: indexes
                .text_indexes
                .into_iter()
                .map(|def| TextIndexDefinition {
                    label: def.label,
                    property: def.property,
                    options: grafeo_core::index::text::TextIndexOptions::default(),
                })
                .collect(),
        }
    }

    /// Replaces the database's contents with the snapshot `data`, for
    /// `action` (`import` into an empty database, `restore`): validates it
    /// and refuses data this build cannot read before anything changes, then
    /// drops every named graph and clears the root, loads every graph's rows
    /// and the RDF triples, continues at the snapshot's epoch, and applies
    /// its schema, index definitions and catalog records, building the
    /// indexes from the rows.
    ///
    /// # Errors
    ///
    /// As `import_snapshot` and `restore_snapshot`: a snapshot that does not
    /// decode or validate, data this build cannot read, a row or record the
    /// stores refuse.
    /// Restores the snapshot `data` into a database of its own, which is
    /// dropped: whether every row and record of it applies, so a restore that
    /// would fail halfway is refused before it changes the database. It holds
    /// the snapshot's data a second time while it runs.
    ///
    /// # Errors
    ///
    /// As [`restore`].
    pub(in crate::database) fn trial_restore(data: &[u8]) -> Result<()> {
        let root = Arc::new(RowGroupStore::new());
        let catalog = Arc::new(Catalog::new());
        #[cfg(feature = "triple-store")]
        let rdf = Arc::new(grafeo_core::graph::rdf::RdfStore::new());
        let parts = Parts {
            root: &root,
            catalog: &catalog,
            #[cfg(feature = "triple-store")]
            rdf: &rdf,
        };
        restore(&parts, data, "restore")
    }

    pub(in crate::database) fn restore(parts: &Parts<'_>, data: &[u8], action: &str) -> Result<()> {
        let decoded = decode_snapshot(data, action)?;
        refuse_unreadable_snapshot(&decoded, action)?;
        restore_decoded(parts, decoded)
    }

    /// As [`restore`], for a snapshot already decoded and checked for data
    /// this build cannot read (`refuse_unreadable_snapshot`): validates its
    /// rows before anything changes, then replaces the database's contents.
    ///
    /// # Errors
    ///
    /// As [`restore`], except the decoding and the refusal, which the
    /// caller did.
    pub(super) fn restore_decoded(parts: &Parts<'_>, decoded: DecodedSnapshot) -> Result<()> {
        let DecodedSnapshot { snapshot, records } = decoded;
        validate_snapshot_data(&snapshot.nodes, &snapshot.edges)?;
        for graph in &snapshot.named_graphs {
            validate_snapshot_data(&graph.nodes, &graph.edges)?;
        }
        #[cfg(feature = "triple-store")]
        let rdf_graphs =
            super::read_rdf_snapshot(&snapshot.rdf_triples, &snapshot.rdf_named_graphs)?;

        let root = parts.root;
        for name in root.graph_names() {
            root.drop_graph(&name);
        }
        root.clear();
        let epoch = EpochId::new(snapshot.epoch);
        populate_store_from_snapshot(&**root, &snapshot.nodes, &snapshot.edges)?;
        root.load_section_epoch(epoch);
        for graph in &snapshot.named_graphs {
            let store = root.load_named_graph(&graph.name)?;
            populate_store_from_snapshot(&*store, &graph.nodes, &graph.edges)?;
            store.load_section_epoch(epoch);
        }
        #[cfg(feature = "triple-store")]
        {
            parts.rdf.clear();
            for name in parts.rdf.graph_names() {
                parts.rdf.drop_graph(&name);
            }
            super::populate_rdf_store(parts.rdf, rdf_graphs);
        }
        // The snapshot's definitions in place of the database's, not merged
        // into them: a restore gives the database the snapshot was taken of.
        parts.catalog.clear_definitions();
        restore_schema_from_snapshot(root, parts.catalog, &snapshot.schema)?;
        let mut indexes = vec![indexes_of_v4(snapshot.indexes)];
        indexes.extend(apply_catalog_records(root, parts.catalog, records)?);
        super::super::row_group_sections::restore_row_group_indexes(root, indexes);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use grafeo_common::types::ReadView;
    use grafeo_common::types::{EdgeId, NodeId, Value};
    use grafeo_core::graph::GraphStore as _;

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
        assert_eq!(
            db.lpg_store()
                .unwrap()
                .node_count(ReadView::latest())
                .unwrap(),
            3
        );

        // Restore original
        db.restore_snapshot(&snapshot).unwrap();

        assert_eq!(
            db.lpg_store()
                .unwrap()
                .node_count(ReadView::latest())
                .unwrap(),
            2
        );
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
        assert_eq!(
            db.lpg_store()
                .unwrap()
                .node_count(ReadView::latest())
                .unwrap(),
            1
        );
    }

    #[test]
    fn test_restore_snapshot_empty_db() {
        let db = GrafeoDB::new_in_memory();

        // Export empty snapshot, then populate, then restore to empty
        let empty_snapshot = db.export_snapshot().unwrap();

        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        assert_eq!(
            db.lpg_store()
                .unwrap()
                .node_count(ReadView::latest())
                .unwrap(),
            1
        );

        db.restore_snapshot(&empty_snapshot).unwrap();
        assert_eq!(
            db.lpg_store()
                .unwrap()
                .node_count(ReadView::latest())
                .unwrap(),
            0
        );
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
        assert_eq!(
            db.lpg_store()
                .unwrap()
                .edge_count(ReadView::latest())
                .unwrap(),
            1
        );

        // Modify: add more data
        session
            .execute("INSERT (:Person {name: 'Vincent'})")
            .unwrap();

        // Restore
        db.restore_snapshot(&snapshot).unwrap();
        assert_eq!(
            db.lpg_store()
                .unwrap()
                .node_count(ReadView::latest())
                .unwrap(),
            2
        );
        assert_eq!(
            db.lpg_store()
                .unwrap()
                .edge_count(ReadView::latest())
                .unwrap(),
            1
        );
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

    /// Copies taken while a transaction is open hold none of
    /// its versions: no property version of the copy is pending (it would
    /// never be finalized there), and no label it added is there (#412).
    #[test]
    fn a_copy_holds_no_version_of_an_open_transaction() {
        use grafeo_common::types::EpochId;

        // History kept: a read at an earlier epoch needs it.
        let db = GrafeoDB::with_config(
            crate::config::Config::in_memory()
                .with_history_retention(crate::config::HistoryRetention::Forever),
        )
        .unwrap();
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
            let store = copy.lpg_store().unwrap();
            let ids = store.node_ids(ReadView::latest()).unwrap();
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
        assert_eq!(
            copy.lpg_store()
                .unwrap()
                .node_count(ReadView::latest())
                .unwrap(),
            0
        );
        assert_eq!(
            copy.lpg_store()
                .unwrap()
                .edge_count(ReadView::latest())
                .unwrap(),
            0
        );
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
        assert_eq!(
            copy.lpg_store()
                .unwrap()
                .node_count(ReadView::latest())
                .unwrap(),
            2
        );

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
        assert_eq!(
            copy.lpg_store()
                .unwrap()
                .node_count(ReadView::latest())
                .unwrap(),
            2
        );
        assert_eq!(
            copy.lpg_store()
                .unwrap()
                .edge_count(ReadView::latest())
                .unwrap(),
            1
        );

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
        assert_eq!(
            db.lpg_store()
                .unwrap()
                .node_count(ReadView::latest())
                .unwrap(),
            2
        );
        assert_eq!(
            copy.lpg_store()
                .unwrap()
                .node_count(ReadView::latest())
                .unwrap(),
            1
        );
    }

    // --- iter_nodes() / iter_edges() ---

    #[test]
    fn test_iter_nodes_empty() {
        let db = GrafeoDB::new_in_memory();
        assert_eq!(db.iter_nodes().unwrap().count(), 0);
    }

    #[test]
    fn test_iter_nodes_returns_all() {
        let db = GrafeoDB::new_in_memory();
        let id1 = db.create_node(&["Person"]).unwrap();
        db.set_node_property(id1, "name", "Alix".into()).unwrap();
        let id2 = db.create_node(&["Animal"]).unwrap();
        db.set_node_property(id2, "name", "Fido".into()).unwrap();

        let nodes: Vec<_> = db.iter_nodes().unwrap().collect();
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
        assert_eq!(db.iter_edges().unwrap().count(), 0);
    }

    #[test]
    fn test_iter_edges_returns_all() {
        let db = GrafeoDB::new_in_memory();
        let a = db.create_node(&["A"]).unwrap();
        let b = db.create_node(&["B"]).unwrap();
        let c = db.create_node(&["C"]).unwrap();
        db.create_edge(a, b, "R1").unwrap();
        db.create_edge(b, c, "R2").unwrap();

        let edges: Vec<_> = db.iter_edges().unwrap().collect();
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
        assert_eq!(
            db.lpg_store()
                .unwrap()
                .node_count(ReadView::latest())
                .unwrap(),
            1
        );
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
                db.lpg_store()
                    .unwrap()
                    .node_count(ReadView::latest())
                    .unwrap(),
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
        // A limit on input: an invalid value naming what it was given as
        // (`named`), never an internal error.
        let invalid = |error: grafeo_common::utils::error::Error, what: &str, named: &str| {
            assert!(
                matches!(error, grafeo_common::utils::error::Error::InvalidValue(_)),
                "{what}: {error:?}"
            );
            let message = error.to_string();
            assert!(message.contains(named), "{what}: {message}");
            assert!(message.contains("GRAFEO-V"), "{what}: {message}");
        };
        let db = GrafeoDB::new_in_memory();
        let alix = db.create_node(&["Person"]).unwrap();
        let too_deep = nested(MAX_PROPERTY_VALUE_DEPTH + 1);
        invalid(
            db.set_node_property(alix, "trips", too_deep.clone())
                .unwrap_err(),
            "set_node_property",
            "\"trips\"",
        );
        invalid(
            db.create_node_with_props(&["Person"], [("trips", too_deep.clone())])
                .unwrap_err(),
            "create_node_with_props",
            "\"trips\"",
        );
        // A statement refuses the parameter itself, before it runs.
        let params = [("trips".to_string(), too_deep)].into_iter().collect();
        invalid(
            db.execute_with_params("INSERT (:Person {trips: $trips})", params)
                .unwrap_err(),
            "an INSERT parameter",
            "parameter $trips",
        );
        assert_eq!(db.node_count().unwrap(), 1, "nothing was written");

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
        assert_eq!(
            db.lpg_store()
                .unwrap()
                .node_count(ReadView::latest())
                .unwrap(),
            1
        );
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
        let found = db2
            .find_nodes_by_property("email", &Value::String("alix@example.com".into()))
            .unwrap();
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
        assert_eq!(written.node_count().unwrap(), 1);
        written.close().unwrap();
    }
}
