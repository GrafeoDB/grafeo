//! The catalog section of the `.grafeo` container format: the schema (node,
//! edge and graph types, graph type bindings, schemas, named constraints,
//! procedures) and the definitions and names of the indexes.
//!
//! # Version 2
//!
//! A checkpoint writes the catalog as [catalog records](grafeo_common::storage::catalog_record):
//!
//! - the metadata chunk ([`ChunkMeta::meta`]): bincode of [`CatalogMeta`],
//!   whose first byte is its layout, 1;
//! - stream 0 of graph 0 ([`ChunkKind::Stream`] pieces of at most the byte
//!   cap): the framed records, one per entry, in this order: schemas, node
//!   types, edge types, graph types, graph type bindings, named
//!   constraints, index definitions, index names, procedures.
//!
//! The records of each kind come in increasing order of their names (a
//! binding by its graph; index definitions by graph, the default graph
//! first, then property, vector and text indexes, each by key or by label
//! and property; index names by name, label, property and kind), so the same
//! catalog is written to the same bytes. A reader refuses a record that
//! repeats or does not follow the one of its kind before it (an index name
//! may repeat: `CREATE INDEX` names one index per property). A catalog
//! without entries is the metadata chunk alone.
//!
//! # Version 1
//!
//! A 0.5.x file holds the catalog as one raw chunk: bincode of
//! [`CatalogSnapshot`] and the in-memory definitions, with the constraint
//! names and the indexes of every graph appended (see
//! [`serialize`](Section::serialize)). It is read through
//! [`deserialize`](Section::deserialize), which stays as it is.

use std::collections::HashSet;
use std::fmt;
use std::io::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};

use grafeo_common::storage::catalog_record::{
    CatalogRecord, GraphBindingRecord, IndexKindRecord, IndexRecord, SchemaRecord,
};
use grafeo_common::storage::chunk::{
    ChunkCaps, ChunkStreamReader, ChunkStreamWriter, stream_error,
};
use grafeo_common::storage::read_catalog_records;
use grafeo_common::storage::section::{
    ChunkKind, ChunkMeta, Section, SectionSink, SectionSource, SectionType, check_version,
    legacy_bytes,
};
use grafeo_common::utils::error::{Error, Result};
use grafeo_core::graph::lpg::LpgStore;
use grafeo_core::index::text::TextIndexOptions;
use grafeo_core::index::vector::{DistanceMetric, QuantizationType};

use super::catalog_records::{
    constraint_definition, constraint_record, edge_type_definition, edge_type_record,
    graph_indexes, graph_type_definition, graph_type_record, index_name, index_name_record,
    index_records, node_type_definition, node_type_record, procedure_definition, procedure_record,
    record_name,
};
use crate::catalog::{
    Catalog, CatalogError, ConstraintDefinition, EdgeTypeDefinition, GraphTypeDefinition,
    IndexType, NodeTypeDefinition, ProcedureDefinition,
};

/// The catalog section version this release writes: catalog records.
const CATALOG_SECTION_VERSION: u8 = 2;

/// The `version` field of the 0.5.x snapshot ([`CatalogSnapshot`]), which
/// [`serialize`](Section::serialize) writes.
const SNAPSHOT_VERSION: u8 = 1;

/// The stream of graph 0 that holds the records.
const CATALOG_STREAM: u32 = 0;

/// The layout of the metadata chunk this release writes and reads.
const META_LAYOUT: u8 = 1;

/// The bincode limit of the metadata chunk's decode.
const META_DECODE_LIMIT: usize = 1 << 24;

/// The metadata chunk of a version 2 catalog section.
#[derive(Serialize, Deserialize)]
struct CatalogMeta {
    /// [`META_LAYOUT`].
    layout: u8,
    /// The byte cap the record stream was cut at. A reader does not need it:
    /// it accepts any caps.
    max_bytes: u32,
}

// ── Snapshot types ──────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct CatalogSnapshot {
    version: u8,
    schema: SnapshotSchema,
    indexes: SnapshotIndexes,
    epoch: u64,
}

#[derive(Serialize, Deserialize, Default)]
struct SnapshotSchema {
    node_types: Vec<NodeTypeDefinition>,
    edge_types: Vec<EdgeTypeDefinition>,
    graph_types: Vec<GraphTypeDefinition>,
    procedures: Vec<ProcedureDefinition>,
    schemas: Vec<String>,
    graph_type_bindings: Vec<(String, String)>,
}

/// Named constraints, appended after the version 1 snapshot when there are
/// any (#420). Readers before 0.5.44 decode the snapshot and ignore the bytes
/// after it, so they still open the file; their constraints keep working
/// through the node types, without names. The version 2 layout (#517) holds
/// them explicitly.
#[derive(Serialize, Deserialize)]
struct ConstraintNames {
    constraints: Vec<ConstraintDefinition>,
}

/// The indexes of every graph and the index names, appended after the
/// constraint names when there are any (0.5.44). The version 1 snapshot
/// holds only the default graph's definitions, without quantization; readers
/// before 0.5.44 ignore both, and did not rebuild indexes from it either.
#[derive(Serialize, Deserialize)]
struct IndexExtension {
    graphs: Vec<GraphIndexes>,
    names: Vec<IndexName>,
}

/// The indexes of one graph, as a checkpoint saves them: definitions only.
/// Loading builds the indexes from the data, or restores the default graph's
/// vector and text indexes from their own sections.
#[derive(Serialize, Deserialize, Default, Debug, Clone, PartialEq)]
pub(crate) struct GraphIndexes {
    /// The graph's storage key; `None` for the default graph.
    pub graph: Option<String>,
    /// Indexed node properties.
    pub property: Vec<String>,
    /// Vector indexes.
    pub vector: Vec<VectorIndexDefinition>,
    /// Text indexes.
    pub text: Vec<TextIndexDefinition>,
}

impl GraphIndexes {
    /// The indexes of `store`, the graph `graph`.
    fn of(store: &LpgStore, graph: Option<String>) -> Self {
        let mut property = store.property_index_keys();
        property.sort();

        #[cfg(feature = "vector-index")]
        let mut vector: Vec<VectorIndexDefinition> = store
            .vector_index_entries()
            .into_iter()
            .filter_map(|(key, index)| {
                let (label, property) = key.split_once(':')?;
                let config = index.config();
                Some(VectorIndexDefinition {
                    label: label.to_string(),
                    property: property.to_string(),
                    dimensions: config.dimensions,
                    metric: config.metric,
                    m: config.m,
                    ef_construction: config.ef_construction,
                    quantization: index.quantization_type(),
                })
            })
            .collect();
        #[cfg(not(feature = "vector-index"))]
        let mut vector: Vec<VectorIndexDefinition> = Vec::new();
        vector.sort_by(|a, b| (&a.label, &a.property).cmp(&(&b.label, &b.property)));

        #[cfg(feature = "text-index")]
        let mut text: Vec<TextIndexDefinition> = store
            .text_index_entries()
            .into_iter()
            .filter_map(|(key, index)| {
                let (label, property) = key.split_once(':')?;
                Some(TextIndexDefinition {
                    label: label.to_string(),
                    property: property.to_string(),
                    options: index.read().options().clone(),
                })
            })
            .collect();
        #[cfg(not(feature = "text-index"))]
        let mut text: Vec<TextIndexDefinition> = Vec::new();
        text.sort_by(|a, b| (&a.label, &a.property).cmp(&(&b.label, &b.property)));

        Self {
            graph,
            property,
            vector,
            text,
        }
    }

    /// Whether the graph has no index.
    pub fn is_empty(&self) -> bool {
        self.property.is_empty() && self.vector.is_empty() && self.text.is_empty()
    }
}

/// A vector index's definition.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub(crate) struct VectorIndexDefinition {
    pub label: String,
    pub property: String,
    pub dimensions: usize,
    pub metric: DistanceMetric,
    pub m: usize,
    pub ef_construction: usize,
    /// `None` for a plain HNSW index.
    pub quantization: Option<QuantizationType>,
}

/// A text index's definition: what it indexes, and its options.
///
/// It serializes as the `(label, property)` pair it replaced, so the indexes
/// that 0.5.44 appended to a version 1 catalog ([`IndexExtension`]) still
/// decode; the options are left out of that layout, and read back as the
/// defaults, which every 0.5.x text index had.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub(crate) struct TextIndexDefinition {
    pub label: String,
    pub property: String,
    #[serde(skip)]
    pub options: TextIndexOptions,
}

/// The name `CREATE INDEX` gave an index, for `SHOW INDEXES` and
/// `DROP INDEX`.
#[derive(Serialize, Deserialize)]
pub(crate) struct IndexName {
    pub name: String,
    pub label: String,
    pub property: String,
    pub index_type: IndexType,
}

impl IndexName {
    /// The order index names are written in: by name, label, property and
    /// kind, so names given to several indexes come in one order.
    fn key(&self) -> (&str, &str, &str, u8) {
        let kind = match self.index_type {
            IndexType::Hash => 0,
            IndexType::BTree => 1,
            IndexType::FullText => 2,
        };
        (&self.name, &self.label, &self.property, kind)
    }
}

#[derive(Serialize, Deserialize, Default)]
struct SnapshotIndexes {
    property_indexes: Vec<String>,
    vector_indexes: Vec<SnapshotVectorIndex>,
    text_indexes: Vec<SnapshotTextIndex>,
}

impl SnapshotIndexes {
    /// The version 1 layout of the default graph's indexes.
    fn from_graph(indexes: &GraphIndexes) -> Self {
        Self {
            property_indexes: indexes.property.clone(),
            vector_indexes: indexes
                .vector
                .iter()
                .map(|def| SnapshotVectorIndex {
                    label: def.label.clone(),
                    property: def.property.clone(),
                    dimensions: def.dimensions,
                    metric: def.metric,
                    m: def.m,
                    ef_construction: def.ef_construction,
                })
                .collect(),
            text_indexes: indexes
                .text
                .iter()
                .map(|def| SnapshotTextIndex {
                    label: def.label.clone(),
                    property: def.property.clone(),
                })
                .collect(),
        }
    }

    /// The default graph's indexes from the version 1 layout: files written
    /// before 0.5.44, which have no [`IndexExtension`].
    fn into_graph(self) -> GraphIndexes {
        GraphIndexes {
            graph: None,
            property: self.property_indexes,
            vector: self
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
            text: self
                .text_indexes
                .into_iter()
                .map(|def| TextIndexDefinition {
                    label: def.label,
                    property: def.property,
                    options: TextIndexOptions::default(),
                })
                .collect(),
        }
    }
}

#[derive(Serialize, Deserialize)]
struct SnapshotVectorIndex {
    label: String,
    property: String,
    dimensions: usize,
    metric: DistanceMetric,
    m: usize,
    ef_construction: usize,
}

#[derive(Serialize, Deserialize)]
struct SnapshotTextIndex {
    label: String,
    property: String,
}

// ── Section implementation ──────────────────────────────────────────

/// Catalog section for the `.grafeo` container.
///
/// Writes and reads the schema and the index definitions and names (see the
/// module documentation). The catalog is always small (typically < 10 KB)
/// and always kept in RAM.
pub struct CatalogSection {
    catalog: Arc<Catalog>,
    store: Arc<LpgStore>,
    epoch_fn: Box<dyn Fn() -> u64 + Send + Sync>,
    dirty: AtomicBool,
    /// The index definitions the last load read.
    loaded_indexes: Vec<GraphIndexes>,
    /// The database epoch the last load read: a 0.5.x catalog holds the epoch
    /// of its checkpoint, a 0.6 one none (0).
    loaded_epoch: u64,
    /// The caps the record stream is cut at, taken when the section is built.
    caps: ChunkCaps,
}

impl CatalogSection {
    /// Create a new catalog section, which writes with the chunk caps of
    /// this moment ([`ChunkCaps::current`]).
    ///
    /// The `epoch_fn` closure returns the current MVCC epoch. This avoids a
    /// dependency on `TransactionManager` which lives in the engine layer.
    pub fn new(
        catalog: Arc<Catalog>,
        store: Arc<LpgStore>,
        epoch_fn: impl Fn() -> u64 + Send + Sync + 'static,
    ) -> Self {
        Self {
            catalog,
            store,
            epoch_fn: Box::new(epoch_fn),
            dirty: AtomicBool::new(false),
            loaded_indexes: Vec::new(),
            loaded_epoch: 0,
            caps: ChunkCaps::current(),
        }
    }

    /// The index definitions of every graph that the last load read, for the
    /// loader to build once the data is in (the catalog holds only their
    /// names).
    pub(crate) fn take_loaded_indexes(&mut self) -> Vec<GraphIndexes> {
        std::mem::take(&mut self.loaded_indexes)
    }

    /// The database epoch the last load read, which the database continues
    /// from: a 0.5.x catalog (the version 1 layout) holds the epoch of its
    /// checkpoint, a 0.6 one none (0, the database header holds it).
    pub(crate) fn loaded_epoch(&self) -> u64 {
        self.loaded_epoch
    }

    /// Mark this section as dirty.
    #[allow(dead_code)] // Wired in Phase 5 checkpoint path
    pub fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }

    fn collect_schema(&self) -> SnapshotSchema {
        SnapshotSchema {
            node_types: self.catalog.all_node_type_defs(),
            edge_types: self.catalog.all_edge_type_defs(),
            graph_types: self.catalog.all_graph_type_defs(),
            procedures: self.catalog.all_procedure_defs(),
            schemas: self.catalog.schema_names(),
            graph_type_bindings: self.catalog.all_graph_type_bindings(),
        }
    }

    /// The indexes of the default graph and of each named graph that has any.
    fn collect_graph_indexes(&self) -> Vec<GraphIndexes> {
        let mut graphs = vec![GraphIndexes::of(&self.store, None)];
        let mut names = self.store.graph_names();
        names.sort();
        for name in names {
            if let Some(graph) = self.store.graph(&name) {
                let indexes = GraphIndexes::of(&graph, Some(name));
                if !indexes.is_empty() {
                    graphs.push(indexes);
                }
            }
        }
        graphs
    }

    /// The names `CREATE INDEX` gave indexes, in the order of
    /// [`IndexName::key`].
    fn collect_index_names(&self) -> Vec<IndexName> {
        let mut names: Vec<IndexName> = self
            .catalog
            .all_indexes()
            .into_iter()
            .filter_map(|def| {
                Some(IndexName {
                    label: self.catalog.get_label_name(def.label)?.to_string(),
                    property: self
                        .catalog
                        .get_property_key_name(def.property_key)?
                        .to_string(),
                    name: def.name,
                    index_type: def.index_type,
                })
            })
            .collect();
        names.sort_by(|a, b| a.key().cmp(&b.key()));
        names
    }

    /// Registers the index names of a version 1 catalog in the catalog
    /// again, each name once.
    fn restore_index_names(&self, names: Vec<IndexName>) {
        for index in names {
            if self.catalog.find_index_by_name(&index.name).is_some() {
                continue;
            }
            let label = self.catalog.get_or_create_label(&index.label);
            let property = self.catalog.get_or_create_property_key(&index.property);
            self.catalog
                .create_index(&index.name, label, property, index.index_type);
        }
    }

    /// Writes the records of version 2 into `stream`, in the order of the
    /// module documentation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] when the stream refuses a write (its own error
    /// is the one its `finish` returns), or the error of a record that does
    /// not encode, with the section and the entry named.
    fn write_records(&self, stream: &mut ChunkStreamWriter<'_>) -> Result<()> {
        // The constraint names and the node types that enforce them, from
        // one moment, as version 1 reads them.
        let (mut schema, constraints) = self
            .catalog
            .with_constraints(|constraints| (self.collect_schema(), constraints));
        let indexes = index_records(&self.collect_graph_indexes()).map_err(in_section)?;
        let index_names = self.collect_index_names();

        schema.schemas.sort();
        schema.node_types.sort_by(|a, b| a.name.cmp(&b.name));
        schema.edge_types.sort_by(|a, b| a.name.cmp(&b.name));
        schema.graph_types.sort_by(|a, b| a.name.cmp(&b.name));
        schema.graph_type_bindings.sort();
        schema.procedures.sort_by(|a, b| a.name.cmp(&b.name));
        // `DROP GRAPH TYPE` leaves the bindings to the type, which then bind
        // nothing (and a version 1 load dropped them, as binding them
        // failed). They are left out, so every binding written names a graph
        // type written, which the reader requires.
        let graph_types: HashSet<&str> = schema
            .graph_types
            .iter()
            .map(|def| def.name.as_str())
            .collect();
        schema
            .graph_type_bindings
            .retain(|(_, graph_type)| graph_types.contains(graph_type.as_str()));

        let mut writer = RecordWriter {
            stream,
            buffer: Vec::new(),
        };
        for name in schema.schemas {
            writer.write(&CatalogRecord::Schema(SchemaRecord { name }))?;
        }
        for def in &schema.node_types {
            writer.write(&CatalogRecord::NodeType(node_type_record(def)))?;
        }
        for def in &schema.edge_types {
            let record = edge_type_record(def).map_err(in_section)?;
            writer.write(&CatalogRecord::EdgeType(record))?;
        }
        for def in &schema.graph_types {
            writer.write(&CatalogRecord::GraphType(graph_type_record(def)))?;
        }
        for (graph, graph_type) in schema.graph_type_bindings {
            writer.write(&CatalogRecord::GraphBinding(GraphBindingRecord {
                graph,
                graph_type,
            }))?;
        }
        for def in &constraints {
            writer.write(&CatalogRecord::Constraint(constraint_record(def)))?;
        }
        for record in indexes {
            writer.write(&CatalogRecord::Index(record))?;
        }
        for name in &index_names {
            writer.write(&CatalogRecord::IndexName(index_name_record(name)))?;
        }
        for def in &schema.procedures {
            writer.write(&CatalogRecord::Procedure(procedure_record(def)))?;
        }
        Ok(())
    }

    /// Applies one record of a version 2 section to the catalog, or keeps it
    /// in `loading` for [`finish_load`](Self::finish_load).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Corruption`] naming the record when it repeats or does
    /// not follow the record of its kind before it, or when the catalog
    /// refuses it; [`Error::Serialization`] when its edge type's endpoints are
    /// not a product (a later release may store those); the error of creating
    /// a schema's default graph.
    fn apply_record(&self, record: CatalogRecord, loading: &mut Loading) -> Result<()> {
        let name = record_name(&record);
        loading.check_order(&record, &name)?;
        let refused = |error: CatalogError| {
            Error::corruption(format!("{name} does not apply to the catalog: {error}"))
        };
        match record {
            CatalogRecord::Schema(SchemaRecord { name: schema }) => {
                self.catalog
                    .register_schema_namespace(schema.clone())
                    .map_err(refused)?;
                self.store.create_graph(&format!("{schema}/__default__"))?;
            }
            CatalogRecord::NodeType(record) => self
                .catalog
                .register_or_replace_node_type(node_type_definition(record)),
            CatalogRecord::EdgeType(record) => self
                .catalog
                .register_or_replace_edge_type_def(edge_type_definition(record)?),
            CatalogRecord::GraphType(record) => self
                .catalog
                .register_graph_type(graph_type_definition(record))
                .map_err(refused)?,
            CatalogRecord::GraphBinding(GraphBindingRecord { graph, graph_type }) => self
                .catalog
                .bind_graph_type(&graph, graph_type)
                .map_err(refused)?,
            CatalogRecord::Constraint(record) => {
                loading.constraints.push(constraint_definition(record));
            }
            CatalogRecord::Index(record) => loading.indexes.push(record),
            CatalogRecord::IndexName(record) => loading.index_names.push(index_name(record)),
            CatalogRecord::Procedure(record) => self
                .catalog
                .replace_procedure(procedure_definition(record))
                .map_err(refused)?,
        }
        Ok(())
    }

    /// Applies what [`apply_record`](Self::apply_record) kept once every
    /// record is read: the constraint names (whose type constraints the node
    /// types hold already), every index name, also a name given to several
    /// indexes, and the index definitions, for the loader.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Serialization`] naming a vector index whose sizes do
    /// not fit this platform.
    fn finish_load(&mut self, loading: Loading) -> Result<()> {
        self.catalog.restore_constraint_names(loading.constraints);
        for index in loading.index_names {
            let label = self.catalog.get_or_create_label(&index.label);
            let property = self.catalog.get_or_create_property_key(&index.property);
            self.catalog
                .create_index(&index.name, label, property, index.index_type);
        }
        self.loaded_indexes = graph_indexes(loading.indexes)?;
        Ok(())
    }
}

/// What a load of a version 2 section keeps between records: the key of the
/// last record of each kind, which the next one must follow, and the records
/// applied once all are read.
#[derive(Default)]
struct Loading {
    schema: Option<String>,
    node_type: Option<String>,
    edge_type: Option<String>,
    graph_type: Option<String>,
    /// By graph.
    binding: Option<String>,
    constraint: Option<String>,
    index: Option<IndexKey>,
    index_name: Option<(String, String, String, u8)>,
    procedure: Option<String>,
    constraints: Vec<ConstraintDefinition>,
    indexes: Vec<IndexRecord>,
    index_names: Vec<IndexName>,
}

/// The order of index records: graph (the default graph first), property,
/// vector or text index, then the key, or the label and property.
type IndexKey = (Option<String>, u8, String, String);

impl Loading {
    /// Refuses `record`, which errors call `name`, when it repeats or does
    /// not follow the record of its kind before it. Only an index name may
    /// repeat.
    fn check_order(&mut self, record: &CatalogRecord, name: &str) -> Result<()> {
        match record {
            CatalogRecord::Schema(record) => follow(&mut self.schema, record.name.clone(), name),
            CatalogRecord::NodeType(record) => {
                follow(&mut self.node_type, record.name.clone(), name)
            }
            CatalogRecord::EdgeType(record) => {
                follow(&mut self.edge_type, record.name.clone(), name)
            }
            CatalogRecord::GraphType(record) => {
                follow(&mut self.graph_type, record.name.clone(), name)
            }
            CatalogRecord::GraphBinding(record) => {
                follow(&mut self.binding, record.graph.clone(), name)
            }
            CatalogRecord::Constraint(record) => {
                follow(&mut self.constraint, record.name.clone(), name)
            }
            CatalogRecord::Index(record) => follow(&mut self.index, index_key(record), name),
            CatalogRecord::IndexName(record) => {
                let index = index_name(record.clone());
                let (name_key, label, property, kind) = index.key();
                let key = (
                    name_key.to_string(),
                    label.to_string(),
                    property.to_string(),
                    kind,
                );
                match &self.index_name {
                    Some(previous) if key < *previous => Err(out_of_order(name)),
                    _ => {
                        self.index_name = Some(key);
                        Ok(())
                    }
                }
            }
            CatalogRecord::Procedure(record) => {
                follow(&mut self.procedure, record.name.clone(), name)
            }
        }
    }
}

/// Keeps `key` as the last key of its kind, or refuses the record `name`
/// when `key` does not come after the last one.
fn follow<K: Ord>(last: &mut Option<K>, key: K, name: &str) -> Result<()> {
    if last.as_ref().is_some_and(|previous| key <= *previous) {
        return Err(out_of_order(name));
    }
    *last = Some(key);
    Ok(())
}

/// The error of a record that repeats or comes before the one of its kind
/// before it.
fn out_of_order(name: &str) -> Error {
    Error::corruption(format!(
        "{name} is out of order: it repeats the record of its kind before it or comes before \
         it, where each kind's records come once each, in increasing order"
    ))
}

/// The place of an index record in the order of [`IndexKey`].
fn index_key(record: &IndexRecord) -> IndexKey {
    let graph = record.graph.clone();
    match &record.index {
        IndexKindRecord::Property { key } => (graph, 0, key.clone(), String::new()),
        IndexKindRecord::Vector {
            label, property, ..
        } => (graph, 1, label.clone(), property.clone()),
        IndexKindRecord::Text {
            label, property, ..
        } => (graph, 2, label.clone(), property.clone()),
    }
}

/// Frames records into the record stream, one at a time.
struct RecordWriter<'w, 's> {
    stream: &'w mut ChunkStreamWriter<'s>,
    /// The record being written, framed.
    buffer: Vec<u8>,
}

impl RecordWriter<'_, '_> {
    /// Frames `record` and writes it into the stream.
    ///
    /// # Errors
    ///
    /// Returns the error of a record that does not encode (more than a
    /// record holds, a property type nested too deep, a default value the
    /// value codec refuses) with the section and the entry named, or
    /// [`Error::Io`] when the stream refuses the write.
    fn write(&mut self, record: &CatalogRecord) -> Result<()> {
        self.buffer.clear();
        record
            .encode_framed(&mut self.buffer)
            .map_err(|error| in_entry(&record_name(record), error))?;
        self.stream.write_all(&self.buffer).map_err(Error::Io)
    }
}

/// `error` with the section and the entry `what` in front of its message,
/// for the errors that carry one.
fn in_entry(what: &str, error: Error) -> Error {
    let message =
        |message: String| format!("section {:?}: {what}: {message}", SectionType::Catalog);
    match error {
        Error::Serialization(text) => Error::Serialization(message(text)),
        Error::InvalidValue(text) => Error::InvalidValue(message(text)),
        Error::Internal(text) => Error::Internal(message(text)),
        other => other,
    }
}

/// `error` with the section in front of its message, as
/// [`stream_error`] names it in a stream's errors: corrupt section data
/// ([`Error::Corruption`]), what a newer release wrote
/// ([`Error::Serialization`]), a misused source ([`Error::Internal`]) or a
/// failed read ([`Error::Io`]). Any other error as it is.
fn in_section(error: Error) -> Error {
    match error {
        Error::Corruption(_) | Error::Serialization(_) | Error::Internal(_) | Error::Io(_) => {
            error.wrapped(format_args!("section {:?}", SectionType::Catalog))
        }
        other => other,
    }
}

/// Checks the chunks of a version 2 section around its records: the first
/// is the metadata chunk ([`ChunkMeta::meta`]), with layout [`META_LAYOUT`]
/// and no bytes after its fields, and every other is a piece of stream
/// [`CATALOG_STREAM`] of graph 0, without codec or rows.
///
/// # Errors
///
/// Returns [`Error::Corruption`] naming the section and the chunk for any
/// other sequence or metadata, or the error of fetching the metadata chunk.
fn read_meta(source: &dyn SectionSource) -> Result<CatalogMeta> {
    let refuse =
        |what: String| Error::corruption(format!("section {:?}: {what}", SectionType::Catalog));
    let chunks = source.chunks();
    match chunks.first() {
        Some(first) if *first == ChunkMeta::meta() => {}
        Some(first) => {
            return Err(refuse(format!(
                "the first chunk is {}, where the metadata chunk belongs",
                ChunkDescription(first)
            )));
        }
        None => return Err(refuse("there is no metadata chunk".to_string())),
    }
    for (index, chunk) in chunks.iter().enumerate().skip(1) {
        if *chunk != ChunkMeta::stream_piece(0, CATALOG_STREAM, chunk.row_start) {
            return Err(refuse(format!(
                "chunk {index} is {}; after its metadata chunk the section holds pieces of \
                 stream {CATALOG_STREAM} of graph 0 only",
                ChunkDescription(chunk)
            )));
        }
    }
    let bytes = source.fetch(0).map_err(in_section)?;
    match bytes.first() {
        Some(&META_LAYOUT) => {}
        Some(layout) => {
            return Err(refuse(format!(
                "the metadata chunk has layout {layout}, this build reads layout {META_LAYOUT}"
            )));
        }
        None => return Err(refuse("the metadata chunk is empty".to_string())),
    }
    let config = bincode::config::standard().with_limit::<META_DECODE_LIMIT>();
    let (meta, read) = bincode::serde::decode_from_slice(&bytes, config)
        .map_err(|e| refuse(format!("the metadata chunk does not decode: {e}")))?;
    if read != bytes.len() {
        return Err(refuse(format!(
            "the metadata chunk holds {} bytes after its fields",
            bytes.len() - read
        )));
    }
    Ok(meta)
}

/// A chunk as an error message names it.
struct ChunkDescription<'a>(&'a ChunkMeta);

impl fmt::Display for ChunkDescription<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let chunk = self.0;
        if chunk.kind == ChunkKind::Stream {
            write!(
                f,
                "a piece of stream {} of graph {} at offset {} (codec {}, rows {})",
                chunk.column_id, chunk.graph_id, chunk.row_start, chunk.codec, chunk.row_count
            )
        } else {
            write!(
                f,
                "a {:?} chunk of graph {}, column {}, first row {}",
                chunk.kind, chunk.graph_id, chunk.column_id, chunk.row_start
            )
        }
    }
}

impl Section for CatalogSection {
    fn section_type(&self) -> SectionType {
        SectionType::Catalog
    }

    fn version(&self) -> u8 {
        CATALOG_SECTION_VERSION
    }

    /// The version 1 layout of 0.5.x files, which only tests write now (a
    /// checkpoint writes version 2 through [`write_to`](Section::write_to)).
    /// Key labels are not in it.
    fn serialize(&self) -> Result<Vec<u8>> {
        // The names and the node types that enforce them, from one moment.
        let (schema, constraints) = self
            .catalog
            .with_constraints(|constraints| (self.collect_schema(), constraints));
        let mut graphs = self.collect_graph_indexes();
        let snapshot = CatalogSnapshot {
            version: SNAPSHOT_VERSION,
            schema,
            indexes: SnapshotIndexes::from_graph(&graphs[0]),
            epoch: (self.epoch_fn)(),
        };

        let config = bincode::config::standard();
        let mut bytes = bincode::serde::encode_to_vec(&snapshot, config)
            .map_err(|e| Error::Internal(format!("Catalog section serialization failed: {e}")))?;

        graphs.retain(|graph| !graph.is_empty());
        let extension = IndexExtension {
            graphs,
            names: self.collect_index_names(),
        };
        let has_extension = !extension.graphs.is_empty() || !extension.names.is_empty();
        // The extension follows the constraint names, so they come first even
        // when there are none.
        if !constraints.is_empty() || has_extension {
            let names = bincode::serde::encode_to_vec(ConstraintNames { constraints }, config)
                .map_err(|e| {
                    Error::Internal(format!("Constraint name serialization failed: {e}"))
                })?;
            bytes.extend_from_slice(&names);
        }
        if has_extension {
            let indexes = bincode::serde::encode_to_vec(&extension, config)
                .map_err(|e| Error::Internal(format!("Index serialization failed: {e}")))?;
            bytes.extend_from_slice(&indexes);
        }
        Ok(bytes)
    }

    /// Reads the version 1 layout of 0.5.x files.
    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        let config = bincode::config::standard();
        let (snapshot, read): (CatalogSnapshot, _) =
            bincode::serde::decode_from_slice(data, config).map_err(|e| {
                Error::corruption(format!("Catalog section deserialization failed: {e}"))
            })?;

        // Restore schema definitions
        for def in &snapshot.schema.node_types {
            self.catalog.register_or_replace_node_type(def.clone());
        }
        for def in &snapshot.schema.edge_types {
            self.catalog.register_or_replace_edge_type_def(def.clone());
        }
        for def in &snapshot.schema.graph_types {
            let _ = self.catalog.register_graph_type(def.clone());
        }
        for def in &snapshot.schema.procedures {
            self.catalog.replace_procedure(def.clone()).ok();
        }
        for name in &snapshot.schema.schemas {
            let _ = self.catalog.register_schema_namespace(name.clone());
            let default_key = format!("{name}/__default__");
            let _ = self.store.create_graph(&default_key);
        }
        for (graph_name, type_name) in &snapshot.schema.graph_type_bindings {
            let _ = self.catalog.bind_graph_type(graph_name, type_name.clone());
        }
        self.loaded_epoch = snapshot.epoch;
        // The node types restored above already hold the constraints.
        let mut rest = &data[read..];
        if !rest.is_empty() {
            let (names, read): (ConstraintNames, _) =
                bincode::serde::decode_from_slice(rest, config).map_err(|e| {
                    Error::corruption(format!("Constraint names deserialization failed: {e}"))
                })?;
            self.catalog.restore_constraint_names(names.constraints);
            rest = &rest[read..];
        }

        // The indexes are built by the loader once the data is in.
        self.loaded_indexes = if rest.is_empty() {
            let root = snapshot.indexes.into_graph();
            if root.is_empty() {
                Vec::new()
            } else {
                vec![root]
            }
        } else {
            let (extension, _): (IndexExtension, _) =
                bincode::serde::decode_from_slice(rest, config)
                    .map_err(|e| Error::corruption(format!("Index deserialization failed: {e}")))?;
            self.restore_index_names(extension.names);
            extension.graphs
        };

        Ok(())
    }

    /// Version 2: the metadata chunk, then the records as stream 0 of graph
    /// 0, cut at the section's byte cap (see the module documentation).
    fn write_to(&self, sink: &mut dyn SectionSink) -> Result<()> {
        let meta = CatalogMeta {
            layout: META_LAYOUT,
            max_bytes: self.caps.max_bytes,
        };
        let meta =
            bincode::serde::encode_to_vec(&meta, bincode::config::standard()).map_err(|e| {
                Error::Serialization(format!(
                    "section {:?}: the metadata chunk does not encode: {e}",
                    SectionType::Catalog
                ))
            })?;
        sink.write_chunk(ChunkMeta::meta(), &meta)?;
        let mut stream = ChunkStreamWriter::new(sink, 0, CATALOG_STREAM, self.caps);
        match self.write_records(&mut stream) {
            Ok(()) => stream.finish().map(drop),
            // Every I/O error comes from the stream, which keeps the sink's
            // own error for `finish` to return.
            Err(Error::Io(error)) => Err(stream
                .finish()
                .err()
                .unwrap_or_else(|| stream_error(SectionType::Catalog, error))),
            Err(error) => Err(error),
        }
    }

    /// A single raw chunk (0.5.x bytes) goes to
    /// [`deserialize`](Section::deserialize). Otherwise the section is
    /// version 2: its metadata chunk and the records of stream 0, applied
    /// one at a time as they are read.
    fn read_from(&mut self, source: &dyn SectionSource) -> Result<()> {
        if let Some(bytes) = legacy_bytes(source).map_err(in_section)? {
            return self.deserialize(&bytes);
        }
        check_version(SectionType::Catalog, source, CATALOG_SECTION_VERSION)?;
        read_meta(source)?;
        let mut reader = ChunkStreamReader::new(source, 0, CATALOG_STREAM);
        let mut loading = Loading::default();
        read_catalog_records(&mut reader, &mut |record| {
            self.apply_record(record, &mut loading)
        })
        .map_err(in_section)?;
        self.finish_load(loading).map_err(in_section)
    }

    fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    fn mark_clean(&self) {
        self.dirty.store(false, Ordering::Release);
    }

    fn memory_usage(&self) -> usize {
        // Catalog is tiny: schema defs + index metadata, typically < 10 KB
        4096
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{EdgeTypeDefinition, NodeTypeDefinition, TypedProperty};

    fn make_section() -> CatalogSection {
        let catalog = Arc::new(Catalog::new());
        let store = Arc::new(grafeo_core::graph::lpg::LpgStore::new().unwrap());
        CatalogSection::new(catalog, store, || 42)
    }

    #[test]
    fn empty_catalog_roundtrip() {
        let section = make_section();
        let bytes = section.serialize().expect("serialize empty catalog");
        assert!(!bytes.is_empty(), "bytes is empty");

        let catalog2 = Arc::new(Catalog::new());
        let store2 = Arc::new(grafeo_core::graph::lpg::LpgStore::new().unwrap());
        let mut section2 = CatalogSection::new(catalog2, store2, || 0);
        section2
            .deserialize(&bytes)
            .expect("deserialize empty catalog");
    }

    /// Named constraints follow the version 1 snapshot: they come back with
    /// their names, and a reader that only knows the snapshot (0.5.43)
    /// still decodes it (#420).
    #[test]
    fn constraint_names_follow_the_v1_snapshot() {
        use crate::catalog::{ConstraintType, TypeConstraint};

        let section = make_section();
        let city_name = ConstraintDefinition {
            name: "city_name".to_string(),
            label: "City".to_string(),
            properties: vec!["name".to_string()],
            kind: ConstraintType::Unique,
        };
        section
            .catalog
            .create_constraint(city_name.clone())
            .unwrap();
        let bytes = section.serialize().unwrap();

        let config = bincode::config::standard();
        let (_, read): (CatalogSnapshot, _) =
            bincode::serde::decode_from_slice(&bytes, config).unwrap();
        assert!(read < bytes.len(), "the names follow the snapshot");

        let catalog = Arc::new(Catalog::new());
        let store = Arc::new(grafeo_core::graph::lpg::LpgStore::new().unwrap());
        let mut reopened = CatalogSection::new(Arc::clone(&catalog), store, || 0);
        reopened.deserialize(&bytes).unwrap();
        assert_eq!(catalog.constraints(), vec![city_name]);
        assert_eq!(
            catalog.get_node_type("City").unwrap().constraints,
            vec![TypeConstraint::Unique(vec!["name".to_string()])],
            "the constraint comes back once, from the node type"
        );
        catalog.drop_constraint("city_name").unwrap();
        assert!(
            catalog
                .get_node_type("City")
                .unwrap()
                .constraints
                .is_empty(),
            "expected no constraints"
        );
    }

    /// A checkpoint sees the constraint names and the node types that enforce
    /// them from one moment: with `CREATE CONSTRAINT` running at the same
    /// time, a name saved without its type constraint would be unenforced
    /// after a reopen, and one saved without its name could not be dropped.
    #[test]
    fn serialized_constraint_names_match_the_node_types() {
        use crate::catalog::{ConstraintType, TypeConstraint};

        let section = make_section();
        let catalog = Arc::clone(&section.catalog);
        let writer = std::thread::spawn(move || {
            for _ in 0..3_000 {
                catalog
                    .create_constraint(ConstraintDefinition {
                        name: "city_name".to_string(),
                        label: "City".to_string(),
                        properties: vec!["name".to_string()],
                        kind: ConstraintType::Unique,
                    })
                    .unwrap();
                catalog.drop_constraint("city_name").unwrap();
            }
        });

        let config = bincode::config::standard();
        let unique_name = TypeConstraint::Unique(vec!["name".to_string()]);
        while !writer.is_finished() {
            let bytes = section.serialize().unwrap();
            let (snapshot, read): (CatalogSnapshot, _) =
                bincode::serde::decode_from_slice(&bytes, config).unwrap();
            let names = if read < bytes.len() {
                let (names, _): (ConstraintNames, _) =
                    bincode::serde::decode_from_slice(&bytes[read..], config).unwrap();
                names.constraints.len()
            } else {
                0
            };
            let enforced = snapshot
                .schema
                .node_types
                .iter()
                .filter(|def| def.name == "City")
                .flat_map(|def| &def.constraints)
                .filter(|constraint| **constraint == unique_name)
                .count();
            assert_eq!(names, enforced, "names and type constraints differ");
        }
        writer.join().unwrap();
    }

    #[test]
    fn catalog_with_node_types_roundtrip() {
        let section = make_section();
        section
            .catalog
            .register_or_replace_node_type(NodeTypeDefinition {
                name: "Person".to_string(),
                properties: vec![TypedProperty {
                    name: "name".to_string(),
                    data_type: crate::catalog::PropertyDataType::String,
                    nullable: false,
                    default_value: None,
                }],
                constraints: vec![],
                parent_types: vec![],
                key_labels: vec![],
            });

        let bytes = section.serialize().unwrap();

        let catalog2 = Arc::new(Catalog::new());
        let store2 = Arc::new(grafeo_core::graph::lpg::LpgStore::new().unwrap());
        let mut section2 = CatalogSection::new(catalog2, store2, || 0);
        section2.deserialize(&bytes).unwrap();

        let types = section2.catalog.all_node_type_defs();
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].name, "Person");
        assert_eq!(types[0].properties.len(), 1);
    }

    #[test]
    fn catalog_with_edge_types_roundtrip() {
        let section = make_section();
        section
            .catalog
            .register_or_replace_edge_type_def(EdgeTypeDefinition {
                name: "KNOWS".to_string(),
                properties: vec![],
                constraints: vec![],
                source_node_types: vec![],
                target_node_types: vec![],
                key_labels: vec![],
            });

        let bytes = section.serialize().unwrap();

        let catalog2 = Arc::new(Catalog::new());
        let store2 = Arc::new(grafeo_core::graph::lpg::LpgStore::new().unwrap());
        let mut section2 = CatalogSection::new(catalog2, store2, || 0);
        section2.deserialize(&bytes).unwrap();

        let types = section2.catalog.all_edge_type_defs();
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].name, "KNOWS");
    }

    #[test]
    fn catalog_section_type_and_version() {
        let section = make_section();
        assert_eq!(section.section_type(), SectionType::Catalog);
        assert_eq!(section.version(), CATALOG_SECTION_VERSION);
    }

    #[test]
    fn catalog_dirty_tracking() {
        let section = make_section();
        assert!(!section.is_dirty());

        section.mark_dirty();
        assert!(section.is_dirty());

        section.mark_clean();
        assert!(!section.is_dirty());
    }

    #[test]
    fn catalog_memory_usage() {
        let section = make_section();
        assert_eq!(section.memory_usage(), 4096);
    }

    #[test]
    fn catalog_deserialize_corrupt_data() {
        let mut section = make_section();
        let result = section.deserialize(&[0xFF, 0xFE, 0xFD, 0x00]);
        assert!(result.is_err(), "corrupt data should fail deserialization");
    }

    /// The indexes 0.5.44 appended to a version 1 catalog hold a text index
    /// as a `(label, property)` pair: a text index definition still encodes
    /// and decodes as one, and reads back with the default options, which
    /// every 0.5.x text index had.
    #[test]
    fn a_text_index_definition_has_the_version_1_layout_of_a_pair() {
        let config = bincode::config::standard();
        let def = TextIndexDefinition {
            label: "Doc".to_string(),
            property: "body".to_string(),
            options: TextIndexOptions::new().with_k1(0.3),
        };
        let bytes = bincode::serde::encode_to_vec(&def, config).unwrap();
        assert_eq!(
            bytes,
            bincode::serde::encode_to_vec(("Doc".to_string(), "body".to_string()), config).unwrap()
        );
        let (back, _): (TextIndexDefinition, _) =
            bincode::serde::decode_from_slice(&bytes, config).unwrap();
        assert_eq!(
            (back.label.as_str(), back.property.as_str(), back.options),
            ("Doc", "body", TextIndexOptions::default())
        );
    }

    /// The indexes of every graph and the index names round trip: loading
    /// hands the definitions to the loader and registers the names.
    #[test]
    fn indexes_of_every_graph_round_trip() {
        let section = make_section();
        section.store.create_property_index("id");
        section.store.create_graph("model").unwrap();
        section
            .store
            .graph("model")
            .unwrap()
            .create_property_index("key");
        let label = section.catalog.get_or_create_label("File");
        let size = section.catalog.get_or_create_property_key("size");
        section
            .catalog
            .create_index("file_size", label, size, IndexType::BTree);
        let bytes = section.serialize().unwrap();

        let catalog = Arc::new(Catalog::new());
        let store = Arc::new(LpgStore::new().unwrap());
        let mut loaded = CatalogSection::new(Arc::clone(&catalog), store, || 0);
        loaded.deserialize(&bytes).unwrap();
        assert_eq!(
            loaded.take_loaded_indexes(),
            [
                GraphIndexes {
                    graph: None,
                    property: vec!["id".to_string()],
                    ..GraphIndexes::default()
                },
                GraphIndexes {
                    graph: Some("model".to_string()),
                    property: vec!["key".to_string()],
                    ..GraphIndexes::default()
                },
            ]
        );
        let index = catalog
            .get_index(catalog.find_index_by_name("file_size").unwrap())
            .unwrap();
        assert_eq!(index.index_type, IndexType::BTree);
        assert_eq!(catalog.get_label_name(index.label).as_deref(), Some("File"));
        assert_eq!(
            catalog.get_property_key_name(index.property_key).as_deref(),
            Some("size")
        );
    }

    /// A catalog written before 0.5.44 names the default graph's indexes in
    /// the version 1 snapshot only.
    #[test]
    fn a_version_1_catalog_names_the_default_graph_indexes() {
        let snapshot = CatalogSnapshot {
            version: 1,
            schema: SnapshotSchema::default(),
            indexes: SnapshotIndexes {
                property_indexes: vec!["id".to_string()],
                ..SnapshotIndexes::default()
            },
            epoch: 7,
        };
        let bytes = bincode::serde::encode_to_vec(&snapshot, bincode::config::standard()).unwrap();

        let mut section = make_section();
        section.deserialize(&bytes).unwrap();
        assert_eq!(
            section.take_loaded_indexes(),
            [GraphIndexes {
                graph: None,
                property: vec!["id".to_string()],
                ..GraphIndexes::default()
            }]
        );
    }

    /// Fills the catalog with every kind of entry a 0.5.x catalog holds, one
    /// entry per hash map (node types, edge types, graph types, procedures,
    /// bindings), so the maps' iteration order cannot change the bytes.
    fn populate_every_0_5_entry(section: &CatalogSection) {
        use crate::catalog::{ConstraintType, PropertyDataType, TypeConstraint};
        use grafeo_common::types::Value;

        let property = |name: &str, data_type: PropertyDataType| TypedProperty {
            name: name.to_string(),
            data_type,
            nullable: true,
            default_value: None,
        };
        let catalog = &section.catalog;
        catalog.register_or_replace_node_type(NodeTypeDefinition {
            name: "City".to_string(),
            properties: vec![
                TypedProperty {
                    name: "name".to_string(),
                    data_type: PropertyDataType::String,
                    nullable: false,
                    default_value: Some(Value::from("Amsterdam")),
                },
                TypedProperty {
                    name: "population".to_string(),
                    data_type: PropertyDataType::Int64,
                    nullable: true,
                    default_value: Some(Value::Int64(88)),
                },
                property("area", PropertyDataType::Float64),
                property("capital", PropertyDataType::Bool),
                property("founded", PropertyDataType::Date),
                property("opens", PropertyDataType::Time),
                property("updated", PropertyDataType::Timestamp),
                property("trip", PropertyDataType::Duration),
                property("districts", PropertyDataType::List),
                property(
                    "zip_codes",
                    PropertyDataType::ListTyped(Box::new(PropertyDataType::Int64)),
                ),
                property("details", PropertyDataType::Map),
                property("flag", PropertyDataType::Bytes),
                property("mayor", PropertyDataType::Node),
                property("road", PropertyDataType::Edge),
                property("notes", PropertyDataType::Any),
            ],
            constraints: vec![
                TypeConstraint::PrimaryKey(vec!["name".to_string()]),
                TypeConstraint::Unique(vec!["name".to_string(), "founded".to_string()]),
                TypeConstraint::NotNull("population".to_string()),
                TypeConstraint::Check {
                    name: Some("populated".to_string()),
                    expression: "population > 3".to_string(),
                },
            ],
            parent_types: vec!["Place".to_string()],
            key_labels: Vec::new(),
        });
        catalog.register_or_replace_edge_type_def(EdgeTypeDefinition {
            name: "ROAD".to_string(),
            properties: vec![TypedProperty {
                name: "km".to_string(),
                data_type: PropertyDataType::Float64,
                nullable: false,
                default_value: Some(Value::Float64(19.0)),
            }],
            constraints: vec![TypeConstraint::Check {
                name: None,
                expression: "km > 0".to_string(),
            }],
            source_node_types: vec!["City".to_string()],
            target_node_types: vec!["City".to_string(), "Place".to_string()],
            key_labels: Vec::new(),
        });
        catalog
            .register_graph_type(GraphTypeDefinition {
                name: "Atlas".to_string(),
                allowed_node_types: vec!["City".to_string()],
                allowed_edge_types: vec!["ROAD".to_string()],
                open: false,
            })
            .unwrap();
        catalog
            .register_schema_namespace("travel".to_string())
            .unwrap();
        catalog
            .bind_graph_type("europe", "Atlas".to_string())
            .unwrap();
        catalog
            .register_procedure(ProcedureDefinition {
                name: "cities_near".to_string(),
                params: vec![("city".to_string(), "STRING".to_string())],
                returns: vec![("name".to_string(), "STRING".to_string())],
                body: "MATCH (c:City) RETURN c.name AS name".to_string(),
            })
            .unwrap();
        for (name, kind) in [
            ("city_exists", ConstraintType::Exists),
            ("city_key", ConstraintType::NodeKey),
            ("city_name", ConstraintType::Unique),
            ("city_present", ConstraintType::NotNull),
        ] {
            catalog
                .create_constraint(ConstraintDefinition {
                    name: name.to_string(),
                    label: "City".to_string(),
                    properties: vec!["name".to_string()],
                    kind,
                })
                .unwrap();
        }
        section.store.create_property_index("name");
        let label = catalog.get_or_create_label("City");
        let name = catalog.get_or_create_property_key("name");
        for (index, index_type) in [
            ("city_btree", IndexType::BTree),
            ("city_hash", IndexType::Hash),
            ("city_text", IndexType::FullText),
        ] {
            catalog.create_index(index, label, name, index_type);
        }
    }

    /// The version 1 catalog (and the snapshot v4, which serializes the same
    /// types) of 0.5.x files is bincode of the in-memory catalog types: this
    /// pins today's bytes, so a variant inserted before another or a new
    /// serialized field fails here instead of misreading released files.
    #[test]
    fn the_0_5_catalog_layout_is_unchanged() {
        const EXPECTED: &[u8] = &[
            1, 1, 4, 67, 105, 116, 121, 15, 4, 110, 97, 109, 101, 0, 0, 1, 4, 9, 65, 109, 115, 116,
            101, 114, 100, 97, 109, 10, 112, 111, 112, 117, 108, 97, 116, 105, 111, 110, 1, 1, 1,
            2, 176, 4, 97, 114, 101, 97, 2, 1, 0, 7, 99, 97, 112, 105, 116, 97, 108, 3, 1, 0, 7,
            102, 111, 117, 110, 100, 101, 100, 4, 1, 0, 5, 111, 112, 101, 110, 115, 5, 1, 0, 7,
            117, 112, 100, 97, 116, 101, 100, 6, 1, 0, 4, 116, 114, 105, 112, 7, 1, 0, 9, 100, 105,
            115, 116, 114, 105, 99, 116, 115, 8, 1, 0, 9, 122, 105, 112, 95, 99, 111, 100, 101,
            115, 9, 1, 1, 0, 7, 100, 101, 116, 97, 105, 108, 115, 10, 1, 0, 4, 102, 108, 97, 103,
            11, 1, 0, 5, 109, 97, 121, 111, 114, 12, 1, 0, 4, 114, 111, 97, 100, 13, 1, 0, 5, 110,
            111, 116, 101, 115, 14, 1, 0, 8, 0, 1, 4, 110, 97, 109, 101, 1, 2, 4, 110, 97, 109,
            101, 7, 102, 111, 117, 110, 100, 101, 100, 2, 10, 112, 111, 112, 117, 108, 97, 116,
            105, 111, 110, 3, 1, 9, 112, 111, 112, 117, 108, 97, 116, 101, 100, 14, 112, 111, 112,
            117, 108, 97, 116, 105, 111, 110, 32, 62, 32, 51, 2, 4, 110, 97, 109, 101, 0, 1, 4,
            110, 97, 109, 101, 1, 1, 4, 110, 97, 109, 101, 2, 4, 110, 97, 109, 101, 1, 5, 80, 108,
            97, 99, 101, 1, 4, 82, 79, 65, 68, 1, 2, 107, 109, 2, 0, 1, 3, 0, 0, 0, 0, 0, 0, 51,
            64, 1, 3, 0, 6, 107, 109, 32, 62, 32, 48, 1, 4, 67, 105, 116, 121, 2, 4, 67, 105, 116,
            121, 5, 80, 108, 97, 99, 101, 1, 5, 65, 116, 108, 97, 115, 1, 4, 67, 105, 116, 121, 1,
            4, 82, 79, 65, 68, 0, 1, 11, 99, 105, 116, 105, 101, 115, 95, 110, 101, 97, 114, 1, 4,
            99, 105, 116, 121, 6, 83, 84, 82, 73, 78, 71, 1, 4, 110, 97, 109, 101, 6, 83, 84, 82,
            73, 78, 71, 36, 77, 65, 84, 67, 72, 32, 40, 99, 58, 67, 105, 116, 121, 41, 32, 82, 69,
            84, 85, 82, 78, 32, 99, 46, 110, 97, 109, 101, 32, 65, 83, 32, 110, 97, 109, 101, 1, 6,
            116, 114, 97, 118, 101, 108, 1, 6, 101, 117, 114, 111, 112, 101, 5, 65, 116, 108, 97,
            115, 1, 4, 110, 97, 109, 101, 0, 0, 42, 4, 11, 99, 105, 116, 121, 95, 101, 120, 105,
            115, 116, 115, 4, 67, 105, 116, 121, 1, 4, 110, 97, 109, 101, 3, 8, 99, 105, 116, 121,
            95, 107, 101, 121, 4, 67, 105, 116, 121, 1, 4, 110, 97, 109, 101, 1, 9, 99, 105, 116,
            121, 95, 110, 97, 109, 101, 4, 67, 105, 116, 121, 1, 4, 110, 97, 109, 101, 0, 12, 99,
            105, 116, 121, 95, 112, 114, 101, 115, 101, 110, 116, 4, 67, 105, 116, 121, 1, 4, 110,
            97, 109, 101, 2, 1, 0, 1, 4, 110, 97, 109, 101, 0, 0, 3, 10, 99, 105, 116, 121, 95, 98,
            116, 114, 101, 101, 4, 67, 105, 116, 121, 4, 110, 97, 109, 101, 1, 9, 99, 105, 116,
            121, 95, 104, 97, 115, 104, 4, 67, 105, 116, 121, 4, 110, 97, 109, 101, 0, 9, 99, 105,
            116, 121, 95, 116, 101, 120, 116, 4, 67, 105, 116, 121, 4, 110, 97, 109, 101, 2,
        ];
        let section = make_section();
        populate_every_0_5_entry(&section);
        let bytes = section.serialize().unwrap();
        assert_eq!(bytes, EXPECTED, "today's layout: {bytes:?}");
    }

    // ── Version 2: catalog records ──────────────────────────────────

    use grafeo_common::storage::catalog_record::{
        GraphBindingRecord, IndexKindRecord, IndexRecord, NodeTypeRecord, ProcedureRecord,
        SchemaRecord,
    };
    use grafeo_common::storage::{
        CatalogRecord, ChunkCaps, ChunkMeta, ImageSource, MemoryImage, RECORD_REQUIRED,
    };
    use grafeo_common::testing::chunk_caps::with_chunk_caps;

    /// The caps of the tests that cut the record stream into many pieces.
    const TINY: ChunkCaps = ChunkCaps {
        max_rows: 3,
        max_bytes: 64,
    };

    /// Everything a catalog holds, in an order that does not depend on the
    /// hash maps it keeps it in.
    fn describe(catalog: &Catalog) -> Vec<String> {
        let mut lines: Vec<String> = Vec::new();
        lines.extend(
            catalog
                .all_node_type_defs()
                .iter()
                .map(|def| format!("node type {def:?}")),
        );
        lines.extend(
            catalog
                .all_edge_type_defs()
                .iter()
                .map(|def| format!("edge type {def:?}")),
        );
        lines.extend(
            catalog
                .all_graph_type_defs()
                .iter()
                .map(|def| format!("graph type {def:?}")),
        );
        lines.extend(
            catalog
                .all_procedure_defs()
                .iter()
                .map(|def| format!("procedure {def:?}")),
        );
        lines.extend(
            catalog
                .schema_names()
                .iter()
                .map(|name| format!("schema {name}")),
        );
        lines.extend(
            catalog
                .all_graph_type_bindings()
                .iter()
                .map(|binding| format!("binding {binding:?}")),
        );
        lines.extend(
            catalog
                .constraints()
                .iter()
                .map(|def| format!("constraint {def:?}")),
        );
        lines.extend(catalog.all_indexes().iter().map(|def| {
            format!(
                "index name {} on {:?}.{:?}, {:?}",
                def.name,
                catalog.get_label_name(def.label),
                catalog.get_property_key_name(def.property_key),
                def.index_type
            )
        }));
        lines.sort();
        lines
    }

    /// [`populate_every_0_5_entry`] plus what only version 2 holds or what
    /// 0.5.x files did not have: `ZONED DATETIME`, `LOCAL DATETIME` and typed
    /// list properties, key labels, a schema more, one index name on two
    /// properties, the default graph's vector and text indexes and the
    /// property and vector indexes of a named graph.
    fn populate_every_entry(section: &CatalogSection) {
        use crate::catalog::PropertyDataType;
        use grafeo_common::types::{Value, ZonedDatetime};

        populate_every_0_5_entry(section);
        let catalog = &section.catalog;
        let property =
            |name: &str, data_type: PropertyDataType, default_value: Option<Value>| TypedProperty {
                name: name.to_string(),
                data_type,
                nullable: true,
                default_value,
            };
        catalog.register_or_replace_node_type(NodeTypeDefinition {
            name: "Event".to_string(),
            properties: vec![
                property(
                    "starts",
                    PropertyDataType::ZonedDatetime,
                    Some(Value::ZonedDatetime(
                        ZonedDatetime::parse("2026-10-05T10:30:00.000019+02:00").unwrap(),
                    )),
                ),
                property("departs", PropertyDataType::LocalDatetime, None),
                property(
                    "tags",
                    PropertyDataType::ListTyped(Box::new(PropertyDataType::String)),
                    Some(Value::List(vec![Value::from("Berlin")].into())),
                ),
                property(
                    "stamps",
                    PropertyDataType::ListTyped(Box::new(PropertyDataType::ListTyped(Box::new(
                        PropertyDataType::ZonedDatetime,
                    )))),
                    None,
                ),
            ],
            constraints: Vec::new(),
            parent_types: vec!["EventKey".to_string()],
            key_labels: vec!["EventKey".to_string()],
        });
        catalog.register_or_replace_edge_type_def(EdgeTypeDefinition {
            name: "BOOKED".to_string(),
            properties: vec![property("booked", PropertyDataType::ZonedDatetime, None)],
            constraints: Vec::new(),
            source_node_types: vec!["Person".to_string(), "Museum".to_string()],
            target_node_types: vec!["Event".to_string()],
            key_labels: vec!["BookingKey".to_string()],
        });
        catalog
            .register_schema_namespace("archive".to_string())
            .unwrap();
        let doc = catalog.get_or_create_label("Doc");
        for property in ["body", "title"] {
            let key = catalog.get_or_create_property_key(property);
            catalog.create_index("doc_terms", doc, key, IndexType::FullText);
        }
        // `CREATE INDEX` run twice for one property: two indexes of one name.
        let summary = catalog.get_or_create_property_key("summary");
        for _ in 0..2 {
            catalog.create_index("doc_summary", doc, summary, IndexType::BTree);
        }

        section.store.create_graph("model").unwrap();
        let model = section.store.graph("model").unwrap();
        model.create_property_index("key");
        #[cfg(feature = "vector-index")]
        {
            use grafeo_core::index::vector::{DistanceMetric, QuantizationType};
            section.store.add_vector_index(
                "Doc",
                "emb",
                Arc::new(crate::GrafeoDB::build_vector_index(
                    3,
                    DistanceMetric::Cosine,
                    Some(19),
                    Some(88),
                    QuantizationType::None,
                    0,
                )),
            );
            model.add_vector_index(
                "Page",
                "emb",
                Arc::new(crate::GrafeoDB::build_vector_index(
                    16,
                    DistanceMetric::Euclidean,
                    None,
                    None,
                    QuantizationType::Product { num_subvectors: 8 },
                    0,
                )),
            );
        }
        #[cfg(feature = "text-index")]
        section.store.add_text_index(
            "Doc",
            "body",
            Arc::new(parking_lot::RwLock::new(
                grafeo_core::index::text::InvertedIndex::new(
                    grafeo_core::index::text::BM25Config::default(),
                ),
            )),
        );
    }

    /// Loads the catalog section of `image` into a new catalog.
    fn load(image: &MemoryImage) -> Result<CatalogSection> {
        let mut section = make_section();
        let source = image
            .section_source(SectionType::Catalog)
            .expect("the image has a catalog section");
        section.read_from(&*source)?;
        Ok(section)
    }

    /// The error of loading the catalog section of `image`.
    fn load_error(image: &MemoryImage) -> String {
        match load(image) {
            Ok(_) => panic!("the catalog section loads"),
            Err(error) => error.to_string(),
        }
    }

    /// The metadata chunk this release writes with the tiny caps.
    fn bincode_meta() -> Vec<u8> {
        bincode::serde::encode_to_vec(
            CatalogMeta {
                layout: 1,
                max_bytes: TINY.max_bytes,
            },
            bincode::config::standard(),
        )
        .unwrap()
    }

    /// A catalog section of `version` holding `chunks`.
    fn image_of(version: u8, chunks: &[(ChunkMeta, Vec<u8>)]) -> MemoryImage {
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::Catalog, version).unwrap();
        for (meta, bytes) in chunks {
            image.write_chunk(*meta, bytes).unwrap();
        }
        image
    }

    /// A version 2 catalog section holding `records`, framed, as one piece.
    fn image_of_records(records: &[CatalogRecord]) -> MemoryImage {
        let mut bytes = Vec::new();
        for record in records {
            record.encode_framed(&mut bytes).unwrap();
        }
        image_of(
            2,
            &[
                (ChunkMeta::meta(), bincode_meta()),
                (ChunkMeta::stream_piece(0, 0, 0), bytes),
            ],
        )
    }

    /// Every chunk of the catalog section of `image`, with its bytes.
    fn chunks_of(image: &MemoryImage) -> Vec<(ChunkMeta, Vec<u8>)> {
        let source = image.section_source(SectionType::Catalog).unwrap();
        (0..source.chunks().len())
            .map(|index| {
                (
                    source.chunks()[index],
                    source.fetch(index).unwrap().to_vec(),
                )
            })
            .collect()
    }

    #[test]
    fn every_kind_of_catalog_entry_round_trips_through_records() {
        // The section takes its caps when it is built, so it is built under
        // the small caps.
        let section = with_chunk_caps(TINY, make_section);
        populate_every_entry(&section);
        let image = MemoryImage::from_sections(&[&section]).unwrap();
        let source = image.section_source(SectionType::Catalog).unwrap();
        assert_eq!(source.section_version(), 2);
        assert!(
            source.chunks().len() > 3,
            "a 64-byte cap cuts the record stream into many pieces: {} chunks",
            source.chunks().len()
        );
        let mut loaded = load(&image).unwrap();
        assert_eq!(describe(&loaded.catalog), describe(&section.catalog));
        let written: Vec<GraphIndexes> = section
            .collect_graph_indexes()
            .into_iter()
            .filter(|graph| !graph.is_empty())
            .collect();
        assert_eq!(written.len(), 2, "the default graph and model: {written:?}");
        assert_eq!(loaded.take_loaded_indexes(), written);
        assert!(
            loaded.store.graph("travel/__default__").is_some()
                && loaded.store.graph("archive/__default__").is_some(),
            "a schema brings its default graph"
        );
        assert_eq!(
            loaded.catalog.get_node_type("Event").unwrap().key_labels,
            ["EventKey"]
        );
        assert_eq!(
            loaded
                .catalog
                .get_edge_type_def("BOOKED")
                .unwrap()
                .key_labels,
            ["BookingKey"]
        );
    }

    /// Two catalogs holding the same entries, registered in opposite orders,
    /// are written to the same bytes: the records come sorted, never in the
    /// order of the hash maps the catalog keeps them in.
    #[test]
    fn the_records_are_written_in_one_order() {
        use crate::catalog::ConstraintType;

        const NAMES: [&str; 8] = [
            "Alix",
            "Gus",
            "Vincent",
            "Jules",
            "Mia",
            "Amsterdam",
            "Berlin",
            "Prague",
        ];
        let written = |names: &mut dyn Iterator<Item = &&str>| {
            let section = with_chunk_caps(TINY, make_section);
            let catalog = &section.catalog;
            for name in names {
                catalog.register_or_replace_node_type(NodeTypeDefinition {
                    name: (*name).to_string(),
                    properties: Vec::new(),
                    constraints: Vec::new(),
                    parent_types: Vec::new(),
                    key_labels: Vec::new(),
                });
                catalog.register_or_replace_edge_type_def(EdgeTypeDefinition {
                    name: name.to_uppercase(),
                    properties: Vec::new(),
                    constraints: Vec::new(),
                    source_node_types: Vec::new(),
                    target_node_types: Vec::new(),
                    key_labels: Vec::new(),
                });
                catalog
                    .register_graph_type(GraphTypeDefinition {
                        name: format!("{name}_type"),
                        allowed_node_types: vec![(*name).to_string()],
                        allowed_edge_types: Vec::new(),
                        open: true,
                    })
                    .unwrap();
                catalog
                    .bind_graph_type(&format!("{name}_graph"), format!("{name}_type"))
                    .unwrap();
                catalog
                    .register_schema_namespace(format!("{name}_schema"))
                    .unwrap();
                catalog
                    .register_procedure(ProcedureDefinition {
                        name: format!("{name}_procedure"),
                        params: Vec::new(),
                        returns: Vec::new(),
                        body: "RETURN 3".to_string(),
                    })
                    .unwrap();
                catalog
                    .create_constraint(ConstraintDefinition {
                        name: format!("{name}_unique"),
                        label: (*name).to_string(),
                        properties: vec!["key".to_string()],
                        kind: ConstraintType::Unique,
                    })
                    .unwrap();
                let label = catalog.get_or_create_label(name);
                let key = catalog.get_or_create_property_key("key");
                catalog.create_index(&format!("{name}_index"), label, key, IndexType::Hash);
                section.store.create_property_index(name);
                section.store.create_graph(name).unwrap();
                section
                    .store
                    .graph(name)
                    .unwrap()
                    .create_property_index(name);
            }
            chunks_of(&MemoryImage::from_sections(&[&section]).unwrap())
        };
        let forward = written(&mut NAMES.iter());
        assert_eq!(forward, written(&mut NAMES.iter().rev()));
        assert!(forward.len() > 3, "{} chunks", forward.len());
    }

    /// A catalog without entries is its metadata chunk alone: the layout
    /// byte and the byte cap it was written with.
    #[test]
    fn an_empty_catalog_is_its_metadata_chunk_alone() {
        let section = with_chunk_caps(TINY, make_section);
        let image = MemoryImage::from_sections(&[&section]).unwrap();
        assert_eq!(chunks_of(&image), [(ChunkMeta::meta(), vec![1, 64])]);
        let mut loaded = load(&image).unwrap();
        assert_eq!(describe(&loaded.catalog), Vec::<String>::new());
        assert_eq!(loaded.take_loaded_indexes(), Vec::new());
    }

    fn person_type() -> NodeTypeDefinition {
        NodeTypeDefinition {
            name: "Person".to_string(),
            properties: vec![TypedProperty {
                name: "name".to_string(),
                data_type: crate::catalog::PropertyDataType::String,
                nullable: false,
                default_value: None,
            }],
            constraints: Vec::new(),
            parent_types: Vec::new(),
            key_labels: Vec::new(),
        }
    }

    #[test]
    fn key_labels_stay_out_of_the_0_5_layouts() {
        let section = make_section();
        let mut person = person_type();
        person.key_labels = vec!["PersonKey".into()];
        section
            .catalog
            .register_or_replace_node_type(person.clone());
        let knows = EdgeTypeDefinition {
            name: "KNOWS".to_string(),
            properties: Vec::new(),
            constraints: Vec::new(),
            source_node_types: Vec::new(),
            target_node_types: Vec::new(),
            key_labels: vec!["KnowsKey".into()],
        };
        section
            .catalog
            .register_or_replace_edge_type_def(knows.clone());
        let with_keys = section.serialize().unwrap();
        person.key_labels.clear();
        section.catalog.register_or_replace_node_type(person);
        section
            .catalog
            .register_or_replace_edge_type_def(EdgeTypeDefinition {
                key_labels: Vec::new(),
                ..knows.clone()
            });
        assert_eq!(
            with_keys,
            section.serialize().unwrap(),
            "the version 1 layout has no key labels"
        );
        // ...and version 2 keeps them:
        section
            .catalog
            .register_or_replace_node_type(NodeTypeDefinition {
                key_labels: vec!["PersonKey".into()],
                ..person_type()
            });
        section.catalog.register_or_replace_edge_type_def(knows);
        let image = MemoryImage::from_sections(&[&section]).unwrap();
        let loaded = load(&image).unwrap();
        assert_eq!(
            loaded.catalog.get_node_type("Person").unwrap().key_labels,
            ["PersonKey"]
        );
        assert_eq!(
            loaded
                .catalog
                .get_edge_type_def("KNOWS")
                .unwrap()
                .key_labels,
            ["KnowsKey"]
        );
    }

    /// The `KEY (...)` labels of the element types a `CREATE GRAPH TYPE`
    /// declares inline reach the records and come back; a node type keeps
    /// them as parent types too, as before.
    #[cfg(feature = "gql")]
    #[test]
    fn key_labels_of_inline_element_types_come_back() {
        let db = crate::GrafeoDB::new_in_memory();
        db.execute(
            "CREATE GRAPH TYPE trips (NODE TYPE Stop KEY (StopKey, Place) (name STRING),              EDGE TYPE LEG KEY (LegKey) (km INT64))",
        )
        .unwrap();
        let section = CatalogSection::new(
            Arc::clone(&db.catalog),
            Arc::new(LpgStore::new().unwrap()),
            || 0,
        );
        let loaded = load(&MemoryImage::from_sections(&[&section]).unwrap()).unwrap();
        let stop = loaded.catalog.get_node_type("Stop").unwrap();
        assert_eq!(stop.key_labels, ["StopKey", "Place"]);
        assert_eq!(stop.parent_types, ["StopKey", "Place"]);
        assert_eq!(
            loaded.catalog.get_edge_type_def("LEG").unwrap().key_labels,
            ["LegKey"]
        );
    }

    #[test]
    fn a_0_5_catalog_still_loads_through_a_raw_chunk() {
        let section = make_section();
        populate_every_0_5_entry(&section);
        let image =
            MemoryImage::from_raw(vec![(SectionType::Catalog, section.serialize().unwrap())])
                .unwrap();
        let mut loaded = load(&image).unwrap();
        assert_eq!(describe(&loaded.catalog), describe(&section.catalog));
        assert_eq!(
            loaded.take_loaded_indexes(),
            [GraphIndexes {
                graph: None,
                property: vec!["name".to_string()],
                ..GraphIndexes::default()
            }]
        );
    }

    /// A binding to a graph type dropped since binds nothing: it is left
    /// out, and the catalog loads with the other bindings.
    #[test]
    fn a_binding_to_a_dropped_graph_type_is_left_out() {
        let section = make_section();
        let catalog = &section.catalog;
        for (graph, graph_type) in [("europe", "Atlas"), ("world", "Globe")] {
            catalog
                .register_graph_type(GraphTypeDefinition {
                    name: graph_type.to_string(),
                    allowed_node_types: vec!["City".to_string()],
                    allowed_edge_types: Vec::new(),
                    open: false,
                })
                .unwrap();
            catalog
                .bind_graph_type(graph, graph_type.to_string())
                .unwrap();
        }
        catalog.drop_graph_type("Atlas").unwrap();
        assert_eq!(
            catalog.get_graph_type_binding("europe").as_deref(),
            Some("Atlas")
        );

        let loaded = load(&MemoryImage::from_sections(&[&section]).unwrap()).unwrap();
        assert_eq!(
            loaded.catalog.all_graph_type_bindings(),
            [("world".to_string(), "Globe".to_string())]
        );
    }

    #[test]
    fn a_record_that_does_not_apply_fails_the_load() {
        // A stream holding a GraphBinding for a graph type no record defines.
        let image = image_of_records(&[CatalogRecord::GraphBinding(GraphBindingRecord {
            graph: "trips".into(),
            graph_type: "missing".into(),
        })]);
        let error = load_error(&image);
        assert!(
            error.contains("Catalog") && error.contains("trips") && error.contains("missing"),
            "{error}"
        );
    }

    /// Chunks a version 2 catalog section never holds are refused, naming
    /// the section: another version, a first chunk that is not the metadata
    /// chunk, another layout, bytes after the metadata, chunks other than
    /// pieces of stream 0 of graph 0, and a raw chunk next to them.
    #[test]
    fn chunks_the_catalog_does_not_write_are_refused() {
        let meta = (ChunkMeta::meta(), bincode_meta());
        let piece = |stream: u32| {
            let mut bytes = Vec::new();
            CatalogRecord::Schema(SchemaRecord {
                name: "travel".into(),
            })
            .encode_framed(&mut bytes)
            .unwrap();
            (ChunkMeta::stream_piece(0, stream, 0), bytes)
        };
        let mut long_meta = bincode_meta();
        long_meta.push(19);
        for (image, expected) in [
            (image_of(3, &[meta.clone(), piece(0)]), "version 3"),
            (image_of(2, &[piece(0)]), "where the metadata chunk belongs"),
            (
                image_of(2, &[(ChunkMeta::meta(), vec![2, 64]), piece(0)]),
                "layout 2",
            ),
            (image_of(2, &[(ChunkMeta::meta(), Vec::new())]), "is empty"),
            (
                image_of(2, &[(ChunkMeta::meta(), long_meta), piece(0)]),
                "after its fields",
            ),
            (image_of(2, &[meta.clone(), piece(1)]), "stream 1"),
            (
                image_of(
                    2,
                    &[
                        meta.clone(),
                        piece(0),
                        (ChunkMeta::stream_piece(3, 0, 0), vec![19]),
                    ],
                ),
                "graph 3",
            ),
            (
                image_of(
                    2,
                    &[meta.clone(), (ChunkMeta::column(0, 0, 0, 1, 0), vec![88])],
                ),
                "Column",
            ),
            (
                image_of(2, &[meta.clone(), (ChunkMeta::raw(), vec![88])]),
                "raw",
            ),
        ] {
            let error = match load(&image) {
                Ok(_) => panic!("the section expected to fail with {expected:?} loads"),
                Err(error) => error.to_string(),
            };
            assert!(
                error.contains("Catalog") && error.contains(expected),
                "expected {expected:?}: {error}"
            );
        }
    }

    /// The records of each kind come once each, in the order of their names
    /// (index records: by graph, the default graph first, then property,
    /// vector and text indexes, each by name): a repeated entry or one out of
    /// order is refused, naming it. A record of a kind this release does not
    /// know is skipped when it is optional.
    #[test]
    fn records_repeated_or_out_of_order_are_refused() {
        let node_type = |name: &str| {
            CatalogRecord::NodeType(NodeTypeRecord {
                name: name.to_string(),
                properties: Vec::new(),
                constraints: Vec::new(),
                parent_types: Vec::new(),
                key_labels: Vec::new(),
            })
        };
        let schema = |name: &str| {
            CatalogRecord::Schema(SchemaRecord {
                name: name.to_string(),
            })
        };
        let property_index = |graph: Option<&str>, key: &str| {
            CatalogRecord::Index(IndexRecord {
                graph: graph.map(str::to_string),
                index: IndexKindRecord::Property {
                    key: key.to_string(),
                },
            })
        };
        let procedure = |name: &str| {
            CatalogRecord::Procedure(ProcedureRecord {
                name: name.to_string(),
                params: Vec::new(),
                returns: Vec::new(),
                body: "RETURN 3".to_string(),
            })
        };
        for (records, expected) in [
            (vec![node_type("City"), node_type("City")], "City"),
            (vec![schema("travel"), schema("archive")], "archive"),
            (vec![procedure("Paris"), procedure("Berlin")], "Berlin"),
            (
                vec![property_index(None, "id"), property_index(None, "id")],
                "id",
            ),
            (
                vec![
                    property_index(Some("model"), "id"),
                    property_index(None, "name"),
                ],
                "name",
            ),
        ] {
            let error = load_error(&image_of_records(&records));
            assert!(
                error.contains("Catalog") && error.contains(expected) && error.contains("order"),
                "expected {expected:?}: {error}"
            );
        }

        // An unknown optional record between two known ones is skipped.
        let mut bytes = Vec::new();
        schema("archive").encode_framed(&mut bytes).unwrap();
        let unknown = bytes.len();
        bytes.extend_from_slice(&[99, 0, 3, 0, 0, 0, 3, 19, 88]);
        schema("travel").encode_framed(&mut bytes).unwrap();
        let image = image_of(
            2,
            &[
                (ChunkMeta::meta(), bincode_meta()),
                (ChunkMeta::stream_piece(0, 0, 0), bytes.clone()),
            ],
        );
        assert_eq!(
            load(&image).unwrap().catalog.schema_names(),
            ["archive", "travel"]
        );
        // The same record required is refused.
        bytes[unknown + 1] = RECORD_REQUIRED;
        let image = image_of(
            2,
            &[
                (ChunkMeta::meta(), bincode_meta()),
                (ChunkMeta::stream_piece(0, 0, 0), bytes),
            ],
        );
        let error = load_error(&image);
        assert!(error.contains("Catalog") && error.contains("99"), "{error}");
    }

    /// An entry too large for a catalog record fails the checkpoint with an
    /// error naming it, instead of writing a record no reader takes.
    #[test]
    fn an_entry_too_large_for_a_record_fails_the_write_naming_it() {
        use grafeo_common::storage::catalog_record::MAX_CATALOG_RECORD_PAYLOAD;

        let section = make_section();
        section
            .catalog
            .register_procedure(ProcedureDefinition {
                name: "cities_near".to_string(),
                params: Vec::new(),
                returns: Vec::new(),
                body: "R".repeat(usize::try_from(MAX_CATALOG_RECORD_PAYLOAD).unwrap()),
            })
            .unwrap();
        let error = MemoryImage::from_sections(&[&section])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("Catalog") && error.contains("cities_near"),
            "{error}"
        );
    }

    /// A node or edge type whose property types nest more `LIST` levels in
    /// all than a catalog record holds (one the catalog API registers, as no
    /// statement builds one) fails the checkpoint with an error naming it.
    #[test]
    fn a_type_past_the_list_level_cap_fails_the_write_naming_it() {
        use grafeo_common::storage::catalog_record::{
            MAX_LIST_LEVELS_PER_RECORD, MAX_LIST_TYPE_DEPTH,
        };

        use crate::catalog::PropertyDataType;

        let deepest = (0..MAX_LIST_TYPE_DEPTH).fold(PropertyDataType::Int64, |inner, _| {
            PropertyDataType::ListTyped(Box::new(inner))
        });
        let mut properties: Vec<TypedProperty> = (0..MAX_LIST_LEVELS_PER_RECORD
            / MAX_LIST_TYPE_DEPTH)
            .map(|n| TypedProperty {
                name: format!("p{n}"),
                data_type: deepest.clone(),
                nullable: true,
                default_value: None,
            })
            .collect();
        properties.push(TypedProperty {
            name: "extra".to_string(),
            data_type: PropertyDataType::ListTyped(Box::new(PropertyDataType::Int64)),
            nullable: true,
            default_value: None,
        });
        for (edge, name) in [(false, "node type 'Wide'"), (true, "edge type 'WIDE'")] {
            let section = make_section();
            if edge {
                section
                    .catalog
                    .register_or_replace_edge_type_def(EdgeTypeDefinition {
                        name: "WIDE".to_string(),
                        properties: properties.clone(),
                        constraints: vec![],
                        source_node_types: vec![],
                        target_node_types: vec![],
                        key_labels: vec![],
                    });
            } else {
                section
                    .catalog
                    .register_or_replace_node_type(NodeTypeDefinition {
                        name: "Wide".to_string(),
                        properties: properties.clone(),
                        constraints: vec![],
                        parent_types: vec![],
                        key_labels: vec![],
                    });
            }
            let error = MemoryImage::from_sections(&[&section])
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("Catalog")
                    && error.contains(name)
                    && error.contains("32769 LIST levels"),
                "{error}"
            );
        }
    }

    /// An edge type whose endpoint lists make more (source, target) pairs
    /// than a catalog record holds fails the checkpoint naming it, from the
    /// lists' sizes, before a pair is built: 1,000 types on each side would
    /// make a million pairs.
    #[test]
    fn an_edge_type_with_too_many_endpoint_pairs_fails_the_write_naming_it() {
        let section = make_section();
        let cities: Vec<String> = (0..1_000).map(|n| format!("City{n:04}")).collect();
        section
            .catalog
            .register_or_replace_edge_type_def(EdgeTypeDefinition {
                name: "ROUTE".to_string(),
                properties: vec![],
                constraints: vec![],
                source_node_types: cities.clone(),
                target_node_types: cities,
                key_labels: vec![],
            });
        let error = MemoryImage::from_sections(&[&section])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("Catalog")
                && error.contains("edge type 'ROUTE'")
                && error.contains("1000000 endpoint pairs"),
            "{error}"
        );
    }
}
