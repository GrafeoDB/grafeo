//! Catalog section serializer for the `.grafeo` container format.
//!
//! Serializes schema definitions (node types, edge types, graph types, procedures),
//! index metadata (property, vector, text), and epoch state into the `CATALOG` section.

// Parts of this module are reserved for Phase 5 checkpoint integration.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};

use grafeo_common::storage::section::{
    Section, SectionSink, SectionSource, SectionType, read_raw, write_raw,
};
use grafeo_common::utils::error::{Error, Result};
use grafeo_core::graph::lpg::LpgStore;
use grafeo_core::index::vector::{DistanceMetric, QuantizationType};

use crate::catalog::{
    Catalog, ConstraintDefinition, EdgeTypeDefinition, GraphTypeDefinition, IndexType,
    NodeTypeDefinition, ProcedureDefinition,
};

/// Current catalog section format version.
const CATALOG_SECTION_VERSION: u8 = 1;

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
    /// Text indexes, as `(label, property)`.
    pub text: Vec<(String, String)>,
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
        let mut text: Vec<(String, String)> = store
            .text_index_entries()
            .into_iter()
            .filter_map(|(key, _)| {
                let (label, property) = key.split_once(':')?;
                Some((label.to_string(), property.to_string()))
            })
            .collect();
        #[cfg(not(feature = "text-index"))]
        let mut text: Vec<(String, String)> = Vec::new();
        text.sort();

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

/// The name `CREATE INDEX` gave an index, for `SHOW INDEXES` and
/// `DROP INDEX`.
#[derive(Serialize, Deserialize)]
struct IndexName {
    name: String,
    label: String,
    property: String,
    index_type: IndexType,
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
                .map(|(label, property)| SnapshotTextIndex {
                    label: label.clone(),
                    property: property.clone(),
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
                .map(|def| (def.label, def.property))
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
/// Serializes schema definitions and index metadata. The catalog is always
/// small (typically < 10 KB) and always kept in RAM.
pub struct CatalogSection {
    catalog: Arc<Catalog>,
    store: Arc<LpgStore>,
    epoch_fn: Box<dyn Fn() -> u64 + Send + Sync>,
    dirty: AtomicBool,
    /// The index definitions the last `deserialize` read.
    loaded_indexes: Vec<GraphIndexes>,
}

impl CatalogSection {
    /// Create a new catalog section.
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
        }
    }

    /// The index definitions of every graph that `deserialize` read, for the
    /// loader to build once the data is in (the catalog holds only their
    /// names).
    pub(crate) fn take_loaded_indexes(&mut self) -> Vec<GraphIndexes> {
        std::mem::take(&mut self.loaded_indexes)
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

    /// The names `CREATE INDEX` gave indexes.
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
        names.sort_by(|a, b| a.name.cmp(&b.name));
        names
    }

    /// Registers the index names in the catalog again.
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
}

impl Section for CatalogSection {
    fn section_type(&self) -> SectionType {
        SectionType::Catalog
    }

    fn version(&self) -> u8 {
        CATALOG_SECTION_VERSION
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        // The names and the node types that enforce them, from one moment.
        let (schema, constraints) = self
            .catalog
            .with_constraints(|constraints| (self.collect_schema(), constraints));
        let mut graphs = self.collect_graph_indexes();
        let snapshot = CatalogSnapshot {
            version: CATALOG_SECTION_VERSION,
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

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        let config = bincode::config::standard();
        let (snapshot, read): (CatalogSnapshot, _) =
            bincode::serde::decode_from_slice(data, config).map_err(|e| {
                Error::Serialization(format!("Catalog section deserialization failed: {e}"))
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
        // The node types restored above already hold the constraints.
        let mut rest = &data[read..];
        if !rest.is_empty() {
            let (names, read): (ConstraintNames, _) =
                bincode::serde::decode_from_slice(rest, config).map_err(|e| {
                    Error::Serialization(format!("Constraint names deserialization failed: {e}"))
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
                bincode::serde::decode_from_slice(rest, config).map_err(|e| {
                    Error::Serialization(format!("Index deserialization failed: {e}"))
                })?;
            self.restore_index_names(extension.names);
            extension.graphs
        };

        Ok(())
    }

    /// The catalog is still written whole: [`serialize`](Section::serialize)
    /// as one raw chunk, the only section of a checkpoint written so. The
    /// catalog records of version 2 (#517) replace this and
    /// [`read_from`](Section::read_from).
    fn write_to(&self, sink: &mut dyn SectionSink) -> Result<()> {
        write_raw(self, sink)
    }

    /// The one raw chunk a checkpoint writes and a 0.5.x file holds, passed
    /// to [`deserialize`](Section::deserialize).
    fn read_from(&mut self, source: &dyn SectionSource) -> Result<()> {
        read_raw(self, source)
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
}
