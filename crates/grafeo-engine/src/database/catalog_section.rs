//! Catalog section serializer for the `.grafeo` container format.
//!
//! Serializes schema definitions (node types, edge types, graph types, procedures),
//! index metadata (property, vector, text), and epoch state into the `CATALOG` section.

// Parts of this module are reserved for Phase 5 checkpoint integration.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};

use grafeo_common::storage::section::{Section, SectionType};
use grafeo_common::utils::error::{Error, Result};

use crate::catalog::{
    Catalog, ConstraintDefinition, EdgeTypeDefinition, GraphTypeDefinition, NodeTypeDefinition,
    ProcedureDefinition,
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

#[derive(Serialize, Deserialize, Default)]
struct SnapshotIndexes {
    property_indexes: Vec<String>,
    vector_indexes: Vec<SnapshotVectorIndex>,
    text_indexes: Vec<SnapshotTextIndex>,
}

#[derive(Serialize, Deserialize)]
struct SnapshotVectorIndex {
    label: String,
    property: String,
    dimensions: usize,
    metric: grafeo_core::index::vector::DistanceMetric,
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
    store: Arc<grafeo_core::graph::lpg::LpgStore>,
    epoch_fn: Box<dyn Fn() -> u64 + Send + Sync>,
    dirty: AtomicBool,
}

impl CatalogSection {
    /// Create a new catalog section.
    ///
    /// The `epoch_fn` closure returns the current MVCC epoch. This avoids a
    /// dependency on `TransactionManager` which lives in the engine layer.
    pub fn new(
        catalog: Arc<Catalog>,
        store: Arc<grafeo_core::graph::lpg::LpgStore>,
        epoch_fn: impl Fn() -> u64 + Send + Sync + 'static,
    ) -> Self {
        Self {
            catalog,
            store,
            epoch_fn: Box::new(epoch_fn),
            dirty: AtomicBool::new(false),
        }
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

    fn collect_indexes(&self) -> SnapshotIndexes {
        let property_indexes = self.store.property_index_keys();

        #[cfg(feature = "vector-index")]
        let vector_indexes: Vec<SnapshotVectorIndex> = self
            .store
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
        let text_indexes: Vec<SnapshotTextIndex> = self
            .store
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
        let snapshot = CatalogSnapshot {
            version: CATALOG_SECTION_VERSION,
            schema,
            indexes: self.collect_indexes(),
            epoch: (self.epoch_fn)(),
        };

        let config = bincode::config::standard();
        let mut bytes = bincode::serde::encode_to_vec(&snapshot, config)
            .map_err(|e| Error::Internal(format!("Catalog section serialization failed: {e}")))?;
        if !constraints.is_empty() {
            let names = bincode::serde::encode_to_vec(ConstraintNames { constraints }, config)
                .map_err(|e| {
                    Error::Internal(format!("Constraint name serialization failed: {e}"))
                })?;
            bytes.extend_from_slice(&names);
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
        if read < data.len() {
            let (names, _): (ConstraintNames, _) =
                bincode::serde::decode_from_slice(&data[read..], config).map_err(|e| {
                    Error::Serialization(format!("Constraint names deserialization failed: {e}"))
                })?;
            self.catalog.restore_constraint_names(names.constraints);
        }

        // Index metadata is stored for reference. Actual index rebuilding
        // happens in the engine after all data sections are loaded.
        // The engine reads the catalog's index defs and calls create_*_index.

        Ok(())
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
        assert!(!bytes.is_empty());

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
                .is_empty()
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
}
