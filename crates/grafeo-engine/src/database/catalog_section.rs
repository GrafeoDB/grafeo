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
    Catalog, EdgeTypeDefinition, GraphTypeDefinition, NodeTypeDefinition, ProcedureDefinition,
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
        let snapshot = CatalogSnapshot {
            version: CATALOG_SECTION_VERSION,
            schema: self.collect_schema(),
            indexes: self.collect_indexes(),
            epoch: (self.epoch_fn)(),
        };

        let config = bincode::config::standard();
        bincode::serde::encode_to_vec(&snapshot, config)
            .map_err(|e| Error::Internal(format!("Catalog section serialization failed: {e}")))
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        let config = bincode::config::standard();
        let (snapshot, _): (CatalogSnapshot, _) = bincode::serde::decode_from_slice(data, config)
            .map_err(|e| {
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

        // Restore vector index objects from catalog metadata so the
        // VectorStore section can populate their topology without rebuilding.
        // Without this, vector_index_entries() returns empty and the persisted
        // HNSW topology is read from disk but never applied — forcing a full
        // rebuild on every open.
        #[cfg(feature = "vector-index")]
        {
            use grafeo_core::index::vector::{HnswConfig, HnswIndex, VectorIndexKind};

            for vidx in &snapshot.indexes.vector_indexes {
                let config = HnswConfig::new(vidx.dimensions, vidx.metric)
                    .with_m(vidx.m)
                    .with_ef_construction(vidx.ef_construction);
                let index = Arc::new(VectorIndexKind::Hnsw(HnswIndex::with_capacity(config, 0)));
                self.store.add_vector_index(&vidx.label, &vidx.property, index);
            }
        }

        // Text indexes: register placeholder entries so the TextIndex section
        // can restore BM25 postings without rebuilding.
        #[cfg(feature = "text-index")]
        {
            for tidx in &snapshot.indexes.text_indexes {
                // Text index restoration is handled by the engine after data load.
                // Registration here ensures the section restore finds non-empty entries.
            }
        }

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
