//! Catalog section serializer for the `.grafeo` container format.
//!
//! Serializes schema definitions (node types, edge types, graph types, procedures),
//! index metadata (property, vector, text), and epoch state into the `CATALOG` section.
//!
//! # Vector index quantization (v2)
//!
//! Catalog version 2 stores per-vector-index [`QuantizationType`] so reopen can
//! re-materialize the correct `VectorIndexKind` shell before VectorStore topology
//! is applied. Version 1 snapshots (no quant field) dual-read as plain HNSW.

// Parts of this module are reserved for Phase 5 checkpoint integration.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};

use grafeo_common::storage::section::{Section, SectionType};
use grafeo_common::utils::error::{Error, Result};

use crate::catalog::{
    Catalog, EdgeTypeDefinition, GraphTypeDefinition, NodeTypeDefinition, ProcedureDefinition,
};

/// Current catalog section format version (includes vector quant metadata).
const CATALOG_SECTION_VERSION: u8 = 2;

/// Legacy catalog section format version (no vector quant field).
const CATALOG_SECTION_VERSION_V1: u8 = 1;

// ── Snapshot types (v2 = current writer) ────────────────────────────

#[derive(Serialize, Deserialize)]
struct CatalogSnapshot {
    version: u8,
    schema: SnapshotSchema,
    indexes: SnapshotIndexes,
    epoch: u64,
}

/// Legacy v1 catalog blob (pre-quant metadata).
#[derive(Serialize, Deserialize)]
struct CatalogSnapshotV1 {
    version: u8,
    schema: SnapshotSchema,
    indexes: SnapshotIndexesV1,
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

#[derive(Serialize, Deserialize, Default)]
struct SnapshotIndexesV1 {
    property_indexes: Vec<String>,
    vector_indexes: Vec<SnapshotVectorIndexV1>,
    text_indexes: Vec<SnapshotTextIndex>,
}

/// Vector index shell metadata including durable quantization mode (v2).
#[derive(Serialize, Deserialize)]
struct SnapshotVectorIndex {
    label: String,
    property: String,
    dimensions: usize,
    metric: grafeo_core::index::vector::DistanceMetric,
    m: usize,
    ef_construction: usize,
    /// Durable quantization mode. `None` means plain HNSW.
    quantization: grafeo_core::index::vector::QuantizationType,
}

/// Pre-v2 vector index shell metadata (no quantization field).
#[derive(Serialize, Deserialize)]
struct SnapshotVectorIndexV1 {
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
                let quantization = index
                    .quantization_type()
                    .unwrap_or(grafeo_core::index::vector::QuantizationType::None);
                Some(SnapshotVectorIndex {
                    label: label.to_string(),
                    property: property.to_string(),
                    dimensions: config.dimensions,
                    metric: config.metric,
                    m: config.m,
                    ef_construction: config.ef_construction,
                    quantization,
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

    fn restore_schema(&self, schema: &SnapshotSchema) {
        for def in &schema.node_types {
            self.catalog.register_or_replace_node_type(def.clone());
        }
        for def in &schema.edge_types {
            self.catalog.register_or_replace_edge_type_def(def.clone());
        }
        for def in &schema.graph_types {
            let _ = self.catalog.register_graph_type(def.clone());
        }
        for def in &schema.procedures {
            self.catalog.replace_procedure(def.clone()).ok();
        }
        for name in &schema.schemas {
            let _ = self.catalog.register_schema_namespace(name.clone());
            let default_key = format!("{name}/__default__");
            let _ = self.store.create_graph(&default_key);
        }
        for (graph_name, type_name) in &schema.graph_type_bindings {
            let _ = self.catalog.bind_graph_type(graph_name, type_name.clone());
        }
    }

    /// Restore vector index shells so VectorStore can apply topology without rebuild.
    #[cfg(feature = "vector-index")]
    fn restore_vector_index_shells(
        &self,
        indexes: &[(
            &str,
            &str,
            usize,
            grafeo_core::index::vector::DistanceMetric,
            usize,
            usize,
            grafeo_core::index::vector::QuantizationType,
        )],
    ) {
        use grafeo_core::index::vector::{
            HnswConfig, HnswIndex, QuantizationType, QuantizedHnswIndex, VectorIndexKind,
        };

        for (label, property, dimensions, metric, m, ef_construction, quantization) in indexes {
            let config = HnswConfig::new(*dimensions, *metric)
                .with_m(*m)
                .with_ef_construction(*ef_construction);
            let index = match quantization {
                QuantizationType::None => {
                    Arc::new(VectorIndexKind::Hnsw(HnswIndex::with_capacity(config, 0)))
                }
                quant => Arc::new(VectorIndexKind::Quantized(QuantizedHnswIndex::new(
                    config, *quant,
                ))),
            };
            self.store.add_vector_index(label, property, index);
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

        // Dispatch on the encoded version before decoding the version-specific
        // shape. A malformed/unknown v2 record must never be reinterpreted as v1.
        let (version, _): (u8, _) =
            bincode::serde::decode_from_slice(data, config).map_err(|e| {
                Error::Serialization(format!("Catalog section version decode failed: {e}"))
            })?;

        if version == CATALOG_SECTION_VERSION {
            let (snapshot, _): (CatalogSnapshot, _) =
                bincode::serde::decode_from_slice(data, config).map_err(|e| {
                    Error::Serialization(format!(
                        "Catalog section v{CATALOG_SECTION_VERSION} deserialization failed: {e}"
                    ))
                })?;
            self.restore_schema(&snapshot.schema);

            #[cfg(feature = "vector-index")]
            {
                let shells: Vec<_> = snapshot
                    .indexes
                    .vector_indexes
                    .iter()
                    .map(|v| {
                        (
                            v.label.as_str(),
                            v.property.as_str(),
                            v.dimensions,
                            v.metric,
                            v.m,
                            v.ef_construction,
                            v.quantization,
                        )
                    })
                    .collect();
                self.restore_vector_index_shells(&shells);
            }

            // Text indexes: registration is handled by the engine after data load.
            let _ = &snapshot.indexes.text_indexes;

            return Ok(());
        }

        if version != CATALOG_SECTION_VERSION_V1 {
            return Err(Error::Serialization(format!(
                "Unsupported catalog section version {version}; expected {CATALOG_SECTION_VERSION_V1} or {CATALOG_SECTION_VERSION}"
            )));
        }

        let (snapshot, _): (CatalogSnapshotV1, _) = bincode::serde::decode_from_slice(data, config)
            .map_err(|e| {
                Error::Serialization(format!("Catalog section deserialization failed: {e}"))
            })?;

        // Accept any successfully-decoded v1-shaped blob (including accidental
        // decode of empty-index catalogs that share layout with a version=1 header).
        let _ = snapshot.version.max(CATALOG_SECTION_VERSION_V1);

        self.restore_schema(&snapshot.schema);

        #[cfg(feature = "vector-index")]
        {
            use grafeo_core::index::vector::QuantizationType;
            let shells: Vec<_> = snapshot
                .indexes
                .vector_indexes
                .iter()
                .map(|v| {
                    (
                        v.label.as_str(),
                        v.property.as_str(),
                        v.dimensions,
                        v.metric,
                        v.m,
                        v.ef_construction,
                        QuantizationType::None,
                    )
                })
                .collect();
            self.restore_vector_index_shells(&shells);
        }

        let _ = &snapshot.indexes.text_indexes;

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
        assert_eq!(section.version(), 2);
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

    #[cfg(feature = "vector-index")]
    #[test]
    fn v2_decode_failure_never_falls_back_to_v1_shape() {
        use grafeo_core::index::vector::DistanceMetric;

        // A v1-shaped payload mislabeled as v2 used to fail v2 decode and then
        // succeed through the unconditional v1 fallback.
        let mislabeled_v1 = CatalogSnapshotV1 {
            version: CATALOG_SECTION_VERSION,
            schema: SnapshotSchema::default(),
            indexes: SnapshotIndexesV1 {
                property_indexes: vec![],
                vector_indexes: vec![SnapshotVectorIndexV1 {
                    label: "Doc".to_string(),
                    property: "emb".to_string(),
                    dimensions: 8,
                    metric: DistanceMetric::Cosine,
                    m: 16,
                    ef_construction: 100,
                }],
                text_indexes: vec![],
            },
            epoch: 1,
        };
        let bytes =
            bincode::serde::encode_to_vec(&mislabeled_v1, bincode::config::standard()).unwrap();

        let mut section = make_section();
        assert!(
            section.deserialize(&bytes).is_err(),
            "v2-tagged malformed payload must fail closed without v1 reinterpretation"
        );
    }

    /// Encode a golden v1 catalog blob (no quant field) and ensure dual-read
    /// opens with plain HNSW shells (quant = None).
    #[cfg(feature = "vector-index")]
    #[test]
    fn catalog_v1_vector_index_opens_as_plain_hnsw() {
        use grafeo_core::index::vector::{
            DistanceMetric, HnswConfig, HnswIndex, QuantizationType, VectorIndexKind,
        };

        let v1 = CatalogSnapshotV1 {
            version: CATALOG_SECTION_VERSION_V1,
            schema: SnapshotSchema::default(),
            indexes: SnapshotIndexesV1 {
                property_indexes: vec![],
                vector_indexes: vec![SnapshotVectorIndexV1 {
                    label: "Doc".to_string(),
                    property: "emb".to_string(),
                    dimensions: 8,
                    metric: DistanceMetric::Cosine,
                    m: 16,
                    ef_construction: 100,
                }],
                text_indexes: vec![],
            },
            epoch: 1,
        };
        let config = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(&v1, config).unwrap();

        let catalog = Arc::new(Catalog::new());
        let store = Arc::new(grafeo_core::graph::lpg::LpgStore::new().unwrap());
        let mut section = CatalogSection::new(catalog, Arc::clone(&store), || 0);
        section.deserialize(&bytes).expect("v1 dual-read");

        let idx = store
            .get_vector_index("Doc", "emb")
            .expect("shell restored");
        assert!(matches!(*idx, VectorIndexKind::Hnsw(_)));
        assert_eq!(
            idx.quantization_type().unwrap_or(QuantizationType::None),
            QuantizationType::None
        );
        assert_eq!(idx.config().dimensions, 8);

        // Sanity: same shell shape as today's plain builder
        let _plain = VectorIndexKind::Hnsw(HnswIndex::with_capacity(
            HnswConfig::new(8, DistanceMetric::Cosine),
            0,
        ));
    }

    #[cfg(feature = "vector-index")]
    #[test]
    fn catalog_v2_vector_index_quant_roundtrip() {
        use grafeo_core::index::vector::{
            DistanceMetric, HnswConfig, QuantizationType, QuantizedHnswIndex, VectorIndexKind,
        };

        let modes = [
            QuantizationType::None,
            QuantizationType::Scalar,
            QuantizationType::Binary,
            QuantizationType::Product { num_subvectors: 4 },
        ];

        for (i, mode) in modes.iter().enumerate() {
            let label = format!("Doc{i}");
            let catalog = Arc::new(Catalog::new());
            let store = Arc::new(grafeo_core::graph::lpg::LpgStore::new().unwrap());
            let config = HnswConfig::new(8, DistanceMetric::Cosine)
                .with_m(16)
                .with_ef_construction(100);
            let index = match mode {
                QuantizationType::None => Arc::new(VectorIndexKind::Hnsw(
                    grafeo_core::index::vector::HnswIndex::with_capacity(config, 0),
                )),
                quant => Arc::new(VectorIndexKind::Quantized(QuantizedHnswIndex::new(
                    config, *quant,
                ))),
            };
            store.add_vector_index(&label, "emb", index);

            let section = CatalogSection::new(catalog, store, || 7);
            let bytes = section.serialize().expect("serialize v2 catalog");

            // Writer always emits version 2.
            let (decoded, _): (CatalogSnapshot, _) =
                bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
            assert_eq!(decoded.version, 2);
            assert_eq!(decoded.indexes.vector_indexes.len(), 1);
            assert_eq!(decoded.indexes.vector_indexes[0].quantization, *mode);

            let catalog2 = Arc::new(Catalog::new());
            let store2 = Arc::new(grafeo_core::graph::lpg::LpgStore::new().unwrap());
            let mut section2 = CatalogSection::new(catalog2, Arc::clone(&store2), || 0);
            section2.deserialize(&bytes).expect("deserialize v2");

            let restored = store2
                .get_vector_index(&label, "emb")
                .expect("restored shell");
            let restored_mode = restored
                .quantization_type()
                .unwrap_or(QuantizationType::None);
            assert_eq!(restored_mode, *mode, "mode mismatch for {mode:?}");
            match mode {
                QuantizationType::None => assert!(matches!(*restored, VectorIndexKind::Hnsw(_))),
                _ => assert!(matches!(*restored, VectorIndexKind::Quantized(_))),
            }
        }
    }

    #[cfg(feature = "vector-index")]
    #[test]
    fn catalog_serialize_captures_quant_from_live_index() {
        use grafeo_core::index::vector::{
            DistanceMetric, HnswConfig, QuantizationType, QuantizedHnswIndex, VectorIndexKind,
        };

        let catalog = Arc::new(Catalog::new());
        let store = Arc::new(grafeo_core::graph::lpg::LpgStore::new().unwrap());
        let config = HnswConfig::new(16, DistanceMetric::Euclidean);
        store.add_vector_index(
            "Mem",
            "embedding",
            Arc::new(VectorIndexKind::Quantized(QuantizedHnswIndex::new(
                config,
                QuantizationType::Scalar,
            ))),
        );
        let section = CatalogSection::new(catalog, store, || 0);
        let bytes = section.serialize().unwrap();
        let (snap, _): (CatalogSnapshot, _) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(
            snap.indexes.vector_indexes[0].quantization,
            QuantizationType::Scalar
        );
    }
}
