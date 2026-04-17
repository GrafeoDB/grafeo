//! Index management for GrafeoDB (property, vector, and text indexes).

use grafeo_common::grafeo_info;
#[cfg(any(feature = "vector-index", feature = "text-index"))]
use std::sync::Arc;

#[cfg(feature = "text-index")]
use parking_lot::RwLock;

use grafeo_common::utils::error::Result;

impl super::GrafeoDB {
    // =========================================================================
    // PROPERTY INDEX API
    // =========================================================================

    /// Creates an index on a node property for O(1) lookups by value.
    ///
    /// After creating an index, calls to [`Self::find_nodes_by_property`] will be
    /// O(1) instead of O(n) for this property. The index is automatically
    /// maintained when properties are set or removed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use grafeo_engine::GrafeoDB;
    /// # use grafeo_common::types::Value;
    /// # let db = GrafeoDB::new_in_memory();
    /// // Create an index on the 'email' property
    /// db.create_property_index("email");
    ///
    /// // Now lookups by email are O(1)
    /// let nodes = db.find_nodes_by_property("email", &Value::from("alix@example.com"));
    /// ```
    pub fn create_property_index(&self, property: &str) {
        self.lpg_store().create_property_index(property);
    }

    /// Drops an index on a node property.
    ///
    /// Returns `true` if the index existed and was removed.
    pub fn drop_property_index(&self, property: &str) -> bool {
        self.lpg_store().drop_property_index(property)
    }

    /// Returns `true` if the property has an index.
    #[must_use]
    pub fn has_property_index(&self, property: &str) -> bool {
        self.lpg_store().has_property_index(property)
    }

    /// Finds all nodes that have a specific property value.
    ///
    /// If the property is indexed, this is O(1). Otherwise, it scans all nodes
    /// which is O(n). Use [`Self::create_property_index`] for frequently queried properties.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use grafeo_engine::GrafeoDB;
    /// # use grafeo_common::types::Value;
    /// # let db = GrafeoDB::new_in_memory();
    /// // Create index for fast lookups (optional but recommended)
    /// db.create_property_index("city");
    ///
    /// // Find all nodes where city = "NYC"
    /// let nyc_nodes = db.find_nodes_by_property("city", &Value::from("NYC"));
    /// ```
    #[must_use]
    pub fn find_nodes_by_property(
        &self,
        property: &str,
        value: &grafeo_common::types::Value,
    ) -> Vec<grafeo_common::types::NodeId> {
        self.lpg_store().find_nodes_by_property(property, value)
    }

    // =========================================================================
    // VECTOR INDEX API
    // =========================================================================

    /// Creates a vector similarity index on a node property.
    ///
    /// This enables efficient approximate nearest-neighbor search on vector
    /// properties. Currently validates the index parameters and scans existing
    /// nodes to verify the property contains vectors of the expected dimensions.
    ///
    /// # Arguments
    ///
    /// * `label` - Node label to index (e.g., `"Doc"`)
    /// * `property` - Property containing vector embeddings (e.g., `"embedding"`)
    /// * `dimensions` - Expected vector dimensions (inferred from data if `None`)
    /// * `metric` - Distance metric: `"cosine"` (default), `"euclidean"`, `"dot_product"`, `"manhattan"`
    /// * `m` - HNSW links per node (default: 16). Higher = better recall, more memory.
    /// * `ef_construction` - Construction beam width (default: 128). Higher = better index quality, slower build.
    /// * `quantization` - Quantization mode: `None` (default), `"scalar"`, `"binary"`, or `"product"`.
    ///   Quantized indexes use less memory at the cost of slightly lower recall.
    ///
    /// # Errors
    ///
    /// Returns an error if the metric is invalid, no vectors are found, or
    /// dimensions don't match.
    #[allow(clippy::too_many_arguments)]
    pub fn create_vector_index(
        &self,
        label: &str,
        property: &str,
        dimensions: Option<usize>,
        metric: Option<&str>,
        m: Option<usize>,
        ef_construction: Option<usize>,
        quantization: Option<&str>,
    ) -> Result<()> {
        use grafeo_common::types::{PropertyKey, Value};
        use grafeo_core::index::vector::DistanceMetric;

        let metric = match metric {
            Some(m) => DistanceMetric::from_str(m).ok_or_else(|| {
                grafeo_common::utils::error::Error::Internal(format!(
                    "Unknown distance metric '{}'. Use: cosine, euclidean, dot_product, manhattan",
                    m
                ))
            })?,
            None => DistanceMetric::Cosine,
        };

        #[cfg(feature = "vector-index")]
        let quantization_type = Self::parse_quantization(quantization)?;
        #[cfg(not(feature = "vector-index"))]
        let _ = quantization;

        // Scan nodes to validate vectors exist and check dimensions
        let prop_key = PropertyKey::new(property);
        let mut found_dims: Option<usize> = dimensions;
        let mut vector_count = 0usize;

        #[cfg(feature = "vector-index")]
        let mut vectors: Vec<(grafeo_common::types::NodeId, Vec<f32>)> = Vec::new();

        let graph = self.graph_store();
        for node_id in graph.nodes_by_label(label) {
            if let Some(Value::Vector(v)) = graph.get_node_property(node_id, &prop_key) {
                if let Some(expected) = found_dims {
                    if v.len() != expected {
                        return Err(grafeo_common::utils::error::Error::Internal(format!(
                            "Vector dimension mismatch: expected {}, found {} on node {}",
                            expected,
                            v.len(),
                            node_id.0
                        )));
                    }
                } else {
                    found_dims = Some(v.len());
                }
                vector_count += 1;
                #[cfg(feature = "vector-index")]
                vectors.push((node_id, v.to_vec()));
            }
        }

        let Some(dims) = found_dims else {
            // No vectors found yet: caller must have supplied explicit dimensions
            // so we can create an empty index that auto-populates via set_node_property.
            return if let Some(d) = dimensions {
                #[cfg(feature = "vector-index")]
                {
                    let index = Self::build_vector_index(
                        d,
                        metric,
                        m,
                        ef_construction,
                        quantization_type,
                        0,
                    );
                    self.lpg_store()
                        .add_vector_index(label, property, Arc::new(index));
                }

                let _ = (m, ef_construction);
                grafeo_info!(
                    "Empty vector index created: :{label}({property}) - 0 vectors, {d} dimensions, metric={metric_name}",
                    metric_name = metric.name()
                );
                Ok(())
            } else {
                Err(grafeo_common::utils::error::Error::Internal(format!(
                    "No vector properties found on :{label}({property}) and no dimensions specified"
                )))
            };
        };

        // Build and populate the vector index
        #[cfg(feature = "vector-index")]
        {
            use grafeo_core::index::vector::VectorIndexKind;

            let index = Self::build_vector_index(
                dims,
                metric,
                m,
                ef_construction,
                quantization_type,
                vectors.len(),
            );

            match &index {
                VectorIndexKind::Hnsw(_) => {
                    let graph = self.graph_store();
                    let accessor = grafeo_core::index::vector::PropertyVectorAccessor::new(
                        &*graph,
                        property,
                    );
                    for (node_id, vec) in &vectors {
                        index.insert(*node_id, vec, &accessor);
                    }
                }
                VectorIndexKind::Quantized(q_idx) => {
                    for (node_id, vec) in &vectors {
                        q_idx.insert(*node_id, vec);
                    }
                }
            }

            self.lpg_store()
                .add_vector_index(label, property, Arc::new(index));
        }

        // Suppress unused variable warnings when vector-index is off
        let _ = (m, ef_construction);

        grafeo_info!(
            "Vector index created: :{label}({property}) - {vector_count} vectors, {dims} dimensions, metric={metric_name}",
            metric_name = metric.name()
        );

        Ok(())
    }

    /// Parses a quantization string into a [`QuantizationType`].
    #[cfg(feature = "vector-index")]
    fn parse_quantization(
        quantization: Option<&str>,
    ) -> Result<grafeo_core::index::vector::QuantizationType> {
        use grafeo_core::index::vector::QuantizationType;
        match quantization {
            None | Some("none") => Ok(QuantizationType::None),
            Some("scalar") => Ok(QuantizationType::Scalar),
            Some("binary") => Ok(QuantizationType::Binary),
            Some("product") => Ok(QuantizationType::Product { num_subvectors: 8 }),
            Some(other) => Err(grafeo_common::utils::error::Error::Internal(format!(
                "Unknown quantization type '{other}'. Use: scalar, binary, product"
            ))),
        }
    }

    /// Builds a [`VectorIndexKind`] from the given parameters.
    #[cfg(feature = "vector-index")]
    fn build_vector_index(
        dims: usize,
        metric: grafeo_core::index::vector::DistanceMetric,
        m: Option<usize>,
        ef_construction: Option<usize>,
        quantization: grafeo_core::index::vector::QuantizationType,
        capacity: usize,
    ) -> grafeo_core::index::vector::VectorIndexKind {
        use grafeo_core::index::vector::{
            HnswConfig, HnswIndex, QuantizationType, QuantizedHnswIndex, VectorIndexKind,
        };

        let mut config = HnswConfig::new(dims, metric);
        if let Some(m_val) = m {
            config = config.with_m(m_val);
        }
        if let Some(ef_c) = ef_construction {
            config = config.with_ef_construction(ef_c);
        }

        match quantization {
            QuantizationType::None => {
                VectorIndexKind::Hnsw(HnswIndex::with_capacity(config, capacity))
            }
            _ => VectorIndexKind::Quantized(QuantizedHnswIndex::new(config, quantization)),
        }
    }

    /// Drops a vector index for the given label and property.
    ///
    /// Returns `true` if the index existed and was removed, `false` if no
    /// index was found.
    ///
    /// After dropping, [`vector_search`](Self::vector_search) for this
    /// label+property pair will return an error.
    #[cfg(feature = "vector-index")]
    pub fn drop_vector_index(&self, label: &str, property: &str) -> bool {
        let removed = self.lpg_store().remove_vector_index(label, property);
        if removed {
            grafeo_info!("Vector index dropped: :{label}({property})");
        }
        removed
    }

    /// Drops and recreates a vector index, rescanning all matching nodes.
    ///
    /// In normal usage you do **not** need to call this. Vector indexes
    /// auto-sync when nodes are created or updated via
    /// [`set_node_property`](Self::set_node_property),
    /// [`batch_create_nodes`](Self::batch_create_nodes), or
    /// [`batch_create_nodes_with_props`](Self::batch_create_nodes_with_props).
    ///
    /// Use `rebuild_vector_index` only when:
    /// - Data was loaded through non-standard paths (e.g., persistence
    ///   restore or direct store manipulation) before the index existed.
    /// - You want to compact the index after many deletions (HNSW does
    ///   not reclaim deleted-node slots automatically).
    /// - The index configuration needs to be refreshed after upgrading.
    ///
    /// When the index still exists, the previous configuration (dimensions,
    /// metric, M, ef\_construction) is preserved. When it has already been
    /// dropped, dimensions are inferred from existing data and default
    /// parameters are used.
    ///
    /// # Errors
    ///
    /// Returns an error if the rebuild fails (e.g., no matching vectors found
    /// and no dimensions can be inferred).
    #[cfg(feature = "vector-index")]
    pub fn rebuild_vector_index(&self, label: &str, property: &str) -> Result<()> {
        // Preserve config and quantization type from existing index if available
        let existing = self.lpg_store().get_vector_index(label, property);

        let (config, quantization_name) = if let Some(ref idx) = existing {
            let qt = match idx.quantization_type() {
                Some(grafeo_core::index::vector::QuantizationType::Scalar) => Some("scalar"),
                Some(grafeo_core::index::vector::QuantizationType::Binary) => Some("binary"),
                Some(grafeo_core::index::vector::QuantizationType::Product { .. }) => {
                    Some("product")
                }
                _ => None,
            };
            (Some(idx.config().clone()), qt)
        } else {
            (None, None)
        };

        self.lpg_store().remove_vector_index(label, property);

        if let Some(config) = config {
            self.create_vector_index(
                label,
                property,
                Some(config.dimensions),
                Some(config.metric.name()),
                Some(config.m),
                Some(config.ef_construction),
                quantization_name,
            )
        } else {
            // Index was already dropped: infer dimensions from data
            self.create_vector_index(label, property, None, None, None, None, None)
        }
    }

    // =========================================================================
    // TEXT INDEX API
    // =========================================================================

    /// Creates a BM25 text index on a node property for full-text search.
    ///
    /// Indexes all existing nodes with the given label and property.
    /// The index stays in sync automatically as nodes are created, updated,
    /// or deleted. Use [`rebuild_text_index`](Self::rebuild_text_index) only
    /// if the index was created before existing data was loaded.
    ///
    /// # Errors
    ///
    /// Returns an error if the label has no nodes or the property contains no text values.
    #[cfg(feature = "text-index")]
    pub fn create_text_index(&self, label: &str, property: &str) -> Result<()> {
        use grafeo_common::types::{PropertyKey, Value};
        use grafeo_core::index::text::{BM25Config, InvertedIndex};

        let mut index = InvertedIndex::new(BM25Config::default());
        let prop_key = PropertyKey::new(property);

        // Index all existing nodes with this label + property
        let graph = self.graph_store();
        let nodes = graph.nodes_by_label(label);
        for node_id in nodes {
            if let Some(Value::String(text)) =
                graph.get_node_property(node_id, &prop_key)
            {
                index.insert(node_id, text.as_str());
            }
        }

        self.lpg_store()
            .add_text_index(label, property, Arc::new(RwLock::new(index)));
        Ok(())
    }

    /// Drops a text index on a label+property pair.
    ///
    /// Returns `true` if the index existed and was removed.
    #[cfg(feature = "text-index")]
    pub fn drop_text_index(&self, label: &str, property: &str) -> bool {
        self.lpg_store().remove_text_index(label, property)
    }

    /// Rebuilds a text index by re-scanning all matching nodes.
    ///
    /// Use after bulk property updates to keep the index current.
    ///
    /// # Errors
    ///
    /// Returns an error if no text index exists for this label+property.
    #[cfg(feature = "text-index")]
    pub fn rebuild_text_index(&self, label: &str, property: &str) -> Result<()> {
        self.lpg_store().remove_text_index(label, property);
        self.create_text_index(label, property)
    }
}
