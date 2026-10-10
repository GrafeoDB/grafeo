//! Index management for GrafeoDB (property, vector, and text indexes).
//!
//! Creating or dropping an index is a standalone change (see
//! [`standalone`](super::standalone)): the index is built from the data
//! first, without holding commits off, then its catalog record is logged and
//! the index installed while commits are held off. Replay applies the
//! record, and builds the index from the replayed data.

#[cfg(feature = "vector-index")]
use grafeo_common::grafeo_info;
#[cfg(any(feature = "vector-index", feature = "text-index"))]
use std::sync::Arc;

use grafeo_common::change::StandaloneOp;
use grafeo_common::storage::catalog_record::{
    CatalogKey, CatalogRecord, IndexKeyRecord, IndexKindRecord, IndexRecord,
};
use grafeo_common::utils::error::{Error, Result};
#[cfg(any(feature = "vector-index", feature = "text-index"))]
use grafeo_core::graph::GraphStoreSearch;
#[cfg(any(feature = "vector-index", feature = "text-index"))]
use grafeo_core::graph::lpg::LpgStore;

#[cfg(any(feature = "vector-index", feature = "text-index"))]
use crate::transaction::BuiltIndex;
use crate::transaction::StandaloneChange;

/// The distance metric `name` names for a new vector index (cosine when
/// `None`).
///
/// # Errors
///
/// An invalid-value error naming the metrics there are, for a name that is
/// none of them.
#[cfg(feature = "vector-index")]
pub(crate) fn vector_index_metric(
    name: Option<&str>,
) -> Result<grafeo_core::index::vector::DistanceMetric> {
    use grafeo_core::index::vector::DistanceMetric;
    match name {
        Some(name) => DistanceMetric::from_str(name).ok_or_else(|| {
            Error::InvalidValue(format!(
                "Unknown distance metric '{name}'. Use: cosine, euclidean, dot_product, manhattan"
            ))
        }),
        None => Ok(DistanceMetric::Cosine),
    }
}

/// Checks the dimensions given for a new vector index.
///
/// # Errors
///
/// An invalid-value error for 0: a vector index measures vectors with at
/// least one value.
#[cfg(feature = "vector-index")]
pub(crate) fn check_vector_index_dimensions(dimensions: Option<usize>) -> Result<()> {
    if dimensions == Some(0) {
        return Err(Error::InvalidValue(
            "a vector index needs at least 1 dimension".to_string(),
        ));
    }
    Ok(())
}

/// Checks a vector a new vector index of `dimensions` would hold, the vector
/// of `node`: the index can measure it only when it has that many values
/// (at least one), none of them NaN or infinite.
///
/// # Errors
///
/// An invalid-value error naming the node and what is wrong with its vector.
#[cfg(feature = "vector-index")]
pub(crate) fn check_vector_for_index(
    node: grafeo_common::types::NodeId,
    vector: &[f32],
    dimensions: usize,
) -> Result<()> {
    check_vector_index_dimensions(Some(dimensions))?;
    if vector.len() != dimensions {
        return Err(Error::InvalidValue(format!(
            "Vector dimension mismatch: expected {dimensions}, found {} on node {}",
            vector.len(),
            node.0
        )));
    }
    if let Some((position, value)) = grafeo_core::index::vector::first_non_finite(vector) {
        return Err(Error::InvalidValue(format!(
            "the vector of node {} has {value} at position {position}: a vector index cannot \
             measure it",
            node.0
        )));
    }
    Ok(())
}

/// The name [`create_vector_index`](super::GrafeoDB::create_vector_index)
/// takes for a quantization.
#[cfg(feature = "vector-index")]
pub(super) fn quantization_name(
    quantization: grafeo_core::index::vector::QuantizationType,
) -> Option<&'static str> {
    use grafeo_core::index::vector::QuantizationType;
    match quantization {
        QuantizationType::Scalar => Some("scalar"),
        QuantizationType::Binary => Some("binary"),
        QuantizationType::Product { .. } => Some("product"),
        _ => None,
    }
}

/// A vector index of `property` on the nodes of `graph` that have `label`,
/// built from their vectors, for `create_vector_index` and `CREATE VECTOR
/// INDEX`: `dimensions` from the first vector when `None`, `metric` cosine
/// when `None`, `m` and `ef_construction` the HNSW defaults when `None`,
/// `quantization` a name `create_vector_index` takes.
///
/// # Errors
///
/// An unknown metric or quantization, 0 dimensions, a vector of other
/// dimensions or with a NaN or infinite value, or no vector and no
/// dimensions.
#[cfg(feature = "vector-index")]
#[allow(
    clippy::too_many_arguments,
    reason = "the arguments of create_vector_index and the graph"
)]
pub(crate) fn vector_index_from_data(
    graph: &dyn GraphStoreSearch,
    label: &str,
    property: &str,
    dimensions: Option<usize>,
    metric: Option<&str>,
    m: Option<usize>,
    ef_construction: Option<usize>,
    quantization: Option<&str>,
) -> Result<grafeo_core::index::vector::VectorIndexKind> {
    use grafeo_common::types::{PropertyKey, Value};
    use grafeo_core::index::vector::VectorIndexKind;

    let metric = vector_index_metric(metric)?;
    check_vector_index_dimensions(dimensions)?;
    let quantization = super::GrafeoDB::parse_quantization(quantization)?;

    let key = PropertyKey::new(property);
    let mut found_dims = dimensions;
    let mut vectors: Vec<(grafeo_common::types::NodeId, Vec<f32>)> = Vec::new();
    for node_id in graph.nodes_by_label(label) {
        if let Some(Value::Vector(v)) = graph.get_node_property(node_id, &key) {
            let expected = *found_dims.get_or_insert(v.len());
            check_vector_for_index(node_id, &v, expected)?;
            vectors.push((node_id, v.to_vec()));
        }
    }
    let Some(dims) = found_dims else {
        return Err(Error::InvalidValue(format!(
            "No vector properties found on :{label}({property}) and no dimensions specified"
        )));
    };

    let index = super::GrafeoDB::build_vector_index(
        dims,
        metric,
        m,
        ef_construction,
        quantization,
        vectors.len(),
    );
    match &index {
        VectorIndexKind::Hnsw(_) => {
            let accessor = grafeo_core::index::vector::PropertyVectorAccessor::new(graph, property);
            for (node_id, vec) in &vectors {
                index.insert(*node_id, vec, &accessor);
            }
        }
        VectorIndexKind::Quantized(quantized) => {
            for (node_id, vec) in &vectors {
                quantized.insert(*node_id, vec);
            }
        }
    }
    grafeo_info!(
        "Vector index built: :{label}({property}) - {} vectors, {dims} dimensions, metric={}",
        vectors.len(),
        metric.name()
    );
    Ok(index)
}

/// A text index of `property` on the nodes of `graph` that have `label`,
/// built from their text values, for `create_text_index` and `CREATE INDEX
/// ... USING TEXT`.
#[cfg(feature = "text-index")]
pub(crate) fn text_index_from_data(
    graph: &dyn GraphStoreSearch,
    label: &str,
    property: &str,
) -> grafeo_core::index::text::InvertedIndex {
    use grafeo_common::types::{PropertyKey, Value};
    use grafeo_core::index::text::{BM25Config, InvertedIndex};

    let mut index = InvertedIndex::new(BM25Config::default());
    let key = PropertyKey::new(property);
    for node_id in graph.nodes_by_label(label) {
        if let Some(Value::String(text)) = graph.get_node_property(node_id, &key) {
            index.insert(node_id, text.as_str());
        }
    }
    index
}

/// The catalog record of `index`, a vector index of `property` on the nodes
/// with `label` in the graph with storage key `graph` (`None` for the
/// default graph), with every parameter it was built with.
///
/// # Errors
///
/// A parameter a catalog record cannot hold (see
/// [`index_records`](super::catalog_records::index_records)).
#[cfg(feature = "vector-index")]
pub(crate) fn vector_index_record(
    graph: Option<&str>,
    label: &str,
    property: &str,
    index: &grafeo_core::index::vector::VectorIndexKind,
) -> Result<IndexRecord> {
    let config = index.config();
    let definition = super::catalog_section::VectorIndexDefinition {
        label: label.to_string(),
        property: property.to_string(),
        dimensions: config.dimensions,
        metric: config.metric,
        m: config.m,
        ef_construction: config.ef_construction,
        quantization: index.quantization_type(),
    };
    let records = super::catalog_records::index_records(&[super::catalog_section::GraphIndexes {
        graph: graph.map(ToString::to_string),
        vector: vec![definition],
        ..super::catalog_section::GraphIndexes::default()
    }])?;
    records
        .into_iter()
        .next()
        .ok_or_else(|| Error::Internal("a vector index made no catalog record".to_string()))
}

/// The put of the index of `kind` in the graph with storage key `graph`.
pub(crate) fn put_index(graph: Option<&str>, kind: IndexKindRecord) -> StandaloneOp {
    StandaloneOp::PutCatalog(CatalogRecord::Index(IndexRecord {
        graph: graph.map(ToString::to_string),
        index: kind,
    }))
}

/// The drop of the index `key` names in the graph with storage key `graph`.
pub(crate) fn drop_index(graph: Option<&str>, key: IndexKeyRecord) -> StandaloneOp {
    StandaloneOp::DropCatalog(CatalogKey::Index {
        graph: graph.map(ToString::to_string),
        index: key,
    })
}

impl super::GrafeoDB {
    // =========================================================================
    // PROPERTY INDEX API
    // =========================================================================

    /// The storage key of the current graph, where the property index calls
    /// work: `None` for the default graph, also when the graph selected no
    /// longer exists (as [`current_lpg_store`](Self::current_lpg_store)).
    fn current_graph_key(&self) -> Option<String> {
        crate::session::graph_storage_key(
            self.current_schema.read().as_deref(),
            self.current_graph.read().as_deref(),
        )
        .filter(|key| self.lpg_store().graph(key).is_some())
    }

    /// Creates an index on a node property of the current graph, for O(1)
    /// lookups by value.
    ///
    /// After creating an index, calls to [`Self::find_nodes_by_property`] will be
    /// O(1) instead of O(n) for this property. The index is automatically
    /// maintained when properties are set or removed. Commits are held off
    /// while it is built, as for `CREATE INDEX`.
    ///
    /// # Errors
    ///
    /// Returns the database-closed error after `close()` of a persistent
    /// database, and the incomplete-commit error after a commit that did not
    /// complete.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use grafeo_engine::GrafeoDB;
    /// # use grafeo_common::types::Value;
    /// # let db = GrafeoDB::new_in_memory();
    /// // Create an index on the 'email' property
    /// db.create_property_index("email")?;
    ///
    /// // Now lookups by email are O(1)
    /// let nodes = db.find_nodes_by_property("email", &Value::from("alix@example.com"));
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    pub fn create_property_index(&self, property: &str) -> Result<()> {
        let held = self.hold_for_standalone(false)?;
        let graph = self.current_graph_key();
        if self.current_lpg_store().has_property_index(property) {
            return Ok(());
        }
        let mut change = StandaloneChange::new();
        change.push(put_index(
            graph.as_deref(),
            IndexKindRecord::Property {
                key: property.to_string(),
            },
        ));
        self.commit_standalone(change, &held)
    }

    /// Drops an index on a node property of the current graph.
    ///
    /// Returns `true` if the index existed and was removed.
    ///
    /// # Errors
    ///
    /// As [`create_property_index`](Self::create_property_index).
    pub fn drop_property_index(&self, property: &str) -> Result<bool> {
        let held = self.hold_for_standalone(false)?;
        let graph = self.current_graph_key();
        if !self.current_lpg_store().has_property_index(property) {
            return Ok(false);
        }
        let mut change = StandaloneChange::new();
        change.push(drop_index(
            graph.as_deref(),
            IndexKeyRecord::Property {
                key: property.to_string(),
            },
        ));
        self.commit_standalone(change, &held)?;
        Ok(true)
    }

    /// Fails before an index is built when it could not be installed: after
    /// `close()` of a persistent database, or after a commit that did not
    /// complete. The install checks again, holding commits off.
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    fn check_index_change(&self) -> Result<()> {
        self.transaction_manager.check_no_incomplete_commit()?;
        self.transaction_manager.check_open()
    }

    /// Returns `true` if the property has an index in the current graph.
    #[must_use]
    pub fn has_property_index(&self, property: &str) -> bool {
        self.current_lpg_store().has_property_index(property)
    }

    /// Finds all nodes of the current graph that have a specific property value.
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
    /// db.create_property_index("city")?;
    ///
    /// // Find all nodes where city = "Amsterdam"
    /// let in_amsterdam = db.find_nodes_by_property("city", &Value::from("Amsterdam"));
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    #[must_use]
    pub fn find_nodes_by_property(
        &self,
        property: &str,
        value: &grafeo_common::types::Value,
    ) -> Vec<grafeo_common::types::NodeId> {
        // The index also holds nodes created by transactions that have not
        // committed yet; return only what a reader at the current epoch sees.
        let epoch = self.read_epoch();
        let Ok(store) = self.read_store(super::direct::DirectTarget::Current) else {
            return Vec::new();
        };
        let candidates = store.find_nodes_by_property(property, value);
        store.filter_visible_node_ids(&candidates, epoch)
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
    /// The index is built from the data without holding commits off, then
    /// logged and installed (in place of an index on the same label and
    /// property) holding them off; a database opened again after a crash
    /// builds it again, with these parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the metric is invalid, no vectors are found, or
    /// dimensions don't match; the database-closed error after `close()` of a
    /// persistent database, and the incomplete-commit error after a commit
    /// that did not complete; an error in a build without the `vector-index`
    /// feature.
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
        #[cfg(feature = "vector-index")]
        {
            self.check_index_change()?;
            let (graph, _) = self.index_target(None)?;
            let index = vector_index_from_data(
                &*graph,
                label,
                property,
                dimensions,
                metric,
                m,
                ef_construction,
                quantization,
            )?;
            let record = vector_index_record(None, label, property, &index)?;
            let held = self.hold_for_standalone(false)?;
            let mut change = StandaloneChange::new();
            change.push_built(
                StandaloneOp::PutCatalog(CatalogRecord::Index(record)),
                BuiltIndex::Vector(index),
            );
            self.commit_standalone(change, &held)
        }
        #[cfg(not(feature = "vector-index"))]
        {
            let _ = (
                label,
                property,
                dimensions,
                metric,
                m,
                ef_construction,
                quantization,
            );
            Err(Error::Internal(
                "Vector index support requires the 'vector-index' feature".to_string(),
            ))
        }
    }

    /// The graph an index reads and the store that holds it, for the graph
    /// with storage key `graph` (`None` for the default graph).
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    fn index_target(
        &self,
        graph: Option<&str>,
    ) -> Result<(Arc<dyn GraphStoreSearch>, Arc<LpgStore>)> {
        match graph {
            None => Ok((self.graph_store(), self.lpg_store())),
            Some(key) => {
                let store = self
                    .lpg_store()
                    .graph(key)
                    .ok_or_else(|| super::direct::missing_graph(key))?;
                Ok((Arc::clone(&store) as Arc<dyn GraphStoreSearch>, store))
            }
        }
    }

    /// The store that takes an index of the graph with storage key `graph`
    /// (`None` for the default graph).
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    fn install_target(&self, graph: Option<&str>) -> Result<Arc<LpgStore>> {
        self.index_target(graph).map(|(_, target)| target)
    }

    /// Builds the vector index a load found defined (in the catalog of the
    /// image, or put by the replayed WAL) from the data of the graph with
    /// storage key `graph` (`None` for the default graph), and installs it
    /// without logging it again.
    ///
    /// # Errors
    ///
    /// As [`create_vector_index`](Self::create_vector_index), and when the
    /// graph does not exist.
    #[cfg(feature = "vector-index")]
    #[allow(
        clippy::too_many_arguments,
        reason = "the arguments of create_vector_index and the graph"
    )]
    pub(super) fn create_vector_index_in(
        &self,
        graph: Option<&str>,
        label: &str,
        property: &str,
        dimensions: Option<usize>,
        metric: Option<&str>,
        m: Option<usize>,
        ef_construction: Option<usize>,
        quantization: Option<&str>,
    ) -> Result<()> {
        self.check_index_change()?;
        let (read, _) = self.index_target(graph)?;
        let index = vector_index_from_data(
            &*read,
            label,
            property,
            dimensions,
            metric,
            m,
            ef_construction,
            quantization,
        )?;
        // The store that takes the index is resolved once commits are held.
        let _held = self.hold_for_standalone(false)?;
        self.install_target(graph)?
            .add_vector_index(label, property, Arc::new(index));
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
    pub(super) fn build_vector_index(
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
    ///
    /// # Errors
    ///
    /// Returns the database-closed error after `close()` of a persistent
    /// database, and the incomplete-commit error after a commit that did not
    /// complete.
    #[cfg(feature = "vector-index")]
    pub fn drop_vector_index(&self, label: &str, property: &str) -> Result<bool> {
        let held = self.hold_for_standalone(false)?;
        if self.lpg_store().get_vector_index(label, property).is_none() {
            return Ok(false);
        }
        let mut change = StandaloneChange::new();
        change.push(drop_index(
            None,
            IndexKeyRecord::Vector {
                label: label.to_string(),
                property: property.to_string(),
            },
        ));
        self.commit_standalone(change, &held)?;
        grafeo_info!("Vector index dropped: :{label}({property})");
        // A spilled column no index reads any more comes back from its
        // cache file: nothing else would reload it (#594).
        #[cfg(all(feature = "vector-index", not(feature = "temporal")))]
        self.reload_unindexed_column(property);
        Ok(true)
    }

    /// Reloads the spilled column `property` when no vector index uses it any
    /// more. A column that cannot be read stays spilled (and readable).
    #[cfg(all(feature = "vector-index", not(feature = "temporal")))]
    fn reload_unindexed_column(&self, property: &str) {
        let store = self.lpg_store();
        let still_indexed = store.vector_index_entries().iter().any(|(key, _)| {
            key.split_once(':')
                .is_some_and(|(_, indexed)| indexed == property)
        });
        if still_indexed {
            return;
        }
        let key = grafeo_common::types::PropertyKey::new(property);
        if let Err(error) = store.reload_node_property_column(&key) {
            grafeo_common::grafeo_warn!("the column {property} stays spilled: {error}");
        }
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
    /// parameters are used. The new index replaces the old one once it is
    /// built; a rebuild that fails leaves the old one.
    ///
    /// # Errors
    ///
    /// Returns an error if the rebuild fails (e.g., no matching vectors found
    /// and no dimensions can be inferred), and the errors of
    /// [`create_vector_index`](Self::create_vector_index).
    #[cfg(feature = "vector-index")]
    pub fn rebuild_vector_index(&self, label: &str, property: &str) -> Result<()> {
        // Preserve config and quantization type from existing index if available
        let existing = self.lpg_store().get_vector_index(label, property);

        let (config, quantization_name) = if let Some(ref idx) = existing {
            let qt = idx.quantization_type().and_then(quantization_name);
            (Some(idx.config().clone()), qt)
        } else {
            (None, None)
        };

        // The new index replaces the old one when it is installed.
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
    /// The index is built from the data without holding commits off, then
    /// logged and installed (in place of an index on the same label and
    /// property) holding them off.
    ///
    /// # Errors
    ///
    /// Returns the database-closed error after `close()` of a persistent
    /// database, and the incomplete-commit error after a commit that did not
    /// complete.
    #[cfg(feature = "text-index")]
    pub fn create_text_index(&self, label: &str, property: &str) -> Result<()> {
        self.check_index_change()?;
        let (graph, _) = self.index_target(None)?;
        let index = text_index_from_data(&*graph, label, property);
        let held = self.hold_for_standalone(false)?;
        let mut change = StandaloneChange::new();
        change.push_built(
            put_index(
                None,
                IndexKindRecord::Text {
                    label: label.to_string(),
                    property: property.to_string(),
                },
            ),
            BuiltIndex::Text(index),
        );
        self.commit_standalone(change, &held)
    }

    /// Builds the text index a load found defined (see
    /// [`create_vector_index_in`](Self::create_vector_index_in)) from the
    /// data of the graph with storage key `graph` (`None` for the default
    /// graph), and installs it without logging it again.
    ///
    /// # Errors
    ///
    /// As [`create_text_index`](Self::create_text_index), and when the
    /// graph does not exist.
    #[cfg(feature = "text-index")]
    pub(super) fn create_text_index_in(
        &self,
        graph: Option<&str>,
        label: &str,
        property: &str,
    ) -> Result<()> {
        self.check_index_change()?;
        let (read, _) = self.index_target(graph)?;
        let index = text_index_from_data(&*read, label, property);
        let _held = self.hold_for_standalone(false)?;
        self.install_target(graph)?.add_text_index(
            label,
            property,
            Arc::new(parking_lot::RwLock::new(index)),
        );
        Ok(())
    }

    /// Drops a text index on a label+property pair.
    ///
    /// Returns `true` if the index existed and was removed.
    ///
    /// # Errors
    ///
    /// Returns the database-closed error after `close()` of a persistent
    /// database, and the incomplete-commit error after a commit that did not
    /// complete.
    #[cfg(feature = "text-index")]
    pub fn drop_text_index(&self, label: &str, property: &str) -> Result<bool> {
        let held = self.hold_for_standalone(false)?;
        if self.lpg_store().get_text_index(label, property).is_none() {
            return Ok(false);
        }
        let mut change = StandaloneChange::new();
        change.push(drop_index(
            None,
            IndexKeyRecord::Text {
                label: label.to_string(),
                property: property.to_string(),
            },
        ));
        self.commit_standalone(change, &held)?;
        Ok(true)
    }

    /// Rebuilds a text index by re-scanning all matching nodes: the new index
    /// replaces the old one once it is built.
    ///
    /// Use after bulk property updates to keep the index current.
    ///
    /// # Errors
    ///
    /// The errors of [`create_text_index`](Self::create_text_index).
    #[cfg(feature = "text-index")]
    pub fn rebuild_text_index(&self, label: &str, property: &str) -> Result<()> {
        self.create_text_index(label, property)
    }
}
