//! Vector, text, and hybrid search operations for GrafeoDB.

#[cfg(any(
    feature = "vector-index",
    feature = "text-index",
    feature = "hybrid-search"
))]
use grafeo_common::types::NodeId;
#[cfg(feature = "vector-index")]
use grafeo_common::types::Value;
#[cfg(any(feature = "text-index", feature = "hybrid-search"))]
use grafeo_common::utils::error::Error;
#[cfg(any(
    feature = "vector-index",
    feature = "text-index",
    feature = "hybrid-search"
))]
use grafeo_common::utils::error::Result;

impl super::GrafeoDB {
    /// Creates a vector accessor for the given label/property, using spilled
    /// MmapStorage if the index has been spilled to disk.
    #[cfg(all(feature = "vector-index", feature = "mmap", not(feature = "temporal")))]
    fn make_vector_accessor<'a>(
        &'a self,
        label: &str,
        property: &str,
    ) -> grafeo_core::index::vector::VectorAccessorKind<'a> {
        let key = format!("{label}:{property}");
        if let Some(ref spill_map) = self.vector_spill_storages {
            let map = spill_map.read();
            if let Some(storage) = map.get(&key) {
                return grafeo_core::index::vector::VectorAccessorKind::Spilled(
                    grafeo_core::index::vector::SpillableVectorAccessor::new(
                        self.graph_store_ref(),
                        property,
                        std::sync::Arc::clone(storage)
                            as std::sync::Arc<dyn grafeo_core::index::vector::VectorStorage>,
                    ),
                );
            }
        }
        grafeo_core::index::vector::VectorAccessorKind::Property(
            grafeo_core::index::vector::PropertyVectorAccessor::new(self.graph_store_ref(), property),
        )
    }

    /// Creates a vector accessor (no spill support when mmap or temporal unavailable).
    #[cfg(not(all(feature = "vector-index", feature = "mmap", not(feature = "temporal"))))]
    #[cfg(feature = "vector-index")]
    fn make_vector_accessor<'a>(
        &'a self,
        _label: &str,
        property: &str,
    ) -> grafeo_core::index::vector::VectorAccessorKind<'a> {
        grafeo_core::index::vector::VectorAccessorKind::Property(
            grafeo_core::index::vector::PropertyVectorAccessor::new(self.graph_store_ref(), property),
        )
    }

    /// Computes a node allowlist from property filters.
    ///
    /// Supports equality filters (scalar values) and operator filters (Map values
    /// with `$`-prefixed keys like `$gt`, `$lt`, `$in`, `$contains`).
    ///
    /// Returns `None` if filters is `None` or empty (meaning no filtering),
    /// or `Some(set)` with the intersection (possibly empty).
    #[cfg(feature = "vector-index")]
    fn compute_filter_allowlist(
        &self,
        label: &str,
        filters: Option<&std::collections::HashMap<String, Value>>,
    ) -> Option<std::collections::HashSet<NodeId>> {
        let filters = filters.filter(|f| !f.is_empty())?;

        let graph = self.graph_store_ref();

        // Start with all nodes for this label
        let label_nodes: std::collections::HashSet<NodeId> =
            graph.nodes_by_label(label).into_iter().collect();

        let mut allowlist = label_nodes;

        for (key, filter_value) in filters {
            // Check if this is an operator filter (Map with $-prefixed keys)
            let is_operator_filter = matches!(filter_value, Value::Map(ops) if ops.keys().any(|k| k.as_str().starts_with('$')));

            if is_operator_filter {
                // Operator filter: scan only the current allowlist (not all nodes).
                // This is much faster when a prior filter has already narrowed the set.
                let prop_key = grafeo_common::types::PropertyKey::new(key);
                allowlist.retain(|&node_id| {
                    graph
                        .get_node_property(node_id, &prop_key)
                        .is_some_and(|v| grafeo_core::LpgStore::matches_filter(&v, filter_value))
                });
            } else {
                // Equality filter: use indexed lookup when available
                let matching: std::collections::HashSet<NodeId> = graph
                    .find_nodes_by_property(key, filter_value)
                    .into_iter()
                    .collect();
                allowlist = allowlist.intersection(&matching).copied().collect();
            }

            // Short-circuit: empty intersection means no results possible
            if allowlist.is_empty() {
                return Some(allowlist);
            }
        }

        Some(allowlist)
    }

    /// Searches for the k nearest neighbors of a query vector.
    ///
    /// Uses the HNSW index created by [`create_vector_index`](Self::create_vector_index).
    ///
    /// # Arguments
    ///
    /// * `label` - Node label that was indexed
    /// * `property` - Property that was indexed
    /// * `query` - Query vector (slice of floats)
    /// * `k` - Number of nearest neighbors to return
    /// * `ef` - Search beam width (higher = better recall, slower). Uses index default if `None`.
    /// * `filters` - Optional property equality filters. Only nodes matching all
    ///   `(key, value)` pairs will appear in results.
    ///
    /// # Returns
    ///
    /// Vector of `(NodeId, distance)` pairs sorted by distance ascending
    /// (lower distance = more similar). The distance scale depends on the
    /// metric configured at index creation: cosine \[0, 2\], euclidean
    /// \[0, inf), dot product (negated, so lower = higher similarity),
    /// manhattan \[0, inf).
    ///
    /// # Errors
    ///
    /// Returns an error if no vector index exists for the given label and property.
    #[cfg(feature = "vector-index")]
    pub fn vector_search(
        &self,
        label: &str,
        property: &str,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
        filters: Option<&std::collections::HashMap<String, Value>>,
    ) -> Result<Vec<(grafeo_common::types::NodeId, f32)>> {
        let index = self.lpg_store().get_vector_index(label, property).ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(format!(
                "No vector index found for :{label}({property}). Call create_vector_index() first."
            ))
        })?;

        let accessor = self.make_vector_accessor(label, property);

        let results = match self.compute_filter_allowlist(label, filters) {
            Some(allowlist) => match ef {
                Some(ef_val) => {
                    index.search_with_ef_and_filter(query, k, ef_val, &allowlist, &accessor)
                }
                None => index.search_with_filter(query, k, &allowlist, &accessor),
            },
            None => match ef {
                Some(ef_val) => index.search_with_ef(query, k, ef_val, &accessor),
                None => index.search(query, k, &accessor),
            },
        };

        Ok(results)
    }

    /// Searches for nearest neighbors for multiple query vectors in parallel.
    ///
    /// Uses rayon parallel iteration under the hood for multi-core throughput.
    ///
    /// # Arguments
    ///
    /// * `label` - Node label that was indexed
    /// * `property` - Property that was indexed
    /// * `queries` - Batch of query vectors
    /// * `k` - Number of nearest neighbors per query
    /// * `ef` - Search beam width (uses index default if `None`)
    /// * `filters` - Optional property equality filters
    ///
    /// # Errors
    ///
    /// Returns an error if no vector index exists for the given label and property.
    #[cfg(feature = "vector-index")]
    pub fn batch_vector_search(
        &self,
        label: &str,
        property: &str,
        queries: &[Vec<f32>],
        k: usize,
        ef: Option<usize>,
        filters: Option<&std::collections::HashMap<String, Value>>,
    ) -> Result<Vec<Vec<(grafeo_common::types::NodeId, f32)>>> {
        let index = self.lpg_store().get_vector_index(label, property).ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(format!(
                "No vector index found for :{label}({property}). Call create_vector_index() first."
            ))
        })?;

        let accessor = self.make_vector_accessor(label, property);

        let results = match self.compute_filter_allowlist(label, filters) {
            Some(allowlist) => match ef {
                Some(ef_val) => {
                    index.batch_search_with_ef_and_filter(queries, k, ef_val, &allowlist, &accessor)
                }
                None => index.batch_search_with_filter(queries, k, &allowlist, &accessor),
            },
            None => match ef {
                Some(ef_val) => index.batch_search_with_ef(queries, k, ef_val, &accessor),
                None => index.batch_search(queries, k, &accessor),
            },
        };

        Ok(results)
    }

    /// Searches for diverse nearest neighbors using Maximal Marginal Relevance (MMR).
    ///
    /// MMR balances relevance (similarity to query) with diversity (dissimilarity
    /// among selected results). This is the algorithm used by LangChain's
    /// `mmr_traversal_search()` for RAG applications.
    ///
    /// # Arguments
    ///
    /// * `label` - Node label that was indexed
    /// * `property` - Property that was indexed
    /// * `query` - Query vector
    /// * `k` - Number of diverse results to return
    /// * `fetch_k` - Number of initial candidates from HNSW (default: `4 * k`)
    /// * `lambda` - Relevance vs. diversity in \[0, 1\] (default: 0.5).
    ///   1.0 = pure relevance, 0.0 = pure diversity.
    /// * `ef` - HNSW search beam width (uses index default if `None`)
    /// * `filters` - Optional property equality filters
    ///
    /// # Returns
    ///
    /// `(NodeId, distance)` pairs in MMR selection order. The f32 is the original
    /// distance from the query, matching [`vector_search`](Self::vector_search).
    ///
    /// # Errors
    ///
    /// Returns an error if no vector index exists for the given label and property.
    #[cfg(feature = "vector-index")]
    #[allow(clippy::too_many_arguments)]
    pub fn mmr_search(
        &self,
        label: &str,
        property: &str,
        query: &[f32],
        k: usize,
        fetch_k: Option<usize>,
        lambda: Option<f32>,
        ef: Option<usize>,
        filters: Option<&std::collections::HashMap<String, Value>>,
    ) -> Result<Vec<(grafeo_common::types::NodeId, f32)>> {
        use grafeo_core::index::vector::mmr_select;

        let index = self.lpg_store().get_vector_index(label, property).ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(format!(
                "No vector index found for :{label}({property}). Call create_vector_index() first."
            ))
        })?;

        let accessor = self.make_vector_accessor(label, property);

        let fetch_k = fetch_k.unwrap_or(k.saturating_mul(4).max(k));
        let lambda = lambda.unwrap_or(0.5);

        // Step 1: Fetch candidates from HNSW (with optional filter)
        let initial_results = match self.compute_filter_allowlist(label, filters) {
            Some(allowlist) => match ef {
                Some(ef_val) => {
                    index.search_with_ef_and_filter(query, fetch_k, ef_val, &allowlist, &accessor)
                }
                None => index.search_with_filter(query, fetch_k, &allowlist, &accessor),
            },
            None => match ef {
                Some(ef_val) => index.search_with_ef(query, fetch_k, ef_val, &accessor),
                None => index.search(query, fetch_k, &accessor),
            },
        };

        if initial_results.is_empty() {
            return Ok(Vec::new());
        }

        // Step 2: Retrieve stored vectors for MMR pairwise comparison
        use grafeo_core::index::vector::VectorAccessor;
        let candidates: Vec<(grafeo_common::types::NodeId, f32, std::sync::Arc<[f32]>)> =
            initial_results
                .into_iter()
                .filter_map(|(id, dist)| accessor.get_vector(id).map(|vec| (id, dist, vec)))
                .collect();

        // Step 3: Build slice-based candidates for mmr_select
        let candidate_refs: Vec<(grafeo_common::types::NodeId, f32, &[f32])> = candidates
            .iter()
            .map(|(id, dist, vec)| (*id, *dist, vec.as_ref()))
            .collect();

        // Step 4: Run MMR selection
        let metric = index.config().metric;
        Ok(mmr_select(query, &candidate_refs, k, lambda, metric))
    }

    /// Searches a text index using BM25 scoring.
    ///
    /// Returns up to `k` results as `(NodeId, score)` pairs sorted by
    /// descending relevance score (higher = more relevant). BM25 scores
    /// are unbounded positive floats whose magnitude depends on corpus
    /// statistics, so compare them only within a single query's results.
    ///
    /// # Errors
    ///
    /// Returns an error if no text index exists for this label+property.
    #[cfg(feature = "text-index")]
    pub fn text_search(
        &self,
        label: &str,
        property: &str,
        query: &str,
        k: usize,
    ) -> Result<Vec<(NodeId, f64)>> {
        let index = self
            .lpg_store()
            .get_text_index(label, property)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "No text index found for :{label}({property}). Call create_text_index() first."
                ))
            })?;

        Ok(index.read().search(query, k))
    }

    /// Performs hybrid search combining text (BM25) and vector similarity.
    ///
    /// Runs both text search and vector search, then fuses results using
    /// the specified method (default: Reciprocal Rank Fusion). If either
    /// index is missing, that source is silently omitted from fusion.
    /// Returns empty results only when both indexes are absent.
    ///
    /// # Arguments
    ///
    /// * `label` - Node label to search within
    /// * `text_property` - Property indexed for text search
    /// * `vector_property` - Property indexed for vector search
    /// * `query_text` - Text query for BM25 search
    /// * `query_vector` - Vector query for similarity search (optional)
    /// * `k` - Number of results to return
    /// * `fusion` - Score fusion method (default: RRF with k=60)
    ///
    /// # Returns
    ///
    /// `(NodeId, score)` pairs sorted by fused score **descending**
    /// (higher = more relevant). These are fusion scores, **not**
    /// distances. With RRF, scores are `sum(1 / (k + rank))` across
    /// sources. With weighted fusion, scores are min-max normalized
    /// and combined with explicit weights.
    ///
    /// **Important:** Do not treat these scores the same as
    /// [`vector_search`](Self::vector_search) distances. For temporal
    /// decay or boosting, multiply fusion scores (higher = better)
    /// rather than dividing (which is appropriate for distances where
    /// lower = better).
    ///
    /// # Errors
    ///
    /// Returns an error if the required indexes don't exist.
    #[cfg(feature = "hybrid-search")]
    #[allow(clippy::too_many_arguments)]
    pub fn hybrid_search(
        &self,
        label: &str,
        text_property: &str,
        vector_property: &str,
        query_text: &str,
        query_vector: Option<&[f32]>,
        k: usize,
        fusion: Option<grafeo_core::index::text::FusionMethod>,
    ) -> Result<Vec<(NodeId, f64)>> {
        use grafeo_core::index::text::fuse_results;

        let fusion_method = fusion.unwrap_or_default();
        let mut sources: Vec<Vec<(NodeId, f64)>> = Vec::new();

        // Text search
        if let Some(text_index) = self.lpg_store().get_text_index(label, text_property) {
            let text_results = text_index.read().search(query_text, k * 2);
            if !text_results.is_empty() {
                sources.push(text_results);
            }
        }

        // Vector search (if query vector provided)
        if let Some(query_vec) = query_vector
            && let Some(vector_index) = self.lpg_store().get_vector_index(label, vector_property)
        {
            let accessor = self.make_vector_accessor(label, vector_property);
            let vector_results = vector_index.search(query_vec, k * 2, &accessor);
            if !vector_results.is_empty() {
                // Negate distances so that "closer = higher score", matching
                // the text source convention (higher = better). This is
                // essential for weighted fusion where min-max normalization
                // would otherwise invert the vector ranking. RRF is
                // unaffected because it uses rank positions, not values.
                sources.push(
                    vector_results
                        .into_iter()
                        .map(|(id, dist)| (id, -f64::from(dist)))
                        .collect(),
                );
            }
        }

        if sources.is_empty() {
            return Ok(Vec::new());
        }

        Ok(fuse_results(&sources, &fusion_method, k))
    }
}
