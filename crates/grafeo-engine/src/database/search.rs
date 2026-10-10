//! Vector, text, and hybrid search operations for GrafeoDB.

#[cfg(any(
    feature = "vector-index",
    feature = "text-index",
    feature = "hybrid-search"
))]
use grafeo_common::types::NodeId;
#[cfg(any(feature = "vector-index", feature = "text-index"))]
use grafeo_common::types::Value;
#[cfg(any(feature = "text-index", feature = "hybrid-search"))]
use grafeo_common::utils::error::Error;
#[cfg(any(
    feature = "vector-index",
    feature = "text-index",
    feature = "hybrid-search"
))]
use grafeo_common::utils::error::Result;
#[cfg(feature = "vector-index")]
use grafeo_core::index::vector::check_query_vector;

impl super::GrafeoDB {
    /// Creates the vector accessor for an indexed property: the property
    /// store, which reads a spilled column through its cache file in place.
    #[cfg(feature = "vector-index")]
    fn make_vector_accessor<'a>(
        &'a self,
        property: &str,
    ) -> grafeo_core::index::vector::VectorAccessorKind<'a> {
        grafeo_core::index::vector::VectorAccessorKind::Property(
            grafeo_core::index::vector::PropertyVectorAccessor::new(
                self.graph_store_ref(),
                property,
            ),
        )
    }

    /// Computes a node allowlist from property filters.
    ///
    /// Supports equality filters (scalar values) and operator filters (Map values
    /// with `$`-prefixed keys like `$gt`, `$lt`, `$in`, `$contains`).
    ///
    /// Returns `None` if filters is `None` or empty (meaning no filtering),
    /// or `Some(set)` with the intersection (possibly empty).
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    fn compute_filter_allowlist(
        &self,
        label: &str,
        filters: Option<&std::collections::HashMap<String, Value>>,
    ) -> Option<std::collections::HashSet<NodeId>> {
        let filters = filters.filter(|f| !f.is_empty())?;

        let graph = self.graph_store();

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
    /// Returns an error if no vector index exists for the given label and
    /// property, and an invalid-value error if the query vector has another
    /// number of values than the index's dimensions, or a NaN or an infinite
    /// value (see [`check_query_vector`]).
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
        let store = self.lpg_store();
        let index = store.get_vector_index(label, property).ok_or_else(|| {
            grafeo_common::utils::error::Error::InvalidValue(format!(
                "there is no vector index on :{label}({property}); create one first"
            ))
        })?;
        check_query_vector(query, index.config().dimensions, label, property)?;

        let accessor = self.make_vector_accessor(property);
        let allowlist = self.compute_filter_allowlist(label, filters);

        Ok(search_vector_index(
            &index,
            query,
            k,
            ef,
            allowlist.as_ref(),
            &accessor,
        ))
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
    /// Returns an error if no vector index exists for the given label and
    /// property, and an invalid-value error if any query vector is one the
    /// index cannot measure (see [`vector_search`](Self::vector_search)): no
    /// query of the batch runs then.
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
        let store = self.lpg_store();
        let index = store.get_vector_index(label, property).ok_or_else(|| {
            grafeo_common::utils::error::Error::InvalidValue(format!(
                "there is no vector index on :{label}({property}); create one first"
            ))
        })?;
        for query in queries {
            check_query_vector(query, index.config().dimensions, label, property)?;
        }

        let accessor = self.make_vector_accessor(property);
        let allowlist = self.compute_filter_allowlist(label, filters);

        let batch = match (&allowlist, ef) {
            (Some(allowlist), Some(ef_val)) => {
                index.batch_search_with_ef_and_filter(queries, k, ef_val, allowlist, &accessor)
            }
            (Some(allowlist), None) => {
                index.batch_search_with_filter(queries, k, allowlist, &accessor)
            }
            (None, Some(ef_val)) => index.batch_search_with_ef(queries, k, ef_val, &accessor),
            (None, None) => index.batch_search(queries, k, &accessor),
        };
        Ok(batch)
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
    /// Returns an error if no vector index exists for the given label and
    /// property, and an invalid-value error if the query vector is one the
    /// index cannot measure (see [`vector_search`](Self::vector_search)).
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

        let store = self.lpg_store();
        let index = store.get_vector_index(label, property).ok_or_else(|| {
            grafeo_common::utils::error::Error::InvalidValue(format!(
                "there is no vector index on :{label}({property}); create one first"
            ))
        })?;
        check_query_vector(query, index.config().dimensions, label, property)?;

        let accessor = self.make_vector_accessor(property);

        let fetch_k = fetch_k.unwrap_or(k.saturating_mul(4).max(k));
        let lambda = lambda.unwrap_or(0.5);

        // Step 1: Fetch candidates from HNSW (with optional filter)
        let allowlist = self.compute_filter_allowlist(label, filters);
        let initial_results =
            search_vector_index(&index, query, fetch_k, ef, allowlist.as_ref(), &accessor);

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
    /// `filters` takes property filters as
    /// [`vector_search`](Self::vector_search) does: equality filters and
    /// operator filters (`$gt`, `$in`, ...). Only the nodes of `label` that
    /// match all of them are searched, so up to `k` matching nodes come
    /// back; their scores are those of the whole index.
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
        filters: Option<&std::collections::HashMap<String, Value>>,
    ) -> Result<Vec<(NodeId, f64)>> {
        let store = self.lpg_store();
        let index = store.get_text_index(label, property).ok_or_else(|| {
            Error::InvalidValue(format!(
                "there is no text index on :{label}({property}); create one first"
            ))
        })?;

        let index = index.read();
        Ok(match self.compute_filter_allowlist(label, filters) {
            Some(allowed) => index.search_with_filter(query, k, &allowed),
            None => index.search(query, k),
        })
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
    /// * `filters` - Optional property filters, as for
    ///   [`vector_search`](Self::vector_search): equality filters and
    ///   operator filters (`$gt`, `$in`, ...). Both the text and the vector
    ///   search keep only the nodes of `label` that match all of them before
    ///   the results are fused, so up to `k` matching nodes come back.
    ///   Text scores stay those of the whole index.
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
    /// Returns an invalid-value error if the vector index exists and the
    /// query vector is one it cannot measure (see
    /// [`vector_search`](Self::vector_search)).
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
        filters: Option<&std::collections::HashMap<String, Value>>,
    ) -> Result<Vec<(NodeId, f64)>> {
        use grafeo_core::index::text::fuse_results;

        let fusion_method = fusion.unwrap_or_default();
        let mut sources: Vec<Vec<(NodeId, f64)>> = Vec::new();
        let store = self.lpg_store();
        let fetch = k.saturating_mul(2);
        let vector_index = store.get_vector_index(label, vector_property);
        if let (Some(query_vec), Some(index)) = (query_vector, &vector_index) {
            check_query_vector(query_vec, index.config().dimensions, label, vector_property)?;
        }

        // Both searches keep only the nodes that match the filters, before
        // fusion: filtering the fused results would return fewer than k.
        let allowlist = self.compute_filter_allowlist(label, filters);
        if allowlist.as_ref().is_some_and(|allowed| allowed.is_empty()) {
            return Ok(Vec::new());
        }

        // Text search
        if let Some(text_index) = store.get_text_index(label, text_property) {
            let text_index = text_index.read();
            let text_results = match &allowlist {
                Some(allowed) => text_index.search_with_filter(query_text, fetch, allowed),
                None => text_index.search(query_text, fetch),
            };
            if !text_results.is_empty() {
                sources.push(text_results);
            }
        }

        // Vector search (if query vector provided)
        if let Some(query_vec) = query_vector
            && let Some(vector_index) = vector_index
        {
            let accessor = self.make_vector_accessor(vector_property);
            let vector_results = search_vector_index(
                &vector_index,
                query_vec,
                fetch,
                None,
                allowlist.as_ref(),
                &accessor,
            );
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

/// One search of `index` for the `fetch` nodes nearest to `query`: with the
/// beam width `ef` (the index's own when `None`), and among the nodes of
/// `allowlist` only, when given.
#[cfg(feature = "vector-index")]
fn search_vector_index(
    index: &grafeo_core::index::vector::VectorIndexKind,
    query: &[f32],
    fetch: usize,
    ef: Option<usize>,
    allowlist: Option<&std::collections::HashSet<NodeId>>,
    accessor: &impl grafeo_core::index::vector::VectorAccessor,
) -> Vec<(NodeId, f32)> {
    match (allowlist, ef) {
        (Some(allowlist), Some(ef)) => {
            index.search_with_ef_and_filter(query, fetch, ef, allowlist, accessor)
        }
        (Some(allowlist), None) => index.search_with_filter(query, fetch, allowlist, accessor),
        (None, Some(ef)) => index.search_with_ef(query, fetch, ef, accessor),
        (None, None) => index.search(query, fetch, accessor),
    }
}
