//! Vector, text, and hybrid search operations for GrafeoDB.
//!
//! A search reads the default graph as a read outside a transaction does: at
//! the published epoch, pinned for the search's run (#412 decision Q6), so a
//! collect meanwhile keeps every version it reads. The filters, the index
//! search and the reranking all read at that one view, through the store's
//! view-taking methods, which take the store's lock and then an index's
//! lock, the order the writes take them in (a search never holds an index
//! lock while it calls into the store).

#[cfg(any(
    feature = "vector-index",
    feature = "text-index",
    feature = "hybrid-search"
))]
use grafeo_common::types::NodeId;
#[cfg(any(feature = "vector-index", feature = "text-index"))]
use grafeo_common::types::{ReadView, Value};
#[cfg(any(feature = "vector-index", feature = "text-index"))]
use grafeo_common::utils::error::Error;
#[cfg(any(
    feature = "vector-index",
    feature = "text-index",
    feature = "hybrid-search"
))]
use grafeo_common::utils::error::Result;
#[cfg(any(feature = "vector-index", feature = "text-index"))]
use grafeo_core::graph::rowgroup::{IndexSearch, RowGroupStore};
#[cfg(any(feature = "vector-index", feature = "text-index"))]
use grafeo_core::graph::{GraphStore, GraphStoreSearch};
#[cfg(feature = "vector-index")]
use grafeo_core::index::vector::check_query_vector;

/// The error of a search through a vector index the default graph lacks.
#[cfg(feature = "vector-index")]
fn no_vector_index(label: &str, property: &str) -> Error {
    Error::InvalidValue(format!(
        "there is no vector index on :{label}({property}); create one first"
    ))
}

impl super::GrafeoDB {
    /// The view a search reads at, and the pin that keeps it while the
    /// search runs: the published epoch, as for every read outside a
    /// transaction.
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    fn search_view(&self) -> (crate::transaction::ReadPin, ReadView) {
        let pin = self.transaction_manager.pin_read();
        (pin, ReadView::committed(self.read_epoch()))
    }

    /// Computes a node allowlist from property filters, as `view` sees the
    /// nodes of `label` in `store`.
    ///
    /// Supports equality filters (scalar values) and operator filters (Map values
    /// with `$`-prefixed keys like `$gt`, `$lt`, `$in`, `$contains`).
    ///
    /// Returns `None` if filters is `None` or empty (meaning no filtering),
    /// or `Some(set)` with the intersection (possibly empty).
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    fn compute_filter_allowlist(
        store: &RowGroupStore,
        label: &str,
        filters: Option<&std::collections::HashMap<String, Value>>,
        view: ReadView,
    ) -> Option<std::collections::HashSet<NodeId>> {
        let filters = filters.filter(|f| !f.is_empty())?;

        // Start with all nodes for this label
        let mut allowlist: std::collections::HashSet<NodeId> =
            store.nodes_by_label_at(label, view).into_iter().collect();

        for (key, filter_value) in filters {
            // Check if this is an operator filter (Map with $-prefixed keys)
            let is_operator_filter = matches!(filter_value, Value::Map(ops) if ops.keys().any(|k| k.as_str().starts_with('$')));

            if is_operator_filter {
                // Operator filter: scan only the current allowlist (not all nodes).
                // This is much faster when a prior filter has already narrowed the set.
                allowlist.retain(|&node_id| {
                    store.node_property_at(node_id, key, view).is_some_and(|v| {
                        grafeo_core::graph::property_filter::matches_filter(&v, filter_value)
                    })
                });
            } else {
                // Equality filter: use indexed lookup when available
                let matching: std::collections::HashSet<NodeId> = store
                    .find_nodes_by_property_at(key, filter_value, view)
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
        let config = store
            .vector_index_config(label, property)
            .ok_or_else(|| no_vector_index(label, property))?;
        check_query_vector(query, config.dimensions, label, property)?;

        let (_pin, view) = self.search_view();
        let allowlist = Self::compute_filter_allowlist(&store, label, filters, view);
        let search = IndexSearch {
            k,
            ef,
            allowed: allowlist.as_ref(),
        };
        store
            .vector_index_search_at(label, property, query, search, view)
            .ok_or_else(|| no_vector_index(label, property))
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
        let config = store
            .vector_index_config(label, property)
            .ok_or_else(|| no_vector_index(label, property))?;
        for query in queries {
            check_query_vector(query, config.dimensions, label, property)?;
        }

        let (_pin, view) = self.search_view();
        let allowlist = Self::compute_filter_allowlist(&store, label, filters, view);
        let search = IndexSearch {
            k,
            ef,
            allowed: allowlist.as_ref(),
        };
        store
            .vector_index_batch_search_at(label, property, queries, search, view)
            .ok_or_else(|| no_vector_index(label, property))
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
        let config = store
            .vector_index_config(label, property)
            .ok_or_else(|| no_vector_index(label, property))?;
        check_query_vector(query, config.dimensions, label, property)?;

        let fetch_k = fetch_k.unwrap_or(k.saturating_mul(4).max(k));
        let lambda = lambda.unwrap_or(0.5);

        // Step 1: Fetch candidates from HNSW (with optional filter)
        let (_pin, view) = self.search_view();
        let allowlist = Self::compute_filter_allowlist(&store, label, filters, view);
        let search = IndexSearch {
            k: fetch_k,
            ef,
            allowed: allowlist.as_ref(),
        };
        let initial_results = store
            .vector_index_search_at(label, property, query, search, view)
            .ok_or_else(|| no_vector_index(label, property))?;

        if initial_results.is_empty() {
            return Ok(Vec::new());
        }

        // Step 2: The candidates' vectors at the view, for MMR's pairwise
        // comparison
        let candidates: Vec<(grafeo_common::types::NodeId, f32, std::sync::Arc<[f32]>)> =
            initial_results
                .into_iter()
                .filter_map(|(id, dist)| {
                    store
                        .vector_at(id, property, view)
                        .map(|vec| (id, dist, vec))
                })
                .collect();

        // Step 3: Build slice-based candidates for mmr_select
        let candidate_refs: Vec<(grafeo_common::types::NodeId, f32, &[f32])> = candidates
            .iter()
            .map(|(id, dist, vec)| (*id, *dist, vec.as_ref()))
            .collect();

        // Step 4: Run MMR selection
        Ok(mmr_select(query, &candidate_refs, k, lambda, config.metric))
    }

    /// Searches a text index using BM25 scoring.
    ///
    /// Returns up to `k` results as `(NodeId, score)` pairs sorted by
    /// descending relevance score (higher = more relevant), equal scores
    /// by ascending node id, so a cut at `k` inside a tie keeps its lowest
    /// ids and identical calls return the same list. BM25 scores
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
        let (_pin, view) = self.search_view();
        let allowlist = Self::compute_filter_allowlist(&store, label, filters, view);
        let search = IndexSearch {
            k,
            ef: None,
            allowed: allowlist.as_ref(),
        };
        store
            .text_index_search_at(label, property, query, search, view)
            .ok_or_else(|| {
                Error::InvalidValue(format!(
                    "there is no text index on :{label}({property}); create one first"
                ))
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
    /// (higher = more relevant), equal scores by ascending node id, so
    /// identical calls return the same list. These are fusion scores, **not**
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
        let vector_config = store.vector_index_config(label, vector_property);
        if let (Some(query_vec), Some(config)) = (query_vector, &vector_config) {
            check_query_vector(query_vec, config.dimensions, label, vector_property)?;
        }

        // Both searches keep only the nodes that match the filters, before
        // fusion: filtering the fused results would return fewer than k.
        let (_pin, view) = self.search_view();
        let allowlist = Self::compute_filter_allowlist(&store, label, filters, view);
        if allowlist.as_ref().is_some_and(|allowed| allowed.is_empty()) {
            return Ok(Vec::new());
        }
        let search = IndexSearch {
            k: fetch,
            ef: None,
            allowed: allowlist.as_ref(),
        };

        // Text search
        if let Some(text_results) =
            store.text_index_search_at(label, text_property, query_text, search, view)
            && !text_results.is_empty()
        {
            sources.push(text_results);
        }

        // Vector search (if query vector provided)
        if let Some(query_vec) = query_vector
            && vector_config.is_some()
            && let Some(vector_results) =
                store.vector_index_search_at(label, vector_property, query_vec, search, view)
            && !vector_results.is_empty()
        {
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

        if sources.is_empty() {
            return Ok(Vec::new());
        }

        Ok(fuse_results(&sources, &fusion_method, k))
    }
}
