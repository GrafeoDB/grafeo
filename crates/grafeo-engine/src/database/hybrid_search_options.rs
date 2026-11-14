//! The optional arguments of a hybrid search.

use grafeo_common::types::Value;

/// The optional arguments of
/// [`GrafeoDB::hybrid_search_with`](super::GrafeoDB::hybrid_search_with).
///
/// Start from [`HybridSearchOptions::new()`] (a text-only search at the
/// defaults) and change options with the `with_*` methods. Later releases may
/// add options, so outside this crate it cannot be built with a struct
/// literal.
///
/// # Examples
///
/// ```
/// use grafeo_engine::database::HybridSearchOptions;
///
/// let query = [0.1_f32, 0.9, 0.0];
/// let options = HybridSearchOptions::new()
///     .with_query_vector(&query)
///     .with_ef(256);
/// assert_eq!(options.ef, Some(256));
/// ```
///
/// ```compile_fail,E0639
/// use grafeo_engine::database::HybridSearchOptions;
///
/// let options = HybridSearchOptions {
///     ef: Some(256),
///     ..HybridSearchOptions::default()
/// };
/// ```
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct HybridSearchOptions<'a> {
    /// The query vector of the vector half (default none: text only).
    pub query_vector: Option<&'a [f32]>,
    /// The beam width of the vector half, as for
    /// [`GrafeoDB::vector_search`](super::GrafeoDB::vector_search) (default
    /// none: the index's default). The text half has no beam and ignores it.
    pub ef: Option<usize>,
    /// How the two halves' results are fused (default none: RRF with k=60).
    pub fusion: Option<grafeo_core::index::text::FusionMethod>,
    /// Property filters, as for
    /// [`GrafeoDB::vector_search`](super::GrafeoDB::vector_search): equality
    /// filters and operator filters (`$gt`, `$in`, ...). Both the text and
    /// the vector search keep only the nodes of the label that match all of
    /// them before the results are fused, so up to `k` matching nodes come
    /// back. Text scores stay those of the whole index.
    pub filters: Option<&'a std::collections::HashMap<String, Value>>,
}

impl<'a> HybridSearchOptions<'a> {
    /// The default options: no query vector, the index's default beam, RRF,
    /// no filters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the query vector of the vector half.
    #[must_use]
    pub fn with_query_vector(mut self, query_vector: &'a [f32]) -> Self {
        self.query_vector = Some(query_vector);
        self
    }

    /// Sets the beam width of the vector half.
    #[must_use]
    pub fn with_ef(mut self, ef: usize) -> Self {
        self.ef = Some(ef);
        self
    }

    /// Sets how the two halves' results are fused.
    #[must_use]
    pub fn with_fusion(mut self, fusion: grafeo_core::index::text::FusionMethod) -> Self {
        self.fusion = Some(fusion);
        self
    }

    /// Sets the property filters.
    #[must_use]
    pub fn with_filters(mut self, filters: &'a std::collections::HashMap<String, Value>) -> Self {
        self.filters = Some(filters);
        self
    }
}
