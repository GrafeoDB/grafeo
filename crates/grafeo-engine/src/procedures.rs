//! Built-in procedure registry for CALL statement execution.
//!
//! Provides the [`Procedure`] trait and [`BuiltinProcedures`] registry used by
//! all supported query languages (GQL, Cypher, SQL/PGQ) to dispatch
//! `CALL grafeo.<name>(...) [YIELD ...]` statements.
//!
//! # Unified dispatch (0.5.41+)
//!
//! Three kinds of procedures share the [`Procedure`] trait:
//!
//! - Graph algorithms (PageRank, BFS, Dijkstra, ...): adapted from
//!   [`GraphAlgorithm`] via [`GraphAlgorithmProcedure`].
//! - Catalog introspection (`db.labels`, `db.relationshipTypes`,
//!   `db.propertyKeys`): implemented directly.
//! - Vector and text search (`grafeo.search.vector`, `grafeo.search.mmr`,
//!   `grafeo.search.text`): implemented directly, require
//!   [`ProcedureContext::lpg_store`].

use std::sync::Arc;

use grafeo_adapters::plugins::algorithms::{
    ArticulationPointsAlgorithm, BellmanFordAlgorithm, BetweennessCentralityAlgorithm,
    BfsAlgorithm, BridgesAlgorithm, ClosenessCentralityAlgorithm, ClusteringCoefficientAlgorithm,
    ConnectedComponentsAlgorithm, DegreeCentralityAlgorithm, DfsAlgorithm, DijkstraAlgorithm,
    FloydWarshallAlgorithm, GraphAlgorithm, KCoreAlgorithm, KruskalAlgorithm,
    LabelPropagationAlgorithm, LouvainAlgorithm, MaxFlowAlgorithm, MinCostFlowAlgorithm,
    PageRankAlgorithm, PrimAlgorithm, SsspAlgorithm, StronglyConnectedComponentsAlgorithm,
    TopologicalSortAlgorithm,
};
use grafeo_adapters::plugins::{AlgorithmResult, ParameterDef, ParameterType, Parameters};
use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};
use grafeo_core::graph::GraphStoreSearch;
#[cfg(feature = "lpg")]
use grafeo_core::graph::lpg::LpgStore;
use hashbrown::HashMap;

/// Unified interface for built-in procedures callable via `CALL`.
///
/// Subsumes graph algorithms (through [`GraphAlgorithmProcedure`]), catalog
/// introspection, and vector/text search procedures. The planner dispatches
/// every `CALL grafeo.<name>(...)` through this trait, so adding a new
/// procedure reduces to implementing [`Procedure`] and registering it in
/// [`BuiltinProcedures::new`].
pub trait Procedure: Send + Sync {
    /// Returns the procedure name (without the `grafeo.` prefix).
    fn name(&self) -> &str;

    /// Returns a short description for `grafeo.procedures()` listings.
    fn description(&self) -> &str;

    /// Returns parameter definitions (name, type, required, default).
    fn parameters(&self) -> &[ParameterDef];

    /// Returns the canonical output column names in order.
    fn output_columns(&self) -> Vec<String>;

    /// Executes the procedure against the supplied context and parameters.
    ///
    /// # Errors
    ///
    /// Returns an error when required parameters are missing, types are
    /// wrong, or the backing operation (graph algorithm, search index, or
    /// catalog lookup) fails.
    fn execute(&self, ctx: &ProcedureContext<'_>, params: &Parameters) -> Result<AlgorithmResult>;
}

/// Runtime context passed to [`Procedure::execute`].
///
/// Graph algorithms and catalog introspection need only [`ProcedureContext::store`].
/// Vector and text search procedures additionally require
/// [`ProcedureContext::lpg_store`] to reach the HNSW and BM25 indexes owned
/// by the LPG store.
pub struct ProcedureContext<'a> {
    /// Read-only graph store, sufficient for graph algorithms and catalog
    /// introspection (labels, edge types, property keys).
    pub store: &'a dyn GraphStoreSearch,

    /// Concrete LPG store reference when available, used by search procedures
    /// to reach vector and text indexes. `None` when the active backend is
    /// not an LPG store (e.g., pure RDF) or in contexts that do not need it.
    #[cfg(feature = "lpg")]
    pub lpg_store: Option<&'a LpgStore>,
}

impl<'a> ProcedureContext<'a> {
    /// Creates a context with only a graph store available.
    #[must_use]
    pub fn new(store: &'a dyn GraphStoreSearch) -> Self {
        Self {
            store,
            #[cfg(feature = "lpg")]
            lpg_store: None,
        }
    }

    /// Creates a context with a graph store and LPG store available.
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn with_lpg_store(store: &'a dyn GraphStoreSearch, lpg_store: &'a LpgStore) -> Self {
        Self {
            store,
            lpg_store: Some(lpg_store),
        }
    }
}

/// Adapter that presents a [`GraphAlgorithm`] as a [`Procedure`].
///
/// Canonical output column names (e.g., `score` instead of `pagerank`) are
/// captured at construction time via [`canonical_output_columns`].
pub struct GraphAlgorithmProcedure {
    inner: Arc<dyn GraphAlgorithm>,
    output_columns: Vec<String>,
}

impl GraphAlgorithmProcedure {
    /// Wraps a graph algorithm as a procedure.
    pub fn new(algorithm: Arc<dyn GraphAlgorithm>) -> Self {
        let output_columns = canonical_output_columns(algorithm.as_ref());
        Self {
            inner: algorithm,
            output_columns,
        }
    }
}

impl Procedure for GraphAlgorithmProcedure {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> &str {
        self.inner.description()
    }

    fn parameters(&self) -> &[ParameterDef] {
        self.inner.parameters()
    }

    fn output_columns(&self) -> Vec<String> {
        self.output_columns.clone()
    }

    fn execute(&self, ctx: &ProcedureContext<'_>, params: &Parameters) -> Result<AlgorithmResult> {
        self.inner.execute(ctx.store, params)
    }
}

// ---------------------------------------------------------------------------
// Catalog introspection procedures
// ---------------------------------------------------------------------------

/// `db.labels` / `grafeo.labels`: lists every label present in the graph.
struct LabelsProcedure;

impl Procedure for LabelsProcedure {
    fn name(&self) -> &str {
        "labels"
    }

    fn description(&self) -> &str {
        "Lists every node label present in the graph"
    }

    fn parameters(&self) -> &[ParameterDef] {
        &[]
    }

    fn output_columns(&self) -> Vec<String> {
        vec!["label".into()]
    }

    fn execute(&self, ctx: &ProcedureContext<'_>, _params: &Parameters) -> Result<AlgorithmResult> {
        let mut result = AlgorithmResult::new(vec!["label".into()]);
        for label in ctx.store.all_labels() {
            result.rows.push(vec![Value::String(label.into())]);
        }
        Ok(result)
    }
}

/// `db.relationshipTypes` / `grafeo.relationshipTypes`: lists every edge type.
struct RelationshipTypesProcedure;

impl Procedure for RelationshipTypesProcedure {
    fn name(&self) -> &str {
        "relationshipTypes"
    }

    fn description(&self) -> &str {
        "Lists every edge type present in the graph"
    }

    fn parameters(&self) -> &[ParameterDef] {
        &[]
    }

    fn output_columns(&self) -> Vec<String> {
        vec!["relationshipType".into()]
    }

    fn execute(&self, ctx: &ProcedureContext<'_>, _params: &Parameters) -> Result<AlgorithmResult> {
        let mut result = AlgorithmResult::new(vec!["relationshipType".into()]);
        for t in ctx.store.all_edge_types() {
            result.rows.push(vec![Value::String(t.into())]);
        }
        Ok(result)
    }
}

/// `db.propertyKeys` / `grafeo.propertyKeys`: lists every property key.
struct PropertyKeysProcedure;

impl Procedure for PropertyKeysProcedure {
    fn name(&self) -> &str {
        "propertyKeys"
    }

    fn description(&self) -> &str {
        "Lists every property key present in the graph"
    }

    fn parameters(&self) -> &[ParameterDef] {
        &[]
    }

    fn output_columns(&self) -> Vec<String> {
        vec!["propertyKey".into()]
    }

    fn execute(&self, ctx: &ProcedureContext<'_>, _params: &Parameters) -> Result<AlgorithmResult> {
        let mut result = AlgorithmResult::new(vec!["propertyKey".into()]);
        for key in ctx.store.all_property_keys() {
            result.rows.push(vec![Value::String(key.into())]);
        }
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Vector / text search procedures
// ---------------------------------------------------------------------------

#[cfg(all(feature = "lpg", feature = "vector-index"))]
fn require_lpg_store<'a>(ctx: &ProcedureContext<'a>, proc_name: &str) -> Result<&'a LpgStore> {
    ctx.lpg_store.ok_or_else(|| {
        grafeo_common::utils::error::Error::Internal(format!(
            "{proc_name} requires an LPG store. Ensure the session is backed by an LPG database \
             (not a pure RDF store or external custom store)."
        ))
    })
}

#[cfg(all(feature = "lpg", feature = "vector-index"))]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "vector indexes store f32; GQL list literals arrive as f64/i64 and must narrow"
)]
fn coerce_params_to_vector(params: &Parameters, key: &str) -> Result<Vec<f32>> {
    if let Some(list) = params.get_list(key) {
        let mut out = Vec::with_capacity(list.len());
        for v in list {
            match v {
                Value::Float64(f) => out.push(*f as f32),
                Value::Int64(i) => out.push(*i as f32),
                other => {
                    return Err(grafeo_common::utils::error::Error::Internal(format!(
                        "Expected numeric list for vector parameter '{key}', found {other:?}"
                    )));
                }
            }
        }
        return Ok(out);
    }
    Err(grafeo_common::utils::error::Error::Internal(format!(
        "Missing required vector parameter '{key}'"
    )))
}

/// Checks that a query vector has the dimensions of the index it searches:
/// the index refuses (panics on) any other length.
#[cfg(all(feature = "lpg", feature = "vector-index"))]
fn check_query_dimensions(procedure: &str, query: &[f32], dimensions: usize) -> Result<()> {
    if query.len() == dimensions {
        return Ok(());
    }
    Err(Error::Query(QueryError::new(
        QueryErrorKind::Semantic,
        format!(
            "Argument 'query' of {procedure} has {} dimensions, the index has {dimensions}",
            query.len()
        ),
    )))
}

/// Converts a `k` parameter (signed, user-supplied) to an unsigned limit.
/// Negative values clamp to 0 so they can't silently flip via cast.
#[cfg(all(feature = "lpg", any(feature = "vector-index", feature = "text-index")))]
fn k_limit(params: &Parameters, default: i64) -> usize {
    usize::try_from(params.get_int("k").unwrap_or(default).max(0)).unwrap_or(0)
}

/// Converts a NodeId into a `Value::Int64`.
/// Using `cast_signed` keeps the bit pattern; ids above `i64::MAX` would
/// become negative, but the ID space is bounded well below that in practice.
#[cfg(all(feature = "lpg", any(feature = "vector-index", feature = "text-index")))]
fn node_id_to_value(node_id: grafeo_common::types::NodeId) -> Value {
    Value::Int64(node_id.as_u64().cast_signed())
}

/// `grafeo.search.vector(label, property, query, k)`:
/// k-nearest-neighbors via the HNSW index.
#[cfg(all(feature = "lpg", feature = "vector-index"))]
struct SearchVectorProcedure {
    parameters: Vec<ParameterDef>,
}

#[cfg(all(feature = "lpg", feature = "vector-index"))]
impl SearchVectorProcedure {
    fn new() -> Self {
        use grafeo_adapters::plugins::ParameterType;
        Self {
            parameters: vec![
                ParameterDef {
                    name: "label".into(),
                    param_type: ParameterType::String,
                    required: true,
                    default: None,
                    description: "Node label that was vector-indexed".into(),
                },
                ParameterDef {
                    name: "property".into(),
                    param_type: ParameterType::String,
                    required: true,
                    default: None,
                    description: "Property name holding the embedding".into(),
                },
                ParameterDef {
                    name: "query".into(),
                    param_type: ParameterType::List,
                    required: true,
                    default: None,
                    description: "Query vector as a list of floats".into(),
                },
                ParameterDef {
                    name: "k".into(),
                    param_type: ParameterType::Integer,
                    required: false,
                    default: Some("10".into()),
                    description: "Number of nearest neighbors to return".into(),
                },
            ],
        }
    }
}

#[cfg(all(feature = "lpg", feature = "vector-index"))]
impl Procedure for SearchVectorProcedure {
    fn name(&self) -> &str {
        "search.vector"
    }

    fn description(&self) -> &str {
        "Approximate k-nearest neighbors via the HNSW vector index"
    }

    fn parameters(&self) -> &[ParameterDef] {
        &self.parameters
    }

    fn output_columns(&self) -> Vec<String> {
        vec!["node_id".into(), "distance".into()]
    }

    fn execute(&self, ctx: &ProcedureContext<'_>, params: &Parameters) -> Result<AlgorithmResult> {
        use grafeo_core::index::vector::{PropertyVectorAccessor, VectorAccessorKind};

        let lpg = require_lpg_store(ctx, "CALL grafeo.search.vector")?;
        let label = params.get_string("label").ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(
                "CALL grafeo.search.vector: missing required parameter 'label'".into(),
            )
        })?;
        let property = params.get_string("property").ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(
                "CALL grafeo.search.vector: missing required parameter 'property'".into(),
            )
        })?;
        let query = coerce_params_to_vector(params, "query")?;
        let k = k_limit(params, 10);

        let index = lpg.get_vector_index(label, property).ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(format!(
                "No vector index found for :{label}({property}). Call CREATE VECTOR INDEX first."
            ))
        })?;
        check_query_dimensions("grafeo.search.vector", &query, index.config().dimensions)?;

        let accessor = VectorAccessorKind::Property(PropertyVectorAccessor::new(
            ctx.store as &dyn grafeo_core::graph::GraphStore,
            property,
        ));
        // The index may hold nodes that are gone: see `live_index_hits`.
        let results = lpg.live_index_hits(k, |fetch| index.search(&query, fetch, &accessor));

        let mut result = AlgorithmResult::new(vec!["node_id".into(), "distance".into()]);
        for (node_id, distance) in results {
            result.rows.push(vec![
                node_id_to_value(node_id),
                Value::Float64(f64::from(distance)),
            ]);
        }
        Ok(result)
    }
}

/// `grafeo.search.mmr(label, property, query, k, fetch_k, lambda)`:
/// Maximal-Marginal-Relevance re-ranking of HNSW candidates for diverse top-k.
#[cfg(all(feature = "lpg", feature = "vector-index"))]
struct SearchMmrProcedure {
    parameters: Vec<ParameterDef>,
}

#[cfg(all(feature = "lpg", feature = "vector-index"))]
impl SearchMmrProcedure {
    fn new() -> Self {
        use grafeo_adapters::plugins::ParameterType;
        Self {
            parameters: vec![
                ParameterDef {
                    name: "label".into(),
                    param_type: ParameterType::String,
                    required: true,
                    default: None,
                    description: "Node label that was vector-indexed".into(),
                },
                ParameterDef {
                    name: "property".into(),
                    param_type: ParameterType::String,
                    required: true,
                    default: None,
                    description: "Property name holding the embedding".into(),
                },
                ParameterDef {
                    name: "query".into(),
                    param_type: ParameterType::List,
                    required: true,
                    default: None,
                    description: "Query vector as a list of floats".into(),
                },
                ParameterDef {
                    name: "k".into(),
                    param_type: ParameterType::Integer,
                    required: false,
                    default: Some("10".into()),
                    description: "Number of diverse results to return".into(),
                },
                ParameterDef {
                    name: "fetch_k".into(),
                    param_type: ParameterType::Integer,
                    required: false,
                    default: None,
                    description: "Initial HNSW candidate count (default: 4*k)".into(),
                },
                ParameterDef {
                    name: "lambda".into(),
                    param_type: ParameterType::Float,
                    required: false,
                    default: Some("0.5".into()),
                    description: "Relevance vs diversity in [0, 1]".into(),
                },
            ],
        }
    }
}

#[cfg(all(feature = "lpg", feature = "vector-index"))]
impl Procedure for SearchMmrProcedure {
    fn name(&self) -> &str {
        "search.mmr"
    }

    fn description(&self) -> &str {
        "Maximal-Marginal-Relevance re-ranking for diverse nearest neighbors"
    }

    fn parameters(&self) -> &[ParameterDef] {
        &self.parameters
    }

    fn output_columns(&self) -> Vec<String> {
        vec!["node_id".into(), "distance".into()]
    }

    fn execute(&self, ctx: &ProcedureContext<'_>, params: &Parameters) -> Result<AlgorithmResult> {
        use grafeo_core::index::vector::{
            PropertyVectorAccessor, VectorAccessor, VectorAccessorKind, mmr_select,
        };

        let lpg = require_lpg_store(ctx, "CALL grafeo.search.mmr")?;
        let label = params.get_string("label").ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(
                "CALL grafeo.search.mmr: missing required parameter 'label'".into(),
            )
        })?;
        let property = params.get_string("property").ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(
                "CALL grafeo.search.mmr: missing required parameter 'property'".into(),
            )
        })?;
        let query = coerce_params_to_vector(params, "query")?;
        let k = k_limit(params, 10);
        let fetch_k = params
            .get_int("fetch_k")
            .map_or(k.saturating_mul(4).max(k), |v| {
                usize::try_from(v.max(0)).unwrap_or(0)
            });
        #[allow(
            clippy::cast_possible_truncation,
            reason = "lambda is a unit-interval weighting factor; f32 precision is sufficient"
        )]
        let lambda = params.get_float("lambda").unwrap_or(0.5) as f32;

        let index = lpg.get_vector_index(label, property).ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(format!(
                "No vector index found for :{label}({property}). Call CREATE VECTOR INDEX first."
            ))
        })?;
        check_query_dimensions("grafeo.search.mmr", &query, index.config().dimensions)?;

        let accessor = VectorAccessorKind::Property(PropertyVectorAccessor::new(
            ctx.store as &dyn grafeo_core::graph::GraphStore,
            property,
        ));
        // The index may hold nodes that are gone: see `live_index_hits`.
        let initial = lpg.live_index_hits(fetch_k, |fetch| index.search(&query, fetch, &accessor));
        if initial.is_empty() {
            return Ok(AlgorithmResult::new(vec![
                "node_id".into(),
                "distance".into(),
            ]));
        }

        let candidates: Vec<(grafeo_common::types::NodeId, f32, std::sync::Arc<[f32]>)> = initial
            .into_iter()
            .filter_map(|(id, dist)| accessor.get_vector(id).map(|v| (id, dist, v)))
            .collect();
        let candidate_refs: Vec<(grafeo_common::types::NodeId, f32, &[f32])> = candidates
            .iter()
            .map(|(id, dist, vec)| (*id, *dist, vec.as_ref()))
            .collect();

        let metric = index.config().metric;
        let selected = mmr_select(&query, &candidate_refs, k, lambda, metric);

        let mut result = AlgorithmResult::new(vec!["node_id".into(), "distance".into()]);
        for (node_id, distance) in selected {
            result.rows.push(vec![
                node_id_to_value(node_id),
                Value::Float64(f64::from(distance)),
            ]);
        }
        Ok(result)
    }
}

/// `grafeo.search.text(label, property, query, k)`:
/// BM25 full-text search over the inverted index.
#[cfg(all(feature = "lpg", feature = "text-index"))]
struct SearchTextProcedure {
    parameters: Vec<ParameterDef>,
}

#[cfg(all(feature = "lpg", feature = "text-index"))]
impl SearchTextProcedure {
    fn new() -> Self {
        use grafeo_adapters::plugins::ParameterType;
        Self {
            parameters: vec![
                ParameterDef {
                    name: "label".into(),
                    param_type: ParameterType::String,
                    required: true,
                    default: None,
                    description: "Node label that was text-indexed".into(),
                },
                ParameterDef {
                    name: "property".into(),
                    param_type: ParameterType::String,
                    required: true,
                    default: None,
                    description: "Property holding the indexed text".into(),
                },
                ParameterDef {
                    name: "query".into(),
                    param_type: ParameterType::String,
                    required: true,
                    default: None,
                    description: "Text query for BM25 scoring".into(),
                },
                ParameterDef {
                    name: "k".into(),
                    param_type: ParameterType::Integer,
                    required: false,
                    default: Some("10".into()),
                    description: "Number of top results to return".into(),
                },
            ],
        }
    }
}

#[cfg(all(feature = "lpg", feature = "text-index"))]
impl Procedure for SearchTextProcedure {
    fn name(&self) -> &str {
        "search.text"
    }

    fn description(&self) -> &str {
        "BM25 full-text search over an inverted text index"
    }

    fn parameters(&self) -> &[ParameterDef] {
        &self.parameters
    }

    fn output_columns(&self) -> Vec<String> {
        vec!["node_id".into(), "score".into()]
    }

    fn execute(&self, ctx: &ProcedureContext<'_>, params: &Parameters) -> Result<AlgorithmResult> {
        let lpg = ctx.lpg_store.ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(
                "CALL grafeo.search.text requires an LPG store".into(),
            )
        })?;
        let label = params.get_string("label").ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(
                "CALL grafeo.search.text: missing required parameter 'label'".into(),
            )
        })?;
        let property = params.get_string("property").ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(
                "CALL grafeo.search.text: missing required parameter 'property'".into(),
            )
        })?;
        let query = params.get_string("query").ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(
                "CALL grafeo.search.text: missing required parameter 'query'".into(),
            )
        })?;
        let k = k_limit(params, 10);

        let index = lpg.get_text_index(label, property).ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(format!(
                "No text index found for :{label}({property}). Call CREATE TEXT INDEX first."
            ))
        })?;

        // The index may hold nodes that are gone: see `live_index_hits`.
        let results = lpg.live_index_hits(k, |fetch| index.read().search(query, fetch));

        let mut result = AlgorithmResult::new(vec!["node_id".into(), "score".into()]);
        for (node_id, score) in results {
            result
                .rows
                .push(vec![node_id_to_value(node_id), Value::Float64(score)]);
        }
        Ok(result)
    }
}

/// Registry of built-in procedures.
///
/// Stores procedures behind `Arc<dyn Procedure>` so dispatch goes through a
/// single path regardless of procedure kind (graph algorithm, catalog, or
/// search).
pub struct BuiltinProcedures {
    procedures: HashMap<String, Arc<dyn Procedure>>,
}

impl BuiltinProcedures {
    /// Creates a new registry with all built-in procedures registered.
    #[must_use]
    pub fn new() -> Self {
        let mut procedures: HashMap<String, Arc<dyn Procedure>> = HashMap::new();

        // Graph algorithms, wrapped via the adapter.
        let mut register_algo = |algo: Arc<dyn GraphAlgorithm>| {
            let proc = Arc::new(GraphAlgorithmProcedure::new(algo));
            procedures.insert(proc.name().to_string(), proc);
        };

        // Centrality
        register_algo(Arc::new(PageRankAlgorithm));
        register_algo(Arc::new(BetweennessCentralityAlgorithm));
        register_algo(Arc::new(ClosenessCentralityAlgorithm));
        register_algo(Arc::new(DegreeCentralityAlgorithm));

        // Traversal
        register_algo(Arc::new(BfsAlgorithm));
        register_algo(Arc::new(DfsAlgorithm));

        // Components
        register_algo(Arc::new(ConnectedComponentsAlgorithm));
        register_algo(Arc::new(StronglyConnectedComponentsAlgorithm));
        register_algo(Arc::new(TopologicalSortAlgorithm));

        // Shortest Path
        register_algo(Arc::new(DijkstraAlgorithm));
        register_algo(Arc::new(SsspAlgorithm));
        register_algo(Arc::new(BellmanFordAlgorithm));
        register_algo(Arc::new(FloydWarshallAlgorithm));

        // Clustering
        register_algo(Arc::new(ClusteringCoefficientAlgorithm));

        // Community
        register_algo(Arc::new(LabelPropagationAlgorithm));
        register_algo(Arc::new(LouvainAlgorithm));

        // MST
        register_algo(Arc::new(KruskalAlgorithm));
        register_algo(Arc::new(PrimAlgorithm));

        // Flow
        register_algo(Arc::new(MaxFlowAlgorithm));
        register_algo(Arc::new(MinCostFlowAlgorithm));

        // Structure
        register_algo(Arc::new(ArticulationPointsAlgorithm));
        register_algo(Arc::new(BridgesAlgorithm));
        register_algo(Arc::new(KCoreAlgorithm));

        // Catalog introspection
        let mut register = |proc: Arc<dyn Procedure>| {
            procedures.insert(proc.name().to_string(), proc);
        };
        register(Arc::new(LabelsProcedure));
        register(Arc::new(RelationshipTypesProcedure));
        register(Arc::new(PropertyKeysProcedure));

        // Search procedures (feature-gated).
        #[cfg(all(feature = "lpg", feature = "vector-index"))]
        {
            register(Arc::new(SearchVectorProcedure::new()));
            register(Arc::new(SearchMmrProcedure::new()));
        }
        #[cfg(all(feature = "lpg", feature = "text-index"))]
        {
            register(Arc::new(SearchTextProcedure::new()));
        }

        Self { procedures }
    }

    /// Resolves a dotted procedure name to its implementation.
    #[must_use]
    pub fn get(&self, name: &[String]) -> Option<Arc<dyn Procedure>> {
        let key = resolve_name(name);
        self.procedures.get(&key).cloned()
    }

    /// Returns metadata for every registered procedure, sorted by name.
    #[must_use]
    pub fn list(&self) -> Vec<ProcedureInfo> {
        let mut result: Vec<ProcedureInfo> = self
            .procedures
            .values()
            .map(|proc| ProcedureInfo {
                name: format!("grafeo.{}", proc.name()),
                description: proc.description().to_string(),
                parameters: proc.parameters().to_vec(),
                output_columns: proc.output_columns(),
            })
            .collect();
        result.sort_by(|a, b| a.name.cmp(&b.name));
        result
    }
}

impl Default for BuiltinProcedures {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata about a registered procedure.
pub struct ProcedureInfo {
    /// Qualified name (e.g., `"grafeo.pagerank"`).
    pub name: String,
    /// Description of what the procedure does.
    pub description: String,
    /// Parameter definitions.
    pub parameters: Vec<ParameterDef>,
    /// Output column names.
    pub output_columns: Vec<String>,
}

/// Resolves a dotted procedure name to its lookup key.
///
/// Strips a leading `"grafeo"` or `"db"` namespace segment if present, then
/// joins the remaining parts with `.`. This lets every form resolve to the
/// same registered key:
///
/// - `CALL pagerank()` / `CALL grafeo.pagerank()` → `"pagerank"`
/// - `CALL db.labels()` / `CALL grafeo.labels()` → `"labels"`
/// - `CALL grafeo.search.vector(...)` → `"search.vector"`
fn resolve_name(parts: &[String]) -> String {
    let slice = match parts {
        [ns, rest @ ..]
            if (ns.eq_ignore_ascii_case("grafeo") || ns.eq_ignore_ascii_case("db"))
                && !rest.is_empty() =>
        {
            rest
        }
        _ => parts,
    };
    slice.join(".")
}

/// Returns canonical output column names for the given graph algorithm.
///
/// These user-facing names (e.g., `"score"` instead of `"pagerank"`) must
/// match the column count produced by each algorithm's `execute()`.
#[must_use]
pub fn canonical_output_columns(algo: &dyn GraphAlgorithm) -> Vec<String> {
    match algo.name() {
        "pagerank" => vec!["node_id".into(), "score".into()],
        "betweenness_centrality" | "closeness_centrality" => {
            vec!["node_id".into(), "centrality".into()]
        }
        "degree_centrality" => vec![
            "node_id".into(),
            "in_degree".into(),
            "out_degree".into(),
            "total_degree".into(),
        ],
        "bfs" => vec!["node_id".into(), "depth".into()],
        "dfs" => vec![
            "node_id".into(),
            "depth".into(),
            "discovery".into(),
            "finish".into(),
        ],
        "connected_components" | "strongly_connected_components" => {
            vec!["node_id".into(), "component_id".into()]
        }
        "topological_sort" => vec!["node_id".into(), "order".into()],
        "dijkstra" | "sssp" => vec!["node_id".into(), "distance".into()],
        "bellman_ford" => vec![
            "node_id".into(),
            "distance".into(),
            "has_negative_cycle".into(),
        ],
        "floyd_warshall" => vec!["source".into(), "target".into(), "distance".into()],
        "clustering_coefficient" => vec![
            "node_id".into(),
            "coefficient".into(),
            "triangle_count".into(),
        ],
        "label_propagation" => vec!["node_id".into(), "community_id".into()],
        "louvain" => vec!["node_id".into(), "community_id".into(), "modularity".into()],
        "kruskal" | "prim" => vec!["source".into(), "target".into(), "weight".into()],
        "max_flow" => vec![
            "source".into(),
            "target".into(),
            "flow".into(),
            "max_flow".into(),
        ],
        "min_cost_max_flow" => vec![
            "source".into(),
            "target".into(),
            "flow".into(),
            "cost".into(),
            "max_flow".into(),
        ],
        "articulation_points" => vec!["node_id".into()],
        "bridges" => vec!["source".into(), "target".into()],
        "kcore" => vec!["node_id".into(), "core_number".into(), "max_core".into()],
        _ => vec!["node_id".into(), "value".into()],
    }
}

/// The name an error gives positional argument `index` of a call: its
/// parameter's name, or its position past the last parameter.
#[must_use]
pub fn argument_name(param_defs: &[ParameterDef], index: usize) -> String {
    param_defs
        .get(index)
        .map_or_else(|| format!("#{}", index + 1), |def| def.name.clone())
}

/// Converts the values of a procedure call's arguments into [`Parameters`].
///
/// Two forms:
/// 1. One map, `{damping: 0.85, max_iterations: 20}` (a map literal or a map
///    parameter), names the parameters.
/// 2. Positional values, `(42, 'weight')`, fill the parameters in the order
///    of `param_defs`; values past the last parameter are ignored.
///
/// Each value must suit its parameter's type: an integer passes for a float
/// and a vector for a list. A null leaves the parameter out, so an optional
/// one keeps its default and a required one fails as missing when the
/// procedure runs. A map key that names no parameter (such as `projection`)
/// passes through with the type of its value.
///
/// # Errors
///
/// Returns a semantic error naming the argument and `procedure` when a value
/// does not suit its parameter: such a value never falls back to the
/// parameter's default.
pub fn evaluate_arguments(
    procedure: &str,
    args: &[Value],
    param_defs: &[ParameterDef],
) -> Result<Parameters> {
    let mut params = Parameters::new();

    if let [Value::Map(entries)] = args {
        for (key, value) in entries.iter() {
            let def = param_defs.iter().find(|def| def.name == key.as_str());
            set_parameter(
                &mut params,
                procedure,
                key.as_str(),
                def.map(|def| def.param_type),
                value,
            )?;
        }
        return Ok(params);
    }

    for (def, value) in param_defs.iter().zip(args) {
        set_parameter(
            &mut params,
            procedure,
            &def.name,
            Some(def.param_type),
            value,
        )?;
    }

    Ok(params)
}

/// Sets parameter `name` to `value`, which must suit `expected` (any value a
/// parameter can hold when `None`).
fn set_parameter(
    params: &mut Parameters,
    procedure: &str,
    name: &str,
    expected: Option<ParameterType>,
    value: &Value,
) -> Result<()> {
    use ParameterType as Type;
    match (expected, value) {
        (_, Value::Null) => {}
        (Some(Type::Integer | Type::NodeId) | None, Value::Int64(v)) => params.set_int(name, *v),
        (Some(Type::Float), Value::Int64(v)) => params.set_float(name, *v as f64),
        (Some(Type::Float) | None, Value::Float64(v)) => params.set_float(name, *v),
        (Some(Type::String) | None, Value::String(v)) => params.set_string(name, v.as_str()),
        (Some(Type::Boolean) | None, Value::Bool(v)) => params.set_bool(name, *v),
        (Some(Type::List) | None, Value::List(items)) => params.set_list(name, items.to_vec()),
        (Some(Type::List) | None, Value::Vector(items)) => params.set_list(
            name,
            items
                .iter()
                .map(|f| Value::Float64(f64::from(*f)))
                .collect(),
        ),
        (expected, value) => {
            let wanted = match expected {
                Some(Type::Integer) => "an integer",
                Some(Type::NodeId) => "a node id (an integer)",
                Some(Type::Float) => "a number",
                Some(Type::String) => "a string",
                Some(Type::Boolean) => "a boolean",
                Some(Type::List) => "a list",
                None => "a number, string, boolean or list",
            };
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!(
                    "Argument '{name}' of {procedure} must be {wanted}, got a {} value",
                    value.type_name()
                ),
            )));
        }
    }
    Ok(())
}

/// Builds a `grafeo.procedures()` result listing all registered procedures.
#[must_use]
pub fn procedures_result(registry: &BuiltinProcedures) -> AlgorithmResult {
    let procedures = registry.list();
    let mut result = AlgorithmResult::new(vec![
        "name".into(),
        "description".into(),
        "parameters".into(),
        "output_columns".into(),
    ]);
    for proc in procedures {
        let param_desc: String = proc
            .parameters
            .iter()
            .map(|p| {
                if p.required {
                    format!("{} ({:?})", p.name, p.param_type)
                } else if let Some(ref default) = p.default {
                    format!("{} ({:?}, default={})", p.name, p.param_type, default)
                } else {
                    format!("{} ({:?}, optional)", p.name, p.param_type)
                }
            })
            .collect::<Vec<_>>()
            .join(", ");

        let columns_desc = proc.output_columns.join(", ");

        result.add_row(vec![
            Value::from(proc.name.as_str()),
            Value::from(proc.description.as_str()),
            Value::from(param_desc.as_str()),
            Value::from(columns_desc.as_str()),
        ]);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_has_all_algorithms() {
        let registry = BuiltinProcedures::new();
        let list = registry.list();
        assert!(
            list.len() >= 22,
            "Expected at least 22 procedures, got {}",
            list.len()
        );
    }

    #[test]
    fn test_resolve_with_namespace() {
        let registry = BuiltinProcedures::new();
        let name = vec!["grafeo".to_string(), "pagerank".to_string()];
        assert!(registry.get(&name).is_some());
    }

    #[test]
    fn test_resolve_without_namespace() {
        let registry = BuiltinProcedures::new();
        let name = vec!["pagerank".to_string()];
        assert!(registry.get(&name).is_some());
    }

    #[test]
    fn test_resolve_unknown() {
        let registry = BuiltinProcedures::new();
        let name = vec!["grafeo".to_string(), "nonexistent".to_string()];
        assert!(registry.get(&name).is_none());
    }

    fn map(entries: &[(&str, Value)]) -> Value {
        Value::Map(Arc::new(
            entries
                .iter()
                .map(|(key, value)| ((*key).into(), value.clone()))
                .collect(),
        ))
    }

    fn def(name: &str, param_type: ParameterType, required: bool) -> ParameterDef {
        ParameterDef {
            name: name.into(),
            param_type,
            required,
            default: None,
            description: String::new(),
        }
    }

    #[test]
    fn test_evaluate_map_arguments() {
        let args = vec![map(&[
            ("damping", Value::Float64(0.85)),
            ("max_iterations", Value::Int64(20)),
        ])];
        let params = evaluate_arguments("pagerank", &args, &[]).unwrap();
        assert_eq!(params.get_float("damping"), Some(0.85));
        assert_eq!(params.get_int("max_iterations"), Some(20));
    }

    #[test]
    fn test_evaluate_empty_arguments() {
        let params = evaluate_arguments("pagerank", &[], &[]).unwrap();
        assert_eq!(params.get_float("damping"), None);
    }

    #[test]
    fn an_integer_argument_fills_a_float_parameter() {
        let defs = [def("damping", ParameterType::Float, false)];
        let positional = evaluate_arguments("pagerank", &[Value::Int64(1)], &defs).unwrap();
        assert_eq!(positional.get_float("damping"), Some(1.0));
        let named =
            evaluate_arguments("pagerank", &[map(&[("damping", Value::Int64(1))])], &defs).unwrap();
        assert_eq!(named.get_float("damping"), Some(1.0));
    }

    #[test]
    fn an_argument_of_the_wrong_type_is_an_error_not_the_default() {
        let defs = [
            def("damping", ParameterType::Float, false),
            def("max_iterations", ParameterType::Integer, false),
            def("directed", ParameterType::Boolean, false),
        ];
        for (args, argument) in [
            (vec![Value::from("high")], "damping"),
            (
                vec![Value::Float64(0.5), Value::Float64(2.5)],
                "max_iterations",
            ),
            (
                vec![Value::Float64(0.5), Value::Int64(3), Value::Int64(1)],
                "directed",
            ),
            (vec![map(&[("damping", Value::Bool(true))])], "damping"),
        ] {
            let Err(error) = evaluate_arguments("grafeo.pagerank", &args, &defs) else {
                panic!("a value of the wrong type for '{argument}' must fail");
            };
            let message = error.to_string();
            assert!(
                message.contains(&format!("Argument '{argument}' of grafeo.pagerank")),
                "{message}"
            );
        }
    }

    #[test]
    fn a_null_argument_leaves_the_parameter_out() {
        let defs = [
            def("k", ParameterType::Integer, false),
            def("fetch_k", ParameterType::Integer, false),
        ];
        let params =
            evaluate_arguments("search.mmr", &[Value::Int64(3), Value::Null], &defs).unwrap();
        assert_eq!(params.get_int("k"), Some(3));
        assert_eq!(params.get_int("fetch_k"), None);
    }

    #[test]
    fn a_map_key_without_a_parameter_passes_through() {
        let params = evaluate_arguments(
            "pagerank",
            &[map(&[("projection", Value::from("people"))])],
            &[def("damping", ParameterType::Float, false)],
        )
        .unwrap();
        assert_eq!(params.get_string("projection"), Some("people"));
        let Err(error) = evaluate_arguments("pagerank", &[map(&[("projection", map(&[]))])], &[])
        else {
            panic!("no parameter holds a map");
        };
        assert!(error.to_string().contains("'projection'"), "{error}");
    }

    #[test]
    fn test_adapter_forwards_metadata() {
        let algo: Arc<dyn GraphAlgorithm> = Arc::new(PageRankAlgorithm);
        let proc = GraphAlgorithmProcedure::new(algo);
        assert_eq!(proc.name(), "pagerank");
        assert_eq!(proc.output_columns(), vec!["node_id", "score"]);
        assert!(!proc.parameters().is_empty());
    }

    // ---------------------------------------------------------------
    // resolve_name: every call site of CALL must dispatch through here,
    // so the namespace-stripping contract is load-bearing.
    // ---------------------------------------------------------------

    #[test]
    fn resolve_name_strips_db_prefix() {
        let registry = BuiltinProcedures::new();
        let name = vec!["db".to_string(), "labels".to_string()];
        assert!(
            registry.get(&name).is_some(),
            "db.labels must route to the labels procedure"
        );
    }

    #[test]
    fn resolve_name_is_case_insensitive_on_prefix() {
        // `CALL DB.Labels()` should route the same as `CALL db.labels()`.
        // (The leaf name itself is case-sensitive by design; we only
        // normalise the namespace token.)
        assert_eq!(
            resolve_name(&["DB".to_string(), "labels".to_string()]),
            "labels"
        );
        assert_eq!(
            resolve_name(&["Grafeo".to_string(), "pagerank".to_string()]),
            "pagerank"
        );
    }

    #[test]
    fn resolve_name_preserves_dotted_paths() {
        // Search procedures live under a dotted sub-namespace
        // (grafeo.search.vector). The resolver must NOT strip the
        // second segment just because the first is "grafeo".
        assert_eq!(
            resolve_name(&[
                "grafeo".to_string(),
                "search".to_string(),
                "vector".to_string()
            ]),
            "search.vector"
        );
    }

    #[test]
    fn resolve_name_leaves_bare_prefix_alone() {
        // `CALL grafeo()` has no child: do not strip, or the resolver
        // would produce the empty string and silently collide with any
        // procedure registered under "".
        assert_eq!(resolve_name(&["grafeo".to_string()]), "grafeo");
        assert_eq!(resolve_name(&["db".to_string()]), "db");
    }

    #[test]
    fn resolve_name_empty_input_returns_empty() {
        assert_eq!(resolve_name(&[]), "");
    }

    // ---------------------------------------------------------------
    // canonical_output_columns: the planner relies on every wrapped
    // algorithm reporting stable, user-facing column names. A rename
    // here silently breaks RETURN / YIELD lists.
    // ---------------------------------------------------------------

    #[test]
    fn canonical_output_columns_for_shortest_path_algorithms() {
        let dijkstra: Arc<dyn GraphAlgorithm> = Arc::new(DijkstraAlgorithm);
        assert_eq!(
            canonical_output_columns(dijkstra.as_ref()),
            vec!["node_id", "distance"]
        );

        let sssp: Arc<dyn GraphAlgorithm> = Arc::new(SsspAlgorithm);
        assert_eq!(
            canonical_output_columns(sssp.as_ref()),
            vec!["node_id", "distance"]
        );

        let bellman: Arc<dyn GraphAlgorithm> = Arc::new(BellmanFordAlgorithm);
        assert_eq!(
            canonical_output_columns(bellman.as_ref()),
            vec!["node_id", "distance", "has_negative_cycle"]
        );
    }

    #[test]
    fn canonical_output_columns_for_community_algorithms() {
        let louvain: Arc<dyn GraphAlgorithm> = Arc::new(LouvainAlgorithm);
        assert_eq!(
            canonical_output_columns(louvain.as_ref()),
            vec!["node_id", "community_id", "modularity"]
        );

        let lpa: Arc<dyn GraphAlgorithm> = Arc::new(LabelPropagationAlgorithm);
        assert_eq!(
            canonical_output_columns(lpa.as_ref()),
            vec!["node_id", "community_id"]
        );
    }

    #[test]
    fn canonical_output_columns_for_mst_algorithms() {
        let kruskal: Arc<dyn GraphAlgorithm> = Arc::new(KruskalAlgorithm);
        assert_eq!(
            canonical_output_columns(kruskal.as_ref()),
            vec!["source", "target", "weight"]
        );
        let prim: Arc<dyn GraphAlgorithm> = Arc::new(PrimAlgorithm);
        assert_eq!(
            canonical_output_columns(prim.as_ref()),
            vec!["source", "target", "weight"]
        );
    }

    #[test]
    fn canonical_output_columns_for_structural_algorithms() {
        let ap: Arc<dyn GraphAlgorithm> = Arc::new(ArticulationPointsAlgorithm);
        assert_eq!(canonical_output_columns(ap.as_ref()), vec!["node_id"]);

        let bridges: Arc<dyn GraphAlgorithm> = Arc::new(BridgesAlgorithm);
        assert_eq!(
            canonical_output_columns(bridges.as_ref()),
            vec!["source", "target"]
        );

        let kcore: Arc<dyn GraphAlgorithm> = Arc::new(KCoreAlgorithm);
        assert_eq!(
            canonical_output_columns(kcore.as_ref()),
            vec!["node_id", "core_number", "max_core"]
        );
    }

    /// A stand-in GraphAlgorithm whose name is not in the canonical list,
    /// used to verify the fallback branch.
    struct UnknownAlgo;
    impl GraphAlgorithm for UnknownAlgo {
        fn name(&self) -> &str {
            "totally_new_algorithm"
        }
        fn description(&self) -> &str {
            "unknown for test"
        }
        fn parameters(&self) -> &[ParameterDef] {
            &[]
        }
        fn execute(
            &self,
            _store: &dyn grafeo_core::graph::GraphStore,
            _params: &Parameters,
        ) -> Result<AlgorithmResult> {
            Ok(AlgorithmResult::new(vec!["node_id".into(), "value".into()]))
        }
    }

    #[test]
    fn canonical_output_columns_falls_back_to_node_id_value() {
        let unknown: Arc<dyn GraphAlgorithm> = Arc::new(UnknownAlgo);
        assert_eq!(
            canonical_output_columns(unknown.as_ref()),
            vec!["node_id", "value"],
            "unknown algorithms must fall back to (node_id, value); otherwise the \
             planner would emit an empty header row"
        );
    }

    // ---------------------------------------------------------------
    // evaluate_arguments: the positional form and each value type.
    // ---------------------------------------------------------------

    #[test]
    fn evaluate_positional_arguments_mapped_by_param_defs() {
        let defs = [
            def("damping", ParameterType::Float, true),
            def("iterations", ParameterType::Integer, true),
        ];
        let args = [Value::Float64(0.9), Value::Int64(50)];
        let params = evaluate_arguments("pagerank", &args, &defs).unwrap();
        assert_eq!(params.get_float("damping"), Some(0.9));
        assert_eq!(params.get_int("iterations"), Some(50));
    }

    #[test]
    fn evaluate_arguments_drops_positional_past_last_param_def() {
        // An extra positional value past the end of `param_defs` is ignored
        // (no panic), the "extra args ignored" contract of procedure calls.
        let defs = [def("only", ParameterType::Integer, true)];
        let args = [Value::Int64(1), Value::Int64(2)];
        let params = evaluate_arguments("only", &args, &defs).unwrap();
        assert_eq!(params.get_int("only"), Some(1));
    }

    #[test]
    fn evaluate_arguments_bool_and_string_values() {
        let defs = [
            def("flag", ParameterType::Boolean, true),
            def("label", ParameterType::String, true),
        ];
        let args = [Value::Bool(true), Value::String("Person".into())];
        let params = evaluate_arguments("flags", &args, &defs).unwrap();
        assert_eq!(params.get_bool("flag"), Some(true));
        assert_eq!(params.get_string("label"), Some("Person"));
    }

    #[test]
    fn evaluate_arguments_vector_becomes_list_of_floats() {
        // A vector (a vector literal or a vector parameter) reaches a list
        // parameter as a List of Float64, so `coerce_params_to_vector` reads
        // it like a typed `[0.1, 0.2, ...]` list.
        let defs = [def("query", ParameterType::List, true)];
        let vector: Arc<[f32]> = Arc::from(vec![0.25_f32, 0.5_f32, 0.75_f32]);
        let params = evaluate_arguments("search.vector", &[Value::Vector(vector)], &defs).unwrap();

        let list = params.get_list("query").expect("query must be a list");
        assert_eq!(
            list,
            &[
                Value::Float64(0.25),
                Value::Float64(0.5),
                Value::Float64(0.75)
            ]
        );
    }

    #[test]
    fn evaluate_arguments_list_value() {
        let defs = [def("items", ParameterType::List, true)];
        let args = [Value::List(vec![Value::Int64(7), Value::Int64(8)].into())];
        let params = evaluate_arguments("items", &args, &defs).unwrap();
        let list = params.get_list("items").unwrap();
        assert_eq!(list, &[Value::Int64(7), Value::Int64(8)]);
    }

    // ---------------------------------------------------------------
    // procedures_result: the user-facing format of `CALL grafeo.procedures()`.
    // Changing the column order or the parameter-description shape is a
    // breaking change for scripts that parse this.
    // ---------------------------------------------------------------

    #[test]
    fn procedures_result_has_stable_column_shape() {
        let registry = BuiltinProcedures::new();
        let result = procedures_result(&registry);
        assert_eq!(
            result.columns,
            vec!["name", "description", "parameters", "output_columns"],
            "column shape is part of the public CALL grafeo.procedures() contract"
        );
        assert!(!result.rows.is_empty(), "result.rows is empty");
    }

    #[test]
    fn procedures_result_rows_are_sorted_by_name() {
        let registry = BuiltinProcedures::new();
        let result = procedures_result(&registry);
        let names: Vec<String> = result
            .rows
            .iter()
            .map(|row| match &row[0] {
                Value::String(s) => s.as_str().to_string(),
                other => panic!("name column must be String, got {other:?}"),
            })
            .collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
    }

    #[test]
    fn procedures_result_names_are_fully_qualified() {
        let registry = BuiltinProcedures::new();
        let result = procedures_result(&registry);
        for row in &result.rows {
            let name = match &row[0] {
                Value::String(s) => s.as_str(),
                _ => panic!(),
            };
            assert!(
                name.starts_with("grafeo."),
                "every listed procedure name must be prefixed with 'grafeo.', got {name:?}"
            );
        }
    }

    #[test]
    fn procedures_result_parameter_description_distinguishes_required_and_default() {
        let registry = BuiltinProcedures::new();
        let result = procedures_result(&registry);
        // pagerank has required and defaulted params; find its row.
        let pagerank_row = result
            .rows
            .iter()
            .find(|row| matches!(&row[0], Value::String(s) if s.as_str() == "grafeo.pagerank"))
            .expect("pagerank must appear in the listing");
        let params_desc = match &pagerank_row[2] {
            Value::String(s) => s.as_str(),
            _ => panic!("parameters column must be String"),
        };
        // The exact param shape depends on PageRankAlgorithm's definition,
        // but the formatter contract is: defaulted params include `default=...`,
        // required params do not. At least one of each should appear in the
        // overall procedures listing.
        let all_desc: String = result
            .rows
            .iter()
            .map(|row| match &row[2] {
                Value::String(s) => s.as_str(),
                _ => "",
            })
            .collect::<Vec<_>>()
            .join(" | ");
        assert!(
            all_desc.contains("default="),
            "at least one procedure must report a parameter default, got: {all_desc}"
        );
        // Spot-check that pagerank's row has at least one parameter listed.
        assert!(
            !params_desc.is_empty(),
            "pagerank must expose at least one parameter in the description, got empty string"
        );
    }

    // ---------------------------------------------------------------
    // Feature-gated helpers (search procedures).
    // ---------------------------------------------------------------

    #[cfg(all(feature = "lpg", feature = "vector-index"))]
    #[test]
    fn coerce_params_to_vector_accepts_mixed_int_and_float() {
        let mut params = Parameters::new();
        params.set_list(
            "query",
            vec![Value::Float64(0.5), Value::Int64(1), Value::Float64(-0.25)],
        );
        let out = coerce_params_to_vector(&params, "query").unwrap();
        assert_eq!(out, vec![0.5_f32, 1.0_f32, -0.25_f32]);
    }

    #[cfg(all(feature = "lpg", feature = "vector-index"))]
    #[test]
    fn coerce_params_to_vector_errors_when_missing() {
        let params = Parameters::new();
        let err = coerce_params_to_vector(&params, "query").unwrap_err();
        assert!(
            err.to_string()
                .contains("Missing required vector parameter"),
            "error must name the parameter: {err}"
        );
    }

    #[cfg(all(feature = "lpg", feature = "vector-index"))]
    #[test]
    fn coerce_params_to_vector_errors_on_non_numeric_element() {
        let mut params = Parameters::new();
        params.set_list(
            "query",
            vec![Value::Float64(0.1), Value::String("nope".into())],
        );
        let err = coerce_params_to_vector(&params, "query").unwrap_err();
        assert!(
            err.to_string().contains("Expected numeric list"),
            "error must describe expected type: {err}"
        );
    }

    #[cfg(all(feature = "lpg", any(feature = "vector-index", feature = "text-index")))]
    #[test]
    fn k_limit_returns_default_when_missing() {
        let params = Parameters::new();
        assert_eq!(k_limit(&params, 7), 7);
    }

    #[cfg(all(feature = "lpg", any(feature = "vector-index", feature = "text-index")))]
    #[test]
    fn k_limit_clamps_negative_to_zero() {
        // A negative `k` must clamp to 0 (not flip to a huge usize via
        // bit-cast), so a caller that passes -1 gets "no results" instead
        // of scanning the entire index.
        let mut params = Parameters::new();
        params.set_int("k", -5);
        assert_eq!(k_limit(&params, 10), 0);
    }

    #[cfg(all(feature = "lpg", any(feature = "vector-index", feature = "text-index")))]
    #[test]
    fn k_limit_passes_through_positive_value() {
        let mut params = Parameters::new();
        params.set_int("k", 25);
        assert_eq!(k_limit(&params, 10), 25);
    }
}
