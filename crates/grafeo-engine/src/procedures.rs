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
//! - The loads (`grafeo.load_nodes_csv`, `grafeo.load_edges_csv`, see
//!   [`crate::loader`]): registered for their parameters and access, planned
//!   with the statement's writer.
//!
//! The registry is in every build; the graph algorithms need `algos`.
//! Each procedure states whether it writes the graph
//! ([`Procedure::graph_access`]), which a read-only session refuses, and
//! whether it reaches outside the database ([`Procedure::external_access`]),
//! which a session without external access refuses. `grafeo.procedures()`
//! lists both.

use std::sync::Arc;

use grafeo_adapters::plugins::Domain;
#[cfg(feature = "algos")]
use grafeo_adapters::plugins::algorithms::{
    ArticulationPointsAlgorithm, BellmanFordAlgorithm, BetweennessCentralityAlgorithm,
    BfsAlgorithm, BridgesAlgorithm, ClosenessCentralityAlgorithm, ClusteringCoefficientAlgorithm,
    ConnectedComponentsAlgorithm, DegreeCentralityAlgorithm, DfsAlgorithm, DijkstraAlgorithm,
    FloydWarshallAlgorithm, GraphAlgorithm, KCoreAlgorithm, KTrussAlgorithm, KruskalAlgorithm,
    LabelPropagationAlgorithm, LouvainAlgorithm, MaxFlowAlgorithm, MinCostFlowAlgorithm,
    PageRankAlgorithm, PrimAlgorithm, SsspAlgorithm, StronglyConnectedComponentsAlgorithm,
    SubgraphIsomorphismAlgorithm, TopologicalSortAlgorithm, TotalTrianglesAlgorithm,
};
use grafeo_adapters::plugins::{
    AlgorithmResult, ParameterDef, ParameterType, Parameters, Prepared, prepare,
};

pub use crate::auth::ExternalAccess;
use grafeo_common::types::{ReadView, Value};
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};
use grafeo_core::graph::GraphStore;
use grafeo_core::graph::rowgroup::RowGroupStore;

use crate::Snapshot;
use hashbrown::HashMap;

/// What a procedure does to the graph it is called on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphAccess {
    /// It only reads the graph.
    Reads,
    /// It writes the graph: a read-only session refuses the call.
    Writes,
}

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

    /// Whether the procedure reads or writes anything outside the database (a file, a URL, a
    /// model), which a session without external access refuses before the call is planned (see
    /// [`Identity::with_external_access`](crate::auth::Identity::with_external_access)). Every
    /// procedure states it: one added later is never let through by a default.
    fn external_access(&self) -> ExternalAccess;

    /// Whether the procedure writes the graph, which a read-only session refuses before the
    /// call is planned (GRAFEO-Q007). Every procedure states it: one added later is never
    /// taken for a read by a default.
    fn graph_access(&self) -> GraphAccess;

    /// Returns the canonical output column names in order, for a call with
    /// the parameters' defaults (as `grafeo.procedures()` lists them).
    fn output_columns(&self) -> Vec<String>;

    /// The output column names of a call with `params`, in order: a
    /// procedure whose rows take another shape for some arguments (a
    /// shortest path to one target, the members of one k-core) names that
    /// shape's columns.
    fn output_columns_for(&self, params: &Parameters) -> Vec<String> {
        let _ = params;
        self.output_columns()
    }

    /// Runs the procedure against the supplied context with the arguments
    /// `params`, prepared first (see [`prepare`]) and named as `CALL` names
    /// it (`grafeo.<name>`): a required argument without a value, or one
    /// outside its domain, is refused before any work, and an optional one
    /// left out takes its declared default. `CALL` prepares its arguments
    /// when it plans the statement (see [`evaluate_arguments`]) and runs
    /// [`compute`](Self::compute) with them.
    ///
    /// Implement [`compute`](Self::compute), never this: an override would
    /// skip the check.
    ///
    /// # Errors
    ///
    /// The invalid parameter value of [`prepare`], and the errors of
    /// [`compute`](Self::compute).
    fn execute(&self, ctx: &ProcedureContext<'_>, params: &Parameters) -> Result<AlgorithmResult> {
        let name = format!("grafeo.{}", self.name());
        let prepared = prepare(&name, self.parameters(), params)?;
        self.compute(ctx, &prepared)
    }

    /// The procedure's work against the supplied context, with arguments
    /// [`prepare`] checked and completed: a required one has a value, and an
    /// optional one that declares a default has it.
    ///
    /// # Errors
    ///
    /// Returns an error when the backing operation (graph algorithm, search
    /// index, or catalog lookup) fails.
    fn compute(&self, ctx: &ProcedureContext<'_>, params: &Prepared) -> Result<AlgorithmResult>;
}

/// Runtime context passed to [`Procedure::execute`].
///
/// Graph algorithms and catalog introspection need only [`ProcedureContext::snapshot`].
/// Vector and text search procedures additionally require
/// [`ProcedureContext::lpg_store`] to reach the HNSW and BM25 indexes owned
/// by the LPG store.
pub struct ProcedureContext<'a> {
    /// The graph the calling query reads, at its view, pinned for the call:
    /// every procedure reads the graph as the query sees it, an algorithm
    /// from start to end, and the search procedures return only the nodes
    /// it sees (a search index holds what open transactions wrote too, #412,
    /// which the store rechecks at the view). Sufficient for graph
    /// algorithms and catalog introspection (labels, edge types, property
    /// keys).
    pub snapshot: &'a Snapshot,

    /// Concrete LPG store reference when available, used by search procedures
    /// to reach vector and text indexes. `None` when the active backend is
    /// not an LPG store (e.g., pure RDF) or in contexts that do not need it.
    pub lpg_store: Option<&'a RowGroupStore>,
}

impl<'a> ProcedureContext<'a> {
    /// Creates a context with only the graph's snapshot available.
    #[must_use]
    pub fn new(snapshot: &'a Snapshot) -> Self {
        Self {
            snapshot,
            lpg_store: None,
        }
    }

    /// Creates a context with the graph's snapshot and the LPG store
    /// available.
    #[must_use]
    pub fn with_lpg_store(snapshot: &'a Snapshot, lpg_store: &'a RowGroupStore) -> Self {
        Self {
            snapshot,
            lpg_store: Some(lpg_store),
        }
    }

    /// The graph's store; read it at [`view`](Self::view).
    #[must_use]
    pub fn store(&self) -> &dyn GraphStore {
        self.snapshot.store()
    }

    /// The labels, edge types and property keys in use, as
    /// `GrafeoDB::schema` reports them.
    fn schema(&self) -> Result<crate::admin::LpgSchemaInfo> {
        crate::admin::LpgSchemaInfo::in_use(self.store(), self.view())
    }

    /// The view the calling query reads at.
    #[must_use]
    pub fn view(&self) -> ReadView {
        self.snapshot.view()
    }
}

/// Adapter that presents a [`GraphAlgorithm`] as a [`Procedure`], with
/// canonical output column names (e.g., `score` instead of `pagerank`, see
/// [`canonical_output_columns`]).
#[cfg(feature = "algos")]
pub struct GraphAlgorithmProcedure {
    inner: Arc<dyn GraphAlgorithm>,
}

#[cfg(feature = "algos")]
impl GraphAlgorithmProcedure {
    /// Wraps a graph algorithm as a procedure.
    pub fn new(algorithm: Arc<dyn GraphAlgorithm>) -> Self {
        Self { inner: algorithm }
    }
}

#[cfg(feature = "algos")]
impl Procedure for GraphAlgorithmProcedure {
    fn external_access(&self) -> ExternalAccess {
        ExternalAccess::None
    }

    fn graph_access(&self) -> GraphAccess {
        GraphAccess::Reads
    }

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
        canonical_output_columns(self.inner.as_ref(), &Parameters::new())
    }

    fn output_columns_for(&self, params: &Parameters) -> Vec<String> {
        canonical_output_columns(self.inner.as_ref(), params)
    }

    fn compute(&self, ctx: &ProcedureContext<'_>, params: &Prepared) -> Result<AlgorithmResult> {
        // The definitions are the algorithm's, so the arguments are its,
        // prepared once.
        self.inner.compute(ctx.store(), ctx.view(), params)
    }
}

// ---------------------------------------------------------------------------
// Catalog introspection procedures
// ---------------------------------------------------------------------------

/// `db.labels` / `grafeo.labels`: lists the labels in use, sorted.
struct LabelsProcedure;

impl Procedure for LabelsProcedure {
    fn external_access(&self) -> ExternalAccess {
        ExternalAccess::None
    }

    fn graph_access(&self) -> GraphAccess {
        GraphAccess::Reads
    }

    fn name(&self) -> &str {
        "labels"
    }

    fn description(&self) -> &str {
        "Lists the node labels some node in the graph has"
    }

    fn parameters(&self) -> &[ParameterDef] {
        &[]
    }

    fn output_columns(&self) -> Vec<String> {
        vec!["label".into()]
    }

    fn compute(&self, ctx: &ProcedureContext<'_>, _params: &Prepared) -> Result<AlgorithmResult> {
        let mut result = AlgorithmResult::new(vec!["label".into()]);
        for label in ctx.schema()?.labels {
            result.rows.push(vec![Value::String(label.name.into())]);
        }
        Ok(result)
    }
}

/// `db.relationshipTypes` / `grafeo.relationshipTypes`: lists the edge types
/// in use, sorted.
struct RelationshipTypesProcedure;

impl Procedure for RelationshipTypesProcedure {
    fn external_access(&self) -> ExternalAccess {
        ExternalAccess::None
    }

    fn graph_access(&self) -> GraphAccess {
        GraphAccess::Reads
    }

    fn name(&self) -> &str {
        "relationshipTypes"
    }

    fn description(&self) -> &str {
        "Lists the edge types some edge in the graph has"
    }

    fn parameters(&self) -> &[ParameterDef] {
        &[]
    }

    fn output_columns(&self) -> Vec<String> {
        vec!["relationshipType".into()]
    }

    fn compute(&self, ctx: &ProcedureContext<'_>, _params: &Prepared) -> Result<AlgorithmResult> {
        let mut result = AlgorithmResult::new(vec!["relationshipType".into()]);
        for edge_type in ctx.schema()?.edge_types {
            result.rows.push(vec![Value::String(edge_type.name.into())]);
        }
        Ok(result)
    }
}

/// `db.propertyKeys` / `grafeo.propertyKeys`: lists the property keys in
/// use, sorted.
struct PropertyKeysProcedure;

impl Procedure for PropertyKeysProcedure {
    fn external_access(&self) -> ExternalAccess {
        ExternalAccess::None
    }

    fn graph_access(&self) -> GraphAccess {
        GraphAccess::Reads
    }

    fn name(&self) -> &str {
        "propertyKeys"
    }

    fn description(&self) -> &str {
        "Lists the property keys some node or edge in the graph holds"
    }

    fn parameters(&self) -> &[ParameterDef] {
        &[]
    }

    fn output_columns(&self) -> Vec<String> {
        vec!["propertyKey".into()]
    }

    fn compute(&self, ctx: &ProcedureContext<'_>, _params: &Prepared) -> Result<AlgorithmResult> {
        let mut result = AlgorithmResult::new(vec!["propertyKey".into()]);
        for key in ctx.schema()?.property_keys {
            result.rows.push(vec![Value::String(key.into())]);
        }
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Vector / text search procedures
// ---------------------------------------------------------------------------

#[cfg(feature = "vector-index")]
fn require_lpg_store<'a>(ctx: &ProcedureContext<'a>, proc_name: &str) -> Result<&'a RowGroupStore> {
    ctx.lpg_store.ok_or_else(|| {
        grafeo_common::utils::error::Error::from(
            grafeo_common::utils::error::QueryError::unsupported(format!(
                "{proc_name} needs an LPG database"
            )),
        )
    })
}

#[cfg(feature = "vector-index")]
fn coerce_params_to_vector(params: &Parameters, key: &str) -> Result<Vec<f32>> {
    if let Some(list) = params.get_list(key) {
        let mut out = Vec::with_capacity(list.len());
        for v in list {
            match v.as_vector_component() {
                Some(component) => out.push(component),
                None => {
                    return Err(grafeo_common::utils::error::Error::InvalidValue(format!(
                        "the vector parameter '{key}' must be a list of numbers, but holds {v}"
                    )));
                }
            }
        }
        return Ok(out);
    }
    Err(grafeo_common::utils::error::Error::InvalidValue(format!(
        "the vector parameter '{key}' is missing or is not a list of numbers"
    )))
}

/// The `k` parameter as a limit, its declared default (10) when the call
/// leaves it out, which [`prepare`] fills in.
///
/// # Errors
///
/// An invalid parameter value for a negative `k`, which its domain refuses
/// before the procedure runs (see [`prepare`]), and an internal error when
/// `params` were not prepared.
#[cfg(any(feature = "vector-index", feature = "text-index"))]
fn k_limit(params: &Parameters) -> Result<usize> {
    params.required_count("k")
}

/// Converts a NodeId into a `Value::Int64`.
/// Using `cast_signed` keeps the bit pattern; ids above `i64::MAX` would
/// become negative, but the ID space is bounded well below that in practice.
#[cfg(any(feature = "vector-index", feature = "text-index"))]
fn node_id_to_value(node_id: grafeo_common::types::NodeId) -> Value {
    Value::Int64(node_id.as_u64().cast_signed())
}

/// `grafeo.search.vector(label, property, query, k)`:
/// k-nearest-neighbors via the HNSW index.
#[cfg(feature = "vector-index")]
struct SearchVectorProcedure {
    parameters: Vec<ParameterDef>,
}

#[cfg(feature = "vector-index")]
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
                    domain: Domain::OfType,
                    description: "Node label that was vector-indexed".into(),
                },
                ParameterDef {
                    name: "property".into(),
                    param_type: ParameterType::String,
                    required: true,
                    default: None,
                    domain: Domain::OfType,
                    description: "Property name holding the embedding".into(),
                },
                ParameterDef {
                    name: "query".into(),
                    param_type: ParameterType::List,
                    required: true,
                    default: None,
                    domain: Domain::OfType,
                    description: "Query vector as a list of floats".into(),
                },
                ParameterDef {
                    name: "k".into(),
                    param_type: ParameterType::Integer,
                    required: false,
                    default: Some("10".into()),
                    domain: Domain::NonNegative,
                    description: "Number of nearest neighbors to return".into(),
                },
            ],
        }
    }
}

#[cfg(feature = "vector-index")]
impl Procedure for SearchVectorProcedure {
    fn external_access(&self) -> ExternalAccess {
        ExternalAccess::None
    }

    fn graph_access(&self) -> GraphAccess {
        GraphAccess::Reads
    }

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

    fn compute(&self, ctx: &ProcedureContext<'_>, params: &Prepared) -> Result<AlgorithmResult> {
        let lpg = require_lpg_store(ctx, "CALL grafeo.search.vector")?;
        let label = params.required_string("label")?;
        let property = params.required_string("property")?;
        let query = coerce_params_to_vector(params, "query")?;
        let k = k_limit(params)?;

        let missing = || {
            grafeo_common::utils::error::Error::InvalidValue(format!(
                "there is no vector index on :{label}({property}); create one first"
            ))
        };
        let config = GraphStore::vector_index_config(lpg, label, property).ok_or_else(missing)?;
        grafeo_core::index::vector::check_query_vector(&query, config.dimensions, label, property)?;

        let search = grafeo_core::graph::rowgroup::IndexSearch {
            k,
            ef: None,
            allowed: None,
        };
        let results = lpg
            .vector_index_search_at(label, property, &query, search, ctx.view())?
            .ok_or_else(missing)?;

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
#[cfg(feature = "vector-index")]
struct SearchMmrProcedure {
    parameters: Vec<ParameterDef>,
}

#[cfg(feature = "vector-index")]
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
                    domain: Domain::OfType,
                    description: "Node label that was vector-indexed".into(),
                },
                ParameterDef {
                    name: "property".into(),
                    param_type: ParameterType::String,
                    required: true,
                    default: None,
                    domain: Domain::OfType,
                    description: "Property name holding the embedding".into(),
                },
                ParameterDef {
                    name: "query".into(),
                    param_type: ParameterType::List,
                    required: true,
                    default: None,
                    domain: Domain::OfType,
                    description: "Query vector as a list of floats".into(),
                },
                ParameterDef {
                    name: "k".into(),
                    param_type: ParameterType::Integer,
                    required: false,
                    default: Some("10".into()),
                    domain: Domain::NonNegative,
                    description: "Number of diverse results to return".into(),
                },
                ParameterDef {
                    name: "fetch_k".into(),
                    param_type: ParameterType::Integer,
                    required: false,
                    default: None,
                    domain: Domain::AtLeast("k"),
                    description: "Initial HNSW candidate count (default: 4*k)".into(),
                },
                ParameterDef {
                    name: "lambda".into(),
                    param_type: ParameterType::Float,
                    required: false,
                    default: Some("0.5".into()),
                    domain: Domain::Unit,
                    description: "Relevance vs diversity in [0, 1]".into(),
                },
            ],
        }
    }
}

#[cfg(feature = "vector-index")]
impl Procedure for SearchMmrProcedure {
    fn external_access(&self) -> ExternalAccess {
        ExternalAccess::None
    }

    fn graph_access(&self) -> GraphAccess {
        GraphAccess::Reads
    }

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

    fn compute(&self, ctx: &ProcedureContext<'_>, params: &Prepared) -> Result<AlgorithmResult> {
        use grafeo_core::index::vector::mmr_select;

        let lpg = require_lpg_store(ctx, "CALL grafeo.search.mmr")?;
        let label = params.required_string("label")?;
        let property = params.required_string("property")?;
        let query = coerce_params_to_vector(params, "query")?;
        let k = k_limit(params)?;
        let fetch_k = params
            .get_count("fetch_k")?
            .unwrap_or_else(|| k.saturating_mul(4));
        #[allow(
            clippy::cast_possible_truncation,
            reason = "lambda is a unit-interval weighting factor; f32 precision is sufficient"
        )]
        let lambda = params.required_float("lambda")? as f32;

        let missing = || {
            grafeo_common::utils::error::Error::InvalidValue(format!(
                "there is no vector index on :{label}({property}); create one first"
            ))
        };
        let config = GraphStore::vector_index_config(lpg, label, property).ok_or_else(missing)?;
        grafeo_core::index::vector::check_query_vector(&query, config.dimensions, label, property)?;

        let view = ctx.view();
        let search = grafeo_core::graph::rowgroup::IndexSearch {
            k: fetch_k,
            ef: None,
            allowed: None,
        };
        let initial = lpg
            .vector_index_search_at(label, property, &query, search, view)?
            .ok_or_else(missing)?;
        if initial.is_empty() {
            return Ok(AlgorithmResult::new(vec![
                "node_id".into(),
                "distance".into(),
            ]));
        }

        // The candidates' vectors at the view.
        let candidates: Vec<(grafeo_common::types::NodeId, f32, std::sync::Arc<[f32]>)> = initial
            .into_iter()
            .filter_map(|(id, dist)| lpg.vector_at(id, property, view).map(|v| (id, dist, v)))
            .collect();
        let candidate_refs: Vec<(grafeo_common::types::NodeId, f32, &[f32])> = candidates
            .iter()
            .map(|(id, dist, vec)| (*id, *dist, vec.as_ref()))
            .collect();

        let selected = mmr_select(&query, &candidate_refs, k, lambda, config.metric);

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
#[cfg(feature = "text-index")]
struct SearchTextProcedure {
    parameters: Vec<ParameterDef>,
}

#[cfg(feature = "text-index")]
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
                    domain: Domain::OfType,
                    description: "Node label that was text-indexed".into(),
                },
                ParameterDef {
                    name: "property".into(),
                    param_type: ParameterType::String,
                    required: true,
                    default: None,
                    domain: Domain::OfType,
                    description: "Property holding the indexed text".into(),
                },
                ParameterDef {
                    name: "query".into(),
                    param_type: ParameterType::String,
                    required: true,
                    default: None,
                    domain: Domain::OfType,
                    description: "Text query for BM25 scoring".into(),
                },
                ParameterDef {
                    name: "k".into(),
                    param_type: ParameterType::Integer,
                    required: false,
                    default: Some("10".into()),
                    domain: Domain::NonNegative,
                    description: "Number of top results to return".into(),
                },
            ],
        }
    }
}

#[cfg(feature = "text-index")]
impl Procedure for SearchTextProcedure {
    fn external_access(&self) -> ExternalAccess {
        ExternalAccess::None
    }

    fn graph_access(&self) -> GraphAccess {
        GraphAccess::Reads
    }

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

    fn compute(&self, ctx: &ProcedureContext<'_>, params: &Prepared) -> Result<AlgorithmResult> {
        let lpg = ctx.lpg_store.ok_or_else(|| {
            grafeo_common::utils::error::Error::from(
                grafeo_common::utils::error::QueryError::unsupported(
                    "CALL grafeo.search.text needs an LPG database",
                ),
            )
        })?;
        let label = params.required_string("label")?;
        let property = params.required_string("property")?;
        let query = params.required_string("query")?;
        let k = k_limit(params)?;

        let search = grafeo_core::graph::rowgroup::IndexSearch {
            k,
            ef: None,
            allowed: None,
        };
        let results = lpg
            .text_index_search_at(label, property, query, search, ctx.view())?
            .ok_or_else(|| {
                grafeo_common::utils::error::Error::InvalidValue(format!(
                    "there is no text index on :{label}({property}); create one first"
                ))
            })?;

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
        #[cfg(feature = "algos")]
        Self::register_algorithms(&mut procedures);

        // Catalog introspection
        let mut register = |proc: Arc<dyn Procedure>| {
            procedures.insert(proc.name().to_string(), proc);
        };
        register(Arc::new(LabelsProcedure));
        register(Arc::new(RelationshipTypesProcedure));
        register(Arc::new(PropertyKeysProcedure));

        // The loads, which read a file.
        register(Arc::new(crate::loader::LoadProcedure::nodes()));
        register(Arc::new(crate::loader::LoadProcedure::edges()));

        // Search procedures (feature-gated).
        #[cfg(feature = "vector-index")]
        {
            register(Arc::new(SearchVectorProcedure::new()));
            register(Arc::new(SearchMmrProcedure::new()));
        }
        #[cfg(feature = "text-index")]
        {
            register(Arc::new(SearchTextProcedure::new()));
        }

        Self { procedures }
    }

    /// The graph algorithms, each wrapped as a procedure.
    #[cfg(feature = "algos")]
    fn register_algorithms(procedures: &mut HashMap<String, Arc<dyn Procedure>>) {
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
        register_algo(Arc::new(TotalTrianglesAlgorithm));

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
        register_algo(Arc::new(KTrussAlgorithm));

        // Pattern matching
        register_algo(Arc::new(SubgraphIsomorphismAlgorithm));
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
                writes: proc.graph_access() == GraphAccess::Writes,
                external_access: proc.external_access(),
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

/// The registry `CALL` resolves procedures in, and the external-access check reads: built
/// once.
pub(crate) fn registry() -> &'static BuiltinProcedures {
    static PROCEDURES: std::sync::OnceLock<BuiltinProcedures> = std::sync::OnceLock::new();
    PROCEDURES.get_or_init(BuiltinProcedures::new)
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
    /// Whether it writes the graph (see [`Procedure::graph_access`]).
    pub writes: bool,
    /// Whether it reaches outside the database (see [`Procedure::external_access`]).
    pub external_access: ExternalAccess,
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
pub(crate) fn resolve_name(parts: &[String]) -> String {
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

/// Returns canonical output column names for the given graph algorithm,
/// called with `params`.
///
/// These user-facing names (e.g., `"score"` instead of `"pagerank"`) name
/// the columns each algorithm's `execute()` gives for those parameters, in
/// order: `dijkstra` to a `target` gives the one path, `kcore` with a `k`
/// the members of that core, a normalized `degree_centrality` one score.
#[cfg(feature = "algos")]
#[must_use]
pub fn canonical_output_columns(algo: &dyn GraphAlgorithm, params: &Parameters) -> Vec<String> {
    match algo.name() {
        "dijkstra" if params.get_int("target").is_some() => vec![
            "source".into(),
            "target".into(),
            "distance".into(),
            "path".into(),
        ],
        "kcore" if params.get_int("k").is_some() => vec!["node_id".into(), "in_k_core".into()],
        "ktruss" if params.get_int("k").is_some() => vec!["source".into(), "target".into()],
        "degree_centrality" if params.get_bool("normalized") == Some(true) => {
            vec!["node_id".into(), "centrality".into()]
        }
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
        "kruskal" | "prim" => vec![
            "source".into(),
            "target".into(),
            "weight".into(),
            "total_weight".into(),
        ],
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
            "total_cost".into(),
        ],
        "articulation_points" => vec!["node_id".into()],
        "bridges" => vec!["source".into(), "target".into()],
        "kcore" => vec!["node_id".into(), "core_number".into(), "max_core".into()],
        "ktruss" => vec!["source".into(), "target".into(), "truss_number".into()],
        "total_triangles" => vec!["total_triangles".into()],
        "subgraph_isomorphism" => vec!["count".into()],
        _ => vec!["node_id".into(), "value".into()],
    }
}

/// The error of a YIELD item that names no column of the procedure, listing the columns it has:
/// an invalid reference (a mistake in the query text), raised when the call is planned.
#[must_use]
pub fn no_such_yield_column(field: &str, columns: &[String]) -> Error {
    Error::from(QueryError::invalid_reference(format!(
        "YIELD column '{field}' is no column of the procedure (it has: {})",
        columns.join(", ")
    )))
}

/// The name an error gives positional argument `index` of a call: its
/// parameter's name, or its position past the last parameter.
#[must_use]
pub fn argument_name(param_defs: &[ParameterDef], index: usize) -> String {
    param_defs
        .get(index)
        .map_or_else(|| format!("#{}", index + 1), |def| def.name.clone())
}

/// The map keys every procedure call takes besides its parameters:
/// `projection` names the graph projection the procedure reads (see the
/// planner's `procedure_store`).
const PASS_THROUGH_KEYS: &[&str] = &["projection"];

/// Converts the values of a procedure call's arguments into [`Parameters`].
///
/// Two forms:
/// 1. One map, `{damping: 0.85, max_iterations: 20}` (a map literal or a map
///    parameter), names the parameters.
/// 2. Positional values, `(42, 'weight')`, fill the parameters in the order
///    of `param_defs`.
///
/// Each value must suit its parameter's type: an integer passes for a float
/// and a vector for a list. The arguments are then prepared (see
/// [`prepare`]), once, when the statement is planned: a value outside its
/// domain is refused, a required parameter left out or given null is refused
/// naming it, and an optional one without a value takes its declared
/// default (a null is no value). A map key that names no
/// parameter passes through only as one of the keys every call takes
/// (`projection`), with the type of its value.
///
/// # Errors
///
/// Returns a semantic error naming the argument and `procedure` when a value
/// does not suit its parameter's type, and an invalid parameter value when a
/// required parameter has none or a value lies outside its domain: such a
/// value never falls back to the
/// parameter's default. Returns an invalid reference naming the parameters
/// for a map key that names none of them (`{dampingg: 0.5}`), and for more
/// positional values than the procedure has parameters: a mistyped or
/// misplaced argument fails instead of being left out.
pub fn evaluate_arguments(
    procedure: &str,
    args: &[Value],
    param_defs: &[ParameterDef],
) -> Result<Prepared> {
    let mut params = Parameters::new();

    if let [Value::Map(entries)] = args {
        for (key, value) in entries.iter() {
            let def = param_defs.iter().find(|def| def.name == key.as_str());
            if def.is_none() && !PASS_THROUGH_KEYS.contains(&key.as_str()) {
                return Err(Error::from(QueryError::invalid_reference(format!(
                    "Argument '{key}' is no parameter of {procedure} (it has: {})",
                    parameter_names(param_defs)
                ))));
            }
            set_parameter(&mut params, procedure, key.as_str(), def, value)?;
        }
        return params.prepared(procedure, param_defs);
    }

    if args.len() > param_defs.len() {
        return Err(Error::from(QueryError::invalid_reference(format!(
            "{procedure} takes at most {} arguments, got {} (its parameters: {})",
            param_defs.len(),
            args.len(),
            parameter_names(param_defs)
        ))));
    }
    for (def, value) in param_defs.iter().zip(args) {
        set_parameter(&mut params, procedure, &def.name, Some(def), value)?;
    }
    params.prepared(procedure, param_defs)
}

/// The names of `param_defs`, for an error: `none` for a procedure without
/// parameters.
fn parameter_names(param_defs: &[ParameterDef]) -> String {
    if param_defs.is_empty() {
        return "none".to_string();
    }
    param_defs
        .iter()
        .map(|def| def.name.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Sets parameter `name`, which `def` defines, to `value`, which must suit its
/// type (any value a parameter can hold for a key every call takes, `def`
/// `None`). One value for a list that declares it takes one
/// ([`Domain::ListOrOne`](grafeo_adapters::plugins::Domain::ListOrOne)) is
/// set as it is, and [`prepare`] makes it a list of one.
fn set_parameter(
    params: &mut Parameters,
    procedure: &str,
    name: &str,
    def: Option<&ParameterDef>,
    value: &Value,
) -> Result<()> {
    use ParameterType as Type;
    let expected = def.map(|def| def.param_type);
    let one_is_a_list = def.is_some_and(|def| def.domain == Domain::ListOrOne);
    match (expected, value) {
        (_, Value::Null) => params.set_null(name),
        (Some(Type::Integer | Type::NodeId) | None, Value::Int64(v)) => params.set_int(name, *v),
        (Some(Type::Float), Value::Int64(v)) => params.set_float(name, *v as f64),
        (Some(Type::Float) | None, Value::Float64(v)) => params.set_float(name, *v),
        (Some(Type::Float) | None, Value::Decimal(d)) => params.set_float(name, d.to_f64()),
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
        (Some(Type::Map), Value::Map(entries)) => params.set_map(name, Arc::clone(entries)),
        (Some(Type::List), value) if one_is_a_list => {
            return set_parameter(params, procedure, name, None, value);
        }
        (expected, value) => {
            let wanted = match expected {
                Some(Type::Integer) => "an integer",
                Some(Type::NodeId) => "a node id (an integer)",
                Some(Type::Float) => "a number",
                Some(Type::String) => "a string",
                Some(Type::Boolean) => "a boolean",
                Some(Type::List) => "a list",
                Some(Type::Map) => "a map",
                None => "a number, string, boolean or list",
            };
            return Err(Error::from(QueryError::new(
                QueryErrorKind::InvalidValueType,
                format!(
                    "Argument '{name}' of {procedure} must be {wanted}, got a {} value",
                    value.type_name()
                ),
            )));
        }
    }
    Ok(())
}

/// Builds a `grafeo.procedures()` result listing all registered procedures: name, description,
/// parameters, output columns, whether it writes the graph (`writes`, a BOOLEAN) and what it
/// reaches outside the database (`external_access`: 'none', 'reads' or 'writes').
#[must_use]
pub fn procedures_result(registry: &BuiltinProcedures) -> AlgorithmResult {
    let procedures = registry.list();
    let mut result = AlgorithmResult::new(vec![
        "name".into(),
        "description".into(),
        "parameters".into(),
        "output_columns".into(),
        "writes".into(),
        "external_access".into(),
    ]);
    for proc in procedures {
        let param_desc: String = proc
            .parameters
            .iter()
            .map(|p| {
                // The values the argument allows, after the type and before the default (which
                // readers take as the text after `default=`): what an invalid parameter value
                // (22023) names.
                let domain = match p.domain {
                    Domain::OfType => String::new(),
                    domain => format!(", {}", domain.describe(None)),
                };
                if p.required {
                    format!("{} ({:?}{domain})", p.name, p.param_type)
                } else if let Some(ref default) = p.default {
                    format!(
                        "{} ({:?}{domain}, default={})",
                        p.name, p.param_type, default
                    )
                } else {
                    format!("{} ({:?}{domain}, optional)", p.name, p.param_type)
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
            Value::Bool(proc.writes),
            Value::from(proc.external_access.name()),
        ]);
    }
    result
}

/// The procedures a build leaves out unless it has a feature: the feature, and the names it
/// registers. A call of one in a build without it is told so (see [`missing_feature`]), not
/// taken for a misspelt name. A test holds this table to the registry of a build with every
/// feature.
const FEATURE_GATED: &[(&str, &[&str])] = &[
    (
        "algos",
        &[
            "articulation_points",
            "bellman_ford",
            "betweenness_centrality",
            "bfs",
            "bridges",
            "closeness_centrality",
            "clustering_coefficient",
            "connected_components",
            "degree_centrality",
            "dfs",
            "dijkstra",
            "floyd_warshall",
            "kcore",
            "kruskal",
            "ktruss",
            "label_propagation",
            "louvain",
            "max_flow",
            "min_cost_max_flow",
            "pagerank",
            "prim",
            "sssp",
            "strongly_connected_components",
            "subgraph_isomorphism",
            "topological_sort",
            "total_triangles",
        ],
    ),
    ("vector-index", &["search.mmr", "search.vector"]),
    ("text-index", &["search.text"]),
];

/// The feature a build needs for the procedure `name` (dotted, as a call writes it) when this
/// build leaves it out: `None` for a procedure it has, or a name no build has.
#[must_use]
pub fn missing_feature(name: &[String]) -> Option<&'static str> {
    let key = resolve_name(name);
    if registry().procedures.contains_key(&key) {
        return None;
    }
    FEATURE_GATED
        .iter()
        .find(|(_, names)| names.contains(&key.as_str()))
        .map(|(feature, _)| *feature)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With every feature, the registry holds every gated name and nothing gated beyond them:
    /// a procedure added behind a feature is told apart from a misspelt name once it is listed.
    #[cfg(all(feature = "algos", feature = "vector-index", feature = "text-index"))]
    #[test]
    fn every_feature_gated_procedure_is_listed() {
        const IN_EVERY_BUILD: &[&str] = &[
            "labels",
            "load_edges_csv",
            "load_nodes_csv",
            "propertyKeys",
            "relationshipTypes",
        ];
        let mut registered: Vec<&str> = registry().procedures.keys().map(String::as_str).collect();
        registered.sort_unstable();
        let mut listed: Vec<&str> = FEATURE_GATED
            .iter()
            .flat_map(|(_, names)| names.iter().copied())
            .chain(IN_EVERY_BUILD.iter().copied())
            .collect();
        listed.sort_unstable();
        assert_eq!(registered, listed);
        assert_eq!(missing_feature(&["grafeo".into(), "pagerank".into()]), None);
        assert_eq!(missing_feature(&["grafeo".into(), "no_such".into()]), None);
    }

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
            domain: Domain::OfType,
            description: String::new(),
        }
    }

    #[test]
    fn test_evaluate_map_arguments() {
        let args = vec![map(&[
            ("damping", Value::Float64(0.85)),
            ("max_iterations", Value::Int64(20)),
        ])];
        let defs = [
            def("damping", ParameterType::Float, false),
            def("max_iterations", ParameterType::Integer, false),
        ];
        let params = evaluate_arguments("pagerank", &args, &defs).unwrap();
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
            canonical_output_columns(dijkstra.as_ref(), &Parameters::new()),
            vec!["node_id", "distance"]
        );

        let sssp: Arc<dyn GraphAlgorithm> = Arc::new(SsspAlgorithm);
        assert_eq!(
            canonical_output_columns(sssp.as_ref(), &Parameters::new()),
            vec!["node_id", "distance"]
        );

        let bellman: Arc<dyn GraphAlgorithm> = Arc::new(BellmanFordAlgorithm);
        assert_eq!(
            canonical_output_columns(bellman.as_ref(), &Parameters::new()),
            vec!["node_id", "distance", "has_negative_cycle"]
        );
    }

    #[test]
    fn canonical_output_columns_for_community_algorithms() {
        let louvain: Arc<dyn GraphAlgorithm> = Arc::new(LouvainAlgorithm);
        assert_eq!(
            canonical_output_columns(louvain.as_ref(), &Parameters::new()),
            vec!["node_id", "community_id", "modularity"]
        );

        let lpa: Arc<dyn GraphAlgorithm> = Arc::new(LabelPropagationAlgorithm);
        assert_eq!(
            canonical_output_columns(lpa.as_ref(), &Parameters::new()),
            vec!["node_id", "community_id"]
        );
    }

    #[test]
    fn canonical_output_columns_for_mst_algorithms() {
        let kruskal: Arc<dyn GraphAlgorithm> = Arc::new(KruskalAlgorithm);
        assert_eq!(
            canonical_output_columns(kruskal.as_ref(), &Parameters::new()),
            vec!["source", "target", "weight", "total_weight"]
        );
        let prim: Arc<dyn GraphAlgorithm> = Arc::new(PrimAlgorithm);
        assert_eq!(
            canonical_output_columns(prim.as_ref(), &Parameters::new()),
            vec!["source", "target", "weight", "total_weight"]
        );
    }

    #[test]
    fn canonical_output_columns_for_structural_algorithms() {
        let ap: Arc<dyn GraphAlgorithm> = Arc::new(ArticulationPointsAlgorithm);
        assert_eq!(
            canonical_output_columns(ap.as_ref(), &Parameters::new()),
            vec!["node_id"]
        );

        let bridges: Arc<dyn GraphAlgorithm> = Arc::new(BridgesAlgorithm);
        assert_eq!(
            canonical_output_columns(bridges.as_ref(), &Parameters::new()),
            vec!["source", "target"]
        );

        let kcore: Arc<dyn GraphAlgorithm> = Arc::new(KCoreAlgorithm);
        assert_eq!(
            canonical_output_columns(kcore.as_ref(), &Parameters::new()),
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
        fn compute(
            &self,
            _store: &dyn grafeo_core::graph::GraphStore,
            _view: grafeo_common::types::ReadView,
            _params: &Prepared,
        ) -> Result<AlgorithmResult> {
            Ok(AlgorithmResult::new(vec!["node_id".into(), "value".into()]))
        }
    }

    #[test]
    fn canonical_output_columns_falls_back_to_node_id_value() {
        let unknown: Arc<dyn GraphAlgorithm> = Arc::new(UnknownAlgo);
        assert_eq!(
            canonical_output_columns(unknown.as_ref(), &Parameters::new()),
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
    fn evaluate_arguments_refuses_positional_past_last_param_def() {
        // A positional value past the end of `param_defs` is a mistake in
        // the call, which fails instead of being left out
        let defs = [def("only", ParameterType::Integer, true)];
        let args = [Value::Int64(1), Value::Int64(2)];
        let Err(error) = evaluate_arguments("only", &args, &defs) else {
            panic!("an extra positional argument is refused");
        };
        assert!(
            error.to_string().contains("at most 1 arguments, got 2"),
            "{error}"
        );
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
            vec![
                "name",
                "description",
                "parameters",
                "output_columns",
                "writes",
                "external_access"
            ],
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

    #[cfg(feature = "vector-index")]
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

    #[cfg(feature = "vector-index")]
    #[test]
    fn coerce_params_to_vector_errors_when_missing() {
        let params = Parameters::new();
        let err = coerce_params_to_vector(&params, "query").unwrap_err();
        assert!(
            matches!(err, grafeo_common::utils::error::Error::InvalidValue(_))
                && err.to_string().contains("'query' is missing"),
            "error must name the parameter: {err}"
        );
    }

    #[cfg(feature = "vector-index")]
    #[test]
    fn coerce_params_to_vector_errors_on_non_numeric_element() {
        let mut params = Parameters::new();
        params.set_list(
            "query",
            vec![Value::Float64(0.1), Value::String("nope".into())],
        );
        let err = coerce_params_to_vector(&params, "query").unwrap_err();
        assert!(
            matches!(err, grafeo_common::utils::error::Error::InvalidValue(_))
                && err.to_string().contains("must be a list of numbers"),
            "error must describe expected type: {err}"
        );
    }

    /// #593: `grafeo.search.vector` and `grafeo.search.mmr` refuse a query
    /// vector with a NaN or an infinite value (a query literal cannot hold
    /// one, so the procedures are called directly), naming the value, as an
    /// invalid value.
    #[cfg(feature = "vector-index")]
    #[test]
    fn search_procedures_refuse_nan_and_infinite_query_values() {
        use grafeo_common::utils::error::ErrorCode;
        use grafeo_core::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind};

        let store = Arc::new(RowGroupStore::new());
        for vector in [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
            let node = store.create_node(&["Doc"]);
            store.set_node_property(node, "emb", Value::Vector(vector.to_vec().into()));
        }
        let index = HnswIndex::with_seed(HnswConfig::new(3, DistanceMetric::Cosine), 3);
        store.add_vector_index(
            "Doc",
            "emb",
            std::sync::Arc::new(VectorIndexKind::Hnsw(index)),
        );
        // The index takes the vectors the store already holds when they are
        // set again.
        for node in store.nodes_by_label("Doc", ReadView::latest()).unwrap() {
            let vector = store
                .node_property(node, "emb", ReadView::latest())
                .unwrap()
                .expect("vector");
            store.set_node_property(node, "emb", vector);
        }
        let snapshot = Snapshot::of_query(
            Arc::clone(&store) as Arc<dyn GraphStore>,
            ReadView::latest(),
        );
        let ctx = ProcedureContext::with_lpg_store(&snapshot, &store);
        let procedures: [Box<dyn Procedure>; 2] = [
            Box::new(SearchVectorProcedure::new()),
            Box::new(SearchMmrProcedure::new()),
        ];
        for procedure in &procedures {
            for (query, message) in [
                (
                    [0.9, f64::NAN, 0.0],
                    "the query vector has NaN at position 1",
                ),
                (
                    [0.9, 0.0, f64::INFINITY],
                    "the query vector has inf at position 2",
                ),
            ] {
                let mut params = Parameters::new();
                params.set_string("label", "Doc");
                params.set_string("property", "emb");
                params.set_list("query", query.iter().map(|v| Value::Float64(*v)).collect());
                params.set_int("k", 2);
                let Err(err) = procedure.execute(&ctx, &params) else {
                    panic!("{}: {message} was not refused", procedure.name());
                };
                assert_eq!(err.error_code(), ErrorCode::InvalidInput, "{err}");
                assert!(
                    err.to_string().contains(message),
                    "{}: {err}",
                    procedure.name()
                );
            }
            let mut params = Parameters::new();
            params.set_string("label", "Doc");
            params.set_string("property", "emb");
            params.set_list(
                "query",
                vec![
                    Value::Float64(1.0),
                    Value::Float64(0.0),
                    Value::Float64(0.0),
                ],
            );
            params.set_int("k", 1);
            let result = procedure
                .execute(&ctx, &params)
                .expect("a query it can measure");
            assert_eq!(result.rows.len(), 1, "{}", procedure.name());
        }
    }

    /// `k` left out takes the default search.vector declares (10), which `prepare` fills in; a
    /// `k` read from parameters nobody prepared is an error, never a default of k_limit's own.
    #[cfg(feature = "vector-index")]
    #[test]
    fn k_limit_reads_the_declared_default_when_k_is_left_out() {
        let procedure = SearchVectorProcedure::new();
        let mut params = Parameters::new();
        params.set_string("label", "Doc");
        params.set_string("property", "emb");
        params.set_list("query", vec![Value::Float64(0.5)]);
        let prepared = prepare("grafeo.search.vector", procedure.parameters(), &params).unwrap();
        assert_eq!(k_limit(&prepared).ok(), Some(10));
        assert!(k_limit(&Parameters::new()).is_err());
    }

    /// A negative `k` lies outside its domain (`k >= 0`) and is refused with 22023 naming the
    /// argument (0.6.2), where it clamped to 0 before; it never flips to a huge count. Zero stays
    /// a valid count.
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    #[test]
    fn k_limit_refuses_a_negative_k() {
        let mut params = Parameters::new();
        for k in [-5, -1, i64::MIN] {
            params.set_int("k", k);
            let error = k_limit(&params).expect_err("a negative k");
            assert_eq!(
                grafeo_common::utils::gqlstatus::GqlStatus::from(&error).as_str(),
                "22023",
                "k = {k}: {error}"
            );
            assert!(error.to_string().contains("'k'"), "k = {k}: {error}");
        }
        params.set_int("k", 0);
        assert_eq!(k_limit(&params).ok(), Some(0));
    }

    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    #[test]
    fn k_limit_passes_through_positive_value() {
        let mut params = Parameters::new();
        params.set_int("k", 25);
        assert_eq!(k_limit(&params).ok(), Some(25));
    }

    /// Every graph algorithm, called in each shape its arguments give, names
    /// as many columns as its rows hold, so a `YIELD` of a listed column
    /// reads that column and a call without `YIELD` names every value.
    #[test]
    fn declared_columns_match_the_rows_of_every_argument_shape() {
        // A small DAG with weights, capacities and costs: a -> b -> c, a -> c
        let store = Arc::new(RowGroupStore::new());
        let a = store.create_node(&["N"]);
        let b = store.create_node(&["N"]);
        let c = store.create_node(&["N"]);
        for (from, to, w) in [(a, b, 1.0), (b, c, 2.0), (a, c, 4.0)] {
            let edge = store.create_edge(from, to, "E");
            store.set_edge_property(edge, "w", Value::Float64(w));
            store.set_edge_property(edge, "capacity", Value::Int64(3));
            store.set_edge_property(edge, "cost", Value::Int64(1));
        }
        let snapshot = Snapshot::of_query(
            Arc::clone(&store) as Arc<dyn GraphStore>,
            ReadView::latest(),
        );
        let ctx = ProcedureContext::with_lpg_store(&snapshot, &store);
        let id = |node: grafeo_common::types::NodeId| i64::try_from(node.0).unwrap();
        let ints = |pairs: &[(&str, i64)]| {
            let mut params = Parameters::new();
            for (name, value) in pairs {
                params.set_int(*name, *value);
            }
            params
        };
        let mut normalized = Parameters::new();
        normalized.set_bool("normalized", true);
        let shapes: Vec<(&str, Parameters, Option<&[&str]>)> = vec![
            ("pagerank", Parameters::new(), None),
            ("betweenness_centrality", Parameters::new(), None),
            ("closeness_centrality", Parameters::new(), None),
            ("degree_centrality", Parameters::new(), None),
            (
                "degree_centrality",
                normalized,
                Some(&["node_id", "centrality"]),
            ),
            ("bfs", ints(&[("start", id(a))]), None),
            ("dfs", ints(&[("start", id(a))]), None),
            ("connected_components", Parameters::new(), None),
            ("strongly_connected_components", Parameters::new(), None),
            ("topological_sort", Parameters::new(), None),
            ("dijkstra", ints(&[("source", id(a))]), None),
            (
                "dijkstra",
                ints(&[("source", id(a)), ("target", id(c))]),
                Some(&["source", "target", "distance", "path"]),
            ),
            ("bellman_ford", ints(&[("source", id(a))]), None),
            ("floyd_warshall", Parameters::new(), None),
            ("clustering_coefficient", Parameters::new(), None),
            ("label_propagation", Parameters::new(), None),
            ("louvain", Parameters::new(), None),
            ("kruskal", Parameters::new(), None),
            ("prim", Parameters::new(), None),
            (
                "max_flow",
                ints(&[("source", id(a)), ("sink", id(c))]),
                None,
            ),
            (
                "min_cost_max_flow",
                ints(&[("source", id(a)), ("sink", id(c))]),
                None,
            ),
            ("articulation_points", Parameters::new(), None),
            ("bridges", Parameters::new(), None),
            ("kcore", Parameters::new(), None),
            ("kcore", ints(&[("k", 1)]), Some(&["node_id", "in_k_core"])),
        ];
        let registry = BuiltinProcedures::new();
        let mut mismatches = Vec::new();
        for (name, params, expected) in &shapes {
            let procedure = registry
                .get(&[(*name).to_string()])
                .unwrap_or_else(|| panic!("no procedure {name}"));
            let result = procedure
                .execute(&ctx, params)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let declared = procedure.output_columns_for(params);
            if declared.len() != result.columns.len() {
                mismatches.push(format!(
                    "{name}: declares {declared:?}, its rows hold {:?}",
                    result.columns
                ));
            }
            if let Some(expected) = expected {
                assert_eq!(declared, *expected, "{name}");
            }
        }
        assert!(
            mismatches.is_empty(),
            "{}",
            mismatches.join(
                "
"
            )
        );
    }

    #[test]
    fn an_unknown_argument_name_or_an_extra_argument_is_refused() {
        let defs = [
            def("damping", ParameterType::Float, false),
            def("max_iterations", ParameterType::Integer, false),
        ];
        let Err(error) = evaluate_arguments(
            "grafeo.pagerank",
            &[map(&[("dampingg", Value::Float64(0.5))])],
            &defs,
        ) else {
            panic!("a misspelled argument name is refused");
        };
        let message = error.to_string();
        assert!(message.contains("'dampingg'"), "{message}");
        assert!(message.contains("damping, max_iterations"), "{message}");
        let Err(error) = evaluate_arguments(
            "grafeo.pagerank",
            &[Value::Float64(0.85), Value::Int64(20), Value::Int64(7)],
            &defs,
        ) else {
            panic!("a third positional argument is refused");
        };
        let message = error.to_string();
        assert!(message.contains("at most 2 arguments, got 3"), "{message}");
        // The keys every call takes still pass
        assert!(
            evaluate_arguments(
                "grafeo.pagerank",
                &[map(&[("projection", Value::from("p"))])],
                &defs
            )
            .is_ok()
        );
    }
}
