//! Prepared plans: a statement with values is optimized once with its
//! parameters in, cached, and each call fills a copy with its own values
//! (DuckDB's prepared statements). See [`Session::plan_with_values`].

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_common::utils::error::Result;
use grafeo_core::graph::GraphStore;

use super::Session;
use crate::query::cache::CacheKey;
use crate::query::plan::LogicalPlan;

impl Session {
    /// The plan cache's key for `query` in `language` in this session: its
    /// current schema and graph, which the plan is made for.
    pub(super) fn plan_cache_key(
        &self,
        query: &str,
        language: crate::query::processor::QueryLanguage,
    ) -> CacheKey {
        CacheKey::in_scope(query, language, self.current_schema(), self.current_graph())
    }

    /// The cached plan of a statement in this session (`key`): for a call
    /// without values its optimized plan, for one with values `params` its
    /// prepared plan, which keeps the parameters (see `plan_with_values`).
    pub(super) fn cached_plan(
        &self,
        key: &CacheKey,
        params: Option<&HashMap<String, Value>>,
    ) -> Option<LogicalPlan> {
        match params {
            None => self.query_cache.get_optimized(key),
            Some(_) => self.query_cache.get_optimized(&key.prepared()),
        }
    }

    /// The plan a statement with values `params` runs, from its translated
    /// plan `plan` (with its parameters in), optimized for `store`, when no
    /// prepared plan of it is cached (DuckDB's prepared statements).
    ///
    /// It binds the plan with the values filled in first, as a statement
    /// with literal values is bound, so a call fails as it would with the
    /// literals, and with the same error. Then, unless the binder answers by
    /// a value (a WITH or RETURN item that is a parameter alone, see
    /// [`Binder::binds_by_value`]), it optimizes the plan with its parameters
    /// in once and caches it as the statement's prepared plan, which every
    /// later call fills with its own values (`Session::fill_prepared`): no value is
    /// ever optimized into a cached plan, and the optimizer reads none (its
    /// estimates take a parameter as an unknown value). A statement that
    /// binds by value (`bound_by_value` once its parsed plan is cached) is
    /// filled, bound and optimized with its values on every call.
    ///
    /// # Errors
    ///
    /// A parameter that is neither given nor defaulted, and what binding or
    /// optimizing the plan with the values fails with.
    pub(super) fn plan_with_values(
        &self,
        key: &CacheKey,
        plan: LogicalPlan,
        params: &HashMap<String, Value>,
        store: &dyn GraphStore,
        bound_by_value: bool,
    ) -> Result<LogicalPlan> {
        use crate::query::{binder::Binder, optimizer::Optimizer};

        let optimizer = Optimizer::from_graph_store(store);
        let mut filled = plan.clone();
        fill_values(&mut filled, params)?;
        Binder::new().bind(&filled)?;
        if !bound_by_value {
            let mut binder = Binder::new();
            binder.bind(&plan)?;
            if !binder.binds_by_value() {
                let prepared = optimizer.optimize(plan)?;
                self.query_cache
                    .put_optimized(key.prepared(), prepared.clone());
                return Self::fill_prepared(prepared, Some(params));
            }
            self.query_cache.put_parsed(key.clone(), plan);
        }
        optimizer.optimize(filled)
    }

    /// The plan a call runs from a cached plan: a prepared plan (a call with
    /// values `params`) filled with them, an optimized plan (a call without) as
    /// it is.
    ///
    /// # Errors
    ///
    /// As [`fill_values`].
    pub(super) fn fill_prepared(
        mut plan: LogicalPlan,
        params: Option<&HashMap<String, Value>>,
    ) -> Result<LogicalPlan> {
        if let Some(params) = params {
            fill_values(&mut plan, params)?;
        }
        Ok(plan)
    }
}

/// Fills `params` (over the plan's own defaults) into `plan`.
///
/// # Errors
///
/// A parameter of the plan that is neither given nor defaulted.
fn fill_values(plan: &mut LogicalPlan, params: &HashMap<String, Value>) -> Result<()> {
    use crate::query::processor::substitute_params;

    if plan.default_params.is_empty() {
        substitute_params(plan, params)
    } else {
        let mut merged = plan.default_params.clone();
        merged.extend(
            params
                .iter()
                .map(|(name, value)| (name.clone(), value.clone())),
        );
        substitute_params(plan, &merged)
    }
}
