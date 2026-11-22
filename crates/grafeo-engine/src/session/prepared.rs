//! Prepared plans: a statement with values is optimized once with its
//! parameters in, cached, and each call fills a copy with its own values
//! (DuckDB's prepared statements). See [`Session::plan_with_values`].

use std::collections::HashMap;
use std::sync::Arc;

use grafeo_common::types::Value;
use grafeo_common::utils::error::Result;
use grafeo_core::graph::GraphStore;

use super::SessionInner;
use crate::query::cache::{CacheKey, CachedPlan, QueryCache, SessionPlans, Stage};
use crate::query::plan::LogicalPlan;

/// Where a session's statements find and store their plans
/// (`SessionInner::plan_cache`).
pub(super) enum PlanCache<'a> {
    /// The open transaction's private cache: it changed the catalog it plans with.
    Private(Arc<QueryCache>),
    /// The session's own cache, in front of the shared one.
    Session(&'a SessionPlans),
}

impl PlanCache<'_> {
    fn cache(&self) -> &QueryCache {
        match self {
            Self::Private(private) => private,
            Self::Session(session) => session.shared(),
        }
    }

    /// The planning generation now: a statement reads it before it plans.
    pub(super) fn generation(&self) -> u64 {
        self.cache().generation()
    }

    /// The plan cached for `key` at `stage`.
    pub(super) fn get(&self, key: &CacheKey, stage: Stage) -> Option<Arc<CachedPlan>> {
        match self {
            Self::Private(private) => private.get(key, stage),
            Self::Session(session) => session.get(key, stage),
        }
    }

    /// Stores `plan`, planned under `generation`, and returns it shared (see
    /// [`QueryCache::put`]).
    pub(super) fn put(
        &self,
        key: CacheKey,
        stage: Stage,
        plan: LogicalPlan,
        generation: u64,
    ) -> Arc<CachedPlan> {
        match self {
            Self::Private(private) => private.put(key, stage, plan, generation),
            Self::Session(session) => session.put(key, stage, plan, generation),
        }
    }
}

/// The plan a statement runs: a cached plan, shared, or one made for this
/// run (filled with its values). Either reads as a [`LogicalPlan`].
pub(crate) enum RunPlan {
    /// A plan the cache holds or just stored.
    Shared(std::sync::Arc<CachedPlan>),
    /// A plan of this run alone.
    Owned(LogicalPlan),
}

impl std::ops::Deref for RunPlan {
    type Target = LogicalPlan;

    fn deref(&self) -> &LogicalPlan {
        match self {
            Self::Shared(cached) => cached.plan(),
            Self::Owned(plan) => plan,
        }
    }
}

impl SessionInner {
    /// The cache of this session's plans: its own cache in front of the shared one, or,
    /// while the open transaction has its own view, the transaction's private cache, whose
    /// plans no other session may run, nor it theirs.
    pub(super) fn plan_cache(&self) -> PlanCache<'_> {
        match &*self.pending_schema.lock() {
            Some(pending) => PlanCache::Private(Arc::clone(&pending.plans)),
            None => PlanCache::Session(&self.session_plans),
        }
    }

    /// The plan cache's key for `query` in `language` in this session: its
    /// current schema and graph and its planner rules, which the plan is made
    /// for.
    ///
    /// # Errors
    ///
    /// A statement longer than the session's query length limit has no key:
    /// it fails here, before the cache is read, as its parse would (the key
    /// leaves out the whitespace around the text, so a longer text can share
    /// the key of a shorter one).
    pub(super) fn plan_cache_key(
        &self,
        query: &str,
        language: crate::query::processor::QueryLanguage,
    ) -> Result<CacheKey> {
        self.plan_options.parse_limits.check_query(query)?;
        Ok(
            CacheKey::in_scope(query, language, self.current_schema(), self.current_graph())
                .with_rules(self.plan_options.rules),
        )
    }

    /// The cached plan of a statement in this session (`key`): for a call
    /// without values its optimized plan, for one with values `params` its
    /// prepared plan, which keeps the parameters (see `plan_with_values`).
    pub(super) fn cached_plan(
        &self,
        key: &CacheKey,
        params: Option<&HashMap<String, Value>>,
    ) -> Option<std::sync::Arc<CachedPlan>> {
        match params {
            None => self.plan_cache().get(key, Stage::Optimized),
            Some(_) => self.plan_cache().get(&key.prepared(), Stage::Optimized),
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
        store: &std::sync::Arc<dyn GraphStore>,
        bound_by_value: bool,
        generation: u64,
    ) -> Result<RunPlan> {
        use crate::query::binder::Binder;

        let optimizer = self.optimizer_for(store, key.language());
        let mut filled = plan.clone();
        fill_values(&mut filled, params)?;
        Binder::new().bind(&filled)?;
        if !bound_by_value {
            let mut binder = Binder::new();
            binder.bind(&plan)?;
            if !binder.binds_by_value() {
                let prepared = optimizer.optimize(plan)?;
                let prepared =
                    self.plan_cache()
                        .put(key.prepared(), Stage::Optimized, prepared, generation);
                return Self::fill_prepared(prepared, Some(params));
            }
            self.plan_cache()
                .put(key.clone(), Stage::Parsed, plan, generation);
        }
        optimizer.optimize(filled).map(RunPlan::Owned)
    }

    /// The plan a call runs from a cached plan: a prepared plan (a call with
    /// values `params`) filled with them, an optimized plan (a call without) as
    /// it is.
    ///
    /// # Errors
    ///
    /// As [`fill_values`].
    pub(super) fn fill_prepared(
        plan: std::sync::Arc<CachedPlan>,
        params: Option<&HashMap<String, Value>>,
    ) -> Result<RunPlan> {
        let Some(params) = params else {
            return Ok(RunPlan::Shared(plan));
        };
        // This call's copy: the cached plan stays as it is for the next.
        let mut filled = plan.plan().clone();
        fill_values(&mut filled, params)?;
        Ok(RunPlan::Owned(filled))
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

#[cfg(all(test, feature = "gql"))]
mod tests {
    use crate::query::optimizer::PlannerRules;
    use crate::query::processor::QueryLanguage;

    /// A session with other planner rules plans the query itself: the cache
    /// holds a plan per set of rules.
    #[test]
    fn sessions_with_other_planner_rules_do_not_share_a_cached_plan() {
        let db = crate::GrafeoDB::new_in_memory();
        let usual = db.session();
        let mut plain = db.session();
        plain.set_planner_rules(PlannerRules::plain());
        let query = "MATCH (n:Person) RETURN n";

        plain.execute(query).unwrap();

        let cached = |session: &crate::session::Session| {
            let key = session
                .inner()
                .unwrap()
                .plan_cache_key(query, QueryLanguage::Gql)
                .unwrap();
            session
                .inner()
                .unwrap()
                .plan_cache()
                .get(&key, crate::query::cache::Stage::Optimized)
                .is_some()
        };
        assert!(cached(&plain));
        assert!(!cached(&usual));
    }
}
