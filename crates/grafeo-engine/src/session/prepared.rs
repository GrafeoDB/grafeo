//! Prepared plans: a statement with values is optimized once with its
//! parameters in, cached, and each call fills a copy with its own values
//! (DuckDB's prepared statements). See [`Session::plan_with_values`].

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_common::utils::error::Result;
use grafeo_core::graph::GraphStore;

use super::Session;
use crate::query::cache::{CacheKey, CachedPlan, Stage};
use crate::query::plan::LogicalPlan;

/// The plan a statement runs: a cached plan, shared, or one made for this
/// run (filled with its values). Either reads as a [`LogicalPlan`].
pub(crate) enum RunPlan {
    /// A plan the cache holds or just stored.
    Shared(std::sync::Arc<CachedPlan>),
    /// A plan of this run alone.
    Owned(LogicalPlan),
    /// A prepared plan whose recipe this call opens with its values, defaults included
    /// (see [`Session::physical_plan`]); it reads as the plan with its parameters.
    Prepared(std::sync::Arc<CachedPlan>, HashMap<String, Value>),
}

impl std::ops::Deref for RunPlan {
    type Target = LogicalPlan;

    fn deref(&self) -> &LogicalPlan {
        match self {
            Self::Shared(cached) => cached.plan(),
            Self::Owned(plan) => plan,
            Self::Prepared(cached, _) => cached.plan(),
        }
    }
}

impl Session {
    /// The plan cache's key for `query` in `language` in this session: its
    /// current schema and graph and its planner rules, which the plan is made
    /// for.
    pub(super) fn plan_cache_key(
        &self,
        query: &str,
        language: crate::query::processor::QueryLanguage,
    ) -> CacheKey {
        CacheKey::in_scope(query, language, self.current_schema(), self.current_graph())
            .with_rules(self.plan_options.rules)
    }

    /// The cached plan of a statement in this session (`key`), with the values
    /// to fill it with ([`fill_prepared`](Self::fill_prepared)): for a call
    /// without values its optimized plan, for one with values `params` its
    /// prepared plan, which keeps the parameters (see `plan_with_values`).
    ///
    /// An empty map fills in nothing, so its call shares the plan of the call
    /// without one (as `params_to_fill` decides after a parse), found before
    /// any parse. Only GraphQL gives parameters defaults, and it does not come
    /// here. An EXPLAIN plan is not shared so: without a map, its parameters
    /// stay unresolved, where an empty map fails as missing them.
    pub(super) fn cached_plan<'a>(
        &self,
        key: &CacheKey,
        params: Option<&'a HashMap<String, Value>>,
    ) -> Option<(
        std::sync::Arc<CachedPlan>,
        Option<&'a HashMap<String, Value>>,
    )> {
        match params {
            None => self
                .plan_cache()
                .get(key, Stage::Optimized)
                .map(|plan| (plan, None)),
            Some(values) if values.is_empty() => self
                .plan_cache()
                .get(key, Stage::Optimized)
                .filter(|plan| !plan.plan().explain)
                .map(|plan| (plan, None)),
            Some(_) => self
                .plan_cache()
                .get(&key.prepared(), Stage::Optimized)
                .map(|plan| (plan, params)),
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
        // A call of a covered plan opens its recipe when the physical plan is made, with
        // no copy of the logical plan; EXPLAIN and PROFILE show the plan filled, as before.
        if !plan.plan().explain
            && !plan.plan().profile
            && let Ok(recipe) = plan.recipe()
            && let Some(values) = recipe.values_of(plan.plan(), params)
        {
            return Ok(RunPlan::Prepared(plan, values));
        }
        // This call's copy: the cached plan stays as it is for the next.
        let mut filled = plan.plan().clone();
        fill_values(&mut filled, params)?;
        Ok(RunPlan::Owned(filled))
    }
}

impl Session {
    /// The physical plan of `run` from `planner`: a prepared plan's recipe opened with the
    /// call's values, or, when the recipe sends this call back (a key of another kind), the
    /// plan filled with them and planned, as every other plan is.
    ///
    /// # Errors
    ///
    /// The errors of planning, and of filling the values.
    pub(super) fn physical_plan(
        run: &RunPlan,
        planner: &crate::query::Planner,
    ) -> Result<crate::query::planner::PhysicalPlan> {
        use crate::query::planner::lpg::recipe::OpenContext;

        let RunPlan::Prepared(cached, values) = run else {
            return planner.plan(run);
        };
        if let Ok(recipe) = cached.recipe()
            && let Some(opened) = recipe.open(cached.plan(), &OpenContext { planner, values })?
        {
            return Ok(opened);
        }
        let mut filled = cached.plan().clone();
        crate::query::processor::substitute_params(&mut filled, values)?;
        planner.plan(&filled)
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

        let cached = |session: &super::Session| {
            let key = session.plan_cache_key(query, QueryLanguage::Gql);
            session
                .plan_cache()
                .get(&key, crate::query::cache::Stage::Optimized)
                .is_some()
        };
        assert!(cached(&plain));
        assert!(!cached(&usual));
    }

    /// The prepared plan the first call of `query` with `params` caches, with its
    /// parameters still in: the plan every later call fills.
    fn prepared_plan(
        query: &str,
        params: &std::collections::HashMap<String, grafeo_common::types::Value>,
    ) -> std::sync::Arc<crate::query::cache::CachedPlan> {
        let db = crate::GrafeoDB::new_in_memory();
        db.execute("INSERT (:Person {id: 1, firstName: 'Alix'})-[:LIVES_IN]->(:City {id: 10})")
            .unwrap();
        let session = db.session();
        session.execute_with_params(query, params.clone()).unwrap();
        let key = session.plan_cache_key(query, QueryLanguage::Gql);
        session
            .plan_cache()
            .get(&key.prepared(), crate::query::cache::Stage::Optimized)
            .expect("the first call caches its prepared plan")
    }

    /// IS1's and IS3's shapes (a start a parameter pins, one hop, a projection) make a
    /// recipe; a plan with a kind outside the covered set names that kind.
    #[test]
    fn the_covered_shapes_make_a_recipe() {
        use crate::query::planner::lpg::recipe::recipe_of;

        let params = std::collections::HashMap::from([(
            "personId".to_string(),
            grafeo_common::types::Value::Int64(1),
        )]);
        for query in [
            "MATCH (n:Person {id: $personId})-[:LIVES_IN]->(c:City) RETURN n.firstName AS firstName, c.id AS cityId",
            "MATCH (n:Person {id: $personId})-[k:KNOWS]-(friend:Person) RETURN friend.id AS friendId, k.creationDate AS since",
        ] {
            let plan = prepared_plan(query, &params);
            assert!(recipe_of(plan.plan()).is_ok(), "{query}");
        }
    }

    #[test]
    fn an_uncovered_kind_names_itself() {
        use crate::query::planner::lpg::recipe::recipe_of;

        let params = std::collections::HashMap::from([(
            "personId".to_string(),
            grafeo_common::types::Value::Int64(1),
        )]);
        for (query, kind) in [
            (
                "MATCH (n:Person {id: $personId})-[:KNOWS]-(f:Person) RETURN f.id AS id ORDER BY id",
                "Sort",
            ),
            (
                "MATCH (n:Person {id: $personId})-[:KNOWS]-(f:Person) RETURN count(f) AS friends",
                "Aggregate",
            ),
            (
                "MATCH (n:Person {id: $personId})-[:KNOWS]-{1,2}(f:Person) RETURN f.id AS id",
                "VariableLengthExpand",
            ),
        ] {
            let plan = prepared_plan(query, &params);
            match recipe_of(plan.plan()) {
                Err(uncovered) => assert_eq!(uncovered.kind, kind, "{query}"),
                Ok(_) => panic!("{query} must not make a recipe"),
            }
        }
    }

    /// The kinds the recipes of IS1's shape and of an unpinned hop hold are the pinned covered
    /// set: a step added without being named there fails here.
    #[test]
    fn the_covered_set_is_pinned() {
        use crate::query::planner::lpg::recipe::{COVERED, recipe_of};

        let params = std::collections::HashMap::from([(
            "personId".to_string(),
            grafeo_common::types::Value::Int64(1),
        )]);
        let mut kinds: Vec<&str> = [
            "MATCH (n:Person {id: $personId})-[:LIVES_IN]->(c:City) RETURN n.firstName AS firstName, c.id AS cityId",
            "MATCH (n:Person)-[:LIVES_IN]->(c:City) RETURN c.id AS cityId",
        ]
        .into_iter()
        .flat_map(|query| recipe_of(prepared_plan(query, &params).plan()).unwrap().kinds())
        .collect();
        kinds.sort_unstable();
        kinds.dedup();
        let mut covered = COVERED.to_vec();
        covered.sort_unstable();
        assert_eq!(kinds, covered);
    }

    /// Opening IS1's recipe gives, for every id, the rows of today's plan of the filled
    /// statement: people with a city, one without, and an id nobody has.
    #[test]
    fn open_gives_the_rows_of_todays_plan() {
        use crate::query::planner::lpg::recipe::{OpenContext, recipe_of};
        use grafeo_common::types::{StatementLanguage, Value};
        use std::collections::HashMap;

        let db = crate::GrafeoDB::new_in_memory();
        db.execute(
            "INSERT (:Person {id: 1, firstName: 'Alix'})-[:LIVES_IN]->(:City {id: 10}),              (:Person {id: 2, firstName: 'Gus'})-[:LIVES_IN]->(:City {id: 20}),              (:Person {id: 3, firstName: 'Vincent'})",
        )
        .unwrap();
        let session = db.session();
        let query = "MATCH (n:Person {id: $personId})-[:LIVES_IN]->(c:City)                      RETURN n.firstName AS firstName, c.id AS cityId";
        let values = |id: i64| HashMap::from([("personId".to_string(), Value::Int64(id))]);
        session.execute_with_params(query, values(1)).unwrap();
        let key = session.plan_cache_key(query, QueryLanguage::Gql);
        let cached = session
            .plan_cache()
            .get(&key.prepared(), crate::query::cache::Stage::Optimized)
            .unwrap();
        let recipe = recipe_of(cached.plan()).unwrap();
        let rows = |mut plan: crate::query::planner::PhysicalPlan| {
            let result = crate::query::executor::Executor::with_columns(plan.columns.clone())
                .execute(plan.operator.as_mut())
                .unwrap();
            let mut rows: Vec<String> =
                result.rows().iter().map(|row| format!("{row:?}")).collect();
            rows.sort();
            rows
        };
        for id in [1, 2, 3, 4] {
            // A statement's operators are built inside its memory, as the session builds them.
            let _memory = session.enter_statement_memory();
            let planner = || {
                let (epoch, transaction, _pin) = session.get_transaction_context().unwrap();
                session
                    .create_planner_for_store_with_read_only(
                        session.active_store(),
                        epoch,
                        transaction,
                        true,
                        StatementLanguage::Gql,
                    )
                    .unwrap()
            };
            let opened = recipe
                .open(
                    cached.plan(),
                    &OpenContext {
                        planner: &planner(),
                        values: &values(id),
                    },
                )
                .unwrap()
                .expect("an integer key opens");
            // Today's path: a copy of the plan filled with the values, then planned.
            let mut filled = cached.plan().clone();
            crate::query::processor::substitute_params(&mut filled, &values(id)).unwrap();
            let today = planner().plan(&filled).unwrap();
            assert_eq!(rows(opened), rows(today), "id {id}");
        }
    }
}
