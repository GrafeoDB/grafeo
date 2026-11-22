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
    /// A prepared plan whose recipe this call opens with its values, defaults included
    /// (see [`SessionInner::physical_plan`]); it reads as the plan with its parameters.
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
                return self.fill_prepared(key, prepared, Some(params));
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
        &self,
        key: &CacheKey,
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
            && let Ok(recipe) = plan.recipe(|| self.route_inputs(key))
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

impl SessionInner {
    /// What the route of an aggregate in a statement with cache key `key` reads besides the
    /// plan, as this session's planner reads it: the key's language and rules, factorized
    /// execution, and the backward adjacency of the store the statement reads.
    pub(super) fn route_inputs(
        &self,
        key: &CacheKey,
    ) -> crate::query::planner::lpg::aggregate::RouteInputs {
        crate::query::planner::lpg::aggregate::RouteInputs {
            language: key.language().statement_language(),
            rules: key.rules(),
            factorized_execution: self.plan_options.factorized_execution_under(key.rules()),
            backward_adjacency: self.active_store().has_backward_adjacency(),
        }
    }

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
        if let Ok(recipe) = cached.recipe(|| planner.route_inputs())
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

    /// The route inputs of a GQL statement in a session of an in-memory database.
    fn inputs() -> crate::query::planner::lpg::aggregate::RouteInputs {
        let db = crate::GrafeoDB::new_in_memory();
        let session = db.session();
        let inner = session.inner().unwrap();
        inner.route_inputs(&inner.plan_cache_key("", QueryLanguage::Gql).unwrap())
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
        let key = session
            .inner()
            .unwrap()
            .plan_cache_key(query, QueryLanguage::Gql)
            .unwrap();
        session
            .inner()
            .unwrap()
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
            "MATCH (n:Person {id: $personId})-[k:KNOWS]-(friend:Person) RETURN friend.id AS friendId, k.creationDate AS since ORDER BY since DESC, friendId ASC",
            "MATCH (:Person {id: $personId})<-[:HAS_CREATOR]-(m:Message) RETURN m.id AS messageId, m.creationDate AS created ORDER BY created DESC, messageId ASC LIMIT 10",
            "MATCH (:Person {id: $personId})<-[:HAS_CREATOR]-(m:Message) RETURN m.id AS messageId ORDER BY m.creationDate DESC",
            "MATCH (:Person {id: $personId})<-[:HAS_CREATOR]-(m:Message) RETURN m.id AS messageId SKIP 1 LIMIT 2",
            "MATCH (p:Person {id: $personId}) RETURN count(p) AS n",
        ] {
            let plan = prepared_plan(query, &params);
            assert!(recipe_of(plan.plan(), inputs()).is_ok(), "{query}");
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
                "MATCH (n:Person {id: $personId})-[:KNOWS]-(f:Person) RETURN f.id AS id ORDER BY abs(f.id)",
                "SortKeys",
            ),
            (
                "MATCH (n:Person {id: $personId})-[:KNOWS]-(f:Person) RETURN f.id AS id ORDER BY cosine_similarity(f.embedding, [1.0, 0.0]) DESC LIMIT 3",
                "SortKeys",
            ),
            (
                "MATCH (n:Person {id: $personId})<-[:HAS_CREATOR]-(m:Message) RETURN m.id + $personId AS x ORDER BY m.creationDate",
                "ReturnItems",
            ),
            (
                "MATCH (n:Person) WHERE n.id > $personId RETURN count(*) AS c",
                "RangeFold",
            ),
            (
                "MATCH (n:Person {id: $personId})-[:KNOWS]-(f:Person) RETURN f.id AS id GROUP BY id HAVING count(*) > 1",
                "Having",
            ),
            (
                "MATCH (n:Person {id: $personId})-[:KNOWS]-(f:Person) RETURN sum(f.id * 2) AS s",
                "AggregateItems",
            ),
            (
                "MATCH (n:Person {id: $personId})-[:KNOWS]-{1,2}(f:Person) RETURN f.id AS id",
                "VariableLengthExpand",
            ),
            (
                "MATCH (n:Person) WHERE n.id + 1 = $personId RETURN n.id AS id",
                "FilterReadsTheStore",
            ),
            (
                "MATCH (n:Person {id: $personId})-[:KNOWS]-(f:Person) WHERE EXISTS { MATCH (f)-[:KNOWS]->(:Person) } RETURN f.id AS id",
                "FilterReadsTheStore",
            ),
        ] {
            let plan = prepared_plan(query, &params);
            match recipe_of(plan.plan(), inputs()) {
                Err(uncovered) => assert_eq!(uncovered.kind, kind, "{query}"),
                Ok(_) => panic!("{query} must not make a recipe"),
            }
        }
    }

    /// The kinds the recipes of IS1's shape, of an unpinned hop and of a triangle hold are the
    /// pinned covered set: a step added without being named there fails here.
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
            "MATCH (n:Person)-[:LIVES_IN]->(c:City) RETURN c.id AS cityId ORDER BY cityId SKIP 1 LIMIT 2",
            "MATCH (n:Person {id: $personId})-[:KNOWS]-(f:Person) RETURN count(f) AS friends",
            "MATCH (a:Person {id: $personId})-[:KNOWS]-(b:Person)-[:KNOWS]-(c:Person)-[:KNOWS]-(a) WHERE b.id < c.id RETURN count(*) AS triangles",
        ]
        .into_iter()
        .flat_map(|query| recipe_of(prepared_plan(query, &params).plan(), inputs()).unwrap().kinds())
        .collect();
        kinds.sort_unstable();
        kinds.dedup();
        let mut covered = COVERED.to_vec();
        covered.sort_unstable();
        assert_eq!(kinds, covered);

        // The filters over hops make a recipe, in a run without work counters too: the square,
        // common friends with the far end's key indexed, and a bound zone maps decide (Cypher's
        // two-hop chain: `the_cypher_chain_is_pinned`).
        let person = |id: i64| {
            (
                "personId".to_string(),
                grafeo_common::types::Value::Int64(id),
            )
        };
        let pins: [(&str, std::collections::HashMap<String, _>); 3] = [
            (
                "MATCH (a:Person {id: $personId})-[:KNOWS]-(b:Person)-[:KNOWS]-(c:Person)-[:KNOWS]-(d:Person)-[:KNOWS]-(a) WHERE b.id < d.id AND c.id <> $personId RETURN count(*) AS cycles",
                std::collections::HashMap::from([person(1)]),
            ),
            (
                "MATCH (a:Person {id: $person1Id})-[:KNOWS]-(c:Person)-[:KNOWS]-(b:Person {id: $person2Id}) RETURN count(DISTINCT c.id) AS common",
                std::collections::HashMap::from([
                    (
                        "person1Id".to_string(),
                        grafeo_common::types::Value::Int64(1),
                    ),
                    (
                        "person2Id".to_string(),
                        grafeo_common::types::Value::Int64(2),
                    ),
                ]),
            ),
            (
                "MATCH (a:Person {id: $personId})-[:KNOWS]-(b:Person) WHERE b.age > $min RETURN b.id AS id",
                std::collections::HashMap::from([
                    person(1),
                    ("min".to_string(), grafeo_common::types::Value::Int64(100)),
                ]),
            ),
        ];
        for (query, params) in pins {
            assert_a_recipe(QueryLanguage::Gql, "gql", true, query, params);
        }
    }

    /// Cypher's two-hop chain with its check that the edges differ (factorized execution off)
    /// makes a recipe, in a run without work counters too.
    #[cfg(feature = "cypher")]
    #[test]
    fn the_cypher_chain_is_pinned() {
        assert_a_recipe(
            QueryLanguage::Cypher,
            "cypher",
            false,
            "MATCH (a:Person)-[:KNOWS]->(m:Person)-[:KNOWS]->(b:Person) RETURN count(*) AS paths",
            std::collections::HashMap::from([(
                "personId".to_string(),
                grafeo_common::types::Value::Int64(1),
            )]),
        );
    }

    /// `query` with `params` in `language` (named `name`), run once on two people with a
    /// property index on `id` (factorized execution as `factorized` says), caches a prepared
    /// plan that makes a recipe.
    fn assert_a_recipe(
        language: QueryLanguage,
        name: &str,
        factorized: bool,
        query: &str,
        params: std::collections::HashMap<String, grafeo_common::types::Value>,
    ) {
        use crate::query::planner::lpg::recipe::recipe_of;

        let config = crate::config::Config::in_memory();
        let config = if factorized {
            config
        } else {
            config.without_factorized_execution()
        };
        let db = crate::GrafeoDB::with_config(config).unwrap();
        db.execute("INSERT (:Person {id: 1, age: 30})-[:KNOWS]->(:Person {id: 2, age: 40})")
            .unwrap();
        db.create_property_index("id").unwrap();
        let session = db.session();
        session.execute_language(query, name, Some(params)).unwrap();
        let key = session
            .inner()
            .unwrap()
            .plan_cache_key(query, language)
            .unwrap();
        let cached = session
            .inner()
            .unwrap()
            .plan_cache()
            .get(&key.prepared(), crate::query::cache::Stage::Optimized)
            .expect("the first call caches its prepared plan");
        let recipe = recipe_of(cached.plan(), session.inner().unwrap().route_inputs(&key));
        assert!(recipe.is_ok(), "{query}: {recipe:?}");
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
            "INSERT (:Person {id: 1, firstName: 'Alix'})-[:LIVES_IN]->(:City {id: 10}), (:Person {id: 2, firstName: 'Gus'})-[:LIVES_IN]->(:City {id: 20}), (:Person {id: 3, firstName: 'Vincent'})",
        )
        .unwrap();
        let session = db.session();
        let query = "MATCH (n:Person {id: $personId})-[:LIVES_IN]->(c:City) RETURN n.firstName AS firstName, c.id AS cityId";
        let values = |id: i64| HashMap::from([("personId".to_string(), Value::Int64(id))]);
        session.execute_with_params(query, values(1)).unwrap();
        let key = session
            .inner()
            .unwrap()
            .plan_cache_key(query, QueryLanguage::Gql)
            .unwrap();
        let cached = session
            .inner()
            .unwrap()
            .plan_cache()
            .get(&key.prepared(), crate::query::cache::Stage::Optimized)
            .unwrap();
        let recipe = recipe_of(cached.plan(), session.inner().unwrap().route_inputs(&key)).unwrap();
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
            let _memory = session.inner().unwrap().enter_statement_memory();
            let planner = || {
                let (epoch, transaction, _pin) =
                    session.inner().unwrap().get_transaction_context().unwrap();
                session
                    .inner()
                    .unwrap()
                    .create_planner_for_store_with_read_only(
                        session.inner().unwrap().active_store(),
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

    /// Opening a recipe whose LIMIT is a parameter builds, for each count, the root operator
    /// today's planner builds for the filled plan: a top-k for a count above 0, a plain limit
    /// over the sort for 0.
    #[test]
    fn open_takes_a_top_k_where_todays_plan_does() {
        use crate::query::planner::lpg::recipe::{OpenContext, recipe_of};
        use grafeo_common::types::{StatementLanguage, Value};
        use std::collections::HashMap;

        let db = crate::GrafeoDB::new_in_memory();
        db.execute(
            "INSERT (alix:Person {id: 1}), (:Message {id: 101, at: 2})-[:HAS_CREATOR]->(alix), (:Message {id: 102, at: 1})-[:HAS_CREATOR]->(alix)",
        )
        .unwrap();
        let session = db.session();
        let query = "MATCH (:Person {id: $personId})<-[:HAS_CREATOR]-(m:Message) RETURN m.id AS id ORDER BY m.at LIMIT $n";
        let values = |count: i64| {
            HashMap::from([
                ("personId".to_string(), Value::Int64(1)),
                ("n".to_string(), Value::Int64(count)),
            ])
        };
        session.execute_with_params(query, values(1)).unwrap();
        let key = session
            .inner()
            .unwrap()
            .plan_cache_key(query, QueryLanguage::Gql)
            .unwrap();
        let cached = session
            .inner()
            .unwrap()
            .plan_cache()
            .get(&key.prepared(), crate::query::cache::Stage::Optimized)
            .unwrap();
        let recipe = recipe_of(cached.plan(), session.inner().unwrap().route_inputs(&key)).unwrap();
        for count in [0, 1, 3] {
            let _memory = session.inner().unwrap().enter_statement_memory();
            let planner = || {
                let (epoch, transaction, _pin) =
                    session.inner().unwrap().get_transaction_context().unwrap();
                session
                    .inner()
                    .unwrap()
                    .create_planner_for_store_with_read_only(
                        session.inner().unwrap().active_store(),
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
                        values: &values(count),
                    },
                )
                .unwrap()
                .expect("an integer key opens");
            let mut filled = cached.plan().clone();
            crate::query::processor::substitute_params(&mut filled, &values(count)).unwrap();
            let today = planner().plan(&filled).unwrap();
            assert_eq!(
                opened.operator.name(),
                today.operator.name(),
                "LIMIT {count}"
            );
        }
    }

    /// The aggregate at the top of `plan`, under its RETURN.
    fn aggregate_of(plan: &crate::query::plan::LogicalPlan) -> &crate::query::plan::AggregateOp {
        let crate::query::plan::LogicalOperator::Return(ret) = &plan.root else {
            panic!("expected a RETURN: {:?}", plan.root);
        };
        let crate::query::plan::LogicalOperator::Aggregate(agg) = ret.input.as_ref() else {
            panic!("expected an aggregate: {:?}", ret.input);
        };
        agg
    }

    /// The route a recipe decides from its inputs alone is the route the planner of a call
    /// takes on the plan filled with the call's values, for each route and both settings of
    /// factorized execution; a range fold whose bound is a parameter has no recipe route.
    #[test]
    fn the_recipe_route_is_the_planners() {
        use crate::query::planner::lpg::recipe::recipe_route;
        use grafeo_common::types::{StatementLanguage, Value};

        let db = crate::GrafeoDB::new_in_memory();
        db.execute(
            "INSERT (:Person {id: 1})-[:KNOWS]->(:Person {id: 2})-[:KNOWS]->(:Person {id: 3})",
        )
        .unwrap();
        let session = db.session();
        let params = std::collections::HashMap::from([("personId".to_string(), Value::Int64(1))]);
        for query in [
            "MATCH (n:Person {id: $personId})-[:KNOWS]-(f:Person) RETURN count(f) AS friends",
            "MATCH (n:Person {id: $personId})-[:KNOWS]-(f:Person) RETURN count(f.id) AS ids",
            "MATCH (a:Person)-[:KNOWS]->(m:Person)-[:KNOWS]->(b:Person) RETURN count(*) AS paths",
            "MATCH (n:Person) RETURN count(n) AS people",
            "MATCH (n:Person) WHERE n.id > 1 RETURN count(*) AS later",
            "MATCH (n:Person) WHERE n.id > TIME('10:00:00') RETURN count(*) AS later",
        ] {
            session.execute_with_params(query, params.clone()).unwrap();
            let key = session
                .inner()
                .unwrap()
                .plan_cache_key(query, QueryLanguage::Gql)
                .unwrap();
            let cached = session
                .inner()
                .unwrap()
                .plan_cache()
                .get(&key.prepared(), crate::query::cache::Stage::Optimized)
                .unwrap();
            let inputs = session.inner().unwrap().route_inputs(&key);
            let mut filled = cached.plan().clone();
            crate::query::processor::substitute_params(&mut filled, &params).unwrap();
            for factorized in [true, false] {
                let inputs = crate::query::planner::lpg::aggregate::RouteInputs {
                    factorized_execution: factorized,
                    ..inputs
                };
                let (epoch, transaction, _pin) =
                    session.inner().unwrap().get_transaction_context().unwrap();
                let planner = session
                    .inner()
                    .unwrap()
                    .create_planner_for_store_with_read_only(
                        session.inner().unwrap().active_store(),
                        epoch,
                        transaction,
                        true,
                        StatementLanguage::Gql,
                    )
                    .unwrap()
                    .with_factorized_execution(factorized);
                assert_eq!(
                    recipe_route(inputs, aggregate_of(cached.plan())),
                    Some(planner.route_kind_of(aggregate_of(&filled))),
                    "{query}, factorized {factorized}"
                );
            }
        }
        let query = "MATCH (n:Person) WHERE n.id > $personId RETURN count(*) AS later";
        session.execute_with_params(query, params.clone()).unwrap();
        let key = session
            .inner()
            .unwrap()
            .plan_cache_key(query, QueryLanguage::Gql)
            .unwrap();
        let cached = session
            .inner()
            .unwrap()
            .plan_cache()
            .get(&key.prepared(), crate::query::cache::Stage::Optimized)
            .unwrap();
        assert_eq!(
            recipe_route(
                session.inner().unwrap().route_inputs(&key),
                aggregate_of(cached.plan())
            ),
            None
        );
    }

    /// Statements in SQL/PGQ, whose parser the build may leave out.
    #[cfg(feature = "sql-pgq")]
    mod sql {
        use super::*;

        /// A SQL/PGQ statement that names its own graph has no recipe (the route's inputs are the
        /// session graph's), and a SQL/PGQ range fold answers that it is not covered.
        #[test]
        fn sql_statements_answer_uncovered() {
            use crate::query::planner::lpg::recipe::recipe_of;
            use grafeo_common::types::Value;

            let db = crate::GrafeoDB::new_in_memory();
            db.execute("CREATE GRAPH g").unwrap();
            db.execute("INSERT (:Person {id: 1}), (:Person {id: 2})")
                .unwrap();
            let session = db.session();
            let params =
                std::collections::HashMap::from([("personId".to_string(), Value::Int64(1))]);
            for (query, kind) in [
                (
                    "SELECT count(*) AS c FROM GRAPH_TABLE (g MATCH (n:Person) COLUMNS (n.id AS id))",
                    "PlanGraph",
                ),
                (
                    "SELECT count(*) AS c FROM GRAPH_TABLE (MATCH (n:Person) WHERE n.id > 1 COLUMNS (n.id AS id))",
                    "RangeFold",
                ),
            ] {
                session
                    .execute_sql_with_params(query, params.clone())
                    .unwrap();
                let key = session
                    .inner()
                    .unwrap()
                    .plan_cache_key(query, QueryLanguage::SqlPgq)
                    .unwrap();
                let cached = session
                    .inner()
                    .unwrap()
                    .plan_cache()
                    .get(&key.prepared(), crate::query::cache::Stage::Optimized)
                    .expect("the call caches its prepared plan");
                match recipe_of(cached.plan(), session.inner().unwrap().route_inputs(&key)) {
                    Err(uncovered) => assert_eq!(uncovered.kind, kind, "{query}"),
                    Ok(recipe) => panic!("{query} made a recipe: {:?}", recipe.kinds()),
                }
            }
        }
    }

    /// The factorized aggregate's only reachable input, a procedure call without YIELD, is no
    /// recipe's: the aggregate answers that its route is not covered.
    #[test]
    fn a_factorized_count_answers_uncovered() {
        use crate::query::planner::lpg::recipe::recipe_of;
        use grafeo_common::types::Value;

        let db = crate::GrafeoDB::new_in_memory();
        db.execute(
            "INSERT (a:P {name: 'Alix'})-[:R]->(g:P {name: 'Gus'}), (g)-[:R]->(v:P {name: 'Vincent'}), (v)-[:R]->(a)",
        )
        .unwrap();
        let session = db.session();
        let query = "CALL grafeo.procedures() MATCH (p:P)-[:R]->(b)-[:R]->(c) RETURN count(*) AS n";
        session
            .execute_with_params(
                query,
                std::collections::HashMap::from([("unused".to_string(), Value::Int64(1))]),
            )
            .unwrap();
        let key = session
            .inner()
            .unwrap()
            .plan_cache_key(query, QueryLanguage::Gql)
            .unwrap();
        let cached = session
            .inner()
            .unwrap()
            .plan_cache()
            .get(&key.prepared(), crate::query::cache::Stage::Optimized)
            .unwrap();
        match recipe_of(cached.plan(), session.inner().unwrap().route_inputs(&key)) {
            Err(uncovered) => assert_eq!(uncovered.kind, "Factorized"),
            Ok(recipe) => panic!("a recipe: {:?}", recipe.kinds()),
        }
    }
}

/// The route each filter of a statement takes in the planner, as the planner records it: one
/// statement per route, the builds that decline and the next route they hand the filter to,
/// and the guards that keep a route out. Each statement also answers as without the planner's
/// rules.
#[cfg(all(
    test,
    feature = "gql",
    feature = "cypher",
    feature = "sql-pgq",
    feature = "vector-index",
    feature = "text-index"
))]
mod filter_route_tests {
    use crate::query::optimizer::PlannerRules;
    use crate::query::processor::QueryLanguage;

    /// People `k` 1 to 4 with names, bios and embeddings, a chain of `R` steps with weights,
    /// things `Q` whose `k` matches some people's, `S` nodes whose indexed `name` holds a
    /// string and an integer, and twelve `Z` nodes; a property index on `P.name` and
    /// `S.name`, a text index on `bio` and a vector index on `embedding`.
    fn graph() -> crate::GrafeoDB {
        let db = crate::GrafeoDB::new_in_memory();
        db.execute(
            "INSERT (a:P {name: 'Alix', k: 1, t: 1, bio: 'graph canals', embedding: [1.0, 0.0]}), \
                    (g:P {name: 'Gus', k: 2, t: 1, bio: 'canals', embedding: [0.0, 1.0]}), \
                    (v:P {name: 'Vincent', k: 3, t: 1, bio: 'bikes', embedding: [1.0, 1.0]}), \
                    (j:P {name: 'Jules', k: 4, t: 1, bio: 'graph', embedding: [0.5, 0.5]}), \
                    (a)-[:R {w: 1}]->(g), (g)-[:R {w: 2}]->(v), (v)-[:R {w: 3}]->(a), (g)-[:R {w: 4}]->(j), \
                    (:Q {k: 2, name: 'Gus'}), (:Q {k: 4, name: 'Jules'}), (:Q {k: 9, name: 'Mia'}), \
                    (:S {name: 'Gus', k: 1}), (:S {name: 'Mia', k: 2}), (:S {name: 7, k: 3}), \
                    (:Z), (:Z), (:Z), (:Z), (:Z), (:Z), (:Z), (:Z), (:Z), (:Z), (:Z), (:Z)",
        )
        .unwrap();
        for ddl in [
            "CREATE INDEX p_name FOR (p:P) ON (p.name)",
            "CREATE INDEX s_name FOR (s:S) ON (s.name)",
            "CREATE INDEX p_bio FOR (p:P) ON (p.bio) USING TEXT",
            "CREATE VECTOR INDEX p_embedding ON :P(embedding) DIMENSION 2 METRIC 'cosine'",
        ] {
            db.execute(ddl).unwrap();
        }
        db
    }

    /// How a case's session is set up.
    #[derive(Clone, Copy)]
    enum Setup {
        /// Every rule on.
        Usual,
        /// No rule on (`PlannerRules::plain`).
        Plain,
        /// A session time zone: a local time equals zoned ones at any offset.
        TimeZone,
        /// Every rule but `SoughtEnd`: a pattern starts where it is written.
        WrittenStart,
    }

    /// A session of `db` set up as `setup` says.
    fn session(db: &crate::GrafeoDB, setup: Setup) -> super::super::Session {
        let mut session = db.session();
        match setup {
            Setup::Usual => {}
            Setup::Plain => session.set_planner_rules(PlannerRules::plain()),
            Setup::TimeZone => session.set_time_zone("+01:00").unwrap(),
            Setup::WrittenStart => session.set_planner_rules(
                PlannerRules::all().with(crate::query::optimizer::PlannerRule::SoughtEnd, false),
            ),
        }
        session
    }

    /// The routes the filters of `query` in `language` take: its optimized plan, planned by a
    /// new planner of `session`, as the session plans a statement.
    fn routes_of(session: &mut super::super::Session, language: &str, query: &str) -> Vec<String> {
        use crate::query::translators::{cypher, gql, sql_pgq};

        let (plan, language) = match language {
            "gql" => match gql::translate_full(
                query,
                &grafeo_adapters::query::limits::ParseLimits::default(),
            )
            .unwrap()
            {
                gql::GqlTranslationResult::Plan(plan) => (plan, QueryLanguage::Gql),
                _ => panic!("{query} is no query"),
            },
            "cypher" => match cypher::translate_full(
                query,
                &grafeo_adapters::query::limits::ParseLimits::default(),
            )
            .unwrap()
            {
                cypher::CypherTranslationResult::Plan(plan) => (plan, QueryLanguage::Cypher),
                _ => panic!("{query} is no query"),
            },
            _ => (
                sql_pgq::translate(
                    query,
                    &grafeo_adapters::query::limits::ParseLimits::default(),
                )
                .unwrap(),
                QueryLanguage::SqlPgq,
            ),
        };
        let mut inner = session.inner_mut().unwrap();
        let session = &mut *inner;
        let plan = session.translated(plan).unwrap();
        crate::query::binder::Binder::new().bind(&plan).unwrap();
        let active = session.active_store();
        let plan = session
            .optimizer_for(&active, language)
            .optimize(plan)
            .unwrap();
        // A write is planned in a transaction, which nothing runs and which rolls back.
        let read_only = !session.plan_writes(&plan.root);
        if !read_only {
            session.begin_transaction().unwrap();
        }
        let routes = {
            let _memory = session.enter_statement_memory();
            let (epoch, transaction, _pin) = session.get_transaction_context().unwrap();
            let planner = session
                .create_planner_for_store_with_read_only(
                    active,
                    epoch,
                    transaction,
                    read_only,
                    language.statement_language(),
                )
                .unwrap();
            planner.plan(&plan).unwrap();
            planner.filter_routes_taken.take()
        };
        if !read_only {
            session.rollback().unwrap();
        }
        routes
    }

    /// The rows of `query` in `language` on a new graph, in a session set up as `setup`,
    /// sorted.
    fn rows_of(setup: Setup, language: &str, query: &str) -> Result<Vec<String>, String> {
        let db = graph();
        let result = session(&db, setup)
            .execute_language(query, language, None)
            .map_err(|error| error.to_string())?;
        let mut rows: Vec<String> = result.rows().iter().map(|row| format!("{row:?}")).collect();
        rows.sort();
        Ok(rows)
    }

    /// A statement, the session it runs in, and the routes its filters take, in the order the
    /// planner builds them (a filter planned inside another's build after it).
    struct Case {
        name: &'static str,
        setup: Setup,
        language: &'static str,
        query: &'static str,
        routes: &'static [&'static str],
    }

    const CASES: &[Case] = &[
        Case {
            name: "exists as a semi join",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (a:P) WHERE EXISTS { MATCH (b:P)-[:R]->(a) } RETURN a.name AS name",
            routes: &["ExistsJoin"],
        },
        Case {
            name: "exists as a semi join, the rest of the predicate planned again",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (a:P) WHERE a.k > 1 AND EXISTS { MATCH (b:P)-[:R]->(a)-[:R]->(c) } RETURN a.name AS name",
            routes: &[
                "ExistsJoin",
                "ZoneMapEmpty",
                "ZoneMapEmpty declined",
                "RangeIndex",
            ],
        },
        Case {
            name: "exists checked inline",
            setup: Setup::Usual,
            language: "cypher",
            query: "MATCH (a:P) WHERE EXISTS { MATCH (a)-[:R]->(b)-[:R]->(c) } RETURN a.name AS name",
            routes: &["ZoneMapEmpty", "ZoneMapEmpty declined", "Generic"],
        },
        Case {
            name: "subquery per row",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (a:P) WHERE COUNT { MATCH (a)-[:R]->(b)-[:R]->(c) } > 1 RETURN a.name AS name",
            routes: &["LiftedSubqueries"],
        },
        Case {
            name: "subquery sharing no variable",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (a:P) WHERE COUNT { MATCH (b:Q) } > 1 RETURN a.name AS name",
            routes: &["LiftedSubqueries"],
        },
        Case {
            name: "intersection closing",
            setup: Setup::Usual,
            language: "cypher",
            query: "MATCH (a:P {k: 1})-[:R]-(b:P)-[:R]-(c:P)-[:R]-(a) RETURN count(*) AS n",
            routes: &[
                "ZoneMapEmpty",
                "ZoneMapEmpty declined",
                "Generic",
                "ZoneMapEmpty",
                "ZoneMapEmpty declined",
                "Generic",
                "hop target",
                "IntersectionClosing",
                "ZoneMapEmpty",
                "ZoneMapEmpty declined",
                "Generic",
                "ZoneMapEmpty",
                "ZoneMapEmpty declined",
                "PropertyIndex",
            ],
        },
        Case {
            name: "zone map excludes",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (a:P)-[:R]->(b) WHERE b.k > 100 RETURN a.name AS name",
            routes: &["ZoneMapEmpty"],
        },
        Case {
            name: "zone map keeps",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (a:P)-[:R]->(b) WHERE b.k > 2 RETURN a.name AS name",
            routes: &["ZoneMapEmpty", "ZoneMapEmpty declined", "Generic"],
        },
        Case {
            name: "edge condition into a path",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH p = (a:P {k: 1})-[:R]->{1,3}(b) WHERE all(e IN edges(p) WHERE e.w > 0) RETURN b.name AS name",
            routes: &[
                "ZoneMapEmpty",
                "ZoneMapEmpty declined",
                "EdgeConditionIntoPath",
                "ZoneMapEmpty",
                "ZoneMapEmpty declined",
                "PropertyIndex",
            ],
        },
        Case {
            name: "seek by an indexed key",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (q:Q) MATCH (a:P) WHERE a.name = q.name RETURN a.k AS k",
            routes: &["ZoneMapEmpty", "ZoneMapEmpty declined", "NodeSeek"],
        },
        Case {
            name: "seek by an id",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (a:P)-[:R]->(b) MATCH (c:P) WHERE id(c) = id(b) RETURN c.name AS name",
            routes: &["ZoneMapEmpty", "ZoneMapEmpty declined", "NodeSeek"],
        },
        Case {
            name: "seek of a variable the input binds",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (a:P) CALL { WITH a MATCH (a:P) WHERE a.name = 'Gus' RETURN a.k AS k2 } RETURN a.k AS k",
            routes: &["ZoneMapEmpty", "ZoneMapEmpty declined", "NodeSeek"],
        },
        Case {
            name: "key without an index per row joins",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (q:Q) MATCH (a:P) WHERE a.k = q.k RETURN a.name AS name",
            routes: &["ZoneMapEmpty", "ZoneMapEmpty declined", "ValueJoin"],
        },
        Case {
            name: "value join",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (q:Q), (a:P) WHERE a.k = q.k RETURN a.name AS name",
            routes: &["ZoneMapEmpty", "ZoneMapEmpty declined", "ValueJoin"],
        },
        Case {
            name: "property index",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (a:P) WHERE a.name = 'Gus' RETURN a.k AS k",
            routes: &["ZoneMapEmpty", "ZoneMapEmpty declined", "PropertyIndex"],
        },
        Case {
            name: "property index, nothing",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (a:P) WHERE a.name = 'Nobody' RETURN a.k AS k",
            routes: &["ZoneMapEmpty", "ZoneMapEmpty declined", "PropertyIndex"],
        },
        Case {
            name: "key column read",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (a:P) WHERE a.k = 2 RETURN a.name AS name",
            routes: &["ZoneMapEmpty", "ZoneMapEmpty declined", "PropertyIndex"],
        },
        Case {
            name: "property index declines, in-list",
            setup: Setup::Usual,
            language: "sql",
            query: "SELECT * FROM GRAPH_TABLE (MATCH (s:S) WHERE s.k IN [1, 2] AND s.name = 'Gus' COLUMNS (s.k AS k))",
            routes: &[
                "ZoneMapEmpty",
                "ZoneMapEmpty declined",
                "PropertyIndex",
                "PropertyIndex declined",
                "InListIndex",
            ],
        },
        Case {
            name: "in-list index",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (a:P) WHERE a.name IN ['Gus', 'Jules'] RETURN a.k AS k",
            routes: &["ZoneMapEmpty", "ZoneMapEmpty declined", "InListIndex"],
        },
        Case {
            name: "in-list index, empty",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (a:P) WHERE a.name IN [] RETURN a.k AS k",
            routes: &["ZoneMapEmpty", "ZoneMapEmpty declined", "InListIndex"],
        },
        Case {
            name: "in-list column read",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (a:P) WHERE a.k IN [1, 2] RETURN a.name AS name",
            routes: &["ZoneMapEmpty", "ZoneMapEmpty declined", "InListIndex"],
        },
        Case {
            name: "in-list column read declines, as long as the nodes",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (a:P) WHERE a.k IN [1, 2, 3, 4] RETURN a.name AS name",
            routes: &[
                "ZoneMapEmpty",
                "ZoneMapEmpty declined",
                "InListIndex",
                "InListIndex declined",
                "Generic",
            ],
        },
        Case {
            name: "range index",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (a:P) WHERE a.k > 2 RETURN a.name AS name",
            routes: &["ZoneMapEmpty", "ZoneMapEmpty declined", "RangeIndex"],
        },
        Case {
            name: "range index under a limit",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (a:P) WHERE a.k > 2 RETURN a.name AS name LIMIT 5",
            routes: &["ZoneMapEmpty", "ZoneMapEmpty declined", "RangeIndex"],
        },
        Case {
            name: "compound hybrid",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (a:P) WHERE text_score(a.bio, 'graph') > 0.0 AND cosine_similarity(a.embedding, [1.0, 0.0]) > 0.5 RETURN a.name AS name",
            routes: &["ZoneMapEmpty", "ZoneMapEmpty declined", "CompoundHybrid"],
        },
        Case {
            name: "vector index",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (a:P) WHERE cosine_similarity(a.embedding, [1.0, 0.0]) > 0.5 RETURN a.name AS name",
            routes: &["ZoneMapEmpty", "ZoneMapEmpty declined", "VectorIndex"],
        },
        Case {
            name: "text index",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (a:P) WHERE text_score(a.bio, 'canals') > 0.0 RETURN a.name AS name",
            routes: &["ZoneMapEmpty", "ZoneMapEmpty declined", "TextIndex"],
        },
        Case {
            name: "label union",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (n) WHERE n:P OR n:Q RETURN n.k AS k",
            routes: &["ZoneMapEmpty", "ZoneMapEmpty declined", "LabelUnion"],
        },
        Case {
            name: "generic",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (a:P)-[:R]->(b) WHERE b.k < a.k RETURN a.name AS name",
            routes: &["ZoneMapEmpty", "ZoneMapEmpty declined", "Generic"],
        },
        Case {
            name: "a local time, indexed equality",
            setup: Setup::TimeZone,
            language: "gql",
            query: "MATCH (a:P) WHERE a.name = TIME '10:00:00' RETURN a.k AS k",
            routes: &["ZoneMapEmpty", "ZoneMapEmpty declined", "Generic"],
        },
        Case {
            name: "a local time, column equality",
            setup: Setup::TimeZone,
            language: "gql",
            query: "MATCH (a:P) WHERE a.t = TIME '10:00:00' RETURN a.k AS k",
            routes: &["ZoneMapEmpty", "ZoneMapEmpty declined", "Generic"],
        },
        Case {
            name: "a local time, in-list column read",
            setup: Setup::TimeZone,
            language: "gql",
            query: "MATCH (a:P) WHERE a.k IN [TIME '10:00:00'] RETURN a.name AS name",
            routes: &["ZoneMapEmpty", "ZoneMapEmpty declined", "Generic"],
        },
        Case {
            name: "a local time, in-list index declines",
            setup: Setup::TimeZone,
            language: "gql",
            query: "MATCH (a:P) WHERE a.name IN [TIME '10:00:00'] RETURN a.k AS k",
            routes: &[
                "ZoneMapEmpty",
                "ZoneMapEmpty declined",
                "InListIndex",
                "InListIndex declined",
                "Generic",
            ],
        },
        Case {
            name: "after a write in the input",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (q:Q) SET q.k = 1 WITH q MATCH (a:P) WHERE a.name = 'Gus' RETURN a.k AS k",
            routes: &["Generic"],
        },
        Case {
            name: "after a write, a key seeks",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (q:Q {name: 'Mia'}) SET q.k = 1 WITH q OPTIONAL MATCH (a:P) WHERE a.name = 'Gus' RETURN a.k AS k",
            routes: &[
                "ZoneMapEmpty",
                "ZoneMapEmpty declined",
                "PropertyIndex",
                "NodeSeek",
            ],
        },
        Case {
            name: "after a write, a key column read",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (q:Q {name: 'Mia'}) SET q.k = 1 WITH q OPTIONAL MATCH (a:P) WHERE a.k = 2 RETURN a.name AS name",
            routes: &[
                "ZoneMapEmpty",
                "ZoneMapEmpty declined",
                "PropertyIndex",
                "PropertyIndex",
            ],
        },
        Case {
            name: "after a write, no zone map",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (q:Q {name: 'Mia'}) SET q.k = 1 WITH q OPTIONAL MATCH (a:P)-[:R]->(b) WHERE b.k > 100 RETURN a.name AS name",
            routes: &[
                "ZoneMapEmpty",
                "ZoneMapEmpty declined",
                "PropertyIndex",
                "Generic",
            ],
        },
        Case {
            name: "rules off",
            setup: Setup::Plain,
            language: "gql",
            query: "MATCH (a:P) WHERE a.name = 'Gus' RETURN a.k AS k",
            routes: &["Generic"],
        },
        Case {
            name: "hop target, a key",
            setup: Setup::Usual,
            language: "gql",
            query: "MATCH (a:P {k: 1})-[:R]->(b:P) WHERE b.name = 'Gus' RETURN a.name AS name",
            routes: &[
                "ZoneMapEmpty",
                "ZoneMapEmpty declined",
                "Generic",
                "hop target",
                "ZoneMapEmpty",
                "ZoneMapEmpty declined",
                "Generic",
                "ZoneMapEmpty",
                "ZoneMapEmpty declined",
                "PropertyIndex",
            ],
        },
        Case {
            name: "hop target, a key from the written start",
            setup: Setup::WrittenStart,
            language: "gql",
            query: "MATCH (a:P)-[:R]->(b:P) WHERE b.name = 'Gus' RETURN a.name AS name",
            routes: &[
                "ZoneMapEmpty",
                "ZoneMapEmpty declined",
                "Generic",
                "hop target",
                "ZoneMapEmpty",
                "ZoneMapEmpty declined",
                "Generic",
            ],
        },
    ];

    /// Each statement's filters take the routes pinned for it, and it answers as without the
    /// planner's rules.
    #[test]
    fn each_filter_records_its_route() {
        for case in CASES {
            let db = graph();
            let routes = routes_of(&mut session(&db, case.setup), case.language, case.query);
            assert_eq!(routes, case.routes, "{}: {}", case.name, case.query);
            assert_eq!(
                rows_of(case.setup, case.language, case.query),
                rows_of(Setup::Plain, case.language, case.query),
                "{}: {}",
                case.name,
                case.query
            );
        }
    }
}
