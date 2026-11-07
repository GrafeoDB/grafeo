//! Where a statement reads and writes, and how its planners are built.
//!
//! A session builds a [`StatementPlanning`] for each statement it plans, and
//! the statement's planner comes from it. The planner keeps the part that is
//! the session's, a [`SessionPlanning`], so the planner of each stored
//! procedure body the statement calls is built the same way, when the call
//! runs, after the session's call returned: on the graph the body's `USE`
//! names, else on the graph of the statement that calls it, in the caller's
//! transaction, with its checks of the schema and the constraints.

use std::sync::Arc;

use grafeo_common::types::{EpochId, TransactionId};
use grafeo_common::utils::error::{Error, QueryError, Result};
use grafeo_core::execution::operators::Recording;
use grafeo_core::graph::GraphBackend;
use grafeo_core::graph::GraphStore;
use grafeo_core::graph::rowgroup::RowGroupStore;

use crate::auth::{Identity, Role};
use crate::catalog::{Catalog, CatalogConstraintValidator};
use crate::transaction::{TransactionChanges, TransactionManager};

#[cfg(lpg_language)]
use super::graph_storage_key;

/// The stores a session reads and writes: the built-in store, which holds
/// the named graphs, and the default graph's stores.
#[derive(Clone)]
pub(crate) struct GraphStores {
    /// The built-in store, which holds the named graphs, their indexes and
    /// the search indexes; `None` for a session on a backend of one's own
    /// (`GrafeoDB::with_store`), which has none of them.
    pub(super) root: Option<Arc<RowGroupStore>>,
    /// The default graph's store, as its reads read it.
    pub(super) graph_store: Arc<dyn GraphStore>,
    /// The default graph's backend, which its writes apply in (`None` for a
    /// read-only one).
    pub(super) graph_store_mut: Option<Arc<dyn GraphBackend>>,
    /// The graphs the session's open transaction created and dropped.
    pub(super) pending_graphs: Arc<parking_lot::RwLock<PendingGraphs>>,
}

/// What an open transaction's own statements see of the named graphs, until its commit.
#[derive(Default)]
pub(crate) struct PendingGraphs {
    /// The graphs it created, by storage key, which nobody else can reach.
    pub(super) created: std::collections::HashMap<String, Arc<RowGroupStore>>,
    /// The graphs it dropped, which it cannot resolve.
    pub(super) dropped: std::collections::HashSet<String>,
    /// The graphs it created, also those it has dropped since: the session may have to
    /// leave one when the transaction ends without its commit.
    pub(super) made: std::collections::HashSet<String>,
}

impl GraphStores {
    /// The named graph with storage key `key`: `None` when there is none, as
    /// on a backend of one's own, which holds no named graphs.
    pub(super) fn named_graph(&self, key: &str) -> Option<Arc<RowGroupStore>> {
        {
            let pending = self.pending_graphs.read();
            if let Some(graph) = pending.created.get(key) {
                return Some(Arc::clone(graph));
            }
            if pending.dropped.contains(key) {
                return None;
            }
        }
        self.root.as_ref().and_then(|store| store.graph(key))
    }

    /// The graph store for the graph with storage key `key` (`None` for the
    /// default graph).
    pub(super) fn store_for_key(&self, key: Option<&str>) -> Arc<dyn GraphStore> {
        match key {
            None => Arc::clone(&self.graph_store),
            Some(name) => match self.named_graph(name) {
                Some(named_store) => named_store as Arc<dyn GraphStore>,
                // Dropped meanwhile: no data, never the default graph's (the
                // graph check before a statement reports the drop).
                None => Arc::new(grafeo_core::graph::NullGraphStore) as Arc<dyn GraphStore>,
            },
        }
    }

    /// The writable store for the graph with storage key `key`: `None` for a
    /// read-only database.
    pub(super) fn write_store_for_key(&self, key: Option<&str>) -> Option<Arc<dyn GraphBackend>> {
        match key {
            None => self.graph_store_mut.as_ref().map(Arc::clone),
            Some(name) => self
                .named_graph(name)
                // Dropped meanwhile: nothing to write to, never the default
                // graph (see `store_for_key`).
                .map(|named_store| named_store as Arc<dyn GraphBackend>),
        }
    }

    /// The storage key of the graph the writes to the graph with storage
    /// key `key` land in: that graph, or `None` (the default graph) when no
    /// graph of that name exists.
    pub(super) fn written_key(&self, key: Option<String>) -> Option<String> {
        key.filter(|name| self.named_graph(name).is_some())
    }

    /// The built-in store of the graph with storage key `written_key` (see
    /// [`written_key`](Self::written_key)): `None` on a backend of one's own.
    pub(super) fn lpg_store(&self, written_key: Option<&str>) -> Option<Arc<RowGroupStore>> {
        written_key
            .and_then(|name| self.named_graph(name))
            .or_else(|| self.root.as_ref().map(Arc::clone))
    }
}

/// What a statement's writes go through and what checks them, whichever
/// graph it reads and writes.
#[derive(Clone)]
pub(crate) struct GraphAccess {
    /// The session's stores.
    pub(super) stores: GraphStores,
    /// The catalog, whose schema and constraints check the writes.
    pub(super) catalog: Arc<Catalog>,
    /// The transaction manager, which the writes' recording and key claims
    /// report to.
    pub(super) transaction_manager: Arc<TransactionManager>,
    /// What the open transaction changed: `None` outside one.
    pub(super) changes: Option<Arc<TransactionChanges>>,
    /// The largest property value a write may store, in bytes.
    pub(super) max_property_size: Option<usize>,
}

impl GraphAccess {
    /// Where a writer of the open transaction writes the graph with storage
    /// key `written` (`None` for the default graph, see
    /// [`GraphStores::written_key`]) and records its writes: the graph's
    /// store and the transaction's changes in it. `None` outside a
    /// transaction, on a read-only external store, and for a named graph
    /// that no longer exists.
    ///
    /// # Errors
    ///
    /// Fails when the transaction wrote the graph through another store: it
    /// was dropped and created again since.
    pub(super) fn recording(&self, written: Option<&str>) -> Result<Option<Recording>> {
        let Some(changes) = &self.changes else {
            return Ok(None);
        };
        let Some(target) = self.stores.write_store_for_key(written) else {
            return Ok(None);
        };
        changes
            .recording(&self.transaction_manager, written, target)
            .map(Some)
    }

    /// The checks for writes to `store`, the store of the graph with storage
    /// key `key` in `schema`: the catalog's schema and constraints (the node
    /// and edge types of the schema), the closed graph type the graph is
    /// bound to, and the property size limit.
    pub(super) fn constraint_validator(
        &self,
        store: Arc<dyn GraphStore>,
        schema: Option<&str>,
        key: Option<&str>,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> CatalogConstraintValidator {
        let validator = CatalogConstraintValidator::new(Arc::clone(&self.catalog))
            .with_store(store)
            .with_max_property_size(self.max_property_size)
            .with_transaction_context(epoch, transaction_id)
            .with_schema(schema)
            .with_key_claims(Arc::clone(&self.transaction_manager), key);
        match key {
            Some(graph) => validator.with_graph_name(graph),
            None => validator,
        }
    }
}

/// Fails when `identity` may not focus a statement on the graph `name` with
/// its `USE`, as GQL's use graph clause checks: a permission error when it
/// has per-graph grants and none for that graph, an invalid reference when
/// no graph has that name in `schema`. The default graph needs neither.
#[cfg(feature = "gql")]
pub(super) fn check_focus(
    identity: &Identity,
    stores: &GraphStores,
    schema: Option<&str>,
    name: &str,
) -> Result<()> {
    if name.eq_ignore_ascii_case("default") {
        return Ok(());
    }
    if identity.has_grants() && !identity.can_access_graph(name, Role::ReadOnly) {
        return Err(Error::from(QueryError::permission_denied(format!(
            "no grant for graph '{name}' (user: {})",
            identity.user_id()
        ))));
    }
    let key = super::graph_storage_key(schema, Some(name));
    if key.is_some_and(|key| stores.named_graph(&key).is_none()) {
        return Err(Error::from(QueryError::invalid_reference(format!(
            "Graph '{name}' does not exist"
        ))));
    }
    Ok(())
}

/// Fails when `identity` has per-graph grants and none covers `graph` at
/// the level a statement needs: any grant to read, a read-write one to
/// write (`writes`). The default graph needs no grant, as for `USE GRAPH`.
pub(super) fn check_graph_grant(
    identity: &Identity,
    graph: Option<&str>,
    writes: bool,
) -> Result<()> {
    if !identity.has_grants() {
        return Ok(());
    }
    let Some(graph) = graph else {
        return Ok(());
    };
    if graph.eq_ignore_ascii_case("default") {
        return Ok(());
    }
    let (role, access) = if writes {
        (Role::ReadWrite, "write")
    } else {
        (Role::ReadOnly, "read")
    };
    if identity.can_access_graph(graph, role) {
        return Ok(());
    }
    Err(Error::from(QueryError::permission_denied(format!(
        "no {access} grant for graph '{graph}' (user: {})",
        identity.user_id()
    ))))
}

/// The part of a statement's planning that is the session's, the same for
/// every planner of the statement: its stores and checks, its transaction
/// and view, and the session's settings as the statement started. The
/// statement's planner keeps it, so the planner of a stored procedure body
/// the statement calls is built from it (see [`for_body`](Self::for_body)).
#[cfg(lpg_language)]
#[derive(Clone)]
pub(crate) struct SessionPlanning {
    /// What the statement's writes go through and what checks them.
    pub(super) access: GraphAccess,
    /// Who runs the statement, for the grants of the graph a body's `USE`
    /// names.
    #[cfg(stored_procedures)]
    pub(super) identity: Arc<Identity>,
    /// How the session's queries are planned.
    pub(super) plan_options: super::PlanOptions,
    /// The database's projections, for `CALL grafeo.<algorithm>({projection: ...})`.
    pub(super) projections: super::ProjectionRegistry,
    /// How GQL compares a zoned datetime or time with a local one, for a
    /// stored procedure's body, which is GQL whatever the language of the
    /// statement that calls it.
    #[cfg(stored_procedures)]
    pub(super) gql_temporal_comparison: grafeo_common::types::TemporalComparison,
    /// How many stored procedure calls the statement runs in: 0 for a
    /// statement the session runs, 1 for the body of a procedure it calls.
    #[cfg(stored_procedures)]
    pub(super) call_depth: u8,
    /// The epoch the statement reads at.
    pub(super) viewing_epoch: EpochId,
    /// The open transaction: `None` outside one.
    pub(super) transaction_id: Option<TransactionId>,
    /// Whether the statement's operators read as a committed reader (see
    /// `Planner::with_read_only`).
    pub(super) read_only: bool,
    /// Whether the plan is only built, never run (`EXPLAIN`): outside a transaction its writers
    /// write through a plan-only recording, which refuses every write, so no transaction is
    /// begun to build them.
    pub(super) plan_only: bool,
}

/// How the planner of one statement is built: the session's part (see
/// [`SessionPlanning`]), and the graph and context the statement runs in.
#[cfg(lpg_language)]
#[derive(Clone)]
pub(crate) struct StatementPlanning {
    /// The session's part, which the statement's planner keeps.
    pub(super) session: SessionPlanning,
    /// The session's schema (`None`: the default schema).
    pub(super) schema: Option<String>,
    /// The graph the statement reads and writes, as its `USE` or the
    /// session names it (`None`: the default graph).
    pub(super) graph: Option<String>,
    /// The session's home schema, which `HOME_SCHEMA` returns.
    pub(super) home_schema: Option<String>,
    /// The session's home graph, which `HOME_GRAPH` returns.
    pub(super) home_graph: Option<String>,
    /// How the statement compares a zoned datetime or time with a local
    /// one, which depends on its language.
    pub(super) temporal_comparison: grafeo_common::types::TemporalComparison,
    /// The statement's language, whose value rules apply (a stored procedure's body is GQL).
    pub(super) language: grafeo_common::types::StatementLanguage,
    /// One instant for the statement, which every clock function reads, a
    /// body's included.
    pub(super) statement_clock: grafeo_core::execution::operators::StatementClock,
    /// Where the statement raises its warnings, a body's included.
    pub(super) warnings: Arc<grafeo_common::utils::warning::WarningSink>,
}

#[cfg(lpg_language)]
impl StatementPlanning {
    /// A planner for the statement that reads `store` (the statement's
    /// graph, or the graph a SQL/PGQ `GRAPH_TABLE` names) and writes the
    /// statement's graph, in the open transaction, with its writes checked
    /// against the schema and the constraints.
    ///
    /// # Errors
    ///
    /// Fails when the open transaction wrote the statement's graph through
    /// another store: the graph was dropped and created again since.
    pub(crate) fn planner(self, store: Arc<dyn GraphStore>) -> Result<crate::query::Planner> {
        use crate::query::Planner;
        use crate::query::optimizer::PlannerRule;
        use grafeo_common::types::ReadView;
        use grafeo_core::execution::operators::{LazyValue, SessionContext};

        let Self {
            session,
            schema,
            graph,
            home_schema,
            home_graph,
            temporal_comparison,
            language,
            statement_clock,
            warnings,
        } = self;
        let access = &session.access;
        let (viewing_epoch, transaction_id) = (session.viewing_epoch, session.transaction_id);

        let key = graph_storage_key(schema.as_deref(), graph.as_deref());
        let write_store = access.stores.write_store_for_key(key.as_deref());
        // The graph the writes land in: the statement's, or the default
        // graph when no graph of that name exists.
        let written = access.stores.written_key(key.clone());
        let recording = match &write_store {
            Some(_) if transaction_id.is_some() => access.recording(written.as_deref())?,
            Some(target) if session.plan_only => {
                Some(grafeo_core::execution::operators::Recording::plan_only(
                    Arc::clone(target),
                    viewing_epoch,
                ))
            }
            _ => None,
        };
        // The graph's built-in store, which the query reads, so CALL
        // grafeo.search.* procedures reach its HNSW / BM25 indexes (#677);
        // a backend of one's own has none.
        let lpg_store = access.stores.lpg_store(written.as_deref());
        // The constraint validator for schema enforcement and property size limits.
        let validator = access.constraint_validator(
            Arc::clone(&store),
            schema.as_deref(),
            key.as_deref(),
            viewing_epoch,
            transaction_id,
        );

        // Capture store reference for lazy introspection (only computed if info()/schema() called).
        let info_store = Arc::clone(&store);
        // The counts as the statement sees the graph.
        let info_view = if session.read_only {
            ReadView::committed(viewing_epoch)
        } else {
            ReadView::new(viewing_epoch, transaction_id)
        };
        let schema_store = Arc::clone(&store);
        let session_context = SessionContext {
            current_schema: schema,
            current_graph: graph,
            home_schema,
            home_graph,
            temporal_comparison,
            language,
            statement_clock,
            db_info: LazyValue::new(move || {
                super::Session::build_info_value(&*info_store, info_view)
            }),
            schema_info: LazyValue::new(move || super::Session::build_schema_value(&*schema_store)),
            warnings,
        };

        let planner = Planner::with_context(
            store,
            write_store,
            Arc::clone(&access.transaction_manager),
            transaction_id,
            viewing_epoch,
        )
        .with_factorized_execution(
            session.plan_options.factorized_execution
                && session
                    .plan_options
                    .rules
                    .has(PlannerRule::FactorizedExecution),
        )
        .with_rules(session.plan_options.rules)
        .with_shuffle_unordered(session.plan_options.shuffle_unordered)
        .with_reachability(session.plan_options.rules.has(PlannerRule::Reachability))
        .with_path_search_budget(session.plan_options.path_search_budget)
        .with_catalog(Arc::clone(&access.catalog))
        .with_session_context(session_context)
        .with_read_only(session.read_only)
        .with_recording(recording)
        .with_projections(Arc::clone(&session.projections))
        .with_validator(Arc::new(validator));
        let planner = match lpg_store {
            Some(store) => planner.with_lpg_store(store),
            None => planner.with_external_store(),
        };

        // A stored procedure the statement calls plans its body from here.
        #[cfg(stored_procedures)]
        let planner = planner.with_session_planning(session);

        Ok(planner)
    }

    /// The planner of a stored procedure's body (see
    /// [`SessionPlanning::for_body`]), which reads and writes its graph.
    ///
    /// # Errors
    ///
    /// As [`planner`](Self::planner).
    #[cfg(stored_procedures)]
    pub(crate) fn body_planner(self) -> Result<crate::query::Planner> {
        let key = graph_storage_key(self.schema.as_deref(), self.graph.as_deref());
        let store = self.session.access.stores.store_for_key(key.as_deref());
        self.planner(store)
    }
}

#[cfg(stored_procedures)]
impl SessionPlanning {
    /// How the body of a stored procedure that a statement in `caller` (the
    /// calling planner's context) calls is planned: on the graph `graph` its
    /// `USE` names, checked as the use graph clause of a statement run
    /// directly is, else on the caller's graph; in the caller's transaction,
    /// at its view and with its clock, and with GQL's semantics. A body that
    /// `writes` needs a read-write grant for its graph.
    ///
    /// # Errors
    ///
    /// A permission error when the user may not read the graph, or may not
    /// write it and the body writes; an invalid reference when no graph has
    /// the name `graph`; an error naming `procedure` when the call nests
    /// deeper than [`MAX_CALL_DEPTH`](crate::query::executor::user_procedure::MAX_CALL_DEPTH).
    pub(crate) fn for_body(
        &self,
        procedure: &str,
        caller: &grafeo_core::execution::operators::SessionContext,
        graph: Option<&str>,
        writes: bool,
    ) -> Result<StatementPlanning> {
        use crate::query::executor::user_procedure::MAX_CALL_DEPTH;

        if self.call_depth >= MAX_CALL_DEPTH {
            return Err(Error::from(QueryError::new(
                grafeo_common::utils::error::QueryErrorKind::Semantic,
                format!(
                    "procedure '{procedure}' is called more than {MAX_CALL_DEPTH} calls deep: \
                     does a procedure call itself?"
                ),
            )));
        }
        let graph = match graph {
            Some(name) => {
                check_focus(
                    &self.identity,
                    &self.access.stores,
                    caller.current_schema.as_deref(),
                    name,
                )?;
                Some(name.to_string())
            }
            None => caller.current_graph.clone(),
        };
        check_graph_grant(&self.identity, graph.as_deref(), writes)?;
        let mut session = self.clone();
        session.call_depth += 1;
        Ok(StatementPlanning {
            session,
            schema: caller.current_schema.clone(),
            graph,
            home_schema: caller.home_schema.clone(),
            home_graph: caller.home_graph.clone(),
            temporal_comparison: self.gql_temporal_comparison,
            language: grafeo_common::types::StatementLanguage::Gql,
            statement_clock: caller.statement_clock,
            warnings: Arc::clone(&caller.warnings),
        })
    }
}
