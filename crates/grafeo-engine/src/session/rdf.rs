//! RDF-specific session methods.
//!
//! This module consolidates all RDF functionality from the session layer.
//! The entire module is gated behind `#[cfg(feature = "triple-store")]` in the parent.
//! The methods that run SPARQL and GraphQL queries on the RDF store are in
//! the `query` submodule, which only a build with one of those languages
//! has: the triple store adds the store, a language adds the queries over it.

use grafeo_core::graph::GraphBackend;
use grafeo_core::graph::GraphStore;
use grafeo_core::graph::rdf::RdfStore;
use grafeo_core::graph::rowgroup::RowGroupStore;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use super::Session;
use super::SessionConfig;

#[cfg(rdf_language)]
mod query;

impl Session {
    /// Creates a session that reads and writes `store` and `rdf_store`.
    pub(crate) fn with_rdf_store(
        store: Arc<RowGroupStore>,
        rdf_store: Arc<RdfStore>,
        cfg: SessionConfig,
    ) -> Self {
        let graph_store = Arc::clone(&store) as Arc<dyn GraphStore>;
        let graph_store_mut = Some(Arc::clone(&store) as Arc<dyn GraphBackend>);
        Self {
            store: Some(store),
            graph_store,
            graph_store_mut,
            catalog: cfg.catalog,
            rdf_store,
            transaction_manager: cfg.transaction_manager,
            query_cache: cfg.query_cache,
            current_transaction: parking_lot::Mutex::new(None),
            read_only_tx: parking_lot::Mutex::new(cfg.read_only),
            db_read_only: cfg.read_only,
            identity: cfg.identity,
            auto_commit: true,
            #[cfg(query_language)]
            plan_options: super::PlanOptions {
                #[cfg(lpg_language)]
                factorized_execution: cfg.factorized_execution,
                shuffle_unordered: cfg.shuffle_unordered,
                #[cfg(lpg_language)]
                reachability: true,
                path_search_budget: cfg.path_search_budget,
            },
            graph_model: cfg.graph_model,
            #[cfg(query_language)]
            query_timeout: cfg.query_timeout,
            max_property_size: cfg.max_property_size,
            #[cfg(feature = "spill")]
            buffer_manager: cfg.buffer_manager,
            commit_counter: cfg.commit_counter,
            gc_interval: cfg.gc_interval,
            active_streams: AtomicUsize::new(0),
            #[cfg(feature = "wal")]
            wal: None,
            #[cfg(feature = "cdc")]
            cdc_log: Arc::new(crate::cdc::CdcLog::new()),
            #[cfg(feature = "cdc")]
            records_cdc: false,
            current_graph: parking_lot::Mutex::new(None),
            statement_graph: parking_lot::Mutex::new(None),
            current_schema: parking_lot::Mutex::new(None),
            home_schema: None,
            home_graph: None,
            time_zone: parking_lot::Mutex::new(None),
            session_params: parking_lot::Mutex::new(std::collections::HashMap::new()),
            viewing_epoch_override: parking_lot::Mutex::new(None),
            savepoints: parking_lot::Mutex::new(Vec::new()),
            transaction_nesting_depth: parking_lot::Mutex::new(0),
            last_commit_epoch: parking_lot::Mutex::new(None),
            transaction_statements: super::TransactionStatements::Allowed,
            changes: parking_lot::Mutex::new(None),
            #[cfg(feature = "metrics")]
            metrics: None,
            #[cfg(feature = "metrics")]
            tx_start_time: parking_lot::Mutex::new(None),
            #[cfg(lpg_language)]
            projections: cfg.projections,
        }
    }
}
