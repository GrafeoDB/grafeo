//! `db.graph(name)`: queries, batches and upserts in one named graph without
//! switching the database's current graph.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use wasm_bindgen::prelude::*;

use crate::{database_error, js_error};

use grafeo_common::types::Value;
use grafeo_engine::session::Session;
use grafeo_engine::{GrafeoDB, GraphHandle as EngineGraphHandle};

use super::writes::{edge_options, edges, ids_to_js, labels, node_options, rows, summary_to_js};
use super::{Database, raw_result_to_js, rows_to_js};

/// A handle on one named graph, from `db.graph(name)`.
///
/// It has the query surface of `Database` (`execute*` and `executeRaw*`),
/// batches and upserts, working in its graph whatever graph the database
/// selects. Each call is its own transaction, so a call throws while the
/// database has a transaction open. Every call throws once the graph no
/// longer exists or the database is closed.
#[wasm_bindgen]
pub struct GraphHandle {
    db: Rc<RefCell<GrafeoDB>>,
    tx: Rc<RefCell<Option<Session>>>,
    closed: Rc<Cell<bool>>,
    schema: Option<String>,
    name: String,
}

#[wasm_bindgen]
impl Database {
    /// A handle on named graph `name` of the current schema, to work in it
    /// without switching the current graph.
    ///
    /// ```js
    /// db.execute("CREATE GRAPH trips");
    /// const trips = db.graph("trips");
    /// trips.execute("INSERT (:City {name: 'Prague'})");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the database is closed or the graph does not
    /// exist.
    pub fn graph(&self, name: &str) -> Result<GraphHandle, JsValue> {
        self.check_open()?;
        let schema = {
            let db = self.db();
            let graph = db.graph(name).map_err(database_error)?;
            graph.schema().map(ToString::to_string)
        };
        Ok(GraphHandle {
            db: Rc::clone(&self.inner),
            tx: Rc::clone(&self.tx),
            closed: Rc::clone(&self.closed),
            schema,
            name: name.to_string(),
        })
    }
}

impl GraphHandle {
    /// Runs `call` with the engine's handle on this graph, after checking that
    /// the database is open and has no transaction open.
    fn with_graph<T>(
        &self,
        call_name: &str,
        call: impl FnOnce(&EngineGraphHandle<'_>) -> grafeo_common::utils::error::Result<T>,
    ) -> Result<T, JsValue> {
        if self.closed.get() {
            return Err(js_error("Database is closed."));
        }
        if self.tx.borrow().is_some() {
            return Err(js_error(&format!(
                "{call_name} on a graph handle runs on its own: commit or roll back the \
                 database's open transaction first"
            )));
        }
        let db = self.db.borrow();
        let graph = db
            .graph_in(self.schema.as_deref(), &self.name)
            .map_err(database_error)?;
        call(&graph).map_err(database_error)
    }

    fn query(
        &self,
        query: &str,
        language: &str,
        params: Option<JsValue>,
    ) -> Result<grafeo_engine::database::QueryResult, JsValue> {
        let params: Option<HashMap<String, Value>> = Database::convert_params(params)?;
        crate::timed(|| {
            self.with_graph("execute", |graph| {
                graph.execute_language(query, language, params)
            })
        })
    }
}

#[wasm_bindgen]
impl GraphHandle {
    /// The graph's name.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// Runs a GQL query in this graph; rows as objects, as
    /// `Database.execute`.
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the query fails or the graph no longer exists.
    pub fn execute(&self, query: &str) -> Result<JsValue, JsValue> {
        Ok(rows_to_js(&self.query(query, "gql", None)?))
    }

    /// Runs a GQL query with parameters in this graph.
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the query fails or the graph no longer exists.
    #[wasm_bindgen(js_name = "executeWithParams")]
    pub fn execute_with_params(&self, query: &str, params: JsValue) -> Result<JsValue, JsValue> {
        Ok(rows_to_js(&self.query(query, "gql", Some(params))?))
    }

    /// Runs a query in a named language (for example `"cypher"`) in this
    /// graph.
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the language is not available, the query fails
    /// or the graph no longer exists.
    #[wasm_bindgen(js_name = "executeWithLanguage")]
    pub fn execute_with_language(&self, query: &str, language: &str) -> Result<JsValue, JsValue> {
        Ok(rows_to_js(&self.query(query, language, None)?))
    }

    /// Runs a query in a named language with parameters in this graph.
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the language is not available, the query fails
    /// or the graph no longer exists.
    #[wasm_bindgen(js_name = "executeWithLanguageAndParams")]
    pub fn execute_with_language_and_params(
        &self,
        query: &str,
        language: &str,
        params: JsValue,
    ) -> Result<JsValue, JsValue> {
        Ok(rows_to_js(&self.query(query, language, Some(params))?))
    }

    /// Runs a GQL query in this graph; `{columns, rows, executionTimeMs,
    /// counters}`, as `Database.executeRaw`.
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the query fails or the graph no longer exists.
    #[wasm_bindgen(js_name = "executeRaw")]
    pub fn execute_raw(&self, query: &str) -> Result<JsValue, JsValue> {
        Ok(raw_result_to_js(&self.query(query, "gql", None)?))
    }

    /// Runs a GQL query with parameters in this graph, raw.
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the query fails or the graph no longer exists.
    #[wasm_bindgen(js_name = "executeRawWithParams")]
    pub fn execute_raw_with_params(
        &self,
        query: &str,
        params: JsValue,
    ) -> Result<JsValue, JsValue> {
        Ok(raw_result_to_js(&self.query(query, "gql", Some(params))?))
    }

    /// Runs a query in a named language in this graph, raw.
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the language is not available, the query fails
    /// or the graph no longer exists.
    #[wasm_bindgen(js_name = "executeRawWithLanguage")]
    pub fn execute_raw_with_language(
        &self,
        query: &str,
        language: &str,
    ) -> Result<JsValue, JsValue> {
        Ok(raw_result_to_js(&self.query(query, language, None)?))
    }

    /// Runs a query in a named language with parameters in this graph, raw.
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the language is not available, the query fails
    /// or the graph no longer exists.
    #[wasm_bindgen(js_name = "executeRawWithLanguageAndParams")]
    pub fn execute_raw_with_language_and_params(
        &self,
        query: &str,
        language: &str,
        params: JsValue,
    ) -> Result<JsValue, JsValue> {
        Ok(raw_result_to_js(&self.query(
            query,
            language,
            Some(params),
        )?))
    }

    /// `Database.batchCreateNodesWithProps` in this graph.
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the arguments are malformed, a node breaks a
    /// constraint or the graph no longer exists.
    #[wasm_bindgen(js_name = "batchCreateNodesWithProps")]
    pub fn batch_create_nodes_with_props(
        &self,
        labels: JsValue,
        rows: JsValue,
    ) -> Result<JsValue, JsValue> {
        let labels = self::labels(labels)?;
        let rows = self::rows(rows)?;
        let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
        let ids = self.with_graph("batchCreateNodesWithProps", |graph| {
            graph.batch_create_nodes_with_labels(&labels, rows)
        })?;
        ids_to_js(ids.into_iter().map(|id| id.as_u64()))
    }

    /// `Database.batchCreateEdges` in this graph.
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the edges are malformed, an edge cannot be
    /// created or the graph no longer exists.
    #[wasm_bindgen(js_name = "batchCreateEdges")]
    pub fn batch_create_edges(&self, edges: JsValue) -> Result<JsValue, JsValue> {
        let edges = self::edges(edges)?;
        let ids = self.with_graph("batchCreateEdges", |graph| graph.batch_create_edges(edges))?;
        ids_to_js(ids.into_iter().map(|id| id.as_u64()))
    }

    /// `Database.upsertNodes` in this graph.
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the arguments are malformed, a row breaks a
    /// constraint or the schema, or the graph no longer exists.
    #[wasm_bindgen(js_name = "upsertNodes")]
    pub fn upsert_nodes(
        &self,
        labels: JsValue,
        rows: JsValue,
        options: JsValue,
    ) -> Result<JsValue, JsValue> {
        let labels = self::labels(labels)?;
        let rows = self::rows(rows)?;
        let (key, replace) = node_options(options)?;
        let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
        let summary = self.with_graph("upsertNodes", |graph| {
            graph.upsert_nodes(&labels, &key, rows, replace)
        })?;
        Ok(summary_to_js(&summary))
    }

    /// `Database.upsertEdges` in this graph.
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the arguments are malformed, a row breaks the
    /// schema, or the graph no longer exists.
    #[wasm_bindgen(js_name = "upsertEdges")]
    pub fn upsert_edges(
        &self,
        edge_type: &str,
        rows: JsValue,
        options: JsValue,
    ) -> Result<JsValue, JsValue> {
        let rows = self::rows(rows)?;
        let options = edge_options(options)?;
        let summary = self.with_graph("upsertEdges", |graph| {
            graph.upsert_edges(edge_type, rows, &options)
        })?;
        Ok(summary_to_js(&summary))
    }
}
