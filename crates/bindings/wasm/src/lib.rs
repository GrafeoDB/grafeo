//! WebAssembly bindings for Grafeo graph database.
//!
//! Use Grafeo from JavaScript in the browser, Deno, or Cloudflare Workers.
//!
//! ```js
//! import init, { Database } from '@grafeo-db/wasm';
//!
//! await init();
//! const db = new Database();
//! db.execute("INSERT (:Person {name: 'Alix', age: 30})");
//! const result = db.execute("MATCH (p:Person) RETURN p.name, p.age");
//! console.log(result); // [{name: "Alix", age: 30}]
//! ```

#![forbid(unsafe_code)]
// WASM target uses 32-bit usize, so usize-to-u32 casts are lossless.
// On 64-bit (clippy host), these are flagged but the code only runs on WASM.
#![allow(clippy::cast_possible_truncation)]

mod signed_snapshot;
mod types;
mod utils;

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use js_sys::Array;
use wasm_bindgen::prelude::*;

use grafeo_bindings_common::json::{json_params_to_map, json_to_value};
use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

/// A Grafeo graph database instance running in WebAssembly.
///
/// All data is held in memory within the WASM heap. For persistence,
/// use `exportSnapshot()` / `importSnapshot()` with IndexedDB or
/// the higher-level `@grafeo-db/web` package.
#[wasm_bindgen]
pub struct Database {
    inner: GrafeoDB,
    /// Active transaction session, set by `beginTransaction()` and cleared by
    /// `commitTransaction()` / `rollbackTransaction()` / `close()`.
    /// When present, `execute*` methods route through this session so
    /// uncommitted writes are visible only to the transaction.
    tx: RefCell<Option<Session>>,
    /// Once set by `close()`, every subsequent method call errors.
    closed: Cell<bool>,
}

#[wasm_bindgen]
impl Database {
    /// Creates a new in-memory database.
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the database fails to initialise.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<Database, JsError> {
        utils::set_panic_hook();
        Ok(Database {
            inner: GrafeoDB::new_in_memory(),
            tx: RefCell::new(None),
            closed: Cell::new(false),
        })
    }

    /// Begins a new transaction.
    ///
    /// Subsequent `execute*` calls see each other's uncommitted writes but
    /// remain invisible to other sessions until `commitTransaction()`. Only
    /// one transaction may be active at a time.
    ///
    /// ```js
    /// db.beginTransaction();
    /// db.execute("INSERT (:Person {name: 'Alix'})");
    /// db.commitTransaction();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the database is closed, a transaction is already
    /// active, or the engine fails to start one.
    #[wasm_bindgen(js_name = "beginTransaction")]
    pub fn begin_transaction(&self) -> Result<(), JsError> {
        self.check_open()?;
        if self.tx.borrow().is_some() {
            return Err(JsError::new(
                "Transaction already active. Commit or rollback before starting a new one.",
            ));
        }
        let mut session = self.inner.session();
        session
            .begin_transaction()
            .map_err(|e| JsError::new(&e.to_string()))?;
        *self.tx.borrow_mut() = Some(session);
        Ok(())
    }

    /// Commits the active transaction.
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the database is closed, no transaction is active,
    /// or the commit fails (e.g., serializable-isolation conflict).
    #[wasm_bindgen(js_name = "commitTransaction")]
    pub fn commit_transaction(&self) -> Result<(), JsError> {
        self.check_open()?;
        let mut tx_slot = self.tx.borrow_mut();
        let session = tx_slot
            .as_mut()
            .ok_or_else(|| JsError::new("No active transaction to commit."))?;
        session.commit().map_err(|e| JsError::new(&e.to_string()))?;
        *tx_slot = None;
        Ok(())
    }

    /// Rolls back the active transaction, discarding all pending writes.
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the database is closed, no transaction is active,
    /// or the rollback fails.
    #[wasm_bindgen(js_name = "rollbackTransaction")]
    pub fn rollback_transaction(&self) -> Result<(), JsError> {
        self.check_open()?;
        let mut tx_slot = self.tx.borrow_mut();
        let session = tx_slot
            .as_mut()
            .ok_or_else(|| JsError::new("No active transaction to roll back."))?;
        session
            .rollback()
            .map_err(|e| JsError::new(&e.to_string()))?;
        *tx_slot = None;
        Ok(())
    }

    /// Returns `true` while a transaction started by `beginTransaction()` is
    /// still active (not yet committed or rolled back).
    #[wasm_bindgen(js_name = "isTransactionActive")]
    pub fn is_transaction_active(&self) -> bool {
        !self.closed.get() && self.tx.borrow().is_some()
    }

    /// Closes the database, rolling back any active transaction and
    /// clearing the query plan cache. Subsequent method calls return an
    /// error. Calling `close()` more than once is a no-op.
    ///
    /// Because WASM keeps all data in memory, `close()` does not persist
    /// anything: call `exportSnapshot()` first if you need persistence.
    pub fn close(&self) {
        if self.closed.get() {
            return;
        }
        if let Some(mut session) = self.tx.borrow_mut().take() {
            let _ = session.rollback();
        }
        self.inner.clear_plan_cache();
        self.closed.set(true);
    }

    fn check_open(&self) -> Result<(), JsError> {
        if self.closed.get() {
            return Err(JsError::new(
                "Database is closed. Create a new instance with `new Database()` or \
                 `Database.importSnapshot()` to continue.",
            ));
        }
        Ok(())
    }

    /// Runs a GQL query, routing through the active transaction session when
    /// one exists so uncommitted writes remain visible to the caller.
    fn run_query(&self, query: &str) -> Result<grafeo_engine::database::QueryResult, JsError> {
        let tx_slot = self.tx.borrow();
        if let Some(session) = tx_slot.as_ref() {
            session
                .execute(query)
                .map_err(|e| JsError::new(&e.to_string()))
        } else {
            drop(tx_slot);
            self.inner
                .execute(query)
                .map_err(|e| JsError::new(&e.to_string()))
        }
    }

    /// Runs a query with an explicit language, routing through the active
    /// transaction session when one exists.
    fn run_language_query(
        &self,
        query: &str,
        language: &str,
        params: Option<HashMap<String, Value>>,
    ) -> Result<grafeo_engine::database::QueryResult, JsError> {
        let tx_slot = self.tx.borrow();
        if let Some(session) = tx_slot.as_ref() {
            session
                .execute_language(query, language, params)
                .map_err(|e| JsError::new(&e.to_string()))
        } else {
            drop(tx_slot);
            self.inner
                .execute_language(query, language, params)
                .map_err(|e| JsError::new(&e.to_string()))
        }
    }

    /// Executes a GQL query and returns results as an array of objects.
    ///
    /// Each row becomes a JavaScript object with column names as keys.
    ///
    /// ```js
    /// const results = db.execute("MATCH (p:Person) RETURN p.name, p.age");
    /// // [{name: "Alix", age: 30}, {name: "Gus", age: 25}]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the query fails to parse or execute.
    pub fn execute(&self, query: &str) -> Result<JsValue, JsError> {
        self.check_open()?;
        let result = self.run_query(query)?;

        let rows = Array::new_with_length(result.rows().len() as u32);
        for (i, row) in result.rows().iter().enumerate() {
            rows.set(i as u32, types::row_to_js_object(&result.columns, row));
        }
        Ok(rows.into())
    }

    /// Executes a GQL query and returns raw columns, rows, and metadata.
    ///
    /// Returns `{ columns: string[], rows: any[][], executionTimeMs?: number }`.
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the query fails to parse or execute.
    #[wasm_bindgen(js_name = "executeRaw")]
    pub fn execute_raw(&self, query: &str) -> Result<JsValue, JsError> {
        self.check_open()?;
        let result = self.run_query(query)?;

        let obj = js_sys::Object::new();

        // columns: string[]
        let cols = Array::new_with_length(result.columns.len() as u32);
        for (i, col) in result.columns.iter().enumerate() {
            cols.set(i as u32, JsValue::from_str(col));
        }
        let _ = js_sys::Reflect::set(&obj, &JsValue::from_str("columns"), &cols);

        // rows: any[][]
        let rows = Array::new_with_length(result.rows().len() as u32);
        for (i, row) in result.rows().iter().enumerate() {
            let js_row = Array::new_with_length(row.len() as u32);
            for (j, val) in row.iter().enumerate() {
                js_row.set(j as u32, types::value_to_js(val));
            }
            rows.set(i as u32, js_row.into());
        }
        let _ = js_sys::Reflect::set(&obj, &JsValue::from_str("rows"), &rows);

        // executionTimeMs?: number
        if let Some(ms) = result.execution_time_ms {
            let _ = js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("executionTimeMs"),
                &JsValue::from_f64(ms),
            );
        }

        Ok(obj.into())
    }

    /// Returns the number of nodes in the database.
    #[wasm_bindgen(js_name = "nodeCount")]
    pub fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Returns the number of edges in the database.
    #[wasm_bindgen(js_name = "edgeCount")]
    pub fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    /// Clears all cached query plans.
    ///
    /// Forces re-parsing and re-optimization on next execution.
    #[wasm_bindgen(js_name = "clearPlanCache")]
    pub fn clear_plan_cache(&self) {
        self.inner.clear_plan_cache();
    }

    /// Executes a query using a specific query language.
    ///
    /// Supported languages: `"gql"`, `"cypher"`, `"sparql"`, `"gremlin"`, `"graphql"`, `"graphql-rdf"`, `"sql"`.
    /// Languages require their corresponding feature flag to be enabled.
    ///
    /// ```js
    /// const results = db.executeWithLanguage(
    ///   "MATCH (p:Person) RETURN p.name",
    ///   "cypher"
    /// );
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the language is unsupported or the query fails to parse or execute.
    #[wasm_bindgen(js_name = "executeWithLanguage")]
    pub fn execute_with_language(&self, query: &str, language: &str) -> Result<JsValue, JsError> {
        self.execute_language_impl(query, language, None)
    }

    /// Exports the database to a binary snapshot.
    ///
    /// Returns a `Uint8Array` that can be stored in IndexedDB, localStorage,
    /// or sent over the network. Restore with `Database.importSnapshot()`.
    ///
    /// ```js
    /// const bytes = db.exportSnapshot();
    /// // Store in IndexedDB, download as file, etc.
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `JsError` if snapshot serialisation fails.
    #[wasm_bindgen(js_name = "exportSnapshot")]
    pub fn export_snapshot(&self) -> Result<Vec<u8>, JsError> {
        self.check_open()?;
        self.inner
            .export_snapshot()
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Exports the database to a tamper-evident binary snapshot.
    ///
    /// Prefixes the bytes with a `GSN1` magic header and appends an
    /// HMAC-SHA256 tag keyed by `key`. The same `key` passed to
    /// `importSnapshotSigned()` verifies integrity before deserialisation;
    /// mismatched or truncated snapshots are rejected.
    ///
    /// Use this form whenever snapshots are stored in locations the user
    /// cannot fully trust (IndexedDB, server-side storage, shared URLs).
    /// For ephemeral in-memory snapshots, `exportSnapshot()` is still fine.
    ///
    /// Key management: the browser has no built-in key store, so the caller
    /// owns key generation, storage, and rotation. Derive from a user
    /// password with `PBKDF2`/`HKDF` in JS, or generate a random 32-byte
    /// key and store it in an IndexedDB record the application controls.
    ///
    /// ```js
    /// const key = crypto.getRandomValues(new Uint8Array(32));
    /// const signed = db.exportSnapshotSigned(key);
    /// // ... later
    /// const restored = Database.importSnapshotSigned(signed, key);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the key is empty or snapshot serialisation fails.
    #[wasm_bindgen(js_name = "exportSnapshotSigned")]
    pub fn export_snapshot_signed(&self, key: &[u8]) -> Result<Vec<u8>, JsError> {
        self.check_open()?;
        if key.is_empty() {
            return Err(JsError::new(
                "exportSnapshotSigned: key must not be empty (recommended: 32 random bytes)",
            ));
        }
        let payload = self
            .inner
            .export_snapshot()
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(signed_snapshot::wrap(key, &payload))
    }

    /// Creates a database from a binary snapshot.
    ///
    /// The `data` must have been produced by `exportSnapshot()`.
    ///
    /// ```js
    /// const db = Database.importSnapshot(bytes);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `JsError` if `data` is not a valid snapshot or deserialisation fails.
    #[wasm_bindgen(js_name = "importSnapshot")]
    pub fn import_snapshot(data: &[u8]) -> Result<Database, JsError> {
        utils::set_panic_hook();
        if data.len() > signed_snapshot::MAX_SNAPSHOT_BYTES {
            return Err(JsError::new(&format!(
                "importSnapshot: snapshot exceeds {} MiB limit",
                signed_snapshot::MAX_SNAPSHOT_BYTES / (1024 * 1024)
            )));
        }
        if signed_snapshot::looks_signed(data) {
            return Err(JsError::new(
                "importSnapshot: this snapshot was produced by exportSnapshotSigned. \
                 Use importSnapshotSigned(data, key) to verify and restore it.",
            ));
        }
        let inner = GrafeoDB::import_snapshot(data).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Database {
            inner,
            tx: RefCell::new(None),
            closed: Cell::new(false),
        })
    }

    /// Creates a database from a tamper-evident binary snapshot.
    ///
    /// Verifies the HMAC-SHA256 tag produced by `exportSnapshotSigned()` in
    /// constant time before deserialising. Rejects snapshots that are
    /// truncated, tagged with a different key, or missing the `GSN1` header.
    ///
    /// ```js
    /// const restored = Database.importSnapshotSigned(signed, key);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the key is empty, the data exceeds the size
    /// limit, the header is missing, the MAC does not verify, or
    /// deserialisation fails.
    #[wasm_bindgen(js_name = "importSnapshotSigned")]
    pub fn import_snapshot_signed(data: &[u8], key: &[u8]) -> Result<Database, JsError> {
        utils::set_panic_hook();
        if key.is_empty() {
            return Err(JsError::new("importSnapshotSigned: key must not be empty"));
        }
        if data.len() > signed_snapshot::MAX_SNAPSHOT_BYTES {
            return Err(JsError::new(&format!(
                "importSnapshotSigned: snapshot exceeds {} MiB limit",
                signed_snapshot::MAX_SNAPSHOT_BYTES / (1024 * 1024)
            )));
        }
        let payload = signed_snapshot::unwrap(key, data).map_err(|msg| JsError::new(&msg))?;
        let inner = GrafeoDB::import_snapshot(payload).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Database {
            inner,
            tx: RefCell::new(None),
            closed: Cell::new(false),
        })
    }

    /// Returns schema information about the database.
    ///
    /// Returns an object describing labels, edge types, and property keys.
    ///
    /// ```js
    /// const schema = db.schema();
    /// // { lpg: { labels: [...], edgeTypes: [...], propertyKeys: [...] } }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the schema info cannot be serialised to a JS value.
    pub fn schema(&self) -> Result<JsValue, JsError> {
        let info = self.inner.schema();
        serde_wasm_bindgen::to_value(&info).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Creates a text index on a label+property pair for full-text (BM25) search.
    ///
    /// Indexes all existing nodes with matching label and string property values.
    ///
    /// ```js
    /// db.createTextIndex("Article", "content");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the text index cannot be created (e.g., invalid label or property).
    #[cfg(feature = "text-index")]
    #[wasm_bindgen(js_name = "createTextIndex")]
    pub fn create_text_index(&self, label: &str, property: &str) -> Result<(), JsError> {
        self.inner
            .create_text_index(label, property)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Drops a text index on a label+property pair.
    ///
    /// Returns `true` if the index existed and was removed.
    #[cfg(feature = "text-index")]
    #[wasm_bindgen(js_name = "dropTextIndex")]
    pub fn drop_text_index(&self, label: &str, property: &str) -> bool {
        self.inner.drop_text_index(label, property)
    }

    /// Rebuilds a text index by re-scanning all matching nodes.
    ///
    /// Use after bulk imports to refresh the index.
    ///
    /// # Errors
    ///
    /// Returns `JsError` if no text index exists for the given label and property, or if rebuilding fails.
    #[cfg(feature = "text-index")]
    #[wasm_bindgen(js_name = "rebuildTextIndex")]
    pub fn rebuild_text_index(&self, label: &str, property: &str) -> Result<(), JsError> {
        self.inner
            .rebuild_text_index(label, property)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Performs full-text search using BM25 ranking.
    ///
    /// Returns an array of `{id, score}` objects, ordered by relevance.
    ///
    /// ```js
    /// db.createTextIndex("Article", "content");
    /// const results = db.textSearch("Article", "content", "graph database", 10);
    /// // [{id: 42, score: 2.5}, {id: 17, score: 1.8}]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `JsError` if no text index exists for the label/property pair, or if the search fails.
    #[cfg(feature = "text-index")]
    #[wasm_bindgen(js_name = "textSearch")]
    pub fn text_search(
        &self,
        label: &str,
        property: &str,
        query: &str,
        k: usize,
    ) -> Result<JsValue, JsError> {
        let results = self
            .inner
            .text_search(label, property, query, k)
            .map_err(|e| JsError::new(&e.to_string()))?;

        let arr = Array::new_with_length(results.len() as u32);
        for (i, (id, score)) in results.iter().enumerate() {
            let obj = js_sys::Object::new();
            let _ = js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("id"),
                &JsValue::from_f64(id.0 as f64),
            );
            let _ = js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("score"),
                &JsValue::from_f64(*score),
            );
            arr.set(i as u32, obj.into());
        }
        Ok(arr.into())
    }

    /// Performs hybrid search combining text (BM25) and vector similarity.
    ///
    /// Uses Reciprocal Rank Fusion to combine results from both indexes.
    /// Returns an array of `{id, score}` objects.
    ///
    /// ```js
    /// const results = db.hybridSearch("Article", "content", "embedding", "graph databases", 10);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the required text or vector indexes are missing, or if the search fails.
    #[cfg(feature = "hybrid-search")]
    #[wasm_bindgen(js_name = "hybridSearch")]
    pub fn hybrid_search(
        &self,
        label: &str,
        text_property: &str,
        vector_property: &str,
        query_text: &str,
        k: usize,
    ) -> Result<JsValue, JsError> {
        let results = self
            .inner
            .hybrid_search(
                label,
                text_property,
                vector_property,
                query_text,
                None,
                k,
                None,
            )
            .map_err(|e| JsError::new(&e.to_string()))?;

        let arr = Array::new_with_length(results.len() as u32);
        for (i, (id, score)) in results.iter().enumerate() {
            let obj = js_sys::Object::new();
            let _ = js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("id"),
                &JsValue::from_f64(id.0 as f64),
            );
            let _ = js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("score"),
                &JsValue::from_f64(*score),
            );
            arr.set(i as u32, obj.into());
        }
        Ok(arr.into())
    }

    // ── Vector Index ──────────────────────────────────────────────────

    /// Creates a vector (HNSW) index on a label+property pair.
    ///
    /// Indexes all existing nodes whose property value is a float array.
    ///
    /// ```js
    /// db.createVectorIndex("Doc", "embedding", {
    ///   dimensions: 384,
    ///   metric: "cosine",       // "cosine" | "euclidean" | "dot_product" | "manhattan"
    ///   m: 16,                  // HNSW links per node
    ///   efConstruction: 128,    // build beam width
    /// });
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `JsError` if `options` cannot be deserialised or the vector index cannot be created.
    #[cfg(feature = "vector-index")]
    #[wasm_bindgen(js_name = "createVectorIndex")]
    pub fn create_vector_index(
        &self,
        label: &str,
        property: &str,
        options: JsValue,
    ) -> Result<(), JsError> {
        let opts: VectorIndexOptions = if options.is_undefined() || options.is_null() {
            VectorIndexOptions::default()
        } else {
            serde_wasm_bindgen::from_value(options)
                .map_err(|e| JsError::new(&format!("Invalid options: {e}")))?
        };

        self.inner
            .create_vector_index(
                label,
                property,
                opts.dimensions,
                opts.metric.as_deref(),
                opts.m,
                opts.ef_construction,
                opts.quantization.as_deref(),
            )
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Drops a vector index on a label+property pair.
    ///
    /// Returns `true` if the index existed and was removed.
    #[cfg(feature = "vector-index")]
    #[wasm_bindgen(js_name = "dropVectorIndex")]
    pub fn drop_vector_index(&self, label: &str, property: &str) -> bool {
        self.inner.drop_vector_index(label, property)
    }

    /// Rebuilds a vector index by re-scanning all matching nodes.
    ///
    /// Use after bulk imports to refresh the index. Preserves existing
    /// configuration (dimensions, metric, M, ef_construction).
    ///
    /// # Errors
    ///
    /// Returns `JsError` if no vector index exists for the given label and property, or if rebuilding fails.
    #[cfg(feature = "vector-index")]
    #[wasm_bindgen(js_name = "rebuildVectorIndex")]
    pub fn rebuild_vector_index(&self, label: &str, property: &str) -> Result<(), JsError> {
        self.inner
            .rebuild_vector_index(label, property)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Performs k-nearest-neighbor vector search.
    ///
    /// Returns an array of `{id, distance}` objects, ordered by proximity.
    ///
    /// ```js
    /// db.createVectorIndex("Doc", "embedding");
    /// const results = db.vectorSearch("Doc", "embedding",
    ///   new Float32Array([1.0, 0.0, 0.0]), 10, { ef: 200 });
    /// // [{id: 42, distance: 0.12}, {id: 17, distance: 0.34}]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `JsError` if `options` cannot be deserialised, no vector index exists, or the search fails.
    #[cfg(feature = "vector-index")]
    #[wasm_bindgen(js_name = "vectorSearch")]
    pub fn vector_search(
        &self,
        label: &str,
        property: &str,
        query: &[f32],
        k: usize,
        options: JsValue,
    ) -> Result<JsValue, JsError> {
        let opts: VectorSearchOptions = if options.is_undefined() || options.is_null() {
            VectorSearchOptions::default()
        } else {
            serde_wasm_bindgen::from_value(options)
                .map_err(|e| JsError::new(&format!("Invalid options: {e}")))?
        };

        let filters = opts.filters.as_ref().map(|f| {
            f.iter()
                .map(|(k, v)| (k.clone(), json_to_value(v)))
                .collect::<HashMap<String, Value>>()
        });

        let results = self
            .inner
            .vector_search(label, property, query, k, opts.ef, filters.as_ref())
            .map_err(|e| JsError::new(&e.to_string()))?;

        Ok(vector_results_to_js(&results))
    }

    /// Performs Maximal Marginal Relevance search for diverse results.
    ///
    /// Balances relevance and diversity via the `lambda` parameter
    /// (1.0 = pure relevance, 0.0 = pure diversity).
    ///
    /// ```js
    /// const results = db.mmrSearch("Doc", "embedding",
    ///   new Float32Array([1.0, 0.0, 0.0]), 5, { fetchK: 20, lambda: 0.7 });
    /// // [{id, distance}]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `JsError` if `options` cannot be deserialised, no vector index exists, or the search fails.
    #[cfg(feature = "vector-index")]
    #[wasm_bindgen(js_name = "mmrSearch")]
    pub fn mmr_search(
        &self,
        label: &str,
        property: &str,
        query: &[f32],
        k: usize,
        options: JsValue,
    ) -> Result<JsValue, JsError> {
        let opts: MmrSearchOptions = if options.is_undefined() || options.is_null() {
            MmrSearchOptions::default()
        } else {
            serde_wasm_bindgen::from_value(options)
                .map_err(|e| JsError::new(&format!("Invalid options: {e}")))?
        };

        let filters = opts.filters.as_ref().map(|f| {
            f.iter()
                .map(|(k, v)| (k.clone(), json_to_value(v)))
                .collect::<HashMap<String, Value>>()
        });

        let results = self
            .inner
            .mmr_search(
                label,
                property,
                query,
                k,
                opts.fetch_k,
                opts.lambda,
                opts.ef,
                filters.as_ref(),
            )
            .map_err(|e| JsError::new(&e.to_string()))?;

        Ok(vector_results_to_js(&results))
    }

    /// Executes a GQL query with parameters and returns results as an array of objects.
    ///
    /// Parameters are passed as a JavaScript object with string keys.
    /// Use `$name` syntax in the query to reference parameters.
    ///
    /// ```js
    /// const results = db.executeWithParams(
    ///   "MATCH (p:Person {name: $name}) RETURN p.name, p.age",
    ///   { name: "Alix" }
    /// );
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `JsError` if `params` is not a valid object, or if the query fails to parse or execute.
    #[wasm_bindgen(js_name = "executeWithParams")]
    pub fn execute_with_params(&self, query: &str, params: JsValue) -> Result<JsValue, JsError> {
        self.execute_language_impl(query, "gql", Some(params))
    }

    /// Executes a query using a specific language with parameters.
    ///
    /// Combines language selection with parameterised queries.
    ///
    /// ```js
    /// const results = db.executeWithLanguageAndParams(
    ///   "MATCH (p:Person {name: $name}) RETURN p.name",
    ///   "cypher",
    ///   { name: "Alix" }
    /// );
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `JsError` if `params` is invalid, the language is unsupported, or the query fails.
    #[wasm_bindgen(js_name = "executeWithLanguageAndParams")]
    pub fn execute_with_language_and_params(
        &self,
        query: &str,
        language: &str,
        params: JsValue,
    ) -> Result<JsValue, JsError> {
        self.execute_language_impl(query, language, Some(params))
    }

    /// Executes a Cypher query and returns results as an array of objects.
    ///
    /// Requires the `cypher` feature flag.
    ///
    /// ```js
    /// const results = db.executeCypher("MATCH (p:Person) RETURN p.name");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the Cypher query fails to parse or execute.
    #[cfg(feature = "cypher")]
    #[wasm_bindgen(js_name = "executeCypher")]
    pub fn execute_cypher(&self, query: &str) -> Result<JsValue, JsError> {
        self.execute_language_impl(query, "cypher", None)
    }

    /// Executes a Gremlin query and returns results as an array of objects.
    ///
    /// Requires the `gremlin` feature flag.
    ///
    /// ```js
    /// const results = db.executeGremlin("g.V().hasLabel('Person').values('name')");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the Gremlin query fails to parse or execute.
    #[cfg(feature = "gremlin")]
    #[wasm_bindgen(js_name = "executeGremlin")]
    pub fn execute_gremlin(&self, query: &str) -> Result<JsValue, JsError> {
        self.execute_language_impl(query, "gremlin", None)
    }

    /// Executes a GraphQL query and returns results as an array of objects.
    ///
    /// Requires the `graphql` feature flag.
    ///
    /// ```js
    /// const results = db.executeGraphql("{ Person { name age } }");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the GraphQL query fails to parse or execute.
    #[cfg(feature = "graphql")]
    #[wasm_bindgen(js_name = "executeGraphql")]
    pub fn execute_graphql(&self, query: &str) -> Result<JsValue, JsError> {
        self.execute_language_impl(query, "graphql", None)
    }

    /// Executes a SPARQL query and returns results as an array of objects.
    ///
    /// Requires the `sparql` feature flag.
    ///
    /// ```js
    /// const results = db.executeSparql("SELECT ?name WHERE { ?p a :Person ; :name ?name }");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the SPARQL query fails to parse or execute.
    #[cfg(feature = "sparql")]
    #[wasm_bindgen(js_name = "executeSparql")]
    pub fn execute_sparql(&self, query: &str) -> Result<JsValue, JsError> {
        self.execute_language_impl(query, "sparql", None)
    }

    /// Executes a SQL/PGQ query and returns results as an array of objects.
    ///
    /// Requires the `sql-pgq` feature flag.
    ///
    /// ```js
    /// const results = db.executeSql("SELECT * FROM GRAPH_TABLE (...)");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the SQL/PGQ query fails to parse or execute.
    #[cfg(feature = "sql-pgq")]
    #[wasm_bindgen(js_name = "executeSql")]
    pub fn execute_sql(&self, query: &str) -> Result<JsValue, JsError> {
        self.execute_language_impl(query, "sql", None)
    }

    /// Executes a query in a specific language and returns raw columns, rows, and metadata.
    ///
    /// Returns `{ columns: string[], rows: any[][], executionTimeMs?: number }`.
    ///
    /// ```js
    /// const raw = db.executeRawWithLanguage("MATCH (p:Person) RETURN p.name", "cypher");
    /// // { columns: ["p.name"], rows: [["Alix"], ["Gus"]], executionTimeMs: 0.5 }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the language is unsupported or the query fails to parse or execute.
    #[wasm_bindgen(js_name = "executeRawWithLanguage")]
    pub fn execute_raw_with_language(
        &self,
        query: &str,
        language: &str,
    ) -> Result<JsValue, JsError> {
        let result = self
            .inner
            .execute_language(query, language, None)
            .map_err(|e| JsError::new(&e.to_string()))?;

        let obj = js_sys::Object::new();

        // columns: string[]
        let cols = Array::new_with_length(result.columns.len() as u32);
        for (i, col) in result.columns.iter().enumerate() {
            cols.set(i as u32, JsValue::from_str(col));
        }
        let _ = js_sys::Reflect::set(&obj, &JsValue::from_str("columns"), &cols);

        // rows: any[][]
        let rows = Array::new_with_length(result.rows().len() as u32);
        for (i, row) in result.rows().iter().enumerate() {
            let js_row = Array::new_with_length(row.len() as u32);
            for (j, val) in row.iter().enumerate() {
                js_row.set(j as u32, types::value_to_js(val));
            }
            rows.set(i as u32, js_row.into());
        }
        let _ = js_sys::Reflect::set(&obj, &JsValue::from_str("rows"), &rows);

        // executionTimeMs?: number
        if let Some(ms) = result.execution_time_ms {
            let _ = js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("executionTimeMs"),
                &JsValue::from_f64(ms),
            );
        }

        Ok(obj.into())
    }

    /// Batch-imports LPG (Labeled Property Graph) data from a structured object.
    ///
    /// Nodes are created first, then edges. Edge `source`/`target` fields are
    /// zero-based indexes into the `nodes` array, so you can reference newly
    /// created nodes without knowing their database IDs.
    ///
    /// Returns `{ nodes: number, edges: number }` with the counts of created
    /// entities.
    ///
    /// ```js
    /// const result = db.importLpg({
    ///   nodes: [
    ///     { labels: ["Person"], properties: { name: "Alix", age: 30 } },
    ///     { labels: ["Person"], properties: { name: "Gus", age: 25 } },
    ///   ],
    ///   edges: [
    ///     { source: 0, target: 1, type: "KNOWS", properties: { since: 2020 } }
    ///   ]
    /// });
    /// // { nodes: 2, edges: 1 }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `JsError` if `data` cannot be deserialised or if an edge references an out-of-bounds node index.
    #[wasm_bindgen(js_name = "importLpg")]
    pub fn import_lpg(&self, data: JsValue) -> Result<JsValue, JsError> {
        let import: LpgImport = serde_wasm_bindgen::from_value(data)
            .map_err(|e| JsError::new(&format!("Invalid LPG data: {e}")))?;

        // Phase 1: create all nodes, collecting their IDs
        let mut node_ids = Vec::with_capacity(import.nodes.len());
        for node in &import.nodes {
            let labels: Vec<&str> = node.labels.iter().map(String::as_str).collect();
            let props: Vec<(PropertyKey, Value)> = node
                .properties
                .as_ref()
                .map(|p| {
                    p.iter()
                        .map(|(k, v)| (PropertyKey::new(k.as_str()), json_to_value(v)))
                        .collect()
                })
                .unwrap_or_default();
            let id = self
                .inner
                .create_node_with_props(&labels, props)
                .map_err(|e| JsError::new(&e.to_string()))?;
            node_ids.push(id);
        }

        // Phase 2: create edges using index-relative source/target
        let mut edge_count: u32 = 0;
        for (i, edge) in import.edges.iter().enumerate() {
            let src = *node_ids.get(edge.source).ok_or_else(|| {
                JsError::new(&format!(
                    "edges[{i}].source index {} out of bounds (0..{})",
                    edge.source,
                    node_ids.len()
                ))
            })?;
            let dst = *node_ids.get(edge.target).ok_or_else(|| {
                JsError::new(&format!(
                    "edges[{i}].target index {} out of bounds (0..{})",
                    edge.target,
                    node_ids.len()
                ))
            })?;
            let props: Vec<(PropertyKey, Value)> = edge
                .properties
                .as_ref()
                .map(|p| {
                    p.iter()
                        .map(|(k, v)| (PropertyKey::new(k.as_str()), json_to_value(v)))
                        .collect()
                })
                .unwrap_or_default();
            self.inner
                .create_edge_with_props(src, dst, &edge.edge_type, props)
                .map_err(|e| JsError::new(&e.to_string()))?;
            edge_count += 1;
        }

        let result = js_sys::Object::new();
        let _ = js_sys::Reflect::set(
            &result,
            &JsValue::from_str("nodes"),
            &JsValue::from_f64(f64::from(node_ids.len() as u32)),
        );
        let _ = js_sys::Reflect::set(
            &result,
            &JsValue::from_str("edges"),
            &JsValue::from_f64(f64::from(edge_count)),
        );
        Ok(result.into())
    }

    /// Batch-imports RDF triples from a structured object.
    ///
    /// Each triple has a `subject`, `predicate`, and `object`. Subjects and
    /// predicates are IRI strings (or blank nodes prefixed with `_:`). The
    /// object can be a plain string (treated as IRI) or a structured literal:
    ///
    /// ```js
    /// const result = db.importRdf({
    ///   triples: [
    ///     {
    ///       subject: "http://example.org/Alix",
    ///       predicate: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    ///       object: "http://example.org/Person"
    ///     },
    ///     {
    ///       subject: "http://example.org/Alix",
    ///       predicate: "http://example.org/name",
    ///       object: { value: "Alix" }
    ///     },
    ///     {
    ///       subject: "http://example.org/Alix",
    ///       predicate: "http://example.org/age",
    ///       object: { value: "30", datatype: "http://www.w3.org/2001/XMLSchema#integer" }
    ///     }
    ///   ]
    /// });
    /// // { triples: 3 }
    /// ```
    ///
    /// Requires the `rdf` feature flag.
    ///
    /// # Errors
    ///
    /// Returns `JsError` if `data` cannot be deserialised as an RDF import payload.
    #[cfg(feature = "rdf")]
    #[wasm_bindgen(js_name = "importRdf")]
    pub fn import_rdf(&self, data: JsValue) -> Result<JsValue, JsError> {
        use grafeo_core::graph::rdf::Term;

        let import: RdfImport = serde_wasm_bindgen::from_value(data)
            .map_err(|e| JsError::new(&format!("Invalid RDF data: {e}")))?;

        let triples = import.triples.into_iter().map(|t| {
            let subject = string_to_rdf_term(&t.subject);
            let predicate = string_to_rdf_term(&t.predicate);
            let object = match t.object {
                RdfObjectSpec::Iri(ref s) => string_to_rdf_term(s),
                RdfObjectSpec::Literal {
                    ref value,
                    ref datatype,
                    ref language,
                } => {
                    if let Some(lang) = language {
                        Term::lang_literal(value.as_str(), lang.as_str())
                    } else if let Some(dt) = datatype {
                        Term::typed_literal(value.as_str(), dt.as_str())
                    } else {
                        Term::literal(value.as_str())
                    }
                }
            };
            grafeo_core::graph::rdf::Triple::new(subject, predicate, object)
        });

        let inserted = self.inner.batch_insert_rdf(triples);

        let result = js_sys::Object::new();
        let _ = js_sys::Reflect::set(
            &result,
            &JsValue::from_str("triples"),
            &JsValue::from_f64(inserted as f64),
        );
        Ok(result.into())
    }

    /// Returns a hierarchical memory usage breakdown.
    ///
    /// The returned object mirrors the engine's `MemoryUsage` struct with
    /// `totalBytes`, `store`, `indexes`, `mvcc`, `caches`, `stringPool`,
    /// and `bufferManager` sections.
    ///
    /// ```js
    /// const usage = db.memoryUsage();
    /// console.log(`Total: ${usage.total_bytes} bytes`);
    /// console.log(`Store: ${usage.store.total_bytes} bytes`);
    /// console.log(`Indexes: ${usage.indexes.total_bytes} bytes`);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the memory usage data cannot be serialised to a JS value.
    #[wasm_bindgen(js_name = "memoryUsage")]
    pub fn memory_usage(&self) -> Result<JsValue, JsError> {
        let usage = self.inner.memory_usage();
        serde_wasm_bindgen::to_value(&usage).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Returns high-level database information (counts, mode, features).
    ///
    /// # Errors
    ///
    /// Returns `JsError` if the info data cannot be serialised to a JS value.
    pub fn info(&self) -> Result<JsValue, JsError> {
        let info = self.inner.info();
        serde_wasm_bindgen::to_value(&info).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Converts the database to a read-only CompactStore for faster queries.
    ///
    /// Takes a snapshot of all nodes and edges, builds a columnar store with
    /// CSR adjacency, and switches to read-only mode. After this call, write
    /// operations will fail. Gives ~60x memory reduction and 100x+ traversal
    /// speedup for read-only workloads.
    ///
    /// # Errors
    ///
    /// Returns `JsError` if compaction fails (e.g., the database is already in compact mode).
    #[cfg(feature = "compact-store")]
    pub fn compact(&mut self) -> Result<(), JsError> {
        self.inner
            .compact()
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Bulk-imports rows (array of objects) as nodes or edges.
    ///
    /// This is the WASM equivalent of Python's `import_df()`: each object
    /// in the array becomes a node or edge, with object keys as property names.
    ///
    /// **Node import** (`mode: "nodes"`): requires `label` (string or string[]).
    /// All object keys become node properties.
    ///
    /// **Edge import** (`mode: "edges"`): requires `edgeType`. The `source`
    /// and `target` keys in each object must contain integer node IDs.
    /// Remaining keys become edge properties. Override column names with
    /// the `source` and `target` options (default `"source"` / `"target"`).
    ///
    /// Returns the number of created entities.
    ///
    /// ```js
    /// // Import nodes
    /// const count = db.importRows(
    ///   [{ name: "Alix", age: 30 }, { name: "Gus", age: 25 }],
    ///   { mode: "nodes", label: "Person" }
    /// );
    ///
    /// // Import edges
    /// const edgeCount = db.importRows(
    ///   [{ source: 0, target: 1, since: 2020 }],
    ///   { mode: "edges", edgeType: "KNOWS" }
    /// );
    ///
    /// // Custom source/target column names
    /// const edgeCount2 = db.importRows(
    ///   [{ from: 0, to: 1 }],
    ///   { mode: "edges", edgeType: "KNOWS", source: "from", target: "to" }
    /// );
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `JsError` if:
    /// - `options` cannot be deserialised or has an invalid `mode`.
    /// - `rows` is not an array of objects.
    /// - A required column (`label`, `edgeType`, `source`, `target`) is missing.
    /// - A source/target value is not a valid non-negative integer.
    #[wasm_bindgen(js_name = "importRows")]
    pub fn import_rows(&self, rows: JsValue, options: JsValue) -> Result<u32, JsError> {
        let opts: ImportRowsOptions = serde_wasm_bindgen::from_value(options)
            .map_err(|e| JsError::new(&format!("Invalid options: {e}")))?;
        let data: Vec<serde_json::Map<String, serde_json::Value>> =
            serde_wasm_bindgen::from_value(rows)
                .map_err(|e| JsError::new(&format!("rows must be an array of objects: {e}")))?;

        let mut count: u32 = 0;

        match opts.mode.as_str() {
            "nodes" => {
                let labels = opts.labels()?;
                let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();

                for row in &data {
                    let props: Vec<(PropertyKey, Value)> = row
                        .iter()
                        .filter(|(_, v)| !v.is_null())
                        .map(|(k, v)| (PropertyKey::new(k.as_str()), json_to_value(v)))
                        .collect();
                    self.inner
                        .create_node_with_props(&label_refs, props)
                        .map_err(|e| JsError::new(&format!("rows[{count}]: {e}")))?;
                    count += 1;
                }
            }
            "edges" => {
                let edge_type = opts
                    .edge_type
                    .as_deref()
                    .ok_or_else(|| JsError::new("edgeType is required for mode 'edges'"))?;
                let source_col = opts.source.as_deref().unwrap_or("source");
                let target_col = opts.target.as_deref().unwrap_or("target");

                for (i, row) in data.iter().enumerate() {
                    let src_val = row.get(source_col).ok_or_else(|| {
                        JsError::new(&format!("rows[{i}]: missing '{source_col}' column"))
                    })?;
                    let dst_val = row.get(target_col).ok_or_else(|| {
                        JsError::new(&format!("rows[{i}]: missing '{target_col}' column"))
                    })?;

                    let src_id = json_to_node_id(src_val, source_col, i)?;
                    let dst_id = json_to_node_id(dst_val, target_col, i)?;

                    let props: Vec<(PropertyKey, Value)> = row
                        .iter()
                        .filter(|(k, v)| {
                            k.as_str() != source_col && k.as_str() != target_col && !v.is_null()
                        })
                        .map(|(k, v)| (PropertyKey::new(k.as_str()), json_to_value(v)))
                        .collect();

                    self.inner
                        .create_edge_with_props(src_id, dst_id, edge_type, props)
                        .map_err(|e| JsError::new(&format!("rows[{i}]: {e}")))?;
                    count += 1;
                }
            }
            other => {
                return Err(JsError::new(&format!(
                    "mode must be 'nodes' or 'edges', got '{other}'"
                )));
            }
        }

        Ok(count)
    }

    /// Returns the Grafeo version.
    pub fn version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    // ── Schema context ───────────────────────────────────────────────────

    /// Sets the current schema for subsequent `execute()` calls.
    ///
    /// Equivalent to `SESSION SET SCHEMA name` but persists across calls.
    /// Call `resetSchema()` to clear it.
    ///
    /// # Errors
    ///
    /// Returns an error if the schema does not exist.
    ///
    /// ```js
    /// db.setSchema("reporting");
    /// const types = db.execute("SHOW GRAPH TYPES"); // only sees 'reporting' types
    /// ```
    #[wasm_bindgen(js_name = "setSchema")]
    pub fn set_schema(&self, name: &str) -> Result<(), JsValue> {
        self.inner
            .set_current_schema(Some(name))
            .map_err(|e| JsError::new(&e.to_string()).into())
    }

    /// Clears the current schema context.
    ///
    /// Subsequent `execute()` calls will use the default (no-schema) namespace.
    #[wasm_bindgen(js_name = "resetSchema")]
    pub fn reset_schema(&self) {
        let _ = self.inner.set_current_schema(None);
    }

    /// Returns the current schema name, or `undefined` if no schema is set.
    #[wasm_bindgen(js_name = "currentSchema")]
    pub fn current_schema(&self) -> Option<String> {
        self.inner.current_schema()
    }

    // ── Graph projections ───────────────────────────────────────────────

    /// Creates a named graph projection. Returns `true` if created, `false`
    /// if a projection with that name already exists.
    ///
    /// A projection is a read-only, filtered view of the default graph.
    /// Only nodes with matching labels and edges with matching types are visible.
    ///
    /// # Errors
    ///
    /// This method does not currently return errors.
    #[wasm_bindgen(js_name = "createProjection")]
    pub fn create_projection(
        &self,
        name: &str,
        node_labels: Option<Vec<String>>,
        edge_types: Option<Vec<String>>,
    ) -> bool {
        use grafeo_engine::ProjectionSpec;

        let mut spec = ProjectionSpec::new();
        if let Some(labels) = node_labels.filter(|l| !l.is_empty()) {
            spec = spec.with_node_labels(labels);
        }
        if let Some(types) = edge_types.filter(|t| !t.is_empty()) {
            spec = spec.with_edge_types(types);
        }
        self.inner.create_projection(name, spec)
    }

    /// Drops a named graph projection. Returns `true` if it existed.
    #[wasm_bindgen(js_name = "dropProjection")]
    pub fn drop_projection(&self, name: &str) -> bool {
        self.inner.drop_projection(name)
    }

    /// Returns the names of all graph projections.
    #[wasm_bindgen(js_name = "listProjections")]
    pub fn list_projections(&self) -> Vec<String> {
        self.inner.list_projections()
    }
}

// ---------------------------------------------------------------------------
// Private helpers (not exported to JS)
// ---------------------------------------------------------------------------

impl Database {
    /// Shared implementation for all language-specific execute methods.
    ///
    /// Converts an optional JS params object to the internal
    /// `HashMap<String, Value>` representation and delegates to
    /// `GrafeoDB::execute_language`.
    fn execute_language_impl(
        &self,
        query: &str,
        language: &str,
        params: Option<JsValue>,
    ) -> Result<JsValue, JsError> {
        self.check_open()?;
        let param_map = Self::convert_params(params)?;
        let result = self.run_language_query(query, language, param_map)?;

        let rows = Array::new_with_length(result.rows().len() as u32);
        for (i, row) in result.rows().iter().enumerate() {
            rows.set(i as u32, types::row_to_js_object(&result.columns, row));
        }
        Ok(rows.into())
    }

    /// Converts a JS params value (object or null/undefined) to an optional
    /// `HashMap<String, Value>` suitable for `execute_language`.
    fn convert_params(params: Option<JsValue>) -> Result<Option<HashMap<String, Value>>, JsError> {
        let Some(js_val) = params else {
            return Ok(None);
        };
        if js_val.is_null() || js_val.is_undefined() {
            return Ok(None);
        }
        let json_val: serde_json::Value =
            serde_wasm_bindgen::from_value(js_val).map_err(|e| JsError::new(&e.to_string()))?;
        json_params_to_map(Some(&json_val)).map_err(|e| JsError::new(&e))
    }
}

// ---------------------------------------------------------------------------
// Vector search option types (serde, not exported to JS)
// ---------------------------------------------------------------------------

/// Options for `createVectorIndex()`.
#[cfg(feature = "vector-index")]
#[derive(Default, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct VectorIndexOptions {
    dimensions: Option<usize>,
    metric: Option<String>,
    m: Option<usize>,
    ef_construction: Option<usize>,
    quantization: Option<String>,
}

/// Options for `vectorSearch()`.
#[cfg(feature = "vector-index")]
#[derive(Default, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct VectorSearchOptions {
    ef: Option<usize>,
    filters: Option<HashMap<String, serde_json::Value>>,
}

/// Options for `mmrSearch()`.
#[cfg(feature = "vector-index")]
#[derive(Default, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct MmrSearchOptions {
    fetch_k: Option<usize>,
    lambda: Option<f32>,
    ef: Option<usize>,
    filters: Option<HashMap<String, serde_json::Value>>,
}

/// Converts a `Vec<(NodeId, f32)>` to a JS array of `{id, distance}` objects.
#[cfg(feature = "vector-index")]
fn vector_results_to_js(results: &[(grafeo_common::types::NodeId, f32)]) -> JsValue {
    let arr = Array::new_with_length(results.len() as u32);
    for (i, (id, distance)) in results.iter().enumerate() {
        let obj = js_sys::Object::new();
        let _ = js_sys::Reflect::set(
            &obj,
            &JsValue::from_str("id"),
            &JsValue::from_f64(id.0 as f64),
        );
        let _ = js_sys::Reflect::set(
            &obj,
            &JsValue::from_str("distance"),
            &JsValue::from_f64(f64::from(*distance)),
        );
        arr.set(i as u32, obj.into());
    }
    arr.into()
}

// ---------------------------------------------------------------------------
// Batch import data types (serde, not exported to JS)
// ---------------------------------------------------------------------------

/// Options for `importRows()`.
#[derive(serde::Deserialize)]
struct ImportRowsOptions {
    mode: String,
    /// Node label(s): a single string or an array of strings.
    #[serde(default)]
    label: Option<ImportLabel>,
    /// Edge type (required for mode "edges").
    #[serde(default, rename = "edgeType")]
    edge_type: Option<String>,
    /// Source column name (default "source").
    #[serde(default)]
    source: Option<String>,
    /// Target column name (default "target").
    #[serde(default)]
    target: Option<String>,
}

/// A label can be a single string or an array of strings.
#[derive(serde::Deserialize)]
#[serde(untagged)]
enum ImportLabel {
    Single(String),
    Multiple(Vec<String>),
}

impl ImportRowsOptions {
    fn labels(&self) -> Result<Vec<String>, JsError> {
        match &self.label {
            Some(ImportLabel::Single(s)) => Ok(vec![s.clone()]),
            Some(ImportLabel::Multiple(v)) => Ok(v.clone()),
            None => Err(JsError::new("label is required for mode 'nodes'")),
        }
    }
}

/// Extracts a `NodeId` from a JSON number value.
fn json_to_node_id(
    val: &serde_json::Value,
    col_name: &str,
    row_idx: usize,
) -> Result<grafeo_common::types::NodeId, JsError> {
    let n = val
        .as_u64()
        .or_else(|| {
            val.as_f64().and_then(|f| {
                // reason: Reject negative, NaN, Infinity, fractional, and out-of-range values
                if (0.0..=9_007_199_254_740_991.0).contains(&f) && f.fract() == 0.0 {
                    // reason: range check above rejects negative values
                    #[allow(clippy::cast_sign_loss)]
                    Some(f as u64)
                } else {
                    None
                }
            })
        })
        .ok_or_else(|| {
            JsError::new(&format!(
                "rows[{row_idx}].{col_name}: expected a non-negative integer, got {val}"
            ))
        })?;
    Ok(grafeo_common::types::NodeId::new(n))
}

/// LPG batch import payload.
#[derive(serde::Deserialize)]
struct LpgImport {
    nodes: Vec<LpgNodeSpec>,
    #[serde(default)]
    edges: Vec<LpgEdgeSpec>,
}

/// A single node in an LPG import.
#[derive(serde::Deserialize)]
struct LpgNodeSpec {
    labels: Vec<String>,
    #[serde(default)]
    properties: Option<serde_json::Map<String, serde_json::Value>>,
}

/// A single edge in an LPG import. `source` and `target` are zero-based
/// indexes into the `nodes` array.
#[derive(serde::Deserialize)]
struct LpgEdgeSpec {
    source: usize,
    target: usize,
    #[serde(rename = "type")]
    edge_type: String,
    #[serde(default)]
    properties: Option<serde_json::Map<String, serde_json::Value>>,
}

/// RDF batch import payload.
#[cfg(feature = "rdf")]
#[derive(serde::Deserialize)]
struct RdfImport {
    triples: Vec<RdfTripleSpec>,
}

/// A single RDF triple in an import.
#[cfg(feature = "rdf")]
#[derive(serde::Deserialize)]
struct RdfTripleSpec {
    subject: String,
    predicate: String,
    object: RdfObjectSpec,
}

/// The object position of an RDF triple: either a plain IRI string or a
/// structured literal with optional datatype/language.
#[cfg(feature = "rdf")]
#[derive(serde::Deserialize)]
#[serde(untagged)]
enum RdfObjectSpec {
    /// Plain string: treated as IRI, or blank node if prefixed with `_:`.
    Iri(String),
    /// Structured literal with optional datatype or language tag.
    Literal {
        value: String,
        #[serde(default)]
        datatype: Option<String>,
        #[serde(default)]
        language: Option<String>,
    },
}

/// Converts a string to an RDF [`Term`]: blank node if prefixed with `_:`,
/// IRI otherwise.
#[cfg(feature = "rdf")]
fn string_to_rdf_term(s: &str) -> grafeo_core::graph::rdf::Term {
    if let Some(id) = s.strip_prefix("_:") {
        grafeo_core::graph::rdf::Term::blank(id)
    } else {
        grafeo_core::graph::rdf::Term::iri(s)
    }
}

// ---------------------------------------------------------------------------
// Unit tests (native, no wasm32 requirement)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    // === Vector options deserialization tests ===

    #[cfg(feature = "vector-index")]
    mod vector_tests {
        use serde_json::json;

        use super::super::*;

        #[test]
        fn vector_index_options_defaults() {
            let opts: VectorIndexOptions = serde_json::from_value(json!({})).unwrap();
            assert!(opts.dimensions.is_none());
            assert!(opts.metric.is_none());
            assert!(opts.m.is_none());
            assert!(opts.ef_construction.is_none());
        }

        #[test]
        fn vector_index_options_full() {
            let opts: VectorIndexOptions = serde_json::from_value(json!({
                "dimensions": 384,
                "metric": "cosine",
                "m": 16,
                "efConstruction": 128
            }))
            .unwrap();
            assert_eq!(opts.dimensions, Some(384));
            assert_eq!(opts.metric.as_deref(), Some("cosine"));
            assert_eq!(opts.m, Some(16));
            assert_eq!(opts.ef_construction, Some(128));
        }

        #[test]
        fn vector_search_options_with_filters() {
            let opts: VectorSearchOptions = serde_json::from_value(json!({
                "ef": 200,
                "filters": { "category": "science" }
            }))
            .unwrap();
            assert_eq!(opts.ef, Some(200));
            assert!(opts.filters.is_some());
            assert_eq!(opts.filters.unwrap()["category"], json!("science"));
        }

        #[test]
        fn mmr_search_options_partial() {
            let opts: MmrSearchOptions =
                serde_json::from_value(json!({ "fetchK": 20, "lambda": 0.7 })).unwrap();
            assert_eq!(opts.fetch_k, Some(20));
            assert_eq!(opts.lambda, Some(0.7));
            assert!(opts.ef.is_none());
            assert!(opts.filters.is_none());
        }

        // vector_results_to_js requires a JS runtime, tested via wasm-bindgen-test

        #[test]
        fn create_vector_index_and_search() {
            use grafeo_common::types::{PropertyKey, Value};

            let db = GrafeoDB::new_in_memory();
            // Create index first, then insert nodes with Value::Vector
            db.create_vector_index(
                "Doc",
                "embedding",
                Some(3),
                Some("cosine"),
                None,
                None,
                None,
            )
            .unwrap();

            let vecs: &[&[f32]] = &[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]];
            for (i, v) in vecs.iter().enumerate() {
                let id = db
                    .create_node_with_props(
                        &["Doc"],
                        vec![(PropertyKey::new("title"), Value::from(format!("doc_{i}")))],
                    )
                    .unwrap();
                db.set_node_property(id, "embedding", Value::Vector(v.to_vec().into()))
                    .unwrap();
            }

            let results = db
                .vector_search("Doc", "embedding", &[1.0, 0.0, 0.0], 2, None, None)
                .unwrap();
            assert_eq!(results.len(), 2);
            assert!(
                results[0].1 <= results[1].1,
                "results should be sorted by distance"
            );
        }

        #[test]
        fn mmr_search_returns_diverse_results() {
            use grafeo_common::types::{PropertyKey, Value};

            let db = GrafeoDB::new_in_memory();
            db.create_vector_index(
                "Doc",
                "embedding",
                Some(3),
                Some("cosine"),
                None,
                None,
                None,
            )
            .unwrap();

            for i in 0..5 {
                let x = if i < 3 { 1.0f32 } else { 0.0 };
                let y = if i >= 3 { 1.0f32 } else { 0.0 };
                let id = db
                    .create_node_with_props(
                        &["Doc"],
                        vec![(PropertyKey::new("idx"), Value::Int64(i))],
                    )
                    .unwrap();
                db.set_node_property(id, "embedding", Value::Vector(vec![x, y, 0.0].into()))
                    .unwrap();
            }

            let results = db
                .mmr_search(
                    "Doc",
                    "embedding",
                    &[1.0, 0.0, 0.0],
                    3,
                    Some(5),
                    Some(0.5),
                    None,
                    None,
                )
                .unwrap();
            assert_eq!(results.len(), 3);
        }
    }

    // === LPG deserialization tests ===

    #[test]
    fn lpg_import_nodes_only() {
        let input = json!({
            "nodes": [
                { "labels": ["Person"], "properties": { "name": "Alix", "age": 30 } },
                { "labels": ["Person"], "properties": { "name": "Gus" } }
            ]
        });
        let import: LpgImport = serde_json::from_value(input).unwrap();
        assert_eq!(import.nodes.len(), 2);
        assert!(import.edges.is_empty(), "edges should default to empty");
    }

    #[test]
    fn lpg_import_nodes_and_edges() {
        let input = json!({
            "nodes": [
                { "labels": ["Person"], "properties": { "name": "Alix" } },
                { "labels": ["Person"], "properties": { "name": "Gus" } }
            ],
            "edges": [
                { "source": 0, "target": 1, "type": "KNOWS", "properties": { "since": 2020 } }
            ]
        });
        let import: LpgImport = serde_json::from_value(input).unwrap();
        assert_eq!(import.nodes.len(), 2);
        assert_eq!(import.edges.len(), 1);
        assert_eq!(import.edges[0].source, 0);
        assert_eq!(import.edges[0].target, 1);
        assert_eq!(import.edges[0].edge_type, "KNOWS");
    }

    #[test]
    fn lpg_import_empty() {
        let input = json!({ "nodes": [] });
        let import: LpgImport = serde_json::from_value(input).unwrap();
        assert!(import.nodes.is_empty());
        assert!(import.edges.is_empty());
    }

    #[test]
    fn lpg_import_node_without_properties() {
        let input = json!({
            "nodes": [{ "labels": ["Tag"] }]
        });
        let import: LpgImport = serde_json::from_value(input).unwrap();
        assert!(import.nodes[0].properties.is_none());
    }

    #[test]
    fn lpg_import_multiple_labels() {
        let input = json!({
            "nodes": [{ "labels": ["Person", "Employee", "Developer"] }]
        });
        let import: LpgImport = serde_json::from_value(input).unwrap();
        assert_eq!(
            import.nodes[0].labels,
            vec!["Person", "Employee", "Developer"]
        );
    }

    #[test]
    fn lpg_import_mixed_property_types() {
        let input = json!({
            "nodes": [{
                "labels": ["Thing"],
                "properties": {
                    "name": "test",
                    "count": 42,
                    "ratio": 1.23,
                    "active": true,
                    "tags": ["a", "b"],
                    "meta": null
                }
            }]
        });
        let import: LpgImport = serde_json::from_value(input).unwrap();
        let props = import.nodes[0].properties.as_ref().unwrap();
        assert_eq!(props.len(), 6);
        assert_eq!(props["name"], json!("test"));
        assert_eq!(props["count"], json!(42));
        assert_eq!(props["ratio"], json!(1.23));
        assert_eq!(props["active"], json!(true));
        assert_eq!(props["tags"], json!(["a", "b"]));
        assert!(props["meta"].is_null());
    }

    #[test]
    fn lpg_import_self_loop_edge() {
        let input = json!({
            "nodes": [{ "labels": ["Node"] }],
            "edges": [{ "source": 0, "target": 0, "type": "SELF" }]
        });
        let import: LpgImport = serde_json::from_value(input).unwrap();
        assert_eq!(import.edges[0].source, 0);
        assert_eq!(import.edges[0].target, 0);
    }

    #[test]
    fn lpg_import_edge_without_properties() {
        let input = json!({
            "nodes": [{ "labels": ["A"] }, { "labels": ["B"] }],
            "edges": [{ "source": 0, "target": 1, "type": "LINKED" }]
        });
        let import: LpgImport = serde_json::from_value(input).unwrap();
        assert!(import.edges[0].properties.is_none());
    }

    #[test]
    fn lpg_import_missing_nodes_field_errors() {
        let input = json!({ "edges": [] });
        let result: Result<LpgImport, _> = serde_json::from_value(input);
        assert!(result.is_err(), "missing 'nodes' field should fail");
    }

    #[test]
    fn lpg_import_missing_edge_type_errors() {
        let input = json!({
            "nodes": [{ "labels": ["A"] }],
            "edges": [{ "source": 0, "target": 0 }]
        });
        let result: Result<LpgImport, _> = serde_json::from_value(input);
        assert!(result.is_err(), "edge without 'type' should fail");
    }

    // === memoryUsage tests ===

    #[test]
    fn memory_usage_returns_hierarchical_breakdown() {
        let db = GrafeoDB::new_in_memory();
        db.create_node_with_props(
            &["Person"],
            vec![
                (PropertyKey::new("name"), Value::from("Alix")),
                (PropertyKey::new("age"), Value::Int64(30)),
            ],
        )
        .unwrap();

        let usage = db.memory_usage();
        assert!(usage.total_bytes > 0, "should report non-zero memory");
        assert!(usage.store.total_bytes > 0, "store should use memory");
        assert!(usage.store.nodes_bytes > 0, "should have node storage");
    }

    #[test]
    fn memory_usage_empty_db() {
        let db = GrafeoDB::new_in_memory();
        let usage = db.memory_usage();
        // Even an empty DB has some baseline allocation
        assert_eq!(usage.store.nodes_bytes, 0);
        assert_eq!(usage.store.edges_bytes, 0);
    }

    #[test]
    fn memory_usage_serializes_to_json() {
        let db = GrafeoDB::new_in_memory();
        let usage = db.memory_usage();
        let json = serde_json::to_value(&usage).unwrap();
        assert!(json.get("total_bytes").is_some());
        assert!(json.get("store").is_some());
        assert!(json.get("indexes").is_some());
        assert!(json.get("mvcc").is_some());
        assert!(json.get("caches").is_some());
        assert!(json.get("string_pool").is_some());
        assert!(json.get("buffer_manager").is_some());
    }

    // === importRows options deserialization tests ===

    #[test]
    fn import_rows_options_single_label() {
        let input = json!({ "mode": "nodes", "label": "Person" });
        let opts: ImportRowsOptions = serde_json::from_value(input).unwrap();
        assert_eq!(opts.mode, "nodes");
        let labels = opts.labels().unwrap();
        assert_eq!(labels, vec!["Person"]);
    }

    #[test]
    fn import_rows_options_multiple_labels() {
        let input = json!({ "mode": "nodes", "label": ["Person", "Employee"] });
        let opts: ImportRowsOptions = serde_json::from_value(input).unwrap();
        let labels = opts.labels().unwrap();
        assert_eq!(labels, vec!["Person", "Employee"]);
    }

    #[test]
    fn import_rows_options_edge_mode() {
        let input = json!({ "mode": "edges", "edgeType": "KNOWS" });
        let opts: ImportRowsOptions = serde_json::from_value(input).unwrap();
        assert_eq!(opts.mode, "edges");
        assert_eq!(opts.edge_type.as_deref(), Some("KNOWS"));
    }

    #[test]
    fn import_rows_options_custom_columns() {
        let input = json!({
            "mode": "edges",
            "edgeType": "LINKED",
            "source": "from",
            "target": "to"
        });
        let opts: ImportRowsOptions = serde_json::from_value(input).unwrap();
        assert_eq!(opts.source.as_deref(), Some("from"));
        assert_eq!(opts.target.as_deref(), Some("to"));
    }

    #[test]
    fn import_rows_options_missing_label_is_none() {
        let input = json!({ "mode": "nodes" });
        let opts: ImportRowsOptions = serde_json::from_value(input).unwrap();
        assert!(opts.label.is_none(), "label should be None when omitted");
    }

    #[test]
    fn json_to_node_id_integer() {
        let val = json!(42);
        let id = json_to_node_id(&val, "source", 0).unwrap();
        assert_eq!(id, grafeo_common::types::NodeId::new(42));
    }

    #[test]
    fn json_to_node_id_float_truncates() {
        let val = json!(7.0);
        let id = json_to_node_id(&val, "target", 0).unwrap();
        assert_eq!(id, grafeo_common::types::NodeId::new(7));
    }

    #[test]
    fn json_to_node_id_string_is_not_u64() {
        let val = json!("not_a_number");
        // as_u64 and as_f64 both return None for strings
        assert!(val.as_u64().is_none());
        assert!(val.as_f64().is_none());
    }

    // === Engine-level importRows tests ===

    #[test]
    fn import_rows_nodes_basic() {
        let db = GrafeoDB::new_in_memory();
        let rows: Vec<serde_json::Map<String, serde_json::Value>> = serde_json::from_value(json!([
            { "name": "Alix", "age": 30 },
            { "name": "Gus", "age": 25 }
        ]))
        .unwrap();

        let label_refs = vec!["Person"];
        for row in &rows {
            let props: Vec<(PropertyKey, Value)> = row
                .iter()
                .filter(|(_, v)| !v.is_null())
                .map(|(k, v)| (PropertyKey::new(k.as_str()), json_to_value(v)))
                .collect();
            db.create_node_with_props(&label_refs, props).unwrap();
        }

        assert_eq!(db.node_count(), 2);
        let session = db.session();
        let result = session
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap();
        assert_eq!(result.rows().len(), 2);
    }

    #[test]
    fn import_rows_edges_basic() {
        let db = GrafeoDB::new_in_memory();
        let alix = db
            .create_node_with_props(
                &["Person"],
                vec![(PropertyKey::new("name"), Value::from("Alix"))],
            )
            .unwrap();
        let gus = db
            .create_node_with_props(
                &["Person"],
                vec![(PropertyKey::new("name"), Value::from("Gus"))],
            )
            .unwrap();

        let rows: Vec<serde_json::Map<String, serde_json::Value>> = serde_json::from_value(json!([
            { "source": alix.0, "target": gus.0, "since": 2020 }
        ]))
        .unwrap();

        for row in &rows {
            let src = json_to_node_id(&row["source"], "source", 0).unwrap();
            let dst = json_to_node_id(&row["target"], "target", 0).unwrap();
            let props: Vec<(PropertyKey, Value)> = row
                .iter()
                .filter(|(k, v)| k.as_str() != "source" && k.as_str() != "target" && !v.is_null())
                .map(|(k, v)| (PropertyKey::new(k.as_str()), json_to_value(v)))
                .collect();
            db.create_edge_with_props(src, dst, "KNOWS", props).unwrap();
        }

        assert_eq!(db.edge_count(), 1);
    }

    #[test]
    fn import_rows_null_values_filtered() {
        let db = GrafeoDB::new_in_memory();
        let rows: Vec<serde_json::Map<String, serde_json::Value>> = serde_json::from_value(json!([
            { "name": "Alix", "nickname": null, "age": 30 }
        ]))
        .unwrap();

        for row in &rows {
            let props: Vec<(PropertyKey, Value)> = row
                .iter()
                .filter(|(_, v)| !v.is_null())
                .map(|(k, v)| (PropertyKey::new(k.as_str()), json_to_value(v)))
                .collect();
            db.create_node_with_props(&["Person"], props).unwrap();
        }

        assert_eq!(db.node_count(), 1);
        let session = db.session();
        let result = session
            .execute("MATCH (p:Person) RETURN p.nickname")
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::Null);
    }

    #[test]
    fn import_rows_large_batch() {
        let db = GrafeoDB::new_in_memory();
        let rows: Vec<serde_json::Map<String, serde_json::Value>> = (0..500)
            .map(|i| {
                let mut map = serde_json::Map::new();
                map.insert("index".to_string(), json!(i));
                map
            })
            .collect();

        for row in &rows {
            let props: Vec<(PropertyKey, Value)> = row
                .iter()
                .map(|(k, v)| (PropertyKey::new(k.as_str()), json_to_value(v)))
                .collect();
            db.create_node_with_props(&["Item"], props).unwrap();
        }

        assert_eq!(db.node_count(), 500);
    }

    // === Engine-level LPG batch tests ===

    #[test]
    fn import_lpg_creates_nodes_and_edges() {
        let db = GrafeoDB::new_in_memory();
        let input: LpgImport = serde_json::from_value(json!({
            "nodes": [
                { "labels": ["Person"], "properties": { "name": "Alix", "age": 30 } },
                { "labels": ["Person"], "properties": { "name": "Gus", "age": 25 } },
                { "labels": ["City"], "properties": { "name": "Amsterdam" } }
            ],
            "edges": [
                { "source": 0, "target": 1, "type": "KNOWS" },
                { "source": 0, "target": 2, "type": "LIVES_IN" }
            ]
        }))
        .unwrap();

        let mut node_ids = Vec::with_capacity(input.nodes.len());
        for node in &input.nodes {
            let labels: Vec<&str> = node.labels.iter().map(String::as_str).collect();
            let props: Vec<(PropertyKey, Value)> = node
                .properties
                .as_ref()
                .map(|p| {
                    p.iter()
                        .map(|(k, v)| (PropertyKey::new(k.as_str()), json_to_value(v)))
                        .collect()
                })
                .unwrap_or_default();
            let id = db.create_node_with_props(&labels, props).unwrap();
            node_ids.push(id);
        }

        for edge in &input.edges {
            let src = node_ids[edge.source];
            let dst = node_ids[edge.target];
            db.create_edge_with_props(
                src,
                dst,
                &edge.edge_type,
                std::iter::empty::<(PropertyKey, Value)>(),
            )
            .unwrap();
        }

        assert_eq!(db.node_count(), 3);
        assert_eq!(db.edge_count(), 2);

        // Verify data via query
        let session = db.session();
        let result = session
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap();
        assert_eq!(result.rows().len(), 2);
    }

    #[test]
    fn import_lpg_empty_dataset() {
        let db = GrafeoDB::new_in_memory();
        let input: LpgImport = serde_json::from_value(json!({ "nodes": [] })).unwrap();
        assert!(input.nodes.is_empty());
        assert!(input.edges.is_empty());
        assert_eq!(db.node_count(), 0);
    }

    #[test]
    fn import_lpg_nodes_without_properties() {
        let db = GrafeoDB::new_in_memory();
        let node_spec: LpgNodeSpec = serde_json::from_value(json!({ "labels": ["Tag"] })).unwrap();
        let labels: Vec<&str> = node_spec.labels.iter().map(String::as_str).collect();
        db.create_node(&labels).unwrap();
        assert_eq!(db.node_count(), 1);
    }

    #[test]
    fn import_lpg_self_loop() {
        let db = GrafeoDB::new_in_memory();
        let id = db.create_node(&["Node"]).unwrap();
        db.create_edge(id, id, "SELF_REF").unwrap();
        assert_eq!(db.edge_count(), 1);

        let session = db.session();
        let result = session
            .execute("MATCH (n)-[e:SELF_REF]->(n) RETURN n")
            .unwrap();
        assert_eq!(result.rows().len(), 1);
    }

    #[test]
    fn import_lpg_multiple_edges_between_same_nodes() {
        let db = GrafeoDB::new_in_memory();
        let alix = db
            .create_node_with_props(
                &["Person"],
                vec![(PropertyKey::new("name"), Value::from("Alix"))],
            )
            .unwrap();
        let gus = db
            .create_node_with_props(
                &["Person"],
                vec![(PropertyKey::new("name"), Value::from("Gus"))],
            )
            .unwrap();
        db.create_edge(alix, gus, "KNOWS").unwrap();
        db.create_edge(alix, gus, "WORKS_WITH").unwrap();
        db.create_edge(gus, alix, "KNOWS").unwrap();

        assert_eq!(db.edge_count(), 3);
    }

    #[test]
    fn import_lpg_large_batch() {
        let db = GrafeoDB::new_in_memory();
        let mut nodes = Vec::new();
        for i in 0..500 {
            nodes.push(json!({
                "labels": ["Item"],
                "properties": { "index": i }
            }));
        }
        let input: LpgImport = serde_json::from_value(json!({ "nodes": nodes })).unwrap();

        for node in &input.nodes {
            let labels: Vec<&str> = node.labels.iter().map(String::as_str).collect();
            let props: Vec<(PropertyKey, Value)> = node
                .properties
                .as_ref()
                .map(|p| {
                    p.iter()
                        .map(|(k, v)| (PropertyKey::new(k.as_str()), json_to_value(v)))
                        .collect()
                })
                .unwrap_or_default();
            db.create_node_with_props(&labels, props).unwrap();
        }

        assert_eq!(db.node_count(), 500);
    }

    #[test]
    fn import_lpg_edge_with_properties() {
        let db = GrafeoDB::new_in_memory();
        let a = db.create_node(&["A"]).unwrap();
        let b = db.create_node(&["B"]).unwrap();
        db.create_edge_with_props(
            a,
            b,
            "REL",
            vec![
                (PropertyKey::new("weight"), Value::Float64(0.75)),
                (PropertyKey::new("label"), Value::from("strong")),
            ],
        )
        .unwrap();

        let session = db.session();
        let result = session
            .execute("MATCH ()-[e:REL]->() RETURN e.weight, e.label")
            .unwrap();
        assert_eq!(result.rows().len(), 1);
    }

    // === RDF deserialization tests ===

    #[cfg(feature = "rdf")]
    mod rdf_tests {
        use serde_json::json;

        use super::super::*;

        #[test]
        fn rdf_import_iri_objects() {
            let input = json!({
                "triples": [
                    {
                        "subject": "http://example.org/Alix",
                        "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                        "object": "http://example.org/Person"
                    }
                ]
            });
            let import: RdfImport = serde_json::from_value(input).unwrap();
            assert_eq!(import.triples.len(), 1);
            assert!(matches!(import.triples[0].object, RdfObjectSpec::Iri(_)));
        }

        #[test]
        fn rdf_import_plain_literal() {
            let input = json!({
                "triples": [{
                    "subject": "http://example.org/Alix",
                    "predicate": "http://example.org/name",
                    "object": { "value": "Alix" }
                }]
            });
            let import: RdfImport = serde_json::from_value(input).unwrap();
            match &import.triples[0].object {
                RdfObjectSpec::Literal {
                    value,
                    datatype,
                    language,
                } => {
                    assert_eq!(value, "Alix");
                    assert!(datatype.is_none());
                    assert!(language.is_none());
                }
                RdfObjectSpec::Iri(_) => panic!("expected literal"),
            }
        }

        #[test]
        fn rdf_import_typed_literal() {
            let input = json!({
                "triples": [{
                    "subject": "http://example.org/Alix",
                    "predicate": "http://example.org/age",
                    "object": {
                        "value": "30",
                        "datatype": "http://www.w3.org/2001/XMLSchema#integer"
                    }
                }]
            });
            let import: RdfImport = serde_json::from_value(input).unwrap();
            match &import.triples[0].object {
                RdfObjectSpec::Literal {
                    value, datatype, ..
                } => {
                    assert_eq!(value, "30");
                    assert_eq!(
                        datatype.as_deref(),
                        Some("http://www.w3.org/2001/XMLSchema#integer")
                    );
                }
                RdfObjectSpec::Iri(_) => panic!("expected typed literal"),
            }
        }

        #[test]
        fn rdf_import_language_literal() {
            let input = json!({
                "triples": [{
                    "subject": "http://example.org/Alix",
                    "predicate": "http://example.org/greeting",
                    "object": { "value": "hallo", "language": "nl" }
                }]
            });
            let import: RdfImport = serde_json::from_value(input).unwrap();
            match &import.triples[0].object {
                RdfObjectSpec::Literal { language, .. } => {
                    assert_eq!(language.as_deref(), Some("nl"));
                }
                RdfObjectSpec::Iri(_) => panic!("expected lang literal"),
            }
        }

        #[test]
        fn rdf_import_blank_node_subject() {
            let input = json!({
                "triples": [{
                    "subject": "_:b1",
                    "predicate": "http://example.org/name",
                    "object": { "value": "anonymous" }
                }]
            });
            let import: RdfImport = serde_json::from_value(input).unwrap();
            assert_eq!(import.triples[0].subject, "_:b1");
        }

        #[test]
        fn rdf_import_empty_triples() {
            let input = json!({ "triples": [] });
            let import: RdfImport = serde_json::from_value(input).unwrap();
            assert!(import.triples.is_empty());
        }

        #[test]
        fn rdf_import_missing_triples_field_errors() {
            let input = json!({});
            let result: Result<RdfImport, _> = serde_json::from_value(input);
            assert!(result.is_err());
        }

        #[test]
        fn string_to_rdf_term_iri() {
            let term = string_to_rdf_term("http://example.org/Alix");
            assert!(term.is_iri());
        }

        #[test]
        fn string_to_rdf_term_blank_node() {
            let term = string_to_rdf_term("_:b42");
            assert!(term.is_blank_node());
        }

        // === Engine-level RDF batch tests ===

        #[test]
        fn batch_insert_rdf_basic() {
            use grafeo_core::graph::rdf::{Term, Triple};

            let db = GrafeoDB::new_in_memory();
            let triples = vec![
                Triple::new(
                    Term::iri("http://example.org/Alix"),
                    Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
                    Term::iri("http://example.org/Person"),
                ),
                Triple::new(
                    Term::iri("http://example.org/Alix"),
                    Term::iri("http://example.org/name"),
                    Term::literal("Alix"),
                ),
            ];

            let inserted = db.batch_insert_rdf(triples);
            assert_eq!(inserted, 2);
        }

        #[test]
        fn batch_insert_rdf_deduplicates() {
            use grafeo_core::graph::rdf::{Term, Triple};

            let db = GrafeoDB::new_in_memory();
            let triple = Triple::new(
                Term::iri("http://example.org/Alix"),
                Term::iri("http://example.org/name"),
                Term::literal("Alix"),
            );

            let first = db.batch_insert_rdf(vec![triple.clone()]);
            assert_eq!(first, 1);

            let second = db.batch_insert_rdf(vec![triple]);
            assert_eq!(second, 0, "duplicate triple should be skipped");
        }

        #[test]
        fn batch_insert_rdf_empty() {
            let db = GrafeoDB::new_in_memory();
            let inserted = db.batch_insert_rdf(Vec::new());
            assert_eq!(inserted, 0);
        }

        #[test]
        fn batch_insert_rdf_blank_nodes() {
            use grafeo_core::graph::rdf::{Term, Triple};

            let db = GrafeoDB::new_in_memory();
            let triples = vec![
                Triple::new(
                    Term::blank("b1"),
                    Term::iri("http://example.org/name"),
                    Term::literal("Anonymous"),
                ),
                Triple::new(
                    Term::blank("b1"),
                    Term::iri("http://example.org/knows"),
                    Term::blank("b2"),
                ),
            ];

            let inserted = db.batch_insert_rdf(triples);
            assert_eq!(inserted, 2);
        }

        #[test]
        fn batch_insert_rdf_typed_and_lang_literals() {
            use grafeo_core::graph::rdf::{Term, Triple};

            let db = GrafeoDB::new_in_memory();
            let triples = vec![
                Triple::new(
                    Term::iri("http://example.org/Alix"),
                    Term::iri("http://example.org/age"),
                    Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer"),
                ),
                Triple::new(
                    Term::iri("http://example.org/Alix"),
                    Term::iri("http://example.org/greeting"),
                    Term::lang_literal("hallo", "nl"),
                ),
            ];

            let inserted = db.batch_insert_rdf(triples);
            assert_eq!(inserted, 2);
        }

        #[test]
        fn batch_insert_rdf_large_batch() {
            use grafeo_core::graph::rdf::{Term, Triple};

            let db = GrafeoDB::new_in_memory();
            let triples: Vec<Triple> = (0..1000)
                .map(|i| {
                    Triple::new(
                        Term::iri(format!("http://example.org/node/{i}")),
                        Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
                        Term::iri("http://example.org/Item"),
                    )
                })
                .collect();

            let inserted = db.batch_insert_rdf(triples);
            assert_eq!(inserted, 1000);
        }

        #[test]
        fn batch_insert_rdf_mixed_duplicates_in_same_batch() {
            use grafeo_core::graph::rdf::{Term, Triple};

            let db = GrafeoDB::new_in_memory();
            let triple = Triple::new(
                Term::iri("http://example.org/a"),
                Term::iri("http://example.org/b"),
                Term::iri("http://example.org/c"),
            );

            // Same triple 3 times in one batch
            let inserted = db.batch_insert_rdf(vec![triple.clone(), triple.clone(), triple]);
            assert_eq!(
                inserted, 1,
                "duplicates within same batch should be deduped"
            );
        }
    }
}
