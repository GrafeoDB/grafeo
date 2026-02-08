//! WebAssembly bindings for Grafeo graph database.
//!
//! Use Grafeo from JavaScript in the browser, Deno, or Cloudflare Workers.
//!
//! ```js
//! import init, { Database } from '@grafeo-db/wasm';
//!
//! await init();
//! const db = new Database();
//! db.execute("INSERT (:Person {name: 'Alice', age: 30})");
//! const result = db.execute("MATCH (p:Person) RETURN p.name, p.age");
//! console.log(result); // [{name: "Alice", age: 30}]
//! ```

mod types;
mod utils;

use js_sys::Array;
use wasm_bindgen::prelude::*;

use grafeo_engine::GrafeoDB;

/// A Grafeo graph database instance running in WebAssembly.
///
/// All data is held in memory within the WASM heap. For persistence,
/// use `exportSnapshot()` / `importSnapshot()` with IndexedDB or
/// the higher-level `@grafeo-db/web` package.
#[wasm_bindgen]
pub struct Database {
    inner: GrafeoDB,
}

#[wasm_bindgen]
impl Database {
    /// Creates a new in-memory database.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<Database, JsError> {
        utils::set_panic_hook();
        Ok(Database {
            inner: GrafeoDB::new_in_memory(),
        })
    }

    /// Executes a GQL query and returns results as an array of objects.
    ///
    /// Each row becomes a JavaScript object with column names as keys.
    ///
    /// ```js
    /// const results = db.execute("MATCH (p:Person) RETURN p.name, p.age");
    /// // [{name: "Alice", age: 30}, {name: "Bob", age: 25}]
    /// ```
    pub fn execute(&self, query: &str) -> Result<JsValue, JsError> {
        let result = self
            .inner
            .execute(query)
            .map_err(|e| JsError::new(&e.to_string()))?;

        let rows = Array::new_with_length(result.rows.len() as u32);
        for (i, row) in result.rows.iter().enumerate() {
            rows.set(i as u32, types::row_to_js_object(&result.columns, row));
        }
        Ok(rows.into())
    }

    /// Executes a GQL query and returns raw columns, rows, and metadata.
    ///
    /// Returns `{ columns: string[], rows: any[][], executionTimeMs?: number }`.
    #[wasm_bindgen(js_name = "executeRaw")]
    pub fn execute_raw(&self, query: &str) -> Result<JsValue, JsError> {
        let result = self
            .inner
            .execute(query)
            .map_err(|e| JsError::new(&e.to_string()))?;

        let obj = js_sys::Object::new();

        // columns: string[]
        let cols = Array::new_with_length(result.columns.len() as u32);
        for (i, col) in result.columns.iter().enumerate() {
            cols.set(i as u32, JsValue::from_str(col));
        }
        let _ = js_sys::Reflect::set(&obj, &JsValue::from_str("columns"), &cols);

        // rows: any[][]
        let rows = Array::new_with_length(result.rows.len() as u32);
        for (i, row) in result.rows.iter().enumerate() {
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

    /// Returns the Grafeo version.
    pub fn version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
}
