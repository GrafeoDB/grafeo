//! Query results and builders for the Python API.

use std::collections::HashMap;
use std::fmt::Write as _;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use grafeo_common::types::Value;

use crate::graph::{PyEdge, PyNode};
use crate::types::PyValue;

/// Results from a GQL query - iterate rows or access nodes and edges directly.
///
/// Iterate with `for row in result:` where each row is a dict. Or use
/// `result.nodes()` and `result.edges()` to get graph elements. For single
/// values, `result.scalar()` grabs the first column of the first row.
///
/// Query performance metrics are available via `execution_time_ms` and
/// `rows_scanned` properties when timing is enabled.
#[pyclass(name = "QueryResult")]
pub struct PyQueryResult {
    pub(crate) columns: Vec<String>,
    pub(crate) rows: Vec<Vec<Value>>,
    pub(crate) nodes: Vec<PyNode>,
    pub(crate) edges: Vec<PyEdge>,
    current_row: usize,
    /// Query execution time in milliseconds.
    pub(crate) execution_time_ms: Option<f64>,
    /// Number of rows scanned during execution.
    pub(crate) rows_scanned: Option<u64>,
    /// What the query's writes changed.
    pub(crate) counters: grafeo_engine::database::WriteCounters,
}

#[pymethods]
impl PyQueryResult {
    /// Get column names.
    #[getter]
    fn columns(&self) -> Vec<String> {
        self.columns.clone()
    }

    /// Get number of rows.
    fn __len__(&self) -> usize {
        self.rows.len()
    }

    /// Get a row by index.
    fn __getitem__(&self, idx: isize, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let idx = if idx < 0 {
            // reason: Python negative indexing: len + negative_idx yields valid positive index,
            // reason: out-of-range values are caught by the bounds check below
            #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
            let resolved = (self.rows.len() as isize + idx) as usize;
            resolved
        } else {
            // reason: non-negative isize fits usize
            #[allow(clippy::cast_sign_loss)]
            let resolved = idx as usize;
            resolved
        };

        if idx >= self.rows.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "Row index out of range",
            ));
        }

        let row = &self.rows[idx];
        let dict = pyo3::types::PyDict::new(py);
        for (col, val) in self.columns.iter().zip(row.iter()) {
            dict.set_item(col, PyValue::to_py(val, py))?;
        }
        Ok(dict.unbind().into_any())
    }

    /// Iterate over rows.
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Get next row.
    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> Option<Py<PyAny>> {
        if slf.current_row >= slf.rows.len() {
            return None;
        }

        let idx = slf.current_row;
        slf.current_row += 1;

        let row = slf.rows[idx].clone();
        let columns = slf.columns.clone();

        let dict = pyo3::types::PyDict::new(py);
        for (col, val) in columns.iter().zip(row.iter()) {
            dict.set_item(col, PyValue::to_py(val, py)).ok()?;
        }
        Some(dict.unbind().into_any())
    }

    /// Get all nodes from the result.
    fn nodes(&self) -> Vec<PyNode> {
        self.nodes.clone()
    }

    /// Get all edges from the result.
    fn edges(&self) -> Vec<PyEdge> {
        self.edges.clone()
    }

    /// Convert to list of dictionaries.
    ///
    /// # Panics
    ///
    /// Panics on memory exhaustion during Python list/dict allocation.
    fn to_list(&self, py: Python<'_>) -> Py<PyAny> {
        let list = pyo3::types::PyList::empty(py);
        for row in &self.rows {
            let dict = pyo3::types::PyDict::new(py);
            for (col, val) in self.columns.iter().zip(row.iter()) {
                dict.set_item(col, PyValue::to_py(val, py))
                    .expect("dict.set_item only fails on memory exhaustion");
            }
            list.append(dict)
                .expect("list.append only fails on memory exhaustion");
        }
        list.unbind().into_any()
    }

    /// Get single value (first column of first row).
    fn scalar(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.rows.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("No rows in result"));
        }
        if self.columns.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "No columns in result",
            ));
        }
        Ok(PyValue::to_py(&self.rows[0][0], py))
    }

    /// Query execution time in milliseconds (if available).
    ///
    /// Example:
    /// ```python
    /// result = db.execute("MATCH (n:Person) RETURN n")
    /// if result.execution_time_ms:
    ///     print(f"Query took {result.execution_time_ms:.2f}ms")
    /// ```
    #[getter]
    fn execution_time_ms(&self) -> Option<f64> {
        self.execution_time_ms
    }

    /// Number of rows scanned during query execution (if available).
    ///
    /// Example:
    /// ```python
    /// result = db.execute("MATCH (n:Person) RETURN n")
    /// if result.rows_scanned:
    ///     print(f"Scanned {result.rows_scanned} rows")
    /// ```
    #[getter]
    fn rows_scanned(&self) -> Option<u64> {
        self.rows_scanned
    }

    /// What the query's writes changed, as a dict: `nodes_created`,
    /// `nodes_deleted`, `edges_created`, `edges_deleted`, `properties_set`,
    /// `labels_added` and `labels_removed`.
    ///
    /// Example:
    /// ```python
    /// result = db.execute("INSERT (:Person {name: 'Alix'})")
    /// result.counters["nodes_created"]  # 1
    /// ```
    #[getter]
    fn counters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let c = &self.counters;
        let dict = PyDict::new(py);
        for (name, count) in [
            ("nodes_created", c.nodes_created),
            ("nodes_deleted", c.nodes_deleted),
            ("edges_created", c.edges_created),
            ("edges_deleted", c.edges_deleted),
            ("properties_set", c.properties_set),
            ("labels_added", c.labels_added),
            ("labels_removed", c.labels_removed),
        ] {
            dict.set_item(name, count)?;
        }
        Ok(dict)
    }

    /// Convert to a pandas DataFrame.
    ///
    /// Requires pandas to be installed (`uv add pandas`). Each column in the
    /// query result becomes a DataFrame column, preserving types where possible.
    ///
    /// Example:
    /// ```python
    /// result = db.execute("MATCH (n:Person) RETURN n.name, n.age")
    /// df = result.to_pandas()
    /// print(df.head())
    /// ```
    #[pyo3(signature = ())]
    fn to_pandas(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let pd = py.import("pandas").map_err(|_| {
            pyo3::exceptions::PyModuleNotFoundError::new_err(
                "pandas is required for to_pandas(). Install it with: uv add pandas",
            )
        })?;

        // Build column-oriented data: dict of {col_name: [values...]}
        let data = pyo3::types::PyDict::new(py);
        for (col_idx, col_name) in self.columns.iter().enumerate() {
            let values = pyo3::types::PyList::empty(py);
            for row in &self.rows {
                let val = row
                    .get(col_idx)
                    .map_or_else(|| py.None(), |v| PyValue::to_py(v, py));
                values.append(val)?;
            }
            data.set_item(col_name, values)?;
        }

        let df = pd.call_method1("DataFrame", (data,))?;
        Ok(df.unbind())
    }

    /// Convert to a polars DataFrame.
    ///
    /// Requires polars to be installed (`uv add polars`). Each column in the
    /// query result becomes a DataFrame column. Values are converted to native
    /// Python types first, then polars infers the best dtype.
    ///
    /// Example:
    /// ```python
    /// result = db.execute("MATCH (n:Person) RETURN n.name, n.age")
    /// df = result.to_polars()
    /// print(df.head())
    /// ```
    #[pyo3(signature = ())]
    fn to_polars(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let pl = py.import("polars").map_err(|_| {
            pyo3::exceptions::PyModuleNotFoundError::new_err(
                "polars is required for to_polars(). Install it with: uv add polars",
            )
        })?;

        // Build column-oriented data: dict of {col_name: [values...]}
        let data = pyo3::types::PyDict::new(py);
        for (col_idx, col_name) in self.columns.iter().enumerate() {
            let values = pyo3::types::PyList::empty(py);
            for row in &self.rows {
                let val = row
                    .get(col_idx)
                    .map_or_else(|| py.None(), |v| PyValue::to_py(v, py));
                values.append(val)?;
            }
            data.set_item(col_name, values)?;
        }

        let df = pl.call_method1("DataFrame", (data,))?;
        Ok(df.unbind())
    }

    /// Convert to Arrow IPC bytes.
    ///
    /// Returns the query result as Arrow IPC stream format bytes. These can be
    /// read by any Arrow implementation:
    ///
    /// - `pyarrow.ipc.open_stream(buf).read_all()` for a PyArrow Table
    /// - `polars.read_ipc(buf)` for a Polars DataFrame
    ///
    /// Example:
    /// ```python
    /// ipc_bytes = result.to_arrow_ipc()
    /// import pyarrow as pa
    /// table = pa.ipc.open_stream(ipc_bytes).read_all()
    /// ```
    #[cfg(feature = "arrow-export")]
    #[pyo3(signature = ())]
    fn to_arrow_ipc(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let ipc_bytes = self.to_ipc_bytes()?;
        Ok(pyo3::types::PyBytes::new(py, &ipc_bytes).into())
    }

    /// Convert to a PyArrow Table.
    ///
    /// Requires pyarrow to be installed (`uv add pyarrow`). Returns an Arrow
    /// Table that can be used directly with DuckDB, Polars, pandas, or any
    /// other Arrow-compatible tool.
    ///
    /// Example:
    /// ```python
    /// table = result.to_arrow()
    /// # Convert to pandas: table.to_pandas()
    /// # Convert to polars: polars.from_arrow(table)
    /// # Use with DuckDB: duckdb.from_arrow(table)
    /// ```
    #[cfg(feature = "arrow-export")]
    #[pyo3(signature = ())]
    fn to_arrow(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let pa = py.import("pyarrow").map_err(|_| {
            pyo3::exceptions::PyModuleNotFoundError::new_err(
                "pyarrow is required for to_arrow(). Install it with: uv add pyarrow",
            )
        })?;
        let ipc_mod = pa.getattr("ipc")?;

        let ipc_bytes = self.to_ipc_bytes()?;
        let py_bytes = pyo3::types::PyBytes::new(py, &ipc_bytes);
        let reader = ipc_mod.call_method1("open_stream", (py_bytes,))?;
        let table = reader.call_method0("read_all")?;
        Ok(table.unbind())
    }

    /// Serialize CONSTRUCT-style results as N-Triples text.
    ///
    /// The result must have columns `["subject", "predicate", "object"]` (the
    /// standard shape returned by SPARQL CONSTRUCT queries). Each row is
    /// formatted as `<subject> <predicate> <object> .\n`.
    ///
    /// Returns a Python string. Raises `ValueError` if the columns do not
    /// match the expected triple pattern.
    ///
    /// Example:
    /// ```python
    /// result = db.execute_sparql("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }")
    /// print(result.to_ntriples())
    /// ```
    fn to_ntriples(&self) -> PyResult<String> {
        self.validate_triple_columns()?;
        let (si, pi, oi) = self.triple_column_indices();
        let mut output = String::new();
        for row in &self.rows {
            let subj = Self::value_to_ntriples_term(&row[si]);
            let pred = Self::value_to_ntriples_term(&row[pi]);
            let obj = Self::value_to_ntriples_term(&row[oi]);
            let _ = writeln!(output, "{subj} {pred} {obj} .");
        }
        Ok(output)
    }

    /// Serialize CONSTRUCT-style results as Turtle text.
    ///
    /// Groups triples by subject and uses `;` to separate predicate-object
    /// pairs sharing the same subject. This is a convenience formatter, not a
    /// full Turtle serializer (no prefix declarations are emitted).
    ///
    /// Returns a Python string. Raises `ValueError` if the columns do not
    /// match the expected triple pattern.
    ///
    /// Example:
    /// ```python
    /// result = db.execute_sparql("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }")
    /// print(result.to_turtle())
    /// ```
    fn to_turtle(&self) -> PyResult<String> {
        self.validate_triple_columns()?;
        let (si, pi, oi) = self.triple_column_indices();

        // Group by subject, preserving insertion order via Vec of (subject, predicates).
        let mut subjects: Vec<(String, Vec<(String, String)>)> = Vec::new();
        let mut subject_index: HashMap<String, usize> = HashMap::new();

        for row in &self.rows {
            let subj = Self::value_to_ntriples_term(&row[si]);
            let pred = Self::value_to_ntriples_term(&row[pi]);
            let obj = Self::value_to_ntriples_term(&row[oi]);

            if let Some(&idx) = subject_index.get(&subj) {
                subjects[idx].1.push((pred, obj));
            } else {
                let idx = subjects.len();
                subject_index.insert(subj.clone(), idx);
                subjects.push((subj, vec![(pred, obj)]));
            }
        }

        let mut output = String::new();
        for (subj, pairs) in &subjects {
            output.push_str(subj);
            for (i, (pred, obj)) in pairs.iter().enumerate() {
                if i == 0 {
                    let _ = write!(output, " {pred} {obj}");
                } else {
                    let _ = write!(output, " ;\n    {pred} {obj}");
                }
            }
            output.push_str(" .\n\n");
        }
        Ok(output)
    }

    fn __repr__(&self) -> String {
        let time_str = self
            .execution_time_ms
            .map(|t| format!(", time={:.2}ms", t))
            .unwrap_or_default();
        format!(
            "QueryResult(columns={:?}, rows={}{})",
            self.columns,
            self.rows.len(),
            time_str
        )
    }

    fn __str__(&self) -> String {
        grafeo_common::fmt::format_result_table(
            &self.columns,
            &self.rows,
            self.execution_time_ms,
            None,
        )
    }
}

impl PyQueryResult {
    /// Creates a new query result (used internally).
    pub fn new(
        columns: Vec<String>,
        rows: Vec<Vec<Value>>,
        nodes: Vec<PyNode>,
        edges: Vec<PyEdge>,
    ) -> Self {
        Self {
            columns,
            rows,
            nodes,
            edges,
            current_row: 0,
            execution_time_ms: None,
            rows_scanned: None,
            counters: grafeo_engine::database::WriteCounters::default(),
        }
    }

    /// Creates a new query result with execution metrics (used internally).
    pub fn with_metrics(
        columns: Vec<String>,
        rows: Vec<Vec<Value>>,
        nodes: Vec<PyNode>,
        edges: Vec<PyEdge>,
        execution_time_ms: Option<f64>,
        rows_scanned: Option<u64>,
    ) -> Self {
        Self {
            columns,
            rows,
            nodes,
            edges,
            current_row: 0,
            execution_time_ms,
            rows_scanned,
            counters: grafeo_engine::database::WriteCounters::default(),
        }
    }

    /// Serializes the query result to Arrow IPC stream bytes.
    #[cfg(feature = "arrow-export")]
    fn to_ipc_bytes(&self) -> PyResult<Vec<u8>> {
        let col_types = vec![grafeo_common::LogicalType::Any; self.columns.len()];
        let batch = grafeo_engine::database::arrow::query_result_to_record_batch(
            &self.columns,
            &col_types,
            &self.rows,
        )
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Arrow export failed: {e}"))
        })?;
        grafeo_engine::database::arrow::record_batch_to_ipc_stream(&batch).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Arrow IPC failed: {e}"))
        })
    }

    /// Creates an empty result (used internally).
    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            rows: Vec::new(),
            nodes: Vec::new(),
            edges: Vec::new(),
            current_row: 0,
            execution_time_ms: None,
            rows_scanned: None,
            counters: grafeo_engine::database::WriteCounters::default(),
        }
    }

    /// Validates that this result has triple-shaped columns (subject, predicate, object).
    fn validate_triple_columns(&self) -> PyResult<()> {
        let normalized: Vec<String> = self.columns.iter().map(|c| c.to_lowercase()).collect();
        let has_s = normalized.iter().any(|c| c == "subject" || c == "s");
        let has_p = normalized.iter().any(|c| c == "predicate" || c == "p");
        let has_o = normalized.iter().any(|c| c == "object" || c == "o");

        if has_s && has_p && has_o {
            Ok(())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(format!(
                "to_ntriples()/to_turtle() requires columns named \
                 [subject, predicate, object] (or [s, p, o]). \
                 Got: {:?}",
                self.columns
            )))
        }
    }

    /// Returns the column indices for (subject, predicate, object).
    /// Must be called after `validate_triple_columns`.
    fn triple_column_indices(&self) -> (usize, usize, usize) {
        let normalized: Vec<String> = self.columns.iter().map(|c| c.to_lowercase()).collect();
        let si = normalized
            .iter()
            .position(|c| c == "subject" || c == "s")
            .unwrap_or(0);
        let pi = normalized
            .iter()
            .position(|c| c == "predicate" || c == "p")
            .unwrap_or(1);
        let oi = normalized
            .iter()
            .position(|c| c == "object" || c == "o")
            .unwrap_or(2);
        (si, pi, oi)
    }

    /// Formats a `Value` as an N-Triples term.
    ///
    /// IRIs are wrapped in angle brackets, strings become quoted literals,
    /// and blank nodes are prefixed with `_:`.
    fn value_to_ntriples_term(val: &Value) -> String {
        match val {
            Value::String(s) => {
                let s_str: &str = s.as_ref();
                if s_str.starts_with("http://")
                    || s_str.starts_with("https://")
                    || s_str.starts_with("urn:")
                {
                    // Looks like an IRI
                    format!("<{s_str}>")
                } else if s_str.starts_with("_:") {
                    // Blank node
                    s_str.to_string()
                } else {
                    // Plain literal: escape special characters
                    let escaped = s_str
                        .replace('\\', "\\\\")
                        .replace('"', "\\\"")
                        .replace('\n', "\\n")
                        .replace('\r', "\\r")
                        .replace('\t', "\\t");
                    format!("\"{escaped}\"")
                }
            }
            Value::Int64(n) => format!("\"{n}\"^^<http://www.w3.org/2001/XMLSchema#integer>"),
            Value::Float64(f) => format!("\"{f}\"^^<http://www.w3.org/2001/XMLSchema#double>"),
            Value::Bool(b) => format!("\"{b}\"^^<http://www.w3.org/2001/XMLSchema#boolean>"),
            Value::Null => "\"\"".to_string(),
            other => {
                // Fallback: quote the display representation.
                format!("\"{}\"", other)
            }
        }
    }
}

/// Builds parameterized queries with a fluent API.
///
/// Add parameters with `.param("name", value)` to safely inject values
/// without string concatenation (prevents injection).
#[pyclass(name = "QueryBuilder")]
pub struct PyQueryBuilder {
    pub(crate) query: String,
    pub(crate) params: HashMap<String, Value>,
}

impl PyQueryBuilder {
    /// Creates a new query builder (Rust API).
    pub fn create(query: String) -> Self {
        Self {
            query,
            params: HashMap::new(),
        }
    }
}

#[pymethods]
impl PyQueryBuilder {
    /// Create a new query builder.
    #[new]
    fn new(query: String) -> Self {
        Self::create(query)
    }

    /// Set a parameter.
    ///
    /// # Errors
    ///
    /// Raises `ValueError` if the value cannot be converted to a Grafeo type.
    fn param(&mut self, name: String, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let v = PyValue::from_py(value).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Cannot convert parameter '{}' to a Grafeo value: {}",
                name, e
            ))
        })?;
        self.params.insert(name, v);
        Ok(())
    }

    /// Get the query string.
    #[getter]
    fn query(&self) -> &str {
        &self.query
    }
}
