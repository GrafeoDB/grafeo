//! Physical operator for CALL procedure execution.
//!
//! Wraps a [`Procedure`] and produces [`DataChunk`]s from its result, with
//! optional YIELD column filtering and aliasing. The `Procedure` trait
//! unifies graph algorithms, catalog introspection, and vector/text search
//! behind a single dispatch path (see [`crate::procedures`]).

use std::sync::Arc;

use grafeo_adapters::plugins::{AlgorithmResult, Parameters};
use grafeo_common::types::{LogicalType, Value};
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::operators::{Operator, OperatorError, OperatorResult};
use grafeo_core::graph::GraphStoreSearch;

use crate::procedures::{Procedure, ProcedureContext};

/// Physical operator that executes a procedure and yields its results.
///
/// On the first call to [`next()`](Operator::next), the procedure runs and
/// the full result is cached. Subsequent calls yield rows in chunks of
/// `CHUNK_SIZE` until exhausted.
pub struct ProcedureCallOperator {
    store: Arc<dyn GraphStoreSearch>,
    procedure: Arc<dyn Procedure>,
    params: Parameters,
    /// Optional LPG store handle, required by search procedures that reach
    /// the HNSW / BM25 indexes rather than the raw graph store.
    #[cfg(feature = "lpg")]
    lpg_store: Option<Arc<grafeo_core::graph::lpg::LpgStore>>,
    /// YIELD items: (original_column, alias). `None` means yield all columns.
    yield_columns: Option<Vec<(String, Option<String>)>>,
    /// Canonical column names from the procedure registry (e.g., `["node_id", "score"]`
    /// for PageRank, even though the algorithm internally names it `"pagerank"`).
    /// Used to remap procedure result columns for YIELD matching.
    canonical_columns: Vec<String>,
    /// Cached procedure result (populated on first next()).
    result: Option<AlgorithmResult>,
    /// Current row position in the cached result.
    row_index: usize,
    /// Output column names (resolved after first next()).
    output_columns: Vec<String>,
    /// Column indices to extract from each result row (resolved after YIELD filtering).
    column_indices: Vec<usize>,
}

/// Number of rows per DataChunk.
const CHUNK_SIZE: usize = 1024;

impl ProcedureCallOperator {
    /// Creates a new procedure call operator.
    pub fn new(
        store: Arc<dyn GraphStoreSearch>,
        procedure: Arc<dyn Procedure>,
        params: Parameters,
        yield_columns: Option<Vec<(String, Option<String>)>>,
        canonical_columns: Vec<String>,
    ) -> Self {
        Self {
            store,
            procedure,
            params,
            #[cfg(feature = "lpg")]
            lpg_store: None,
            yield_columns,
            canonical_columns,
            result: None,
            row_index: 0,
            output_columns: Vec::new(),
            column_indices: Vec::new(),
        }
    }

    /// Attaches an LPG store handle so search procedures can reach vector /
    /// text indexes.
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn with_lpg_store(mut self, lpg_store: Arc<grafeo_core::graph::lpg::LpgStore>) -> Self {
        self.lpg_store = Some(lpg_store);
        self
    }

    /// Executes the procedure and resolves YIELD column mapping.
    fn execute_algorithm(&mut self) -> Result<(), OperatorError> {
        #[cfg(feature = "lpg")]
        let ctx = match self.lpg_store.as_deref() {
            Some(lpg) => ProcedureContext::with_lpg_store(&*self.store, lpg),
            None => ProcedureContext::new(&*self.store),
        };
        #[cfg(not(feature = "lpg"))]
        let ctx = ProcedureContext::new(&*self.store);
        let result = self
            .procedure
            .execute(&ctx, &self.params)
            // The procedure's error keeps its code: a mistake in the
            // arguments stays one (a query vector the index cannot measure,
            // #593), with the procedure's own message.
            .map_err(OperatorError::from)?;

        // Use canonical column names if available (same length as result columns),
        // otherwise fall back to the algorithm's own column names.
        let display_columns = if self.canonical_columns.len() == result.columns.len() {
            &self.canonical_columns
        } else {
            &result.columns
        };

        // Resolve YIELD columns → indices (matching against canonical names)
        if let Some(ref yield_cols) = self.yield_columns {
            for (field_name, alias) in yield_cols {
                let idx = display_columns
                    .iter()
                    .position(|c| c == field_name)
                    .ok_or_else(|| {
                        // A mistake in the query text: the procedure has no
                        // such column.
                        OperatorError::from(grafeo_common::utils::error::Error::Query(
                            grafeo_common::utils::error::QueryError::semantic(format!(
                                "YIELD column '{}' is no column of the procedure (it has: {})",
                                field_name,
                                display_columns.join(", ")
                            )),
                        ))
                    })?;
                self.column_indices.push(idx);
                self.output_columns
                    .push(alias.clone().unwrap_or_else(|| field_name.clone()));
            }
        } else {
            // No YIELD: return all columns with canonical names
            self.column_indices = (0..result.columns.len()).collect();
            self.output_columns = display_columns.clone();
        }

        self.result = Some(result);
        Ok(())
    }

    /// Returns the output column names (available after first next() call).
    pub fn output_columns(&self) -> &[String] {
        &self.output_columns
    }
}

impl Operator for ProcedureCallOperator {
    fn next(&mut self) -> OperatorResult {
        // Lazy execution: run algorithm on first call
        if self.result.is_none() {
            self.execute_algorithm()?;
        }

        let result = self
            .result
            .as_ref()
            .expect("result populated by execute_algorithm");

        if self.row_index >= result.rows.len() {
            return Ok(None);
        }

        let remaining = result.rows.len() - self.row_index;
        let chunk_rows = remaining.min(CHUNK_SIZE);

        // Build column types from first row
        let col_types: Vec<LogicalType> = if !result.rows.is_empty() {
            self.column_indices
                .iter()
                .map(|&idx| value_to_logical_type(&result.rows[0][idx]))
                .collect()
        } else {
            vec![LogicalType::Any; self.column_indices.len()]
        };

        let mut chunk = DataChunk::with_capacity(&col_types, chunk_rows);

        for row_offset in 0..chunk_rows {
            let row = &result.rows[self.row_index + row_offset];
            for (col_idx, &src_idx) in self.column_indices.iter().enumerate() {
                let value = row.get(src_idx).cloned().unwrap_or(Value::Null);
                if let Some(col) = chunk.column_mut(col_idx) {
                    col.push_value(value);
                }
            }
        }
        chunk.set_count(chunk_rows);

        self.row_index += chunk_rows;
        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.row_index = 0;
        // Keep the cached result for re-iteration
    }

    fn name(&self) -> &'static str {
        "ProcedureCall"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Maps a `Value` to its `LogicalType`.
fn value_to_logical_type(value: &Value) -> LogicalType {
    match value {
        Value::Null => LogicalType::Any,
        Value::Bool(_) => LogicalType::Bool,
        Value::Int64(_) => LogicalType::Int64,
        Value::Float64(_) => LogicalType::Float64,
        Value::String(_) => LogicalType::String,
        _ => LogicalType::Any,
    }
}
