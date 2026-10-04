//! Operator for executing user-defined stored procedures.
//!
//! Re-parses the stored GQL body, substitutes parameters, and executes
//! as a sub-query using a fresh planner/executor pipeline.

use std::collections::HashMap;
use std::sync::Arc;

use grafeo_common::types::{EpochId, TransactionId, Value};
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::operators::{Operator, OperatorError, OperatorResult, WriteCounter};
use grafeo_core::graph::{GraphStoreMut, GraphStoreSearch};

use crate::catalog::Catalog;
use crate::database::QueryResult;
use crate::query::planner::Planner;
use crate::transaction::TransactionManager;

/// Execution context shared across sub-query operators.
pub struct ProcedureContext {
    /// Store for sub-query execution (read path).
    pub store: Arc<dyn GraphStoreSearch>,
    /// Writable store for sub-query mutations (None for read-only).
    pub store_mut: Option<Arc<dyn GraphStoreMut>>,
    /// Transaction manager for sub-query coordination.
    pub transaction_manager: Option<Arc<TransactionManager>>,
    /// Current transaction ID.
    pub transaction_id: Option<TransactionId>,
    /// Viewing epoch for MVCC reads.
    pub viewing_epoch: EpochId,
    /// Catalog for sub-planner resolution.
    pub catalog: Option<Arc<Catalog>>,
    /// The calling statement's write counter: the body's writes count there.
    pub write_counter: Arc<WriteCounter>,
}

/// An operator that executes a user-defined stored procedure.
///
/// On first call to `next()`, it:
/// 1. Parses the stored GQL body
/// 2. Substitutes parameter values
/// 3. Executes the query via a sub-planner
/// 4. Buffers the results
/// 5. Returns results in chunks
pub struct UserProcedureOperator {
    /// Raw GQL body of the procedure.
    body: String,
    /// Parameter name to value mapping.
    params: HashMap<String, Value>,
    /// Return column names (from procedure definition).
    return_columns: Vec<String>,
    /// YIELD column filter (if specified by caller).
    yield_columns: Option<Vec<String>>,
    /// Store for sub-query execution (read path).
    store: Arc<dyn GraphStoreSearch>,
    /// Writable store for sub-query mutations (None for read-only).
    store_mut: Option<Arc<dyn GraphStoreMut>>,
    /// Transaction manager for sub-query.
    transaction_manager: Option<Arc<TransactionManager>>,
    /// Current transaction ID.
    transaction_id: Option<TransactionId>,
    /// Viewing epoch.
    viewing_epoch: EpochId,
    /// Catalog for sub-planner.
    catalog: Option<Arc<Catalog>>,
    /// The calling statement's write counter.
    write_counter: Arc<WriteCounter>,
    /// Buffered result rows from execution.
    result_rows: Option<Vec<Vec<Value>>>,
    /// Current row index into buffered results.
    row_index: usize,
    /// Output column names after YIELD filtering.
    output_columns: Vec<String>,
}

impl UserProcedureOperator {
    /// Creates a new user procedure operator.
    pub fn new(
        body: String,
        params: HashMap<String, Value>,
        return_columns: Vec<String>,
        yield_columns: Option<Vec<String>>,
        ctx: ProcedureContext,
    ) -> Self {
        let output_columns = if let Some(ref yields) = yield_columns {
            yields.clone()
        } else {
            return_columns.clone()
        };
        Self {
            body,
            params,
            return_columns,
            yield_columns,
            store: ctx.store,
            store_mut: ctx.store_mut,
            transaction_manager: ctx.transaction_manager,
            transaction_id: ctx.transaction_id,
            viewing_epoch: ctx.viewing_epoch,
            catalog: ctx.catalog,
            write_counter: ctx.write_counter,
            result_rows: None,
            row_index: 0,
            output_columns,
        }
    }

    /// Executes the stored procedure body and buffers the results.
    fn execute_body(&mut self) -> Result<(), OperatorError> {
        // Substitute parameters into the body
        let mut body = self.body.clone();
        for (name, value) in &self.params {
            let placeholder = format!("${name}");
            let replacement = value_to_gql_literal(value);
            body = body.replace(&placeholder, &replacement);
        }

        // Use the module-level translate function
        let mut logical_plan = crate::query::translators::gql::translate(&body).map_err(|e| {
            OperatorError::Execution(format!("Failed to translate procedure body: {e}"))
        })?;
        // A pattern through a node or edge bound before is checked, as in a
        // session's plan (the optimizer's first pass; the body skips the
        // optimizer).
        logical_plan.root = crate::query::optimizer::close_cycles(logical_plan.root);

        // Plan physical operators
        let planner = if let Some(ref tx_mgr) = self.transaction_manager {
            let mut p = Planner::with_context(
                Arc::clone(&self.store),
                self.store_mut.as_ref().map(Arc::clone),
                Arc::clone(tx_mgr),
                self.transaction_id,
                self.viewing_epoch,
            );
            if let Some(ref cat) = self.catalog {
                p = p.with_catalog(Arc::clone(cat));
            }
            p
        } else {
            let mut p = Planner::new(Arc::clone(&self.store));
            if let Some(ref cat) = self.catalog {
                p = p.with_catalog(Arc::clone(cat));
            }
            p
        }
        .with_write_counter(Arc::clone(&self.write_counter));

        let physical = planner
            .plan(&logical_plan)
            .map_err(|e| OperatorError::Execution(format!("Failed to plan procedure body: {e}")))?;

        // Execute
        let executor = crate::query::executor::Executor::with_columns(physical.columns().to_vec());
        let mut root_op = physical.into_operator();
        let result: QueryResult = executor
            .execute(root_op.as_mut())
            .map_err(|e| OperatorError::Execution(format!("Procedure execution failed: {e}")))?;

        // Map result columns to expected return columns, handling YIELD filtering
        let column_indices = if let Some(ref yields) = self.yield_columns {
            yields
                .iter()
                .filter_map(|y| result.columns.iter().position(|c| c == y))
                .collect::<Vec<_>>()
        } else {
            // Map return columns to result columns
            self.return_columns
                .iter()
                .filter_map(|r| result.columns.iter().position(|c| c == r))
                .collect::<Vec<_>>()
        };

        // If no columns matched, return all result columns
        let rows = if column_indices.is_empty() {
            result.rows
        } else {
            result
                .rows
                .into_iter()
                .map(|row| column_indices.iter().map(|&i| row[i].clone()).collect())
                .collect()
        };

        self.result_rows = Some(rows);
        Ok(())
    }
}

impl Operator for UserProcedureOperator {
    fn next(&mut self) -> OperatorResult {
        // Execute on first call
        if self.result_rows.is_none() {
            self.execute_body()?;
        }

        let rows = self
            .result_rows
            .as_ref()
            .expect("result_rows populated by execute_body");
        if self.row_index >= rows.len() {
            return Ok(None);
        }

        // Return up to CHUNK_SIZE rows
        const CHUNK_SIZE: usize = 1024;
        let end = (self.row_index + CHUNK_SIZE).min(rows.len());
        let chunk_rows = end - self.row_index;

        let col_count = self
            .output_columns
            .len()
            .max(rows.first().map_or(self.output_columns.len(), |r| r.len()));

        let types: Vec<grafeo_common::types::LogicalType> =
            vec![grafeo_common::types::LogicalType::Any; col_count];
        let mut chunk = DataChunk::with_capacity(&types, chunk_rows);

        for row_idx in self.row_index..end {
            let row = &rows[row_idx];
            for (col_idx, val) in row.iter().enumerate() {
                if let Some(col) = chunk.column_mut(col_idx) {
                    col.push_value(val.clone());
                }
            }
        }
        chunk.set_count(chunk_rows);

        self.row_index = end;
        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.row_index = 0;
        self.result_rows = None;
    }

    fn name(&self) -> &'static str {
        "UserProcedure"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Converts a Value to a GQL literal string for parameter substitution.
fn value_to_gql_literal(value: &Value) -> String {
    match value {
        Value::Null => "NULL".to_string(),
        Value::Bool(b) => if *b { "TRUE" } else { "FALSE" }.to_string(),
        Value::Int64(n) => n.to_string(),
        Value::Float64(f) => format!("{f:?}"),
        Value::String(s) => format!("'{}'", s.replace('\'', "''")),
        _ => format!("'{value}'"),
    }
}
