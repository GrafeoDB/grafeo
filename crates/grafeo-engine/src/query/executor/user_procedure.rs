//! Operator for executing user-defined stored procedures.
//!
//! `bind_body` puts the call's arguments into the stored GQL body, and
//! translates and binds it when the call is planned; the operator plans and
//! executes it as a sub-query using a fresh planner/executor pipeline.

use std::collections::HashMap;
use std::sync::Arc;

use grafeo_common::types::{EpochId, TransactionId, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::operators::{
    Operator, OperatorError, OperatorResult, Recording, WriteCounter,
};
use grafeo_core::graph::{GraphStoreMut, GraphStoreSearch};

use crate::catalog::Catalog;
use crate::database::QueryResult;
use crate::query::binder::Binder;
use crate::query::plan::LogicalPlan;
use crate::query::planner::Planner;
use crate::transaction::TransactionManager;

/// The body of procedure `name` with the call's arguments in place of its
/// parameters, translated and bound as a statement is, so a mistake in it is
/// found when the call is planned, before anything runs.
///
/// # Errors
///
/// Returns the syntax or semantic error of the body (an unbound variable,
/// for one), its message naming the procedure.
pub(crate) fn bind_body(
    name: &str,
    body: &str,
    params: &HashMap<String, Value>,
) -> Result<LogicalPlan> {
    let mut body = body.to_string();
    for (param, value) in params {
        body = body.replace(&format!("${param}"), &value_to_gql_literal(value));
    }
    let in_procedure = |error: Error| match error {
        Error::Query(mut query) => {
            query.message = format!("procedure '{name}': {}", query.message);
            Error::Query(query)
        }
        other => other,
    };
    let mut plan = crate::query::translators::gql::translate(&body).map_err(in_procedure)?;
    Binder::new().bind(&plan).map_err(in_procedure)?;
    // A pattern through a node or edge bound before is checked, as in a
    // session's plan (the optimizer's first pass; the body skips the
    // optimizer).
    plan.root = crate::query::optimizer::close_cycles(plan.root);
    Ok(plan)
}

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
    /// The calling statement's recording: the body's writes are its
    /// transaction's.
    pub recording: Option<Recording>,
    /// The database's projections, so a `CALL ... {projection: ...}` in the
    /// body runs as it does at the top level.
    #[cfg(feature = "lpg")]
    pub projections: Option<crate::session::ProjectionRegistry>,
}

/// An operator that executes a user-defined stored procedure.
///
/// On first call to `next()`, it:
/// 1. Plans the bound body (`bind_body`) with a sub-planner
/// 2. Executes it
/// 3. Buffers the results
/// 4. Returns results in chunks
pub struct UserProcedureOperator {
    /// The bound body, with the call's arguments in place.
    plan: LogicalPlan,
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
    /// The calling statement's recording.
    recording: Option<Recording>,
    /// The database's projections, for `CALL ... {projection: ...}` in the body.
    #[cfg(feature = "lpg")]
    projections: Option<crate::session::ProjectionRegistry>,
    /// Buffered result rows from execution.
    result_rows: Option<Vec<Vec<Value>>>,
    /// Current row index into buffered results.
    row_index: usize,
    /// Output column names after YIELD filtering.
    output_columns: Vec<String>,
}

impl UserProcedureOperator {
    /// Creates a new user procedure operator running `plan`, a body
    /// `bind_body` bound.
    pub fn new(
        plan: LogicalPlan,
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
            plan,
            return_columns,
            yield_columns,
            store: ctx.store,
            store_mut: ctx.store_mut,
            transaction_manager: ctx.transaction_manager,
            transaction_id: ctx.transaction_id,
            viewing_epoch: ctx.viewing_epoch,
            catalog: ctx.catalog,
            write_counter: ctx.write_counter,
            recording: ctx.recording,
            #[cfg(feature = "lpg")]
            projections: ctx.projections,
            result_rows: None,
            row_index: 0,
            output_columns,
        }
    }

    /// Executes the stored procedure body and buffers the results.
    fn execute_body(&mut self) -> std::result::Result<(), OperatorError> {
        let logical_plan = &self.plan;

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
            p.with_recording(self.recording.clone())
        } else {
            let mut p = Planner::new(Arc::clone(&self.store));
            if let Some(ref cat) = self.catalog {
                p = p.with_catalog(Arc::clone(cat));
            }
            p
        }
        .with_write_counter(Arc::clone(&self.write_counter));
        #[cfg(feature = "lpg")]
        let planner = match &self.projections {
            Some(projections) => planner.with_projections(Arc::clone(projections)),
            None => planner,
        };

        let physical = planner.plan(logical_plan).map_err(OperatorError::from)?;

        // Execute
        let executor = crate::query::executor::Executor::with_columns(physical.columns().to_vec());
        let mut root_op = physical.into_operator();
        let result: QueryResult = executor
            .execute(root_op.as_mut())
            .map_err(OperatorError::from)?;

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
