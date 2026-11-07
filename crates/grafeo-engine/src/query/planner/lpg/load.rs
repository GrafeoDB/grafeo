//! The load procedures, `CALL grafeo.load_nodes_csv({...})` and
//! `grafeo.load_edges_csv({...})` (see [`crate::loader`]): registered
//! procedures whose arguments `CALL` prepares as every procedure's, planned
//! with the statement's writer.

use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};

use super::Operator;
use crate::loader::{LoadCall, LoadCallOperator, REPORT_COLUMNS};
use crate::query::plan::CallProcedureOp;

/// Whether `call` is a load procedure.
pub(super) fn is_load(call: &CallProcedureOp) -> bool {
    crate::loader::is_load(&call.name)
}

impl super::Planner {
    /// Plans a load procedure: its arguments, constants, prepared for its
    /// parameters (in the map or the positional form, see
    /// [`evaluate_arguments`](crate::procedures::evaluate_arguments)) and read
    /// as the load; its YIELD names columns of the report.
    pub(super) fn plan_load_call(
        &self,
        call: &CallProcedureOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let load = crate::procedures::registry()
            .get(&call.name)
            .ok_or_else(|| Error::Internal(format!("{} is not registered", call.name.join("."))))?;
        let procedure = format!("grafeo.{}", load.name());
        let values =
            self.procedure_argument_values(&procedure, &call.arguments, load.parameters())?;
        // A load writes into the graph the statement reads: it reads no projection, so naming
        // one (with a value or null, in a literal map or a parameter's) is refused rather than
        // ignored, before its other arguments are checked.
        if let [Value::Map(entries)] = values.as_slice()
            && entries.keys().any(|key| key.as_str() == "projection")
        {
            return Err(Error::from(QueryError::new(
                QueryErrorKind::InvalidParameterValue,
                format!("{procedure} takes no 'projection': it loads into the graph it runs on"),
            )));
        }
        let params = crate::procedures::evaluate_arguments(&procedure, &values, load.parameters())?;
        let load = LoadCall::from_prepared(&procedure, &params)?;
        let (yielded, columns) = match &call.yield_items {
            None => (
                (0..REPORT_COLUMNS.len()).collect(),
                REPORT_COLUMNS.iter().map(ToString::to_string).collect(),
            ),
            Some(items) => {
                let mut yielded = Vec::with_capacity(items.len());
                let mut columns = Vec::with_capacity(items.len());
                for item in items {
                    let index = REPORT_COLUMNS
                        .iter()
                        .position(|column| *column == item.field_name)
                        .ok_or_else(|| {
                            Error::from(QueryError::invalid_reference(format!(
                                "{procedure} yields {}, not '{}'",
                                REPORT_COLUMNS.join(", "),
                                item.field_name
                            )))
                        })?;
                    yielded.push(index);
                    columns.push(
                        item.alias
                            .clone()
                            .unwrap_or_else(|| item.field_name.clone()),
                    );
                }
                (yielded, columns)
            }
        };
        for column in &columns {
            self.scalar_columns.borrow_mut().insert(column.clone());
        }
        let operator = LoadCallOperator::new(load, self.graph_writer()?, yielded);
        Ok((Box::new(operator), columns))
    }
}
