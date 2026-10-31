//! The load procedures, `CALL grafeo.load_nodes_csv({...})` and
//! `grafeo.load_edges_csv({...})` (see [`crate::loader`]): planned with the
//! statement's writer, in any build (the other procedures need `algos`).

use std::collections::BTreeMap;

use grafeo_common::types::{PropertyKey, Value};
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};

use super::Operator;
use crate::loader::{LoadCall, LoadCallOperator, REPORT_COLUMNS};
use crate::query::plan::{CallProcedureOp, LogicalExpression};

/// Whether `call` is a load procedure.
pub(super) fn is_load(call: &CallProcedureOp) -> bool {
    call.writes()
}

impl super::Planner {
    /// Plans a load procedure: its one map argument, constant, read as the
    /// load's options; its YIELD names columns of the report.
    pub(super) fn plan_load_call(
        &self,
        call: &CallProcedureOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let procedure = call.name.join(".");
        let semantic =
            |message: String| Error::from(QueryError::new(QueryErrorKind::Semantic, message));
        let options = match call.arguments.as_slice() {
            [argument] => match constant(argument) {
                Some(Value::Map(options)) => options,
                _ => {
                    return Err(semantic(format!(
                        "{procedure} takes one map of constant options"
                    )));
                }
            },
            _ => {
                return Err(semantic(format!(
                    "{procedure} takes one map of options, such as {{path: 'nodes.csv'}}"
                )));
            }
        };
        let load = LoadCall::from_options(&procedure, &options)?;
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

/// The value of a constant argument: a literal, or a list or map of them
/// (parameters are literals by the time a plan is made).
fn constant(expression: &LogicalExpression) -> Option<Value> {
    match expression {
        LogicalExpression::Literal(value) => Some(value.clone()),
        LogicalExpression::List(items) => Some(Value::List(
            items
                .iter()
                .map(constant)
                .collect::<Option<Vec<_>>>()?
                .into(),
        )),
        LogicalExpression::Map(entries) => {
            let mut map = BTreeMap::new();
            for (key, value) in entries {
                map.insert(PropertyKey::new(key.as_str()), constant(value)?);
            }
            Some(Value::Map(std::sync::Arc::new(map)))
        }
        _ => None,
    }
}
