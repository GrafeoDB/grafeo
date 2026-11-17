//! An aggregate without grouping folded over a label's range in one store
//! call.
//!
//! `MATCH (c:Comment) WHERE c.length > $min RETURN count(*), avg(c.length)`
//! needs no node id: the store walks the label's rows in range and hands each
//! row's values of the properties the aggregates read straight to their
//! states ([`GraphStore::visit_nodes_with_label_in_range`]), without a list of
//! the nodes and without a second read of each value by id. The rows reach
//! the states in the range scan's order through the same per-row update as
//! [`SimpleAggregateOperator`](super::SimpleAggregateOperator), so the answer
//! is the one the range scan and the aggregate give: the same integer sum and
//! its out-of-range error, the same float average in row order.

use std::ops::ControlFlow;
use std::sync::Arc;

use grafeo_common::types::{LogicalType, ReadView};

use super::accumulator::{AggregateExpr, AggregateFunction};
use super::aggregate::{AggregateState, update_row};
use super::{Operator, OperatorError, OperatorResult, ReadCounter};
use crate::execution::chunk::DataChunkBuilder;
use crate::graph::{GraphStore, ValueRange};

/// One row: the aggregates of the nodes of `label` whose `property` lies in
/// `range`, each aggregate's column an index into `read` (`None` for
/// `count(*)`). See the [module](self) documentation.
pub struct RangeFoldOperator {
    store: Arc<dyn GraphStore>,
    label: String,
    property: String,
    range: ValueRange,
    read: Vec<String>,
    aggregates: Vec<AggregateExpr>,
    output_schema: Vec<LogicalType>,
    view: ReadView,
    done: bool,
    reads: Option<Arc<ReadCounter>>,
}

impl RangeFoldOperator {
    /// The fold of `aggregates` over the nodes of `label` in `range` of
    /// `property` that `view` sees, reading the properties `read`. A count's
    /// column is an integer, every other aggregate's a value of any kind, as
    /// the aggregate it stands for types them.
    #[must_use]
    pub fn new(
        store: Arc<dyn GraphStore>,
        label: String,
        property: String,
        range: ValueRange,
        read: Vec<String>,
        aggregates: Vec<AggregateExpr>,
        view: ReadView,
    ) -> Self {
        let output_schema = aggregates
            .iter()
            .map(|aggregate| match aggregate.function {
                AggregateFunction::Count | AggregateFunction::CountNonNull => LogicalType::Int64,
                _ => LogicalType::Any,
            })
            .collect();
        Self {
            store,
            label,
            property,
            range,
            read,
            aggregates,
            output_schema,
            view,
            done: false,
            reads: None,
        }
    }

    /// Counts the rows the fold reads in `reads`, as the range scan counts
    /// the nodes it returns.
    #[must_use]
    pub fn with_read_counter(mut self, reads: Arc<ReadCounter>) -> Self {
        self.reads = Some(reads);
        self
    }
}

impl Operator for RangeFoldOperator {
    fn next(&mut self) -> OperatorResult {
        if self.done {
            return Ok(None);
        }
        self.done = true;
        let mut states: Vec<AggregateState> = self
            .aggregates
            .iter()
            .map(AggregateState::for_expr)
            .collect();
        let read: Vec<&str> = self.read.iter().map(String::as_str).collect();
        let aggregates = &self.aggregates;
        // The query's deadline, checked as the walk goes: a timeout or a
        // cancellation stops it with the executor's error.
        let mut ticker = crate::execution::deadline::Ticker::new();
        let mut stopped: Option<OperatorError> = None;
        let mut rows = 0_usize;
        self.store.visit_nodes_with_label_in_range(
            &self.label,
            &self.property,
            &self.range,
            &read,
            self.view,
            &mut |values| {
                if let Err(error) = ticker.tick() {
                    stopped = Some(error);
                    return ControlFlow::Break(());
                }
                rows += 1;
                update_row(aggregates, &mut states, |column| {
                    values.get(column).cloned().flatten()
                });
                ControlFlow::Continue(())
            },
        )?;
        if let Some(error) = stopped {
            return Err(error);
        }
        super::read_counter::count(self.reads.as_deref(), rows);
        let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, 1);
        for (i, (state, expr)) in states.iter().zip(&self.aggregates).enumerate() {
            if let Some(column) = builder.column_mut(i) {
                column.push_value(state.finalize_for(expr)?);
            }
        }
        builder.advance_row();
        Ok(Some(builder.finish()))
    }

    fn reset(&mut self) {
        self.done = false;
    }

    fn name(&self) -> &'static str {
        "RangeFold"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}
