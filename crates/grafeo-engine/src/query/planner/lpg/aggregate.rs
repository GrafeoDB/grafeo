//! Aggregate and factorized aggregate planning.

use grafeo_core::execution::operators::MAX_COUNTED_DIFFERENT_HOPS;

use super::{
    AggregateOp, Arc, Direction, EntityValue, Error, ExpandDirection, ExpandOp, ExpandStep,
    ExpressionPredicate, FactorizedAggregate, FactorizedAggregateOperator, FilterExpression,
    FilterOperator, GraphStore, HashAggregateOperator, HashMap, LazyFactorizedChainOperator,
    LogicalAggregateFunction, LogicalExpression, LogicalOperator, LogicalType, Operator, PathMode,
    PhysicalAggregateExpr, ProjectExpr, ProjectOperator, Result, SimpleAggregateOperator,
    convert_aggregate_function, expression_to_string, resolved_column_name,
};
use crate::query::optimizer::{PlannerRule, PlannerRules};
use crate::query::plan::{AggregateExpr, RangeFoldShape};
use grafeo_common::types::{StatementLanguage, Value};
use grafeo_common::utils::error::{QueryError, QueryErrorKind};

type Built = (Box<dyn Operator>, Vec<String>);

/// What a range fold reads: the property and its range.
pub(super) type RangeRead = (String, grafeo_core::graph::ValueRange);

/// Everything the route of an aggregate reads besides the plan, fixed for a cache key and a
/// database, so a prepared plan's recipe decides it without a planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RouteInputs {
    /// The statement's language: in SQL/PGQ a range fold reads the column's kinds.
    pub(crate) language: StatementLanguage,
    /// The planner rules.
    pub(crate) rules: PlannerRules,
    /// Factorized execution: the database's setting and its rule.
    pub(crate) factorized_execution: bool,
    /// Whether the store keeps backward adjacency, which the middle sum reads.
    pub(crate) backward_adjacency: bool,
}

/// A route with what its builder reads.
pub(super) enum Route<'a, T> {
    MiddleSum(EdgeCountShape<'a>, Vec<String>),
    EdgeCount(EdgeCountShape<'a>),
    Factorized,
    LabelCount(&'a str),
    RangeFold(RangeFoldShape<'a>, T),
    Regular,
}

impl<'a, T> Route<'a, T> {
    /// The operator the route's builder reads its rows from, `None` for one that reads the
    /// store itself.
    ///
    /// # Errors
    ///
    /// A shape the route does not hold, which [`route`] never gives.
    pub(super) fn input(&self, agg: &'a AggregateOp) -> Result<Option<&'a LogicalOperator>> {
        Ok(match self {
            Self::EdgeCount(shape) => Some(edge_count_input(shape)?),
            Self::Factorized => Some(factorized_base(agg)?),
            Self::Regular => Some(agg.input.as_ref()),
            Self::MiddleSum(..) | Self::LabelCount(_) | Self::RangeFold(..) => None,
        })
    }
}

/// The route `agg` is built by: the first whose conditions hold, in the order of [`Route`]'s
/// variants. It reads the plan and `inputs`, and gives a range fold's filter to `range`, which
/// answers what the fold reads, `None` when the filter reads no range.
pub(super) fn route<'a, T>(
    inputs: RouteInputs,
    agg: &'a AggregateOp,
    range: impl FnOnce(&RangeFoldShape<'a>) -> Option<T>,
) -> Route<'a, T> {
    if inputs.factorized_execution
        && let Some(shape) = EdgeCountShape::of(agg)
    {
        // A count of the last hop's edges, also over a chain of expands that the factorized
        // aggregate would count too: counting each row's edges from the adjacency, without
        // listing them, takes about half the time of the factorized chain, which lists the
        // edges of its last level.
        return match middle_sum_first_labels(inputs, agg, &shape) {
            Some(first_labels) => Route::MiddleSum(shape, first_labels),
            None => Route::EdgeCount(shape),
        };
    }
    // Factorized aggregation, over a chain of two or more expands without GROUP BY, of counts
    // of the chain's rows. Not after a write (the regular path reads the writing input whole
    // first, see `after_write`). A `count(*)` or an aggregated variable reads nothing itself,
    // and the chain reads its whole source before its first expand (the lazy chain collects
    // every batch, and the node scan of a MATCH from a bound node reads a writing input whole,
    // see `plan_node_scan`): the count over the chain counts what the whole write left.
    if inputs.factorized_execution
        && agg.group_by.is_empty()
        && !(super::after_write::aggregate_reads(agg)
            && super::after_write::writes_pending(&agg.input))
        && super::Planner::count_expand_chain(&agg.input).0 >= 2
        && super::Planner::is_simple_aggregate(agg)
    {
        return Route::Factorized;
    }
    if let Some(label) = agg.counted_label() {
        return Route::LabelCount(label);
    }
    if inputs.rules.has(PlannerRule::RangeFold)
        && inputs.rules.has(PlannerRule::IndexRange)
        && let Some(shape) = agg.range_fold_shape()
        && let Some(read) = range(&shape)
    {
        return Route::RangeFold(shape, read);
    }
    Route::Regular
}

impl super::Planner {
    /// The inputs of the route this planner takes ([`RouteInputs`]).
    pub(crate) fn route_inputs(&self) -> RouteInputs {
        RouteInputs {
            language: self.session_context.language,
            rules: self.rules,
            factorized_execution: self.factorized_execution,
            backward_adjacency: self.store.has_backward_adjacency(),
        }
    }

    /// The route `agg` is built by ([`route`]), with what its builder reads.
    pub(super) fn aggregate_route<'a>(&self, agg: &'a AggregateOp) -> Route<'a, RangeRead> {
        route(self.route_inputs(), agg, |shape| {
            self.filter_range(&shape.filter.predicate, shape.variable)
        })
    }

    /// Plans an AGGREGATE operator by its route ([`route`]).
    pub(super) fn plan_aggregate(&self, agg: &AggregateOp) -> Result<Built> {
        let route = self.aggregate_route(agg);
        let input = match (&route, route.input(agg)?) {
            // An aggregate that comes first (`RETURN count(*)`) aggregates the one empty row.
            (Route::Regular, Some(input)) => Some(self.plan_input(input)?),
            (_, Some(input)) => Some(self.plan_operator(input)?),
            (_, None) => None,
        };
        self.build_route(agg, route, input)
    }

    /// The operators of `agg` by `route`, over `input`, the plan of the route's
    /// [`input`](Route::input).
    ///
    /// # Errors
    ///
    /// The errors of building the operators, and a missing input the route reads.
    pub(super) fn build_route(
        &self,
        agg: &AggregateOp,
        route: Route<'_, RangeRead>,
        input: Option<Built>,
    ) -> Result<Built> {
        let input = || input.ok_or_else(|| Error::Internal("the route reads an input".to_string()));
        match route {
            Route::MiddleSum(shape, first_labels) => {
                let (counts, columns) = self.middle_sum_of(&shape, first_labels)?;
                self.finish_edge_count(agg, &shape, counts, columns)
            }
            Route::EdgeCount(shape) => {
                let (input_op, columns) = input()?;
                self.edge_count_over(agg, &shape, input_op, columns)
            }
            Route::Factorized => {
                let (base_op, base_columns) = input()?;
                self.factorized_over(agg, base_op, base_columns)
            }
            Route::LabelCount(label) => Ok(self.plan_label_count(agg, label)),
            Route::RangeFold(shape, (property, range)) => {
                Ok(self.plan_range_fold(agg, &shape, property, range))
            }
            Route::Regular => {
                let (mut input_op, input_columns) = input()?;
                // Values that read the graph after a write (`... SET h.c = i RETURN
                // sum(h.c)`) read what the whole write left (see `after_write`).
                if super::after_write::aggregate_reads(agg) {
                    input_op = super::mutation::read_first_after_a_write(input_op, &agg.input);
                }
                self.aggregate_over(agg, input_op, input_columns)
            }
        }
    }

    /// The percentile of a `PERCENTILE_DISC` or `PERCENTILE_CONT`, evaluated
    /// once: it is the same for every row (a literal, a parameter or an
    /// expression of them, the parameters already substituted) and a number
    /// from 0 to 1. `None` for every other aggregate.
    ///
    /// # Errors
    ///
    /// A percentile aggregate without its percentile, or one that reads the
    /// rows (42000); a percentile that is null or no number (22G03), or
    /// outside 0 to 1 (22003).
    pub(super) fn constant_percentile(
        &self,
        function: LogicalAggregateFunction,
        percentile: Option<&LogicalExpression>,
    ) -> Result<Option<f64>> {
        let refuse = |kind, message: String| Err(Error::from(QueryError::new(kind, message)));
        if !matches!(
            function,
            LogicalAggregateFunction::PercentileDisc | LogicalAggregateFunction::PercentileCont
        ) {
            return Ok(None);
        }
        let Some(expression) = percentile else {
            return refuse(
                QueryErrorKind::Semantic,
                "a percentile aggregate takes its percentile as the second argument".to_string(),
            );
        };
        if !super::seek::value_variables(expression).is_some_and(|read| read.is_empty()) {
            return refuse(
                QueryErrorKind::Semantic,
                format!(
                    "the percentile {} must be the same for every row: a literal, a parameter \
                     or an expression of them",
                    expression_to_string(expression)
                ),
            );
        }
        let value = ExpressionPredicate::new(
            self.convert_expression(expression)?,
            HashMap::new(),
            Arc::clone(&self.store),
            self.read_view(),
        )
        .with_session_context(self.session_context.clone())
        .eval_at(&grafeo_core::execution::DataChunk::empty(), 0)
        .map_err(crate::query::executor::convert_operator_error)?;
        let number = match value {
            Some(Value::Int64(n)) => n as f64,
            Some(Value::Float64(f)) => f,
            Some(Value::Decimal(d)) => d.to_f64(),
            Some(Value::Null) | None => {
                return refuse(
                    QueryErrorKind::InvalidValueType,
                    "the percentile is null; it takes a number from 0 to 1".to_string(),
                );
            }
            Some(other) => {
                return refuse(
                    QueryErrorKind::InvalidValueType,
                    format!("the percentile {other} is no number; it takes one from 0 to 1"),
                );
            }
        };
        if !(0.0..=1.0).contains(&number) {
            return refuse(
                QueryErrorKind::NumericOutOfRange,
                format!("the percentile {number} is outside 0 to 1"),
            );
        }
        Ok(Some(number))
    }

    /// Plans an AGGREGATE operator.
    /// `count(*)` and `count(n)` of a scan of one label and nothing else
    /// (`MATCH (n:Graph) RETURN count(*)`): the store's count of the label's
    /// nodes, read when the query runs (see `LabelCountOperator`), without
    /// a scan, of `label` ([`AggregateOp::counted_label`]).
    fn plan_label_count(&self, agg: &AggregateOp, label: &str) -> (Box<dyn Operator>, Vec<String>) {
        let columns: Vec<String> = agg
            .aggregates
            .iter()
            .map(|aggregate| {
                aggregate.alias.clone().unwrap_or_else(|| {
                    crate::query::planner::common::aggregate_column_name(aggregate)
                })
            })
            .collect();
        for column in &columns {
            self.scalar_columns.borrow_mut().insert(column.clone());
        }
        // PROFILE names the scan the count stands in for.
        self.record_absorbed_scan_entry("NodeScan", &agg.input);
        let operator = grafeo_core::execution::operators::LabelCountOperator::new(
            Arc::clone(&self.store) as Arc<dyn GraphStore>,
            label.to_string(),
            self.read_view(),
            columns.len(),
        )
        .with_read_counter(self.read_counter());
        (Box::new(operator), columns)
    }

    /// `count(*)`, `count(n)`, and `count`, `sum`, `avg`, `min` and `max` of
    /// properties of `n`, none distinct, with no grouping and no `HAVING`, over
    /// a range filter on a scan of one label (`MATCH (n:L) WHERE n.k > $v
    /// RETURN count(*), avg(n.k)`), also through a projection that only names
    /// properties of `n` (SQL/PGQ's `COLUMNS (n.k AS len)`): the range's
    /// values folded in one store call (see `RangeFoldOperator`), without the
    /// range scan's node list or a second read of each value, of the `shape`,
    /// `property` and `range` its route found ([`route`]).
    fn plan_range_fold(
        &self,
        agg: &AggregateOp,
        shape: &RangeFoldShape<'_>,
        property: String,
        range: grafeo_core::graph::ValueRange,
    ) -> (Box<dyn Operator>, Vec<String>) {
        let mut read: Vec<String> = Vec::new();
        let aggregates: Vec<PhysicalAggregateExpr> = agg
            .aggregates
            .iter()
            .zip(&shape.reads)
            .map(|(aggregate, property)| {
                let column = property.map(|property| {
                    read.iter()
                        .position(|known| known == property)
                        .unwrap_or_else(|| {
                            read.push(property.to_string());
                            read.len() - 1
                        })
                });
                // A count of the rows reads no value, as `count(*)` does.
                let function = match (aggregate.function, column) {
                    (LogicalAggregateFunction::CountNonNull, None) => {
                        LogicalAggregateFunction::Count
                    }
                    (function, _) => function,
                };
                PhysicalAggregateExpr {
                    function: convert_aggregate_function(function),
                    column,
                    column2: None,
                    distinct: false,
                    alias: aggregate.alias.clone(),
                    percentile: None,
                    separator: None,
                    rdf_literals: false,
                    temporal: self.session_context.temporal_comparison,
                    keep_nulls: aggregate.keep_nulls,
                }
            })
            .collect();
        let columns: Vec<String> = agg
            .aggregates
            .iter()
            .map(|aggregate| {
                aggregate.alias.clone().unwrap_or_else(|| {
                    crate::query::planner::common::aggregate_column_name(aggregate)
                })
            })
            .collect();
        for column in &columns {
            self.scalar_columns.borrow_mut().insert(column.clone());
        }
        // PROFILE names what the fold stands in for.
        let filter = match agg.input.as_ref() {
            LogicalOperator::Project(project) => {
                self.record_absorbed_scan_entry("Project", &agg.input);
                &project.input
            }
            LogicalOperator::Return(project) => {
                self.record_absorbed_scan_entry("Return", &agg.input);
                &project.input
            }
            _ => &agg.input,
        };
        self.record_absorbed_scan_entry("Filter", filter);
        self.record_absorbed_scan_entry("NodeScan", &shape.filter.input);
        let operator = grafeo_core::execution::operators::RangeFoldOperator::new(
            Arc::clone(&self.store) as Arc<dyn GraphStore>,
            shape.label.to_string(),
            property,
            range,
            read,
            aggregates,
            self.read_view(),
        )
        .with_read_counter(self.read_counter());
        (Box::new(operator), columns)
    }

    /// A count over one expand that reads neither the edge nor its far end
    /// (`MATCH (a:V)-[:E]->(:V) RETURN a.id, count(*)`): the count of each
    /// input row's edges, read from the adjacency (see `EdgeCountOperator`),
    /// summed per group, or totalled in one row without groups (`MATCH
    /// (a)-[:E]->() RETURN count(*)`), where the expand would make a row of
    /// every edge for the aggregate to count. Labels on the far end stay a
    /// check of each counted edge, as the store keeps no count of the edges
    /// to a label. Over a chain of expands the last one is counted, for
    /// each row of the others, `shape` ([`EdgeCountShape::of`]), over `input_op`, the plan of
    /// [`edge_count_input`].
    fn edge_count_over(
        &self,
        agg: &AggregateOp,
        shape: &EdgeCountShape<'_>,
        input_op: Box<dyn Operator>,
        columns: Vec<String>,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let Some(first) = shape.hops.last() else {
            return Err(Error::Internal("an edge count without a hop".to_string()));
        };
        let expand = first.expand;
        let source_column = columns
            .iter()
            .position(|c| c == &expand.from_variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Source variable '{}' not found in input columns",
                    expand.from_variable
                ))
            })?;
        let direction_of = |expand: &ExpandOp| match expand.direction {
            ExpandDirection::Outgoing => Direction::Outgoing,
            ExpandDirection::Incoming => Direction::Incoming,
            ExpandDirection::Both => Direction::Both,
        };
        let direction = direction_of(expand);
        let counts = grafeo_core::execution::operators::EdgeCountOperator::new(
            Arc::clone(&self.store) as Arc<dyn GraphStore>,
            input_op,
            source_column,
            direction,
            self.stored_edge_types(&expand.edge_types)?,
            first.end_labels.clone(),
            self.read_view(),
        )
        .with_read_counter(self.read_counter());
        // A chain goes on from the first hop, bottom up.
        let counts = if shape.hops.len() > 1 || !shape.different.is_empty() {
            let mut rest = Vec::with_capacity(shape.hops.len() - 1);
            for hop in shape.hops.iter().rev().skip(1) {
                rest.push(grafeo_core::execution::operators::CountedHop {
                    direction: direction_of(hop.expand),
                    edge_types: self
                        .stored_edge_types(&hop.expand.edge_types)?
                        .into_iter()
                        .map(Into::into)
                        .collect(),
                    end_labels: hop
                        .end_labels
                        .iter()
                        .map(|label| label.as_str().into())
                        .collect(),
                });
            }
            counts.with_chain(rest, &shape.different)
        } else {
            counts
        };
        self.finish_edge_count(agg, shape, counts, columns)
    }

    /// The middle sum of `shape`, a two-hop count whose first node has `first_labels`
    /// ([`Self::middle_sum_first_labels`]).
    fn middle_sum_of(
        &self,
        shape: &EdgeCountShape<'_>,
        first_labels: Vec<String>,
    ) -> Result<PlannedCount> {
        let [on, near] = shape.hops.as_slice() else {
            return Err(Error::Internal(
                "a middle sum of other than two hops".to_string(),
            ));
        };
        let direction_of = |direction: ExpandDirection| match direction {
            ExpandDirection::Outgoing => Direction::Outgoing,
            ExpandDirection::Incoming => Direction::Incoming,
            ExpandDirection::Both => Direction::Both,
        };
        let back_direction = match near.expand.direction {
            ExpandDirection::Outgoing => Direction::Incoming,
            ExpandDirection::Incoming => Direction::Outgoing,
            ExpandDirection::Both => Direction::Both,
        };
        let stored = |types: &[String]| -> Result<Vec<grafeo_common::types::ArcStr>> {
            Ok(self
                .stored_edge_types(types)?
                .into_iter()
                .map(Into::into)
                .collect())
        };
        let labels = |labels: &[String]| -> Vec<grafeo_common::types::ArcStr> {
            labels.iter().map(|label| label.as_str().into()).collect()
        };
        let middle_label = near.end_labels.first().cloned();
        let sum = grafeo_core::execution::operators::MiddleSum {
            back: grafeo_core::execution::operators::CountedHop {
                direction: back_direction,
                edge_types: stored(&near.expand.edge_types)?,
                end_labels: labels(&first_labels),
            },
            on: grafeo_core::execution::operators::CountedHop {
                direction: direction_of(on.expand.direction),
                edge_types: stored(&on.expand.edge_types)?,
                end_labels: labels(&on.end_labels),
            },
            middle_labels: labels(near.end_labels.get(1..).unwrap_or_default()),
            different_edges: !shape.different.is_empty(),
        };
        let middle = LogicalOperator::NodeScan(crate::query::plan::NodeScanOp {
            variable: on.expand.from_variable.clone(),
            label: middle_label,
            input: None,
        });
        let (input, columns) = self.plan_operator(&middle)?;
        let counts = grafeo_core::execution::operators::EdgeCountOperator::new(
            Arc::clone(&self.store) as Arc<dyn GraphStore>,
            input,
            0,
            Direction::Outgoing,
            Vec::new(),
            Vec::new(),
            self.read_view(),
        )
        .with_read_counter(self.read_counter())
        .with_middle_sum(sum);
        Ok((counts, columns))
    }

    /// The plan of `counts` over `columns`, with the aggregate it stands in for:
    /// each hop's profile entries, then a total or a sum per group.
    fn finish_edge_count(
        &self,
        agg: &AggregateOp,
        shape: &EdgeCountShape<'_>,
        counts: grafeo_core::execution::operators::EdgeCountOperator,
        columns: Vec<String>,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Without groups the count totals itself in one row, which holds 0
        // when no edge matches (a sum of the counts over no rows is null),
        // and stands in for the aggregate too.
        let ungrouped = agg.group_by.is_empty();
        let counts: Box<dyn Operator> = if ungrouped {
            Box::new(counts.totalled(agg.aggregates.len()))
        } else {
            Box::new(counts)
        };
        // PROFILE shows the count in the expand's place, with the label
        // checks of the far end it does.
        for hop in shape.hops.iter().skip(1).rev() {
            self.record_absorbed_scan_entry("Expand", hop.node);
            for check in hop.checks.iter().rev() {
                self.record_absorbed_scan_entry("Filter", check);
            }
        }
        let last = &shape.hops[0];
        let (counts, mut columns) = self.maybe_profile(Ok((counts, columns)), last.node)?;
        for check in last.checks.iter().rev() {
            self.record_absorbed_scan_entry("Filter", check);
        }
        if ungrouped {
            let totals: Vec<String> = agg
                .aggregates
                .iter()
                .map(|aggregate| {
                    aggregate.alias.clone().unwrap_or_else(|| {
                        crate::query::planner::common::aggregate_column_name(aggregate)
                    })
                })
                .collect();
            for total in &totals {
                self.set_column_entity(total, None);
            }
            let operator = self.filter_having(counts, &totals, agg.having.as_ref())?;
            return Ok((operator, totals));
        }
        // A name no variable has: the count column of this aggregate.
        let mut count_column = String::from("_edge_count");
        while columns.contains(&count_column) {
            count_column.push('_');
        }
        columns.push(count_column.clone());
        self.scalar_columns
            .borrow_mut()
            .insert(count_column.clone());
        // Each count sums the rows' counts, under the name it had.
        let summed = AggregateOp {
            group_by: agg.group_by.clone(),
            aggregates: agg
                .aggregates
                .iter()
                .map(|aggregate| AggregateExpr {
                    function: LogicalAggregateFunction::Sum,
                    expression: Some(LogicalExpression::Variable(count_column.clone())),
                    expression2: None,
                    distinct: false,
                    alias: Some(aggregate.alias.clone().unwrap_or_else(|| {
                        crate::query::planner::common::aggregate_column_name(aggregate)
                    })),
                    percentile: None,
                    separator: None,
                    keep_nulls: false,
                })
                .collect(),
            input: Box::new(LogicalOperator::Empty),
            having: agg.having.clone(),
        };
        self.aggregate_over(&summed, counts, columns)
    }

    /// Plans `agg` over `input_op`, whose rows hold `input_columns`.
    fn aggregate_over(
        &self,
        agg: &AggregateOp,
        mut input_op: Box<dyn Operator>,
        mut input_columns: Vec<String>,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // A subquery in a grouping key or an aggregate's argument
        // (`sum(COUNT { ... })`) runs per input row, as one in a projection
        // does (see `lift_subqueries`): its value is a column of the rows
        // the aggregate reads. A key column keeps the name of the key as
        // written (`COUNT {...}`), which the RETURN above it reads.
        let key_names: Vec<String> = agg.group_by.iter().map(expression_to_string).collect();
        let input_writes = agg.input.has_mutations();
        let lifted_aggregate;
        let agg = if agg
            .group_by
            .iter()
            .chain(
                agg.aggregates.iter().flat_map(|aggregate| {
                    aggregate.expression.iter().chain(&aggregate.expression2)
                }),
            )
            .any(|expression| {
                self.has_subquery_to_lift(expression, Some(&input_columns), input_writes)
            }) {
            let mut owned = agg.clone();
            let expressions =
                owned
                    .group_by
                    .iter_mut()
                    .chain(owned.aggregates.iter_mut().flat_map(|aggregate| {
                        aggregate
                            .expression
                            .iter_mut()
                            .chain(aggregate.expression2.iter_mut())
                    }));
            for expression in expressions {
                let (lifted, lifted_input, lifted_columns) =
                    self.lift_subqueries(expression, input_op, input_columns, input_writes)?;
                *expression = lifted;
                input_op = lifted_input;
                input_columns = lifted_columns;
            }
            lifted_aggregate = owned;
            &lifted_aggregate
        } else {
            agg
        };

        // Build variable to column index mapping
        let mut variable_columns: HashMap<String, usize> = input_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Collect all extra projections (property access and complex expressions)
        // in a single ordered list so that column index assignment matches the
        // order they are added to the ProjectOperator.
        enum ExtraProjection {
            Property { variable: String, property: String },
            Expression { filter_expr: FilterExpression },
        }
        let mut extra_projections: Vec<ExtraProjection> = Vec::new();
        let mut next_col_idx = input_columns.len();
        // The group keys that are `labels(n)` of a variable: grouped by the
        // store's label mask, one integer, and named after the grouping.
        let mut mask_keys: Vec<usize> = Vec::new();

        // Check group-by expressions for properties and complex expressions
        // (Labels, Type, FunctionCall, IndexAccess, etc.)
        for expr in &agg.group_by {
            match expr {
                LogicalExpression::Property { variable, property } => {
                    let col_name = resolved_column_name(expr);
                    if !variable_columns.contains_key(&col_name) {
                        extra_projections.push(ExtraProjection::Property {
                            variable: variable.clone(),
                            property: property.clone(),
                        });
                        variable_columns.insert(col_name, next_col_idx);
                        next_col_idx += 1;
                    }
                }
                LogicalExpression::Variable(_) => {
                    // Already in variable_columns, nothing to project
                }
                _ => {
                    // Complex expression (Labels, Type, FunctionCall, IndexAccess,
                    // CASE, Binary, etc.): project as computed column
                    let col_name = resolved_column_name(expr);
                    if !variable_columns.contains_key(&col_name) {
                        let filter_expr = match labels_of_variable(expr) {
                            Some(variable) => {
                                mask_keys.push(
                                    agg.group_by
                                        .iter()
                                        .position(|key| std::ptr::eq(key, expr))
                                        .expect("the key is one of the group keys"),
                                );
                                FilterExpression::LabelMask(variable.to_string())
                            }
                            None => self.convert_expression(expr)?,
                        };
                        extra_projections.push(ExtraProjection::Expression { filter_expr });
                        variable_columns.insert(col_name, next_col_idx);
                        next_col_idx += 1;
                    }
                }
            }
        }

        // Check aggregate expressions for properties and complex expressions
        // (both first and second arguments)
        for agg_expr in &agg.aggregates {
            for expr_opt in [&agg_expr.expression, &agg_expr.expression2] {
                let Some(expr) = expr_opt else { continue };
                match expr {
                    LogicalExpression::Property { variable, property } => {
                        let col_name = resolved_column_name(expr);
                        if !variable_columns.contains_key(&col_name) {
                            extra_projections.push(ExtraProjection::Property {
                                variable: variable.clone(),
                                property: property.clone(),
                            });
                            variable_columns.insert(col_name, next_col_idx);
                            next_col_idx += 1;
                        }
                    }
                    LogicalExpression::Variable(_) => {
                        // Already in variable_columns, nothing to project
                    }
                    _ => {
                        // Complex expression (CASE, Binary, etc.): project as computed column
                        let col_name = resolved_column_name(expr);
                        if !variable_columns.contains_key(&col_name) {
                            let filter_expr = self.convert_expression(expr)?;
                            extra_projections.push(ExtraProjection::Expression { filter_expr });
                            variable_columns.insert(col_name, next_col_idx);
                            next_col_idx += 1;
                        }
                    }
                }
            }
        }

        // If we have extra projections, add a projection to materialize them
        if !extra_projections.is_empty() {
            let mut projections = Vec::new();
            let mut output_types = Vec::new();

            // First, pass through every input column as it is, typed as the
            // other pass-through projections type it: an edge column as
            // edges, every other column `Any`, a copy that keeps the input's
            // vector type (node IDs stay nodes) and copies strings, floats,
            // booleans, lists, maps and paths unchanged. A copy typed `Node`
            // turns every value that is not an integer into node 0.
            for (i, column_type) in self
                .derive_schema_from_columns(&input_columns)
                .into_iter()
                .enumerate()
            {
                projections.push(ProjectExpr::Column(i));
                output_types.push(column_type);
            }

            // Add extra projections in the same order as index assignment
            for proj in &extra_projections {
                match proj {
                    ExtraProjection::Property {
                        variable, property, ..
                    } => {
                        let source_col = *variable_columns.get(variable).ok_or_else(|| {
                            Error::Internal(format!(
                                "Variable '{}' not found for property projection",
                                variable
                            ))
                        })?;
                        projections.push(ProjectExpr::PropertyAccess {
                            column: source_col,
                            property: property.clone(),
                        });
                        output_types.push(LogicalType::Any);
                    }
                    ExtraProjection::Expression { filter_expr, .. } => {
                        projections.push(ProjectExpr::Expression {
                            expr: filter_expr.clone(),
                            variable_columns: variable_columns.clone(),
                        });
                        output_types.push(LogicalType::Any);
                    }
                }
            }

            input_op = Box::new(
                ProjectOperator::with_store(
                    input_op,
                    projections,
                    output_types,
                    Arc::clone(&self.store) as Arc<dyn GraphStore>,
                    self.read_view(),
                )
                .with_session_context(self.session_context.clone()),
            );
        }

        // Convert group-by expressions to column indices
        let group_columns: Vec<usize> = agg
            .group_by
            .iter()
            .map(|expr| self.resolve_expression_to_column_with_properties(expr, &variable_columns))
            .collect::<Result<Vec<_>>>()?;

        // Convert aggregate expressions to physical form
        let physical_aggregates: Vec<PhysicalAggregateExpr> = agg
            .aggregates
            .iter()
            .map(|agg_expr| {
                let column = agg_expr
                    .expression
                    .as_ref()
                    .map(|e| {
                        self.resolve_expression_to_column_with_properties(e, &variable_columns)
                    })
                    .transpose()?;

                let column2 = agg_expr
                    .expression2
                    .as_ref()
                    .map(|e| {
                        self.resolve_expression_to_column_with_properties(e, &variable_columns)
                    })
                    .transpose()?;

                Ok(PhysicalAggregateExpr {
                    function: convert_aggregate_function(agg_expr.function),
                    column,
                    column2,
                    distinct: agg_expr.distinct,
                    alias: agg_expr.alias.clone(),
                    percentile: self
                        .constant_percentile(agg_expr.function, agg_expr.percentile.as_ref())?,
                    separator: agg_expr.separator.clone(),
                    keep_nulls: agg_expr.keep_nulls,
                    rdf_literals: false,
                    temporal: self.session_context.temporal_comparison,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Build output schema and column names, and what each column holds: a
        // group key that is a node or an edge (or a list of them) stays one,
        // and so does the list `collect` makes of nodes or edges. Every other
        // column holds values.
        let mut output_schema = Vec::new();
        let mut output_columns = Vec::new();
        let mut output_entities = Vec::new();

        // Add group-by columns: a group key keeps what it holds, also one
        // computed by an expression (`{msg: m}`, `x.msg`, `startNode(r)`).
        for (expr, name) in agg.group_by.iter().zip(key_names) {
            let entity = self.held_entity(expr);
            output_schema.push(entity_type(entity.as_ref()));
            output_columns.push(name);
            output_entities.push(entity);
        }

        // Add aggregate result columns
        for agg_expr in &agg.aggregates {
            let collected = match (agg_expr.function, &agg_expr.expression) {
                // The list of what `collect` gathers keeps its kind: a node or
                // edge column, or an expression that yields one (`head(rs)`,
                // `last(relationships(p))`), or a value with them inside
                // (`collect({msg: m})`, `collect(p)`).
                (LogicalAggregateFunction::Collect, Some(expression)) => {
                    self.held_entity(expression).map(|item| item.list())
                }
                _ => None,
            };
            let result_type = match agg_expr.function {
                LogicalAggregateFunction::Count | LogicalAggregateFunction::CountNonNull => {
                    LogicalType::Int64
                }
                LogicalAggregateFunction::Sum => LogicalType::Any,
                // A float, or a decimal for decimals and integers.
                LogicalAggregateFunction::Avg => LogicalType::Any,
                LogicalAggregateFunction::Min | LogicalAggregateFunction::Max => {
                    // MIN/MAX preserve input type: the result can be any type
                    // (Int64, Float64, String, Date, etc.), so use Any/Generic
                    // to avoid type mismatch when pushing the finalized value.
                    LogicalType::Any
                }
                // A list of nodes or edges, or of any values
                LogicalAggregateFunction::Collect => entity_type(collected.as_ref()),
                LogicalAggregateFunction::GroupConcat => LogicalType::String,
                LogicalAggregateFunction::Sample => LogicalType::Any,
                // Statistical functions return Float64
                LogicalAggregateFunction::StdDev
                | LogicalAggregateFunction::StdDevPop
                | LogicalAggregateFunction::Variance
                | LogicalAggregateFunction::VariancePop
                | LogicalAggregateFunction::PercentileDisc
                | LogicalAggregateFunction::PercentileCont
                | LogicalAggregateFunction::CovarSamp
                | LogicalAggregateFunction::CovarPop
                | LogicalAggregateFunction::Corr
                | LogicalAggregateFunction::RegrSlope
                | LogicalAggregateFunction::RegrIntercept
                | LogicalAggregateFunction::RegrR2
                | LogicalAggregateFunction::RegrSxx
                | LogicalAggregateFunction::RegrSyy
                | LogicalAggregateFunction::RegrSxy
                | LogicalAggregateFunction::RegrAvgx
                | LogicalAggregateFunction::RegrAvgy => LogicalType::Float64,
                // REGR_COUNT returns Int64
                LogicalAggregateFunction::RegrCount => LogicalType::Int64,
            };
            output_schema.push(result_type);
            output_columns.push(
                agg_expr.alias.clone().unwrap_or_else(|| {
                    crate::query::planner::common::aggregate_column_name(agg_expr)
                }),
            );
            output_entities.push(collected);
        }

        for (column, entity) in output_columns.iter().zip(&output_entities) {
            self.set_column_entity(column, entity.clone());
        }

        let key_types = output_schema.clone();
        // Choose operator based on whether there are group-by columns
        let mut operator: Box<dyn Operator> = if group_columns.is_empty() {
            Box::new(
                SimpleAggregateOperator::new(input_op, physical_aggregates, output_schema)
                    .with_temporal(self.session_context.temporal_comparison),
            )
        } else {
            Box::new(
                HashAggregateOperator::new(
                    input_op,
                    group_columns,
                    physical_aggregates,
                    output_schema,
                )
                .with_temporal(self.session_context.temporal_comparison),
            )
        };

        // The groups keyed by a label mask get their labels back, once a
        // group, before HAVING and what follows read them.
        if !mask_keys.is_empty() {
            let output_variables: HashMap<String, usize> = output_columns
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), i))
                .collect();
            let mut projections = Vec::with_capacity(output_columns.len());
            let mut types = Vec::with_capacity(output_columns.len());
            for (i, name) in output_columns.iter().enumerate() {
                if mask_keys.contains(&i) {
                    projections.push(ProjectExpr::Expression {
                        expr: FilterExpression::LabelsOfMask(name.clone()),
                        variable_columns: output_variables.clone(),
                    });
                    types.push(LogicalType::Any);
                } else {
                    projections.push(ProjectExpr::Column(i));
                    types.push(key_types[i].clone());
                }
            }
            operator = Box::new(
                ProjectOperator::with_store(
                    operator,
                    projections,
                    types,
                    Arc::clone(&self.store) as Arc<dyn GraphStore>,
                    self.read_view(),
                )
                .with_session_context(self.session_context.clone()),
            );
        }

        let operator = self.filter_having(operator, &output_columns, agg.having.as_ref())?;
        Ok((operator, output_columns))
    }

    /// The rows of `operator`, an aggregate's, whose columns are
    /// `output_columns`, that `having` keeps: all of them without one.
    fn filter_having(
        &self,
        operator: Box<dyn Operator>,
        output_columns: &[String],
        having: Option<&LogicalExpression>,
    ) -> Result<Box<dyn Operator>> {
        let Some(having) = having else {
            return Ok(operator);
        };
        let having_columns: HashMap<String, usize> = output_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();
        let predicate = ExpressionPredicate::new(
            self.convert_expression(having)?,
            having_columns,
            Arc::clone(&self.store) as Arc<dyn GraphStore>,
            self.read_view(),
        )
        .with_session_context(self.session_context.clone());
        Ok(Box::new(FilterOperator::new(operator, Box::new(predicate))))
    }

    /// Whether every aggregate of `agg` gives on the factorized chain what it
    /// gives on the flat rows: a count of the chain's rows ([`counts_chain_rows`]).
    /// The factorized aggregate reads the chain's last column, which is never
    /// null, so it counts the paths: a count of a variable the chain binds, or
    /// of its start, is that; a count of a variable bound below the start may
    /// skip nulls, and any other aggregate (or `DISTINCT`) reads a column the
    /// chain does not hold, so both run on the flat rows.
    pub(super) fn is_simple_aggregate(agg: &AggregateOp) -> bool {
        let chain = Self::collect_expand_chain(&agg.input);
        let Some(first) = chain.first() else {
            return false;
        };
        agg.aggregates.iter().all(|aggregate| {
            counts_chain_rows(aggregate, &first.from_variable, chain.iter().copied())
        })
    }

    /// The factorized aggregate of `agg` over `base_op`, the plan of [`factorized_base`].
    pub(super) fn factorized_over(
        &self,
        agg: &AggregateOp,
        base_op: Box<dyn Operator>,
        base_columns: Vec<String>,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Build the expand chain - this returns a LazyFactorizedChainOperator
        let expands = Self::collect_expand_chain(&agg.input);
        // The route takes a chain of two or more.
        debug_assert!(expands.len() >= 2, "a factorized aggregate over no chain");
        if expands.len() < 2 {
            return Err(Error::Internal(
                "Expected expand chain for factorized aggregate".to_string(),
            ));
        }
        // The expands run inside the chain, with no operator of their own:
        // PROFILE names each in the aggregate's detail, from the innermost
        // up, as the post-order walk of the plan meets them.
        let mut expand_nodes = Vec::with_capacity(expands.len());
        let mut node = agg.input.as_ref();
        while expand_nodes.len() < expands.len() {
            // `collect_expand_chain` walked these same expands down from the input.
            debug_assert!(
                matches!(node, LogicalOperator::Expand(_)),
                "the factorized chain is not the aggregate's input"
            );
            let LogicalOperator::Expand(expand) = node else {
                return Err(Error::Internal(
                    "the factorized chain is not the aggregate's input".to_string(),
                ));
            };
            expand_nodes.push(node);
            node = &expand.input;
        }
        for expand in expand_nodes.into_iter().rev() {
            self.record_absorbed_scan_entry("Expand", expand);
        }

        let mut columns = base_columns.clone();
        let mut steps = Vec::new();
        let mut is_first = true;

        for expand in &expands {
            // Find source column for this expand
            let source_column = if is_first {
                base_columns
                    .iter()
                    .position(|c| c == &expand.from_variable)
                    .ok_or_else(|| {
                        Error::Internal(format!(
                            "Source variable '{}' not found in input columns",
                            expand.from_variable
                        ))
                    })?
            } else {
                1 // Target from previous level
            };

            let direction = match expand.direction {
                ExpandDirection::Outgoing => Direction::Outgoing,
                ExpandDirection::Incoming => Direction::Incoming,
                ExpandDirection::Both => Direction::Both,
            };

            steps.push(ExpandStep {
                source_column,
                direction,
                edge_types: self.stored_edge_types(&expand.edge_types)?,
            });

            let edge_col_name = self.register_edge_column(&expand.edge_variable);
            columns.push(edge_col_name);
            columns.push(expand.to_variable.clone());

            is_first = false;
        }

        // Create the lazy factorized chain operator
        let lazy_op = LazyFactorizedChainOperator::new(
            Arc::clone(&self.store) as Arc<dyn GraphStore>,
            base_op,
            steps,
            self.read_view(),
        )
        .with_read_counter(self.read_counter());

        // Convert logical aggregates to factorized aggregates
        let factorized_aggs: Vec<FactorizedAggregate> = agg
            .aggregates
            .iter()
            .map(|agg_expr| {
                match agg_expr.function {
                    LogicalAggregateFunction::Count | LogicalAggregateFunction::CountNonNull => {
                        // COUNT(*) uses simple count, COUNT(col) uses column count
                        if agg_expr.expression.is_none() {
                            FactorizedAggregate::count()
                        } else {
                            // For COUNT(variable), we use the deepest level's target column
                            // which is the last column added to the schema
                            FactorizedAggregate::count_column(1) // Target is at index 1 in deepest level
                        }
                    }
                    // Only counts are planned here (see `is_simple_aggregate`)
                    _ => FactorizedAggregate::count(),
                }
            })
            .collect();

        // Build output column names
        let output_columns: Vec<String> = agg
            .aggregates
            .iter()
            .map(|agg_expr| {
                agg_expr.alias.clone().unwrap_or_else(|| {
                    crate::query::planner::common::aggregate_column_name(agg_expr)
                })
            })
            .collect();

        // Register output columns as scalar (aggregate results are materialized
        // scalar values, not entity references). Without this, a post-Return
        // projection would treat them as node IDs and attempt NodeResolve, which
        // corrupts the result on 3+ hop queries.
        for col in &output_columns {
            self.scalar_columns.borrow_mut().insert(col.clone());
        }

        // Create the factorized aggregate operator
        let factorized_agg_op = FactorizedAggregateOperator::new(lazy_op, factorized_aggs);

        Ok((Box::new(factorized_agg_op), output_columns))
    }

    /// Resolves a logical expression to a column index, using projected property columns.
    ///
    /// This is used for aggregations where properties have been projected into their own columns.
    pub(super) fn resolve_expression_to_column_with_properties(
        &self,
        expr: &LogicalExpression,
        variable_columns: &HashMap<String, usize>,
    ) -> Result<usize> {
        crate::query::planner::common::resolve_expression_to_column(expr, variable_columns, "")
    }
}

/// A count operator over its input, with the input's columns.
type PlannedCount = (
    grafeo_core::execution::operators::EdgeCountOperator,
    Vec<String>,
);

/// A grouped count over one expand that its edge counts answer (see
/// `edge_count_over`).
pub(super) struct EdgeCountShape<'a> {
    /// The expands whose paths are counted, the last hop of the path first.
    hops: Vec<CountedExpand<'a>>,
    /// Pairs of positions on the path, the first hop at 0, whose edges must
    /// differ: what Cypher's edge uniqueness asks of hops of one type.
    different: Vec<(usize, usize)>,
}

/// One expand of a counted chain.
pub(super) struct CountedExpand<'a> {
    /// The expand.
    expand: &'a ExpandOp,
    /// The expand as a plan operator.
    node: &'a LogicalOperator,
    /// The filters right above it, from the top down: checks of its far end's
    /// labels, or of edges that must differ.
    checks: Vec<&'a LogicalOperator>,
    /// The labels the far end of each counted edge has.
    end_labels: Vec<String>,
}

impl<'a> EdgeCountShape<'a> {
    /// The shape of `agg` when it counts the paths of a chain of expands per
    /// group, or in all without groups:
    ///
    /// - with every aggregate a `count(*)` or a `count(x)` without
    ///   `DISTINCT` of the chain's start, edges or far ends, which are never
    ///   null on its rows, so each counts the rows;
    /// - over expands of one edge each as a walk or a trail (a one-edge path
    ///   is both; a simple or acyclic one drops a self-loop) without a path
    ///   variable, each from the far end of the one below, whose edge and far
    ///   end nothing below binds (an expand to a bound node checks it is that
    ///   node);
    /// - with nothing between them but checks of labels on a far end and of
    ///   edges that must differ, at most [`MAX_COUNTED_DIFFERENT_HOPS`] hops for
    ///   those. An expand with anything else above it, or that binds what a group
    ///   key or count reads, is part of the chain's input;
    /// - whose group keys are the same for each path of a row (no subquery, no
    ///   volatile call);
    /// - and reads no write (the input writes nothing).
    pub(super) fn of(agg: &'a AggregateOp) -> Option<Self> {
        if agg.input.has_mutations() {
            return None;
        }
        let mut hops: Vec<CountedExpand<'a>> = Vec::new();
        let mut pairs: Vec<(String, String)> = Vec::new();
        let mut below = agg.input.as_ref();
        loop {
            let mut checks = Vec::new();
            let mut at = below;
            while let LogicalOperator::Filter(filter) = at {
                checks.push(at);
                at = &filter.input;
            }
            let LogicalOperator::Expand(expand) = at else {
                break;
            };
            if expand.min_hops != 1
                || expand.max_hops != Some(1)
                || expand.quantified
                || expand.path_alias.is_some()
                || !matches!(expand.path_mode, PathMode::Walk | PathMode::Trail)
                || hops
                    .last()
                    .is_some_and(|upper| upper.expand.from_variable != expand.to_variable)
            {
                break;
            }
            let Some(end_labels) = read_checks(&checks, &expand.to_variable, &mut pairs) else {
                break;
            };
            hops.push(CountedExpand {
                expand,
                node: at,
                checks,
                end_labels,
            });
            below = &expand.input;
        }

        // A group key or count may read what the lower expands bind: they are
        // then the chain's input.
        let mut reads = Vec::new();
        for key in &agg.group_by {
            reads.extend(super::seek::value_variables(key)?);
        }
        let binds = |hop: &CountedExpand<'_>, variable: &str| {
            variable == hop.expand.to_variable
                || hop.expand.edge_variable.as_deref() == Some(variable)
        };
        for index in 0..hops.len() {
            if reads.iter().any(|variable| binds(&hops[index], variable)) {
                hops.truncate(index);
                break;
            }
        }
        let first = hops.last()?;
        let bound = first.expand.input.bound_variables(None)?;
        for hop in &hops {
            if bound.contains(&hop.expand.to_variable)
                || hop
                    .expand
                    .edge_variable
                    .as_ref()
                    .is_some_and(|edge| bound.contains(edge))
            {
                return None;
            }
        }
        let counts_rows = |aggregate: &AggregateExpr| {
            counts_chain_rows(
                aggregate,
                &first.expand.from_variable,
                hops.iter().map(|hop| hop.expand),
            )
        };
        if agg.aggregates.is_empty() || !agg.aggregates.iter().all(counts_rows) {
            return None;
        }

        // Every pair names two edges of this chain.
        let position = |edge: &str| {
            hops.iter()
                .position(|hop| hop.expand.edge_variable.as_deref() == Some(edge))
                .map(|index| hops.len() - 1 - index)
        };
        let mut different = Vec::with_capacity(pairs.len());
        for (left, right) in &pairs {
            different.push((position(left)?, position(right)?));
        }
        if !different.is_empty() && hops.len() > MAX_COUNTED_DIFFERENT_HOPS {
            return None;
        }
        Some(Self { hops, different })
    }
}

/// The count of a whole-graph path of two hops by its middle nodes: a scan of
/// them, each counting its two degrees. Only where nothing but labels narrows
/// the first node, nothing is grouped, and the store keeps what a hop read from
/// its middle needs. Cypher's different edges need both hops to read an edge the
/// same way: then only a self-loop can serve both. The labels of the first node
/// when it applies, `None` otherwise: the chain's walk from each first node holds for
/// every other shape.
pub(super) fn middle_sum_first_labels(
    inputs: RouteInputs,
    agg: &AggregateOp,
    shape: &EdgeCountShape<'_>,
) -> Option<Vec<String>> {
    let [on, near] = shape.hops.as_slice() else {
        return None;
    };
    if !agg.group_by.is_empty() || !inputs.backward_adjacency {
        return None;
    }
    let mut below = near.expand.input.as_ref();
    let mut first_labels: Vec<String> = Vec::new();
    while let LogicalOperator::Filter(filter) = below {
        let mut conjuncts = Vec::new();
        conjuncts_of(&filter.predicate, &mut conjuncts);
        for conjunct in conjuncts {
            let label = super::scan::label_check(conjunct, &near.expand.from_variable)?;
            first_labels.push(label.to_string());
        }
        below = &filter.input;
    }
    let LogicalOperator::NodeScan(scan) = below else {
        return None;
    };
    if scan.input.is_some() || scan.variable != near.expand.from_variable {
        return None;
    }
    first_labels.extend(scan.label.clone());
    let same_way = near.expand.direction == on.expand.direction
        && near.expand.direction != ExpandDirection::Both
        && near.expand.edge_types == on.expand.edge_types;
    if !shape.different.is_empty() && !same_way {
        return None;
    }
    Some(first_labels)
}

/// The operator an edge count of `shape` is built over: the input of its chain's first hop.
///
/// # Errors
///
/// A shape without a hop, which [`EdgeCountShape::of`] never finds.
pub(super) fn edge_count_input<'a>(shape: &EdgeCountShape<'a>) -> Result<&'a LogicalOperator> {
    // `EdgeCountShape::of` finds a shape only with a hop.
    debug_assert!(!shape.hops.is_empty(), "an edge count without a hop");
    shape
        .hops
        .last()
        .map(|first| first.expand.input.as_ref())
        .ok_or_else(|| Error::Internal("an edge count without a hop".to_string()))
}

/// The operator a factorized aggregate of `agg` is built over: the input of its chain's first
/// expand.
///
/// # Errors
///
/// An aggregate over no chain, which the route never takes factorized.
pub(super) fn factorized_base(agg: &AggregateOp) -> Result<&LogicalOperator> {
    super::Planner::collect_expand_chain(&agg.input)
        .into_iter()
        .next()
        .map(|first| first.input.as_ref())
        .ok_or_else(|| {
            Error::Internal("Expected expand chain for factorized aggregate".to_string())
        })
}

/// Whether `aggregate` counts the rows of a chain of `expands` from `start`: a `count(*)`, or a
/// `count(x)` without `DISTINCT` of the chain's start or of an edge or far end of its hops,
/// which are never null on its rows. A count of any other variable, one bound below the chain's
/// start, counts that variable's values, which may be null (an `OPTIONAL MATCH`), so it is not a
/// count of the rows. The edge count and the factorized aggregate both count rows.
fn counts_chain_rows<'e>(
    aggregate: &AggregateExpr,
    start: &str,
    mut expands: impl Iterator<Item = &'e ExpandOp>,
) -> bool {
    !aggregate.distinct
        && aggregate.expression2.is_none()
        && match (aggregate.function, &aggregate.expression) {
            (LogicalAggregateFunction::Count, None) => true,
            (
                LogicalAggregateFunction::Count | LogicalAggregateFunction::CountNonNull,
                Some(LogicalExpression::Variable(variable)),
            ) => {
                variable == start
                    || expands.any(|expand| {
                        expand.to_variable == *variable
                            || expand.edge_variable.as_deref() == Some(variable.as_str())
                    })
            }
            _ => false,
        }
}

/// The labels `checks` require of `far`, adding the pairs of edges they require to
/// differ; `None` when they check anything else.
fn read_checks(
    checks: &[&LogicalOperator],
    far: &str,
    pairs: &mut Vec<(String, String)>,
) -> Option<Vec<String>> {
    let mut end_labels = Vec::new();
    for check in checks {
        let LogicalOperator::Filter(filter) = check else {
            return None;
        };
        let mut conjuncts = Vec::new();
        conjuncts_of(&filter.predicate, &mut conjuncts);
        for conjunct in conjuncts {
            if let Some(label) = super::scan::label_check(conjunct, far) {
                end_labels.push(label.to_string());
            } else {
                pairs.push(edges_that_differ(conjunct)?);
            }
        }
    }
    Some(end_labels)
}

/// The two variables of `id(a) <> id(b)`.
fn edges_that_differ(conjunct: &LogicalExpression) -> Option<(String, String)> {
    let LogicalExpression::Binary {
        left,
        op: crate::query::plan::BinaryOp::Ne,
        right,
    } = conjunct
    else {
        return None;
    };
    let id_of = |side: &LogicalExpression| match side {
        LogicalExpression::Id(variable) => Some(variable.clone()),
        _ => None,
    };
    Some((id_of(left)?, id_of(right)?))
}

/// The conjuncts of an `AND` chain, in order.
fn conjuncts_of<'a>(expression: &'a LogicalExpression, out: &mut Vec<&'a LogicalExpression>) {
    match expression {
        LogicalExpression::Binary {
            left,
            op: crate::query::plan::BinaryOp::And,
            right,
        } => {
            conjuncts_of(left, out);
            conjuncts_of(right, out);
        }
        other => out.push(other),
    }
}

/// The declared type of a column that holds `entity`: a node, an edge, a list
/// of them, a value with them inside, or any value.
fn entity_type(entity: Option<&EntityValue>) -> LogicalType {
    entity.map_or(LogicalType::Any, EntityValue::logical_type)
}

/// The variable of a `labels(n)` group key (the function or the `Labels`
/// expression), which groups by the store's label mask (see
/// `FilterExpression::LabelMask`).
fn labels_of_variable(expression: &LogicalExpression) -> Option<&str> {
    match expression {
        LogicalExpression::Labels(variable) => Some(variable),
        LogicalExpression::FunctionCall {
            callee: crate::query::plan::Callee::Builtin(crate::query::plan::BuiltinFunction::Labels),
            args,
            ..
        } => match args.as_slice() {
            [LogicalExpression::Variable(variable)] => Some(variable),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::Planner;
    use crate::query::plan::{
        AggregateExpr, AggregateFunction, AggregateOp, CreateNodeOp, ExpandDirection, ExpandOp,
        LogicalExpression, LogicalOperator, LogicalPlan, NodeScanOp, PathMode, ProjectOp,
        Projection,
    };
    use crate::transaction::TransactionManager;
    use grafeo_core::graph::GraphStore;
    use grafeo_core::graph::rowgroup::RowGroupStore;
    use std::sync::Arc;

    /// A one-hop expand from `from` to `to` along `edge_type` over `input`.
    fn expand(from: &str, edge_type: &str, to: &str, input: LogicalOperator) -> LogicalOperator {
        LogicalOperator::Expand(ExpandOp {
            from_variable: from.to_string(),
            to_variable: to.to_string(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_types: vec![edge_type.to_string()],
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(input),
            path_alias: None,
            path_mode: PathMode::Walk,
            quantified: false,
        })
    }

    /// `CREATE (h:Hub) WITH h MATCH (h)-[:R]->(q)-[:S]->(t) RETURN count(*)`
    /// runs factorized: the chain reads its whole source (the node scan of
    /// the bound `h`, which reads its writing input whole, see
    /// `plan_node_scan`) before its first expand, so the count reads after
    /// the whole write without a step of its own (the queries in
    /// `tests/pattern_after_write.rs` check the count).
    #[test]
    fn a_count_over_a_chain_from_a_written_node_is_planned_factorized() {
        let store = Arc::new(RowGroupStore::new());
        let transaction_manager = Arc::new(TransactionManager::new());
        let transaction_id = transaction_manager.begin();
        let epoch = transaction_manager.current_epoch();
        // The transaction's writers record in its change set.
        let recording = transaction_manager
            .changes(transaction_id)
            .unwrap()
            .recording(
                &transaction_manager,
                None,
                Arc::clone(&store) as Arc<dyn grafeo_core::graph::apply::GraphBackend>,
            )
            .unwrap();
        let planner = Planner::with_context(
            Arc::clone(&store) as Arc<dyn GraphStore>,
            Some(Arc::clone(&store) as Arc<dyn grafeo_core::graph::GraphBackend>),
            Arc::clone(&transaction_manager),
            Some(transaction_id),
            epoch,
        )
        .with_recording(Some(recording));
        let write = LogicalOperator::Project(ProjectOp {
            projections: vec![Projection {
                expression: LogicalExpression::Variable("h".to_string()),
                alias: None,
            }],
            input: Box::new(LogicalOperator::CreateNode(CreateNodeOp {
                variable: "h".to_string(),
                labels: vec!["Hub".to_string()],
                properties: Vec::new(),
                input: None,
            })),
            pass_through_input: false,
        });
        let scan = LogicalOperator::NodeScan(NodeScanOp {
            variable: "h".to_string(),
            label: None,
            input: Some(Box::new(write)),
        });
        let count = LogicalOperator::Aggregate(AggregateOp {
            group_by: Vec::new(),
            aggregates: vec![AggregateExpr {
                function: AggregateFunction::Count,
                expression: None,
                expression2: None,
                distinct: false,
                alias: Some("found".to_string()),
                percentile: None,
                separator: None,
                keep_nulls: false,
            }],
            input: Box::new(expand("q", "S", "t", expand("h", "R", "q", scan))),
            having: None,
        });

        let physical = planner.plan(&LogicalPlan::new(count)).unwrap();

        assert_eq!(physical.operator.name(), "FactorizedAggregate");
        assert_eq!(physical.columns, ["found"]);
    }
}
