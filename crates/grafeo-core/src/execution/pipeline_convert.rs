//! Converts pull-based operator trees into push-based pipelines.
//!
//! The converter walks the operator tree top-down, decomposing operators that
//! have push equivalents. Source operators (scan, expand, join) stay pull-based
//! and get wrapped in [`OperatorSource`](super::source::OperatorSource).
//!
//! This enables the documented push-based execution model without modifying
//! the planner, which continues to emit pull-based operator trees.

use super::chunk::DataChunk;
use super::operators::push::FilterPredicate;
use super::operators::{
    AggregatePushOperator, DistinctPushOperator, FilterPushOperator, LimitPushOperator,
    SortPushOperator,
};
use super::operators::{
    DistinctOperator, FilterOperator, HashAggregateOperator, LimitOperator, Operator, Predicate,
    SortOperator,
};
use super::pipeline::PushOperator;

// -------------------------------------------------------------------------
// Type adapters (bridge pull types to push types)
// -------------------------------------------------------------------------

/// Adapts a pull-based [`Predicate`] to the push [`FilterPredicate`] trait.
pub struct PredicateAdapter(pub Box<dyn Predicate>);

impl FilterPredicate for PredicateAdapter {
    fn evaluate(&self, chunk: &DataChunk, row: usize) -> bool {
        self.0.evaluate(chunk, row)
    }
}

// NOTE: ProjectExprAdapter and is_simple_project are intentionally omitted.
// ProjectOperator carries store references, transaction context, and session
// context that cannot be transferred to push operators. Project stays pull-based.
// When a dedicated PushProjectOperator with store access is added, revisit this.

/// Converts a pull-based sort key to the push-based equivalent.
///
/// Both types have identical fields but are separate types in separate modules.
fn convert_sort_key(pull: &super::operators::SortKey) -> super::operators::push::SortKey {
    use super::operators::{NullOrder, SortDirection};
    super::operators::push::SortKey {
        column: pull.column,
        direction: match pull.direction {
            SortDirection::Ascending => super::operators::push::SortDirection::Ascending,
            SortDirection::Descending => super::operators::push::SortDirection::Descending,
        },
        null_order: match pull.null_order {
            NullOrder::NullsFirst => super::operators::push::NullOrder::First,
            NullOrder::NullsLast => super::operators::push::NullOrder::Last,
        },
    }
}

// -------------------------------------------------------------------------
// Pipeline converter
// -------------------------------------------------------------------------

/// Converts a pull-based operator tree into a source operator and a chain of push operators.
///
/// Walks the tree from the root, decomposing operators that have push equivalents
/// (Filter, Sort, Aggregate, Limit, Distinct). Stops at source operators
/// (scan, expand, join, etc.) which stay pull-based.
///
/// Returns `(source, push_ops)` where:
/// - `source` is the deepest non-convertible operator (pull-based)
/// - `push_ops` is the chain of push operators in pipeline order (source-first)
///
/// If the root operator has no push equivalent (e.g., a bare scan), returns
/// an empty `push_ops` vec.
pub fn convert_to_pipeline(
    root: Box<dyn Operator>,
) -> (Box<dyn Operator>, Vec<Box<dyn PushOperator>>) {
    let mut push_ops: Vec<Box<dyn PushOperator>> = Vec::new();
    let source = decompose_recursive(root, &mut push_ops);
    // Push ops are collected root-first (outermost first), reverse for pipeline order
    push_ops.reverse();
    (source, push_ops)
}

/// Converts a pull-based operator tree into a push pipeline with memory-aware spilling.
///
/// When `memory_ctx` is `Some`, Sort and Aggregate operators are created as their
/// spillable variants that register with the `BufferManager` and spill based on
/// system memory pressure. When `memory_ctx` is `None`, delegates to
/// [`convert_to_pipeline`] (non-spillable operators).
#[cfg(feature = "spill")]
pub fn convert_to_pipeline_with_memory(
    root: Box<dyn Operator>,
    memory_ctx: Option<super::memory::OperatorMemoryContext>,
) -> (Box<dyn Operator>, Vec<Box<dyn PushOperator>>) {
    let Some(ctx) = memory_ctx else {
        return convert_to_pipeline(root);
    };
    let mut push_ops: Vec<Box<dyn PushOperator>> = Vec::new();
    let source = decompose_recursive_memory(root, &mut push_ops, &ctx);
    push_ops.reverse();
    (source, push_ops)
}

/// Recursively decomposes operators with memory-aware spillable variants.
#[cfg(feature = "spill")]
fn decompose_recursive_memory(
    op: Box<dyn Operator>,
    push_ops: &mut Vec<Box<dyn PushOperator>>,
    ctx: &super::memory::OperatorMemoryContext,
) -> Box<dyn Operator> {
    use super::operators::{SpillableAggregatePushOperator, SpillableSortPushOperator};

    match op.name() {
        "Filter" => {
            let any = op.into_any();
            let filter = any
                .downcast::<FilterOperator>()
                .expect("name() returned 'Filter' but downcast failed");
            let (child, predicate) = filter.into_parts();
            push_ops.push(Box::new(FilterPushOperator::new(Box::new(
                PredicateAdapter(predicate),
            ))));
            decompose_recursive_memory(child, push_ops, ctx)
        }
        "Sort" => {
            let any = op.into_any();
            let sort = any
                .downcast::<SortOperator>()
                .expect("name() returned 'Sort' but downcast failed");
            if sort.drops_columns() {
                // The push-based sort returns every column.
                return sort;
            }
            let (child, sort_keys) = sort.into_parts();
            let push_keys: Vec<_> = sort_keys.iter().map(convert_sort_key).collect();
            push_ops.push(Box::new(SpillableSortPushOperator::with_memory_context(
                push_keys,
                ctx.clone(),
            )));
            decompose_recursive_memory(child, push_ops, ctx)
        }
        "HashAggregate" => {
            let any = op.into_any();
            let agg = any
                .downcast::<HashAggregateOperator>()
                .expect("name() returned 'HashAggregate' but downcast failed");
            let (child, group_columns, aggregates) = agg.into_parts();
            push_ops.push(Box::new(
                SpillableAggregatePushOperator::with_memory_context(
                    group_columns,
                    aggregates,
                    ctx.clone(),
                ),
            ));
            decompose_recursive_memory(child, push_ops, ctx)
        }
        "Limit" => {
            let any = op.into_any();
            let limit = any
                .downcast::<LimitOperator>()
                .expect("name() returned 'Limit' but downcast failed");
            let (child, count) = limit.into_parts();
            push_ops.push(Box::new(LimitPushOperator::new(count)));
            decompose_recursive_memory(child, push_ops, ctx)
        }
        "Distinct" => {
            let any = op.into_any();
            let distinct = any
                .downcast::<DistinctOperator>()
                .expect("name() returned 'Distinct' but downcast failed");
            let (child, columns) = distinct.into_parts();
            let push_distinct = if let Some(cols) = columns {
                DistinctPushOperator::on_columns(cols)
            } else {
                DistinctPushOperator::new()
            };
            push_ops.push(Box::new(push_distinct));
            decompose_recursive_memory(child, push_ops, ctx)
        }
        _ => op,
    }
}

/// Recursively decomposes operators, collecting push equivalents.
///
/// Uses `name()` to identify the operator type, then `into_any()` + downcast
/// to decompose it into child + push operator.
fn decompose_recursive(
    op: Box<dyn Operator>,
    push_ops: &mut Vec<Box<dyn PushOperator>>,
) -> Box<dyn Operator> {
    match op.name() {
        "Filter" => {
            let any = op.into_any();
            let filter = any
                .downcast::<FilterOperator>()
                .expect("name() returned 'Filter' but downcast failed");
            let (child, predicate) = filter.into_parts();
            push_ops.push(Box::new(FilterPushOperator::new(Box::new(
                PredicateAdapter(predicate),
            ))));
            decompose_recursive(child, push_ops)
        }
        // Project is NOT decomposed because it often holds store references,
        // transaction context, and session context that cannot be transferred
        // to push operators. Treat as a source operator boundary.
        //
        // TODO: when a dedicated PushProjectOperator with store access exists,
        // revisit this decision.
        "Sort" => {
            let any = op.into_any();
            let sort = any
                .downcast::<SortOperator>()
                .expect("name() returned 'Sort' but downcast failed");
            if sort.drops_columns() {
                // The push-based sort returns every column.
                return sort;
            }
            let (child, sort_keys) = sort.into_parts();
            let push_keys: Vec<_> = sort_keys.iter().map(convert_sort_key).collect();
            push_ops.push(Box::new(SortPushOperator::new(push_keys)));
            decompose_recursive(child, push_ops)
        }
        "HashAggregate" => {
            let any = op.into_any();
            let agg = any
                .downcast::<HashAggregateOperator>()
                .expect("name() returned 'HashAggregate' but downcast failed");
            let (child, group_columns, aggregates) = agg.into_parts();
            push_ops.push(Box::new(AggregatePushOperator::new(
                group_columns,
                aggregates,
            )));
            decompose_recursive(child, push_ops)
        }
        "Limit" => {
            let any = op.into_any();
            let limit = any
                .downcast::<LimitOperator>()
                .expect("name() returned 'Limit' but downcast failed");
            let (child, count) = limit.into_parts();
            push_ops.push(Box::new(LimitPushOperator::new(count)));
            decompose_recursive(child, push_ops)
        }
        "Distinct" => {
            let any = op.into_any();
            let distinct = any
                .downcast::<DistinctOperator>()
                .expect("name() returned 'Distinct' but downcast failed");
            let (child, columns) = distinct.into_parts();
            let push_distinct = if let Some(cols) = columns {
                DistinctPushOperator::on_columns(cols)
            } else {
                DistinctPushOperator::new()
            };
            push_ops.push(Box::new(push_distinct));
            decompose_recursive(child, push_ops)
        }
        // Not convertible: this is a source operator (scan, expand, join, etc.)
        _ => op,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::operators::{OperatorResult, SortKey};
    use grafeo_common::types::LogicalType;

    /// A trivial predicate that always returns true (for testing decomposition only).
    struct AlwaysTruePredicate;

    impl Predicate for AlwaysTruePredicate {
        fn evaluate(&self, _chunk: &DataChunk, _row: usize) -> bool {
            true
        }
    }

    /// A minimal test operator that produces one chunk.
    struct TestScanOperator {
        emitted: bool,
    }

    impl TestScanOperator {
        fn new() -> Self {
            Self { emitted: false }
        }
    }

    impl Operator for TestScanOperator {
        fn next(&mut self) -> OperatorResult {
            if self.emitted {
                return Ok(None);
            }
            self.emitted = true;
            let mut col = crate::execution::vector::ValueVector::with_type(LogicalType::Int64);
            col.push_int64(1);
            col.push_int64(2);
            col.push_int64(3);
            Ok(Some(DataChunk::new(vec![col])))
        }

        fn reset(&mut self) {
            self.emitted = false;
        }

        fn name(&self) -> &'static str {
            "TestScan"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    #[test]
    fn convert_bare_scan_produces_empty_pipeline() {
        let scan: Box<dyn Operator> = Box::new(TestScanOperator::new());
        let (source, push_ops) = convert_to_pipeline(scan);
        assert!(push_ops.is_empty());
        assert_eq!(source.name(), "TestScan");
    }

    #[test]
    fn convert_filter_scan_produces_one_push_op() {
        let scan: Box<dyn Operator> = Box::new(TestScanOperator::new());
        let predicate: Box<dyn Predicate> = Box::new(AlwaysTruePredicate);
        let filter: Box<dyn Operator> = Box::new(FilterOperator::new(scan, predicate));

        let (source, push_ops) = convert_to_pipeline(filter);
        assert_eq!(source.name(), "TestScan");
        assert_eq!(push_ops.len(), 1);
        assert_eq!(push_ops.len(), 1);
        // Push operators have their own naming convention
        assert!(
            push_ops[0].name().contains("Filter"),
            "expected filter push op, got {}",
            push_ops[0].name()
        );
    }

    #[test]
    fn convert_limit_filter_scan_produces_two_push_ops() {
        let scan: Box<dyn Operator> = Box::new(TestScanOperator::new());
        let predicate: Box<dyn Predicate> = Box::new(AlwaysTruePredicate);
        let filter: Box<dyn Operator> = Box::new(FilterOperator::new(scan, predicate));
        let limit: Box<dyn Operator> = Box::new(LimitOperator::new(filter, 10));

        let (source, push_ops) = convert_to_pipeline(limit);
        assert_eq!(source.name(), "TestScan");
        assert_eq!(push_ops.len(), 2);
        // Pipeline order: filter first, then limit
        assert!(push_ops[0].name().contains("Filter"));
        assert!(push_ops[1].name().contains("Limit"));
    }

    #[test]
    fn convert_sort_scan_produces_one_push_op() {
        let scan: Box<dyn Operator> = Box::new(TestScanOperator::new());
        let keys = vec![SortKey::ascending(0)];
        let sort: Box<dyn Operator> = Box::new(SortOperator::new(scan, keys));

        let (source, push_ops) = convert_to_pipeline(sort);
        assert_eq!(source.name(), "TestScan");
        assert_eq!(push_ops.len(), 1);
        assert!(push_ops[0].name().contains("Sort"));
    }

    #[test]
    fn convert_aggregate_scan_produces_one_push_op() {
        use crate::execution::operators::{AggregateExpr, AggregateFunction};

        let scan: Box<dyn Operator> = Box::new(TestScanOperator::new());
        let aggregates = vec![AggregateExpr {
            function: AggregateFunction::Count,
            column: None,
            column2: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
        }];
        let agg: Box<dyn Operator> = Box::new(HashAggregateOperator::new(
            scan,
            vec![],
            aggregates,
            vec![LogicalType::Int64],
        ));

        let (source, push_ops) = convert_to_pipeline(agg);
        assert_eq!(source.name(), "TestScan");
        assert_eq!(push_ops.len(), 1);
        assert!(push_ops[0].name().contains("Aggregate"));
    }

    #[test]
    fn convert_distinct_scan_produces_one_push_op() {
        let scan: Box<dyn Operator> = Box::new(TestScanOperator::new());
        let distinct: Box<dyn Operator> = Box::new(DistinctOperator::new(scan));

        let (source, push_ops) = convert_to_pipeline(distinct);
        assert_eq!(source.name(), "TestScan");
        assert_eq!(push_ops.len(), 1);
        assert!(push_ops[0].name().contains("Distinct"));
    }

    #[test]
    fn convert_distinct_on_columns_scan() {
        let scan: Box<dyn Operator> = Box::new(TestScanOperator::new());
        let distinct: Box<dyn Operator> = Box::new(DistinctOperator::on_columns(scan, vec![0]));

        let (source, push_ops) = convert_to_pipeline(distinct);
        assert_eq!(source.name(), "TestScan");
        assert_eq!(push_ops.len(), 1);
        assert!(push_ops[0].name().contains("Distinct"));
    }

    #[test]
    fn convert_deep_pipeline_sort_filter_limit() {
        let scan: Box<dyn Operator> = Box::new(TestScanOperator::new());
        let predicate: Box<dyn Predicate> = Box::new(AlwaysTruePredicate);
        let filter: Box<dyn Operator> = Box::new(FilterOperator::new(scan, predicate));
        let keys = vec![SortKey::ascending(0)];
        let sort: Box<dyn Operator> = Box::new(SortOperator::new(filter, keys));
        let limit: Box<dyn Operator> = Box::new(LimitOperator::new(sort, 5));

        let (source, push_ops) = convert_to_pipeline(limit);
        assert_eq!(source.name(), "TestScan");
        assert_eq!(push_ops.len(), 3);
        // Pipeline order: filter, sort, limit (source-first)
        assert!(push_ops[0].name().contains("Filter"));
        assert!(push_ops[1].name().contains("Sort"));
        assert!(push_ops[2].name().contains("Limit"));
    }

    #[test]
    fn pipeline_roundtrip_produces_correct_results() {
        use crate::execution::pipeline::Pipeline;
        use crate::execution::sink::CollectorSink;
        use crate::execution::source::OperatorSource;

        // Build: Scan -> Filter(always true) -> Sort(col 0 ASC)
        let scan: Box<dyn Operator> = Box::new(TestScanOperator::new());
        let predicate: Box<dyn Predicate> = Box::new(AlwaysTruePredicate);
        let filter: Box<dyn Operator> = Box::new(FilterOperator::new(scan, predicate));
        let keys = vec![SortKey::ascending(0)];
        let sort: Box<dyn Operator> = Box::new(SortOperator::new(filter, keys));

        // Convert to pipeline
        let (source, push_ops) = convert_to_pipeline(sort);
        assert_eq!(push_ops.len(), 2); // Filter + Sort

        // Execute the pipeline
        let source = Box::new(OperatorSource::new(source));
        let collector = CollectorSink::new();
        let mut pipeline = Pipeline::new(source, push_ops, Box::new(collector));
        pipeline.execute().unwrap();

        // Extract results
        let sink_box = pipeline.into_sink();
        let any_sink: Box<dyn std::any::Any> = sink_box.into_any();
        let collector = any_sink.downcast::<CollectorSink>().unwrap();
        assert_eq!(collector.row_count(), 3);
    }

    #[test]
    fn predicate_adapter_delegates_correctly() {
        let mut col = crate::execution::vector::ValueVector::with_type(LogicalType::Int64);
        col.push_int64(42);
        let chunk = DataChunk::new(vec![col]);

        let adapter = PredicateAdapter(Box::new(AlwaysTruePredicate));
        assert!(adapter.evaluate(&chunk, 0));
    }

    #[test]
    fn convert_sort_key_maps_directions() {
        use crate::execution::operators::{NullOrder, SortDirection};

        use crate::execution::operators::push::{
            NullOrder as PushNullOrder, SortDirection as PushSortDirection,
        };

        let asc = super::convert_sort_key(&SortKey {
            column: 3,
            direction: SortDirection::Ascending,
            null_order: NullOrder::NullsFirst,
        });
        assert_eq!(asc.column, 3);
        assert_eq!(asc.direction, PushSortDirection::Ascending);
        assert_eq!(asc.null_order, PushNullOrder::First);

        let desc = super::convert_sort_key(&SortKey {
            column: 7,
            direction: SortDirection::Descending,
            null_order: NullOrder::NullsLast,
        });
        assert_eq!(desc.column, 7);
        assert_eq!(desc.direction, PushSortDirection::Descending);
        assert_eq!(desc.null_order, PushNullOrder::Last);
    }

    #[test]
    fn test_distinct_on_columns_pipeline_execution() {
        use crate::execution::pipeline::Pipeline;
        use crate::execution::sink::CollectorSink;

        // Build: Scan -> Distinct(on column 0)
        let scan: Box<dyn Operator> = Box::new(TestScanOperator::new());
        let distinct: Box<dyn Operator> = Box::new(DistinctOperator::on_columns(scan, vec![0]));

        let (source, push_ops) = convert_to_pipeline(distinct);
        assert_eq!(push_ops.len(), 1);
        assert!(push_ops[0].name().contains("Distinct"));

        // Execute the pipeline and verify results
        let source = Box::new(crate::execution::source::OperatorSource::new(source));
        let collector = CollectorSink::new();
        let mut pipeline = Pipeline::new(source, push_ops, Box::new(collector));
        pipeline.execute().unwrap();

        let sink_box = pipeline.into_sink();
        let any_sink: Box<dyn std::any::Any> = sink_box.into_any();
        let collector = any_sink.downcast::<CollectorSink>().unwrap();
        // TestScan produces [1, 2, 3], all distinct, so 3 rows
        assert_eq!(collector.row_count(), 3);
    }

    #[test]
    fn test_unrecognized_operator_stays_as_source() {
        /// A custom operator with an unrecognized name.
        struct CustomJoinOperator;

        impl Operator for CustomJoinOperator {
            fn next(&mut self) -> OperatorResult {
                Ok(None)
            }

            fn reset(&mut self) {}

            fn name(&self) -> &'static str {
                "CustomNestedLoopJoin"
            }

            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
                self
            }
        }

        let join: Box<dyn Operator> = Box::new(CustomJoinOperator);
        let (source, push_ops) = convert_to_pipeline(join);
        assert_eq!(source.name(), "CustomNestedLoopJoin");
        assert!(
            push_ops.is_empty(),
            "unrecognized operator should produce no push ops"
        );
    }
}
