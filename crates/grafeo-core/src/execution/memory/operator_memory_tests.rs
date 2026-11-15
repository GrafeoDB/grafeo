//! One statement's operators share one memory cap (#674): every operator that
//! materializes its input takes grants against it, a chunk or a table resize
//! at a time, and one that cannot spill fails with `GRAFEO-S001` naming the
//! bound that refused. Consumers of concurrent operators stay their own (C18).

use std::sync::Arc;

use grafeo_common::memory::buffer::{BufferManager, MemoryRegion};
use grafeo_common::types::{LogicalType, Value};
use grafeo_common::utils::error::LimitBound;

use crate::execution::chunk::DataChunk;
use crate::execution::memory::{self, OperatorMemory, OperatorMemoryContext, StatementMemory};
use crate::execution::operators::push::SortKey as PushSortKey;
use crate::execution::operators::{
    AggregateExpr, AggregatePushOperator, ApplyOperator, DelimBuffer, DelimReplayOperator,
    DistinctMaterializingOperator, DistinctOperator, DistinctPushOperator, EagerOperator,
    ExceptOperator, HashAggregateOperator, HashJoinOperator, IntersectOperator, JoinType,
    MapCollectOperator, NestedLoopJoinOperator, Operator, OperatorError, OperatorResult,
    ShuffleOperator, SimpleAggregateOperator, SortKey, SortOperator,
    SpillableAggregatePushOperator, SpillableSortPushOperator, TopKOperator,
};
use crate::execution::pipeline::PushOperator;
use crate::execution::sink::CollectorSink;
use crate::execution::vector::ValueVector;

const KIB: usize = 1 << 10;
const MIB: usize = 1 << 20;

/// `chunks` chunks of 1024 rows, `columns` Int64 columns each, counting up.
struct Numbers {
    next: i64,
    left: usize,
    columns: usize,
}

impl Numbers {
    fn boxed(chunks: usize, columns: usize) -> Box<dyn Operator> {
        Box::new(Self {
            next: 0,
            left: chunks,
            columns,
        })
    }
}

impl Operator for Numbers {
    fn next(&mut self) -> OperatorResult {
        if self.left == 0 {
            return Ok(None);
        }
        self.left -= 1;
        let values: Vec<Value> = (self.next..self.next + 1024).map(Value::Int64).collect();
        self.next += 1024;
        let columns = (0..self.columns)
            .map(|_| ValueVector::from_values(&values))
            .collect();
        Ok(Some(DataChunk::new(columns)))
    }

    fn reset(&mut self) {}

    fn name(&self) -> &'static str {
        "Numbers"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// A statement's context with a large budget and a `limit` byte cap.
fn statement(limit: usize) -> OperatorMemoryContext {
    OperatorMemoryContext::new(BufferManager::with_budget(256 * MIB), None).with_query_limit(limit)
}

/// `operator`, built inside the statement scope of `ctx`.
fn built_in<T>(ctx: &OperatorMemoryContext, build: impl FnOnce() -> T) -> T {
    let _scope = memory::enter(StatementMemory::Budgeted(ctx.clone()));
    build()
}

/// Runs `operator` to its end.
fn drain(operator: &mut dyn Operator) -> Result<usize, OperatorError> {
    let mut rows = 0;
    while let Some(chunk) = operator.next()? {
        rows += chunk.row_count();
    }
    Ok(rows)
}

/// `result` is the S001 refusal of `bound`, naming it.
fn assert_refused<T: std::fmt::Debug>(result: Result<T, OperatorError>, bound: LimitBound) {
    let error = result.expect_err("refused at the bound");
    let OperatorError::Wrapped(inner) = &error else {
        panic!("not a refusal: {error}");
    };
    let text = error.to_string();
    assert_eq!(inner.error_code().as_str(), "GRAFEO-S001", "{text}");
    assert_eq!(
        inner.limit_refusal().map(|refusal| refusal.bound()),
        Some(bound),
        "{text}"
    );
    assert!(
        text.contains(bound.name()),
        "names the {}: {text}",
        bound.name()
    );
}

/// Every pull operator that materializes its input and cannot spill: built
/// in a statement capped at 64 KiB it fails naming the query memory limit;
/// at a large cap it runs, and gives back all it held when it drops.
#[test]
fn every_materializing_pull_operator_refuses_at_the_query_limit() {
    type Build = fn() -> Box<dyn Operator>;
    let operators: [(&str, Build); 16] = [
        ("sort", || {
            Box::new(SortOperator::new(
                Numbers::boxed(20, 1),
                vec![SortKey::ascending(0)],
            ))
        }),
        ("hash aggregate", || {
            Box::new(HashAggregateOperator::new(
                Numbers::boxed(20, 1),
                vec![0],
                vec![AggregateExpr::count_star()],
                vec![LogicalType::Int64, LogicalType::Int64],
            ))
        }),
        ("hash join", || {
            Box::new(HashJoinOperator::new(
                Numbers::boxed(1, 1),
                Numbers::boxed(20, 1),
                vec![0],
                vec![0],
                JoinType::Inner,
                vec![LogicalType::Int64, LogicalType::Int64],
            ))
        }),
        ("distinct", || {
            Box::new(DistinctOperator::new(Numbers::boxed(20, 1)))
        }),
        ("eager", || {
            Box::new(EagerOperator::new(Numbers::boxed(20, 1)))
        }),
        ("map collect", || {
            Box::new(MapCollectOperator::new(Numbers::boxed(20, 2), 0, 1))
        }),
        ("except", || {
            Box::new(ExceptOperator::new(
                Numbers::boxed(20, 1),
                Numbers::boxed(1, 1),
                false,
            ))
        }),
        ("intersect", || {
            Box::new(IntersectOperator::new(
                Numbers::boxed(20, 1),
                Numbers::boxed(1, 1),
                false,
            ))
        }),
        ("shuffle", || {
            Box::new(ShuffleOperator::new(Numbers::boxed(20, 1)))
        }),
        ("delim", || {
            Box::new(DelimReplayOperator::new(DelimBuffer::new(
                Numbers::boxed(20, 1),
                vec![0],
            )))
        }),
        ("hash join reading its probe side first", || {
            Box::new(
                HashJoinOperator::new(
                    Numbers::boxed(20, 1),
                    Numbers::boxed(1, 1),
                    vec![0],
                    vec![0],
                    JoinType::Inner,
                    vec![LogicalType::Int64, LogicalType::Int64],
                )
                .with_probe_first(),
            )
        }),
        ("nested loop join", || {
            Box::new(NestedLoopJoinOperator::new(
                Numbers::boxed(1, 1),
                Numbers::boxed(20, 1),
                None,
                JoinType::Inner,
                vec![LogicalType::Int64, LogicalType::Int64],
            ))
        }),
        ("top k", || {
            Box::new(TopKOperator::new(
                Numbers::boxed(20, 1),
                vec![SortKey::ascending(0)],
                100_000,
            ))
        }),
        ("apply joining one outer row", || {
            Box::new(ApplyOperator::new(
                Numbers::boxed(1, 1),
                Numbers::boxed(20, 1),
            ))
        }),
        ("count distinct", || {
            Box::new(SimpleAggregateOperator::new(
                Numbers::boxed(20, 1),
                vec![count_distinct(0)],
                vec![LogicalType::Int64],
            ))
        }),
        ("apply reading its outer input first", || {
            Box::new(
                ApplyOperator::new(Numbers::boxed(20, 1), Numbers::boxed(1, 1)).with_outer_first(),
            )
        }),
    ];
    for (name, build) in operators {
        let small = statement(64 * KIB);
        let mut operator = built_in(&small, build);
        let result = drain(operator.as_mut());
        assert!(result.is_err(), "{name}: refused");
        assert_refused(result, LimitBound::QueryMemoryLimit);
        drop(operator);
        assert_eq!(small.query().used(), 0, "{name}: gave back what it held");

        let large = statement(64 * MIB);
        let mut operator = built_in(&large, build);
        assert!(drain(operator.as_mut()).unwrap() > 0, "{name}: ran");
        assert!(large.query().used() > 0, "{name}: counted what it held");
        drop(operator);
        assert_eq!(large.query().used(), 0, "{name}: gave back what it held");
        assert_eq!(large.buffer_manager().stats().total_allocated, 0, "{name}");
    }
}

/// `COUNT(DISTINCT column)`.
fn count_distinct(column: usize) -> AggregateExpr {
    let mut expr = AggregateExpr::count(column);
    expr.distinct = true;
    expr
}

/// A push aggregate of `COUNT(DISTINCT)` in the statement of `ctx`: the one
/// without spilling, or the spillable one (without a spill path here, so it
/// has nowhere to spill).
fn push_count_distinct(ctx: &OperatorMemoryContext, spillable: bool) -> Box<dyn PushOperator> {
    if spillable {
        Box::new(SpillableAggregatePushOperator::with_memory_context(
            Vec::new(),
            vec![count_distinct(0)],
            ctx.clone(),
        ))
    } else {
        Box::new(built_in(ctx, || {
            AggregatePushOperator::global(vec![count_distinct(0)])
        }))
    }
}

/// The push aggregates count their groups' DISTINCT sets: past a small cap
/// they fail naming the query memory limit, under a large one they run, and
/// they give back what they held when they drop.
#[test]
fn the_push_aggregates_count_their_distinct_sets() {
    for spillable in [false, true] {
        let small = statement(64 * KIB);
        let mut aggregate = push_count_distinct(&small, spillable);
        let mut sink = CollectorSink::new();
        let mut source = Numbers::boxed(20, 1);
        let mut result = Ok(true);
        while let Some(chunk) = source.next().unwrap() {
            result = aggregate.push(chunk, &mut sink);
            if result.is_err() {
                break;
            }
        }
        assert_refused(result, LimitBound::QueryMemoryLimit);
        drop(aggregate);
        assert_eq!(
            small.query().used(),
            0,
            "spillable {spillable}: gave it back"
        );

        let large = statement(64 * MIB);
        let mut aggregate = push_count_distinct(&large, spillable);
        let mut source = Numbers::boxed(20, 1);
        while let Some(chunk) = source.next().unwrap() {
            aggregate.push(chunk, &mut sink).unwrap();
        }
        assert!(large.query().used() > 0, "spillable {spillable}: counted");
        aggregate.finalize(&mut sink).unwrap();
        drop(aggregate);
        assert_eq!(
            large.query().used(),
            0,
            "spillable {spillable}: gave it back"
        );
    }
}

/// Built outside a budgeted statement (a unit test, an `EXPLAIN`) an
/// operator takes no grants.
#[test]
fn an_operator_built_outside_a_budgeted_statement_counts_nothing() {
    let mut outside = SortOperator::new(Numbers::boxed(20, 1), vec![SortKey::ascending(0)]);
    assert_eq!(drain(&mut outside).unwrap(), 20 * 1024);
    let mut unbudgeted = {
        let _scope = memory::enter(StatementMemory::Unbudgeted);
        assert!(memory::in_scope());
        assert!(memory::current().is_none());
        EagerOperator::new(Numbers::boxed(20, 1))
    };
    assert!(!memory::in_scope());
    assert_eq!(drain(&mut unbudgeted).unwrap(), 20 * 1024);
}

/// The push distinct counts its seen keys as the pull one does.
#[test]
fn the_push_distinct_refuses_at_the_query_limit() {
    let ctx = statement(64 * KIB);
    let mut distinct = built_in(&ctx, DistinctPushOperator::new);
    let mut sink = CollectorSink::new();
    let mut source = Numbers::boxed(20, 1);
    let mut result = Ok(true);
    while let Some(chunk) = source.next().unwrap() {
        result = distinct.push(chunk, &mut sink);
        if result.is_err() {
            break;
        }
    }
    assert_refused(result, LimitBound::QueryMemoryLimit);
}

/// The materializing push distinct counts its rows and keys.
#[test]
fn the_materializing_push_distinct_refuses_at_the_query_limit() {
    let ctx = statement(64 * KIB);
    let mut distinct = built_in(&ctx, DistinctMaterializingOperator::new);
    let mut sink = CollectorSink::new();
    let mut source = Numbers::boxed(20, 1);
    let mut result = Ok(true);
    while let Some(chunk) = source.next().unwrap() {
        result = distinct.push(chunk, &mut sink);
        if result.is_err() {
            break;
        }
    }
    assert_refused(result, LimitBound::QueryMemoryLimit);
}

/// Two operators of one statement share its cap: what one holds, the other
/// cannot take; given back, it can.
#[test]
fn the_operators_of_one_statement_share_its_cap() {
    let ctx = statement(256 * KIB);
    let mut first = OperatorMemory::new(ctx.clone());
    let mut second = OperatorMemory::new(ctx.clone());
    first.add(160 * KIB).unwrap();
    assert_refused(second.add(160 * KIB), LimitBound::QueryMemoryLimit);
    assert_eq!(second.bytes(), 160 * KIB, "the count keeps the bytes asked");
    first.release();
    second.set(160 * KIB).unwrap();
    assert!(ctx.query().used() >= 160 * KIB);
    // Another statement has a cap of its own.
    let mut other = OperatorMemory::new(statement(256 * KIB));
    other.add(160 * KIB).unwrap();
}

/// Past the buffer manager's limit the refusal names the memory limit, also
/// when the statement's cap would allow the bytes, after the eviction cycle
/// found nothing to free.
#[test]
fn a_grant_past_the_memory_limit_names_the_memory_limit() {
    let manager = BufferManager::with_budget(MIB);
    let _held = manager
        .try_allocate(900 * KIB, MemoryRegion::ExecutionBuffers)
        .unwrap();
    let ctx = OperatorMemoryContext::new(Arc::clone(&manager), None);
    assert_eq!(ctx.query_limit(), MIB);
    let mut memory = OperatorMemory::new(ctx.clone());
    assert_refused(memory.add(200 * KIB), LimitBound::MemoryLimit);
    assert_eq!(ctx.query().used(), 0, "the cap gave the bytes back");
    // What fits under the limit is granted, exactly when the step would not fit.
    memory.clear();
    memory.add(120 * KIB).unwrap();
    assert_eq!(ctx.query().used(), 120 * KIB);
}

/// A query limit above the memory limit is the memory limit.
#[test]
fn the_query_limit_never_exceeds_the_budget() {
    let manager = BufferManager::with_budget(MIB);
    let ctx = OperatorMemoryContext::new(manager, None).with_query_limit(8 * MIB);
    assert_eq!(ctx.query_limit(), MIB);
}

/// The scope restores the one before it, so a statement planned while
/// another plans (a procedure body) leaves the outer one in place.
#[test]
fn a_nested_scope_restores_the_outer_one() {
    let outer = statement(MIB);
    let _outer = memory::enter(StatementMemory::Budgeted(outer.clone()));
    {
        let _inner = memory::enter(StatementMemory::Unbudgeted);
        assert!(memory::current().is_none());
    }
    let current = memory::current().expect("the outer statement");
    assert!(Arc::ptr_eq(current.query(), outer.query()));
}

/// Two sorts of one database each register a consumer of their own name:
/// one sort's end unregisters only its own, and a sort dropped without
/// finishing (a failed statement) unregisters its own too (C18).
#[test]
fn concurrent_sorts_keep_their_own_consumers() {
    let manager = BufferManager::with_budget(64 * MIB);
    let ctx = OperatorMemoryContext::new(Arc::clone(&manager), None);
    let mut first = SpillableSortPushOperator::with_memory_context(
        vec![PushSortKey::ascending(0)],
        ctx.clone(),
    );
    let second = SpillableSortPushOperator::with_memory_context(
        vec![PushSortKey::ascending(0)],
        ctx.clone(),
    );
    let names: Vec<String> = manager
        .consumer_stats()
        .into_iter()
        .map(|stats| stats.name)
        .collect();
    assert_eq!(names.len(), 2);
    assert_ne!(names[0], names[1], "each its own name");
    assert!(
        names
            .iter()
            .all(|name| name.starts_with("SpillableSortPush#"))
    );

    let mut sink = CollectorSink::new();
    let mut source = Numbers::boxed(2, 1);
    while let Some(chunk) = source.next().unwrap() {
        first.push(chunk, &mut sink).unwrap();
    }
    first.finalize(&mut sink).unwrap();
    assert_eq!(manager.consumer_stats().len(), 1, "the other sort's stays");
    drop(second);
    assert_eq!(manager.consumer_stats().len(), 0, "a dropped sort's goes");
    drop(first);
    assert_eq!(ctx.query().used(), 0);
}

/// A sort past the statement's cap without a spill path fails naming the
/// query memory limit, and its consumer goes when it drops.
#[test]
fn a_push_sort_without_a_spill_path_refuses_at_the_query_limit() {
    let manager = BufferManager::with_budget(64 * MIB);
    let ctx = OperatorMemoryContext::new(Arc::clone(&manager), None).with_query_limit(64 * KIB);
    let mut sort = SpillableSortPushOperator::with_memory_context(
        vec![PushSortKey::ascending(0)],
        ctx.clone(),
    );
    let mut sink = CollectorSink::new();
    let mut source = Numbers::boxed(20, 1);
    let mut result = Ok(true);
    while let Some(chunk) = source.next().unwrap() {
        result = sort.push(chunk, &mut sink);
        if result.is_err() {
            break;
        }
    }
    assert_refused(result, LimitBound::QueryMemoryLimit);
    drop(sort);
    assert_eq!(manager.consumer_stats().len(), 0);
    assert_eq!(ctx.query().used(), 0);
}

/// A reset counts 0 again and keeps the grant, so a rerun (an inner plan of a
/// correlated subquery, reset per outer row) grows within it; the grant goes
/// back when the operator drops, and the buffer manager's execution region
/// is where it was before.
#[test]
fn a_reset_keeps_the_grant_and_a_drop_gives_it_back() {
    let ctx = statement(64 * MIB);
    let execution = || {
        ctx.buffer_manager()
            .stats()
            .region_usage(MemoryRegion::ExecutionBuffers)
    };
    let before = execution();

    let mut memory = OperatorMemory::new(ctx.clone());
    memory.add(100 * KIB).unwrap();
    let granted = memory.granted();
    assert!(granted >= 100 * KIB);
    memory.clear();
    assert_eq!(memory.bytes(), 0, "a reset counts 0");
    assert_eq!(memory.granted(), granted, "and keeps the grant");
    assert_eq!(ctx.query().used(), granted);
    memory.add(100 * KIB).unwrap();
    assert_eq!(memory.granted(), granted, "a rerun grows within it");
    memory.release();
    assert_eq!(memory.granted(), 0, "a release gives it back");
    assert_eq!(ctx.query().used(), 0);
    assert_eq!(execution(), before);

    // An operator reset between runs keeps its grant, and gives it back when
    // it drops.
    let mut sort = built_in(&ctx, || {
        SortOperator::new(Numbers::boxed(20, 1), vec![SortKey::ascending(0)])
    });
    assert_eq!(drain(&mut sort).unwrap(), 20 * 1024);
    let held = ctx.query().used();
    assert!(held > 0);
    sort.reset();
    assert_eq!(ctx.query().used(), held, "the reset keeps the grant");
    drop(sort);
    assert_eq!(ctx.query().used(), 0, "the drop gives it back");
    assert_eq!(execution(), before, "nothing leaks");
}

/// A source of one chunk of 1024 integers that starts over at each reset.
struct Again(bool);

impl Operator for Again {
    fn next(&mut self) -> OperatorResult {
        if self.0 {
            return Ok(None);
        }
        self.0 = true;
        let values: Vec<Value> = (0..1024).map(Value::Int64).collect();
        Ok(Some(DataChunk::new(vec![ValueVector::from_values(
            &values,
        )])))
    }

    fn reset(&mut self) {
        self.0 = false;
    }

    fn name(&self) -> &'static str {
        "Again"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// An inner sort under an Apply, reset per outer row, holds one run's grant:
/// the statement's count stays where the first runs left it, and is 0 once
/// the Apply drops, with the execution region where it was.
#[test]
fn an_inner_plan_reset_per_outer_row_keeps_one_runs_grant() {
    let ctx = statement(64 * MIB);
    let execution = || {
        ctx.buffer_manager()
            .stats()
            .region_usage(MemoryRegion::ExecutionBuffers)
    };
    let before = execution();
    let mut apply = built_in(&ctx, || {
        ApplyOperator::new(
            Numbers::boxed(1, 1),
            Box::new(SortOperator::new(
                Box::new(Again(false)),
                vec![SortKey::ascending(0)],
            )),
        )
    });
    let mut rows = 0;
    let mut peak = 0;
    let mut first_chunk = None;
    while let Some(chunk) = apply.next().unwrap() {
        rows += chunk.row_count();
        peak = peak.max(ctx.query().used());
        first_chunk.get_or_insert(ctx.query().used());
    }
    assert_eq!(rows, 1024 * 1024);
    let first = first_chunk.expect("rows came");
    assert!(
        peak <= first + 64 * KIB,
        "{peak} bytes held at most, {first} after the first outer rows"
    );
    drop(apply);
    assert_eq!(ctx.query().used(), 0);
    assert_eq!(execution(), before);
}

/// What a chunk holds counts its strings' and generic values' heap, and
/// nothing more for flat columns.
#[test]
fn chunk_bytes_count_the_value_heap() {
    let mut integers = ValueVector::with_type(LogicalType::Int64);
    integers.push_int64(1);
    integers.push_int64(2);
    let flat = DataChunk::new(vec![integers]);
    assert_eq!(memory::chunk_heap_bytes(&flat), 0);
    assert!(memory::HeapColumns::of(&flat, None).is_empty());

    let mut strings = ValueVector::with_type(LogicalType::String);
    strings.push_string("x".repeat(4000));
    strings.push_string("y".repeat(1000));
    let chunk = DataChunk::new(vec![strings]);
    assert_eq!(memory::chunk_heap_bytes(&chunk), 5000);
    let mut short = ValueVector::with_type(LogicalType::String);
    short.push_string("");
    short.push_string("");
    assert_eq!(
        memory::chunk_bytes(&chunk),
        memory::chunk_bytes(&DataChunk::new(vec![short])) + 5000
    );
    let heap = memory::HeapColumns::of(&chunk, None);
    assert_eq!(heap.row_bytes(0), 4000);
    assert_eq!(heap.rows_bytes(0..2), 5000);

    let lists = DataChunk::new(vec![ValueVector::from_values(&[Value::List(
        vec![Value::from("z".repeat(500).as_str())].into(),
    )])]);
    assert!(memory::chunk_heap_bytes(&lists) >= 500);
}

/// One chunk of 2,000 distinct strings of 4 KB.
struct Long(bool);

impl Operator for Long {
    fn next(&mut self) -> OperatorResult {
        if self.0 {
            return Ok(None);
        }
        self.0 = true;
        let mut column = ValueVector::with_type(LogicalType::String);
        for i in 0..2000 {
            column.push_string(format!("{i}{}", "x".repeat(4000)));
        }
        Ok(Some(DataChunk::new(vec![column])))
    }

    fn reset(&mut self) {}

    fn name(&self) -> &'static str {
        "Long"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// A distinct over long strings counts the strings its seen keys keep: 2,000
/// keys of 4 KB pass a 1 MiB cap that as many integers fit in.
#[test]
fn a_distinct_counts_the_strings_its_keys_keep() {
    let small = statement(MIB);
    let mut distinct = built_in(&small, || DistinctOperator::new(Box::new(Long(false))));
    assert_refused(drain(&mut distinct), LimitBound::QueryMemoryLimit);
    drop(distinct);
    let mut integers = built_in(&small, || DistinctOperator::new(Numbers::boxed(2, 1)));
    assert_eq!(drain(&mut integers).unwrap(), 2048);
}

/// A grouping past the query limit with a spill path refuses at once: its
/// estimate counts spilled groups too, so it never spills to get under the
/// cap (a spilled partition is loaded again by its next key).
#[test]
fn a_spillable_grouping_past_the_query_limit_refuses_without_spilling() {
    let dir = tempfile::tempdir().unwrap();
    let spill = Arc::new(crate::execution::spill::SpillManager::new(dir.path()).unwrap());
    let ctx = OperatorMemoryContext::new(
        BufferManager::with_budget(256 * MIB),
        Some(Arc::clone(&spill)),
    )
    .with_query_limit(64 * KIB);
    let mut aggregate = SpillableAggregatePushOperator::with_memory_context(
        vec![0],
        vec![AggregateExpr::count_star()],
        ctx.clone(),
    );
    let mut sink = CollectorSink::new();
    let mut source = Numbers::boxed(20, 1);
    let mut result = Ok(true);
    while let Some(chunk) = source.next().unwrap() {
        result = aggregate.push(chunk, &mut sink);
        if result.is_err() {
            break;
        }
    }
    assert_refused(result, LimitBound::QueryMemoryLimit);
    assert_eq!(spill.active_file_count(), 0, "nothing spilled");
    drop(aggregate);
    assert_eq!(ctx.query().used(), 0);
}
