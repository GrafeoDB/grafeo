//! The gate for morsel-driven parallel query execution.
//!
//! Two workloads over a label scan of `Person` nodes, at 100k and 1M nodes:
//!
//! - `filter`: load `age` and `city`, keep `age >= 90` (10% of the rows).
//! - `group_by`: load `age` and `city`, keep `age >= 18` (82%), group by city
//!   with `count(*)` and `sum(age)`. Merging the per-worker partial groups is
//!   part of the timed work.
//!
//! Each runs single-threaded through the push [`Pipeline`] and through
//! [`ParallelPipeline`] (default morsel size) at 1, 4 and 8 workers. Before
//! anything is timed, every variant's answer is asserted equal to the
//! single-threaded one, and that one to the answer the data was built to give,
//! so a speedup never comes from a wrong result.
//!
//! Queries do not run on the morsel pipeline yet: this benchmark decides
//! whether wiring it in pays off. Run it with
//! `cargo bench -p grafeo-core --bench morsel_pipeline`.

use std::collections::BTreeMap;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput};

use grafeo_common::types::{LogicalType, NodeId, PropertyKey, Value};
use grafeo_core::execution::operators::push::CompareOp;
use grafeo_core::execution::operators::{
    AggregateExpr, AggregatePushOperator, FilterPushOperator, OperatorError,
};
use grafeo_core::execution::parallel::ParallelNodeScanSource;
use grafeo_core::execution::{
    ChunkCollector, CloneableOperatorFactory, DataChunk, ParallelPipeline, ParallelPipelineConfig,
    Pipeline, PushOperator, Sink, ValueVector,
};
use grafeo_core::graph::GraphStoreSearch;
use grafeo_core::graph::lpg::LpgStore;

const CITIES: [&str; 5] = ["Amsterdam", "Berlin", "Paris", "Prague", "Barcelona"];
const SIZES: [usize; 2] = [100_000, 1_000_000];
const WORKERS: [usize; 3] = [1, 4, 8];

/// Columns after [`FetchProperties`]: the node, its age, its city.
const ID: usize = 0;
const AGE: usize = 1;
const CITY: usize = 2;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Workload {
    Filter,
    GroupBy,
}

impl Workload {
    fn name(self) -> &'static str {
        match self {
            Self::Filter => "filter",
            Self::GroupBy => "group_by",
        }
    }

    /// The lowest age the filter keeps: ages are uniform over 0 to 99.
    fn min_age(self) -> i64 {
        match self {
            Self::Filter => 90,
            Self::GroupBy => 18,
        }
    }
}

/// What a run returns, independent of how the rows were split over chunks
/// and workers.
#[derive(Debug, PartialEq, Eq)]
enum Answer {
    /// The rows out: how many, and the sums of their ages and node IDs.
    Rows {
        rows: usize,
        age_sum: i64,
        id_sum: u64,
    },
    /// Per city: the count and the age sum, merged over partial groups.
    Groups(BTreeMap<String, (i64, i64)>),
}

/// Person `i`'s age: `37` is coprime to `100`, so every 100 persons hold each
/// age from 0 to 99 once.
fn age_of(i: usize) -> i64 {
    i64::try_from(i * 37 % 100).expect("an age is below 100")
}

/// Loads `persons` Person nodes and a tenth as many Company nodes, which have
/// an age too: the label scan must skip them.
fn load(persons: usize) -> Arc<LpgStore> {
    let store = Arc::new(LpgStore::new().expect("arena for the benchmark store"));
    for i in 0..persons {
        store.create_node_with_props(
            &["Person"],
            [
                ("age", Value::Int64(age_of(i))),
                ("city", Value::from(CITIES[i % CITIES.len()])),
            ],
        );
    }
    for i in 0..persons / 10 {
        store.create_node_with_props(&["Company"], [("age", Value::Int64(age_of(i)))]);
    }
    store
}

/// Loads `age` and `city` for the nodes of column 0.
struct FetchProperties {
    store: Arc<LpgStore>,
    age: PropertyKey,
    city: PropertyKey,
}

impl PushOperator for FetchProperties {
    fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
        let column = chunk
            .column(ID)
            .ok_or_else(|| OperatorError::ColumnNotFound("node id".to_string()))?;
        let ids: Vec<NodeId> = match column.as_node_id_slice() {
            Some(slice) => slice.to_vec(),
            None => chunk
                .selected_indices()
                .filter_map(|row| column.get_node_id(row))
                .collect(),
        };
        let mut ages = ValueVector::with_capacity(LogicalType::Int64, ids.len());
        for age in self.store.get_node_property_batch(&ids, &self.age) {
            ages.push_value(age.unwrap_or(Value::Null));
        }
        let mut cities = ValueVector::with_capacity(LogicalType::String, ids.len());
        for city in self.store.get_node_property_batch(&ids, &self.city) {
            cities.push_value(city.unwrap_or(Value::Null));
        }
        let mut nodes = ValueVector::with_capacity(LogicalType::Node, ids.len());
        for id in ids {
            nodes.push_node_id(id);
        }
        sink.consume(DataChunk::new(vec![nodes, ages, cities]))
    }

    fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
        Ok(())
    }

    fn name(&self) -> &'static str {
        "FetchProperties"
    }
}

fn fetch(store: &Arc<LpgStore>) -> Box<dyn PushOperator> {
    Box::new(FetchProperties {
        store: Arc::clone(store),
        age: PropertyKey::from("age"),
        city: PropertyKey::from("city"),
    })
}

fn filter(workload: Workload) -> Box<dyn PushOperator> {
    Box::new(FilterPushOperator::column_compare(
        AGE,
        CompareOp::Ge,
        Value::Int64(workload.min_age()),
    ))
}

fn group_by_city() -> Box<dyn PushOperator> {
    Box::new(AggregatePushOperator::new(
        vec![CITY],
        vec![AggregateExpr::count_star(), AggregateExpr::sum(AGE)],
    ))
}

fn person_scan(store: &Arc<LpgStore>) -> ParallelNodeScanSource {
    ParallelNodeScanSource::with_label(Arc::clone(store) as Arc<dyn GraphStoreSearch>, "Person")
}

fn run_single(store: &Arc<LpgStore>, workload: Workload) -> Vec<DataChunk> {
    let mut operators = vec![fetch(store), filter(workload)];
    if workload == Workload::GroupBy {
        operators.push(group_by_city());
    }
    let mut pipeline = Pipeline::new(
        Box::new(person_scan(store)),
        operators,
        Box::new(ChunkCollector::new()),
    );
    pipeline.execute().expect("the push pipeline runs");
    pipeline
        .into_sink()
        .into_any()
        .downcast::<ChunkCollector>()
        .expect("the sink is the chunk collector")
        .into_chunks()
}

fn run_parallel(store: &Arc<LpgStore>, workload: Workload, workers: usize) -> Vec<DataChunk> {
    let fetch_store = Arc::clone(store);
    let mut factory = CloneableOperatorFactory::new()
        .with_operator(move || fetch(&fetch_store))
        .with_operator(move || filter(workload));
    if workload == Workload::GroupBy {
        factory = factory
            .with_operator(group_by_city)
            .with_pipeline_breakers();
    }
    let config = ParallelPipelineConfig::default().with_workers(workers);
    ParallelPipeline::new(Arc::new(person_scan(store)), Arc::new(factory), config)
        .execute()
        .expect("the parallel pipeline runs")
        .chunks
}

fn int(chunk: &DataChunk, column: usize, row: usize) -> i64 {
    match chunk
        .column(column)
        .and_then(|values| values.get_value(row))
    {
        Some(Value::Int64(value)) => value,
        other => panic!("column {column} row {row} holds {other:?}, not an integer"),
    }
}

/// The answer of a run's chunks. For `group_by` this merges the workers'
/// partial groups.
fn answer(workload: Workload, chunks: &[DataChunk]) -> Answer {
    match workload {
        Workload::Filter => {
            let (mut rows, mut age_sum, mut id_sum) = (0, 0, 0);
            for chunk in chunks {
                for row in chunk.selected_indices() {
                    let id = chunk
                        .column(ID)
                        .and_then(|nodes| nodes.get_node_id(row))
                        .expect("every row has its node");
                    rows += 1;
                    age_sum += int(chunk, AGE, row);
                    id_sum += id.as_u64();
                }
            }
            Answer::Rows {
                rows,
                age_sum,
                id_sum,
            }
        }
        Workload::GroupBy => {
            let mut groups: BTreeMap<String, (i64, i64)> = BTreeMap::new();
            for chunk in chunks {
                for row in chunk.selected_indices() {
                    let city = match chunk.column(0).and_then(|keys| keys.get_value(row)) {
                        Some(Value::String(city)) => city.to_string(),
                        other => panic!("group row {row} has key {other:?}, not a city"),
                    };
                    let group = groups.entry(city).or_default();
                    group.0 += int(chunk, 1, row);
                    group.1 += int(chunk, 2, row);
                }
            }
            Answer::Groups(groups)
        }
    }
}

/// The timed work after the pipeline: merging the partial groups.
fn finish(workload: Workload, chunks: Vec<DataChunk>) {
    if workload == Workload::GroupBy {
        black_box(answer(workload, &chunks));
    } else {
        black_box(chunks);
    }
}

/// Checks the single-threaded answer against what the data was built to
/// give: every 100 persons hold each age from 0 to 99 once.
fn assert_built_answer(workload: Workload, persons: usize, answer: &Answer) {
    let kept_per_hundred = 100 - workload.min_age();
    let kept = i64::try_from(persons).expect("the size fits an i64") / 100 * kept_per_hundred;
    match answer {
        Answer::Rows { rows, .. } => assert_eq!(
            i64::try_from(*rows).expect("the row count fits an i64"),
            kept,
            "the filter keeps the persons aged {} and up",
            workload.min_age()
        ),
        Answer::Groups(groups) => {
            assert_eq!(groups.len(), CITIES.len(), "one group per city");
            let count: i64 = groups.values().map(|group| group.0).sum();
            assert_eq!(count, kept, "the groups count every kept person once");
        }
    }
}

fn bench_morsel_pipeline(c: &mut Criterion) {
    for persons in SIZES {
        let store = load(persons);
        for workload in [Workload::Filter, Workload::GroupBy] {
            let expected = answer(workload, &run_single(&store, workload));
            assert_built_answer(workload, persons, &expected);
            for workers in WORKERS {
                assert_eq!(
                    answer(workload, &run_parallel(&store, workload, workers)),
                    expected,
                    "{} over {persons} persons on {workers} workers must give the \
                     single-threaded answer",
                    workload.name()
                );
            }

            let mut group = c.benchmark_group(format!("morsel_{}/{persons}", workload.name()));
            group.throughput(Throughput::Elements(
                u64::try_from(persons).expect("the size fits a u64"),
            ));
            group.sample_size(10);
            group.warm_up_time(Duration::from_secs(1));
            group.measurement_time(Duration::from_secs(if persons > 100_000 { 6 } else { 2 }));
            group.bench_function("push_pipeline_1_thread", |b| {
                b.iter(|| finish(workload, run_single(&store, workload)));
            });
            for workers in WORKERS {
                group.bench_with_input(
                    BenchmarkId::new("parallel_pipeline_workers", workers),
                    &workers,
                    |b, &workers| {
                        b.iter(|| finish(workload, run_parallel(&store, workload, workers)));
                    },
                );
            }
            group.finish();
        }
    }
}

/// The generated group function is public: a private module keeps it out of
/// the crate's documented surface.
mod gate {
    criterion::criterion_group!(benches, super::bench_morsel_pipeline);
}

criterion::criterion_main!(gate::benches);
