//! Write-path benchmarks: query writes, batch calls, rollback and the reads
//! next to them.
//!
//! Every case builds its input outside the timed part (a fresh database,
//! preloaded nodes, the parameter maps) and drops the database outside it
//! too, so a file database's closing checkpoint is not measured. A file
//! database is opened with [`GrafeoDB::open`] and the default durability
//! (the WAL with batched syncs) in a fresh temporary directory per
//! iteration.
//!
//! The scale defaults to 10,000 statements (100,000 rows for the large batch
//! calls and 100,000 nodes for the reads); `GRAFEO_BENCH_SCALE` divides it,
//! for example `GRAFEO_BENCH_SCALE=10` for a quick run or under CodSpeed's
//! simulation.
//!
//! Run with: cargo bench -p grafeo-engine --features full --bench write_path_bench

use std::collections::HashMap;
use std::hint::black_box;
use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{BatchSize, Bencher, BenchmarkGroup, Criterion, SamplingMode, criterion_main};

use grafeo_common::types::{NodeId, PropertyKey, Value};
use grafeo_engine::database::BatchEdge;
use grafeo_engine::{GrafeoDB, Session};

// ============================================================================
// Scale and data
// ============================================================================

/// Statements per write case.
const STATEMENTS: usize = 10_000;
/// Rows of the large batch calls.
const LARGE_BATCH: usize = 100_000;
/// Nodes of the read graph (with five edges each).
const READ_NODES: usize = 100_000;
/// Nodes the `DETACH DELETE` cases delete.
const DELETES: usize = 1_000;
/// Index lookups per iteration of the lookup read.
const LOOKUPS: usize = 1_000;

const NAMES: [&str; 8] = [
    "Alix", "Gus", "Vincent", "Mia", "Jules", "Butch", "Django", "Beatrix",
];

/// `count` divided by `GRAFEO_BENCH_SCALE` (at least 1).
fn scaled(count: usize) -> usize {
    let divisor = std::env::var("GRAFEO_BENCH_SCALE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|divisor| *divisor > 0)
        .unwrap_or(1);
    (count / divisor).max(1)
}

/// The unique name of person `index`.
fn name(index: usize) -> String {
    format!("{}-{index}", NAMES[index % NAMES.len()])
}

/// The age of person `index`: 3 to 90.
fn age(index: usize) -> i64 {
    3 + i64::try_from(index % 88).expect("index % 88 fits in i64")
}

/// The target of edge `index` among `nodes` nodes, whose source is
/// `index % nodes`: the edges of one source (one per round of `nodes`) go to
/// different targets.
fn edge_target(index: usize, nodes: usize) -> usize {
    let (round, source) = (index / nodes, index % nodes);
    (source * 7 + 13 + round * 101) % nodes
}

fn params(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_string(), value.clone()))
        .collect()
}

/// `{n, a}` for person `index`: its name and a new age.
fn person_params(count: usize) -> Vec<HashMap<String, Value>> {
    (0..count)
        .map(|index| {
            params(&[
                ("n", Value::from(name(index))),
                ("a", Value::Int64(age(index + 19))),
            ])
        })
        .collect()
}

/// `{a, b}` naming the endpoints of edge `index` among `nodes` people.
fn edge_params(count: usize, nodes: usize) -> Vec<HashMap<String, Value>> {
    (0..count)
        .map(|index| {
            params(&[
                ("a", Value::from(name(index % nodes))),
                ("b", Value::from(name(edge_target(index, nodes)))),
            ])
        })
        .collect()
}

/// The position of person `index`, as a property value.
fn idx(index: usize) -> Value {
    Value::Int64(i64::try_from(index).expect("bench indexes fit in i64"))
}

fn person_properties(index: usize) -> HashMap<PropertyKey, Value> {
    let mut properties = HashMap::new();
    properties.insert(PropertyKey::from("idx"), idx(index));
    properties.insert(PropertyKey::from("name"), Value::from(name(index)));
    properties.insert(PropertyKey::from("age"), Value::Int64(age(index)));
    properties
}

/// Creates `count` people (with `idx`, `name` and `age`) in one batch.
fn load_people(db: &GrafeoDB, count: usize) -> Vec<NodeId> {
    db.batch_create_nodes_with_props("Person", (0..count).map(person_properties).collect())
        .expect("load people")
}

/// Creates `per_node` KNOWS edges from every person in one batch.
fn load_knows(db: &GrafeoDB, people: &[NodeId], per_node: usize) {
    let edges = (0..people.len() * per_node)
        .map(|index| {
            BatchEdge::new(
                people[index % people.len()],
                people[edge_target(index, people.len())],
                "KNOWS",
            )
        })
        .collect();
    db.batch_create_edges(edges).expect("load edges");
}

// ============================================================================
// Databases
// ============================================================================

/// Where a case runs.
#[derive(Clone, Copy)]
enum Storage {
    Memory,
    File,
}

impl Storage {
    fn label(self) -> &'static str {
        match self {
            Self::Memory => "mem",
            Self::File => "file",
        }
    }
}

/// A database for one iteration; the session drops first, then the
/// database, then its directory.
struct Fixture {
    session: Session,
    db: GrafeoDB,
    _dir: Option<tempfile::TempDir>,
}

impl Fixture {
    fn new(storage: Storage) -> Self {
        let (db, dir) = match storage {
            Storage::Memory => (GrafeoDB::new_in_memory(), None),
            Storage::File => {
                let dir = tempfile::tempdir().expect("temporary directory");
                let db =
                    GrafeoDB::open(dir.path().join("bench.grafeo")).expect("open file database");
                (db, Some(dir))
            }
        };
        Self {
            session: db.session(),
            db,
            _dir: dir,
        }
    }

    /// A fixture with `count` people and an index on `name`.
    fn with_people(storage: Storage, count: usize) -> (Self, Vec<NodeId>) {
        let fixture = Self::new(storage);
        let people = load_people(&fixture.db, count);
        fixture
            .db
            .create_property_index("name")
            .expect("name index");
        (fixture, people)
    }
}

/// Measures `routine` on a fresh `setup()` per iteration: neither the setup
/// nor the drop of its value is measured (`iter_batched`, which CodSpeed
/// supports, unlike `iter_custom`).
fn per_iteration<S>(
    b: &mut Bencher<'_>,
    setup: impl FnMut() -> S,
    mut routine: impl FnMut(&mut S),
) {
    b.iter_batched(
        setup,
        |mut state| {
            routine(&mut state);
            state
        },
        BatchSize::PerIteration,
    );
}

fn gql(session: &Session, query: &str, params: HashMap<String, Value>) {
    black_box(session.execute_with_params(query, params).expect(query));
}

fn cypher(session: &Session, query: &str, params: HashMap<String, Value>) {
    black_box(
        session
            .execute_cypher_with_params(query, params)
            .expect(query),
    );
}

fn slow_group<'a>(c: &'a mut Criterion, name: &str) -> BenchmarkGroup<'a, WallTime> {
    let mut group = c.benchmark_group(name);
    group
        .sample_size(10)
        .sampling_mode(SamplingMode::Flat)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(3));
    group
}

// ============================================================================
// Query writes
// ============================================================================

const GQL_INSERT: &str = "INSERT (:Person {name: $n, age: $a})";
const GQL_SET: &str = "MATCH (p:Person {name: $n}) SET p.age = $a";

fn bench_query_writes(c: &mut Criterion) {
    let statements = scaled(STATEMENTS);
    let deletes = scaled(DELETES);
    let mut group = slow_group(c, "write_query");
    for storage in [Storage::Memory, Storage::File] {
        let tag = storage.label();

        group.bench_function(format!("gql_insert_autocommit/{tag}"), |b| {
            per_iteration(
                b,
                || (Fixture::new(storage), person_params(statements)),
                |(fixture, params)| {
                    for p in params.drain(..) {
                        gql(&fixture.session, GQL_INSERT, p);
                    }
                },
            );
        });

        group.bench_function(format!("gql_insert_one_txn/{tag}"), |b| {
            per_iteration(
                b,
                || (Fixture::new(storage), person_params(statements)),
                |(fixture, params)| {
                    fixture.session.begin_transaction().expect("begin");
                    for p in params.drain(..) {
                        gql(&fixture.session, GQL_INSERT, p);
                    }
                    fixture.session.commit().expect("commit");
                },
            );
        });

        group.bench_function(format!("cypher_create_node/{tag}"), |b| {
            per_iteration(
                b,
                || (Fixture::new(storage), person_params(statements)),
                |(fixture, params)| {
                    for p in params.drain(..) {
                        cypher(&fixture.session, "CREATE (:Person {name: $n, age: $a})", p);
                    }
                },
            );
        });

        group.bench_function(format!("cypher_merge_node/{tag}"), |b| {
            per_iteration(
                b,
                || {
                    let fixture = Fixture::new(storage);
                    fixture
                        .db
                        .create_property_index("name")
                        .expect("name index");
                    (fixture, person_params(statements))
                },
                |(fixture, params)| {
                    for p in params.drain(..) {
                        cypher(&fixture.session, "MERGE (:Person {name: $n, age: $a})", p);
                    }
                },
            );
        });

        group.bench_function(format!("cypher_create_rel/{tag}"), |b| {
            per_iteration(
                b,
                    || {
                        let (fixture, _) = Fixture::with_people(storage, statements);
                        (fixture, edge_params(statements, statements))
                    },
                    |(fixture, params)| {
                        for p in params.drain(..) {
                            cypher(
                                &fixture.session,
                                "MATCH (a:Person {name: $a}), (b:Person {name: $b}) CREATE (a)-[:KNOWS]->(b)",
                                p,
                            );
                        }
                    },
                );
        });

        group.bench_function(format!("cypher_merge_rel/{tag}"), |b| {
            per_iteration(
                b,
                    || {
                        let (fixture, _) = Fixture::with_people(storage, statements);
                        (fixture, edge_params(statements, statements))
                    },
                    |(fixture, params)| {
                        for p in params.drain(..) {
                            cypher(
                                &fixture.session,
                                "MATCH (a:Person {name: $a}), (b:Person {name: $b}) MERGE (a)-[:KNOWS]->(b)",
                                p,
                            );
                        }
                    },
                );
        });

        group.bench_function(format!("gql_set_autocommit/{tag}"), |b| {
            per_iteration(
                b,
                || {
                    let (fixture, _) = Fixture::with_people(storage, statements);
                    (fixture, person_params(statements))
                },
                |(fixture, params)| {
                    for p in params.drain(..) {
                        gql(&fixture.session, GQL_SET, p);
                    }
                },
            );
        });

        group.bench_function(format!("gql_set_one_txn/{tag}"), |b| {
            per_iteration(
                b,
                || {
                    let (fixture, _) = Fixture::with_people(storage, statements);
                    (fixture, person_params(statements))
                },
                |(fixture, params)| {
                    fixture.session.begin_transaction().expect("begin");
                    for p in params.drain(..) {
                        gql(&fixture.session, GQL_SET, p);
                    }
                    fixture.session.commit().expect("commit");
                },
            );
        });

        group.bench_function(format!("gql_match_create_edge/{tag}"), |b| {
            per_iteration(
                b,
                    || {
                        let (fixture, _) = Fixture::with_people(storage, statements);
                        (fixture, edge_params(statements, statements))
                    },
                    |(fixture, params)| {
                        for p in params.drain(..) {
                            gql(
                                &fixture.session,
                                "MATCH (a:Person), (b:Person) WHERE a.name = $a AND b.name = $b CREATE (a)-[:KNOWS]->(b)",
                                p,
                            );
                        }
                    },
                );
        });

        // One statement deleting `deletes` people, each with about ten edges.
        group.bench_function(format!("gql_detach_delete_one_stmt/{tag}"), |b| {
            per_iteration(
                b,
                || {
                    let (fixture, people) = Fixture::with_people(storage, statements);
                    load_knows(&fixture.db, &people, 5);
                    fixture
                },
                |fixture| {
                    gql(
                        &fixture.session,
                        "MATCH (p:Person) WHERE p.idx < $limit DETACH DELETE p",
                        params(&[("limit", idx(deletes))]),
                    );
                },
            );
        });

        // `deletes` auto-commit statements deleting one person each.
        group.bench_function(format!("gql_detach_delete_autocommit/{tag}"), |b| {
            per_iteration(
                b,
                || {
                    let (fixture, people) = Fixture::with_people(storage, statements);
                    load_knows(&fixture.db, &people, 5);
                    (fixture, person_params(deletes))
                },
                |(fixture, params)| {
                    for p in params.drain(..) {
                        gql(
                            &fixture.session,
                            "MATCH (p:Person {name: $n}) DETACH DELETE p",
                            p,
                        );
                    }
                },
            );
        });
    }
    group.finish();
}

// ============================================================================
// Batch calls and the direct API
// ============================================================================

fn bench_batch_calls(c: &mut Criterion) {
    let mut group = slow_group(c, "write_batch");
    for storage in [Storage::Memory, Storage::File] {
        let tag = storage.label();
        for rows in [scaled(STATEMENTS), scaled(LARGE_BATCH)] {
            group.bench_function(format!("batch_create_nodes/{rows}/{tag}"), |b| {
                per_iteration(
                    b,
                    || {
                        let fixture = Fixture::new(storage);
                        let rows: Vec<_> = (0..rows).map(person_properties).collect();
                        (fixture, Some(rows))
                    },
                    |(fixture, rows)| {
                        let rows = rows.take().expect("rows");
                        black_box(
                            fixture
                                .db
                                .batch_create_nodes_with_props("Person", rows)
                                .expect("batch nodes"),
                        );
                    },
                );
            });

            group.bench_function(format!("batch_create_edges/{rows}/{tag}"), |b| {
                per_iteration(
                    b,
                    || {
                        let fixture = Fixture::new(storage);
                        let people = load_people(&fixture.db, (rows / 5).max(19));
                        let edges: Vec<_> = (0..rows)
                            .map(|index| {
                                BatchEdge::new(
                                    people[index % people.len()],
                                    people[edge_target(index, people.len())],
                                    "KNOWS",
                                )
                                .with_properties([("since", age(index) + 1900)])
                            })
                            .collect();
                        (fixture, Some(edges))
                    },
                    |(fixture, edges)| {
                        let edges = edges.take().expect("edges");
                        black_box(fixture.db.batch_create_edges(edges).expect("batch edges"));
                    },
                );
            });
        }
    }

    // The direct calls one at a time, as memory_bench builds its graph: a
    // node, three properties, then five edges per node.
    let nodes = scaled(STATEMENTS);
    group.bench_function(format!("direct_calls_nodes_edges/{nodes}/mem"), |b| {
        per_iteration(
            b,
            || Fixture::new(Storage::Memory),
            |fixture| {
                let db = &fixture.db;
                let mut people = Vec::with_capacity(nodes);
                for index in 0..nodes {
                    let id = db.create_node(&["Person"]).expect("node");
                    db.set_node_property(id, "idx", idx(index)).expect("idx");
                    db.set_node_property(id, "name", Value::from(name(index)))
                        .expect("name");
                    db.set_node_property(id, "age", Value::Int64(age(index)))
                        .expect("age");
                    people.push(id);
                }
                for index in 0..nodes * 5 {
                    db.create_edge(
                        people[index % nodes],
                        people[edge_target(index, nodes)],
                        "KNOWS",
                    )
                    .expect("edge");
                }
            },
        );
    });

    // The same calls split: the nodes with their properties, then the edges
    // between existing nodes.
    group.bench_function(format!("direct_create_nodes/{nodes}/mem"), |b| {
        per_iteration(
            b,
            || Fixture::new(Storage::Memory),
            |fixture| {
                let db = &fixture.db;
                for index in 0..nodes {
                    let id = db.create_node(&["Person"]).expect("node");
                    db.set_node_property(id, "idx", idx(index)).expect("idx");
                    db.set_node_property(id, "name", Value::from(name(index)))
                        .expect("name");
                    db.set_node_property(id, "age", Value::Int64(age(index)))
                        .expect("age");
                }
            },
        );
    });
    group.bench_function(format!("direct_create_edges/{}/mem", nodes * 5), |b| {
        per_iteration(
            b,
            || {
                let fixture = Fixture::new(Storage::Memory);
                let people = load_people(&fixture.db, nodes);
                (fixture, people)
            },
            |(fixture, people)| {
                for index in 0..nodes * 5 {
                    fixture
                        .db
                        .create_edge(
                            people[index % nodes],
                            people[edge_target(index, nodes)],
                            "KNOWS",
                        )
                        .expect("edge");
                }
            },
        );
    });
    group.finish();
}

// ============================================================================
// Rollback
// ============================================================================

fn bench_rollback(c: &mut Criterion) {
    let statements = scaled(STATEMENTS);
    let mut group = slow_group(c, "write_rollback");
    for storage in [Storage::Memory, Storage::File] {
        let tag = storage.label();

        // The whole transaction: 10k inserts, then the rollback.
        group.bench_function(format!("insert_then_rollback/{tag}"), |b| {
            per_iteration(
                b,
                || (Fixture::new(storage), person_params(statements)),
                |(fixture, params)| {
                    fixture.session.begin_transaction().expect("begin");
                    for p in params.drain(..) {
                        gql(&fixture.session, GQL_INSERT, p);
                    }
                    fixture.session.rollback().expect("rollback");
                },
            );
        });

        // Only the rollback of 10k inserts.
        group.bench_function(format!("rollback_only_inserts/{tag}"), |b| {
            per_iteration(
                b,
                || {
                    let mut fixture = Fixture::new(storage);
                    fixture.session.begin_transaction().expect("begin");
                    for p in person_params(statements) {
                        gql(&fixture.session, GQL_INSERT, p);
                    }
                    fixture
                },
                |fixture| fixture.session.rollback().expect("rollback"),
            );
        });

        // Only the rollback of 10k property updates on existing people.
        group.bench_function(format!("rollback_only_sets/{tag}"), |b| {
            per_iteration(
                b,
                || {
                    let (mut fixture, _) = Fixture::with_people(storage, statements);
                    fixture.session.begin_transaction().expect("begin");
                    for p in person_params(statements) {
                        gql(&fixture.session, GQL_SET, p);
                    }
                    fixture
                },
                |fixture| fixture.session.rollback().expect("rollback"),
            );
        });
    }
    group.finish();
}

// ============================================================================
// Reads (the write path must not move them)
// ============================================================================

/// The read graph: `nodes` people with five KNOWS edges each and an index on
/// `name`, and a session on it.
fn read_graph(nodes: usize) -> Fixture {
    let fixture = Fixture::new(Storage::Memory);
    let people = load_people(&fixture.db, nodes);
    load_knows(&fixture.db, &people, 5);
    fixture
        .db
        .create_property_index("name")
        .expect("name index");
    fixture
}

fn bench_reads(c: &mut Criterion) {
    let nodes = scaled(READ_NODES);
    let lookups = scaled(LOOKUPS).min(nodes);
    // Built by the first case that runs, so a filtered run without reads
    // skips it.
    let graph = std::cell::OnceCell::new();

    let mut group = c.benchmark_group("read");
    group
        .sample_size(20)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(3));

    group.bench_function("label_scan_filter", |b| {
        let session = &graph.get_or_init(|| read_graph(nodes)).session;
        b.iter(|| {
            black_box(
                session
                    .execute("MATCH (p:Person) WHERE p.age > 80 RETURN p.name")
                    .expect("scan"),
            )
        });
    });

    group.bench_function("two_hop_count", |b| {
        let session = &graph.get_or_init(|| read_graph(nodes)).session;
        b.iter(|| {
            black_box(
                session
                    .execute(
                        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) WHERE a.age = 19 RETURN count(c)",
                    )
                    .expect("two hops"),
            )
        });
    });

    let lookup_params: Vec<_> = (0..lookups)
        .map(|index| params(&[("n", Value::from(name(index * (nodes / lookups))))]))
        .collect();
    group.bench_function(format!("index_lookup_x{lookups}"), |b| {
        let session = &graph.get_or_init(|| read_graph(nodes)).session;
        b.iter(|| {
            for p in &lookup_params {
                black_box(
                    session
                        .execute_with_params("MATCH (p:Person {name: $n}) RETURN p.age", p.clone())
                        .expect("lookup"),
                );
            }
        });
    });
    group.finish();
}

/// The benchmark groups, in a private module so the generated function needs
/// no doc comment.
mod groups {
    use super::{bench_batch_calls, bench_query_writes, bench_reads, bench_rollback};

    criterion::criterion_group!(
        benches,
        bench_query_writes,
        bench_batch_calls,
        bench_rollback,
        bench_reads
    );
}

criterion_main!(groups::benches);
