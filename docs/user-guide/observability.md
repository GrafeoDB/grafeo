---
title: Observability
description: Metrics, tracing and profiling for Grafeo databases.
tags:
  - metrics
  - observability
  - monitoring
---

# Observability

Grafeo provides built-in observability through the `metrics` feature flag: query metrics, transaction metrics, plan cache statistics, Prometheus export and structured tracing spans. All counters use lock-free atomics, so recording a metric is a single atomic increment with no contention.

## Metrics

Enable the `metrics` feature in your `Cargo.toml`:

```toml
[dependencies]
grafeo = { version = "0.5", features = ["metrics"] }
```

!!! note
    The `metrics` feature is included in the `server` profile. For other profiles, add it explicitly with `features = ["metrics"]`.

### Retrieving a Snapshot

Call `db.metrics()` to get a serializable point-in-time snapshot of all tracked metrics:

```rust
use grafeo::GrafeoDB;

let db = GrafeoDB::open(":memory:")?;
let session = db.session();

session.execute("INSERT (:Person {name: 'Alix'})")?;
session.execute("INSERT (:Person {name: 'Gus'})")?;
session.execute("MATCH (n:Person) RETURN n.name")?;

let snapshot = db.metrics();
println!("Queries executed: {}", snapshot.query_count);
println!("Mean latency:     {:.2}ms", snapshot.query_latency_mean_ms);
println!("Rows returned:    {}", snapshot.rows_returned);
println!("Cache hit rate:   {}/{}", snapshot.cache_hits, snapshot.cache_hits + snapshot.cache_misses);
```

### Resetting Counters

Call `db.reset_metrics()` to zero out all counters and histograms. This is useful when collecting metrics over fixed windows:

```rust
db.reset_metrics();
// ... run workload ...
let window_snapshot = db.metrics();
```

### Tracked Metrics

#### Query Metrics

| Field | Type | Description |
|-------|------|-------------|
| `query_count` | counter | Total queries executed |
| `query_errors` | counter | Queries that returned an error |
| `query_timeouts` | counter | Queries cancelled by timeout |
| `query_latency_p50_ms` | gauge | 50th percentile query latency (ms) |
| `query_latency_p99_ms` | gauge | 99th percentile query latency (ms) |
| `query_latency_mean_ms` | gauge | Mean query latency (ms) |
| `rows_returned` | counter | Cumulative rows returned |
| `rows_scanned` | counter | Cumulative rows scanned |
| `queries_gql` | counter | GQL queries executed |
| `queries_cypher` | counter | Cypher queries executed |
| `queries_sparql` | counter | SPARQL queries executed |
| `queries_gremlin` | counter | Gremlin queries executed |
| `queries_graphql` | counter | GraphQL queries executed |
| `queries_sql_pgq` | counter | SQL/PGQ queries executed |

#### Transaction Metrics

| Field | Type | Description |
|-------|------|-------------|
| `tx_active` | gauge | Currently open transactions |
| `tx_committed` | counter | Total transactions committed |
| `tx_rolled_back` | counter | Total transactions rolled back |
| `tx_conflicts` | counter | Write-write conflicts detected |
| `tx_duration_p50_ms` | gauge | 50th percentile transaction duration (ms) |
| `tx_duration_p99_ms` | gauge | 99th percentile transaction duration (ms) |
| `tx_duration_mean_ms` | gauge | Mean transaction duration (ms) |

#### Session and GC Metrics

| Field | Type | Description |
|-------|------|-------------|
| `session_active` | gauge | Currently active sessions |
| `session_created` | counter | Total sessions created |
| `gc_runs` | counter | Total garbage collection sweep runs |

#### Plan Cache Metrics

| Field | Type | Description |
|-------|------|-------------|
| `cache_hits` | counter | Plan cache hits (parsed + optimized) |
| `cache_misses` | counter | Plan cache misses (parsed + optimized) |
| `cache_size` | gauge | Current number of cached plans |
| `cache_invalidations` | counter | Cache invalidations triggered by DDL |

### Per-Query Metrics (Python)

Each query result in Python includes per-query performance data:

```python
from grafeo import GrafeoDB

db = GrafeoDB()
db.execute("INSERT (:Person {name: 'Alix'})")
db.execute("INSERT (:Person {name: 'Gus'})")

result = db.execute("MATCH (n:Person) RETURN n.name")
print(f"Execution time: {result.execution_time_ms:.2f}ms")
print(f"Rows scanned:   {result.rows_scanned}")
```

## Prometheus Export

_Since 0.5.23_

Call `db.metrics_prometheus()` to get all metrics in Prometheus text exposition format, ready to serve from an HTTP `/metrics` endpoint:

```rust
let prometheus_output = db.metrics_prometheus();
println!("{prometheus_output}");
```

Example output:

```text
# HELP grafeo_query_count Total queries executed.
# TYPE grafeo_query_count counter
grafeo_query_count 42

# HELP grafeo_query_errors Queries that returned an error.
# TYPE grafeo_query_errors counter
grafeo_query_errors 1

# HELP grafeo_query_latency_ms Query latency in milliseconds.
# TYPE grafeo_query_latency_ms histogram
grafeo_query_latency_ms_bucket{le="0.1"} 5
grafeo_query_latency_ms_bucket{le="0.25"} 18
grafeo_query_latency_ms_bucket{le="0.5"} 30
grafeo_query_latency_ms_bucket{le="1"} 38
grafeo_query_latency_ms_bucket{le="2.5"} 40
grafeo_query_latency_ms_bucket{le="5"} 41
grafeo_query_latency_ms_bucket{le="10"} 42
grafeo_query_latency_ms_bucket{le="+Inf"} 42
grafeo_query_latency_ms_sum 28.5
grafeo_query_latency_ms_count 42

# HELP grafeo_query_count_by_language Queries executed per language.
# TYPE grafeo_query_count_by_language counter
grafeo_query_count_by_language{language="gql"} 42

# HELP grafeo_tx_active Currently active transactions.
# TYPE grafeo_tx_active gauge
grafeo_tx_active 0

# HELP grafeo_tx_committed Total transactions committed.
# TYPE grafeo_tx_committed counter
grafeo_tx_committed 10

# HELP grafeo_gc_runs Total garbage collection runs.
# TYPE grafeo_gc_runs counter
grafeo_gc_runs 3
```

In `grafeo-server`, this output is served directly from the `/metrics` HTTP endpoint for Prometheus scraping.

## Tracing Spans

_Since 0.5.23_

Grafeo emits structured [`tracing`](https://docs.rs/tracing) spans at key points in the query and transaction lifecycle. When no `tracing` subscriber is registered, these spans compile down to no-ops with zero runtime cost.

### Span Names

| Span | Level | Description |
|------|-------|-------------|
| `grafeo::session::execute` | `INFO` | Full query execution (includes language field) |
| `grafeo::query::parse` | `DEBUG` | Query parsing (includes language field) |
| `grafeo::query::optimize` | `DEBUG` | Logical plan optimization |
| `grafeo::query::plan` | `DEBUG` | Physical plan generation |
| `grafeo::query::execute` | `DEBUG` | Physical operator execution |
| `grafeo::tx::begin` | `DEBUG` | Transaction begin (includes read_only field) |
| `grafeo::tx::commit` | `DEBUG` | Transaction commit |
| `grafeo::tx::rollback` | `DEBUG` | Transaction rollback |

### Example: Enabling Tracing in Rust

```rust
use grafeo::GrafeoDB;
use tracing_subscriber::{fmt, EnvFilter};

fn main() -> grafeo::Result<()> {
    // Initialize a subscriber that prints spans to stderr.
    // Set RUST_LOG=grafeo=debug for full span output.
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let db = GrafeoDB::open(":memory:")?;
    let session = db.session();

    session.execute("INSERT (:Person {name: 'Alix'})")?;
    session.execute("MATCH (n:Person) RETURN n.name")?;

    Ok(())
}
```

Running with `RUST_LOG=grafeo=debug` produces output like:

```text
DEBUG grafeo::query::parse{language=Gql}: parsing query
DEBUG grafeo::query::optimize: optimizing logical plan
DEBUG grafeo::query::plan: generating physical plan
DEBUG grafeo::query::execute: executing operators
 INFO grafeo::session::execute{language="gql"}: query complete
```

## EXPLAIN

_Since 0.5.14_

Prefix any query with `EXPLAIN` to see the optimized logical plan without executing it. The plan shows operator ordering, pushdown hints and index usage:

```python
from grafeo import GrafeoDB

db = GrafeoDB()
db.execute("INSERT (:Person {name: 'Alix', age: 30})")
db.execute("INSERT (:Person {name: 'Gus', age: 25})")

result = db.execute("EXPLAIN MATCH (n:Person) WHERE n.age > 20 RETURN n.name")
print(result[0][0])
```

Example output:

```text
Projection [n.name]
  Filter (n.age > 20)
    NodeScan (n:Person) [label-first]
```

Pushdown hints in square brackets indicate optimizer decisions:

| Hint | Meaning |
|------|---------|
| `[label-first]` | Label filter applied at scan level |
| `[index: prop]` | Property index used for filtering |
| `[inline-filter]` | Filter merged into scan operator |

EXPLAIN works the same way in Rust:

```rust
let result = session.execute("EXPLAIN MATCH (n:Person) RETURN n")?;
println!("{}", result.rows()[0][0]);
```

## PROFILE

_Since 0.5.16_

Prefix any query with `PROFILE` to execute it and get per-operator runtime metrics. Unlike EXPLAIN, PROFILE runs the query and returns actual performance data:

```python
from grafeo import GrafeoDB

db = GrafeoDB()
db.execute("INSERT (:Person {name: 'Alix', age: 30})")
db.execute("INSERT (:Person {name: 'Gus', age: 25})")

result = db.execute("PROFILE MATCH (n:Person) WHERE n.age > 20 RETURN n.name")
print(result[0][0])
```

Example output:

```text
Projection (n.name)  rows=2  time=0.01ms
  Filter (n.age > 20)  rows=2  time=0.03ms
    NodeScan (n:Person)  rows=2  time=0.05ms

Total time: 0.12ms
```

Each operator line includes:

| Metric | Description |
|--------|-------------|
| `rows` | Number of rows produced by this operator |
| `time` | Self-time for this operator (wall clock minus children) |

The underlying `ProfileStats` struct also tracks `calls` (number of `next()` invocations on the operator), available when using the Rust API directly.

## Plan Cache Statistics

Grafeo caches parsed and optimized query plans to avoid redundant work on repeated queries. The cache operates transparently, but you can inspect and manage it.

### Inspecting Cache Stats

Cache statistics are included in the `MetricsSnapshot` returned by `db.metrics()`:

```rust
let snapshot = db.metrics();
println!("Cache hits:          {}", snapshot.cache_hits);
println!("Cache misses:        {}", snapshot.cache_misses);
println!("Cached plans:        {}", snapshot.cache_size);
println!("Cache invalidations: {}", snapshot.cache_invalidations);
```

### Clearing the Cache

All bindings expose a `clear_plan_cache()` method:

=== "Python"

    ```python
    db.clear_plan_cache()
    ```

=== "Rust"

    ```rust
    db.clear_plan_cache();
    ```

=== "Node.js"

    ```javascript
    db.clearPlanCache();
    ```

=== "WASM"

    ```javascript
    db.clearPlanCache();
    ```

=== "C"

    ```c
    grafeo_clear_plan_cache(db);
    ```

### Auto-Invalidation

The plan cache is automatically invalidated after DDL operations such as `CREATE INDEX`, `DROP INDEX` and `DROP TYPE`. This ensures that queries are re-optimized to take advantage of new indexes or reflect schema changes. Manual clearing is only needed after external schema modifications or when you want to force re-optimization after a bulk data import.

## Change Data Capture

Change Data Capture (CDC, since 0.5.19) tracks every mutation to nodes and edges as an append-only event log. Enable it with the `cdc` feature flag:

```toml
[dependencies]
grafeo = { version = "0.5", features = ["cdc"] }
```

!!! note
    The `cdc` feature is included in the `ai`, `server`, and `full` profiles. For other profiles, add it explicitly with `features = ["cdc"]`.

### Change Events

Each event is a dictionary (Python) / struct (Rust) with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `entity_id` | `int` | ID of the affected node or edge |
| `entity_type` | `str` | `"node"` or `"edge"` |
| `kind` | `str` | `"create"`, `"update"`, or `"delete"` |
| `epoch` | `int` | MVCC epoch when the change was committed |
| `timestamp` | `int` | Wall-clock time in milliseconds since Unix epoch |
| `before` | `dict or None` | Property snapshot before the change (`None` for creates) |
| `after` | `dict or None` | Property snapshot after the change (`None` for deletes) |

### Per-Entity History

```python
from grafeo import GrafeoDB

db = GrafeoDB()
session = db.session()

session.execute("INSERT (:Server {name: 'web-01', status: 'active'})")
session.commit()

session.execute("MATCH (s:Server {name: 'web-01'}) SET s.status = 'retired'")
session.commit()

node_id = db.execute("MATCH (s:Server) RETURN id(s)")[0][0]

# Full history for a single node
events = db.node_history(node_id)
for e in events:
    print(f"epoch={e['epoch']}  kind={e['kind']}  after={e['after']}")
# epoch=1  kind=create  after={'name': 'web-01', 'status': 'active'}
# epoch=2  kind=update  after={'status': 'retired'}

# History since a known epoch (incremental polling)
recent = db.node_history_since(node_id, since_epoch=2)

# Edge history
edge_events = db.edge_history(edge_id)
```

### Range Queries

`changes_between(start_epoch, end_epoch)` returns all change events across all entities within an epoch range. This is the foundation for replication and offline sync:

```python
# Collect everything that changed between epoch 10 and 20
events = db.changes_between(start_epoch=10, end_epoch=20)
for e in events:
    print(f"{e['entity_type']} {e['entity_id']}: {e['kind']} at epoch {e['epoch']}")
```

The grafeo-server HTTP API exposes this as `GET /db/{name}/changes?since={epoch}`. See [Offline Sync](offline-sync.md) for the full pull/push protocol.
