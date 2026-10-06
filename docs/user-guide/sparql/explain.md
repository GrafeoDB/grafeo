---
title: EXPLAIN and EXPLAIN ANALYZE
description: Inspect SPARQL query execution plans in Grafeo.
tags:
  - sparql
  - explain
  - performance
  - profiling
---

# EXPLAIN and EXPLAIN ANALYZE

Grafeo provides two query introspection modes for SPARQL, allowing you to understand how a query will be (or was) executed.

## EXPLAIN

`EXPLAIN` shows the physical execution plan without running the query. Use it to understand which operators Grafeo will use, how joins are ordered, and where filters are applied.

```sparql
EXPLAIN SELECT ?name WHERE {
    ?person <http://xmlns.com/foaf/0.1/name> ?name .
    ?person <http://xmlns.com/foaf/0.1/age> ?age
    FILTER(?age > 30)
}
```

The result is a tree of physical operators with estimated costs. No data is read or returned.

### When to Use EXPLAIN

- Verify that index scans are chosen over full scans
- Check join ordering before running expensive queries
- Confirm that filters are pushed down close to the scan operators
- Understand the shape of the plan for complex queries with OPTIONAL, UNION, or subqueries

## EXPLAIN ANALYZE

`EXPLAIN ANALYZE` executes the query with profiling enabled, then reports per-operator timing and row counts alongside the plan tree.

```sparql
EXPLAIN ANALYZE SELECT ?name WHERE {
    ?person <http://xmlns.com/foaf/0.1/name> ?name .
    ?person <http://xmlns.com/foaf/0.1/age> ?age
    FILTER(?age > 30)
}
```

The result includes both the plan structure and actual runtime statistics: wall-clock time per operator, rows produced, and total execution time.

### When to Use EXPLAIN ANALYZE

- Profile slow queries to find the bottleneck operator
- Compare estimated vs actual row counts to detect stale statistics
- Measure the effect of adding or removing indexes
- Validate that optimizer improvements have real-world impact

## Python API

=== "Python"

    ```python
    import grafeo

    db = grafeo.GrafeoDB()

    # Insert some data first
    db.execute_sparql("""
        INSERT DATA {
            <http://ex.org/alix> <http://xmlns.com/foaf/0.1/name> "Alix" .
            <http://ex.org/alix> <http://xmlns.com/foaf/0.1/age> 30 .
            <http://ex.org/gus> <http://xmlns.com/foaf/0.1/name> "Gus" .
            <http://ex.org/gus> <http://xmlns.com/foaf/0.1/age> 25 .
        }
    """)

    # Show the plan without executing
    plan = db.explain_sparql("""
        SELECT ?name WHERE {
            ?person <http://xmlns.com/foaf/0.1/name> ?name .
            ?person <http://xmlns.com/foaf/0.1/age> ?age
            FILTER(?age > 28)
        }
    """)
    for row in plan:
        print(row)

    # Execute with profiling
    profile = db.execute_sparql("""
        EXPLAIN ANALYZE SELECT ?name WHERE {
            ?person <http://xmlns.com/foaf/0.1/name> ?name .
            ?person <http://xmlns.com/foaf/0.1/age> ?age
            FILTER(?age > 28)
        }
    """)
    for row in profile:
        print(row)
    ```

The `explain_sparql(query)` helper is equivalent to `execute_sparql("EXPLAIN " + query)`.

=== "Rust"

    ```rust
    use grafeo_engine::GrafeoDB;

    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // EXPLAIN (plan only)
    let plan = session.execute_sparql("EXPLAIN SELECT ?s ?p ?o WHERE { ?s ?p ?o }")?;

    // EXPLAIN ANALYZE (plan + runtime stats)
    let profile = session.execute_sparql(
        "EXPLAIN ANALYZE SELECT ?s ?p ?o WHERE { ?s ?p ?o }"
    )?;
    ```

## Interpreting the Output

### EXPLAIN Output

The plan is returned as a result set with a single `plan` column. Each row represents one line of the plan tree, indented to show the operator hierarchy:

```text
Project [?name]
  Filter (?age > 28)
    HashJoin [?person]
      TripleScan (?person, foaf:name, ?name)
      TripleScan (?person, foaf:age, ?age)
```

Key operators to look for:

| Operator | Description |
|----------|-------------|
| `TripleScan` | Scans the triple index for matching patterns |
| `HashJoin` | Joins two inputs on shared variables |
| `NestedLoopJoin` | Row-by-row join (less efficient for large inputs) |
| `Filter` | Applies a condition to incoming rows |
| `Project` | Selects and reorders output columns |
| `Sort` | Orders rows by the given keys |
| `Aggregate` | Groups and aggregates (COUNT, SUM, etc.) |

### EXPLAIN ANALYZE Output

The profiled output adds timing and row-count columns to each operator:

```text
Project [?name]                    (rows: 1, time: 0.02ms)
  Filter (?age > 28)               (rows: 1, time: 0.01ms)
    HashJoin [?person]              (rows: 2, time: 0.05ms)
      TripleScan (foaf:name)        (rows: 2, time: 0.03ms)
      TripleScan (foaf:age)         (rows: 2, time: 0.02ms)

Total execution time: 0.13ms
```

Look for operators with unexpectedly high row counts or time. A `TripleScan` producing many more rows than the final result suggests a missing filter push-down or an unselective pattern.

## Tips

- Run `EXPLAIN` first to check the plan shape, then `EXPLAIN ANALYZE` only when you need actual timings.
- Profiling adds overhead, so `EXPLAIN ANALYZE` timings are slightly higher than normal execution.
- Both modes work with all SPARQL query forms: SELECT, ASK, CONSTRUCT, and DESCRIBE.
