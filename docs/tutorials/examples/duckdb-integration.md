---
title: DuckDB Integration
description: Combine Grafeo graph storage with DuckDB analytical SQL queries via zero-copy Apache Arrow.
tags:
  - example
  - duckdb
  - arrow
  - analytics
---

# DuckDB Integration

Combine Grafeo for graph storage and traversal with DuckDB for analytical SQL queries. Data flows between the two via Apache Arrow Tables with zero-copy overhead: no serialization, no parsing, no intermediate formats.

## Topics Covered

- Exporting graph data to Arrow Tables
- Querying nodes and edges with DuckDB SQL
- Joining nodes and edges for connectivity analysis
- Combining GQL pattern matching with SQL aggregation
- RDF/SPARQL results in DuckDB
- Writing results to Parquet for data lake workflows

## Setup

```bash
uv add grafeo pyarrow duckdb
```

```python
import grafeo
import duckdb
```

## Build a Social Graph

```python
db = grafeo.GrafeoDB()

# People in three cities
db.execute("""
    INSERT (:Person {name: 'Alix', age: 30, city: 'Amsterdam'})
    INSERT (:Person {name: 'Gus', age: 35, city: 'Berlin'})
    INSERT (:Person {name: 'Vincent', age: 28, city: 'Paris'})
    INSERT (:Person {name: 'Jules', age: 42, city: 'Amsterdam'})
    INSERT (:Person {name: 'Mia', age: 26, city: 'Berlin'})
    INSERT (:Person {name: 'Butch', age: 38, city: 'Paris'})
    INSERT (:Person {name: 'Django', age: 31, city: 'Amsterdam'})
    INSERT (:Person {name: 'Shosanna', age: 29, city: 'Berlin'})
""")

# Companies
db.execute("""
    INSERT (:Company {name: 'GraphTech', industry: 'Technology', founded: 2018})
    INSERT (:Company {name: 'DataFlow', industry: 'Technology', founded: 2020})
    INSERT (:Company {name: 'CanalBrew', industry: 'Food & Drink', founded: 2015})
""")

# Relationships
db.execute("""
    MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
    INSERT (a)-[:KNOWS {since: 2019}]->(b)
""")
db.execute("""
    MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Jules'})
    INSERT (a)-[:KNOWS {since: 2020}]->(b)
""")
db.execute("""
    MATCH (a:Person {name: 'Gus'}), (b:Person {name: 'Vincent'})
    INSERT (a)-[:KNOWS {since: 2021}]->(b)
""")
db.execute("""
    MATCH (a:Person {name: 'Vincent'}), (b:Person {name: 'Butch'})
    INSERT (a)-[:KNOWS {since: 2018}]->(b)
""")
db.execute("""
    MATCH (a:Person {name: 'Jules'}), (b:Person {name: 'Django'})
    INSERT (a)-[:KNOWS {since: 2022}]->(b)
""")
db.execute("""
    MATCH (a:Person {name: 'Mia'}), (b:Person {name: 'Shosanna'})
    INSERT (a)-[:KNOWS {since: 2023}]->(b)
""")
db.execute("""
    MATCH (a:Person {name: 'Django'}), (b:Person {name: 'Shosanna'})
    INSERT (a)-[:KNOWS {since: 2021}]->(b)
""")

# Employment edges
db.execute("""
    MATCH (p:Person {name: 'Alix'}), (c:Company {name: 'GraphTech'})
    INSERT (p)-[:WORKS_AT {role: 'Engineer', started: 2019}]->(c)
""")
db.execute("""
    MATCH (p:Person {name: 'Gus'}), (c:Company {name: 'GraphTech'})
    INSERT (p)-[:WORKS_AT {role: 'Manager', started: 2018}]->(c)
""")
db.execute("""
    MATCH (p:Person {name: 'Vincent'}), (c:Company {name: 'DataFlow'})
    INSERT (p)-[:WORKS_AT {role: 'Analyst', started: 2021}]->(c)
""")
db.execute("""
    MATCH (p:Person {name: 'Jules'}), (c:Company {name: 'CanalBrew'})
    INSERT (p)-[:WORKS_AT {role: 'Owner', started: 2015}]->(c)
""")
db.execute("""
    MATCH (p:Person {name: 'Mia'}), (c:Company {name: 'DataFlow'})
    INSERT (p)-[:WORKS_AT {role: 'Designer', started: 2022}]->(c)
""")

print(f"Graph: {db.node_count} nodes, {db.edge_count} edges")
```

```title="Output"
Graph: 11 nodes, 12 edges
```

## Query Nodes with SQL

Export all nodes as an Arrow Table and query them directly in DuckDB:

```python
nodes = db.nodes_to_arrow()

print(nodes.schema)
# id: uint64
# labels: list<item: string>
# name: string
# age: int64
# city: string
# industry: string
# founded: int64
```

DuckDB can query Arrow Tables in place, with no data copying:

```python
result = duckdb.sql("""
    SELECT city, COUNT(*) AS count, ROUND(AVG(age), 1) AS avg_age
    FROM nodes
    WHERE list_contains(labels, 'Person')
    GROUP BY city
    ORDER BY count DESC
""")
print(result)
```

```title="Output"
┌───────────┬───────┬─────────┐
│   city    │ count │ avg_age │
│  varchar  │ int64 │ double  │
├───────────┼───────┼─────────┤
│ Amsterdam │     3 │    34.3 │
│ Berlin    │     3 │    30.0 │
│ Paris     │     2 │    33.0 │
└───────────┴───────┴─────────┘
```

### Label Distribution

```python
result = duckdb.sql("""
    SELECT
        unnest(labels) AS label,
        COUNT(*) AS count
    FROM nodes
    GROUP BY label
    ORDER BY count DESC
""")
print(result)
```

```title="Output"
┌─────────┬───────┐
│  label  │ count │
│ varchar │ int64 │
├─────────┼───────┤
│ Person  │     8 │
│ Company │     3 │
└─────────┴───────┘
```

## Edge Analytics

Export edges and register both tables in a DuckDB connection for joins:

```python
edges = db.edges_to_arrow()

print(edges.schema)
# id: uint64
# type: string
# source: uint64
# target: uint64
# since: int64
# role: string
# started: int64
```

```python
conn = duckdb.connect()
conn.register("nodes", nodes)
conn.register("edges", edges)
```

### Most Connected People

Count outgoing and incoming KNOWS relationships per person:

```python
result = conn.sql("""
    WITH connections AS (
        SELECT source AS node_id, COUNT(*) AS out_degree
        FROM edges
        WHERE type = 'KNOWS'
        GROUP BY source
        UNION ALL
        SELECT target AS node_id, COUNT(*) AS out_degree
        FROM edges
        WHERE type = 'KNOWS'
        GROUP BY target
    )
    SELECT n.name, SUM(c.out_degree) AS total_connections
    FROM connections c
    JOIN nodes n ON c.node_id = n.id
    GROUP BY n.name
    ORDER BY total_connections DESC
    LIMIT 5
""")
print(result)
```

```title="Output"
┌──────────┬───────────────────┐
│   name   │ total_connections │
│ varchar  │      int128       │
├──────────┼───────────────────┤
│ Django   │                 2 │
│ Shosanna │                 2 │
│ Alix     │                 2 │
│ Gus      │                 2 │
│ Vincent  │                 2 │
└──────────┴───────────────────┘
```

### Company Headcount

```python
result = conn.sql("""
    SELECT
        c.name AS company,
        c.industry,
        COUNT(*) AS employees,
        ROUND(AVG(p.age), 1) AS avg_employee_age
    FROM edges e
    JOIN nodes p ON e.source = p.id
    JOIN nodes c ON e.target = c.id
    WHERE e.type = 'WORKS_AT'
    GROUP BY c.name, c.industry
    ORDER BY employees DESC
""")
print(result)
```

```title="Output"
┌───────────┬──────────────┬───────────┬──────────────────┐
│  company  │   industry   │ employees │ avg_employee_age │
│  varchar  │   varchar    │   int64   │      double      │
├───────────┼──────────────┼───────────┼──────────────────┤
│ GraphTech │ Technology   │         2 │             32.5 │
│ DataFlow  │ Technology   │         2 │             27.0 │
│ CanalBrew │ Food & Drink │         1 │             42.0 │
└───────────┴──────────────┴───────────┴──────────────────┘
```

## Graph-to-Warehouse Pattern

Use Grafeo for graph pattern matching, then hand the results to DuckDB for aggregation. This combines the strengths of both: Grafeo handles traversals and pattern matching, DuckDB handles analytical SQL.

### Step 1: Pattern match in Grafeo

```python
# Find all two-hop connections: who knows someone who knows someone?
result = db.execute("""
    MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)
    WHERE a <> c
    RETURN a.name AS person, b.name AS via, c.name AS reaches
""")

# Export the result to an Arrow Table
friend_paths = result.to_arrow()
print(f"Found {friend_paths.num_rows} two-hop paths")
```

```title="Output"
Found 5 two-hop paths
```

### Step 2: Analyze in DuckDB

```python
result = duckdb.sql("""
    SELECT
        person,
        COUNT(DISTINCT reaches) AS reachable_people,
        LIST(DISTINCT reaches) AS who
    FROM friend_paths
    GROUP BY person
    ORDER BY reachable_people DESC
""")
print(result)
```

This pattern works well for any graph-then-aggregate workflow:

1. **MATCH** a subgraph pattern with GQL (paths, neighborhoods, motifs)
2. **Export** the tabular result to Arrow via `result.to_arrow()`
3. **Analyze** in DuckDB with window functions, pivots, GROUP BY

## RDF/SPARQL + DuckDB

If you work with RDF data, you can run SPARQL queries in Grafeo and analyze the results in DuckDB.

### Load RDF triples

```python
db_rdf = grafeo.GrafeoDB()

db_rdf.execute_sparql("""
    PREFIX ex: <http://example.org/>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>

    INSERT DATA {
        ex:alix  a foaf:Person ; foaf:name "Alix"  ; ex:city "Amsterdam" ; ex:age 30 .
        ex:gus   a foaf:Person ; foaf:name "Gus"   ; ex:city "Berlin"    ; ex:age 35 .
        ex:vincent a foaf:Person ; foaf:name "Vincent" ; ex:city "Paris"  ; ex:age 28 .
        ex:jules a foaf:Person ; foaf:name "Jules"  ; ex:city "Amsterdam" ; ex:age 42 .

        ex:alix foaf:knows ex:gus .
        ex:alix foaf:knows ex:jules .
        ex:gus  foaf:knows ex:vincent .
    }
""")
```

### Query with SPARQL, analyze with SQL

```python
result = db_rdf.execute_sparql("""
    PREFIX ex: <http://example.org/>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>

    SELECT ?name ?city ?age
    WHERE {
        ?person a foaf:Person ;
                foaf:name ?name ;
                ex:city ?city ;
                ex:age ?age .
    }
""")

# Export SPARQL results to Arrow
people_table = result.to_arrow()

# Aggregate in DuckDB
stats = duckdb.sql("""
    SELECT city, COUNT(*) AS count, AVG(age) AS avg_age
    FROM people_table
    GROUP BY city
    ORDER BY count DESC
""")
print(stats)
```

The SPARQL query handles the graph pattern matching (triple patterns, OPTIONAL joins, property paths). DuckDB handles the analytical heavy lifting (aggregation, window functions, HAVING clauses).

## Export to Parquet

DuckDB can write query results directly to Parquet files, turning your graph data into data lake assets:

```python
conn = duckdb.connect()
conn.register("nodes", db.nodes_to_arrow())
conn.register("edges", db.edges_to_arrow())

# Export a summary table to Parquet
conn.sql("""
    COPY (
        SELECT n.name, n.city, n.age, COUNT(e.id) AS connections
        FROM nodes n
        LEFT JOIN edges e ON n.id = e.source AND e.type = 'KNOWS'
        WHERE list_contains(n.labels, 'Person')
        GROUP BY n.name, n.city, n.age
        ORDER BY connections DESC
    ) TO 'people_connections.parquet' (FORMAT PARQUET)
""")
```

This Parquet file can then be loaded into any data warehouse, shared with colleagues, or queried later with DuckDB, Spark, or Polars.

## Performance Notes

| Aspect | Detail |
|--------|--------|
| **Zero-copy transfer** | Arrow Tables pass to DuckDB without copying data. The column pointers are shared directly. |
| **Scale** | For graphs with hundreds of thousands of nodes, Arrow export is 10-100x faster than converting through pandas row by row. |
| **Memory** | DuckDB scans the Arrow Table in-place. No second copy of the data exists in memory. |
| **Predicate pushdown** | DuckDB pushes filters down into Arrow scans, reading only the columns and rows it needs. |
| **Parallelism** | DuckDB's vectorized engine processes Arrow batches in parallel across CPU cores. |

## When to Use Each

| Task | Use |
|------|-----|
| Graph traversal, shortest path, pattern matching | Grafeo (GQL) |
| Aggregation, GROUP BY, window functions, pivots | DuckDB (SQL) |
| RDF triple patterns, OPTIONAL, UNION | Grafeo (SPARQL) |
| Export to Parquet, CSV, JSON | DuckDB |
| Subgraph extraction then analytics | Grafeo then DuckDB |
| Full-text or vector search | Grafeo |

## Next Steps

- [NetworkX Integration](networkx-integration.md) for graph algorithms and visualization
- [Graph Visualization](graph-visualization.md) for interactive exploration with anywidget-graph
- [Fraud Detection](fraud-detection-example.md) for a practical graph analytics example
- [Python API](../../user-guide/python/index.md) for the full Python binding reference
