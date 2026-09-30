---
title: grafeo.GrafeoDB
description: Database class reference.
tags:
  - api
  - python
---

# grafeo.GrafeoDB

The main database class.

## Constructor

```python
GrafeoDB(
    path: Optional[str] = None,
    *,
    cdc: bool = False,
    shuffle_unordered: bool = False
)
```

### Parameters

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `path` | `str` | `None` | Database file path (None for in-memory) |
| `cdc` | `bool` | `False` | Enable change data capture (keyword-only). When `True`, mutations are tracked and queryable via `node_history()` / `edge_history()`. |
| `shuffle_unordered` | `bool` | `False` | Return the rows of every query without `ORDER BY` in random order (keyword-only). For tests: without `ORDER BY` the row order is unspecified, and this finds code that relies on it. |

### Examples

```python
# In-memory database
db = grafeo.GrafeoDB()

# Persistent database
db = grafeo.GrafeoDB("my_graph.db")
```

## Static Constructors

### open()

Open an existing database.

```python
@staticmethod
def open(path: str) -> GrafeoDB
```

### open_read_only()

Open a database in read-only mode. Uses a shared file lock, so multiple processes can read the same `.grafeo` file concurrently. Mutations will raise an error.

```python
@staticmethod
def open_read_only(path: str) -> GrafeoDB
```

```python
db = GrafeoDB.open_read_only("./my_graph.grafeo")
result = db.execute("MATCH (n) RETURN n LIMIT 10")
```

### open_in_memory()

Open a persistent database file and load it entirely into memory. The returned database has no connection to the original file: changes will not be written back.

```python
@staticmethod
def open_in_memory(path: str) -> GrafeoDB
```

```python
db = GrafeoDB.open_in_memory("./mydb")
db.create_node(["Test"], {})  # does not affect the file
```

## Query Methods

### execute()

Execute a GQL query.

```python
def execute(self, query: str, params: Optional[Dict] = None) -> QueryResult
```

### execute_cypher()

Execute a Cypher query.

```python
def execute_cypher(self, query: str, params: Optional[Dict] = None) -> QueryResult
```

### execute_gremlin()

Execute a Gremlin query.

```python
def execute_gremlin(self, query: str, params: Optional[Dict] = None) -> QueryResult
```

### execute_graphql()

Execute a GraphQL query.

```python
def execute_graphql(self, query: str, params: Optional[Dict] = None) -> QueryResult
```

### execute_sparql()

Execute a SPARQL query.

```python
def execute_sparql(self, query: str, params: Optional[Dict] = None) -> QueryResult
```

### explain_sparql()

Execute a SPARQL query with `EXPLAIN` or `EXPLAIN ANALYZE` to inspect the physical plan tree. Returns a `QueryResult` containing the query plan as text rows. Use `EXPLAIN` for the plan without execution, or `EXPLAIN ANALYZE` for a profiled execution that includes timing and row counts.

```python
def explain_sparql(self, query: str) -> QueryResult
```

```python
# View the query plan without executing
result = db.explain_sparql("EXPLAIN SELECT ?name WHERE { ?p a :Person ; :name ?name }")
for row in result:
    print(row)

# Profiled execution with timing
result = db.explain_sparql("EXPLAIN ANALYZE SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100")
for row in result:
    print(row)
```

### execute_sql()

Execute a SQL/PGQ query.

```python
def execute_sql(self, query: str, params: Optional[Dict] = None) -> QueryResult
```

### execute_async()

Execute a GQL query asynchronously. Returns a Python awaitable for use with asyncio.

```python
def execute_async(self, query: str, params: Optional[Dict] = None) -> Awaitable[AsyncQueryResult]
```

```python
import asyncio

async def main():
    db = GrafeoDB()
    result = await db.execute_async("MATCH (n:Person) RETURN n")
    for row in result:
        print(row)

asyncio.run(main())
```

### execute_at_epoch()

Execute a GQL query at a specific historical epoch. Returns results as they would have appeared at that point in time.

```python
def execute_at_epoch(self, query: str, epoch: int, params: Optional[Dict] = None) -> QueryResult
```

```python
result = db.execute_at_epoch("MATCH (n:Server) RETURN n.status", epoch=5)
```

## Node Operations

### create_node()

Create a node with labels and properties.

```python
def create_node(self, labels: List[str], properties: Optional[Dict[str, Any]] = None) -> Node
```

### get_node()

Get a node by ID. Returns `None` if the node does not exist.

```python
def get_node(self, id: int) -> Optional[Node]
```

### delete_node()

Delete a node by ID. Returns `True` if the node existed and was deleted.

```python
def delete_node(self, id: int) -> bool
```

### add_node_label()

Add a label to an existing node. Returns `True` if the label was added, `False` if the node does not exist or already has the label.

```python
def add_node_label(self, node_id: int, label: str) -> bool
```

### remove_node_label()

Remove a label from a node. Returns `True` if the label was removed, `False` if the node does not exist or does not have the label.

```python
def remove_node_label(self, node_id: int, label: str) -> bool
```

### get_node_labels()

Get all labels for a node. Returns `None` if the node does not exist.

```python
def get_node_labels(self, node_id: int) -> Optional[List[str]]
```

### set_node_property()

Set a property on a node.

```python
def set_node_property(self, node_id: int, key: str, value: Any) -> None
```

### remove_node_property()

Remove a property from a node. Returns `True` if the property existed and was removed.

```python
def remove_node_property(self, node_id: int, key: str) -> bool
```

### get_nodes_by_label()

Get all nodes with a specific label and their properties. Supports pagination with `limit` and `offset`. More efficient than calling `get_node()` in a loop because it batches property lookups.

```python
def get_nodes_by_label(
    self,
    label: str,
    limit: Optional[int] = None,
    offset: int = 0
) -> List[Tuple[int, Dict[str, Any]]]
```

Returns a list of `(node_id, properties_dict)` tuples.

```python
people = db.get_nodes_by_label("Person", limit=100)
for node_id, props in people:
    print(f"Node {node_id}: {props}")

# Pagination
page = db.get_nodes_by_label("Person", limit=50, offset=100)
```

### get_property_batch()

Get a specific property value for multiple nodes at once. More efficient than calling `get_node()` in a loop when you only need one property.

```python
def get_property_batch(self, node_ids: List[int], property: str) -> List[Optional[Any]]
```

```python
ages = db.get_property_batch([1, 2, 3, 4, 5], "age")
for node_id, age in zip([1, 2, 3, 4, 5], ages):
    if age is not None:
        print(f"Node {node_id} is {age} years old")
```

## Edge Operations

### create_edge()

Create an edge between two nodes.

```python
def create_edge(
    self,
    source_id: int,
    target_id: int,
    edge_type: str,
    properties: Optional[Dict[str, Any]] = None
) -> Edge
```

### get_edge()

Get an edge by ID. Returns `None` if the edge does not exist.

```python
def get_edge(self, id: int) -> Optional[Edge]
```

### delete_edge()

Delete an edge by ID. Returns `True` if the edge existed and was deleted.

```python
def delete_edge(self, id: int) -> bool
```

### set_edge_property()

Set a property on an edge.

```python
def set_edge_property(self, edge_id: int, key: str, value: Any) -> None
```

### remove_edge_property()

Remove a property from an edge. Returns `True` if the property existed and was removed.

```python
def remove_edge_property(self, edge_id: int, key: str) -> bool
```

## Upserts

Create or update nodes and edges by a key property, many rows in one statement. The rows are checked like
a query (constraints, schema), and a call writes all of its rows or none. Rows apply in order: a key
repeated within one call creates one node or edge, which the later rows update. Both return a dict with
`created`, `updated`, `skipped` and `skipped_rows` (the indices of the skipped rows, at most 1,000). Graph
handles (`db.graph(name)`) have the same methods.

### upsert_nodes()

One node per row, matched by `key` and all of `labels`. A row without the key is skipped. By default a
row's properties are merged into the node's and none is removed; with `replace=True` they become exactly
the row's. Labels are never removed.

```python
def upsert_nodes(self, labels: list[str], rows: list[dict], key: str = "id", replace: bool = False) -> dict
```

```python
db.upsert_nodes(["Graph", "File"], [{"id": "f1", "size": 3}, {"id": "f2", "size": 5}])
# {'created': 2, 'updated': 0, 'skipped': 0, 'skipped_rows': []}
```

### upsert_edges()

One edge of `edge_type` per row, between the nodes whose `endpoint_key` is the row's `src_field` and
`dst_field` value (restricted to `endpoint_labels` when given). The edge is identified by its endpoints,
type and `key`; every other field of the row is an edge property. A row whose endpoint does not exist, or
without the key, is skipped, never created. A property index on `endpoint_key` makes the lookups fast.

```python
def upsert_edges(
    self,
    edge_type: str,
    rows: list[dict],
    key: str = "id",
    endpoint_key: str = "id",
    endpoint_labels: list[str] | None = None,
    src_field: str = "src",
    dst_field: str = "dst",
    replace: bool = False,
) -> dict
```

```python
db.create_property_index("id")
db.upsert_edges("USES", [{"src": "f1", "dst": "f2", "id": "u1", "weight": 1}])
# {'created': 1, 'updated': 0, 'skipped': 0, 'skipped_rows': []}
```

## DataFrame Integration

These methods convert between Grafeo and pandas/polars DataFrames. Requires `pandas` or `polars` to be installed (`uv add pandas` or `uv add polars`).

### nodes_df()

Export all nodes as a pandas DataFrame. Columns: `id` (int), `labels` (list[str]), plus one column per unique property key. Missing properties are `None`.

```python
def nodes_df(self) -> pandas.DataFrame
```

```python
df = db.nodes_df()
print(df[df["labels"].apply(lambda l: "Person" in l)])
```

### edges_df()

Export all edges as a pandas DataFrame. Columns: `id` (int), `source` (int), `target` (int), `type` (str), plus one column per unique property key. Missing properties are `None`.

```python
def edges_df(self) -> pandas.DataFrame
```

```python
df = db.edges_df()
print(df[df["type"] == "KNOWS"])
```

### import_df()

Bulk import nodes or edges from a pandas or polars DataFrame. Returns the number of rows imported.

```python
def import_df(
    self,
    df: DataFrame,
    mode: str,                    # "nodes" or "edges"
    *,
    label: Optional[str | List[str]] = None,  # required for mode="nodes"
    edge_type: Optional[str] = None,          # required for mode="edges"
    source: str = "source",       # column name for source node IDs
    target: str = "target"        # column name for target node IDs
) -> int
```

**Node import** (`mode='nodes'`): each row becomes a node. The `label` parameter sets the label(s). All DataFrame columns become properties.

**Edge import** (`mode='edges'`): each row becomes an edge. The `source` and `target` columns must contain integer node IDs. Remaining columns become edge properties.

```python
import pandas as pd

# Import nodes
people = pd.DataFrame({"name": ["Alix", "Gus"], "age": [30, 25]})
db.import_df(people, mode="nodes", label="Person")

# Import edges (source/target are node IDs)
edges = pd.DataFrame({"source": [0, 1], "target": [1, 0], "since": [2020, 2021]})
db.import_df(edges, mode="edges", edge_type="KNOWS")
```

## Arrow Bulk Export

Zero-copy bulk export using Apache Arrow. These methods are faster than `nodes_df()`/`edges_df()` for large graphs because they build Arrow record batches directly in Rust, avoiding row-by-row Python object creation.

!!! note
    `nodes_df()` and `edges_df()` now auto-detect pyarrow at runtime. When pyarrow is available, they use the Arrow fast path internally, so you get the same speed as `nodes_to_pandas()` without changing existing code.

**Node schema:** `id` (uint64), `labels` (list&lt;utf8&gt;), plus one column per unique property key.

**Edge schema:** `id` (uint64), `type` (utf8), `source` (uint64), `target` (uint64), plus one column per unique property key.

### nodes_to_arrow()

Export all nodes as a `pyarrow.Table`. Requires pyarrow (`uv add pyarrow`).

```python
def nodes_to_arrow(self) -> pyarrow.Table
```

```python
table = db.nodes_to_arrow()
print(table.schema)
print(table.to_pandas())
```

### edges_to_arrow()

Export all edges as a `pyarrow.Table`. Requires pyarrow (`uv add pyarrow`).

```python
def edges_to_arrow(self) -> pyarrow.Table
```

```python
table = db.edges_to_arrow()
print(table.filter(table.column("type") == "KNOWS"))
```

### nodes_to_polars()

Export all nodes as a `polars.DataFrame`. Requires polars (`uv add polars`), but does not require pyarrow. Uses the Arrow IPC format internally for zero-copy transfer.

```python
def nodes_to_polars(self) -> polars.DataFrame
```

```python
df = db.nodes_to_polars()
print(df.filter(pl.col("labels").list.contains("Person")))
```

### edges_to_polars()

Export all edges as a `polars.DataFrame`. Requires polars (`uv add polars`), but does not require pyarrow.

```python
def edges_to_polars(self) -> polars.DataFrame
```

```python
df = db.edges_to_polars()
print(df.filter(pl.col("type") == "KNOWS"))
```

### nodes_to_pandas()

Export all nodes as a `pandas.DataFrame` via the Arrow fast path. Requires both pandas and pyarrow (`uv add pandas pyarrow`). Faster than `nodes_df()` on older versions, but equivalent now that `nodes_df()` auto-detects pyarrow.

```python
def nodes_to_pandas(self) -> pandas.DataFrame
```

```python
df = db.nodes_to_pandas()
print(df[df["labels"].apply(lambda l: "Person" in l)])
```

### edges_to_pandas()

Export all edges as a `pandas.DataFrame` via the Arrow fast path. Requires both pandas and pyarrow (`uv add pandas pyarrow`).

```python
def edges_to_pandas(self) -> pandas.DataFrame
```

```python
df = db.edges_to_pandas()
print(df.groupby("type").size())
```

## Batch Operations

### batch_create_nodes()

Bulk-insert nodes with a single vector property each. All nodes get the same label. Much faster than calling `create_node()` in a loop.

```python
def batch_create_nodes(self, label: str, property: str, vectors: List[List[float]]) -> List[int]
```

Returns a list of created node IDs.

```python
ids = db.batch_create_nodes("Doc", "embedding", [[1.0, 0.0], [0.0, 1.0]])
```

### batch_create_nodes_with_props()

Batch-create nodes with full property maps. Each dict in the list is a complete
set of properties for one node. Vector values are auto-inserted into matching
vector indexes.

```python
def batch_create_nodes_with_props(self, label: str | list[str], properties_list: List[Dict[str, Any]]) -> List[int]
```

| Parameter | Type | Description |
| --- | --- | --- |
| `label` | `str` or `list[str]` | Label for all created nodes, or labels they all get |
| `properties_list` | `list[dict]` | One property dict per node |

Returns a list of created node IDs. The batch is one transaction: if a node breaks a constraint, none is
created.

```python
ids = db.batch_create_nodes_with_props("Person", [
    {"name": "Alix", "age": 30},
    {"name": "Gus", "age": 25},
])
db.batch_create_nodes_with_props(["Graph", "File"], [{"id": "f1"}, {"id": "f2"}])
```

### batch_create_edges()

Batch-create edges, each with its own type and properties, from `(src, dst, type)` or
`(src, dst, type, properties)` tuples. The batch is one transaction: if an edge names a node that does not
exist or breaks the schema, none is created.

```python
def batch_create_edges(self, edges: List[tuple]) -> List[int]
```

Returns a list of created edge IDs, in input order.

```python
alix, gus = db.batch_create_nodes_with_props("Person", [{"name": "Alix"}, {"name": "Gus"}])
ids = db.batch_create_edges([
    (alix, gus, "KNOWS", {"since": 2020}),
    (gus, alix, "KNOWS"),
])
```

### batch_vector_search()

Search for nearest neighbors of multiple query vectors in parallel across all available CPU cores.

```python
def batch_vector_search(
    self,
    label: str,
    property: str,
    queries: List[List[float]],
    k: int,
    ef: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None
) -> List[List[Tuple[int, float]]]
```

Returns a list of results per query. Each result is a list of `(node_id, distance)` tuples.

```python
results = db.batch_vector_search("Doc", "embedding", [[1.0, 0.0], [0.0, 1.0]], k=5)
for i, hits in enumerate(results):
    print(f"Query {i}: {hits}")
```

## Search

### vector_search()

Search for the k nearest neighbors of a query vector using the HNSW index.

```python
def vector_search(
    self,
    label: str,
    property: str,
    query: List[float],
    k: int,
    ef: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None
) -> List[Tuple[int, float]]
```

Returns a list of `(node_id, distance)` tuples sorted by distance ascending (lower distance = more similar). The distance scale depends on the metric configured at index creation: cosine `[0, 2]`, euclidean `[0, inf)`, dot_product (negated, so lower = higher similarity), manhattan `[0, inf)`.

```python
results = db.vector_search("Doc", "embedding", [1.0, 0.0, 0.0], k=10, ef=200)
for node_id, distance in results:
    print(f"Node {node_id}: distance={distance:.4f}")

# With property filters
results = db.vector_search("Doc", "embedding", query, k=10, filters={"user_id": 42})
```

### mmr_search()

Maximal Marginal Relevance search. Balances relevance to the query with diversity among results, avoiding redundant results in RAG pipelines.

```python
def mmr_search(
    self,
    label: str,
    property: str,
    query: List[float],
    k: int,
    fetch_k: Optional[int] = None,       # initial candidates, default 4*k
    lambda_mult: Optional[float] = None,  # 0=diverse, 1=relevant, default 0.5
    ef: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None
) -> List[Tuple[int, float]]
```

Returns a list of `(node_id, distance)` tuples in MMR selection order. The `distance` values are identical to those returned by `vector_search()` for the same nodes (lower = more similar). The list ordering reflects MMR's relevance-diversity balance, not pure distance sorting.

```python
results = db.mmr_search("Doc", "embedding", [1.0, 0.0, 0.0], k=4, lambda_mult=0.5)
for node_id, distance in results:
    print(f"Node {node_id}: distance={distance:.4f}")
```

### text_search()

BM25 full-text search. Requires the `text-index` feature and a text index created with `create_text_index()`.

```python
def text_search(self, label: str, property: str, query: str, k: int) -> List[Tuple[int, float]]
```

Returns a list of `(node_id, score)` tuples sorted by descending relevance (higher score = more relevant). BM25 scores are unbounded positive floats; compare them only within a single query's results.

```python
db.create_text_index("Article", "title")
results = db.text_search("Article", "title", "graph database", k=10)
for node_id, score in results:
    print(f"Node {node_id}: score={score:.4f}")
```

### hybrid_search()

Combined text and vector search using Reciprocal Rank Fusion (RRF) or weighted fusion. Requires the `hybrid-search` feature. For best results, create both a text index (`create_text_index()`) and a vector index (`create_vector_index()`). If either index is missing, that source is silently omitted from fusion.

```python
def hybrid_search(
    self,
    label: str,
    text_property: str,
    vector_property: str,
    query_text: str,
    k: int,
    query_vector: Optional[List[float]] = None,
    fusion: Optional[str] = None,          # "rrf" (default) or "weighted"
    weights: Optional[List[float]] = None, # [text_weight, vector_weight]
    rrf_k: Optional[int] = None
) -> List[Tuple[int, float]]
```

Returns a list of `(node_id, score)` tuples sorted by fused score **descending** (higher = more relevant). These are fusion scores, **not** distances. With RRF (default), scores are `sum(1/(k+rank))` across sources. With weighted fusion, scores are normalized to `[0, 1]` and combined with explicit weights.

!!! warning "Score convention differs from vector_search"
    `hybrid_search()` returns fusion scores where higher = better.
    `vector_search()` returns distances where lower = better.
    For temporal decay, **multiply** fusion scores but **divide** distances.

```python
results = db.hybrid_search(
    "Article", "title", "embedding",
    "graph databases", k=10,
    query_vector=[1.0, 0.0, 0.0]
)
```

## Property Indexes

### create_property_index()

Create an index on a node property for O(1) lookups.

```python
def create_property_index(self, property: str) -> None
```

### drop_property_index()

Remove a property index. Returns `True` if the index existed and was removed.

```python
def drop_property_index(self, property: str) -> bool
```

### has_property_index()

Check whether a property has an index.

```python
def has_property_index(self, property: str) -> bool
```

### find_nodes_by_property()

Find all nodes with a specific property value. O(1) if the property is indexed, O(n) otherwise.

```python
def find_nodes_by_property(self, property: str, value: Any) -> List[int]
```

```python
db.create_property_index("email")
node_ids = db.find_nodes_by_property("email", "alix@example.com")
```

## Vector Index Management

### create_vector_index()

Create an HNSW vector similarity index on a node property.

```python
def create_vector_index(
    self,
    label: str,
    property: str,
    dimensions: Optional[int] = None,
    metric: Optional[str] = None,       # "cosine" (default), "euclidean", "dot_product", "manhattan"
    m: Optional[int] = None,            # HNSW links per node, default 16
    ef_construction: Optional[int] = None  # build beam width, default 128
) -> None
```

### drop_vector_index()

Drop a vector index. Returns `True` if the index existed and was removed.

```python
def drop_vector_index(self, label: str, property: str) -> bool
```

### rebuild_vector_index()

Rebuild a vector index from scratch, preserving its configuration (dimensions, metric, M, ef_construction).

!!! note "Auto-sync: rebuild is rarely needed"
    Vector indexes auto-sync when you call `set_node_property()`, `batch_create_nodes()`, or `batch_create_nodes_with_props()` with vector data. You only need `rebuild_vector_index()` after importing data through non-standard paths or to compact the index after many deletions.

```python
def rebuild_vector_index(self, label: str, property: str) -> None
```

## Text Index Management

Requires the `text-index` feature.

### create_text_index()

Create a BM25 text index on a node property. The index is automatically kept in sync as nodes are created, updated, or deleted. You do not need to call `rebuild_text_index()` after normal write operations.

```python
def create_text_index(self, label: str, property: str) -> None
```

### drop_text_index()

Drop a text index. Returns `True` if the index existed and was removed.

```python
def drop_text_index(self, label: str, property: str) -> bool
```

### rebuild_text_index()

Rebuild a text index from scratch. Text indexes auto-sync on normal writes; you only need this after importing data through non-standard paths.

```python
def rebuild_text_index(self, label: str, property: str) -> None
```

## Transaction Methods

### begin_transaction()

Start a new transaction. Returns a `Transaction` object that can be used as a context manager. The `isolation_level` parameter accepts a string (e.g., `"snapshot"`, `"serializable"`).

```python
def begin_transaction(self, isolation_level: Optional[str] = None) -> Transaction
```

```python
with db.begin_transaction() as tx:
    tx.execute("INSERT (:Person {name: 'Alix'})")
    tx.execute("INSERT (:Person {name: 'Gus'})")
    tx.commit()  # both nodes created atomically

# With explicit isolation level
with db.begin_transaction("serializable") as tx:
    tx.execute("MATCH (n:Counter) SET n.val = n.val + 1")
    tx.commit()
```

### begin_transaction_with_cdc()

Start a transaction with an explicit CDC (change data capture) override. When `cdc_enabled` is `True`, mutations in this transaction are tracked regardless of the database-level default. When `False`, tracking is disabled for this transaction only. Requires the `cdc` feature.

```python
def begin_transaction_with_cdc(
    self,
    cdc_enabled: bool,
    isolation_level: Optional[str] = None
) -> Transaction
```

```python
with db.begin_transaction_with_cdc(True) as tx:
    tx.execute("INSERT (:Person {name: 'Alix'})")
    tx.commit()
# This transaction's changes appear in node_history()
```

## Schema Context

### set_schema()

Set the current schema for subsequent `execute()` calls.

```python
def set_schema(self, name: str) -> None
```

```python
db.set_schema("reporting")
result = db.execute("SHOW GRAPH TYPES")  # only sees types in 'reporting'
```

### reset_schema()

Clear the current schema context. Subsequent `execute()` calls will use the default namespace.

```python
def reset_schema(self) -> None
```

### current_schema()

Returns the current schema name, or `None` if no schema is set.

```python
def current_schema(self) -> Optional[str]
```

## SHACL Validation

Validate graph data against [SHACL](https://www.w3.org/TR/shacl/) (Shapes Constraint Language) shapes. Requires the `triple-store` feature.

### validate_shacl()

Validate the current graph against a SHACL shapes graph provided as a Turtle string. Returns a dict with the validation report.

```python
def validate_shacl(self, shapes_graph: str) -> Dict[str, Any]
```

**Return value keys:**

| Key | Type | Description |
| --- | ---- | ----------- |
| `conforms` | `bool` | `True` if no violations were found |
| `results` | `list[dict]` | One dict per violation |
| `results_text` | `str` | Human-readable summary of all violations |

**Each result dict contains:**

| Key | Type | Description |
| --- | ---- | ----------- |
| `focus_node` | `str` | The node that was validated |
| `severity` | `str` | `"Violation"`, `"Warning"`, or `"Info"` |
| `source_constraint_component` | `str` | The SHACL constraint that failed |
| `source_shape` | `str` | The shape that triggered the violation |
| `value` | `str` (optional) | The invalid value, if applicable |
| `message` | `str` (optional) | A human-readable description of the violation |

```python
shapes = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/> .

ex:PersonShape a sh:NodeShape ;
    sh:targetClass ex:Person ;
    sh:property [
        sh:path ex:name ;
        sh:minCount 1 ;
        sh:datatype xsd:string ;
    ] .
"""

report = db.validate_shacl(shapes)
if report["conforms"]:
    print("All data is valid")
else:
    for r in report["results"]:
        print(f"{r['severity']}: {r['focus_node']} - {r['source_constraint_component']}")
    print(report["results_text"])
```

## Admin Methods

### info()

Get database information. Returns a dict with keys: `mode`, `node_count`, `edge_count`, `is_persistent`, `path`, `wal_enabled`, `version`.

```python
def info(self) -> Dict[str, Any]
```

### detailed_stats()

Get detailed statistics. Returns a dict with keys: `node_count`, `edge_count`, `label_count`, `edge_type_count`, `property_key_count`, `index_count`, `memory_bytes`, `disk_bytes`.

```python
def detailed_stats(self) -> Dict[str, Any]
```

### memory_usage()

Returns a hierarchical memory usage breakdown. Walks all internal structures (store, indexes, MVCC chains, caches, string pools, buffer manager) and returns estimated heap bytes.

```python
def memory_usage(self) -> Dict[str, Any]
```

Returns a dict with keys: `total_bytes`, `store`, `indexes`, `mvcc`, `caches`, `string_pool`, `buffer_manager`. Each sub-key is itself a dict with a detailed breakdown.

```python
usage = db.memory_usage()
print(f"Total: {usage['total_bytes']} bytes")
print(f"Store: {usage['store']['total_bytes']} bytes")
```

### schema()

Get schema information. Returns a dict with keys: `labels` (list of dicts with `name` and `count`), `edge_types` (list of dicts), `property_keys` (list of strings).

```python
def schema(self) -> Dict[str, Any]
```

### validate()

Validate database integrity. Returns a list of error dicts (empty list means valid). Each error dict has keys: `code`, `message`, `context`.

```python
def validate(self) -> List[Dict[str, str]]
```

```python
errors = db.validate()
if not errors:
    print("Database is valid")
else:
    for err in errors:
        print(f"[{err['code']}] {err['message']}")
```

### wal_status()

Returns WAL (Write-Ahead Log) status. Returns a dict with keys: `enabled`, `path`, `size_bytes`, `record_count`, `last_checkpoint`, `current_epoch`.

```python
def wal_status(self) -> Dict[str, Any]
```

```python
wal = db.wal_status()
print(f"WAL size: {wal['size_bytes']} bytes")
```

### save()

Save the database to a file path. For in-memory databases, creates a new persistent copy. For file-backed databases, creates a copy at the new path. The original database remains unchanged.

```python
def save(self, path: str) -> None
```

```python
db = GrafeoDB()  # in-memory
db.create_node(["Person"], {"name": "Alix"})
db.save("./mydb")  # persist to disk
```

### to_memory()

Create an independent in-memory copy of this database. Changes to the copy do not affect the original. The copy has every graph with its data, the schema and constraints, and the property, vector and text indexes: what reopening the database from a checkpoint would give.

```python
def to_memory(self) -> GrafeoDB
```

```python
file_db = GrafeoDB("./production.grafeo")
test_db = file_db.to_memory()  # safe copy for experiments, indexes included
```

### compact()

Converts the database to a read-only [CompactStore](../../user-guide/compact-store.md) for faster queries. Takes a snapshot of all nodes and edges, builds a columnar store with CSR adjacency, and switches to read-only mode. The original store is dropped to free memory.

After calling this, write queries will raise an error. Gives ~60x memory reduction and 100x+ traversal speedup for read-only workloads.

```python
def compact(self) -> None
```

```python
db = grafeo.GrafeoDB()
db.execute("INSERT (:Person {name: 'Alix', age: 30})")
db.execute("INSERT (:Person {name: 'Gus', age: 25})")

db.compact()  # switch to read-only columnar mode

result = db.execute("MATCH (p:Person) RETURN p.name")  # fast
db.execute("INSERT (:Person {name: 'Vincent'})")        # raises error
```

!!! note
    Requires the `compact-store` feature (included in the default `lpg` profile).

### close()

Close the database, flushing any pending writes.

```python
def close(self) -> None
```

### clear_plan_cache()

Clear all cached query plans, forcing re-parsing and re-optimization on next execution. Called automatically after DDL operations, but can be invoked manually after external schema changes.

```python
def clear_plan_cache(self) -> None
```

### Properties

| Property | Type | Description |
| -------- | ---- | ----------- |
| `node_count` | `int` | Number of nodes in the database |
| `edge_count` | `int` | Number of edges in the database |
| `is_persistent` | `bool` | `True` if backed by a file |
| `path` | `Optional[str]` | Database file path, or `None` for in-memory |

## Temporal Queries

### get_node_at_epoch()

Get a node as it existed at a specific historical epoch. Returns `None` if the node did not exist at that epoch.

```python
def get_node_at_epoch(self, id: int, epoch: int) -> Optional[Node]
```

### get_edge_at_epoch()

Get an edge as it existed at a specific historical epoch. Returns `None` if the edge did not exist at that epoch.

```python
def get_edge_at_epoch(self, id: int, epoch: int) -> Optional[Edge]
```

### get_node_history()

Get the version history of a node. Returns a list of `(created_epoch, deleted_epoch, node)` tuples.

```python
def get_node_history(self, id: int) -> List[Tuple[int, Optional[int], Node]]
```

### get_edge_history()

Get the version history of an edge. Returns a list of `(created_epoch, deleted_epoch, edge)` tuples.

```python
def get_edge_history(self, id: int) -> List[Tuple[int, Optional[int], Edge]]
```

### get_node_property_at_epoch()

Returns a property value as it existed at a specific epoch. Requires the `temporal` feature.

```python
def get_node_property_at_epoch(self, id: int, key: str, epoch: int) -> Optional[Any]
```

### get_node_property_history()

Returns the full version timeline for a single property: list of (epoch, value) tuples.
Requires the `temporal` feature.

```python
def get_node_property_history(self, id: int, key: str) -> List[Tuple[int, Any]]
```

### get_all_node_property_history()

Returns version history for all properties: dict mapping property names to lists of
(epoch, value) tuples. Requires the `temporal` feature.

```python
def get_all_node_property_history(self, id: int) -> Dict[str, List[Tuple[int, Any]]]
```

### current_epoch()

Returns the current epoch of the database. The epoch increments with each committed transaction.

```python
def current_epoch(self) -> int
```

## Algorithms

Access graph algorithms via the `algorithms` property (not a method call). All algorithms run directly on the Rust graph store with no data copying. Requires the `algos` feature.

```python
algos = db.algorithms  # property, not db.algorithms()
```

### Traversal

| Method | Returns | Description |
| ------ | ------- | ----------- |
| `bfs(start)` | `List[int]` | Breadth-first search from a starting node |
| `bfs_layers(start)` | `List[List[int]]` | BFS nodes grouped by distance from start |
| `dfs(start)` | `List[int]` | Depth-first search (post-order) from a starting node |
| `dfs_all()` | `List[int]` | DFS visiting all nodes in the graph |

### Shortest Paths

#### dijkstra()

Dijkstra's algorithm. When `target` is provided, returns `(distance, path)` or `None`. When `target` is omitted, returns a dict mapping node IDs to distances.

```python
def dijkstra(
    self,
    source: int,
    target: Optional[int] = None,
    weight: Optional[str] = None
) -> Union[Dict[int, float], Tuple[float, List[int]], None]
```

```python
# Single target
result = db.algorithms.dijkstra(1, 5, weight="cost")
if result:
    distance, path = result
    print(f"Distance: {distance}, Path: {path}")

# All distances from source
distances = db.algorithms.dijkstra(1)
```

#### floyd_warshall()

All-pairs shortest paths. Returns a dict mapping `(source, target)` tuples to distances.

```python
def floyd_warshall(self, weight: Optional[str] = None) -> Dict[Tuple[int, int], float]
```

| Method | Returns | Description |
| ------ | ------- | ----------- |
| `astar(source, target, heuristic=None, weight=None)` | `Tuple[float, List[int]]` or `None` | A* shortest path |
| `bellman_ford(source, weight=None)` | `Dict` | Shortest paths with negative weights |
| `sssp(source, weight_attr=None)` | `Dict[str, float]` | SSSP with string node name support |

### Centrality

| Method | Returns | Description |
| ------ | ------- | ----------- |
| `pagerank(damping=0.85, max_iterations=100, tolerance=1e-6)` | `Dict[int, float]` | PageRank scores |
| `betweenness_centrality(normalized=True)` | `Dict[int, float]` | Betweenness centrality (Brandes) |
| `closeness_centrality(wf_improved=False)` | `Dict[int, float]` | Closeness centrality |
| `degree_centrality(normalized=False)` | `Dict` | Degree centrality (in/out/total) |

### Community Detection

| Method | Returns | Description |
| ------ | ------- | ----------- |
| `connected_components()` | `Dict[int, int]` | Component ID per node (undirected) |
| `connected_component_count()` | `int` | Number of connected components |
| `strongly_connected_components()` | `List[List[int]]` | Strongly connected components |
| `label_propagation(max_iterations=100)` | `Dict[int, int]` | Label Propagation communities |
| `louvain(resolution=1.0)` | `Dict` | Louvain communities with modularity |
| `topological_sort()` | `Optional[List[int]]` | Topological ordering, or `None` if cyclic |
| `is_dag()` | `bool` | Check if the graph is a DAG |

### Clustering and Triangles

| Method | Returns | Description |
| ------ | ------- | ----------- |
| `triangle_count()` | `Dict[int, int]` | Triangle count per node |
| `total_triangles()` | `int` | Total unique triangles in the graph |
| `global_clustering_coefficient()` | `float` | Average clustering coefficient (0.0 to 1.0) |
| `local_clustering_coefficient()` | `Dict[int, float]` | Per-node clustering coefficient |
| `clustering_coefficient(parallel=True)` | `Dict` | Full clustering info (coefficients, triangles, global) |

### Structure Analysis

| Method | Returns | Description |
| ------ | ------- | ----------- |
| `articulation_points()` | `List[int]` | Cut vertices whose removal disconnects the graph |
| `bridges()` | `List[Tuple[int, int]]` | Cut edges whose removal disconnects the graph |
| `kcore(k=None)` | `Dict` or `List[int]` | k-core decomposition (all cores or specific k) |

### Minimum Spanning Tree

| Method | Returns | Description |
| ------ | ------- | ----------- |
| `kruskal(weight=None)` | `Dict` | MST via Kruskal's (edges + total_weight) |
| `prim(weight=None, start=None)` | `Dict` | MST via Prim's (edges + total_weight) |

### Network Flow

| Method | Returns | Description |
| ------ | ------- | ----------- |
| `max_flow(source, sink, capacity=None)` | `Dict` | Maximum flow (Edmonds-Karp) |
| `min_cost_max_flow(source, sink, capacity=None, cost=None)` | `Dict` | Min-cost max-flow |

## QueryResult

Returned by `execute()` and other query methods. Iterable: each row is a dict keyed by column name.

### Result Properties

| Property | Type | Description |
| -------- | ---- | ----------- |
| `columns` | `List[str]` | Column names |
| `execution_time_ms` | `Optional[float]` | Query execution time in milliseconds |
| `rows_scanned` | `Optional[int]` | Number of rows scanned during execution |

### Result Methods

| Method | Returns | Description |
| ------ | ------- | ----------- |
| `nodes()` | `List[Node]` | All nodes from the result |
| `edges()` | `List[Edge]` | All edges from the result |
| `scalar()` | `Any` | First column of the first row |
| `to_list()` | `List[Dict]` | All rows as a list of dicts |
| `to_pandas()` | `pandas.DataFrame` | Convert to pandas DataFrame (requires pandas) |
| `to_polars()` | `polars.DataFrame` | Convert to polars DataFrame (requires polars) |

```python
result = db.execute("MATCH (p:Person) RETURN p.name, p.age")

# Iterate rows
for row in result:
    print(row["p.name"])

# Convert to DataFrame
df = result.to_pandas()

# Access metrics
if result.execution_time_ms:
    print(f"Query took {result.execution_time_ms:.2f}ms")
```

## Example

```python
import grafeo

db = grafeo.GrafeoDB()

# Execute queries
db.execute("INSERT (:Person {name: 'Alix', age: 30})")

result = db.execute("MATCH (p:Person) RETURN p.name")
for row in result:
    print(row['p.name'])

# Use transactions
with db.begin_transaction() as tx:
    tx.execute("INSERT (:Person {name: 'Gus'})")
    tx.commit()

# DataFrame integration
df = db.nodes_df()
print(df.head())

# Graph algorithms
pr = db.algorithms.pagerank()
path = db.algorithms.dijkstra(1, 5)
```
