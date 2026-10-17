---
title: C# / .NET API
description: C# API reference for Grafeo via .NET 8 P/Invoke bindings.
---

# C# / .NET API Reference

Complete reference for the `Grafeo` NuGet package, .NET 8 bindings that wrap the `grafeo-c` native library via P/Invoke.

## Installation

```bash
dotnet add package Grafeo
```

The NuGet package bundles pre-built native libraries for Windows (x64), Linux (x64, arm64) and macOS (x64, arm64). No separate native install is required.

## Quick Start

```csharp
using Grafeo;

using var db = GrafeoDB.Memory();

db.Execute("INSERT (:Person {name: 'Alix', age: 30})");
db.Execute("INSERT (:Person {name: 'Gus', age: 25})");
db.Execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})");

var result = db.Execute("MATCH (p:Person) RETURN p.name, p.age");

foreach (var row in result.Rows)
{
    Console.WriteLine($"{row["p.name"]}: {row["p.age"]}");
}

Console.WriteLine($"Nodes: {db.NodeCount}, Edges: {db.EdgeCount}");
```

## GrafeoDB Class

The primary database handle. Thread-safe (the Rust engine uses `Arc<RwLock>`). Implements `IDisposable` and `IAsyncDisposable` for deterministic cleanup.

### Lifecycle

```csharp
// In-memory database
using var db = GrafeoDB.Memory();

// Persistent database (opens or creates at path)
using var db = GrafeoDB.Open("/path/to/data");

// Async disposal
await using var db = GrafeoDB.Memory();

// Library version
Console.WriteLine(GrafeoDB.Version);
```

| Member | Returns | Description |
|--------|---------|-------------|
| `GrafeoDB.Memory()` | `GrafeoDB` | Create an in-memory database |
| `GrafeoDB.Open(string path)` | `GrafeoDB` | Open or create a persistent database at `path` |
| `Dispose()` | `void` | Free the native database handle |
| `DisposeAsync()` | `ValueTask` | Async disposal (calls `Dispose` synchronously) |
| `Version` | `string` | Static property returning the Grafeo library version |

### Query Execution (Sync)

```csharp
// GQL (ISO standard)
var result = db.Execute("MATCH (p:Person) RETURN p.name");

// GQL with parameters
var result = db.ExecuteWithParams(
    "MATCH (p:Person) WHERE p.name = $name RETURN p",
    new Dictionary<string, object?> { ["name"] = "Alix" });

// Other query languages
var result = db.ExecuteCypher("MATCH (p:Person) RETURN p.name");
var result = db.ExecuteSparql("SELECT ?s WHERE { ?s a <http://example.org/Person> }");
var result = db.ExecuteGremlin("g.V().hasLabel('Person').values('name')");
var result = db.ExecuteGraphql("{ persons { name } }");
var result = db.ExecuteSql("SELECT p.name FROM GRAPH_TABLE (g MATCH (p:Person)) AS t");
```

| Method | Returns | Description |
|--------|---------|-------------|
| `Execute(string query)` | `QueryResult` | Execute a GQL query |
| `ExecuteWithParams(string query, Dictionary<string, object?> parameters)` | `QueryResult` | Execute a GQL query with parameter binding |
| `ExecuteCypher(string query)` | `QueryResult` | Execute a Cypher query |
| `ExecuteSparql(string query)` | `QueryResult` | Execute a SPARQL query |
| `ExecuteGremlin(string query)` | `QueryResult` | Execute a Gremlin query |
| `ExecuteGraphql(string query)` | `QueryResult` | Execute a GraphQL query |
| `ExecuteSql(string query)` | `QueryResult` | Execute a SQL/PGQ query |

### Query Execution (Async)

All sync query methods have async counterparts that run on the thread pool. Each accepts an optional `CancellationToken`.

```csharp
var result = await db.ExecuteAsync("MATCH (p:Person) RETURN p.name");

var result = await db.ExecuteWithParamsAsync(
    "MATCH (p:Person) WHERE p.age > $min RETURN p.name",
    new Dictionary<string, object?> { ["min"] = 18 });

var result = await db.ExecuteCypherAsync("MATCH (p:Person) RETURN p", ct);
var result = await db.ExecuteSparqlAsync(query, ct);
var result = await db.ExecuteGremlinAsync(query, ct);
var result = await db.ExecuteGraphqlAsync(query, ct);
var result = await db.ExecuteSqlAsync(query, ct);
```

| Method | Returns | Description |
|--------|---------|-------------|
| `ExecuteAsync(string query, CancellationToken ct = default)` | `Task<QueryResult>` | Execute GQL on the thread pool |
| `ExecuteWithParamsAsync(string query, Dictionary<string, object?> parameters, CancellationToken ct = default)` | `Task<QueryResult>` | Execute GQL with params on the thread pool |
| `ExecuteCypherAsync(string query, CancellationToken ct = default)` | `Task<QueryResult>` | Execute Cypher on the thread pool |
| `ExecuteSparqlAsync(string query, CancellationToken ct = default)` | `Task<QueryResult>` | Execute SPARQL on the thread pool |
| `ExecuteGremlinAsync(string query, CancellationToken ct = default)` | `Task<QueryResult>` | Execute Gremlin on the thread pool |
| `ExecuteGraphqlAsync(string query, CancellationToken ct = default)` | `Task<QueryResult>` | Execute GraphQL on the thread pool |
| `ExecuteSqlAsync(string query, CancellationToken ct = default)` | `Task<QueryResult>` | Execute SQL/PGQ on the thread pool |

### Transactions

```csharp
using var tx = db.BeginTransaction();
tx.Execute("INSERT (:Person {name: 'Alix'})");
tx.Execute("INSERT (:Person {name: 'Gus'})");
tx.Commit();

// With isolation level
using var tx = db.BeginTransaction("serializable");
```

| Method | Returns | Description |
|--------|---------|-------------|
| `BeginTransaction()` | `Transaction` | Begin with default isolation (snapshot) |
| `BeginTransaction(string)` | `Transaction` | Begin with isolation level: `"read_committed"`, `"snapshot"`, `"serializable"` |
| `BeginTransaction(IsolationLevel)` | `Transaction` | Type-safe overload: `IsolationLevel.ReadCommitted`, `.Snapshot`, `.Serializable` |

### Node CRUD

```csharp
// Create a node with labels and properties
long nodeId = db.CreateNode(
    ["Person"],
    new Dictionary<string, object?> { ["name"] = "Alix", ["age"] = 30 });

// Get a node by ID (returns null if not found)
Node? node = db.GetNode(nodeId);

// Modify properties
db.SetNodeProperty(nodeId, "city", "Amsterdam");
db.RemoveNodeProperty(nodeId, "city");

// Modify labels
db.AddNodeLabel(nodeId, "Employee");
db.RemoveNodeLabel(nodeId, "Employee");

// Delete
bool deleted = db.DeleteNode(nodeId);
```

| Method | Returns | Description |
|--------|---------|-------------|
| `CreateNode(IEnumerable<string> labels, Dictionary<string, object?>? properties = null)` | `long` | Create a node, returns the new node ID |
| `GetNode(long id)` | `Node?` | Get a node by ID, or `null` if not found |
| `DeleteNode(long id)` | `bool` | Delete a node, returns `true` if deleted |
| `SetNodeProperty(long id, string key, object? value)` | `void` | Set a property on a node |
| `RemoveNodeProperty(long id, string key)` | `bool` | Remove a property, returns `true` if removed |
| `AddNodeLabel(long id, string label)` | `bool` | Add a label, returns `true` if added |
| `RemoveNodeLabel(long id, string label)` | `bool` | Remove a label, returns `true` if removed |

### Edge CRUD

```csharp
long alixId = db.CreateNode(["Person"], new() { ["name"] = "Alix" });
long gusId = db.CreateNode(["Person"], new() { ["name"] = "Gus" });

// Create an edge with type and optional properties
long edgeId = db.CreateEdge(alixId, gusId, "KNOWS",
    new Dictionary<string, object?> { ["since"] = 2024 });

// Get an edge by ID
Edge? edge = db.GetEdge(edgeId);

// Modify properties
db.SetEdgeProperty(edgeId, "weight", 0.9);
db.RemoveEdgeProperty(edgeId, "weight");

// Delete
bool deleted = db.DeleteEdge(edgeId);
```

| Method | Returns | Description |
|--------|---------|-------------|
| `CreateEdge(long sourceId, long targetId, string edgeType, Dictionary<string, object?>? properties = null)` | `long` | Create an edge, returns the new edge ID |
| `GetEdge(long id)` | `Edge?` | Get an edge by ID, or `null` if not found |
| `DeleteEdge(long id)` | `bool` | Delete an edge, returns `true` if deleted |
| `SetEdgeProperty(long id, string key, object? value)` | `void` | Set a property on an edge |
| `RemoveEdgeProperty(long id, string key)` | `bool` | Remove a property, returns `true` if removed |

### Stats and Info

```csharp
long nodes = db.NodeCount;
long edges = db.EdgeCount;

IReadOnlyDictionary<string, object?> info = db.Info();
Console.WriteLine(info["version"]);
```

| Member | Returns | Description |
|--------|---------|-------------|
| `NodeCount` | `long` | Number of nodes in the database |
| `EdgeCount` | `long` | Number of edges in the database |
| `Info()` | `IReadOnlyDictionary<string, object?>` | Database metadata (version, node count, edge count, etc.) |

### Persistence

```csharp
db.Save("/path/to/backup.grafeo");
```

| Method | Returns | Description |
|--------|---------|-------------|
| `Save(string path)` | `void` | Save the database to a file |

### Vector Search

```csharp
// Create a vector index via GQL
db.Execute("CREATE VECTOR INDEX ON Document(embedding) OPTIONS {dimensions: 384}");

// Similarity search: returns nearest k nodes
float[] queryVector = GetEmbedding("graph database");
IReadOnlyList<VectorResult> results = db.VectorSearch(
    "Document", "embedding", queryVector, k: 10, ef: 64);

foreach (var hit in results)
{
    Console.WriteLine($"Node {hit.NodeId}, distance: {hit.Distance}");
}

// MMR search: balances relevance with diversity
IReadOnlyList<VectorResult> diverse = db.MmrSearch(
    "Document", "embedding", queryVector,
    k: 10, fetchK: 50, lambda: 0.7f, ef: 64);

// Index management
db.RebuildVectorIndex("Document", "embedding");
db.DropVectorIndex("Document", "embedding");
```

| Method | Returns | Description |
|--------|---------|-------------|
| `VectorSearch(string label, string property, float[] query, int k, uint ef = 0)` | `IReadOnlyList<VectorResult>` | k-NN similarity search ordered by distance |
| `MmrSearch(string label, string property, float[] query, int k, int fetchK, float lambda, int ef = 0)` | `IReadOnlyList<VectorResult>` | Maximal Marginal Relevance search for diverse results |
| `DropVectorIndex(string label, string property)` | `bool` | Drop a vector index. Returns `true` if dropped, `false` if not found |
| `RebuildVectorIndex(string label, string property)` | `void` | Rebuild a vector index |

## Transaction Class

An ACID transaction handle. Implements `IDisposable` and `IAsyncDisposable`. If `Commit()` is not called before disposal, the transaction is automatically rolled back.

```csharp
using var tx = db.BeginTransaction();

tx.Execute("INSERT (:Person {name: 'Alix'})");
tx.Execute("INSERT (:Person {name: 'Gus'})");

var result = tx.ExecuteWithParams(
    "MATCH (p:Person) WHERE p.name = $name RETURN p",
    new Dictionary<string, object?> { ["name"] = "Alix" });

tx.Commit(); // if this line is not reached, the transaction auto-rolls back
```

| Method | Returns | Description |
|--------|---------|-------------|
| `Execute(string query)` | `QueryResult` | Execute a GQL query within the transaction |
| `ExecuteAsync(string query, CancellationToken ct = default)` | `Task<QueryResult>` | Execute GQL on the thread pool within the transaction |
| `ExecuteWithParams(string query, Dictionary<string, object?> parameters)` | `QueryResult` | Execute GQL with params within the transaction |
| `ExecuteWithParamsAsync(string query, Dictionary<string, object?> parameters, CancellationToken ct = default)` | `Task<QueryResult>` | Execute GQL with params on the thread pool |
| `Commit()` | `void` | Commit the transaction, making all changes permanent |
| `Rollback()` | `void` | Roll back the transaction, discarding all changes |
| `Dispose()` | `void` | Dispose the handle; auto-rolls back if not committed |
| `DisposeAsync()` | `ValueTask` | Async disposal (calls `Dispose` synchronously) |

## Data Types

All entity types are immutable C# records.

### Node

```csharp
public sealed record Node(
    long Id,
    IReadOnlyList<string> Labels,
    IReadOnlyDictionary<string, object?> Properties);
```

### Edge

```csharp
public sealed record Edge(
    long Id,
    string Type,
    long SourceId,
    long TargetId,
    IReadOnlyDictionary<string, object?> Properties);
```

### QueryResult

```csharp
public sealed record QueryResult(
    IReadOnlyList<string> Columns,
    IReadOnlyList<IReadOnlyDictionary<string, object?>> Rows,
    IReadOnlyList<Node> Nodes,
    IReadOnlyList<Edge> Edges,
    double ExecutionTimeMs,
    long RowsScanned);
```

| Field | Type | Description |
|-------|------|-------------|
| `Columns` | `IReadOnlyList<string>` | Column names from the RETURN clause |
| `Rows` | `IReadOnlyList<IReadOnlyDictionary<string, object?>>` | Row data as column-name to value maps |
| `Nodes` | `IReadOnlyList<Node>` | Node entities extracted from the result |
| `Edges` | `IReadOnlyList<Edge>` | Edge entities extracted from the result |
| `ExecutionTimeMs` | `double` | Query execution time in milliseconds |
| `RowsScanned` | `long` | Number of rows scanned by the engine |

### VectorResult

```csharp
public sealed record VectorResult(long NodeId, double Distance);
```

| Field | Type | Description |
|-------|------|-------------|
| `NodeId` | `long` | The matching node's ID |
| `Distance` | `double` | Distance from the query vector |

## Exception Hierarchy

All exceptions derive from `GrafeoException`, which carries a `GrafeoStatus` code from the native layer.

```
GrafeoException             (base, Status property)
├── QueryException          (parse or execution errors)
├── TransactionException    (commit, rollback, isolation errors)
├── StorageException        (WAL, persistence, I/O errors)
└── SerializationException  (JSON or value conversion errors)
```

```csharp
try
{
    db.Execute("INVALID QUERY");
}
catch (QueryException ex)
{
    Console.WriteLine($"Query failed: {ex.Message}");
    Console.WriteLine($"Status: {ex.Status}");
}
catch (GrafeoException ex)
{
    Console.WriteLine($"Grafeo error [{ex.Status}]: {ex.Message}");
}
```

### GrafeoStatus Enum

| Value | Name | Description |
|-------|------|-------------|
| 0 | `Ok` | Success |
| 1 | `Database` | General database error |
| 2 | `Query` | Query parsing or execution error |
| 3 | `Transaction` | Transaction lifecycle error |
| 4 | `Storage` | Storage or persistence error |
| 5 | `Io` | I/O error |
| 6 | `Serialization` | JSON or value serialization error |
| 7 | `Internal` | Internal engine error |
| 8 | `NullPointer` | Null pointer passed to FFI |
| 9 | `InvalidUtf8` | Invalid UTF-8 string passed to FFI |

## Type Mapping

| C# Type | Grafeo Type | Notes |
|---------|-------------|-------|
| `null` | Null | |
| `bool` | Bool | |
| `int` | Int64 | Widened to `long` |
| `long` | Int64 | |
| `float` | Float64 | Widened to `double` |
| `double` | Float64 | |
| `string` | String | |
| `DateTime` | Timestamp | Converted to UTC microseconds |
| `DateTimeOffset` | Timestamp | Converted to UTC microseconds |
| `DateOnly` | Date | ISO 8601 date string |
| `TimeOnly` | Time | ISO 8601 time string |
| `TimeSpan` | Duration | ISO 8601 duration string |
| `byte[]` | Bytes | Base64-encoded for transport |
| `float[]` | Vector | For embeddings and similarity search |
| `ReadOnlyMemory<float>` | Vector | Zero-copy span-based vector input |
| `IList<object?>` | List | Elements converted recursively |
| `IDictionary<string, object?>` | Map | Keys must be strings |

## Links

- [NuGet package](https://www.nuget.org/packages/Grafeo)
- [GitHub](https://github.com/GrafeoDB/grafeo/tree/main/crates/bindings/csharp)
