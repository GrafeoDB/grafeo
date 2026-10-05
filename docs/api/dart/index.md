---
title: Dart API
description: Dart API reference for Grafeo via dart:ffi bindings.
---

# Dart API Reference

Complete reference for the `grafeo` Dart package, native bindings for Grafeo via [dart:ffi](https://dart.dev/interop/c-interop). Wraps the grafeo-c shared library for high-performance FFI access.

## Installation

Add the package from [pub.dev](https://pub.dev/packages/grafeo):

```yaml
# pubspec.yaml
dependencies:
  grafeo: ^0.5.39
```

```bash
dart pub get
```

The package requires the `grafeo_c` shared library at runtime. Place `libgrafeo_c.so` (Linux), `libgrafeo_c.dylib` (macOS) or `grafeo_c.dll` (Windows) in a location visible to your application. You can also pass a custom path via the `libraryPath` parameter on factory constructors.

## Quick Start

```dart
import 'package:grafeo/grafeo.dart';

void main() {
  final db = GrafeoDB.memory();

  db.execute('''
    INSERT (:Person {name: "Alix", age: 30}),
           (:Person {name: "Gus",  age: 28})
  ''');

  db.execute('''
    MATCH (a:Person {name: "Alix"}), (b:Person {name: "Gus"})
    INSERT (a)-[:KNOWS]->(b)
  ''');

  final result = db.execute('''
    MATCH (a:Person)-[r:KNOWS]->(b:Person)
    RETURN a.name AS from, b.name AS to
  ''');

  for (final row in result.rows) {
    print('${row['from']} knows ${row['to']}');
  }
  // Output: Alix knows Gus

  db.close();
}
```

## GrafeoDB

The main database class. All methods are synchronous (FFI calls block the current isolate).

### Lifecycle

#### GrafeoDB.memory()

Create a new in-memory database.

```dart
static GrafeoDB memory({String? libraryPath})
```

```dart
final db = GrafeoDB.memory();
```

#### GrafeoDB.open()

Open a persistent database at the given path.

```dart
static GrafeoDB open(String path, {String? libraryPath})
```

```dart
final db = GrafeoDB.open('/tmp/my_graph.db');
```

#### GrafeoDB.openSingleFile()

Deprecated since 0.6.0 and removed in 0.7.0: use `open()`. Opens or creates a
persistent database at the given path (a single file), with a sidecar WAL for
crash safety. Since 0.6 every database is a single file, whatever the extension
of its path, so this is the same as `open()`.

```dart
static GrafeoDB openSingleFile(String path, {String? libraryPath})
```

```dart
final db = GrafeoDB.open('/home/user/my_graph.grafeo');
```

#### GrafeoDB.openReadOnly()

Open an existing database in read-only mode. Multiple read-only handles may be
opened concurrently on the same path. Write operations throw `DatabaseException`.

```dart
static GrafeoDB openReadOnly(String path, {String? libraryPath})
```

```dart
final db = GrafeoDB.openReadOnly('/home/user/my_graph.grafeo');
```

#### close()

Close the database, flushing all writes. Safe to call multiple times. After closing, all other methods throw a `DatabaseException`.

```dart
void close()
```

#### version()

Returns the grafeo-c library version string.

```dart
static String version({String? libraryPath})
```

```dart
print(GrafeoDB.version()); // "0.5.39"
```

### Query Execution

#### execute()

Execute a GQL (ISO standard) query and return a `QueryResult`.

```dart
QueryResult execute(String query)
```

```dart
final result = db.execute('MATCH (p:Person) RETURN p.name, p.age');
for (final row in result.rows) {
  print('${row['p.name']}: ${row['p.age']}');
}
```

#### executeWithParams()

Execute a GQL query with named parameters. Parameters are JSON-encoded using the grafeo-bindings-common wire format. Temporal types (`DateTime`, `Duration`) are automatically encoded.

```dart
QueryResult executeWithParams(String query, Map<String, dynamic> params)
```

```dart
final result = db.executeWithParams(
  'MATCH (p:Person) WHERE p.age > \$minAge RETURN p.name',
  {'minAge': 25},
);
```

#### executeCypher()

Execute a Cypher query. Requires the `cypher` feature in grafeo-c.

```dart
QueryResult executeCypher(String query)
```

#### executeSparql()

Execute a SPARQL query. Requires the `sparql` feature in grafeo-c.

```dart
QueryResult executeSparql(String query)
```

#### executeGremlin()

Execute a Gremlin query. Requires the `gremlin` feature in grafeo-c.

```dart
QueryResult executeGremlin(String query)
```

#### executeGraphql()

Execute a GraphQL query. Requires the `graphql` feature in grafeo-c.

```dart
QueryResult executeGraphql(String query)
```

#### executeCypherWithParams()

Execute a Cypher query with named parameters.

```dart
QueryResult executeCypherWithParams(String query, Map<String, dynamic> params)
```

#### executeGremlinWithParams()

Execute a Gremlin query with named parameters.

```dart
QueryResult executeGremlinWithParams(String query, Map<String, dynamic> params)
```

#### executeGraphqlWithParams()

Execute a GraphQL query with named parameters.

```dart
QueryResult executeGraphqlWithParams(String query, Map<String, dynamic> params)
```

#### executeSparqlWithParams()

Execute a SPARQL query with named parameters.

```dart
QueryResult executeSparqlWithParams(String query, Map<String, dynamic> params)
```

#### executeLanguage()

Execute a query in any supported language with optional parameters. The `language` parameter is one of: `"gql"`, `"cypher"`, `"gremlin"`, `"graphql"`, `"sparql"`, `"sql"`.

```dart
QueryResult executeLanguage(String language, String query, {Map<String, dynamic>? params})
```

```dart
final result = db.executeLanguage('cypher', 'MATCH (n) RETURN n LIMIT 10');
final result2 = db.executeLanguage('gql',
  r'MATCH (p:Person) WHERE p.age > $min RETURN p',
  params: {'min': 25},
);
```

### Statistics

| Member | Type | Description |
|--------|------|-------------|
| `nodeCount` | `int` (getter) | Number of nodes in the database (O(1)) |
| `edgeCount` | `int` (getter) | Number of edges in the database (O(1)) |

#### info()

Returns high-level database information as a parsed JSON map.

```dart
Map<String, dynamic> info()
```

```dart
final dbInfo = db.info();
print(dbInfo['node_count']); // 42
```

### Schema Context

#### setSchema()

Set the active schema for subsequent queries. Equivalent to `SESSION SET SCHEMA` in GQL.

```dart
void setSchema(String schemaName)
```

```dart
db.setSchema('analytics');
db.execute('MATCH (n) RETURN n');  // scoped to 'analytics' schema
```

#### resetSchema()

Clear the active schema, reverting to the default graph store.

```dart
void resetSchema()
```

#### currentSchema()

Return the currently active schema name, or `null` if none is set.

```dart
String? currentSchema()
```

```dart
db.setSchema('analytics');
print(db.currentSchema()); // 'analytics'
db.resetSchema();
print(db.currentSchema()); // null
```

### Transactions

#### beginTransaction()

Begin a new transaction with the default isolation level.

```dart
Transaction beginTransaction()
```

```dart
final tx = db.beginTransaction();
try {
  tx.execute('INSERT (:Person {name: "Vincent"})');

  tx.commit();
} catch (_) {
  tx.rollback();
  rethrow;
}
```

#### beginTransactionWithIsolation()

Begin a transaction with a specific isolation level.

```dart
Transaction beginTransactionWithIsolation(IsolationLevel isolationLevel)
```

```dart
final tx = db.beginTransactionWithIsolation(IsolationLevel.serializable);
```

### Node CRUD

#### createNode()

Create a node with labels and properties. Returns the new node ID.

```dart
int createNode(List<String> labels, Map<String, dynamic> properties)
```

```dart
final id = db.createNode(['Person'], {'name': 'Alix', 'age': 30});
print(id); // 0
```

#### getNode()

Get a node by ID. Throws on error if the node does not exist.

```dart
Node getNode(int id)
```

```dart
final node = db.getNode(0);
print(node.labels);     // ['Person']
print(node.properties); // {'name': 'Alix', 'age': 30}
```

#### getNodeLabels()

Return the labels of a node without fetching full node data. More efficient than `getNode()` when only labels are needed.

```dart
List<String> getNodeLabels(int id)
```

```dart
final labels = db.getNodeLabels(0);
print(labels); // ['Person', 'Employee']
```

#### deleteNode()

Delete a node by ID. Returns `true` if the node existed.

```dart
bool deleteNode(int id)
```

#### setNodeProperty()

Set a property on a node.

```dart
void setNodeProperty(int id, String key, dynamic value)
```

```dart
db.setNodeProperty(0, 'email', 'alix@example.com');
```

#### removeNodeProperty()

Remove a property from a node.

```dart
void removeNodeProperty(int id, String key)
```

#### addNodeLabel()

Add a label to an existing node.

```dart
void addNodeLabel(int id, String label)
```

```dart
db.addNodeLabel(0, 'Employee');
```

#### removeNodeLabel()

Remove a label from a node.

```dart
void removeNodeLabel(int id, String label)
```

### Edge CRUD

#### createEdge()

Create an edge from a source node to a target node with a type and properties. Returns the new edge ID.

```dart
int createEdge(int sourceId, int targetId, String type, Map<String, dynamic> properties)
```

```dart
final edgeId = db.createEdge(0, 1, 'KNOWS', {'since': 2024});
```

#### getEdge()

Get an edge by ID. Throws on error if the edge does not exist.

```dart
Edge getEdge(int id)
```

```dart
final edge = db.getEdge(0);
print(edge.type);     // 'KNOWS'
print(edge.sourceId); // 0
print(edge.targetId); // 1
```

#### deleteEdge()

Delete an edge by ID. Returns `true` if the edge existed.

```dart
bool deleteEdge(int id)
```

#### setEdgeProperty()

Set a property on an edge.

```dart
void setEdgeProperty(int id, String key, dynamic value)
```

#### removeEdgeProperty()

Remove a property from an edge.

```dart
void removeEdgeProperty(int id, String key)
```

### Property Indexes

#### createPropertyIndex()

Create a property index for fast point-lookup queries.

```dart
void createPropertyIndex(String propertyKey)
```

```dart
db.createPropertyIndex('name');
```

#### dropPropertyIndex()

Drop a property index. Returns `true` if it existed.

```dart
bool dropPropertyIndex(String propertyKey)
```

#### hasPropertyIndex()

Check if a property index exists.

```dart
bool hasPropertyIndex(String propertyKey)
```

#### findNodesByProperty()

Find all node IDs where the given property equals the given value. Requires a property index.

```dart
List<int> findNodesByProperty(String propertyKey, dynamic value)
```

```dart
db.createPropertyIndex('name');
final ids = db.findNodesByProperty('name', 'Alix');
for (final id in ids) {
  final node = db.getNode(id);
  print(node.properties);
}
```

### Vector Search

#### createVectorIndex()

Create an HNSW vector index on nodes with the given label.

```dart
void createVectorIndex(
  String label,
  String property,
  int dimensions,
  String metric, {
  int m = 16,
  int efConstruction = 200,
})
```

| Parameter | Description |
|-----------|-------------|
| `label` | Node label to index |
| `property` | Property name containing the float vector |
| `dimensions` | Vector dimensions (e.g. 384, 768, 1536) |
| `metric` | Distance metric: `"cosine"`, `"euclidean"`, or `"dot"` |
| `m` | Bidirectional links per HNSW node (default: 16) |
| `efConstruction` | Index build quality (default: 200) |

```dart
db.createVectorIndex('Document', 'embedding', 384, 'cosine');
```

#### vectorSearch()

Perform a k-nearest-neighbour vector search. For diversity-aware results, use `mmrSearch()` instead.

```dart
List<VectorResult> vectorSearch(
  String label,
  String property,
  List<double> query, {
  required int k,
  int ef = 64,
})
```

```dart
final results = db.vectorSearch('Document', 'embedding', queryVec, k: 10);
for (final r in results) {
  print('Node ${r.nodeId}: distance ${r.distance}');
}
```

#### batchCreateNodes()

Bulk-create nodes with vector embeddings. All nodes get the same label; each
receives a unique embedding vector stored under the given property.

```dart
List<int> batchCreateNodes(
  String label,
  String embeddingProperty,
  List<List<double>> vectors,
)
```

```dart
final vectors = [
  [0.1, 0.2, 0.3],
  [0.4, 0.5, 0.6],
  [0.7, 0.8, 0.9],
];
final ids = db.batchCreateNodes('Document', 'embedding', vectors);
print(ids); // [0, 1, 2]
```

!!! note "Requires `vector-index` feature"
    `batchCreateNodes` is only available when grafeo-c is built with
    `--features vector-index` (included in the `ai` and `analytics` profiles).

#### mmrSearch()

Perform a Maximal Marginal Relevance (MMR) vector search for diverse nearest neighbors.

```dart
List<VectorResult> mmrSearch(
  String label,
  String property,
  List<double> query, {
  required int k,
  required int fetchK,
  required double lambda,
  required int ef,
})
```

| Parameter | Description |
|-----------|-------------|
| `label` | Node label to search |
| `property` | Vector property name |
| `query` | Query vector as a list of doubles |
| `k` | Number of results to return |
| `fetchK` | Number of candidates to fetch before reranking |
| `lambda` | Balance between relevance (1.0) and diversity (0.0) |
| `ef` | Search beam width (higher is more accurate but slower) |

```dart
final results = db.mmrSearch('Document', 'embedding', queryVec,
    k: 5, fetchK: 20, lambda: 0.7, ef: 64);
for (final r in results) {
  print('Node ${r.nodeId}: distance ${r.distance}');
}
```

#### dropVectorIndex()

Drop a vector index. Returns `true` if the index existed.

```dart
bool dropVectorIndex(String label, String property)
```

#### rebuildVectorIndex()

Rebuild a vector index by rescanning all matching nodes.

```dart
void rebuildVectorIndex(String label, String property)
```

### Admin

#### save()

Save a database snapshot to the given path.

```dart
void save(String path)
```

```dart
db.save('/tmp/backup.db');
```

#### walCheckpoint()

Force a write-ahead log checkpoint, flushing buffered writes to the main data file.

```dart
void walCheckpoint()
```

## Transaction

An ACID transaction on a Grafeo database. Obtain via `GrafeoDB.beginTransaction()` or `GrafeoDB.beginTransactionWithIsolation()`. Must be explicitly committed or rolled back. If dropped without either, the Rust Drop implementation performs an automatic rollback.

### Transaction.execute()

Execute a GQL query within this transaction.

```dart
QueryResult execute(String query)
```

### Transaction.executeWithParams()

Execute a parameterized GQL query within this transaction.

```dart
QueryResult executeWithParams(String query, Map<String, dynamic> params)
```

### Transaction.executeLanguage()

Execute a query in any supported language within this transaction.

```dart
QueryResult executeLanguage(String language, String query, {Map<String, dynamic>? params})
```

```dart
final tx = db.beginTransaction();
tx.executeLanguage('cypher', "CREATE (n:Person {name: 'Alix'})");
tx.commit();
```

### commit()

Commit the transaction, making all changes permanent.

```dart
void commit()
```

### rollback()

Roll back the transaction, discarding all changes.

```dart
void rollback()
```

### Example

```dart
final tx = db.beginTransaction();
try {
  tx.execute("INSERT (:Person {name: 'Alix'})");
  tx.execute("INSERT (:Person {name: 'Gus'})");

  final result = tx.executeWithParams(
    'MATCH (p:Person) WHERE p.name = \$name RETURN p',
    {'name': 'Alix'},
  );
  print(result.rows.length); // 1

  tx.commit();
} catch (e) {
  tx.rollback();
  rethrow;
}
```

## IsolationLevel

Transaction isolation levels, matching the C `GrafeoStatus` enum codes.

```dart
enum IsolationLevel {
  readCommitted(0),
  snapshotIsolation(1),
  serializable(2);
}
```

| Value | Code | Description |
|-------|------|-------------|
| `readCommitted` | 0 | Reads see only committed data; no dirty reads |
| `snapshotIsolation` | 1 | Reads use a consistent snapshot taken at transaction start |
| `serializable` | 2 | Full serializability; transactions appear to execute sequentially |

## Data Types

### Node

A graph node with an ID, labels and properties.

```dart
class Node {
  final int id;
  final List<String> labels;
  final Map<String, dynamic> properties;
}
```

Equality is based on `id` only.

### Edge

A graph edge with an ID, type, source/target endpoints and properties.

```dart
class Edge {
  final int id;
  final String type;
  final int sourceId;
  final int targetId;
  final Map<String, dynamic> properties;
}
```

Equality is based on `id` only.

### QueryResult

The result of a query execution.

```dart
class QueryResult {
  final List<String> columns;
  final List<Map<String, dynamic>> rows;
  final List<Node> nodes;
  final List<Edge> edges;
  final double executionTimeMs;
  final int rowsScanned;
}
```

| Field | Description |
|-------|-------------|
| `columns` | Column names extracted from the first row |
| `rows` | List of row maps with column-name keys |
| `nodes` | Deduplicated `Node` entities found in the result rows |
| `edges` | Deduplicated `Edge` entities found in the result rows |
| `executionTimeMs` | Query execution time in milliseconds |
| `rowsScanned` | Number of rows scanned during execution |

### VectorResult

A single vector search result with a node ID and distance score.

```dart
class VectorResult {
  final int nodeId;
  final double distance;
}
```

## Exception Hierarchy

All Grafeo errors extend the `sealed` base class `GrafeoException`. Use pattern matching to handle specific error types:

```dart
try {
  db.execute('INVALID QUERY');
} on QueryException catch (e) {
  print('Query error: ${e.message}');
} on TransactionException catch (e) {
  print('Transaction error: ${e.message}');
} on GrafeoException catch (e) {
  print('Grafeo error (${e.status.name}): ${e.message}');
}
```

### Class Hierarchy

```text
GrafeoException (sealed)
  +-- QueryException           // query parsing or execution error
  +-- TransactionException     // conflict or invalid transaction state
  +-- StorageException         // storage or IO error
  +-- SerializationException   // data serialization error
  +-- DatabaseException        // generic database error (catch-all)
```

Every exception carries a `message` string and a `GrafeoStatus` enum value.

### GrafeoStatus

Status codes returned by grafeo-c FFI functions, matching the C enum exactly.

```dart
enum GrafeoStatus {
  ok(0),
  database(1),
  query(2),
  transaction(3),
  storage(4),
  io(5),
  serialization(6),
  internal(7),
  nullPointer(8),
  invalidUtf8(9);
}
```

## Type Mapping

| Dart | Grafeo | Notes |
|------|--------|-------|
| `null` | Null | |
| `bool` | Bool | |
| `int` | Int64 | |
| `double` | Float64 | |
| `String` | String | |
| `DateTime` | Timestamp | Encoded as `$timestamp_us` (microseconds, UTC) |
| `Duration` | Duration | Encoded as ISO 8601 duration string |
| `List` | List | Elements converted recursively |
| `Map<String, dynamic>` | Map | Keys must be strings |
| `Uint8List` | Bytes | Encoded as base64 |
| `Float32List` | Vector | For embeddings and similarity search |
| `Float64List` | Vector | Converted to list of doubles |

## NativeFinalizer

Both `GrafeoDB` and `Transaction` implement `Finalizable` and register a `NativeFinalizer` for automatic cleanup. If you forget to call `close()` on a database or `commit()`/`rollback()` on a transaction, the Dart garbage collector will invoke the native destructor to prevent resource leaks.

You should still call `close()` and `commit()`/`rollback()` explicitly for deterministic resource management. The finalizer is a safety net, not a substitute for proper lifecycle handling.

```dart
// Recommended: explicit cleanup
final db = GrafeoDB.memory();
try {
  // ... use the database ...
} finally {
  db.close();
}
```

## Links

- [Flutter Integration Guide](../../user-guide/dart/flutter.md)
- [pub.dev package](https://pub.dev/packages/grafeo)
- [GitHub](https://github.com/GrafeoDB/grafeo/tree/main/crates/bindings/dart)
