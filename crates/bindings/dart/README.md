# Grafeo Dart Bindings

Dart FFI bindings for the [Grafeo](https://grafeo.dev) graph database. Wraps the `grafeo-c` shared library for native performance with a Dart-idiomatic API.

## Installation

```yaml
dependencies:
  grafeo: ^0.5.43
```

You also need the `grafeo-c` native library for your platform. See [Building from Source](#building-from-source) below.

## Quick Start

```dart
import 'package:grafeo/grafeo.dart';

void main() {
  final db = GrafeoDB.memory();

  // Insert data
  db.execute("INSERT (:Person {name: 'Alix', age: 30})");
  db.execute("INSERT (:Person {name: 'Gus', age: 28})");

  // Query with parameters
  final result = db.executeWithParams(
    r'MATCH (p:Person) WHERE p.age > $minAge RETURN p.name, p.age',
    {'minAge': 25},
  );

  for (final row in result.rows) {
    print('${row['p.name']}: ${row['p.age']}');
  }

  // Transactions
  final tx = db.beginTransaction();
  tx.execute("INSERT (:City {name: 'Amsterdam'})");
  tx.execute("INSERT (:City {name: 'Berlin'})");
  tx.commit();

  // CRUD operations
  final nodeId = db.createNode(['Person'], {'name': 'Vincent'});
  db.setNodeProperty(nodeId, 'role', 'hitman');
  final node = db.getNode(nodeId);
  print(node); // Node(id, [Person], {name: Vincent, role: hitman})

  db.close();
}
```

## API Reference

### GrafeoDB

| Method | Description |
|--------|-------------|
| `GrafeoDB.memory()` | Open an in-memory database |
| `GrafeoDB.open(path)` | Open a persistent database (directory or single-file) |
| `GrafeoDB.openSingleFile(path)` | Open a single-file `.grafeo` database |
| `GrafeoDB.openReadOnly(path)` | Open an existing database in read-only mode |
| `GrafeoDB.version()` | Get the library version string |
| `execute(query)` | Execute a GQL query |
| `executeWithParams(query, params)` | Execute GQL with parameters |
| `executeCypher(query)` | Execute a Cypher query |
| `executeCypherWithParams(query, params)` | Execute Cypher with parameters |
| `executeGremlin(query)` | Execute a Gremlin query |
| `executeGremlinWithParams(query, params)` | Execute Gremlin with parameters |
| `executeGraphql(query)` | Execute a GraphQL query |
| `executeGraphqlWithParams(query, params)` | Execute GraphQL with parameters |
| `executeSparql(query)` | Execute a SPARQL query |
| `executeSparqlWithParams(query, params)` | Execute SPARQL with parameters |
| `executeLanguage(lang, query, {params})` | Execute in any supported language |
| `setSchema(name)` | Set active schema for subsequent queries |
| `resetSchema()` | Revert to default graph store |
| `currentSchema()` | Get active schema name (or null) |
| `beginTransaction()` | Start an ACID transaction |
| `beginTransactionWithIsolation(level)` | Start transaction with isolation level |
| `createNode(labels, properties)` | Create a node, returns ID |
| `getNode(id)` | Get a node by ID |
| `getNodeLabels(id)` | Get labels only (faster than getNode) |
| `deleteNode(id)` | Delete a node |
| `createEdge(src, dst, type, props)` | Create an edge, returns ID |
| `getEdge(id)` | Get an edge by ID |
| `deleteEdge(id)` | Delete an edge |
| `setNodeProperty(id, key, value)` | Set a node property |
| `setEdgeProperty(id, key, value)` | Set an edge property |
| `removeNodeProperty(id, key)` | Remove a node property |
| `removeEdgeProperty(id, key)` | Remove an edge property |
| `addNodeLabel(id, label)` | Add a label to a node |
| `removeNodeLabel(id, label)` | Remove a label from a node |
| `createPropertyIndex(key)` | Create a property index |
| `dropPropertyIndex(key)` | Drop a property index |
| `hasPropertyIndex(key)` | Check if property index exists |
| `findNodesByProperty(key, value)` | Find node IDs by indexed property |
| `createVectorIndex(label, prop, dims, metric)` | Create an HNSW vector index |
| `vectorSearch(label, prop, query, {k, ef})` | k-NN vector search |
| `mmrSearch(label, prop, query, {k, ...})` | MMR diversity-aware search |
| `batchCreateNodes(label, prop, vectors)` | Bulk-create nodes with embeddings |
| `dropVectorIndex(label, property)` | Drop a vector index |
| `rebuildVectorIndex(label, property)` | Rebuild a vector index |
| `nodeCount` | Number of nodes |
| `edgeCount` | Number of edges |
| `info()` | Database metadata as JSON map |
| `save(path)` | Save snapshot to path |
| `walCheckpoint()` | Force WAL checkpoint |
| `close()` | Close and flush |

### Transaction

| Method | Description |
|--------|-------------|
| `execute(query)` | Execute GQL within transaction |
| `executeWithParams(query, params)` | Execute GQL with parameters |
| `executeLanguage(lang, query, {params})` | Execute in any language |
| `commit()` | Make changes permanent |
| `rollback()` | Discard changes |

### Types

- **`QueryResult`**: rows, columns, nodes, edges, executionTimeMs, rowsScanned
- **`Node`**: id, labels, properties
- **`Edge`**: id, type, sourceId, targetId, properties
- **`VectorResult`**: nodeId, distance

## Building from Source

```bash
# Clone and build the native library
git clone https://github.com/GrafeoDB/grafeo.git
cd grafeo
cargo build --release -p grafeo-c

# Copy to the Dart package (or your project)
# Linux:   cp target/release/libgrafeo_c.so crates/bindings/dart/
# macOS:   cp target/release/libgrafeo_c.dylib crates/bindings/dart/
# Windows: copy target\release\grafeo_c.dll crates\bindings\dart\

# Run tests
cd crates/bindings/dart
dart pub get
dart test
```

## License

Apache-2.0. See [LICENSE](../../LICENSE) for details.
