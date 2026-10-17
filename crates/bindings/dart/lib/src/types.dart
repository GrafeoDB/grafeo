/// Data types for the Grafeo Dart binding.
library;

import 'dart:ffi';

/// Opaque pointer to a Grafeo database.
typedef GrafeoDatabasePtr = Pointer<Void>;

/// Opaque pointer to a Grafeo query result.
typedef GrafeoResultPtr = Pointer<Void>;

/// Opaque pointer to a Grafeo transaction.
typedef GrafeoTransactionPtr = Pointer<Void>;

/// Transaction isolation levels.
enum IsolationLevel {
  readCommitted(0),
  snapshotIsolation(1),
  serializable(2);

  final int code;
  const IsolationLevel(this.code);
}

/// A graph node with an ID, labels, and properties.
class Node {
  final int id;
  final List<String> labels;
  final Map<String, dynamic> properties;

  const Node(this.id, this.labels, this.properties);

  @override
  String toString() => 'Node($id, $labels, $properties)';

  @override
  bool operator ==(Object other) =>
      identical(this, other) || (other is Node && other.id == id);

  @override
  int get hashCode => id.hashCode;
}

/// A graph edge with an ID, type, source/target, and properties.
class Edge {
  /// Unique numeric identifier for this edge.
  final int id;

  /// The relationship type label (e.g. `"KNOWS"`, `"WORKS_AT"`).
  final String type;

  /// The ID of the source (start) node.
  final int sourceId;

  /// The ID of the target (end) node.
  final int targetId;

  /// Key/value properties stored on this edge.
  final Map<String, dynamic> properties;

  /// Creates an edge with the given [id], [type], [sourceId], [targetId], and [properties].
  const Edge(this.id, this.type, this.sourceId, this.targetId, this.properties);

  @override
  String toString() => 'Edge($id, $type, $sourceId->$targetId, $properties)';

  @override
  bool operator ==(Object other) =>
      identical(this, other) || (other is Edge && other.id == id);

  @override
  int get hashCode => id.hashCode;
}

/// The result of a query execution.
class QueryResult {
  final List<String> columns;
  final List<Map<String, dynamic>> rows;
  final List<Node> nodes;
  final List<Edge> edges;
  final double executionTimeMs;
  final int rowsScanned;

  const QueryResult({
    required this.columns,
    required this.rows,
    required this.nodes,
    required this.edges,
    required this.executionTimeMs,
    required this.rowsScanned,
  });

  @override
  String toString() =>
      'QueryResult(${rows.length} rows, ${nodes.length} nodes, '
      '${edges.length} edges, ${executionTimeMs.toStringAsFixed(2)}ms)';
}

/// A single vector search result with a node ID and distance score.
class VectorResult {
  final int nodeId;
  final double distance;

  const VectorResult(this.nodeId, this.distance);

  @override
  String toString() => 'VectorResult(nodeId: $nodeId, distance: $distance)';
}
