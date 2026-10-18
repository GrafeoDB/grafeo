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

  /// What a write statement changed (all zero for a read).
  final WriteCounters counters;

  const QueryResult({
    required this.columns,
    required this.rows,
    required this.nodes,
    required this.edges,
    required this.executionTimeMs,
    required this.rowsScanned,
    this.counters = WriteCounters.none,
  });

  @override
  String toString() =>
      'QueryResult(${rows.length} rows, ${nodes.length} nodes, '
      '${edges.length} edges, ${executionTimeMs.toStringAsFixed(2)}ms)';
}

/// What a write statement changed.
class WriteCounters {
  final int nodesCreated;
  final int nodesDeleted;
  final int edgesCreated;
  final int edgesDeleted;
  final int propertiesSet;
  final int labelsAdded;
  final int labelsRemoved;

  const WriteCounters({
    this.nodesCreated = 0,
    this.nodesDeleted = 0,
    this.edgesCreated = 0,
    this.edgesDeleted = 0,
    this.propertiesSet = 0,
    this.labelsAdded = 0,
    this.labelsRemoved = 0,
  });

  /// No changes.
  static const none = WriteCounters();

  @override
  bool operator ==(Object other) =>
      other is WriteCounters &&
      other.nodesCreated == nodesCreated &&
      other.nodesDeleted == nodesDeleted &&
      other.edgesCreated == edgesCreated &&
      other.edgesDeleted == edgesDeleted &&
      other.propertiesSet == propertiesSet &&
      other.labelsAdded == labelsAdded &&
      other.labelsRemoved == labelsRemoved;

  @override
  int get hashCode => Object.hash(nodesCreated, nodesDeleted, edgesCreated,
      edgesDeleted, propertiesSet, labelsAdded, labelsRemoved);

  @override
  String toString() => 'WriteCounters(nodes +$nodesCreated -$nodesDeleted, '
      'edges +$edgesCreated -$edgesDeleted, properties $propertiesSet, '
      'labels +$labelsAdded -$labelsRemoved)';
}

/// A single vector search result with a node ID and distance score.
class VectorResult {
  final int nodeId;
  final double distance;

  const VectorResult(this.nodeId, this.distance);

  @override
  String toString() => 'VectorResult(nodeId: $nodeId, distance: $distance)';
}

/// An exact decimal from Grafeo: up to 38 digits, kept as its text with the
/// scale it was written with (`1.50`).
///
/// Equal values are equal whatever their scale (`1.50` equals `1.5`). As a
/// query parameter it is passed exactly; Dart has no decimal type of its own,
/// so [toDouble] is the nearest double.
final class GrafeoDecimal {
  /// A decimal from its text: `[+-]digits[.digits]`.
  ///
  /// Throws a [FormatException] when [text] is not a decimal number.
  GrafeoDecimal(this.text) {
    if (!_pattern.hasMatch(text)) {
      throw FormatException('not a decimal number', text);
    }
  }

  /// The text with its scale, as Grafeo writes it.
  final String text;

  static final _pattern = RegExp(r'^[+-]?(\d+(\.\d*)?|\.\d+)$');

  /// The nearest double.
  double toDouble() => double.parse(text);

  @override
  String toString() => text;

  @override
  bool operator ==(Object other) =>
      other is GrafeoDecimal && _normalized(other.text) == _normalized(text);

  @override
  int get hashCode => _normalized(text).hashCode;

  /// The value without leading or trailing zeros: one text per value.
  static String _normalized(String text) {
    final negative = text.startsWith('-');
    final body =
        text.startsWith('-') || text.startsWith('+') ? text.substring(1) : text;
    final point = body.indexOf('.');
    final integer = (point < 0 ? body : body.substring(0, point))
        .replaceFirst(RegExp('^0+'), '');
    final fraction = (point < 0 ? '' : body.substring(point + 1))
        .replaceFirst(RegExp(r'0+$'), '');
    if (integer.isEmpty && fraction.isEmpty) return '0';
    return '${negative ? '-' : ''}$integer.$fraction';
  }
}
