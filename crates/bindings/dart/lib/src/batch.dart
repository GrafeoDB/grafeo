/// Types of the batch writes, upserts and change history.
library;

/// One edge of [GrafeoDB.batchCreateEdges].
class BatchEdge {
  /// Source node ID.
  final int sourceId;

  /// Target node ID.
  final int targetId;

  /// Edge type.
  final String type;

  /// Optional properties.
  final Map<String, dynamic>? properties;

  const BatchEdge(this.sourceId, this.targetId, this.type, [this.properties]);

  @override
  String toString() => 'BatchEdge($sourceId->$targetId, $type, $properties)';
}

/// What an upsert did with its rows.
class UpsertSummary {
  /// Rows that created a node or edge.
  final int created;

  /// Rows that updated one.
  final int updated;

  /// Rows without their key, and edge rows whose endpoint matches no node or
  /// several.
  final int skipped;

  /// Indexes of the skipped rows.
  final List<int> skippedRows;

  const UpsertSummary(
      this.created, this.updated, this.skipped, this.skippedRows);

  @override
  String toString() =>
      'UpsertSummary(created: $created, updated: $updated, skipped: $skipped)';
}

/// One recorded change of a node or edge (change data capture).
class ChangeEvent {
  /// The node or edge ID.
  final int entityId;

  /// `node` or `edge`.
  final String entityType;

  /// The named graph, null for the default graph.
  final String? graph;

  /// `create`, `update` or `delete`.
  final String kind;

  /// The epoch of the commit.
  final int epoch;

  /// A hybrid logical clock value.
  final int timestamp;

  /// Properties before the change (null on a create).
  final Map<String, dynamic>? before;

  /// Properties after the change (null on a delete).
  final Map<String, dynamic>? after;

  /// A node's labels after the change.
  final List<String>? labels;

  /// A node's labels before the change.
  final List<String>? beforeLabels;

  /// An edge's type.
  final String? edgeType;

  /// An edge's source node.
  final int? sourceId;

  /// An edge's target node.
  final int? targetId;

  /// On an edge a detach delete removed, the node it was removed with.
  final int? deletedWith;

  const ChangeEvent({
    required this.entityId,
    required this.entityType,
    required this.graph,
    required this.kind,
    required this.epoch,
    required this.timestamp,
    this.before,
    this.after,
    this.labels,
    this.beforeLabels,
    this.edgeType,
    this.sourceId,
    this.targetId,
    this.deletedWith,
  });

  @override
  String toString() => 'ChangeEvent($kind $entityType $entityId at $epoch)';
}
