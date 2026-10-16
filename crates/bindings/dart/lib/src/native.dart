/// Marshalling the database, transaction and graph handles share.
library;

import 'dart:convert';
import 'dart:ffi';

import 'package:ffi/ffi.dart';

import 'batch.dart';
import 'error.dart';
import 'ffi/bindings.dart';
import 'types.dart';
import 'value.dart';

/// Runs [body] with [strings] as native UTF-8 (null as a null pointer) and
/// frees them after.
T withNativeStrings<T>(
  List<String?> strings,
  T Function(List<Pointer<Utf8>> pointers) body,
) {
  final pointers = [
    for (final s in strings)
      s == null ? nullptr.cast<Utf8>() : s.toNativeUtf8(allocator: malloc),
  ];
  try {
    return body(pointers);
  } finally {
    for (final p in pointers) {
      if (p != nullptr) malloc.free(p);
    }
  }
}

/// Extracts a [QueryResult] from a native result pointer, then frees it.
QueryResult buildResult(GrafeoBindings bindings, Pointer<Void> resultPtr) {
  try {
    final rows = parseRows(bindings.grafeoResultJson(resultPtr).toDartString());
    final (nodes, edges) = extractEntities(rows);
    return QueryResult(
      columns: extractColumns(rows),
      rows: rows,
      nodes: nodes,
      edges: edges,
      executionTimeMs: bindings.grafeoResultExecutionTimeMs(resultPtr),
      rowsScanned: bindings.grafeoResultRowsScanned(resultPtr),
      counters: WriteCounters(
        nodesCreated: bindings.grafeoResultNodesCreated(resultPtr),
        nodesDeleted: bindings.grafeoResultNodesDeleted(resultPtr),
        edgesCreated: bindings.grafeoResultEdgesCreated(resultPtr),
        edgesDeleted: bindings.grafeoResultEdgesDeleted(resultPtr),
        propertiesSet: bindings.grafeoResultPropertiesSet(resultPtr),
        labelsAdded: bindings.grafeoResultLabelsAdded(resultPtr),
        labelsRemoved: bindings.grafeoResultLabelsRemoved(resultPtr),
      ),
    );
  } finally {
    bindings.grafeoFreeResult(resultPtr);
  }
}

/// Reads a node the library handed out into [out] by [call], then frees it.
Node takeNode(
  GrafeoBindings bindings,
  int Function(Pointer<Pointer<Void>> out) call,
) {
  final out = malloc<Pointer<Void>>();
  try {
    final status = call(out);
    if (status != GrafeoStatus.ok.code) throwStatus(bindings, status);
    final node = out.value;
    try {
      return Node(
        bindings.grafeoNodeId(node),
        parseStringArray(bindings.grafeoNodeLabelsJson(node).toDartString()),
        parseObject(bindings.grafeoNodePropertiesJson(node).toDartString()),
      );
    } finally {
      bindings.grafeoFreeNode(node);
    }
  } finally {
    malloc.free(out);
  }
}

/// Reads an edge the library handed out into [out] by [call], then frees it.
Edge takeEdge(
  GrafeoBindings bindings,
  int Function(Pointer<Pointer<Void>> out) call,
) {
  final out = malloc<Pointer<Void>>();
  try {
    final status = call(out);
    if (status != GrafeoStatus.ok.code) throwStatus(bindings, status);
    final edge = out.value;
    try {
      return Edge(
        bindings.grafeoEdgeId(edge),
        bindings.grafeoEdgeType(edge).toDartString(),
        bindings.grafeoEdgeSourceId(edge),
        bindings.grafeoEdgeTargetId(edge),
        parseObject(bindings.grafeoEdgePropertiesJson(edge).toDartString()),
      );
    } finally {
      bindings.grafeoFreeEdge(edge);
    }
  } finally {
    malloc.free(out);
  }
}

/// Runs [call], copies and frees the ids it hands out.
List<int> takeIds(
  GrafeoBindings bindings,
  int Function(Pointer<Pointer<Uint64>> ids, Pointer<IntPtr> count) call,
) {
  final idsPtr = malloc<Pointer<Uint64>>();
  final countPtr = malloc<IntPtr>();
  try {
    final status = call(idsPtr, countPtr);
    if (status != GrafeoStatus.ok.code) throwStatus(bindings, status);
    final count = countPtr.value;
    if (count == 0) return [];
    final ids = idsPtr.value;
    final result = [for (var i = 0; i < count; i++) ids[i]];
    bindings.grafeoFreeNodeIds(ids, count);
    return result;
  } finally {
    malloc.free(idsPtr);
    malloc.free(countPtr);
  }
}

/// Runs [call] and takes the JSON string it hands out.
String takeJson(
  GrafeoBindings bindings,
  int Function(Pointer<Pointer<Utf8>> json) call,
) {
  final out = malloc<Pointer<Utf8>>();
  try {
    final status = call(out);
    if (status != GrafeoStatus.ok.code) throwStatus(bindings, status);
    return takeString(bindings, out.value);
  } finally {
    malloc.free(out);
  }
}

/// Copies and frees a string the library handed out; null throws its error.
String takeString(GrafeoBindings bindings, Pointer<Utf8> ptr) {
  if (ptr == nullptr) throwLastError(bindings);
  try {
    return ptr.toDartString();
  } finally {
    bindings.grafeoFreeString(ptr);
  }
}

/// A created ID, throwing the error on the u64::MAX sentinel.
int createdId(GrafeoBindings bindings, int id) {
  if (id == -1) throwLastError(bindings);
  return id;
}

/// A 1/0 flag, throwing the error on -1.
bool flag(GrafeoBindings bindings, int result) {
  if (result < 0) throwLastError(bindings);
  return result == 1;
}

String _rows(List<Map<String, dynamic>> rows) =>
    '[${rows.map(encodeParams).join(',')}]';

String _edges(List<BatchEdge> edges) => '[${edges.map((e) {
      final properties = e.properties;
      return '{"src":${e.sourceId},"dst":${e.targetId},'
          '"type":${jsonEncode(e.type)}'
          '${properties == null ? '' : ',"properties":${encodeParams(properties)}'}}';
    }).join(',')}]';

/// A node batch through [call], which gets the labels and rows JSON.
List<int> batchNodes(
  GrafeoBindings bindings,
  List<String> labels,
  List<Map<String, dynamic>> rows,
  int Function(Pointer<Utf8> labels, Pointer<Utf8> rows,
          Pointer<Pointer<Uint64>> ids, Pointer<IntPtr> count)
      call,
) =>
    withNativeStrings([jsonEncode(labels), _rows(rows)], (p) {
      return takeIds(bindings, (ids, count) => call(p[0], p[1], ids, count));
    });

/// An edge batch through [call], which gets the edges JSON.
List<int> batchEdges(
  GrafeoBindings bindings,
  List<BatchEdge> edges,
  int Function(Pointer<Utf8> edges, Pointer<Pointer<Uint64>> ids,
          Pointer<IntPtr> count)
      call,
) =>
    withNativeStrings([_edges(edges)], (p) {
      return takeIds(bindings, (ids, count) => call(p[0], ids, count));
    });

/// A node upsert through [call], which gets the labels, rows and options JSON.
UpsertSummary upsertNodes(
  GrafeoBindings bindings,
  List<String> labels,
  List<Map<String, dynamic>> rows,
  String? key,
  bool replace,
  int Function(Pointer<Utf8> labels, Pointer<Utf8> rows, Pointer<Utf8> options,
          Pointer<Pointer<Utf8>> json)
      call,
) {
  final options = jsonEncode({if (key != null) 'key': key, 'replace': replace});
  return withNativeStrings([jsonEncode(labels), _rows(rows), options], (p) {
    return _summary(takeJson(bindings, (json) => call(p[0], p[1], p[2], json)));
  });
}

/// An edge upsert through [call], which gets the type, rows and options JSON.
UpsertSummary upsertEdges(
  GrafeoBindings bindings,
  String edgeType,
  List<Map<String, dynamic>> rows,
  Map<String, Object?> options,
  int Function(Pointer<Utf8> edgeType, Pointer<Utf8> rows,
          Pointer<Utf8> options, Pointer<Pointer<Utf8>> json)
      call,
) {
  final optionsJson = jsonEncode({
    for (final entry in options.entries)
      if (entry.value != null) entry.key: entry.value,
  });
  return withNativeStrings([edgeType, _rows(rows), optionsJson], (p) {
    return _summary(takeJson(bindings, (json) => call(p[0], p[1], p[2], json)));
  });
}

UpsertSummary _summary(String json) {
  final map = jsonDecode(json) as Map<String, dynamic>;
  return UpsertSummary(
    map['created'] as int,
    map['updated'] as int,
    map['skipped'] as int,
    (map['skipped_rows'] as List).cast<int>(),
  );
}

/// Parses a JSON array of change events.
List<ChangeEvent> parseEvents(String json) {
  List<String>? strings(Object? value) =>
      value == null ? null : (value as List).cast<String>();
  Map<String, dynamic>? properties(Object? value) =>
      value == null ? null : decodeObject(value);
  return [
    for (final e in (jsonDecode(json) as List).cast<Map<String, dynamic>>())
      ChangeEvent(
        entityId: e['entity_id'] as int,
        entityType: e['entity_type'] as String,
        graph: e['graph'] as String?,
        kind: e['kind'] as String,
        epoch: e['epoch'] as int,
        timestamp: e['timestamp'] as int,
        before: properties(e['before']),
        after: properties(e['after']),
        labels: strings(e['labels']),
        beforeLabels: strings(e['before_labels']),
        edgeType: e['edge_type'] as String?,
        sourceId: e['src_id'] as int?,
        targetId: e['dst_id'] as int?,
        deletedWith: e['deleted_with'] as int?,
      ),
  ];
}

/// Parses a batch vector search answer: one list per query.
List<List<VectorResult>> parseVectorResults(String json) => [
      for (final hits in jsonDecode(json) as List)
        [
          for (final hit in (hits as List).cast<Map<String, dynamic>>())
            VectorResult(hit['id'] as int, (hit['distance'] as num).toDouble()),
        ],
    ];
