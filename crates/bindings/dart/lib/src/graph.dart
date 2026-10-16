/// A handle on one named graph of a database.
library;

import 'dart:convert';
import 'dart:ffi';

import 'batch.dart';
import 'error.dart';
import 'ffi/bindings.dart';
import 'native.dart' hide upsertEdges, upsertNodes;
import 'native.dart' as native show upsertEdges, upsertNodes;
import 'types.dart';
import 'value.dart';

/// Works in one named graph of a database without switching the database's
/// current graph, from [GrafeoDB.graph]. Each call looks the graph up again:
/// once it is dropped, every call throws. Call [close] when done, or rely on
/// the [NativeFinalizer] as a safety net.
class GrafeoGraph implements Finalizable {
  final GrafeoBindings _bindings;
  Pointer<Void> _handle;
  bool _closed = false;

  static NativeFinalizer? _finalizer;

  /// Wraps a native graph handle; typically called by [GrafeoDB.graph].
  GrafeoGraph(this._handle, this._bindings) {
    _finalizer ??= NativeFinalizer(
      _bindings.library.lookup<NativeFunction<Void Function(Pointer<Void>)>>(
        'grafeo_free_graph',
      ),
    );
    _finalizer!.attach(this, _handle.cast(), detach: this);
  }

  /// Free the handle. Safe to call multiple times.
  void close() {
    if (_closed) return;
    _closed = true;
    _finalizer!.detach(this);
    _bindings.grafeoFreeGraph(_handle);
    _handle = nullptr;
  }

  Pointer<Void> get _open {
    if (_closed) {
      throw DatabaseException('Graph handle is closed', GrafeoStatus.database);
    }
    return _handle;
  }

  QueryResult _result(Pointer<Void> resultPtr) {
    if (resultPtr == nullptr) throwLastError(_bindings);
    return buildResult(_bindings, resultPtr);
  }

  // ===========================================================================
  // Query execution
  // ===========================================================================

  /// Execute a GQL query in the graph.
  QueryResult execute(String query) => withNativeStrings(
      [query], (p) => _result(_bindings.grafeoGraphExecute(_open, p[0])));

  /// Execute a GQL query with parameters in the graph.
  QueryResult executeWithParams(String query, Map<String, dynamic> params) =>
      withNativeStrings(
          [query, encodeParams(params)],
          (p) => _result(
              _bindings.grafeoGraphExecuteWithParams(_open, p[0], p[1])));

  /// Execute a query in the given [language] (`"gql"`, `"cypher"`, ...) with
  /// optional [params] in the graph.
  QueryResult executeLanguage(
    String language,
    String query, {
    Map<String, dynamic>? params,
  }) =>
      withNativeStrings(
          [language, query, params == null ? null : encodeParams(params)],
          (p) => _result(
              _bindings.grafeoGraphExecuteLanguage(_open, p[0], p[1], p[2])));

  // ===========================================================================
  // Nodes and edges
  // ===========================================================================

  /// Create a node with [labels] and [properties]. Returns the new node ID.
  int createNode(List<String> labels, Map<String, dynamic> properties) =>
      withNativeStrings(
          [jsonEncode(labels), encodeParams(properties)],
          (p) => createdId(
              _bindings, _bindings.grafeoGraphCreateNode(_open, p[0], p[1])));

  /// Create an edge from [sourceId] to [targetId]. Returns the new edge ID.
  int createEdge(
    int sourceId,
    int targetId,
    String type,
    Map<String, dynamic> properties,
  ) =>
      withNativeStrings(
          [type, encodeParams(properties)],
          (p) => createdId(
              _bindings,
              _bindings.grafeoGraphCreateEdge(
                  _open, sourceId, targetId, p[0], p[1])));

  /// Get a node by [id]; throws when it does not exist.
  Node getNode(int id) => takeNode(
      _bindings, (out) => _bindings.grafeoGraphGetNode(_open, id, out));

  /// Get an edge by [id]; throws when it does not exist.
  Edge getEdge(int id) => takeEdge(
      _bindings, (out) => _bindings.grafeoGraphGetEdge(_open, id, out));

  /// [GrafeoDB.getNodeEdges] in the graph.
  List<NodeEdge> getNodeEdges(
    int id, {
    Direction direction = Direction.out,
    List<String>? edgeTypes,
  }) =>
      nodeEdges(
          _bindings,
          edgeTypes,
          (types, neighbors, edges, count) => _bindings.grafeoGraphGetNodeEdges(
              _open, id, direction.code, types, neighbors, edges, count));

  /// Set a property on node [id].
  void setNodeProperty(int id, String key, dynamic value) =>
      withNativeStrings([key, encodeValue(value)], (p) {
        final status =
            _bindings.grafeoGraphSetNodeProperty(_open, id, p[0], p[1]);
        if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
      });

  /// Set a property on edge [id].
  void setEdgeProperty(int id, String key, dynamic value) =>
      withNativeStrings([key, encodeValue(value)], (p) {
        final status =
            _bindings.grafeoGraphSetEdgeProperty(_open, id, p[0], p[1]);
        if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
      });

  /// Remove a property from node [id]; whether it was there.
  bool removeNodeProperty(int id, String key) => withNativeStrings(
      [key],
      (p) => flag(
          _bindings, _bindings.grafeoGraphRemoveNodeProperty(_open, id, p[0])));

  /// Remove a property from edge [id]; whether it was there.
  bool removeEdgeProperty(int id, String key) => withNativeStrings(
      [key],
      (p) => flag(
          _bindings, _bindings.grafeoGraphRemoveEdgeProperty(_open, id, p[0])));

  /// Add a label to node [id]; false if it had it.
  bool addNodeLabel(int id, String label) => withNativeStrings(
      [label],
      (p) =>
          flag(_bindings, _bindings.grafeoGraphAddNodeLabel(_open, id, p[0])));

  /// Remove a label from node [id]; whether it had it.
  bool removeNodeLabel(int id, String label) => withNativeStrings(
      [label],
      (p) => flag(
          _bindings, _bindings.grafeoGraphRemoveNodeLabel(_open, id, p[0])));

  /// Delete node [id] (one with edges throws); whether it existed.
  bool deleteNode(int id) =>
      flag(_bindings, _bindings.grafeoGraphDeleteNode(_open, id));

  /// Delete node [id] with all its edges, as `DETACH DELETE` does; whether it
  /// existed.
  bool detachDeleteNode(int id) =>
      flag(_bindings, _bindings.grafeoGraphDetachDeleteNode(_open, id));

  /// Delete edge [id]; whether it existed.
  bool deleteEdge(int id) =>
      flag(_bindings, _bindings.grafeoGraphDeleteEdge(_open, id));

  // ===========================================================================
  // Batch writes and upserts
  // ===========================================================================

  /// [GrafeoDB.batchCreateNodesWithProps] in the graph.
  List<int> batchCreateNodesWithProps(
    List<String> labels,
    List<Map<String, dynamic>> rows,
  ) =>
      batchNodes(
          _bindings,
          labels,
          rows,
          (labels, rows, ids, count) =>
              _bindings.grafeoGraphBatchCreateNodesWithProps(
                  _open, labels, rows, ids, count));

  /// [GrafeoDB.batchCreateEdges] in the graph.
  List<int> batchCreateEdges(List<BatchEdge> edges) => batchEdges(
      _bindings,
      edges,
      (edges, ids, count) =>
          _bindings.grafeoGraphBatchCreateEdges(_open, edges, ids, count));

  /// [GrafeoDB.upsertNodes] in the graph.
  UpsertSummary upsertNodes(
    List<String> labels,
    List<Map<String, dynamic>> rows, {
    String? key,
    bool replace = false,
  }) =>
      native.upsertNodes(
          _bindings,
          labels,
          rows,
          key,
          replace,
          (labels, rows, options, json) => _bindings.grafeoGraphUpsertNodes(
              _open, labels, rows, options, json));

  /// [GrafeoDB.upsertEdges] in the graph.
  UpsertSummary upsertEdges(
    String edgeType,
    List<Map<String, dynamic>> rows, {
    String? key,
    String? endpointKey,
    List<String>? endpointLabels,
    String? sourceField,
    String? targetField,
    bool replace = false,
  }) =>
      native.upsertEdges(
          _bindings,
          edgeType,
          rows,
          {
            'key': key,
            'endpoint_key': endpointKey,
            'endpoint_labels': endpointLabels,
            'src_field': sourceField,
            'dst_field': targetField,
            'replace': replace,
          },
          (edgeType, rows, options, json) => _bindings.grafeoGraphUpsertEdges(
              _open, edgeType, rows, options, json));
}
