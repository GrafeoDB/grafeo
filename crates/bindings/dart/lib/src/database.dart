/// The main GrafeoDB database class.
///
/// Wraps the grafeo-c shared library via FFI. Uses [NativeFinalizer] to
/// prevent leaks if [close] is not called explicitly.
library;

import 'dart:convert';
import 'dart:ffi';

import 'package:ffi/ffi.dart';

import 'error.dart';
import 'ffi/bindings.dart';
import 'ffi/loader.dart';
import 'stream.dart';
import 'transaction.dart';
import 'types.dart';
import 'value.dart';

/// A Grafeo graph database instance.
///
/// Create with [GrafeoDB.memory] (in-memory) or [GrafeoDB.open] (persistent).
/// Always call [close] when done, or rely on [NativeFinalizer] as a safety net.
class GrafeoDB implements Finalizable {
  final GrafeoBindings _bindings;
  Pointer<Void> _handle;
  bool _closed = false;

  static NativeFinalizer? _finalizer;

  GrafeoDB._(this._handle, this._bindings) {
    // Lazily create a finalizer that calls grafeo_free_database on the handle.
    // The Rust Drop impl for the inner Arc<RwLock<GrafeoDB>> flushes writes.
    _finalizer ??= NativeFinalizer(
      _bindings.library.lookup<NativeFunction<Void Function(Pointer<Void>)>>(
        'grafeo_free_database',
      ),
    );
    _finalizer!.attach(this, _handle.cast(), detach: this);
  }

  // ===========================================================================
  // Lifecycle
  // ===========================================================================

  /// Create a new in-memory database.
  static GrafeoDB memory({String? libraryPath}) {
    final lib = loadNativeLibrary(libraryPath);
    final bindings = GrafeoBindings(lib);
    final ptr = bindings.grafeoOpenMemory();
    if (ptr == nullptr) throwLastError(bindings);
    return GrafeoDB._(ptr, bindings);
  }

  /// Open a persistent database at [path].
  static GrafeoDB open(String path, {String? libraryPath}) {
    final lib = loadNativeLibrary(libraryPath);
    final bindings = GrafeoBindings(lib);
    final pathPtr = path.toNativeUtf8(allocator: malloc);
    try {
      final ptr = bindings.grafeoOpen(pathPtr);
      if (ptr == nullptr) throwLastError(bindings);
      return GrafeoDB._(ptr, bindings);
    } finally {
      malloc.free(pathPtr);
    }
  }

  /// Open or create a persistent database at [path] (a single file).
  ///
  /// All data is stored in one file with a sidecar WAL for crash safety,
  /// similar to DuckDB's `.duckdb` format. Since 0.6 every database is a
  /// single file, whatever the extension of its path, so this is the same
  /// as [open].
  static GrafeoDB openSingleFile(String path, {String? libraryPath}) {
    final lib = loadNativeLibrary(libraryPath);
    final bindings = GrafeoBindings(lib);
    final pathPtr = path.toNativeUtf8(allocator: malloc);
    try {
      final ptr = bindings.grafeoOpenSingleFile(pathPtr);
      if (ptr == nullptr) throwLastError(bindings);
      return GrafeoDB._(ptr, bindings);
    } finally {
      malloc.free(pathPtr);
    }
  }

  /// Open an existing database at [path] in read-only mode.
  ///
  /// Multiple read-only handles may be opened concurrently on the same path.
  /// Write operations on a read-only database will throw [DatabaseException].
  static GrafeoDB openReadOnly(String path, {String? libraryPath}) {
    final lib = loadNativeLibrary(libraryPath);
    final bindings = GrafeoBindings(lib);
    final pathPtr = path.toNativeUtf8(allocator: malloc);
    try {
      final ptr = bindings.grafeoOpenReadOnly(pathPtr);
      if (ptr == nullptr) throwLastError(bindings);
      return GrafeoDB._(ptr, bindings);
    } finally {
      malloc.free(pathPtr);
    }
  }

  /// Close the database, flushing all writes.
  ///
  /// Safe to call multiple times. After close, all other methods throw.
  void close() {
    if (_closed) return;
    _closed = true;
    _finalizer!.detach(this);
    final status = _bindings.grafeoClose(_handle);
    _bindings.grafeoFreeDatabase(_handle);
    _handle = nullptr;
    if (status != GrafeoStatus.ok.code) {
      throw classifyError(status, lastError(_bindings));
    }
  }

  /// Returns the grafeo-c library version.
  ///
  /// The C function returns a pointer to a static string that must NOT
  /// be freed (it lives in the binary's read-only data segment).
  static String version({String? libraryPath}) {
    final lib = loadNativeLibrary(libraryPath);
    final bindings = GrafeoBindings(lib);
    final ptr = bindings.grafeoVersion();
    return ptr.toDartString();
  }

  void _checkOpen() {
    if (_closed) {
      throw DatabaseException('Database is closed', GrafeoStatus.database);
    }
  }

  // ===========================================================================
  // Query execution
  // ===========================================================================

  /// Execute a GQL query and return a [QueryResult].
  QueryResult execute(String query) {
    _checkOpen();
    final queryPtr = query.toNativeUtf8(allocator: malloc);
    try {
      final resultPtr = _bindings.grafeoExecute(_handle, queryPtr);
      if (resultPtr == nullptr) throwLastError(_bindings);
      return _buildResult(resultPtr);
    } finally {
      malloc.free(queryPtr);
    }
  }

  /// Open a lazy cursor over a read-only GQL query.
  ///
  /// Rows are pulled from the Rust engine on demand, so memory stays bounded
  /// regardless of result size. Rejects mutations, EXPLAIN / PROFILE,
  /// session/schema commands, and queries that compile to push-based
  /// pipelines (ORDER BY, aggregate, DISTINCT). Use [execute] for those.
  ///
  /// Always call `stream.close()` (or rely on the [NativeFinalizer] as a
  /// safety net) to release the underlying Rust iterator.
  ///
  /// Experimental (0.5.40+): API may change before Beta.
  ResultStream executeStream(String query) {
    _checkOpen();
    return ResultStream.open(_handle, _bindings, query);
  }

  /// Execute a GQL query with parameters.
  ///
  /// Parameters are JSON-encoded using the grafeo-bindings-common wire format.
  /// Temporal types (DateTime, Duration) are automatically encoded as
  /// `$timestamp_us`, `$duration`, etc.
  QueryResult executeWithParams(
    String query,
    Map<String, dynamic> params,
  ) {
    _checkOpen();
    final queryPtr = query.toNativeUtf8(allocator: malloc);
    final paramsJson = encodeParams(params);
    final paramsPtr = paramsJson.toNativeUtf8(allocator: malloc);
    try {
      final resultPtr = _bindings.grafeoExecuteWithParams(
        _handle,
        queryPtr,
        paramsPtr,
      );
      if (resultPtr == nullptr) throwLastError(_bindings);
      return _buildResult(resultPtr);
    } finally {
      malloc.free(queryPtr);
      malloc.free(paramsPtr);
    }
  }

  /// Execute a Cypher query (requires `cypher` feature in grafeo-c).
  QueryResult executeCypher(String query) => _executeLanguage(
        query,
        _bindings.grafeoExecuteCypher,
      );

  /// Execute a Gremlin query (requires `gremlin` feature in grafeo-c).
  QueryResult executeGremlin(String query) => _executeLanguage(
        query,
        _bindings.grafeoExecuteGremlin,
      );

  /// Execute a GraphQL query (requires `graphql` feature in grafeo-c).
  QueryResult executeGraphql(String query) => _executeLanguage(
        query,
        _bindings.grafeoExecuteGraphql,
      );

  /// Execute a SPARQL query (requires `sparql` feature in grafeo-c).
  QueryResult executeSparql(String query) => _executeLanguage(
        query,
        _bindings.grafeoExecuteSparql,
      );

  /// Execute a Cypher query with parameters (requires `cypher` feature in grafeo-c).
  QueryResult executeCypherWithParams(
    String query,
    Map<String, dynamic> params,
  ) =>
      _executeLanguageWithParams(
        query,
        params,
        _bindings.grafeoExecuteCypherWithParams,
      );

  /// Execute a Gremlin query with parameters (requires `gremlin` feature in grafeo-c).
  QueryResult executeGremlinWithParams(
    String query,
    Map<String, dynamic> params,
  ) =>
      _executeLanguageWithParams(
        query,
        params,
        _bindings.grafeoExecuteGremlinWithParams,
      );

  /// Execute a GraphQL query with parameters (requires `graphql` feature in grafeo-c).
  QueryResult executeGraphqlWithParams(
    String query,
    Map<String, dynamic> params,
  ) =>
      _executeLanguageWithParams(
        query,
        params,
        _bindings.grafeoExecuteGraphqlWithParams,
      );

  /// Execute a SPARQL query with parameters (requires `sparql` feature in grafeo-c).
  QueryResult executeSparqlWithParams(
    String query,
    Map<String, dynamic> params,
  ) =>
      _executeLanguageWithParams(
        query,
        params,
        _bindings.grafeoExecuteSparqlWithParams,
      );

  /// Execute a query in the given [language] with optional [params].
  ///
  /// [language] is one of: `"gql"`, `"cypher"`, `"gremlin"`, `"graphql"`,
  /// `"sparql"`, `"sql"`. Omit [params] for queries without parameters.
  QueryResult executeLanguage(
    String language,
    String query, {
    Map<String, dynamic>? params,
  }) {
    _checkOpen();
    final langPtr = language.toNativeUtf8(allocator: malloc);
    final queryPtr = query.toNativeUtf8(allocator: malloc);
    Pointer<Utf8>? paramsPtr;
    if (params != null) {
      paramsPtr = encodeParams(params).toNativeUtf8(allocator: malloc);
    }
    try {
      final resultPtr = _bindings.grafeoExecuteLanguage(
        _handle,
        langPtr,
        queryPtr,
        paramsPtr ?? nullptr.cast<Utf8>(),
      );
      if (resultPtr == nullptr) throwLastError(_bindings);
      return _buildResult(resultPtr);
    } finally {
      malloc.free(langPtr);
      malloc.free(queryPtr);
      if (paramsPtr != null) malloc.free(paramsPtr);
    }
  }

  QueryResult _executeLanguage(
    String query,
    Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>) fn,
  ) {
    _checkOpen();
    final queryPtr = query.toNativeUtf8(allocator: malloc);
    try {
      final resultPtr = fn(_handle, queryPtr);
      if (resultPtr == nullptr) throwLastError(_bindings);
      return _buildResult(resultPtr);
    } finally {
      malloc.free(queryPtr);
    }
  }

  QueryResult _executeLanguageWithParams(
    String query,
    Map<String, dynamic> params,
    Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>) fn,
  ) {
    _checkOpen();
    final queryPtr = query.toNativeUtf8(allocator: malloc);
    final paramsPtr = encodeParams(params).toNativeUtf8(allocator: malloc);
    try {
      final resultPtr = fn(_handle, queryPtr, paramsPtr);
      if (resultPtr == nullptr) throwLastError(_bindings);
      return _buildResult(resultPtr);
    } finally {
      malloc.free(queryPtr);
      malloc.free(paramsPtr);
    }
  }

  // ===========================================================================
  // Statistics and info
  // ===========================================================================

  /// Get the number of nodes in the database (O(1), synchronous).
  int get nodeCount {
    _checkOpen();
    return _bindings.grafeoNodeCount(_handle);
  }

  /// Get the number of edges in the database (O(1), synchronous).
  int get edgeCount {
    _checkOpen();
    return _bindings.grafeoEdgeCount(_handle);
  }

  /// Get database info as a parsed JSON map.
  ///
  /// The C function allocates a string that must be freed with
  /// grafeo_free_string.
  Map<String, dynamic> info() {
    _checkOpen();
    final ptr = _bindings.grafeoInfo(_handle);
    if (ptr == nullptr) throwLastError(_bindings);
    try {
      return parseObject(ptr.toDartString());
    } finally {
      _bindings.grafeoFreeString(ptr);
    }
  }

  // ===========================================================================
  // Schema context
  // ===========================================================================

  /// Set the active schema for subsequent execute calls on this database.
  ///
  /// Equivalent to running `SESSION SET SCHEMA 'schemaName'` via GQL, but
  /// without requiring a round-trip query. All queries issued after this call
  /// will be scoped to [schemaName] until [resetSchema] is called.
  ///
  /// Throws [DatabaseException] if the schema does not exist.
  void setSchema(String schemaName) {
    _checkOpen();
    final schemaPtr = schemaName.toNativeUtf8(allocator: malloc);
    try {
      final status = _bindings.grafeoSetSchema(_handle, schemaPtr);
      if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
    } finally {
      malloc.free(schemaPtr);
    }
  }

  /// Clear the active schema context, reverting to the default graph store.
  void resetSchema() {
    _checkOpen();
    final status = _bindings.grafeoResetSchema(_handle);
    if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
  }

  /// Return the currently active schema name, or `null` if none is set.
  String? currentSchema() {
    _checkOpen();
    final ptr = _bindings.grafeoCurrentSchema(_handle);
    if (ptr == nullptr) return null;
    try {
      return ptr.toDartString();
    } finally {
      _bindings.grafeoFreeString(ptr);
    }
  }

  // ===========================================================================
  // Transactions
  // ===========================================================================

  /// Begin a new transaction with the default isolation level.
  Transaction beginTransaction() {
    _checkOpen();
    final txPtr = _bindings.grafeoBeginTransaction(_handle);
    if (txPtr == nullptr) throwLastError(_bindings);
    return Transaction(txPtr, _bindings);
  }

  /// Begin a transaction with a specific [isolationLevel].
  Transaction beginTransactionWithIsolation(IsolationLevel isolationLevel) {
    _checkOpen();
    final txPtr = _bindings.grafeoBeginTransactionWithIsolation(
      _handle,
      isolationLevel.code,
    );
    if (txPtr == nullptr) throwLastError(_bindings);
    return Transaction(txPtr, _bindings);
  }

  // ===========================================================================
  // Node CRUD
  // ===========================================================================

  /// Create a node with [labels] and [properties]. Returns the new node ID.
  int createNode(List<String> labels, Map<String, dynamic> properties) {
    _checkOpen();
    final labelsJson = jsonEncode(labels);
    final propsJson = encodeParams(properties);
    final labelsPtr = labelsJson.toNativeUtf8(allocator: malloc);
    final propsPtr = propsJson.toNativeUtf8(allocator: malloc);
    try {
      final id = _bindings.grafeoCreateNode(_handle, labelsPtr, propsPtr);
      if (id == -1) throwLastError(_bindings); // C returns u64::MAX on error
      return id;
    } finally {
      malloc.free(labelsPtr);
      malloc.free(propsPtr);
    }
  }

  /// Get a node by [id]. Returns a [Node] or throws on error.
  Node getNode(int id) {
    _checkOpen();
    final outPtr = malloc<Pointer<Void>>();
    try {
      final status = _bindings.grafeoGetNode(_handle, id, outPtr);
      if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
      final nodePtr = outPtr.value;
      try {
        final nodeId = _bindings.grafeoNodeId(nodePtr);
        final labelsJson =
            _bindings.grafeoNodeLabelsJson(nodePtr).toDartString();
        final propsJson =
            _bindings.grafeoNodePropertiesJson(nodePtr).toDartString();
        return Node(
          nodeId,
          parseStringArray(labelsJson),
          parseObject(propsJson),
        );
      } finally {
        _bindings.grafeoFreeNode(nodePtr);
      }
    } finally {
      malloc.free(outPtr);
    }
  }

  /// Return the labels of node [id] without fetching the full node.
  ///
  /// More efficient than [getNode] when only labels are needed.
  List<String> getNodeLabels(int id) {
    _checkOpen();
    final ptr = _bindings.grafeoGetNodeLabels(_handle, id);
    if (ptr == nullptr) throwLastError(_bindings);
    try {
      return parseStringArray(ptr.toDartString());
    } finally {
      _bindings.grafeoFreeString(ptr);
    }
  }

  /// Delete a node by [id]. Returns true on success.
  bool deleteNode(int id) {
    _checkOpen();
    final result = _bindings.grafeoDeleteNode(_handle, id);
    if (result < 0) throwLastError(_bindings);
    return result == 1;
  }

  /// Set a property on node [id].
  void setNodeProperty(int id, String key, dynamic value) {
    _checkOpen();
    final keyPtr = key.toNativeUtf8(allocator: malloc);
    final valueJson = encodeValue(value);
    final valuePtr = valueJson.toNativeUtf8(allocator: malloc);
    try {
      final status = _bindings.grafeoSetNodeProperty(
        _handle,
        id,
        keyPtr,
        valuePtr,
      );
      if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
    } finally {
      malloc.free(keyPtr);
      malloc.free(valuePtr);
    }
  }

  /// Remove a property from node [id].
  void removeNodeProperty(int id, String key) {
    _checkOpen();
    final keyPtr = key.toNativeUtf8(allocator: malloc);
    try {
      final result = _bindings.grafeoRemoveNodeProperty(_handle, id, keyPtr);
      if (result < 0) throwLastError(_bindings);
    } finally {
      malloc.free(keyPtr);
    }
  }

  /// Add a label to node [id].
  void addNodeLabel(int id, String label) {
    _checkOpen();
    final labelPtr = label.toNativeUtf8(allocator: malloc);
    try {
      final result = _bindings.grafeoAddNodeLabel(_handle, id, labelPtr);
      if (result < 0) throwLastError(_bindings);
    } finally {
      malloc.free(labelPtr);
    }
  }

  /// Remove a label from node [id].
  void removeNodeLabel(int id, String label) {
    _checkOpen();
    final labelPtr = label.toNativeUtf8(allocator: malloc);
    try {
      final result = _bindings.grafeoRemoveNodeLabel(_handle, id, labelPtr);
      if (result < 0) throwLastError(_bindings);
    } finally {
      malloc.free(labelPtr);
    }
  }

  // ===========================================================================
  // Edge CRUD
  // ===========================================================================

  /// Create an edge from [sourceId] to [targetId] with the given [type] and
  /// [properties]. Returns the new edge ID.
  int createEdge(
    int sourceId,
    int targetId,
    String type,
    Map<String, dynamic> properties,
  ) {
    _checkOpen();
    final typePtr = type.toNativeUtf8(allocator: malloc);
    final propsJson = encodeParams(properties);
    final propsPtr = propsJson.toNativeUtf8(allocator: malloc);
    try {
      final id = _bindings.grafeoCreateEdge(
        _handle,
        sourceId,
        targetId,
        typePtr,
        propsPtr,
      );
      if (id == -1) throwLastError(_bindings); // C returns u64::MAX on error
      return id;
    } finally {
      malloc.free(typePtr);
      malloc.free(propsPtr);
    }
  }

  /// Get an edge by [id]. Returns an [Edge] or throws on error.
  Edge getEdge(int id) {
    _checkOpen();
    final outPtr = malloc<Pointer<Void>>();
    try {
      final status = _bindings.grafeoGetEdge(_handle, id, outPtr);
      if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
      final edgePtr = outPtr.value;
      try {
        final edgeId = _bindings.grafeoEdgeId(edgePtr);
        final sourceId = _bindings.grafeoEdgeSourceId(edgePtr);
        final targetId = _bindings.grafeoEdgeTargetId(edgePtr);
        final edgeType = _bindings.grafeoEdgeType(edgePtr).toDartString();
        final propsJson =
            _bindings.grafeoEdgePropertiesJson(edgePtr).toDartString();
        return Edge(
          edgeId,
          edgeType,
          sourceId,
          targetId,
          parseObject(propsJson),
        );
      } finally {
        _bindings.grafeoFreeEdge(edgePtr);
      }
    } finally {
      malloc.free(outPtr);
    }
  }

  /// Delete an edge by [id]. Returns true on success.
  bool deleteEdge(int id) {
    _checkOpen();
    final result = _bindings.grafeoDeleteEdge(_handle, id);
    if (result < 0) throwLastError(_bindings);
    return result == 1;
  }

  /// Set a property on edge [id].
  void setEdgeProperty(int id, String key, dynamic value) {
    _checkOpen();
    final keyPtr = key.toNativeUtf8(allocator: malloc);
    final valueJson = encodeValue(value);
    final valuePtr = valueJson.toNativeUtf8(allocator: malloc);
    try {
      final status = _bindings.grafeoSetEdgeProperty(
        _handle,
        id,
        keyPtr,
        valuePtr,
      );
      if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
    } finally {
      malloc.free(keyPtr);
      malloc.free(valuePtr);
    }
  }

  /// Remove a property from edge [id].
  void removeEdgeProperty(int id, String key) {
    _checkOpen();
    final keyPtr = key.toNativeUtf8(allocator: malloc);
    try {
      final result = _bindings.grafeoRemoveEdgeProperty(_handle, id, keyPtr);
      if (result < 0) throwLastError(_bindings);
    } finally {
      malloc.free(keyPtr);
    }
  }

  // ===========================================================================
  // Property indexes
  // ===========================================================================

  /// Create a property index on [propertyKey] for fast point-lookup queries.
  ///
  /// After creation, `MATCH (n {name: $name})` style queries use the index
  /// instead of a full scan. [propertyKey] is the bare property name (e.g.
  /// `"name"`, not `"n.name"`).
  void createPropertyIndex(String propertyKey) {
    _checkOpen();
    final keyPtr = propertyKey.toNativeUtf8(allocator: malloc);
    try {
      final status = _bindings.grafeoCreatePropertyIndex(_handle, keyPtr);
      if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
    } finally {
      malloc.free(keyPtr);
    }
  }

  /// Drop the property index on [propertyKey]. Returns true if it existed.
  bool dropPropertyIndex(String propertyKey) {
    _checkOpen();
    final keyPtr = propertyKey.toNativeUtf8(allocator: malloc);
    try {
      final result = _bindings.grafeoDropPropertyIndex(_handle, keyPtr);
      if (result < 0) throwLastError(_bindings);
      return result == 1;
    } finally {
      malloc.free(keyPtr);
    }
  }

  /// Returns true if a property index exists for [propertyKey].
  bool hasPropertyIndex(String propertyKey) {
    _checkOpen();
    final keyPtr = propertyKey.toNativeUtf8(allocator: malloc);
    try {
      return _bindings.grafeoHasPropertyIndex(_handle, keyPtr) != 0;
    } finally {
      malloc.free(keyPtr);
    }
  }

  /// Find all node IDs where [propertyKey] equals [value].
  ///
  /// Requires that a property index exists for [propertyKey] (see
  /// [createPropertyIndex]). Returns the matching node IDs; call [getNode]
  /// to retrieve full node data.
  List<int> findNodesByProperty(String propertyKey, dynamic value) {
    _checkOpen();
    final keyPtr = propertyKey.toNativeUtf8(allocator: malloc);
    final valueJson = encodeValue(value);
    final valuePtr = valueJson.toNativeUtf8(allocator: malloc);
    final outIdsPtr = malloc<Pointer<Uint64>>();
    final outCountPtr = malloc<IntPtr>();
    try {
      final status = _bindings.grafeoFindNodesByProperty(
        _handle,
        keyPtr,
        valuePtr,
        outIdsPtr,
        outCountPtr,
      );
      if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
      final count = outCountPtr.value;
      if (count == 0) return [];
      final ids = outIdsPtr.value;
      final result = [for (var i = 0; i < count; i++) ids[i]];
      _bindings.grafeoFreeNodeIds(ids, count);
      return result;
    } finally {
      malloc.free(keyPtr);
      malloc.free(valuePtr);
      malloc.free(outIdsPtr);
      malloc.free(outCountPtr);
    }
  }

  // ===========================================================================
  // Vector operations
  // ===========================================================================

  /// Create an HNSW vector index on nodes with [label] for the [property]
  /// field, which must contain a list of [dimensions] floats.
  ///
  /// [metric] is one of: `"cosine"`, `"euclidean"`, `"dot"`.
  /// [m] is the number of bidirectional links per node (typical: 16).
  /// [efConstruction] controls index quality vs. build time (typical: 200).
  void createVectorIndex(
    String label,
    String property,
    int dimensions,
    String metric, {
    int m = 16,
    int efConstruction = 200,
  }) {
    _checkOpen();
    final labelPtr = label.toNativeUtf8(allocator: malloc);
    final propertyPtr = property.toNativeUtf8(allocator: malloc);
    final metricPtr = metric.toNativeUtf8(allocator: malloc);
    try {
      final status = _bindings.grafeoCreateVectorIndex(
        _handle,
        labelPtr,
        propertyPtr,
        dimensions,
        metricPtr,
        m,
        efConstruction,
      );
      if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
    } finally {
      malloc.free(labelPtr);
      malloc.free(propertyPtr);
      malloc.free(metricPtr);
    }
  }

  /// Perform a k-nearest-neighbour vector search.
  ///
  /// Returns up to [k] results ordered by similarity. [ef] controls the
  /// search quality vs. speed trade-off (typical: 64). For diversity-aware
  /// search, use [mmrSearch] instead.
  List<VectorResult> vectorSearch(
    String label,
    String property,
    List<double> query, {
    required int k,
    int ef = 64,
  }) {
    _checkOpen();
    final labelPtr = label.toNativeUtf8(allocator: malloc);
    final propertyPtr = property.toNativeUtf8(allocator: malloc);
    final queryPtr = malloc<Float>(query.length);
    for (var i = 0; i < query.length; i++) {
      queryPtr[i] = query[i];
    }
    final outIdsPtr = malloc<Pointer<Uint64>>();
    final outDistsPtr = malloc<Pointer<Float>>();
    final outCountPtr = malloc<IntPtr>();

    try {
      final status = _bindings.grafeoVectorSearch(
        _handle,
        labelPtr,
        propertyPtr,
        queryPtr,
        query.length,
        k,
        ef,
        outIdsPtr,
        outDistsPtr,
        outCountPtr,
      );
      if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);

      final count = outCountPtr.value;
      if (count == 0) return [];

      final ids = outIdsPtr.value;
      final dists = outDistsPtr.value;
      final results = <VectorResult>[
        for (var i = 0; i < count; i++) VectorResult(ids[i], dists[i]),
      ];

      _bindings.grafeoFreeVectorResults(ids, dists, count);
      return results;
    } finally {
      malloc.free(labelPtr);
      malloc.free(propertyPtr);
      malloc.free(queryPtr);
      malloc.free(outIdsPtr);
      malloc.free(outDistsPtr);
      malloc.free(outCountPtr);
    }
  }

  /// Bulk-create [vectors.length] nodes, each labelled [label], with the
  /// embedding stored under [embeddingProperty].
  ///
  /// [vectors] is a list of N equal-length float vectors (one per node).
  /// Returns the list of created node IDs in insertion order.
  ///
  /// Requires the `vector-index` feature in grafeo-c.
  List<int> batchCreateNodes(
    String label,
    String embeddingProperty,
    List<List<double>> vectors,
  ) {
    _checkOpen();
    if (vectors.isEmpty) return [];
    final dimensions = vectors.first.length;
    final count = vectors.length;
    final labelPtr = label.toNativeUtf8(allocator: malloc);
    final propertyPtr = embeddingProperty.toNativeUtf8(allocator: malloc);
    final vectorPtr = malloc<Float>(count * dimensions);
    for (var i = 0; i < count; i++) {
      for (var j = 0; j < dimensions; j++) {
        vectorPtr[i * dimensions + j] = vectors[i][j];
      }
    }
    final outIdsPtr = malloc<Pointer<Uint64>>();
    final outCountPtr = malloc<IntPtr>();
    try {
      final status = _bindings.grafeoBatchCreateNodes(
        _handle,
        labelPtr,
        propertyPtr,
        vectorPtr,
        count,
        dimensions,
        outIdsPtr,
        outCountPtr,
      );
      if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
      final resultCount = outCountPtr.value;
      if (resultCount == 0) return [];
      final ids = outIdsPtr.value;
      final result = [for (var i = 0; i < resultCount; i++) ids[i]];
      _bindings.grafeoFreeNodeIds(ids, resultCount);
      return result;
    } finally {
      malloc.free(labelPtr);
      malloc.free(propertyPtr);
      malloc.free(vectorPtr);
      malloc.free(outIdsPtr);
      malloc.free(outCountPtr);
    }
  }

  /// Drop a vector index. Returns true if the index existed.
  bool dropVectorIndex(String label, String property) {
    _checkOpen();
    final labelPtr = label.toNativeUtf8(allocator: malloc);
    final propertyPtr = property.toNativeUtf8(allocator: malloc);
    try {
      final result = _bindings.grafeoDropVectorIndex(
        _handle,
        labelPtr,
        propertyPtr,
      );
      return result != 0;
    } finally {
      malloc.free(labelPtr);
      malloc.free(propertyPtr);
    }
  }

  /// Rebuild a vector index.
  void rebuildVectorIndex(String label, String property) {
    _checkOpen();
    final labelPtr = label.toNativeUtf8(allocator: malloc);
    final propertyPtr = property.toNativeUtf8(allocator: malloc);
    try {
      final status = _bindings.grafeoRebuildVectorIndex(
        _handle,
        labelPtr,
        propertyPtr,
      );
      if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
    } finally {
      malloc.free(labelPtr);
      malloc.free(propertyPtr);
    }
  }

  /// Perform an MMR (Maximal Marginal Relevance) vector search.
  List<VectorResult> mmrSearch(
    String label,
    String property,
    List<double> query, {
    required int k,
    required int fetchK,
    required double lambda,
    required int ef,
  }) {
    _checkOpen();
    final labelPtr = label.toNativeUtf8(allocator: malloc);
    final propertyPtr = property.toNativeUtf8(allocator: malloc);
    final queryPtr = malloc<Float>(query.length);
    for (var i = 0; i < query.length; i++) {
      queryPtr[i] = query[i];
    }
    final outIdsPtr = malloc<Pointer<Uint64>>();
    final outDistsPtr = malloc<Pointer<Float>>();
    final outCountPtr = malloc<IntPtr>();

    try {
      final status = _bindings.grafeoMmrSearch(
        _handle,
        labelPtr,
        propertyPtr,
        queryPtr,
        query.length,
        k,
        fetchK,
        lambda,
        ef,
        outIdsPtr,
        outDistsPtr,
        outCountPtr,
      );
      if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);

      final count = outCountPtr.value;
      if (count == 0) return [];

      final ids = outIdsPtr.value;
      final dists = outDistsPtr.value;
      final results = <VectorResult>[
        for (var i = 0; i < count; i++) VectorResult(ids[i], dists[i]),
      ];

      _bindings.grafeoFreeVectorResults(ids, dists, count);
      return results;
    } finally {
      malloc.free(labelPtr);
      malloc.free(propertyPtr);
      malloc.free(queryPtr);
      malloc.free(outIdsPtr);
      malloc.free(outDistsPtr);
      malloc.free(outCountPtr);
    }
  }

  // ===========================================================================
  // Admin
  // ===========================================================================

  /// Save a database snapshot to [path].
  void save(String path) {
    _checkOpen();
    final pathPtr = path.toNativeUtf8(allocator: malloc);
    try {
      final status = _bindings.grafeoSave(_handle, pathPtr);
      if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
    } finally {
      malloc.free(pathPtr);
    }
  }

  /// Force a WAL checkpoint.
  void walCheckpoint() {
    _checkOpen();
    final status = _bindings.grafeoWalCheckpoint(_handle);
    if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
  }

  // ===========================================================================
  // Internal helpers
  // ===========================================================================

  /// Extract a [QueryResult] from a native result pointer, then free it.
  QueryResult _buildResult(Pointer<Void> resultPtr) {
    try {
      final jsonPtr = _bindings.grafeoResultJson(resultPtr);
      final jsonString = jsonPtr.toDartString();
      final executionTimeMs = _bindings.grafeoResultExecutionTimeMs(resultPtr);
      final rowsScanned = _bindings.grafeoResultRowsScanned(resultPtr);

      final rows = parseRows(jsonString);
      final columns = extractColumns(rows);
      final (nodes, edges) = extractEntities(rows);

      return QueryResult(
        columns: columns,
        rows: rows,
        nodes: nodes,
        edges: edges,
        executionTimeMs: executionTimeMs,
        rowsScanned: rowsScanned,
      );
    } finally {
      _bindings.grafeoFreeResult(resultPtr);
    }
  }

}
