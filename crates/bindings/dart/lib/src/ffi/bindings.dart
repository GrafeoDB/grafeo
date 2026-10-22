/// Raw FFI bindings for every grafeo-c function.
///
/// All lookups use `late final` so the symbol resolution happens once per
/// binding instance, not on every call.
library;

import 'dart:ffi';

import 'package:ffi/ffi.dart';

/// Statically-typed FFI bindings for the grafeo-c shared library.
final class GrafeoBindings {
  /// The underlying [DynamicLibrary] used for symbol lookups.
  final DynamicLibrary library;

  GrafeoBindings(this.library);

  // ===========================================================================
  // Error handling
  // ===========================================================================

  /// Returns the last error message (thread-local). Do NOT free the pointer.
  late final grafeoLastError = library
      .lookupFunction<Pointer<Utf8> Function(), Pointer<Utf8> Function()>(
    'grafeo_last_error',
  );

  /// Clears the thread-local error state.
  late final grafeoClearError =
      library.lookupFunction<Void Function(), void Function()>(
    'grafeo_clear_error',
  );

  /// Free a heap-allocated string returned by grafeo-c (e.g. grafeo_info).
  late final grafeoFreeString = library.lookupFunction<
      Void Function(Pointer<Utf8>),
      void Function(
        Pointer<Utf8>,
      )>('grafeo_free_string');

  // ===========================================================================
  // Lifecycle
  // ===========================================================================

  /// Create a new in-memory database. Returns null on error.
  late final grafeoOpenMemory = library.lookupFunction<Pointer<Void> Function(),
      Pointer<Void> Function()>('grafeo_open_memory');

  /// Open a persistent database at [path]. Returns null on error.
  late final grafeoOpen = library.lookupFunction<
      Pointer<Void> Function(Pointer<Utf8>),
      Pointer<Void> Function(Pointer<Utf8>)>('grafeo_open');

  /// Open or create a persistent database (a single file, as with `grafeo_open`).
  late final grafeoOpenSingleFile = library.lookupFunction<
      Pointer<Void> Function(Pointer<Utf8>),
      Pointer<Void> Function(Pointer<Utf8>)>('grafeo_open_single_file');

  /// Open a persistent database in read-only mode. Returns null on error.
  late final grafeoOpenReadOnly = library.lookupFunction<
      Pointer<Void> Function(Pointer<Utf8>),
      Pointer<Void> Function(Pointer<Utf8>)>('grafeo_open_read_only');

  /// Close a database, flushing writes. Returns GrafeoStatus.
  late final grafeoClose = library.lookupFunction<Int32 Function(Pointer<Void>),
      int Function(Pointer<Void>)>('grafeo_close');

  /// Free the opaque database handle.
  late final grafeoFreeDatabase = library.lookupFunction<
      Void Function(Pointer<Void>),
      void Function(Pointer<Void>)>('grafeo_free_database');

  /// Returns the library version string. Static pointer, do NOT free.
  late final grafeoVersion = library
      .lookupFunction<Pointer<Utf8> Function(), Pointer<Utf8> Function()>(
    'grafeo_version',
  );

  // ===========================================================================
  // Query execution
  // ===========================================================================

  /// Execute a GQL query. Returns result pointer or null on error.
  late final grafeoExecute = library.lookupFunction<
      Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>),
      Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>)>('grafeo_execute');

  /// Execute a GQL query with JSON-encoded parameters.
  late final grafeoExecuteWithParams = library.lookupFunction<
      Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>),
      Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>,
          Pointer<Utf8>)>('grafeo_execute_with_params');

  /// Execute a Cypher query.
  late final grafeoExecuteCypher = library.lookupFunction<
      Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>),
      Pointer<Void> Function(
          Pointer<Void>, Pointer<Utf8>)>('grafeo_execute_cypher');

  /// Execute a Gremlin query.
  late final grafeoExecuteGremlin = library.lookupFunction<
      Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>),
      Pointer<Void> Function(
          Pointer<Void>, Pointer<Utf8>)>('grafeo_execute_gremlin');

  /// Execute a GraphQL query.
  late final grafeoExecuteGraphql = library.lookupFunction<
      Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>),
      Pointer<Void> Function(
          Pointer<Void>, Pointer<Utf8>)>('grafeo_execute_graphql');

  /// Execute a SPARQL query.
  late final grafeoExecuteSparql = library.lookupFunction<
      Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>),
      Pointer<Void> Function(
          Pointer<Void>, Pointer<Utf8>)>('grafeo_execute_sparql');

  /// Execute a Cypher query with JSON-encoded parameters.
  late final grafeoExecuteCypherWithParams = library.lookupFunction<
      Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>),
      Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>,
          Pointer<Utf8>)>('grafeo_execute_cypher_with_params');

  /// Execute a Gremlin query with JSON-encoded parameters.
  late final grafeoExecuteGremlinWithParams = library.lookupFunction<
      Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>),
      Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>,
          Pointer<Utf8>)>('grafeo_execute_gremlin_with_params');

  /// Execute a GraphQL query with JSON-encoded parameters.
  late final grafeoExecuteGraphqlWithParams = library.lookupFunction<
      Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>),
      Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>,
          Pointer<Utf8>)>('grafeo_execute_graphql_with_params');

  /// Execute a SPARQL query with JSON-encoded parameters.
  late final grafeoExecuteSparqlWithParams = library.lookupFunction<
      Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>),
      Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>,
          Pointer<Utf8>)>('grafeo_execute_sparql_with_params');

  /// Execute a query in any supported language with optional JSON parameters.
  /// [language] is one of: "gql", "cypher", "gremlin", "graphql", "sparql", "sql".
  late final grafeoExecuteLanguage = library.lookupFunction<
      Pointer<Void> Function(
          Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>),
      Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>,
          Pointer<Utf8>)>('grafeo_execute_language');

  // ===========================================================================
  // Result access
  // ===========================================================================

  /// Get JSON string from a result. Pointer valid until grafeo_free_result.
  late final grafeoResultJson = library.lookupFunction<
      Pointer<Utf8> Function(Pointer<Void>),
      Pointer<Utf8> Function(Pointer<Void>)>('grafeo_result_json');

  /// Get the number of rows in a result.
  late final grafeoResultRowCount = library.lookupFunction<
      IntPtr Function(Pointer<Void>),
      int Function(Pointer<Void>)>('grafeo_result_row_count');

  /// Get execution time in milliseconds.
  late final grafeoResultExecutionTimeMs = library.lookupFunction<
      Double Function(Pointer<Void>),
      double Function(Pointer<Void>)>('grafeo_result_execution_time_ms');

  /// Get the number of rows scanned.
  late final grafeoResultRowsScanned = library.lookupFunction<
      Uint64 Function(Pointer<Void>),
      int Function(Pointer<Void>)>('grafeo_result_rows_scanned');

  /// Free a result handle.
  late final grafeoFreeResult = library.lookupFunction<
      Void Function(Pointer<Void>),
      void Function(Pointer<Void>)>('grafeo_free_result');

  // ===========================================================================
  // Streaming (experimental, 0.5.40+)
  // ===========================================================================

  /// Open a streaming GQL query. Returns null on error.
  late final grafeoStreamOpen = library.lookupFunction<
      Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>),
      Pointer<Void> Function(
          Pointer<Void>, Pointer<Utf8>)>('grafeo_stream_open');

  /// Returns the column names as a JSON array string. Caller must
  /// [grafeoFreeString] the pointer.
  late final grafeoStreamColumnsJson = library.lookupFunction<
      Pointer<Utf8> Function(Pointer<Void>),
      Pointer<Utf8> Function(Pointer<Void>)>('grafeo_stream_columns_json');

  /// Pulls the next row into `*out_json`. Returns GrafeoStatus:
  /// - 0 (Ok) + non-null *out_json → row; caller frees the string
  /// - 0 (Ok) + null *out_json → stream exhausted
  /// - non-zero → error (call grafeoLastError for details)
  late final grafeoStreamNextRowJson = library.lookupFunction<
      Int32 Function(Pointer<Void>, Pointer<Pointer<Utf8>>),
      int Function(Pointer<Void>,
          Pointer<Pointer<Utf8>>)>('grafeo_stream_next_row_json');

  /// Frees a stream handle.
  late final grafeoStreamFree = library.lookupFunction<
      Void Function(Pointer<Void>),
      void Function(Pointer<Void>)>('grafeo_stream_free');

  // ===========================================================================
  // Schema context
  // ===========================================================================

  /// Set the current schema for subsequent execute calls.
  late final grafeoSetSchema = library.lookupFunction<
      Int32 Function(Pointer<Void>, Pointer<Utf8>),
      int Function(Pointer<Void>, Pointer<Utf8>)>('grafeo_set_schema');

  /// Clear the current schema context.
  late final grafeoResetSchema = library.lookupFunction<
      Int32 Function(Pointer<Void>),
      int Function(Pointer<Void>)>('grafeo_reset_schema');

  /// Returns the current schema name, or null if none is set.
  late final grafeoCurrentSchema = library.lookupFunction<
      Pointer<Utf8> Function(Pointer<Void>),
      Pointer<Utf8> Function(Pointer<Void>)>('grafeo_current_schema');

  // ===========================================================================
  // Node CRUD
  // ===========================================================================

  /// Create a node with JSON labels and properties. Returns node ID (0 = error).
  late final grafeoCreateNode = library.lookupFunction<
      Uint64 Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>),
      int Function(
          Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>)>('grafeo_create_node');

  /// Get a node by ID. Writes to [out]. Returns GrafeoStatus.
  late final grafeoGetNode = library.lookupFunction<
      Int32 Function(Pointer<Void>, Uint64, Pointer<Pointer<Void>>),
      int Function(
          Pointer<Void>, int, Pointer<Pointer<Void>>)>('grafeo_get_node');

  /// Delete a node by ID. Returns 0 on success, -1 on error.
  late final grafeoDeleteNode = library.lookupFunction<
      Int32 Function(Pointer<Void>, Uint64),
      int Function(Pointer<Void>, int)>('grafeo_delete_node');

  /// Set a property on a node. Returns GrafeoStatus.
  late final grafeoSetNodeProperty = library.lookupFunction<
          Int32 Function(
            Pointer<Void>,
            Uint64,
            Pointer<Utf8>,
            Pointer<Utf8>,
          ),
          int Function(Pointer<Void>, int, Pointer<Utf8>, Pointer<Utf8>)>(
      'grafeo_set_node_property');

  /// Remove a property from a node. Returns 0 on success, -1 on error.
  late final grafeoRemoveNodeProperty = library.lookupFunction<
      Int32 Function(Pointer<Void>, Uint64, Pointer<Utf8>),
      int Function(
          Pointer<Void>, int, Pointer<Utf8>)>('grafeo_remove_node_property');

  /// Add a label to a node. Returns 0 on success, -1 on error.
  late final grafeoAddNodeLabel = library.lookupFunction<
      Int32 Function(Pointer<Void>, Uint64, Pointer<Utf8>),
      int Function(Pointer<Void>, int, Pointer<Utf8>)>('grafeo_add_node_label');

  /// Remove a label from a node. Returns 0 on success, -1 on error.
  late final grafeoRemoveNodeLabel = library.lookupFunction<
      Int32 Function(Pointer<Void>, Uint64, Pointer<Utf8>),
      int Function(
          Pointer<Void>, int, Pointer<Utf8>)>('grafeo_remove_node_label');

  /// Get labels for a node as JSON. Caller must free with grafeoFreeString.
  late final grafeoGetNodeLabels = library.lookupFunction<
      Pointer<Utf8> Function(Pointer<Void>, Uint64),
      Pointer<Utf8> Function(Pointer<Void>, int)>('grafeo_get_node_labels');

  /// Get node ID from an opaque node pointer.
  late final grafeoNodeId = library.lookupFunction<
      Uint64 Function(Pointer<Void>),
      int Function(Pointer<Void>)>('grafeo_node_id');

  /// Get node labels JSON. Valid until grafeo_free_node.
  late final grafeoNodeLabelsJson = library.lookupFunction<
      Pointer<Utf8> Function(Pointer<Void>),
      Pointer<Utf8> Function(Pointer<Void>)>('grafeo_node_labels_json');

  /// Get node properties JSON. Valid until grafeo_free_node.
  late final grafeoNodePropertiesJson = library.lookupFunction<
      Pointer<Utf8> Function(Pointer<Void>),
      Pointer<Utf8> Function(Pointer<Void>)>('grafeo_node_properties_json');

  /// Free an opaque node handle.
  late final grafeoFreeNode = library.lookupFunction<
      Void Function(Pointer<Void>),
      void Function(Pointer<Void>)>('grafeo_free_node');

  // ===========================================================================
  // Edge CRUD
  // ===========================================================================

  /// Create an edge. Returns edge ID (0 = error).
  late final grafeoCreateEdge = library.lookupFunction<
          Uint64 Function(
            Pointer<Void>,
            Uint64,
            Uint64,
            Pointer<Utf8>,
            Pointer<Utf8>,
          ),
          int Function(Pointer<Void>, int, int, Pointer<Utf8>, Pointer<Utf8>)>(
      'grafeo_create_edge');

  /// Get an edge by ID. Writes to [out]. Returns GrafeoStatus.
  late final grafeoGetEdge = library.lookupFunction<
      Int32 Function(Pointer<Void>, Uint64, Pointer<Pointer<Void>>),
      int Function(
          Pointer<Void>, int, Pointer<Pointer<Void>>)>('grafeo_get_edge');

  /// Delete an edge by ID. Returns 0 on success, -1 on error.
  late final grafeoDeleteEdge = library.lookupFunction<
      Int32 Function(Pointer<Void>, Uint64),
      int Function(Pointer<Void>, int)>('grafeo_delete_edge');

  /// Set a property on an edge. Returns GrafeoStatus.
  late final grafeoSetEdgeProperty = library.lookupFunction<
          Int32 Function(
            Pointer<Void>,
            Uint64,
            Pointer<Utf8>,
            Pointer<Utf8>,
          ),
          int Function(Pointer<Void>, int, Pointer<Utf8>, Pointer<Utf8>)>(
      'grafeo_set_edge_property');

  /// Remove a property from an edge. Returns 0 on success, -1 on error.
  late final grafeoRemoveEdgeProperty = library.lookupFunction<
      Int32 Function(Pointer<Void>, Uint64, Pointer<Utf8>),
      int Function(
          Pointer<Void>, int, Pointer<Utf8>)>('grafeo_remove_edge_property');

  /// Get edge ID from an opaque edge pointer.
  late final grafeoEdgeId = library.lookupFunction<
      Uint64 Function(Pointer<Void>),
      int Function(Pointer<Void>)>('grafeo_edge_id');

  /// Get source node ID from an edge pointer.
  late final grafeoEdgeSourceId = library.lookupFunction<
      Uint64 Function(Pointer<Void>),
      int Function(Pointer<Void>)>('grafeo_edge_source_id');

  /// Get target node ID from an edge pointer.
  late final grafeoEdgeTargetId = library.lookupFunction<
      Uint64 Function(Pointer<Void>),
      int Function(Pointer<Void>)>('grafeo_edge_target_id');

  /// Get edge type string. Valid until grafeo_free_edge.
  late final grafeoEdgeType = library.lookupFunction<
      Pointer<Utf8> Function(Pointer<Void>),
      Pointer<Utf8> Function(Pointer<Void>)>('grafeo_edge_type');

  /// Get edge properties JSON. Valid until grafeo_free_edge.
  late final grafeoEdgePropertiesJson = library.lookupFunction<
      Pointer<Utf8> Function(Pointer<Void>),
      Pointer<Utf8> Function(Pointer<Void>)>('grafeo_edge_properties_json');

  /// Free an opaque edge handle.
  late final grafeoFreeEdge = library.lookupFunction<
      Void Function(Pointer<Void>),
      void Function(Pointer<Void>)>('grafeo_free_edge');

  // ===========================================================================
  // Property indexes
  // ===========================================================================

  /// Create a property index. Returns GrafeoStatus.
  late final grafeoCreatePropertyIndex = library.lookupFunction<
      Int32 Function(Pointer<Void>, Pointer<Utf8>),
      int Function(
          Pointer<Void>, Pointer<Utf8>)>('grafeo_create_property_index');

  /// Drop a property index. Returns 0 on success, -1 on error.
  late final grafeoDropPropertyIndex = library.lookupFunction<
      Int32 Function(Pointer<Void>, Pointer<Utf8>),
      int Function(Pointer<Void>, Pointer<Utf8>)>('grafeo_drop_property_index');

  /// Check if a property index exists. Returns 1 if exists, 0 if not.
  late final grafeoHasPropertyIndex = library.lookupFunction<
      Int32 Function(Pointer<Void>, Pointer<Utf8>),
      int Function(Pointer<Void>, Pointer<Utf8>)>('grafeo_has_property_index');

  /// Find nodes by property value. Writes IDs to [outIds], count to [outCount].
  late final grafeoFindNodesByProperty = library.lookupFunction<
      Int32 Function(
        Pointer<Void>,
        Pointer<Utf8>,
        Pointer<Utf8>,
        Pointer<Pointer<Uint64>>,
        Pointer<IntPtr>,
      ),
      int Function(
        Pointer<Void>,
        Pointer<Utf8>,
        Pointer<Utf8>,
        Pointer<Pointer<Uint64>>,
        Pointer<IntPtr>,
      )>('grafeo_find_nodes_by_property');

  /// Free a node ID array returned by grafeoFindNodesByProperty.
  late final grafeoFreeNodeIds = library.lookupFunction<
      Void Function(Pointer<Uint64>, IntPtr),
      void Function(Pointer<Uint64>, int)>('grafeo_free_node_ids');

  // ===========================================================================
  // Vector operations
  // ===========================================================================

  /// Create a vector index. Returns GrafeoStatus.
  late final grafeoCreateVectorIndex = library.lookupFunction<
      Int32 Function(
        Pointer<Void>,
        Pointer<Utf8>,
        Pointer<Utf8>,
        Int32,
        Pointer<Utf8>,
        Int32,
        Int32,
      ),
      int Function(
        Pointer<Void>,
        Pointer<Utf8>,
        Pointer<Utf8>,
        int,
        Pointer<Utf8>,
        int,
        int,
      )>('grafeo_create_vector_index');

  /// Drop a vector index. Returns 0 on success, -1 on error.
  late final grafeoDropVectorIndex = library.lookupFunction<
      Int32 Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>),
      int Function(Pointer<Void>, Pointer<Utf8>,
          Pointer<Utf8>)>('grafeo_drop_vector_index');

  /// Rebuild a vector index. Returns GrafeoStatus.
  late final grafeoRebuildVectorIndex = library.lookupFunction<
      Int32 Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>),
      int Function(Pointer<Void>, Pointer<Utf8>,
          Pointer<Utf8>)>('grafeo_rebuild_vector_index');

  /// Vector similarity search.
  late final grafeoVectorSearch = library.lookupFunction<
      Int32 Function(
        Pointer<Void>,
        Pointer<Utf8>,
        Pointer<Utf8>,
        Pointer<Float>,
        IntPtr,
        IntPtr,
        Int32,
        Pointer<Pointer<Uint64>>,
        Pointer<Pointer<Float>>,
        Pointer<IntPtr>,
      ),
      int Function(
        Pointer<Void>,
        Pointer<Utf8>,
        Pointer<Utf8>,
        Pointer<Float>,
        int,
        int,
        int,
        Pointer<Pointer<Uint64>>,
        Pointer<Pointer<Float>>,
        Pointer<IntPtr>,
      )>('grafeo_vector_search');

  /// MMR (Maximal Marginal Relevance) search.
  late final grafeoMmrSearch = library.lookupFunction<
      Int32 Function(
        Pointer<Void>,
        Pointer<Utf8>,
        Pointer<Utf8>,
        Pointer<Float>,
        IntPtr,
        IntPtr,
        Int32,
        Float,
        Int32,
        Pointer<Pointer<Uint64>>,
        Pointer<Pointer<Float>>,
        Pointer<IntPtr>,
      ),
      int Function(
        Pointer<Void>,
        Pointer<Utf8>,
        Pointer<Utf8>,
        Pointer<Float>,
        int,
        int,
        int,
        double,
        int,
        Pointer<Pointer<Uint64>>,
        Pointer<Pointer<Float>>,
        Pointer<IntPtr>,
      )>('grafeo_mmr_search');

  /// Batch-create nodes with vector embeddings.
  /// [outCount] receives the number of IDs written to [outIds].
  late final grafeoBatchCreateNodes = library.lookupFunction<
      Int32 Function(
        Pointer<Void>,
        Pointer<Utf8>,
        Pointer<Utf8>,
        Pointer<Float>,
        IntPtr,
        IntPtr,
        Pointer<Pointer<Uint64>>,
        Pointer<IntPtr>,
      ),
      int Function(
        Pointer<Void>,
        Pointer<Utf8>,
        Pointer<Utf8>,
        Pointer<Float>,
        int,
        int,
        Pointer<Pointer<Uint64>>,
        Pointer<IntPtr>,
      )>('grafeo_batch_create_nodes');

  /// Free vector search results.
  late final grafeoFreeVectorResults = library.lookupFunction<
      Void Function(Pointer<Uint64>, Pointer<Float>, IntPtr),
      void Function(
          Pointer<Uint64>, Pointer<Float>, int)>('grafeo_free_vector_results');

  // ===========================================================================
  // Statistics
  // ===========================================================================

  /// Get the number of nodes.
  late final grafeoNodeCount = library.lookupFunction<
      IntPtr Function(Pointer<Void>),
      int Function(Pointer<Void>)>('grafeo_node_count');

  /// Get the number of edges.
  late final grafeoEdgeCount = library.lookupFunction<
      IntPtr Function(Pointer<Void>),
      int Function(Pointer<Void>)>('grafeo_edge_count');

  // ===========================================================================
  // Transactions
  // ===========================================================================

  /// Begin a transaction. Returns null on error.
  late final grafeoBeginTransaction = library.lookupFunction<
      Pointer<Void> Function(Pointer<Void>),
      Pointer<Void> Function(Pointer<Void>)>('grafeo_begin_transaction');

  /// Begin a transaction with a specific isolation level.
  late final grafeoBeginTransactionWithIsolation = library.lookupFunction<
      Pointer<Void> Function(Pointer<Void>, Int32),
      Pointer<Void> Function(
          Pointer<Void>, int)>('grafeo_begin_transaction_with_isolation');

  /// Execute a query within a transaction.
  late final grafeoTransactionExecute = library.lookupFunction<
      Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>),
      Pointer<Void> Function(
          Pointer<Void>, Pointer<Utf8>)>('grafeo_transaction_execute');

  /// Execute a parameterized query within a transaction.
  late final grafeoTransactionExecuteWithParams = library.lookupFunction<
      Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>),
      Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>,
          Pointer<Utf8>)>('grafeo_transaction_execute_with_params');

  /// Execute a query in any supported language within a transaction.
  /// [language] is one of: "gql", "cypher", "gremlin", "graphql", "sparql", "sql".
  late final grafeoTransactionExecuteLanguage = library.lookupFunction<
      Pointer<Void> Function(
          Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>),
      Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>,
          Pointer<Utf8>)>('grafeo_transaction_execute_language');

  /// Commit a transaction. Returns GrafeoStatus.
  late final grafeoCommit = library.lookupFunction<
      Int32 Function(Pointer<Void>),
      int Function(Pointer<Void>)>('grafeo_commit');

  /// Rollback a transaction. Returns GrafeoStatus.
  late final grafeoRollback = library.lookupFunction<
      Int32 Function(Pointer<Void>),
      int Function(Pointer<Void>)>('grafeo_rollback');

  /// Free a transaction handle.
  late final grafeoFreeTransaction = library.lookupFunction<
      Void Function(Pointer<Void>),
      void Function(Pointer<Void>)>('grafeo_free_transaction');

  // ===========================================================================
  // Admin
  // ===========================================================================

  /// Get database info as JSON. Caller must free with grafeoFreeString.
  late final grafeoInfo = library.lookupFunction<
      Pointer<Utf8> Function(Pointer<Void>),
      Pointer<Utf8> Function(Pointer<Void>)>('grafeo_info');

  /// Save a snapshot to the given path. Returns GrafeoStatus.
  late final grafeoSave = library.lookupFunction<
      Int32 Function(Pointer<Void>, Pointer<Utf8>),
      int Function(Pointer<Void>, Pointer<Utf8>)>('grafeo_save');

  /// Force a WAL checkpoint. Returns GrafeoStatus.
  late final grafeoWalCheckpoint = library.lookupFunction<
      Int32 Function(Pointer<Void>),
      int Function(Pointer<Void>)>('grafeo_wal_checkpoint');
}
