// P/Invoke declarations for all grafeo-c exported functions.
// Uses .NET 8 LibraryImport source generation for AOT-safe marshalling.

using System.Runtime.InteropServices;

namespace Grafeo.Native;

internal static partial class NativeMethods
{
    private const string LibName = "grafeo_c";

    // =========================================================================
    // Lifecycle
    // =========================================================================

    /// <summary>Create a new in-memory database. Returns null on error.</summary>
    [LibraryImport(LibName)]
    internal static partial nint grafeo_open_memory();

    /// <summary>Open or create a persistent database at path. Returns null on error.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint grafeo_open(string path);

    /// <summary>Open an existing database in read-only mode. Returns null on error.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint grafeo_open_read_only(string path);

    /// <summary>Close the database, flushing pending writes. Returns status code.</summary>
    [LibraryImport(LibName)]
    internal static partial int grafeo_close(nint db);

    /// <summary>Free a database handle. Must be called after grafeo_close.</summary>
    [LibraryImport(LibName)]
    internal static partial void grafeo_free_database(nint db);

    /// <summary>Returns the library version string. The pointer is static, do NOT free.</summary>
    [LibraryImport(LibName)]
    internal static partial nint grafeo_version();

    // =========================================================================
    // Query Execution
    // =========================================================================

    /// <summary>Execute a GQL query. Returns result pointer, or null on error.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint grafeo_execute(nint db, string query);

    /// <summary>Execute a GQL query with JSON-encoded parameters.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint grafeo_execute_with_params(nint db, string query, string paramsJson);

    /// <summary>Execute a Cypher query.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint grafeo_execute_cypher(nint db, string query);

    /// <summary>Execute a Gremlin query.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint grafeo_execute_gremlin(nint db, string query);

    /// <summary>Execute a GraphQL query.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint grafeo_execute_graphql(nint db, string query);

    /// <summary>Execute a SPARQL query.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint grafeo_execute_sparql(nint db, string query);

    /// <summary>Execute a SQL/PGQ query.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint grafeo_execute_sql(nint db, string query);

    /// <summary>Execute a Cypher query with JSON-encoded parameters.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint grafeo_execute_cypher_with_params(nint db, string query, string paramsJson);

    /// <summary>Execute a Gremlin query with JSON-encoded parameters.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint grafeo_execute_gremlin_with_params(nint db, string query, string paramsJson);

    /// <summary>Execute a GraphQL query with JSON-encoded parameters.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint grafeo_execute_graphql_with_params(nint db, string query, string paramsJson);

    /// <summary>Execute a SPARQL query with JSON-encoded parameters.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint grafeo_execute_sparql_with_params(nint db, string query, string paramsJson);

    /// <summary>Execute a SQL/PGQ query with JSON-encoded parameters.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint grafeo_execute_sql_with_params(nint db, string query, string paramsJson);

    /// <summary>Execute a query in any supported language. Language: "gql", "cypher", "sparql", "gremlin", "graphql", "sql".</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint grafeo_execute_language(nint db, string language, string query, string? paramsJson);

    // =========================================================================
    // Result Access
    // =========================================================================

    /// <summary>Get the JSON string from a result. Valid until grafeo_free_result.</summary>
    [LibraryImport(LibName)]
    internal static partial nint grafeo_result_json(nint result);

    /// <summary>Get the execution time in milliseconds.</summary>
    [LibraryImport(LibName)]
    internal static partial double grafeo_result_execution_time_ms(nint result);

    /// <summary>Get the number of rows scanned.</summary>
    [LibraryImport(LibName)]
    internal static partial ulong grafeo_result_rows_scanned(nint result);

    /// <summary>Get pre-extracted nodes as JSON. Valid until grafeo_free_result.</summary>
    [LibraryImport(LibName)]
    internal static partial nint grafeo_result_nodes_json(nint result);

    /// <summary>Get pre-extracted edges as JSON. Valid until grafeo_free_result.</summary>
    [LibraryImport(LibName)]
    internal static partial nint grafeo_result_edges_json(nint result);

    /// <summary>Free a query result.</summary>
    [LibraryImport(LibName)]
    internal static partial void grafeo_free_result(nint result);

    // =========================================================================
    // Streaming (experimental, 0.5.40+)
    // =========================================================================

    /// <summary>Open a streaming GQL query. Returns stream handle or null on error.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint grafeo_stream_open(nint db, string query);

    /// <summary>Returns the column names as a JSON array. Caller must free with grafeo_free_string.</summary>
    [LibraryImport(LibName)]
    internal static partial nint grafeo_stream_columns_json(nint stream);

    /// <summary>Pulls the next row as a JSON object into out_json.
    /// Returns 0 (Ok) with non-null out_json: caller frees the string.
    /// Returns 0 (Ok) with null out_json: stream exhausted.
    /// Returns non-zero: error (call grafeo_last_error).</summary>
    [LibraryImport(LibName)]
    internal static partial int grafeo_stream_next_row_json(nint stream, out nint outJson);

    /// <summary>Frees a stream handle.</summary>
    [LibraryImport(LibName)]
    internal static partial void grafeo_stream_free(nint stream);

    // =========================================================================
    // Schema context
    // =========================================================================

    /// <summary>Set the current schema for subsequent execute calls.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int grafeo_set_schema(nint db, string name);

    /// <summary>Clear the current schema context.</summary>
    [LibraryImport(LibName)]
    internal static partial int grafeo_reset_schema(nint db);

    /// <summary>Returns the current schema name as a UTF-8 string pointer, or null if none is set.</summary>
    [LibraryImport(LibName)]
    internal static partial nint grafeo_current_schema(nint db);

    // =========================================================================
    // Node CRUD
    // =========================================================================

    /// <summary>Create a node with labels (JSON array) and properties (JSON object). Returns node ID or u64.MaxValue on error.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial ulong grafeo_create_node(nint db, string labelsJson, string? propertiesJson);

    /// <summary>Get a node by ID. Writes into out pointer. Returns status.</summary>
    [LibraryImport(LibName)]
    internal static partial int grafeo_get_node(nint db, ulong id, out nint nodeOut);

    /// <summary>Delete a node by ID. Returns 1 if deleted, 0 if not found, -1 on error.</summary>
    [LibraryImport(LibName)]
    internal static partial int grafeo_delete_node(nint db, ulong id);

    /// <summary>Set a property on a node.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int grafeo_set_node_property(nint db, ulong id, string key, string valueJson);

    /// <summary>Remove a property from a node. Returns 1 if removed, 0 if not found.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int grafeo_remove_node_property(nint db, ulong id, string key);

    /// <summary>Add a label to a node. Returns 1 if added, 0 if already present.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int grafeo_add_node_label(nint db, ulong id, string label);

    /// <summary>Remove a label from a node. Returns 1 if removed, 0 if not present.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int grafeo_remove_node_label(nint db, ulong id, string label);

    /// <summary>Free a GrafeoNode.</summary>
    [LibraryImport(LibName)]
    internal static partial void grafeo_free_node(nint node);

    /// <summary>Access node ID.</summary>
    [LibraryImport(LibName)]
    internal static partial ulong grafeo_node_id(nint node);

    /// <summary>Access labels JSON. Valid until grafeo_free_node.</summary>
    [LibraryImport(LibName)]
    internal static partial nint grafeo_node_labels_json(nint node);

    /// <summary>Access properties JSON. Valid until grafeo_free_node.</summary>
    [LibraryImport(LibName)]
    internal static partial nint grafeo_node_properties_json(nint node);

    // =========================================================================
    // Edge CRUD
    // =========================================================================

    /// <summary>Create an edge. Returns edge ID or u64.MaxValue on error.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial ulong grafeo_create_edge(nint db, ulong sourceId, ulong targetId, string edgeType, string? propertiesJson);

    /// <summary>Get an edge by ID. Writes into out pointer. Returns status.</summary>
    [LibraryImport(LibName)]
    internal static partial int grafeo_get_edge(nint db, ulong id, out nint edgeOut);

    /// <summary>Delete an edge by ID. Returns 1 if deleted, 0 if not found, -1 on error.</summary>
    [LibraryImport(LibName)]
    internal static partial int grafeo_delete_edge(nint db, ulong id);

    /// <summary>Set a property on an edge.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int grafeo_set_edge_property(nint db, ulong id, string key, string valueJson);

    /// <summary>Remove a property from an edge. Returns 1 if removed, 0 if not found.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int grafeo_remove_edge_property(nint db, ulong id, string key);

    /// <summary>Free a GrafeoEdge.</summary>
    [LibraryImport(LibName)]
    internal static partial void grafeo_free_edge(nint edge);

    /// <summary>Access edge ID.</summary>
    [LibraryImport(LibName)]
    internal static partial ulong grafeo_edge_id(nint edge);

    /// <summary>Access source node ID.</summary>
    [LibraryImport(LibName)]
    internal static partial ulong grafeo_edge_source_id(nint edge);

    /// <summary>Access target node ID.</summary>
    [LibraryImport(LibName)]
    internal static partial ulong grafeo_edge_target_id(nint edge);

    /// <summary>Access edge type string. Valid until grafeo_free_edge.</summary>
    [LibraryImport(LibName)]
    internal static partial nint grafeo_edge_type(nint edge);

    /// <summary>Access edge properties JSON. Valid until grafeo_free_edge.</summary>
    [LibraryImport(LibName)]
    internal static partial nint grafeo_edge_properties_json(nint edge);

    // =========================================================================
    // Indexes
    // =========================================================================

    /// <summary>Create a property index on a property key.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int grafeo_create_property_index(nint db, string property);

    /// <summary>Drop a property index.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int grafeo_drop_property_index(nint db, string property);

    /// <summary>Check if a property index exists. Returns 1 if exists, 0 if not.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int grafeo_has_property_index(nint db, string property);

    /// <summary>Find nodes by property value. Writes IDs and count to out pointers.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int grafeo_find_nodes_by_property(
        nint db, string property, string valueJson,
        out nint idsOut, out nuint countOut);

    /// <summary>Free node IDs returned by grafeo_find_nodes_by_property.</summary>
    [LibraryImport(LibName)]
    internal static partial void grafeo_free_node_ids(nint ids, nuint count);

    // =========================================================================
    // Vector Search
    // =========================================================================

    /// <summary>Create a vector index.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int grafeo_create_vector_index(
        nint db, string label, string property,
        int dimensions, string metric, int m, int efConstruction);

    /// <summary>Drop a vector index.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int grafeo_drop_vector_index(nint db, string label, string property);

    /// <summary>Rebuild a vector index.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int grafeo_rebuild_vector_index(nint db, string label, string property);

    /// <summary>Perform a vector search.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static unsafe partial int grafeo_vector_search(
        nint db, string label, string property,
        float* query, nuint queryLen, nuint k, uint ef,
        out nint idsOut, out nint distsOut, out nuint countOut);

    /// <summary>Perform an MMR search.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static unsafe partial int grafeo_mmr_search(
        nint db, string label, string property,
        float* query, nuint queryLen, nuint k,
        int fetchK, float lambda, int ef,
        out nint idsOut, out nint distsOut, out nuint countOut);

    /// <summary>Free vector search results.</summary>
    [LibraryImport(LibName)]
    internal static partial void grafeo_free_vector_results(nint ids, nint dists, nuint count);

    // =========================================================================
    // Batch Operations
    // =========================================================================

    /// <summary>Batch create nodes with vector properties.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static unsafe partial int grafeo_batch_create_nodes(
        nint db, string label, string property,
        float* vectors, nuint vectorCount, nuint dimensions,
        out nint outIds, out nuint outCount);

    // =========================================================================
    // Admin
    // =========================================================================

    /// <summary>Get the number of nodes.</summary>
    [LibraryImport(LibName)]
    internal static partial nuint grafeo_node_count(nint db);

    /// <summary>Get the number of edges.</summary>
    [LibraryImport(LibName)]
    internal static partial nuint grafeo_edge_count(nint db);

    /// <summary>Get database info as JSON. Caller must free with grafeo_free_string.</summary>
    [LibraryImport(LibName)]
    internal static partial nint grafeo_info(nint db);

    /// <summary>Save database to path.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int grafeo_save(nint db, string path);

    /// <summary>Checkpoint the WAL.</summary>
    [LibraryImport(LibName)]
    internal static partial int grafeo_wal_checkpoint(nint db);

    /// <summary>Clear all cached query plans.</summary>
    [LibraryImport(LibName)]
    internal static partial int grafeo_clear_plan_cache(nint db);

    // =========================================================================
    // Backup / Restore
    // =========================================================================

    /// <summary>Create a full backup at the given path.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int grafeo_backup_full(nint db, string path);

    /// <summary>Create an incremental backup at the given path.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int grafeo_backup_incremental(nint db, string path);

    /// <summary>Restore database to a specific epoch from a backup directory.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int grafeo_restore_to_epoch(string backupDir, ulong epoch, string outputPath);

    // =========================================================================
    // Maintenance
    // =========================================================================

    /// <summary>Compact the database store.</summary>
    [LibraryImport(LibName)]
    internal static partial int grafeo_compact(nint db);

    // =========================================================================
    // Projections
    // =========================================================================

    /// <summary>Create a named graph projection from label/type filters.
    /// Returns 1 if created, 0 if the name is taken, -1 on error.
    /// All string pointers are manually marshalled to avoid source-gen issues
    /// with mixed string + pointer parameters.</summary>
    [LibraryImport(LibName)]
    internal static unsafe partial int grafeo_create_projection(
        nint db, nint name,
        nint nodeLabels, nuint numLabels,
        nint edgeTypes, nuint numTypes);

    /// <summary>Drop a named graph projection. Returns 1 if it existed, 0 if not, -1 on error.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int grafeo_drop_projection(nint db, string name);

    /// <summary>List all projections as JSON. Caller must free with grafeo_free_string.</summary>
    [LibraryImport(LibName)]
    internal static partial nint grafeo_list_projections(nint db);

    // =========================================================================
    // Change Data Capture
    // =========================================================================

    /// <summary>Enable or disable CDC tracking.</summary>
    [LibraryImport(LibName)]
    internal static partial void grafeo_set_cdc_enabled(nint db, [MarshalAs(UnmanagedType.U1)] bool enabled);

    /// <summary>Check if CDC is enabled.</summary>
    [LibraryImport(LibName)]
    [return: MarshalAs(UnmanagedType.U1)]
    internal static partial bool grafeo_is_cdc_enabled(nint db);

    // =========================================================================
    // Transactions
    // =========================================================================

    /// <summary>Begin a new transaction. Returns null on error.</summary>
    [LibraryImport(LibName)]
    internal static partial nint grafeo_begin_transaction(nint db);

    /// <summary>Begin a transaction with a specific isolation level (0=ReadCommitted, 1=Snapshot, 2=Serializable).</summary>
    [LibraryImport(LibName)]
    internal static partial nint grafeo_begin_transaction_with_isolation(nint db, int isolationLevel);

    /// <summary>Execute a query within a transaction.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint grafeo_transaction_execute(nint tx, string query);

    /// <summary>Execute a query with parameters within a transaction.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint grafeo_transaction_execute_with_params(nint tx, string query, string paramsJson);

    /// <summary>Execute a query in any language within a transaction.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint grafeo_transaction_execute_language(nint tx, string language, string query, string? paramsJson);

    /// <summary>Commit a transaction.</summary>
    [LibraryImport(LibName)]
    internal static partial int grafeo_commit(nint tx);

    /// <summary>Rollback a transaction.</summary>
    [LibraryImport(LibName)]
    internal static partial int grafeo_rollback(nint tx);

    /// <summary>Free a transaction handle.</summary>
    [LibraryImport(LibName)]
    internal static partial void grafeo_free_transaction(nint tx);

    // =========================================================================
    // Error Handling
    // =========================================================================

    /// <summary>Get the last error message. Valid until next FFI call on this thread. Do NOT free.</summary>
    [LibraryImport(LibName)]
    internal static partial nint grafeo_last_error();

    /// <summary>Clear the last error.</summary>
    [LibraryImport(LibName)]
    internal static partial void grafeo_clear_error();

    /// <summary>Free a string returned by the API (info, labels, etc.).</summary>
    [LibraryImport(LibName)]
    internal static partial void grafeo_free_string(nint str);
}
