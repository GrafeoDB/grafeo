// Primary database handle for the Grafeo graph database.

using System.Runtime.InteropServices;

using Grafeo.Native;

namespace Grafeo;

/// <summary>
/// Transaction isolation levels supported by Grafeo.
/// Values match the <c>GrafeoIsolationLevel</c> C enum.
/// </summary>
public enum IsolationLevel
{
    /// <summary>Read committed: each statement sees the latest committed data.</summary>
    ReadCommitted = 0,

    /// <summary>Snapshot isolation: the transaction sees a consistent snapshot taken at begin.</summary>
    Snapshot = 1,

    /// <summary>Serializable: full serializability, the strongest isolation guarantee.</summary>
    Serializable = 2,
}

/// <summary>
/// Primary handle to a Grafeo graph database.
/// Thread-safe: the underlying engine uses <c>Arc&lt;RwLock&gt;</c>.
/// Implements <see cref="IDisposable"/> and <see cref="IAsyncDisposable"/>
/// for deterministic cleanup via <c>using</c>/<c>await using</c>.
/// </summary>
public sealed class GrafeoDB : IGrafeoDB, IDisposable, IAsyncDisposable
{
    private readonly DatabaseHandle _handle;
    private volatile bool _disposed;

    private GrafeoDB(DatabaseHandle handle) => _handle = handle;

    // =========================================================================
    // Lifecycle
    // =========================================================================

    /// <summary>Create a new in-memory database.</summary>
    public static GrafeoDB Memory()
    {
        var ptr = NativeMethods.grafeo_open_memory();
        if (ptr == nint.Zero)
            throw GrafeoException.FromLastError();

        var handle = new DatabaseHandle();
        Marshal.InitHandle(handle, ptr);
        return new GrafeoDB(handle);
    }

    /// <summary>Open or create a persistent database at <paramref name="path"/>.</summary>
    public static GrafeoDB Open(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);

        var ptr = NativeMethods.grafeo_open(path);
        if (ptr == nint.Zero)
            throw GrafeoException.FromLastError();

        var handle = new DatabaseHandle();
        Marshal.InitHandle(handle, ptr);
        return new GrafeoDB(handle);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _handle.Dispose();
    }

    /// <inheritdoc/>
    public ValueTask DisposeAsync()
    {
        Dispose();
        return ValueTask.CompletedTask;
    }

    // =========================================================================
    // Query Execution
    // =========================================================================

    /// <summary>Execute a GQL query synchronously.</summary>
    public QueryResult Execute(string query)
    {
        ThrowIfDisposed();
        var resultPtr = NativeMethods.grafeo_execute(Handle, query);
        if (resultPtr == nint.Zero)
            throw GrafeoException.FromLastError(GrafeoStatus.Query);
        return BuildResult(resultPtr);
    }

    /// <summary>
    /// Open a lazy cursor over a read-only GQL query. Memory is bounded to
    /// one chunk at a time regardless of total result size.
    /// Rejects mutations, EXPLAIN / PROFILE, session/schema commands, and
    /// queries that compile to push-based pipelines (ORDER BY, aggregate,
    /// DISTINCT). Use <see cref="Execute(string)"/> for those.
    /// Experimental (0.5.40+): signature may change before Beta.
    /// </summary>
    public ResultStream ExecuteStream(string query)
    {
        ThrowIfDisposed();
        var streamPtr = NativeMethods.grafeo_stream_open(Handle, query);
        if (streamPtr == nint.Zero)
            throw GrafeoException.FromLastError(GrafeoStatus.Query);

        // Read columns once; the JSON string is heap-allocated and must be freed.
        // Any failure past this point must free streamPtr before propagating.
        try
        {
            var colsPtr = NativeMethods.grafeo_stream_columns_json(streamPtr);
            if (colsPtr == nint.Zero)
            {
                throw GrafeoException.FromLastError(GrafeoStatus.Query);
            }
            IReadOnlyList<string> columns;
            try
            {
                var json = Marshal.PtrToStringUTF8(colsPtr) ?? "[]";
                columns = System.Text.Json.JsonSerializer.Deserialize<List<string>>(json)
                    ?? new List<string>();
            }
            finally
            {
                NativeMethods.grafeo_free_string(colsPtr);
            }
            return new ResultStream(streamPtr, columns);
        }
        catch
        {
            NativeMethods.grafeo_stream_free(streamPtr);
            throw;
        }
    }

    /// <summary>Execute a GQL query on the thread pool.</summary>
    public Task<QueryResult> ExecuteAsync(string query, CancellationToken ct = default)
    {
        ThrowIfDisposed();
        var h = Handle;
        return Task.Run(() =>
        {
            ct.ThrowIfCancellationRequested();
            var resultPtr = NativeMethods.grafeo_execute(h, query);
            if (resultPtr == nint.Zero)
                throw GrafeoException.FromLastError(GrafeoStatus.Query);
            return BuildResult(resultPtr);
        }, ct);
    }

    /// <summary>Execute a GQL query with parameters.</summary>
    public QueryResult ExecuteWithParams(string query, Dictionary<string, object?> parameters)
    {
        ThrowIfDisposed();
        var paramsJson = ValueConverter.EncodeParams(parameters);
        var resultPtr = NativeMethods.grafeo_execute_with_params(Handle, query, paramsJson);
        if (resultPtr == nint.Zero)
            throw GrafeoException.FromLastError(GrafeoStatus.Query);
        return BuildResult(resultPtr);
    }

    /// <summary>Execute a GQL query with parameters on the thread pool.</summary>
    public Task<QueryResult> ExecuteWithParamsAsync(
        string query,
        Dictionary<string, object?> parameters,
        CancellationToken ct = default)
    {
        ThrowIfDisposed();
        var paramsJson = ValueConverter.EncodeParams(parameters);
        var h = Handle;
        return Task.Run(() =>
        {
            ct.ThrowIfCancellationRequested();
            var resultPtr = NativeMethods.grafeo_execute_with_params(h, query, paramsJson);
            if (resultPtr == nint.Zero)
                throw GrafeoException.FromLastError(GrafeoStatus.Query);
            return BuildResult(resultPtr);
        }, ct);
    }

    /// <summary>Execute a Cypher query.</summary>
    public QueryResult ExecuteCypher(string query)
    {
        ThrowIfDisposed();
        var resultPtr = NativeMethods.grafeo_execute_cypher(Handle, query);
        if (resultPtr == nint.Zero)
            throw GrafeoException.FromLastError(GrafeoStatus.Query);
        return BuildResult(resultPtr);
    }

    /// <summary>Execute a Cypher query on the thread pool.</summary>
    public Task<QueryResult> ExecuteCypherAsync(string query, CancellationToken ct = default)
        => ExecuteLanguageAsync(NativeMethods.grafeo_execute_cypher, query, ct);

    /// <summary>Execute a SPARQL query.</summary>
    public QueryResult ExecuteSparql(string query)
    {
        ThrowIfDisposed();
        var resultPtr = NativeMethods.grafeo_execute_sparql(Handle, query);
        if (resultPtr == nint.Zero)
            throw GrafeoException.FromLastError(GrafeoStatus.Query);
        return BuildResult(resultPtr);
    }

    /// <summary>Execute a SPARQL query on the thread pool.</summary>
    public Task<QueryResult> ExecuteSparqlAsync(string query, CancellationToken ct = default)
        => ExecuteLanguageAsync(NativeMethods.grafeo_execute_sparql, query, ct);

    /// <summary>Execute a Gremlin query.</summary>
    public QueryResult ExecuteGremlin(string query)
    {
        ThrowIfDisposed();
        var resultPtr = NativeMethods.grafeo_execute_gremlin(Handle, query);
        if (resultPtr == nint.Zero)
            throw GrafeoException.FromLastError(GrafeoStatus.Query);
        return BuildResult(resultPtr);
    }

    /// <summary>Execute a Gremlin query on the thread pool.</summary>
    public Task<QueryResult> ExecuteGremlinAsync(string query, CancellationToken ct = default)
        => ExecuteLanguageAsync(NativeMethods.grafeo_execute_gremlin, query, ct);

    /// <summary>Execute a GraphQL query.</summary>
    public QueryResult ExecuteGraphql(string query)
    {
        ThrowIfDisposed();
        var resultPtr = NativeMethods.grafeo_execute_graphql(Handle, query);
        if (resultPtr == nint.Zero)
            throw GrafeoException.FromLastError(GrafeoStatus.Query);
        return BuildResult(resultPtr);
    }

    /// <summary>Execute a GraphQL query on the thread pool.</summary>
    public Task<QueryResult> ExecuteGraphqlAsync(string query, CancellationToken ct = default)
        => ExecuteLanguageAsync(NativeMethods.grafeo_execute_graphql, query, ct);

    /// <summary>Execute a SQL/PGQ query.</summary>
    public QueryResult ExecuteSql(string query)
    {
        ThrowIfDisposed();
        var resultPtr = NativeMethods.grafeo_execute_sql(Handle, query);
        if (resultPtr == nint.Zero)
            throw GrafeoException.FromLastError(GrafeoStatus.Query);
        return BuildResult(resultPtr);
    }

    /// <summary>Execute a SQL/PGQ query on the thread pool.</summary>
    public Task<QueryResult> ExecuteSqlAsync(string query, CancellationToken ct = default)
        => ExecuteLanguageAsync(NativeMethods.grafeo_execute_sql, query, ct);

    /// <summary>
    /// Execute a query in any supported language, optionally with parameters.
    /// </summary>
    /// <param name="language">Query language: "gql", "cypher", "gremlin", "graphql", "sparql", "sql", etc.</param>
    /// <param name="query">The query string.</param>
    /// <param name="parameters">Optional typed parameters to bind.</param>
    public QueryResult ExecuteLanguage(
        string language, string query, Dictionary<string, object?>? parameters = null)
    {
        ThrowIfDisposed();
        string? paramsJson = parameters is not null ? ValueConverter.EncodeParams(parameters) : null;
        var resultPtr = NativeMethods.grafeo_execute_language(Handle, language, query, paramsJson);
        if (resultPtr == nint.Zero)
            throw GrafeoException.FromLastError(GrafeoStatus.Query);
        return BuildResult(resultPtr);
    }

    // =========================================================================
    // Transactions
    // =========================================================================

    /// <summary>Begin a new ACID transaction with the default isolation level.</summary>
    public Transaction BeginTransaction()
    {
        ThrowIfDisposed();
        var txPtr = NativeMethods.grafeo_begin_transaction(Handle);
        if (txPtr == nint.Zero)
            throw GrafeoException.FromLastError(GrafeoStatus.Transaction);
        return new Transaction(txPtr);
    }

    /// <summary>Begin a transaction with a specific isolation level.</summary>
    /// <param name="isolationLevel">
    /// Accepted values (case-insensitive): "read_committed" / "ReadCommitted",
    /// "snapshot" / "Snapshot" / "snapshot_isolation" / "SnapshotIsolation",
    /// "serializable" / "Serializable".
    /// </param>
    public Transaction BeginTransaction(string isolationLevel)
    {
        ThrowIfDisposed();
        var level = ParseIsolationLevel(isolationLevel);
        var txPtr = NativeMethods.grafeo_begin_transaction_with_isolation(Handle, level);
        if (txPtr == nint.Zero)
            throw GrafeoException.FromLastError(GrafeoStatus.Transaction);
        return new Transaction(txPtr);
    }

    /// <summary>Begin a transaction with a specific isolation level.</summary>
    public Transaction BeginTransaction(IsolationLevel isolationLevel)
    {
        ThrowIfDisposed();
        var txPtr = NativeMethods.grafeo_begin_transaction_with_isolation(Handle, (int)isolationLevel);
        if (txPtr == nint.Zero)
            throw GrafeoException.FromLastError(GrafeoStatus.Transaction);
        return new Transaction(txPtr);
    }

    // Explicit interface implementations for ITransaction return type
    ITransaction IGrafeoDB.BeginTransaction() => BeginTransaction();
    ITransaction IGrafeoDB.BeginTransaction(IsolationLevel level) => BeginTransaction(level);

    // =========================================================================
    // Node CRUD
    // =========================================================================

    /// <summary>Create a node with labels and optional properties. Returns the new node ID.</summary>
    public long CreateNode(IEnumerable<string> labels, Dictionary<string, object?>? properties = null)
    {
        ThrowIfDisposed();
        var labelsJson = System.Text.Json.JsonSerializer.Serialize(labels);
        var propsJson = properties is not null ? ValueConverter.EncodeParams(properties) : null;
        var id = NativeMethods.grafeo_create_node(Handle, labelsJson, propsJson);
        if (id == ulong.MaxValue)
            throw GrafeoException.FromLastError();
        return (long)id;
    }

    /// <summary>Get a node by ID. Returns null if not found.</summary>
    public Node? GetNode(long id)
    {
        ThrowIfDisposed();
        var status = NativeMethods.grafeo_get_node(Handle, (ulong)id, out var nodePtr);
        if (status != (int)GrafeoStatus.Ok)
            return null;
        try
        {
            return ReadNode(nodePtr);
        }
        finally
        {
            NativeMethods.grafeo_free_node(nodePtr);
        }
    }

    /// <summary>Delete a node by ID. Returns true if deleted.</summary>
    public bool DeleteNode(long id)
    {
        ThrowIfDisposed();
        return NativeMethods.grafeo_delete_node(Handle, (ulong)id) == 1;
    }

    /// <summary>Set a property on a node.</summary>
    public void SetNodeProperty(long id, string key, object? value)
    {
        ThrowIfDisposed();
        var valueJson = ValueConverter.EncodeValue(value);
        GrafeoException.ThrowIfFailed(
            NativeMethods.grafeo_set_node_property(Handle, (ulong)id, key, valueJson));
    }

    /// <summary>Remove a property from a node. Returns true if removed.</summary>
    public bool RemoveNodeProperty(long id, string key)
    {
        ThrowIfDisposed();
        return NativeMethods.grafeo_remove_node_property(Handle, (ulong)id, key) == 1;
    }

    /// <summary>Add a label to a node. Returns true if added.</summary>
    public bool AddNodeLabel(long id, string label)
    {
        ThrowIfDisposed();
        return NativeMethods.grafeo_add_node_label(Handle, (ulong)id, label) == 1;
    }

    /// <summary>Remove a label from a node. Returns true if removed.</summary>
    public bool RemoveNodeLabel(long id, string label)
    {
        ThrowIfDisposed();
        return NativeMethods.grafeo_remove_node_label(Handle, (ulong)id, label) == 1;
    }

    // =========================================================================
    // Edge CRUD
    // =========================================================================

    /// <summary>Create an edge between two nodes. Returns the new edge ID.</summary>
    public long CreateEdge(
        long sourceId,
        long targetId,
        string edgeType,
        Dictionary<string, object?>? properties = null)
    {
        ThrowIfDisposed();
        var propsJson = properties is not null ? ValueConverter.EncodeParams(properties) : null;
        var id = NativeMethods.grafeo_create_edge(
            Handle, (ulong)sourceId, (ulong)targetId, edgeType, propsJson);
        if (id == ulong.MaxValue)
            throw GrafeoException.FromLastError();
        return (long)id;
    }

    /// <summary>Get an edge by ID. Returns null if not found.</summary>
    public Edge? GetEdge(long id)
    {
        ThrowIfDisposed();
        var status = NativeMethods.grafeo_get_edge(Handle, (ulong)id, out var edgePtr);
        if (status != (int)GrafeoStatus.Ok)
            return null;
        try
        {
            return ReadEdge(edgePtr);
        }
        finally
        {
            NativeMethods.grafeo_free_edge(edgePtr);
        }
    }

    /// <summary>Delete an edge by ID. Returns true if deleted.</summary>
    public bool DeleteEdge(long id)
    {
        ThrowIfDisposed();
        return NativeMethods.grafeo_delete_edge(Handle, (ulong)id) == 1;
    }

    /// <summary>Set a property on an edge.</summary>
    public void SetEdgeProperty(long id, string key, object? value)
    {
        ThrowIfDisposed();
        var valueJson = ValueConverter.EncodeValue(value);
        GrafeoException.ThrowIfFailed(
            NativeMethods.grafeo_set_edge_property(Handle, (ulong)id, key, valueJson));
    }

    /// <summary>Remove a property from an edge. Returns true if removed.</summary>
    public bool RemoveEdgeProperty(long id, string key)
    {
        ThrowIfDisposed();
        return NativeMethods.grafeo_remove_edge_property(Handle, (ulong)id, key) == 1;
    }

    // =========================================================================
    // Admin
    // =========================================================================

    /// <summary>Number of nodes in the database.</summary>
    public long NodeCount
    {
        get
        {
            ThrowIfDisposed();
            return (long)NativeMethods.grafeo_node_count(Handle);
        }
    }

    /// <summary>Number of edges in the database.</summary>
    public long EdgeCount
    {
        get
        {
            ThrowIfDisposed();
            return (long)NativeMethods.grafeo_edge_count(Handle);
        }
    }

    /// <summary>Get database info as a dictionary (version, node count, edge count, etc.).</summary>
    public IReadOnlyDictionary<string, object?> Info()
    {
        ThrowIfDisposed();
        var ptr = NativeMethods.grafeo_info(Handle);
        if (ptr == nint.Zero)
            throw GrafeoException.FromLastError();
        try
        {
            var json = Marshal.PtrToStringUTF8(ptr)!;
            return ValueConverter.ParseObject(json);
        }
        finally
        {
            NativeMethods.grafeo_free_string(ptr);
        }
    }

    /// <summary>Get the Grafeo library version string.</summary>
    public static string Version
    {
        get
        {
            var ptr = NativeMethods.grafeo_version();
            return Marshal.PtrToStringUTF8(ptr) ?? "unknown";
        }
    }

    /// <summary>Save the database to a file path.</summary>
    public void Save(string path)
    {
        ThrowIfDisposed();
        GrafeoException.ThrowIfFailed(NativeMethods.grafeo_save(Handle, path));
    }

    /// <summary>Clear all cached query plans, forcing re-parsing on next execution.</summary>
    public void ClearPlanCache()
    {
        ThrowIfDisposed();
        GrafeoException.ThrowIfFailed(NativeMethods.grafeo_clear_plan_cache(Handle));
    }

    // =========================================================================
    // Schema Context
    // =========================================================================

    /// <summary>Set the current schema for subsequent queries.</summary>
    public void SetSchema(string name)
    {
        ThrowIfDisposed();
        GrafeoException.ThrowIfFailed(NativeMethods.grafeo_set_schema(Handle, name));
    }

    /// <summary>Clear the current schema context.</summary>
    public void ResetSchema()
    {
        ThrowIfDisposed();
        GrafeoException.ThrowIfFailed(NativeMethods.grafeo_reset_schema(Handle));
    }

    /// <summary>Get the current schema name, or <c>null</c> if none is set.</summary>
    public string? CurrentSchema()
    {
        ThrowIfDisposed();
        var ptr = NativeMethods.grafeo_current_schema(Handle);
        return ptr == nint.Zero ? null : Marshal.PtrToStringUTF8(ptr);
    }

    // =========================================================================
    // Backup / Restore
    // =========================================================================

    /// <summary>Create a full backup at the given directory path.</summary>
    public void BackupFull(string path)
    {
        ThrowIfDisposed();
        GrafeoException.ThrowIfFailed(NativeMethods.grafeo_backup_full(Handle, path));
    }

    /// <summary>Create an incremental backup at the given directory path.</summary>
    public void BackupIncremental(string path)
    {
        ThrowIfDisposed();
        GrafeoException.ThrowIfFailed(NativeMethods.grafeo_backup_incremental(Handle, path));
    }

    /// <summary>Restore database to a specific epoch from a backup directory.</summary>
    /// <param name="backupDir">Directory containing backup files.</param>
    /// <param name="epoch">Target epoch to restore to.</param>
    /// <param name="outputPath">Path for the restored database.</param>
    public static void RestoreToEpoch(string backupDir, ulong epoch, string outputPath)
    {
        GrafeoException.ThrowIfFailed(
            NativeMethods.grafeo_restore_to_epoch(backupDir, epoch, outputPath));
    }

    // =========================================================================
    // Maintenance
    // =========================================================================

    /// <summary>Compact the database, reclaiming storage from deleted data.</summary>
    public void Compact()
    {
        ThrowIfDisposed();
        GrafeoException.ThrowIfFailed(NativeMethods.grafeo_compact(Handle));
    }

    // =========================================================================
    // Projections
    // =========================================================================

    /// <summary>Create a named graph projection over the graph selected now (the default graph
    /// when none is selected) that includes the specified node labels and edge types.</summary>
    /// <param name="name">Unique name for the projection.</param>
    /// <param name="nodeLabels">Node labels to include (null or empty for all).</param>
    /// <param name="edgeTypes">Edge types to include (null or empty for all).</param>
    /// <returns><c>true</c> if the projection was created, <c>false</c> if the name exists.</returns>
    /// <exception cref="GrafeoException">The selected graph no longer exists.</exception>
    public bool CreateProjection(string name, IEnumerable<string>? nodeLabels = null, IEnumerable<string>? edgeTypes = null)
    {
        ThrowIfDisposed();
        var labels = nodeLabels?.ToArray() ?? [];
        var types = edgeTypes?.ToArray() ?? [];

        // Marshal all strings manually (name + string arrays) to avoid
        // source-gen issues with mixed auto-marshalled + pointer parameters.
        var namePtr = Marshal.StringToCoTaskMemUTF8(name);
        var labelPtrs = labels.Select(Marshal.StringToCoTaskMemUTF8).ToArray();
        var typePtrs = types.Select(Marshal.StringToCoTaskMemUTF8).ToArray();

        try
        {
            unsafe
            {
                fixed (nint* lp = labelPtrs.Length > 0 ? labelPtrs : null)
                fixed (nint* tp = typePtrs.Length > 0 ? typePtrs : null)
                {
                    var result = NativeMethods.grafeo_create_projection(
                        Handle, namePtr,
                        (nint)lp, (nuint)labels.Length,
                        (nint)tp, (nuint)types.Length);
                    if (result < 0)
                        throw GrafeoException.FromLastError();
                    return result == 1;
                }
            }
        }
        finally
        {
            Marshal.FreeCoTaskMem(namePtr);
            foreach (var p in labelPtrs) Marshal.FreeCoTaskMem(p);
            foreach (var p in typePtrs) Marshal.FreeCoTaskMem(p);
        }
    }

    /// <summary>Drop a named graph projection.</summary>
    /// <returns><c>true</c> if the projection existed, <c>false</c> otherwise.</returns>
    public bool DropProjection(string name)
    {
        ThrowIfDisposed();
        var result = NativeMethods.grafeo_drop_projection(Handle, name);
        if (result < 0)
            throw GrafeoException.FromLastError();
        return result == 1;
    }

    /// <summary>List all named graph projections as a JSON array.</summary>
    public string ListProjections()
    {
        ThrowIfDisposed();
        var ptr = NativeMethods.grafeo_list_projections(Handle);
        if (ptr == nint.Zero) return "[]";
        try
        {
            return Marshal.PtrToStringUTF8(ptr) ?? "[]";
        }
        finally
        {
            NativeMethods.grafeo_free_string(ptr);
        }
    }

    // =========================================================================
    // Change Data Capture
    // =========================================================================

    /// <summary>Gets or sets whether Change Data Capture is enabled.</summary>
    public bool CdcEnabled
    {
        get
        {
            ThrowIfDisposed();
            return NativeMethods.grafeo_is_cdc_enabled(Handle);
        }
        set
        {
            ThrowIfDisposed();
            NativeMethods.grafeo_set_cdc_enabled(Handle, value);
        }
    }

    // =========================================================================
    // Vector Search
    // =========================================================================

    /// <summary>Drop a vector index on the given label and property.</summary>
    /// <returns><c>true</c> if the index was dropped, <c>false</c> if no such index existed.</returns>
    public bool DropVectorIndex(string label, string property)
    {
        ThrowIfDisposed();
        var result = NativeMethods.grafeo_drop_vector_index(Handle, label, property);
        if (result < 0)
            throw GrafeoException.FromLastError();
        return result == 1;
    }

    /// <summary>Rebuild a vector index on the given label and property.</summary>
    public void RebuildVectorIndex(string label, string property)
    {
        ThrowIfDisposed();
        GrafeoException.ThrowIfFailed(
            NativeMethods.grafeo_rebuild_vector_index(Handle, label, property));
    }

    /// <summary>Perform a vector similarity search.</summary>
    /// <returns>List of (nodeId, distance) results ordered by distance.</returns>
    public IReadOnlyList<VectorResult> VectorSearch(
        string label, string property, float[] query, int k, uint ef = 0)
    {
        ThrowIfDisposed();
        unsafe
        {
            fixed (float* queryPtr = query)
            {
                var status = NativeMethods.grafeo_vector_search(
                    Handle, label, property,
                    queryPtr, (nuint)query.Length, (nuint)k, ef,
                    out var idsPtr, out var distsPtr, out var count);

                GrafeoException.ThrowIfFailed(status);
                return ReadVectorResults(idsPtr, distsPtr, count);
            }
        }
    }

    /// <summary>Perform a Maximal Marginal Relevance (MMR) search.</summary>
    public IReadOnlyList<VectorResult> MmrSearch(
        string label, string property, float[] query,
        int k, int fetchK, float lambda, int ef = 0)
    {
        ThrowIfDisposed();
        unsafe
        {
            fixed (float* queryPtr = query)
            {
                var status = NativeMethods.grafeo_mmr_search(
                    Handle, label, property,
                    queryPtr, (nuint)query.Length, (nuint)k,
                    fetchK, lambda, ef,
                    out var idsPtr, out var distsPtr, out var count);

                GrafeoException.ThrowIfFailed(status);
                return ReadVectorResults(idsPtr, distsPtr, count);
            }
        }
    }

    // =========================================================================
    // Internals
    // =========================================================================

    private Task<QueryResult> ExecuteLanguageAsync(
        Func<nint, string, nint> nativeMethod,
        string query,
        CancellationToken ct)
    {
        ThrowIfDisposed();
        var h = Handle;
        return Task.Run(() =>
        {
            ct.ThrowIfCancellationRequested();
            var resultPtr = nativeMethod(h, query);
            if (resultPtr == nint.Zero)
                throw GrafeoException.FromLastError(GrafeoStatus.Query);
            return BuildResult(resultPtr);
        }, ct);
    }

    /// <summary>Get the raw handle, checking for disposal first.</summary>
    private nint Handle
    {
        get
        {
            ThrowIfDisposed();
            return _handle.DangerousGetHandle();
        }
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    /// <summary>Parse a string isolation level name to the integer value expected by the C API.</summary>
    private static int ParseIsolationLevel(string isolationLevel) =>
        isolationLevel.ToLowerInvariant() switch
        {
            "read_committed" or "readcommitted" => (int)IsolationLevel.ReadCommitted,
            "snapshot" or "snapshot_isolation" or "snapshotisolation" => (int)IsolationLevel.Snapshot,
            "serializable" => (int)IsolationLevel.Serializable,
            _ => throw new ArgumentException(
                $"Unknown isolation level: '{isolationLevel}'. " +
                "Use \"read_committed\", \"snapshot\", or \"serializable\".",
                nameof(isolationLevel)),
        };

    /// <summary>Parse a native result pointer into a QueryResult, then free the native result.</summary>
    private static QueryResult BuildResult(nint resultPtr)
    {
        try
        {
            var jsonPtr = NativeMethods.grafeo_result_json(resultPtr);
            var json = Marshal.PtrToStringUTF8(jsonPtr) ?? "[]";
            var executionTimeMs = NativeMethods.grafeo_result_execution_time_ms(resultPtr);
            var rowsScanned = (long)NativeMethods.grafeo_result_rows_scanned(resultPtr);

            var rows = ValueConverter.ParseRows(json);
            var columns = ValueConverter.ExtractColumns(rows);
            var (nodes, edges) = ValueConverter.ExtractEntities(rows);

            return new QueryResult(columns, rows, nodes, edges, executionTimeMs, rowsScanned);
        }
        finally
        {
            NativeMethods.grafeo_free_result(resultPtr);
        }
    }

    /// <summary>Read a node from a native GrafeoNode pointer.</summary>
    private static Node ReadNode(nint nodePtr)
    {
        var id = (long)NativeMethods.grafeo_node_id(nodePtr);
        var labelsJsonPtr = NativeMethods.grafeo_node_labels_json(nodePtr);
        var propsJsonPtr = NativeMethods.grafeo_node_properties_json(nodePtr);

        var labelsJson = Marshal.PtrToStringUTF8(labelsJsonPtr) ?? "[]";
        var propsJson = Marshal.PtrToStringUTF8(propsJsonPtr) ?? "{}";

        var labels = ValueConverter.ParseStringArray(labelsJson);
        var properties = ValueConverter.ParseObject(propsJson);
        return new Node(id, labels, properties);
    }

    /// <summary>Read an edge from a native GrafeoEdge pointer.</summary>
    private static Edge ReadEdge(nint edgePtr)
    {
        var id = (long)NativeMethods.grafeo_edge_id(edgePtr);
        var sourceId = (long)NativeMethods.grafeo_edge_source_id(edgePtr);
        var targetId = (long)NativeMethods.grafeo_edge_target_id(edgePtr);
        var typePtr = NativeMethods.grafeo_edge_type(edgePtr);
        var propsPtr = NativeMethods.grafeo_edge_properties_json(edgePtr);

        var edgeType = Marshal.PtrToStringUTF8(typePtr) ?? "";
        var propsJson = Marshal.PtrToStringUTF8(propsPtr) ?? "{}";
        var properties = ValueConverter.ParseObject(propsJson);

        return new Edge(id, edgeType, sourceId, targetId, properties);
    }

    /// <summary>Read vector search results from native pointers, then free them.</summary>
    private static IReadOnlyList<VectorResult> ReadVectorResults(
        nint idsPtr, nint distsPtr, nuint count)
    {
        if (count == 0)
            return Array.Empty<VectorResult>();

        var results = new VectorResult[(int)count];
        unsafe
        {
            var ids = (ulong*)idsPtr;
            var dists = (float*)distsPtr;
            for (var i = 0; i < (int)count; i++)
            {
                results[i] = new VectorResult((long)ids[i], dists[i]);
            }
        }
        NativeMethods.grafeo_free_vector_results(idsPtr, distsPtr, count);
        return results;
    }
}
