// A handle on one named graph of a database.

using System.Runtime.InteropServices;

using Grafeo.Native;

namespace Grafeo;

/// <summary>
/// Works in one named graph of a database without switching the database's current graph,
/// from <see cref="GrafeoDB.Graph"/>. Each call looks the graph up again: once it is dropped,
/// every call throws. Dispose frees the handle.
/// </summary>
public sealed class GrafeoGraph : IDisposable
{
    private readonly GraphHandle _handle;
    private volatile bool _disposed;

    internal GrafeoGraph(nint ptr)
    {
        _handle = new GraphHandle();
        Marshal.InitHandle(_handle, ptr);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _handle.Dispose();
    }

    private nint Handle
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _handle.DangerousGetHandle();
        }
    }

    // =========================================================================
    // Query Execution
    // =========================================================================

    /// <summary>Execute a GQL query in the graph.</summary>
    public QueryResult Execute(string query) =>
        Result(NativeMethods.grafeo_graph_execute(Handle, query));

    /// <summary>Execute a GQL query with parameters in the graph.</summary>
    public QueryResult ExecuteWithParams(string query, Dictionary<string, object?> parameters) =>
        Result(NativeMethods.grafeo_graph_execute_with_params(
            Handle, query, ValueConverter.EncodeParams(parameters)));

    /// <summary>Execute a query in any supported language ("gql", "cypher", ...) in the graph.</summary>
    public QueryResult ExecuteLanguage(
        string language, string query, Dictionary<string, object?>? parameters = null) =>
        Result(NativeMethods.grafeo_graph_execute_language(
            Handle, language, query, parameters is not null ? ValueConverter.EncodeParams(parameters) : null));

    private static QueryResult Result(nint resultPtr)
    {
        if (resultPtr == nint.Zero)
            throw GrafeoException.FromLastError(GrafeoStatus.Query);
        return GrafeoDB.BuildResult(resultPtr);
    }

    // =========================================================================
    // Nodes and edges
    // =========================================================================

    /// <summary>Create a node with labels and optional properties. Returns the new node ID.</summary>
    public long CreateNode(IEnumerable<string> labels, Dictionary<string, object?>? properties = null)
    {
        var propsJson = properties is not null ? ValueConverter.EncodeParams(properties) : null;
        return Created(NativeMethods.grafeo_graph_create_node(Handle, BatchCalls.Labels(labels), propsJson));
    }

    /// <summary>Create an edge between two nodes. Returns the new edge ID.</summary>
    public long CreateEdge(
        long sourceId, long targetId, string edgeType, Dictionary<string, object?>? properties = null)
    {
        var propsJson = properties is not null ? ValueConverter.EncodeParams(properties) : null;
        return Created(NativeMethods.grafeo_graph_create_edge(
            Handle, (ulong)sourceId, (ulong)targetId, edgeType, propsJson));
    }

    /// <summary>Get a node by ID, or null when it does not exist.</summary>
    /// <exception cref="GrafeoException">The graph was dropped, or the read failed.</exception>
    public Node? GetNode(long id)
    {
        GrafeoException.ThrowIfFailed(NativeMethods.grafeo_graph_get_node(Handle, (ulong)id, out var nodePtr));
        if (nodePtr == nint.Zero)
            return null;
        try
        {
            return GrafeoDB.ReadNode(nodePtr);
        }
        finally
        {
            NativeMethods.grafeo_free_node(nodePtr);
        }
    }

    /// <summary>Get an edge by ID, or null when it does not exist.</summary>
    /// <exception cref="GrafeoException">The graph was dropped, or the read failed.</exception>
    public Edge? GetEdge(long id)
    {
        GrafeoException.ThrowIfFailed(NativeMethods.grafeo_graph_get_edge(Handle, (ulong)id, out var edgePtr));
        if (edgePtr == nint.Zero)
            return null;
        try
        {
            return GrafeoDB.ReadEdge(edgePtr);
        }
        finally
        {
            NativeMethods.grafeo_free_edge(edgePtr);
        }
    }

    /// <summary><see cref="GrafeoDB.GetNodeEdges"/> in the graph.</summary>
    public IReadOnlyList<NodeEdge> GetNodeEdges(
        long nodeId, Direction direction = Direction.Out, IEnumerable<string>? edgeTypes = null)
    {
        var typesJson = BatchCalls.EdgeTypes(edgeTypes);
        return BatchCalls.NodeEdges((out nint neighbors, out nint edges, out nuint count) =>
            NativeMethods.grafeo_graph_get_node_edges(
                Handle, (ulong)nodeId, (int)direction, typesJson, out neighbors, out edges, out count));
    }

    /// <summary>Set a property on a node.</summary>
    public void SetNodeProperty(long id, string key, object? value) =>
        GrafeoException.ThrowIfFailed(NativeMethods.grafeo_graph_set_node_property(
            Handle, (ulong)id, key, ValueConverter.EncodeValue(value)));

    /// <summary>Set a property on an edge.</summary>
    public void SetEdgeProperty(long id, string key, object? value) =>
        GrafeoException.ThrowIfFailed(NativeMethods.grafeo_graph_set_edge_property(
            Handle, (ulong)id, key, ValueConverter.EncodeValue(value)));

    /// <summary>Remove a property from a node. Returns true if it was there.</summary>
    public bool RemoveNodeProperty(long id, string key) =>
        Flag(NativeMethods.grafeo_graph_remove_node_property(Handle, (ulong)id, key));

    /// <summary>Remove a property from an edge. Returns true if it was there.</summary>
    public bool RemoveEdgeProperty(long id, string key) =>
        Flag(NativeMethods.grafeo_graph_remove_edge_property(Handle, (ulong)id, key));

    /// <summary>Add a label to a node. Returns false if it had it.</summary>
    public bool AddNodeLabel(long id, string label) =>
        Flag(NativeMethods.grafeo_graph_add_node_label(Handle, (ulong)id, label));

    /// <summary>Remove a label from a node. Returns true if it had it.</summary>
    public bool RemoveNodeLabel(long id, string label) =>
        Flag(NativeMethods.grafeo_graph_remove_node_label(Handle, (ulong)id, label));

    /// <summary>Delete a node (one with edges throws). Returns true if it existed.</summary>
    public bool DeleteNode(long id) =>
        Flag(NativeMethods.grafeo_graph_delete_node(Handle, (ulong)id));

    /// <summary>Delete a node with all its edges, as <c>DETACH DELETE</c> does. Returns true if it existed.</summary>
    public bool DetachDeleteNode(long id) =>
        Flag(NativeMethods.grafeo_graph_detach_delete_node(Handle, (ulong)id));

    /// <summary>Delete an edge. Returns true if it existed.</summary>
    public bool DeleteEdge(long id) =>
        Flag(NativeMethods.grafeo_graph_delete_edge(Handle, (ulong)id));

    // =========================================================================
    // Batch writes and upserts
    // =========================================================================

    /// <summary><see cref="GrafeoDB.BatchCreateNodesWithProps"/> in the graph.</summary>
    public IReadOnlyList<long> BatchCreateNodesWithProps(
        IEnumerable<string> labels, IEnumerable<IDictionary<string, object?>> rows)
    {
        var labelsJson = BatchCalls.Labels(labels);
        var rowsJson = BatchCalls.Rows(rows);
        return BatchCalls.Ids((out nint ids, out nuint count) =>
            NativeMethods.grafeo_graph_batch_create_nodes_with_props(Handle, labelsJson, rowsJson, out ids, out count));
    }

    /// <summary><see cref="GrafeoDB.BatchCreateEdges"/> in the graph.</summary>
    public IReadOnlyList<long> BatchCreateEdges(IEnumerable<BatchEdge> edges)
    {
        var edgesJson = BatchCalls.Edges(edges);
        return BatchCalls.Ids((out nint ids, out nuint count) =>
            NativeMethods.grafeo_graph_batch_create_edges(Handle, edgesJson, out ids, out count));
    }

    /// <summary><see cref="GrafeoDB.UpsertNodes"/> in the graph.</summary>
    public UpsertSummary UpsertNodes(
        IEnumerable<string> labels,
        IEnumerable<IDictionary<string, object?>> rows,
        UpsertNodesOptions? options = null)
    {
        var labelsJson = BatchCalls.Labels(labels);
        var rowsJson = BatchCalls.Rows(rows);
        var optionsJson = BatchCalls.Options(options);
        return BatchCalls.Summary(BatchCalls.Json((out nint json) =>
            NativeMethods.grafeo_graph_upsert_nodes(Handle, labelsJson, rowsJson, optionsJson, out json)));
    }

    /// <summary><see cref="GrafeoDB.UpsertEdges"/> in the graph.</summary>
    public UpsertSummary UpsertEdges(
        string edgeType,
        IEnumerable<IDictionary<string, object?>> rows,
        UpsertEdgesOptions? options = null)
    {
        var rowsJson = BatchCalls.Rows(rows);
        var optionsJson = BatchCalls.Options(options);
        return BatchCalls.Summary(BatchCalls.Json((out nint json) =>
            NativeMethods.grafeo_graph_upsert_edges(Handle, edgeType, rowsJson, optionsJson, out json)));
    }

    private static long Created(ulong id)
    {
        if (id == ulong.MaxValue)
            throw GrafeoException.FromLastError();
        return (long)id;
    }

    private static bool Flag(int result)
    {
        if (result < 0)
            throw GrafeoException.FromLastError();
        return result == 1;
    }
}
