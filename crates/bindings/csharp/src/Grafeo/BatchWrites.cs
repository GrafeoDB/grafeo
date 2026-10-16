// Batch writes, upserts and change history: the types they take and return, and the
// marshalling GrafeoDB and GrafeoGraph share.

using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Nodes;

using Grafeo.Native;

namespace Grafeo;

/// <summary>One edge of <see cref="GrafeoDB.BatchCreateEdges"/>.</summary>
/// <param name="SourceId">Source node ID.</param>
/// <param name="TargetId">Target node ID.</param>
/// <param name="Type">Edge type.</param>
/// <param name="Properties">Optional properties.</param>
public sealed record BatchEdge(
    long SourceId,
    long TargetId,
    string Type,
    IDictionary<string, object?>? Properties = null);

/// <summary>What an upsert did with its rows.</summary>
/// <param name="Created">Rows that created a node or edge.</param>
/// <param name="Updated">Rows that updated one.</param>
/// <param name="Skipped">Rows without their key, and edge rows whose endpoint matches no node or several.</param>
/// <param name="SkippedRows">Indexes of the skipped rows.</param>
public sealed record UpsertSummary(
    int Created,
    int Updated,
    int Skipped,
    IReadOnlyList<int> SkippedRows);

/// <summary>Options of <see cref="GrafeoDB.UpsertNodes"/>; null means the defaults.</summary>
public sealed record UpsertNodesOptions
{
    /// <summary>The property that identifies a node (default <c>id</c>).</summary>
    public string? Key { get; init; }

    /// <summary>Make a node's properties exactly the row's (default: the row's are merged in).</summary>
    public bool Replace { get; init; }
}

/// <summary>Options of <see cref="GrafeoDB.UpsertEdges"/>; null means the defaults.</summary>
public sealed record UpsertEdgesOptions
{
    /// <summary>The property that identifies an edge (default <c>id</c>).</summary>
    public string? Key { get; init; }

    /// <summary>The node property the endpoint fields hold (default <c>id</c>).</summary>
    public string? EndpointKey { get; init; }

    /// <summary>Labels an endpoint must have (default: any).</summary>
    public IReadOnlyList<string>? EndpointLabels { get; init; }

    /// <summary>The row field naming the source (default <c>src</c>).</summary>
    public string? SourceField { get; init; }

    /// <summary>The row field naming the target (default <c>dst</c>).</summary>
    public string? TargetField { get; init; }

    /// <summary>Make an edge's properties exactly the row's (default: the row's are merged in).</summary>
    public bool Replace { get; init; }
}

/// <summary>One recorded change of a node or edge (change data capture).</summary>
/// <param name="EntityId">The node or edge ID.</param>
/// <param name="EntityType"><c>node</c> or <c>edge</c>.</param>
/// <param name="Graph">The named graph, null for the default graph.</param>
/// <param name="Kind"><c>create</c>, <c>update</c> or <c>delete</c>.</param>
/// <param name="Epoch">The epoch of the commit.</param>
/// <param name="Timestamp">A hybrid logical clock value.</param>
/// <param name="Before">Properties before the change (null on a create).</param>
/// <param name="After">Properties after the change (null on a delete).</param>
/// <param name="Labels">A node's labels after the change.</param>
/// <param name="BeforeLabels">A node's labels before the change.</param>
/// <param name="EdgeType">An edge's type.</param>
/// <param name="SourceId">An edge's source node.</param>
/// <param name="TargetId">An edge's target node.</param>
/// <param name="DeletedWith">On an edge a detach delete removed, the node it was removed with.</param>
public sealed record ChangeEvent(
    long EntityId,
    string EntityType,
    string? Graph,
    string Kind,
    ulong Epoch,
    ulong Timestamp,
    IReadOnlyDictionary<string, object?>? Before,
    IReadOnlyDictionary<string, object?>? After,
    IReadOnlyList<string>? Labels,
    IReadOnlyList<string>? BeforeLabels,
    string? EdgeType,
    long? SourceId,
    long? TargetId,
    long? DeletedWith);

/// <summary>Which edges <see cref="GrafeoDB.GetNodeEdges"/> lists. Values match <c>GrafeoDirection</c>.</summary>
public enum Direction
{
    /// <summary>The edges that leave the node.</summary>
    Out = 0,

    /// <summary>The edges that reach the node.</summary>
    In = 1,

    /// <summary>Both, a self-loop once.</summary>
    Both = 2,
}

/// <summary>One edge of a node and the neighbor it reaches.</summary>
/// <param name="NeighborId">The node at the other end.</param>
/// <param name="EdgeId">The edge.</param>
public readonly record struct NodeEdge(long NeighborId, long EdgeId);

/// <summary>A native call that hands out neighbor and edge id arrays.</summary>
internal delegate int EdgesCall(out nint neighbors, out nint edges, out nuint count);

/// <summary>A native call that hands out an id array.</summary>
internal delegate int IdsCall(out nint ids, out nuint count);

/// <summary>A native call that hands out a JSON string.</summary>
internal delegate int JsonCall(out nint json);

/// <summary>Marshalling the batch, upsert and history calls share.</summary>
internal static class BatchCalls
{
    /// <summary>Runs <paramref name="call"/>, copies and frees the ids it hands out.</summary>
    internal static IReadOnlyList<long> Ids(IdsCall call)
    {
        GrafeoException.ThrowIfFailed(call(out var ids, out var count));
        if (count == 0)
            return Array.Empty<long>();
        try
        {
            var result = new long[(int)count];
            Marshal.Copy(ids, result, 0, (int)count);
            return result;
        }
        finally
        {
            NativeMethods.grafeo_free_node_ids(ids, count);
        }
    }

    /// <summary>Runs <paramref name="call"/>, pairs and frees the neighbor and edge ids it hands out.</summary>
    internal static IReadOnlyList<NodeEdge> NodeEdges(EdgesCall call)
    {
        GrafeoException.ThrowIfFailed(call(out var neighbors, out var edges, out var count));
        if (count == 0)
            return Array.Empty<NodeEdge>();
        try
        {
            var neighborIds = new long[(int)count];
            var edgeIds = new long[(int)count];
            Marshal.Copy(neighbors, neighborIds, 0, (int)count);
            Marshal.Copy(edges, edgeIds, 0, (int)count);
            var result = new NodeEdge[(int)count];
            for (var i = 0; i < result.Length; i++)
                result[i] = new NodeEdge(neighborIds[i], edgeIds[i]);
            return result;
        }
        finally
        {
            NativeMethods.grafeo_free_node_ids(neighbors, count);
            NativeMethods.grafeo_free_node_ids(edges, count);
        }
    }

    /// <summary>The edge types argument: null for every type, else a JSON array.</summary>
    internal static string? EdgeTypes(IEnumerable<string>? edgeTypes)
    {
        var types = edgeTypes?.ToArray();
        return types is { Length: > 0 } ? JsonSerializer.Serialize(types) : null;
    }

    /// <summary>Runs <paramref name="call"/> and takes the JSON string it hands out.</summary>
    internal static string Json(JsonCall call)
    {
        GrafeoException.ThrowIfFailed(call(out var json));
        return Take(json);
    }

    /// <summary>Copies and frees a string the library handed out; null throws its error.</summary>
    internal static string Take(nint ptr)
    {
        if (ptr == nint.Zero)
            throw GrafeoException.FromLastError();
        try
        {
            return Marshal.PtrToStringUTF8(ptr) ?? "";
        }
        finally
        {
            NativeMethods.grafeo_free_string(ptr);
        }
    }

    internal static string Labels(IEnumerable<string> labels) => JsonSerializer.Serialize(labels);

    internal static string Rows(IEnumerable<IDictionary<string, object?>> rows)
    {
        var array = new JsonArray();
        foreach (var row in rows)
            array.Add(ValueConverter.ToJsonNode(row));
        return array.ToJsonString();
    }

    internal static string Edges(IEnumerable<BatchEdge> edges)
    {
        var array = new JsonArray();
        foreach (var edge in edges)
        {
            var obj = new JsonObject
            {
                ["src"] = edge.SourceId,
                ["dst"] = edge.TargetId,
                ["type"] = edge.Type,
            };
            if (edge.Properties is not null)
                obj["properties"] = ValueConverter.ToJsonNode(edge.Properties);
            array.Add(obj);
        }
        return array.ToJsonString();
    }

    internal static string? Options(UpsertNodesOptions? options)
    {
        if (options is null)
            return null;
        var obj = new JsonObject { ["replace"] = options.Replace };
        if (options.Key is not null)
            obj["key"] = options.Key;
        return obj.ToJsonString();
    }

    internal static string? Options(UpsertEdgesOptions? options)
    {
        if (options is null)
            return null;
        var obj = new JsonObject { ["replace"] = options.Replace };
        if (options.Key is not null)
            obj["key"] = options.Key;
        if (options.EndpointKey is not null)
            obj["endpoint_key"] = options.EndpointKey;
        if (options.EndpointLabels is not null)
            obj["endpoint_labels"] = new JsonArray(options.EndpointLabels.Select(l => (JsonNode?)l).ToArray());
        if (options.SourceField is not null)
            obj["src_field"] = options.SourceField;
        if (options.TargetField is not null)
            obj["dst_field"] = options.TargetField;
        return obj.ToJsonString();
    }

    internal static UpsertSummary Summary(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        var skippedRows = root.GetProperty("skipped_rows").EnumerateArray().Select(e => e.GetInt32()).ToList();
        return new UpsertSummary(
            root.GetProperty("created").GetInt32(),
            root.GetProperty("updated").GetInt32(),
            root.GetProperty("skipped").GetInt32(),
            skippedRows);
    }

    internal static IReadOnlyList<ChangeEvent> Events(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var events = new List<ChangeEvent>();
        foreach (var e in doc.RootElement.EnumerateArray())
        {
            events.Add(new ChangeEvent(
                (long)e.GetProperty("entity_id").GetUInt64(),
                e.GetProperty("entity_type").GetString()!,
                Text(e, "graph"),
                e.GetProperty("kind").GetString()!,
                e.GetProperty("epoch").GetUInt64(),
                e.GetProperty("timestamp").GetUInt64(),
                Properties(e, "before"),
                Properties(e, "after"),
                Strings(e, "labels"),
                Strings(e, "before_labels"),
                Text(e, "edge_type"),
                Id(e, "src_id"),
                Id(e, "dst_id"),
                Id(e, "deleted_with")));
        }
        return events;
    }

    internal static IReadOnlyList<IReadOnlyList<VectorResult>> VectorResults(string json)
    {
        using var doc = JsonDocument.Parse(json);
        return doc.RootElement.EnumerateArray()
            .Select(hits => (IReadOnlyList<VectorResult>)hits.EnumerateArray()
                .Select(hit => new VectorResult(
                    (long)hit.GetProperty("id").GetUInt64(),
                    hit.GetProperty("distance").GetDouble()))
                .ToList())
            .ToList();
    }

    private static bool Present(JsonElement e, string name, out JsonElement value) =>
        e.TryGetProperty(name, out value) && value.ValueKind != JsonValueKind.Null;

    private static string? Text(JsonElement e, string name) =>
        Present(e, name, out var value) ? value.GetString() : null;

    private static long? Id(JsonElement e, string name) =>
        Present(e, name, out var value) ? (long)value.GetUInt64() : null;

    private static IReadOnlyList<string>? Strings(JsonElement e, string name) =>
        Present(e, name, out var value) ? value.EnumerateArray().Select(s => s.GetString()!).ToList() : null;

    private static IReadOnlyDictionary<string, object?>? Properties(JsonElement e, string name) =>
        Present(e, name, out var value)
            ? value.EnumerateObject().ToDictionary(p => p.Name, p => ValueConverter.FromJsonElement(p.Value))
            : null;
}
