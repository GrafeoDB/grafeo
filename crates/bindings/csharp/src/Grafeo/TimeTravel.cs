// Time travel: reads of the database as it was at an earlier epoch or time.

using System.Runtime.InteropServices;
using System.Text.Json;

using Grafeo.Native;

namespace Grafeo;

/// <summary>
/// A point in the database's history: an epoch (<see cref="Epoch"/>), a time (<see cref="At"/>)
/// or a duration before now (<see cref="Ago"/>). Times have the resolution of a millisecond.
/// </summary>
public readonly record struct TimePoint
{
    // The GRAFEO_POINT_* kinds of the C API.
    private const int EpochKind = 0;
    private const int TimeKind = 1;
    private const int AgoKind = 2;

    private TimePoint(int kind, long value) => (Kind, Value) = (kind, value);

    internal int Kind { get; }

    internal long Value { get; }

    /// <summary>The committed state of <paramref name="epoch"/>; a later epoch reads now.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The epoch is above <see cref="long.MaxValue"/>.</exception>
    public static TimePoint Epoch(ulong epoch)
    {
        ArgumentOutOfRangeException.ThrowIfGreaterThan(epoch, (ulong)long.MaxValue);
        return new TimePoint(EpochKind, (long)epoch);
    }

    /// <summary>The last commit at or before <paramref name="time"/>.</summary>
    public static TimePoint At(DateTimeOffset time) => new(TimeKind, time.ToUnixTimeMilliseconds());

    /// <summary>The last commit at or before <paramref name="duration"/> ago.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The duration is negative.</exception>
    public static TimePoint Ago(TimeSpan duration)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(duration, TimeSpan.Zero);
        return new TimePoint(AgoKind, (long)duration.TotalMilliseconds);
    }
}

/// <summary>
/// How much history a database keeps for time-travel reads: none (<see cref="Off"/>, the default),
/// all of it (<see cref="Forever"/>) or what was committed within a duration before now
/// (<see cref="For"/>). A database file stores the setting; one given at open replaces it.
/// </summary>
public readonly record struct HistoryRetention
{
    // The GRAFEO_HISTORY_* kinds of the C API.
    private const int OffKind = 0;
    private const int ForeverKind = 1;
    private const int DurationKind = 2;

    private HistoryRetention(int kind, ulong durationMs) => (Kind, DurationMs) = (kind, durationMs);

    internal int Kind { get; }

    internal ulong DurationMs { get; }

    /// <summary>No history: only the current epoch can be read.</summary>
    public static HistoryRetention Off => new(OffKind, 0);

    /// <summary>All history.</summary>
    public static HistoryRetention Forever => new(ForeverKind, 0);

    /// <summary>The history committed within <paramref name="duration"/> before now.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The duration is negative.</exception>
    public static HistoryRetention For(TimeSpan duration)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(duration, TimeSpan.Zero);
        return new HistoryRetention(DurationKind, (ulong)duration.TotalMilliseconds);
    }
}

/// <summary>
/// A read-only view of the database as it was at an earlier epoch, from <see cref="GrafeoDB.AsOf"/>.
/// Its queries and reads see the committed state of <see cref="Epoch"/>; a query that writes throws. It keeps
/// the history of its epoch until disposed.
/// </summary>
public sealed class AsOfView : IDisposable
{
    private readonly AsOfHandle _handle;
    private volatile bool _disposed;

    internal AsOfView(nint ptr)
    {
        _handle = new AsOfHandle();
        Marshal.InitHandle(_handle, ptr);
        Epoch = NativeMethods.grafeo_as_of_epoch(ptr);
    }

    /// <summary>The epoch the view reads at.</summary>
    public ulong Epoch { get; }

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

    /// <summary>Execute a GQL query as the database was at the view's epoch.</summary>
    public QueryResult Execute(string query) =>
        Result(NativeMethods.grafeo_as_of_execute(Handle, query, null));

    /// <summary>Execute a GQL query with parameters as the database was at the view's epoch.</summary>
    public QueryResult ExecuteWithParams(string query, Dictionary<string, object?> parameters) =>
        Result(NativeMethods.grafeo_as_of_execute(Handle, query, ValueConverter.EncodeParams(parameters)));

    /// <summary>The node as it was at the view's epoch, or null when it did not exist then.</summary>
    public Node? GetNode(long id)
    {
        if (NativeMethods.grafeo_as_of_get_node(Handle, (ulong)id, out var nodePtr) != (int)GrafeoStatus.Ok)
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

    /// <summary>The edge as it was at the view's epoch, or null when it did not exist then.</summary>
    public Edge? GetEdge(long id)
    {
        if (NativeMethods.grafeo_as_of_get_edge(Handle, (ulong)id, out var edgePtr) != (int)GrafeoStatus.Ok)
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

    /// <summary>
    /// A node's value of <paramref name="key"/> as it was at the view's epoch, or null when the node
    /// or the property did not exist then.
    /// </summary>
    public object? GetNodeProperty(long id, string key)
    {
        var ptr = NativeMethods.grafeo_as_of_get_node_property(Handle, (ulong)id, key);
        if (ptr == nint.Zero)
            return null;
        try
        {
            using var json = JsonDocument.Parse(Marshal.PtrToStringUTF8(ptr)!);
            return ValueConverter.FromJsonElement(json.RootElement);
        }
        finally
        {
            NativeMethods.grafeo_free_string(ptr);
        }
    }

    private static QueryResult Result(nint resultPtr)
    {
        if (resultPtr == nint.Zero)
            throw GrafeoException.FromLastError(GrafeoStatus.Query);
        return GrafeoDB.BuildResult(resultPtr);
    }
}
