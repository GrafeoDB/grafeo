// Lazy, cursor-based GQL result stream.

using System.Runtime.InteropServices;
using System.Text.Json;

using Grafeo.Native;

namespace Grafeo;

/// <summary>
/// Async cursor over the rows of a read-only GQL query.
/// Pulls rows from the Rust engine on demand, so memory stays bounded
/// regardless of result size.
///
/// Rejects mutating queries (INSERT / DELETE / SET), EXPLAIN / PROFILE,
/// session/schema commands, and queries that compile to push-based pipelines
/// (ORDER BY, aggregate, DISTINCT). Use <see cref="GrafeoDB.Execute(string)"/>
/// for those.
///
/// Dispose the stream when done (via <c>using</c> or <c>await using</c>) to
/// release the underlying Rust iterator.
///
/// Experimental (0.5.40+): API may change before Beta.
/// </summary>
public sealed class ResultStream : IDisposable, IAsyncDisposable
{
    private readonly IReadOnlyList<string> _columns;
    // _sync serializes Next() with Dispose() so the native handle cannot be
    // freed between ThrowIfDisposed and grafeo_stream_next_row_json.
    private readonly object _sync = new();
    private nint _handle;
    private int _disposed; // 0 = live, 1 = disposed; updated via Interlocked for atomic guard.
    // What the stream had when it was disposed.
    private IReadOnlyList<QueryWarning> _warnings = [];

    internal ResultStream(nint handle, IReadOnlyList<string> columns)
    {
        _handle = handle;
        _columns = columns;
    }

    /// <summary>Column names in the order they appear in each row dictionary.</summary>
    public IReadOnlyList<string> Columns => _columns;

    /// <summary>
    /// The warnings the statement raised so far, ordered by code: its
    /// translation's from the open, its run's as the rows that raise them are
    /// pulled, all of them once <see cref="Next"/> returned <c>null</c>. After
    /// <see cref="Dispose"/> it holds what the stream had then.
    /// Experimental (0.6.2+).
    /// </summary>
    public IReadOnlyList<QueryWarning> Warnings
    {
        get
        {
            lock (_sync)
            {
                return _handle == nint.Zero ? _warnings : ReadWarnings();
            }
        }
    }

    // Reads the warnings through the native handle; the caller holds _sync.
    private IReadOnlyList<QueryWarning> ReadWarnings() =>
        GrafeoDB.ReadWarnings(
            NativeMethods.grafeo_stream_warning_count(_handle),
            index => NativeMethods.grafeo_stream_warning_code(_handle, index),
            index => NativeMethods.grafeo_stream_warning_kind(_handle, index),
            index => NativeMethods.grafeo_stream_warning_message(_handle, index));

    /// <summary>
    /// Pulls the next row, or returns <c>null</c> when the stream is exhausted.
    /// </summary>
    public Dictionary<string, object?>? Next()
    {
        nint rowPtr;
        int status;
        // Hold _sync across the FFI call so a concurrent Dispose cannot free
        // _handle while grafeo_stream_next_row_json is running. JSON decoding
        // happens after the call returns and can be done outside the lock.
        lock (_sync)
        {
            ThrowIfDisposed();
            status = NativeMethods.grafeo_stream_next_row_json(_handle, out rowPtr);
        }
        if (status != 0)
        {
            throw GrafeoException.FromLastError((GrafeoStatus)status);
        }
        if (rowPtr == nint.Zero)
        {
            return null;
        }
        try
        {
            var json = Marshal.PtrToStringUTF8(rowPtr)
                ?? throw new GrafeoException("Failed to decode row JSON", GrafeoStatus.Serialization);
            // Each value decoded as a query row's is ($bytes, $decimal, dates, ...)
            using var doc = JsonDocument.Parse(json);
            if (doc.RootElement.ValueKind != JsonValueKind.Object)
                throw new GrafeoException("Row JSON is not an object", GrafeoStatus.Serialization);
            var dict = new Dictionary<string, object?>();
            foreach (var prop in doc.RootElement.EnumerateObject())
                dict[prop.Name] = ValueConverter.FromJsonElement(prop.Value);
            return dict;
        }
        finally
        {
            NativeMethods.grafeo_free_string(rowPtr);
        }
    }

    /// <summary>
    /// Iterates the remaining rows. Convenient for
    /// <c>foreach (var row in stream.Rows())</c>.
    /// </summary>
    public IEnumerable<Dictionary<string, object?>> Rows()
    {
        while (true)
        {
            var row = Next();
            if (row is null) yield break;
            yield return row;
        }
    }

    /// <summary>Drains the stream into a list. Prefer <see cref="Rows"/> for large result sets.</summary>
    public List<Dictionary<string, object?>> ToList()
    {
        var list = new List<Dictionary<string, object?>>();
        foreach (var row in Rows())
        {
            list.Add(row);
        }
        return list;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        // CompareExchange guarantees a single Dispose call owns the free,
        // so concurrent callers cannot double-free the native handle.
        if (Interlocked.CompareExchange(ref _disposed, 1, 0) != 0) return;
        // _sync waits for any in-flight Next() to exit its FFI call before we
        // release the handle, preventing a use-after-free across threads.
        lock (_sync)
        {
            if (_handle != nint.Zero)
            {
                _warnings = ReadWarnings();
                NativeMethods.grafeo_stream_free(_handle);
                _handle = nint.Zero;
            }
        }
    }

    /// <inheritdoc/>
    public ValueTask DisposeAsync()
    {
        Dispose();
        return ValueTask.CompletedTask;
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0)
        {
            throw new ObjectDisposedException(nameof(ResultStream));
        }
    }
}
