// The settings a database opens with.

using Grafeo.Native;

namespace Grafeo;

/// <summary>
/// The memory and spill settings a database opens with, for
/// <see cref="GrafeoDB.Open(string?, GrafeoOptions?)"/>. Unset means the default: 75% of the
/// memory the process may use for the memory limit, the memory limit for the query memory
/// limit, a quarter of the memory limit (at most 256 MiB) for the path search budget, a quarter
/// of the memory limit (at least 1 MiB) for the snapshot retention limit, half the spill
/// volume's free space for the spill limit. A write past the memory limit fails with
/// GRAFEO-S001, and so does a path search over its budget. One object may open several
/// databases.
/// </summary>
/// <remarks>
/// It holds no native resource: the settings reach the library when a database opens.
/// <see cref="Dispose"/> lets it stand in a <c>using</c> as the other objects of the binding do.
/// </remarks>
public sealed class GrafeoOptions : IDisposable
{
    private ulong? _memoryLimit;
    private ulong? _queryMemoryLimit;
    private ulong? _pathSearchBudget;
    private ulong? _snapshotRetentionLimit;
    private ulong? _historyRetentionLimit;
    private ulong? _spillLimit;
    private bool _unlimitedSpill;

    /// <summary>
    /// How many bytes the database may hold. A zero or negative limit fails at the open.
    /// </summary>
    public GrafeoOptions SetMemoryLimit(long bytes)
    {
        _memoryLimit = Bytes(bytes);
        return this;
    }

    /// <summary>
    /// How many bytes one statement's sorts, joins and aggregates may hold, never more than the
    /// memory limit. A zero or negative limit fails at the open.
    /// </summary>
    public GrafeoOptions SetQueryMemoryLimit(long bytes)
    {
        _queryMemoryLimit = Bytes(bytes);
        return this;
    }

    /// <summary>
    /// How many bytes one path search may hold: the paths a variable-length or shortest-path
    /// search keeps at once, or what an RDF property path search visits (the nodes an ANY SHORTEST
    /// or shortestPath search visits are not counted yet). Never more than the memory limit. A
    /// search over it fails with GRAFEO-S001. A zero or negative budget fails at the open.
    /// </summary>
    public GrafeoOptions SetPathSearchBudget(long bytes)
    {
        _pathSearchBudget = Bytes(bytes);
        return this;
    }

    /// <summary>
    /// How many bytes of replaced data an open transaction's snapshot may keep: what the commits
    /// since it began replaced that it can still read. Past it the transaction's next call fails
    /// with the retryable GRAFEO-T012 and rolls it back. Below 1 MiB fails at the open.
    /// </summary>
    public GrafeoOptions SetSnapshotRetentionLimit(long bytes)
    {
        _snapshotRetentionLimit = Bytes(bytes);
        return this;
    }

    /// <summary>
    /// How many bytes of replaced data the history window may keep: once the commits since its
    /// start replaced more, the start moves forward, whichever of time or bytes comes first, and a
    /// read before it is refused. No limit by default; zero fails at the open. Not stored with the
    /// database: pass it at every open.
    /// </summary>
    public GrafeoOptions SetHistoryRetentionLimit(long bytes)
    {
        _historyRetentionLimit = Bytes(bytes);
        return this;
    }

    /// <summary>How many bytes may spill to disk; zero means nothing spills.</summary>
    public GrafeoOptions SetSpillLimit(long bytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(bytes);
        _spillLimit = (ulong)bytes;
        return this;
    }

    /// <summary>Lifts the spill limit; it wins over <see cref="SetSpillLimit"/>.</summary>
    public GrafeoOptions SetUnlimitedSpill(bool unlimited)
    {
        _unlimitedSpill = unlimited;
        return this;
    }

    /// <summary>Nothing to release (see the remarks).</summary>
    public void Dispose()
    {
    }

    /// <summary>A size the library checks: a negative one reaches it as zero, which it refuses.</summary>
    private static ulong Bytes(long bytes) => bytes < 0 ? 0 : (ulong)bytes;

    /// <summary>Sets these settings on the native options object <paramref name="native"/>.</summary>
    internal void ApplyTo(OptionsHandle native)
    {
        if (_memoryLimit is { } memory)
            GrafeoException.ThrowIfFailed(NativeMethods.grafeo_options_set_memory_limit(native, memory));
        if (_queryMemoryLimit is { } query)
            GrafeoException.ThrowIfFailed(NativeMethods.grafeo_options_set_query_memory_limit(native, query));
        if (_pathSearchBudget is { } paths)
            GrafeoException.ThrowIfFailed(NativeMethods.grafeo_options_set_path_search_budget(native, paths));
        if (_snapshotRetentionLimit is { } snapshots)
            GrafeoException.ThrowIfFailed(NativeMethods.grafeo_options_set_snapshot_retention_limit(native, snapshots));
        if (_historyRetentionLimit is { } history)
            GrafeoException.ThrowIfFailed(NativeMethods.grafeo_options_set_history_retention_limit(native, history));
        if (_unlimitedSpill)
            GrafeoException.ThrowIfFailed(NativeMethods.grafeo_options_set_unlimited_spill(native));
        else if (_spillLimit is { } spill)
            GrafeoException.ThrowIfFailed(NativeMethods.grafeo_options_set_spill_limit(native, spill));
    }
}
