// The settings a database opens with.

using Grafeo.Native;

namespace Grafeo;

/// <summary>
/// The memory, spill and query limit settings a database opens with, for
/// <see cref="GrafeoDB.Open(string?, GrafeoOptions?)"/>. Unset means the default: 75% of the
/// memory the process may use for the memory limit, the memory limit for the query memory
/// limit, a quarter of the memory limit (at most 256 MiB) for the path search budget, 100
/// million for the path expansion limit, 16 MiB for the query length and 16 KiB for the
/// identifier length, a quarter of the memory limit (at least 1 MiB) for the snapshot retention
/// limit, half the spill
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
    private ulong? _pathExpansionLimit;
    private ulong? _maxQueryLength;
    private ulong? _maxIdentifierLength;
    private ulong? _snapshotRetentionLimit;
    private ulong? _historyRetentionLimit;
    private ulong? _spillLimit;
    private bool _unlimitedSpill;
    private ulong? _planCacheLimit;

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
    /// How many bytes one path search may hold: everything a variable-length, shortest-path or
    /// RDF property path search holds at once (the nodes it visits, its queues and the paths it
    /// keeps), which counts against the query memory limit too. Never more than the memory limit. A
    /// search over it fails with GRAFEO-S001. A zero or negative budget fails at the open.
    /// </summary>
    public GrafeoOptions SetPathSearchBudget(long bytes)
    {
        _pathSearchBudget = Bytes(bytes);
        return this;
    }

    /// <summary>
    /// How many path expansions one statement's variable-length and shortest-path searches may
    /// make together: each edge they read to extend a path, and each node and edge they copy out
    /// of a path into a row. 100 million by default. A statement over it fails with GRAFEO-Q009.
    /// A zero or negative limit fails at the open.
    /// </summary>
    public GrafeoOptions SetPathExpansionLimit(long expansions)
    {
        _pathExpansionLimit = Bytes(expansions);
        return this;
    }

    /// <summary>
    /// How many bytes a statement's text may have; the values passed as parameters do not count.
    /// 16 MiB by default. A longer statement fails with GRAFEO-Q002. A zero or negative length
    /// fails at the open.
    /// </summary>
    public GrafeoOptions SetMaxQueryLength(long bytes)
    {
        _maxQueryLength = Bytes(bytes);
        return this;
    }

    /// <summary>
    /// How many bytes one name in a statement may have, as written between its delimiters
    /// (string literals and IRIs are not names). 16 KiB by default. A statement with a longer
    /// name fails with GRAFEO-Q002. A zero or negative length fails at the open.
    /// </summary>
    public GrafeoOptions SetMaxIdentifierLength(long bytes)
    {
        _maxIdentifierLength = Bytes(bytes);
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

    /// <summary>
    /// How many bytes of plans the database keeps for repeated statements, by a deep estimate of
    /// what each plan holds; zero keeps none, so every statement plans. By default a sixty-fourth
    /// of the memory limit, from 4 MiB to 64 MiB; never more than the memory limit. The report's
    /// <c>caches.plan_cache_limit_bytes</c> reads it back.
    /// </summary>
    public GrafeoOptions SetPlanCacheLimit(long bytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(bytes);
        _planCacheLimit = (ulong)bytes;
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
        if (_pathExpansionLimit is { } expansions)
            GrafeoException.ThrowIfFailed(NativeMethods.grafeo_options_set_path_expansion_limit(native, expansions));
        if (_maxQueryLength is { } queryLength)
            GrafeoException.ThrowIfFailed(NativeMethods.grafeo_options_set_max_query_length(native, queryLength));
        if (_maxIdentifierLength is { } identifierLength)
            GrafeoException.ThrowIfFailed(NativeMethods.grafeo_options_set_max_identifier_length(native, identifierLength));
        if (_snapshotRetentionLimit is { } snapshots)
            GrafeoException.ThrowIfFailed(NativeMethods.grafeo_options_set_snapshot_retention_limit(native, snapshots));
        if (_historyRetentionLimit is { } history)
            GrafeoException.ThrowIfFailed(NativeMethods.grafeo_options_set_history_retention_limit(native, history));
        if (_planCacheLimit is { } plans)
            GrafeoException.ThrowIfFailed(NativeMethods.grafeo_options_set_plan_cache_limit(native, plans));
        if (_unlimitedSpill)
            GrafeoException.ThrowIfFailed(NativeMethods.grafeo_options_set_unlimited_spill(native));
        else if (_spillLimit is { } spill)
            GrafeoException.ThrowIfFailed(NativeMethods.grafeo_options_set_spill_limit(native, spill));
    }
}
