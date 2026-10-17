// ACID transaction handle with auto-rollback on dispose.

using System.Runtime.InteropServices;

using Grafeo.Native;

namespace Grafeo;

/// <summary>
/// An ACID transaction. Auto-rolls back on <see cref="Dispose"/> if
/// <see cref="Commit"/> was not called, making it exception-safe:
/// <code>
/// using var tx = db.BeginTransaction();
/// tx.Execute("INSERT (:Person {name: 'Alix'})");
/// tx.Commit(); // if this line is not reached, the transaction rolls back
/// </code>
/// </summary>
public sealed class Transaction : ITransaction, IDisposable, IAsyncDisposable
{
    private readonly TransactionHandle _handle;
    private volatile bool _finished;
    private volatile bool _disposed;

    internal Transaction(nint ptr)
    {
        _handle = new TransactionHandle();
        Marshal.InitHandle(_handle, ptr);
    }

    // =========================================================================
    // Query Execution
    // =========================================================================

    /// <summary>Execute a GQL query within this transaction.</summary>
    public QueryResult Execute(string query)
    {
        ThrowIfFinished();
        var resultPtr = NativeMethods.grafeo_transaction_execute(Handle, query);
        if (resultPtr == nint.Zero)
            throw GrafeoException.FromLastError(GrafeoStatus.Query);
        return BuildResult(resultPtr);
    }

    /// <summary>Execute a GQL query within this transaction on the thread pool.</summary>
    public Task<QueryResult> ExecuteAsync(string query, CancellationToken ct = default)
    {
        ThrowIfFinished();
        var h = Handle;
        return Task.Run(() =>
        {
            ct.ThrowIfCancellationRequested();
            var resultPtr = NativeMethods.grafeo_transaction_execute(h, query);
            if (resultPtr == nint.Zero)
                throw GrafeoException.FromLastError(GrafeoStatus.Query);
            return BuildResult(resultPtr);
        }, ct);
    }

    /// <summary>Execute a GQL query with parameters within this transaction.</summary>
    public QueryResult ExecuteWithParams(string query, Dictionary<string, object?> parameters)
    {
        ThrowIfFinished();
        var paramsJson = ValueConverter.EncodeParams(parameters);
        var resultPtr = NativeMethods.grafeo_transaction_execute_with_params(Handle, query, paramsJson);
        if (resultPtr == nint.Zero)
            throw GrafeoException.FromLastError(GrafeoStatus.Query);
        return BuildResult(resultPtr);
    }

    /// <summary>Execute a GQL query with parameters within this transaction on the thread pool.</summary>
    public Task<QueryResult> ExecuteWithParamsAsync(
        string query,
        Dictionary<string, object?> parameters,
        CancellationToken ct = default)
    {
        ThrowIfFinished();
        var paramsJson = ValueConverter.EncodeParams(parameters);
        var h = Handle;
        return Task.Run(() =>
        {
            ct.ThrowIfCancellationRequested();
            var resultPtr = NativeMethods.grafeo_transaction_execute_with_params(h, query, paramsJson);
            if (resultPtr == nint.Zero)
                throw GrafeoException.FromLastError(GrafeoStatus.Query);
            return BuildResult(resultPtr);
        }, ct);
    }

    /// <summary>
    /// Execute a query in the specified language within this transaction.
    /// </summary>
    /// <param name="language">Query language: "gql", "cypher", "gremlin", "graphql", "sparql", or "sql".</param>
    /// <param name="query">The query string.</param>
    /// <param name="paramsJson">Optional JSON-encoded parameters (null for none).</param>
    public QueryResult ExecuteLanguage(string language, string query, string? paramsJson = null)
    {
        ThrowIfFinished();
        var resultPtr = NativeMethods.grafeo_transaction_execute_language(Handle, language, query, paramsJson);
        if (resultPtr == nint.Zero)
            throw GrafeoException.FromLastError(GrafeoStatus.Query);
        return BuildResult(resultPtr);
    }

    /// <summary>
    /// Execute a query in the specified language within this transaction on the thread pool.
    /// </summary>
    /// <param name="language">Query language: "gql", "cypher", "gremlin", "graphql", "sparql", or "sql".</param>
    /// <param name="query">The query string.</param>
    /// <param name="paramsJson">Optional JSON-encoded parameters (null for none).</param>
    /// <param name="ct">Cancellation token.</param>
    public Task<QueryResult> ExecuteLanguageAsync(
        string language,
        string query,
        string? paramsJson = null,
        CancellationToken ct = default)
    {
        ThrowIfFinished();
        var h = Handle;
        return Task.Run(() =>
        {
            ct.ThrowIfCancellationRequested();
            var resultPtr = NativeMethods.grafeo_transaction_execute_language(h, language, query, paramsJson);
            if (resultPtr == nint.Zero)
                throw GrafeoException.FromLastError(GrafeoStatus.Query);
            return BuildResult(resultPtr);
        }, ct);
    }

    // =========================================================================
    // Commit / Rollback
    // =========================================================================

    /// <summary>Commit the transaction, making all changes permanent.</summary>
    public void Commit()
    {
        ThrowIfFinished();
        var h = Handle;
        _finished = true;
        var status = NativeMethods.grafeo_commit(h);
        if (status != (int)GrafeoStatus.Ok)
            throw GrafeoException.FromLastError(GrafeoStatus.Transaction);
        _handle.Committed = true;
    }

    /// <summary>Roll back the transaction, discarding all changes.</summary>
    public void Rollback()
    {
        if (_finished) return;
        var h = Handle;
        _finished = true;
        var status = NativeMethods.grafeo_rollback(h);
        if (status != (int)GrafeoStatus.Ok)
            throw GrafeoException.FromLastError(GrafeoStatus.Transaction);
    }

    // =========================================================================
    // Dispose
    // =========================================================================

    /// <summary>
    /// Dispose the transaction. If <see cref="Commit"/> was not called,
    /// the transaction is automatically rolled back.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (!_finished)
        {
            // Best-effort rollback; swallow errors during dispose.
            _finished = true;
            NativeMethods.grafeo_rollback(_handle.DangerousGetHandle());
        }
        _handle.Dispose();
    }

    /// <inheritdoc/>
    public ValueTask DisposeAsync()
    {
        Dispose();
        return ValueTask.CompletedTask;
    }

    // =========================================================================
    // Internals
    // =========================================================================

    private nint Handle
    {
        get
        {
            ThrowIfFinished();
            return _handle.DangerousGetHandle();
        }
    }

    private void ThrowIfFinished()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_finished)
            throw new TransactionException("Transaction is already committed or rolled back");
    }

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
}
