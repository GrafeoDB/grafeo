// SafeHandle subclasses for automatic native resource cleanup.

using System.Runtime.InteropServices;

namespace Grafeo.Native;

/// <summary>
/// Safe handle wrapping a <c>GrafeoDatabase*</c>.
/// Calls <c>grafeo_close</c> + <c>grafeo_free_database</c> on release.
/// </summary>
internal sealed class DatabaseHandle : SafeHandle
{
    public DatabaseHandle() : base(nint.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == nint.Zero;

    protected override bool ReleaseHandle()
    {
        ReleaseMethods.grafeo_close(handle);
        ReleaseMethods.grafeo_free_database(handle);
        return true;
    }
}

/// <summary>
/// Safe handle wrapping a <c>GrafeoTransaction*</c>.
/// Auto-rolls back and frees on release.
/// </summary>
internal sealed class TransactionHandle : SafeHandle
{
    public TransactionHandle() : base(nint.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == nint.Zero;

    internal volatile bool Committed;

    protected override bool ReleaseHandle()
    {
        if (!Committed)
        {
            ReleaseMethods.grafeo_rollback(handle);
        }
        ReleaseMethods.grafeo_free_transaction(handle);
        return true;
    }
}

/// <summary>
/// Safe handle wrapping a <c>GrafeoResult*</c>.
/// Frees the result on release.
/// </summary>
internal sealed class ResultHandle : SafeHandle
{
    public ResultHandle() : base(nint.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == nint.Zero;

    protected override bool ReleaseHandle()
    {
        ReleaseMethods.grafeo_free_result(handle);
        return true;
    }
}

/// <summary>
/// Safe handle wrapping a <c>GrafeoGraph*</c>.
/// Frees the handle on release.
/// </summary>
internal sealed class GraphHandle : SafeHandle
{
    public GraphHandle() : base(nint.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == nint.Zero;

    protected override bool ReleaseHandle()
    {
        ReleaseMethods.grafeo_free_graph(handle);
        return true;
    }
}

/// <summary>
/// Safe handle wrapping a <c>GrafeoAsOf*</c> view.
/// Frees the view on release.
/// </summary>
internal sealed class AsOfHandle : SafeHandle
{
    public AsOfHandle() : base(nint.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == nint.Zero;

    protected override bool ReleaseHandle()
    {
        ReleaseMethods.grafeo_free_as_of(handle);
        return true;
    }
}

/// <summary>
/// Safe handle wrapping a <c>GrafeoStream*</c>.
/// Frees the stream, and its hold on the database, on release.
/// </summary>
internal sealed class StreamHandle : SafeHandle
{
    public StreamHandle() : base(nint.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == nint.Zero;

    protected override bool ReleaseHandle()
    {
        ReleaseMethods.grafeo_stream_free(handle);
        return true;
    }
}

/// <summary>
/// Safe handle wrapping a <c>GrafeoNode*</c>.
/// Frees the node on release.
/// </summary>
internal sealed class NodeHandle : SafeHandle
{
    public NodeHandle() : base(nint.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == nint.Zero;

    protected override bool ReleaseHandle()
    {
        ReleaseMethods.grafeo_free_node(handle);
        return true;
    }
}

/// <summary>
/// Safe handle wrapping a <c>GrafeoEdge*</c>.
/// Frees the edge on release.
/// </summary>
internal sealed class EdgeHandle : SafeHandle
{
    public EdgeHandle() : base(nint.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == nint.Zero;

    protected override bool ReleaseHandle()
    {
        ReleaseMethods.grafeo_free_edge(handle);
        return true;
    }
}

/// <summary>
/// Safe handle wrapping a <c>GrafeoOptions*</c>.
/// Frees the open options on release.
/// </summary>
internal sealed class OptionsHandle : SafeHandle
{
    public OptionsHandle() : base(nint.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == nint.Zero;

    protected override bool ReleaseHandle()
    {
        ReleaseMethods.grafeo_options_free(handle);
        return true;
    }
}

/// <summary>
/// The native calls a handle's <c>ReleaseHandle</c> makes, with the raw
/// pointer: a SafeHandle being released cannot add a reference to itself.
/// Every other native call takes the SafeHandle type (see
/// <see cref="NativeMethods"/>), so the handle is never freed during one.
/// </summary>
internal static partial class ReleaseMethods
{
    /// <summary>Close the database, flushing pending writes. Returns status code.</summary>
    [LibraryImport(NativeMethods.LibName)]
    internal static partial int grafeo_close(nint db);

    /// <summary>Free a database handle. Must be called after grafeo_close.</summary>
    [LibraryImport(NativeMethods.LibName)]
    internal static partial void grafeo_free_database(nint db);

    /// <summary>Free a transaction handle.</summary>
    [LibraryImport(NativeMethods.LibName)]
    internal static partial void grafeo_free_transaction(nint tx);

    /// <summary>Free a graph handle.</summary>
    [LibraryImport(NativeMethods.LibName)]
    internal static partial void grafeo_free_graph(nint graph);

    [LibraryImport(NativeMethods.LibName)]
    internal static partial void grafeo_free_as_of(nint view);

    /// <summary>Rollback a transaction.</summary>
    [LibraryImport(NativeMethods.LibName)]
    internal static partial int grafeo_rollback(nint tx);

    /// <summary>Free a query result.</summary>
    [LibraryImport(NativeMethods.LibName)]
    internal static partial void grafeo_free_result(nint result);

    /// <summary>Free a stream, and its hold on the database.</summary>
    [LibraryImport(NativeMethods.LibName)]
    internal static partial void grafeo_stream_free(nint stream);

    /// <summary>Free a node.</summary>
    [LibraryImport(NativeMethods.LibName)]
    internal static partial void grafeo_free_node(nint node);

    /// <summary>Free an edge.</summary>
    [LibraryImport(NativeMethods.LibName)]
    internal static partial void grafeo_free_edge(nint edge);

    /// <summary>Free open options; null is ignored.</summary>
    [LibraryImport(NativeMethods.LibName)]
    internal static partial void grafeo_options_free(nint options);
}
