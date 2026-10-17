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
        NativeMethods.grafeo_close(handle);
        NativeMethods.grafeo_free_database(handle);
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
            NativeMethods.grafeo_rollback(handle);
        }
        NativeMethods.grafeo_free_transaction(handle);
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
        NativeMethods.grafeo_free_result(handle);
        return true;
    }
}
