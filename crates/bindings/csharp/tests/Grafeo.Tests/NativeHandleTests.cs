using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using Xunit;

namespace Grafeo.Tests;

/// <summary>
/// Every native object the binding holds is a <see cref="SafeHandle"/>, so the finalizer frees
/// one the caller never disposed, and a handle is never freed during a call that uses it (#712).
/// The test reads the C header and the binding's native calls, so a new call is covered without
/// a row here: a parameter, an out parameter or a return value the header types as a pointer to
/// an opaque <c>Grafeo*</c> object must be a SafeHandle type in <c>NativeMethods</c>; only
/// <c>ReleaseMethods</c>, which a handle's release calls, passes the raw pointer, and every free
/// of an opaque object is declared there alone.
/// </summary>
public sealed class NativeHandleTests
{
    /// <summary>One C function: its return type and its parameters' types, in order.</summary>
    private sealed record CFunction(string Return, IReadOnlyList<string> Parameters);

    private static readonly Regex Opaque = new(@"\bGrafeo(?!Status\b)[A-Za-z]+\s*\*", RegexOptions.Compiled);

    /// <summary>The header beside the C binding, found from the test's folder upwards.</summary>
    private static string HeaderText()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            var header = Path.Combine(dir.FullName, "crates", "bindings", "c", "grafeo.h");
            if (File.Exists(header))
                return File.ReadAllText(header);
            var sibling = Path.Combine(dir.FullName, "c", "grafeo.h");
            if (File.Exists(sibling))
                return File.ReadAllText(sibling);
        }
        throw new FileNotFoundException("crates/bindings/c/grafeo.h above " + AppContext.BaseDirectory);
    }

    /// <summary>The header's functions by name, comments taken out.</summary>
    private static Dictionary<string, CFunction> HeaderFunctions()
    {
        var text = Regex.Replace(HeaderText(), @"/\*.*?\*/|//[^\n]*", " ", RegexOptions.Singleline);
        text = Regex.Replace(text, @"^[ \t]*#[^\n]*", " ", RegexOptions.Multiline);
        var functions = new Dictionary<string, CFunction>();
        foreach (Match match in Regex.Matches(text, @"([A-Za-z_][\w\s\*]*?)\b(grafeo_\w+)\s*\(([^)]*)\)\s*;"))
        {
            var parameters = match.Groups[3].Value.Trim();
            functions[match.Groups[2].Value] = new CFunction(
                match.Groups[1].Value.Trim(),
                parameters is "" or "void"
                    ? []
                    : parameters.Split(',').Select(p => p.Trim()).ToList());
        }
        Assert.True(functions.Count > 100, $"the header's functions were read: {functions.Count}");
        return functions;
    }

    private static Type Native(string name) =>
        typeof(GrafeoDB).Assembly.GetType("Grafeo.Native." + name)
        ?? throw new InvalidOperationException("no type Grafeo.Native." + name);

    private static IEnumerable<MethodInfo> Imports(Type type) =>
        type.GetMethods(BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public)
            .Where(method => method.Name.StartsWith("grafeo_", StringComparison.Ordinal));

    private static bool IsSafeHandle(Type type) =>
        typeof(SafeHandle).IsAssignableFrom(type.IsByRef ? type.GetElementType()! : type);

    [Fact]
    public void EveryOpaqueObjectCrossesAsASafeHandle()
    {
        var header = HeaderFunctions();
        var wrong = new List<string>();
        foreach (var method in Imports(Native("NativeMethods")))
        {
            if (!header.TryGetValue(method.Name, out var function))
            {
                wrong.Add($"{method.Name}: not in grafeo.h");
                continue;
            }
            if (Opaque.IsMatch(function.Return) && !IsSafeHandle(method.ReturnType))
                wrong.Add($"{method.Name}: returns {function.Return} as {method.ReturnType.Name}");
            var parameters = method.GetParameters();
            for (var i = 0; i < Math.Min(parameters.Length, function.Parameters.Count); i++)
            {
                if (Opaque.IsMatch(function.Parameters[i]) && !IsSafeHandle(parameters[i].ParameterType))
                    wrong.Add($"{method.Name}: `{function.Parameters[i]}` as {parameters[i].ParameterType.Name}");
            }
        }
        Assert.True(wrong.Count == 0, string.Join("\n", wrong));
    }

    [Fact]
    public void OnlyReleaseMethodsFreeAnOpaqueObject()
    {
        var header = HeaderFunctions();
        static bool Frees(string name) =>
            name.StartsWith("grafeo_free_", StringComparison.Ordinal) || name.EndsWith("_free", StringComparison.Ordinal);
        var wrong = Imports(Native("NativeMethods"))
            .Where(method => Frees(method.Name)
                && header.TryGetValue(method.Name, out var function)
                && function.Parameters.Count > 0
                && Opaque.IsMatch(function.Parameters[0]))
            .Select(method => method.Name)
            .ToList();
        Assert.True(wrong.Count == 0, "free in NativeMethods, not ReleaseMethods: " + string.Join(", ", wrong));
    }
}
