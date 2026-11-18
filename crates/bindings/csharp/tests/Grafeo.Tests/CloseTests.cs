using System.Collections;
using System.Reflection;
using Xunit;

namespace Grafeo.Tests;

/// <summary>
/// After <see cref="GrafeoDB.Dispose"/> every member of the database throws
/// <see cref="ObjectDisposedException"/> with the GRAFEO-T007 text, and every call on what it
/// handed out before throws a <see cref="GrafeoException"/> with GRAFEO-T007.
/// </summary>
public sealed class CloseTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"grafeo-cs-close-{Guid.NewGuid():N}");

    public CloseTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private GrafeoDB OpenPersistent()
    {
        var db = GrafeoDB.Open(Path.Combine(_dir, "amsterdam.grafeo"));
        db.Execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})");
        return db;
    }

    /// <summary>A value of <paramref name="type"/> that passes argument checks.</summary>
    private static object? Argument(Type type)
    {
        if (type == typeof(string)) return "name";
        if (type == typeof(object)) return "Amsterdam";
        if (type.IsArray) return Array.CreateInstance(type.GetElementType()!, 0);
        if (Nullable.GetUnderlyingType(type) is not null) return null;
        if (type.IsValueType) return Activator.CreateInstance(type);
        if (type.IsGenericType && typeof(IEnumerable).IsAssignableFrom(type))
        {
            var arguments = type.GetGenericArguments();
            if (arguments.Length == 2)
                return Activator.CreateInstance(typeof(Dictionary<,>).MakeGenericType(arguments));
            return Activator.CreateInstance(typeof(List<>).MakeGenericType(arguments[0]));
        }
        return null;
    }

    /// <summary>Runs the member and returns what it threw, awaiting a task it returns.</summary>
    private static Exception? Thrown(Func<object?> call)
    {
        try
        {
            var result = call();
            if (result is Task task) task.GetAwaiter().GetResult();
            else if (result is ValueTask valueTask) valueTask.AsTask().GetAwaiter().GetResult();
            return null;
        }
        catch (TargetInvocationException wrapped) { return wrapped.InnerException; }
        catch (Exception thrown) { return thrown; }
    }

    [Fact]
    public void EveryMemberOfADisposedDatabaseThrowsWithT007()
    {
        var db = OpenPersistent();
        db.Dispose();
        db.Dispose(); // a second Dispose does nothing

        var missed = new List<string>();
        var type = typeof(GrafeoDB);
        foreach (var method in type.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly))
        {
            if (method.Name is nameof(GrafeoDB.Dispose) or nameof(GrafeoDB.DisposeAsync)
                || method.IsGenericMethodDefinition || method.IsSpecialName)
                continue;
            var arguments = method.GetParameters().Select(p => p.HasDefaultValue ? p.DefaultValue : Argument(p.ParameterType)).ToArray();
            Check(method.Name, Thrown(() => method.Invoke(db, arguments)), missed);
        }
        foreach (var property in type.GetProperties(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly))
        {
            if (property.GetMethod is { IsPublic: true } getter)
                Check(property.Name, Thrown(() => getter.Invoke(db, null)), missed);
        }
        Assert.Empty(missed);
    }

    private static void Check(string name, Exception? thrown, List<string> missed)
    {
        if (thrown is not ObjectDisposedException disposed || !disposed.Message.Contains("GRAFEO-T007"))
            missed.Add($"{name}: {thrown?.GetType().Name ?? "answered"} {thrown?.Message}");
    }

    [Fact]
    public void WhatADisposedDatabaseHandedOutThrowsWithT007()
    {
        var db = OpenPersistent();
        db.Execute("CREATE GRAPH berlin");
        var graph = db.Graph("berlin");
        var transaction = db.BeginTransaction();
        transaction.Execute("INSERT (:Person {name: 'Mia'})");
        var stream = db.ExecuteStream("MATCH (p:Person) RETURN p.name");
        var view = db.AsOf(TimePoint.Epoch(db.CurrentEpoch));
        db.Dispose();

        const string query = "MATCH (p:Person) RETURN p.name";
        var rows = new Dictionary<string, Func<object?>>
        {
            ["graph.Execute"] = () => graph.Execute(query),
            ["transaction.Execute"] = () => transaction.Execute(query),
            ["transaction.Commit"] = () => { transaction.Commit(); return null; },
            ["stream.Next"] = () => stream.Next(),
            ["stream.Columns"] = () => stream.Columns,
            ["stream.Warnings"] = () => stream.Warnings,
            ["view.Epoch"] = () => view.Epoch,
            ["view.Execute"] = () => view.Execute(query),
        };
        var missed = new List<string>();
        foreach (var (name, call) in rows)
        {
            var thrown = Thrown(call);
            if (thrown is not GrafeoException grafeo || !grafeo.Message.Contains("GRAFEO-T007"))
                missed.Add($"{name}: {thrown?.GetType().Name ?? "answered"} {thrown?.Message}");
        }
        Assert.Empty(missed);

        view.Dispose();
        stream.Dispose();
        transaction.Dispose();
        graph.Dispose();

        using var reopened = GrafeoDB.Open(Path.Combine(_dir, "amsterdam.grafeo"));
        Assert.Equal(2, reopened.NodeCount); // the transaction open across Dispose left nothing
    }
}
