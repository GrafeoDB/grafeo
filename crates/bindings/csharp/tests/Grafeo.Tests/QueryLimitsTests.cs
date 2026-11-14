using Xunit;

namespace Grafeo.Tests;

/// <summary>
/// The path search budget through C# (#492): <see cref="GrafeoOptions.SetPathSearchBudget"/>.
/// Zero fails at the open; the budget binds, with its error code and the setting named in the
/// message.
/// </summary>
public sealed class QueryLimitsTests
{
    private const string Star =
        "INSERT (c:Star {n: 0}), (c)-[:E]->(:Leaf {n: 1}), (c)-[:E]->(:Leaf {n: 2}), " +
        "(c)-[:E]->(:Leaf {n: 3}), (c)-[:E]->(:Leaf {n: 4}), (c)-[:E]->(:Leaf {n: 5})";

    private const string Paths = "MATCH (s:Star)-[:E*1..2]->(t) RETURN t.n AS n";

    private static GrafeoDB Opened(Action<GrafeoOptions> set)
    {
        using var options = new GrafeoOptions();
        set(options);
        var db = GrafeoDB.Open(null, options);
        db.Execute(Star);
        return db;
    }

    private static void AssertRefused(GrafeoDB db, string query, string code, string setting)
    {
        var error = Assert.Throws<GrafeoException>(() => db.Execute(query));
        Assert.Contains(code, error.Message);
        Assert.Contains(setting, error.Message);
    }

    [Fact]
    public void ZeroIsRefused()
    {
        using var options = new GrafeoOptions();
        Assert.ThrowsAny<Exception>(() =>
        {
            options.SetPathSearchBudget(0);
            using var db = GrafeoDB.Open(null, options);
        });
    }

    [Fact]
    public void ThePathSearchBudgetBinds()
    {
        using var db = Opened(o => o.SetPathSearchBudget(1));
        AssertRefused(db, Paths, "GRAFEO-S001", "path_search_budget");
    }
}
