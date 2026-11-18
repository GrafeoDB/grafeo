using Xunit;

namespace Grafeo.Tests;

/// <summary>
/// The path search budget through C# (#492): <see cref="GrafeoOptions.SetPathSearchBudget"/>.
/// Zero fails at the open; the budget binds, with its error code and the setting named in the
/// message. The snapshot retention limit, <see cref="GrafeoOptions.SetSnapshotRetentionLimit"/>:
/// below 1 MiB fails at the open; past it an open transaction fails with GRAFEO-T012.
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

    // Any GrafeoException: the code and the setting in the message are what the refusal promises.
    // Execute raises every failure as a QueryException today, where the native category of a
    // limit refusal is Storage.
    private static void AssertRefused(GrafeoDB db, string query, string code, string setting)
    {
        var error = Assert.ThrowsAny<GrafeoException>(() => db.Execute(query));
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

    private const string ReadBio = "MATCH (p:Person {name: 'Alix'}) RETURN p.bio AS bio";

    [Fact]
    public void ASnapshotRetentionLimitBelow1MiBIsRefused()
    {
        using var options = new GrafeoOptions().SetSnapshotRetentionLimit(512 * 1024);
        var error = Assert.ThrowsAny<Exception>(() =>
        {
            using var db = GrafeoDB.Open(null, options);
        });
        Assert.Contains("at least 1 MiB", error.Message);
    }

    // Past the limit (20 commits replace 64 KiB each, over 1 MiB) the open transaction that read
    // Alix fails with GRAFEO-T012, and rolls back.
    [Fact]
    public void TheSnapshotRetentionLimitBinds()
    {
        using var options = new GrafeoOptions().SetSnapshotRetentionLimit(1 << 20);
        using var db = GrafeoDB.Open(null, options);
        db.Execute("INSERT (:Person {name: 'Alix', bio: 'Amsterdam'})");
        using var tx = db.BeginTransaction();
        tx.Execute(ReadBio);
        var text = string.Concat(Enumerable.Repeat("Vincent ", 8 * 1024));
        for (var round = 0; round < 20; round++)
            db.Execute($"MATCH (p:Person {{name: 'Alix'}}) SET p.bio = '{round}{text}'");
        var error = Assert.ThrowsAny<GrafeoException>(() => tx.Execute(ReadBio));
        Assert.Contains("GRAFEO-T012", error.Message);
        tx.Rollback();
    }
}
