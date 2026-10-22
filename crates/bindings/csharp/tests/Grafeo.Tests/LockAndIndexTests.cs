using Xunit;

namespace Grafeo.Tests;

/// <summary>
/// One writer per database file (#405), a property index after a reopen
/// (#638), and SetSchema of a missing schema (#245).
/// </summary>
public sealed class LockAndIndexTests
{
    private static string TempDbPath(string name)
    {
        var dir = Path.Join(Path.GetTempPath(), $"grafeo-csharp-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        return Path.Join(dir, name);
    }

    private static List<object?> Names(GrafeoDB db) =>
        db.Execute("MATCH (p:Person) RETURN p.name AS n ORDER BY n").Rows.Select(row => row["n"]).ToList();

    [Theory]
    [InlineData("people")]
    [InlineData("people.grafeo")]
    public void SecondOpenIsRefusedWhileTheFirstIsOpen(string name)
    {
        var path = TempDbPath(name);
        using (var first = GrafeoDB.Open(path))
        {
            first.Execute("INSERT (:Person {name: 'A'})");
            var error = Assert.ThrowsAny<GrafeoException>(() => GrafeoDB.Open(path));
            Assert.Contains("locked", error.Message);
            first.Execute("INSERT (:Person {name: 'B'})");
        }
        using var reopened = GrafeoDB.Open(path);
        Assert.Equal(new List<object?> { "A", "B" }, Names(reopened));
    }

    [Fact]
    public void PropertyIndexSurvivesCloseAndReopen()
    {
        var path = TempDbPath("indexed.grafeo");
        using (var db = GrafeoDB.Open(path))
        {
            db.Execute("INSERT (:Person {name: 'Alix'}), (:Person {name: 'Gus'})");
            db.Execute("CREATE INDEX person_name FOR (p:Person) ON (p.name)");
            db.Execute("INSERT (:Person {name: 'Vincent'})");
        }
        using var reopened = GrafeoDB.Open(path);
        const string query = "MATCH (p:Person {name: 'Vincent'}) RETURN p.name AS n";
        var plan = reopened.Execute("EXPLAIN " + query);
        var text = plan.Rows[0][plan.Columns[0]] as string;
        Assert.Contains("[index: name]", text);
        var rows = reopened.Execute(query).Rows;
        Assert.Single(rows);
        Assert.Equal("Vincent", rows[0]["n"]);
    }

    [Fact]
    public void SetSchemaOfAMissingSchemaThrowsAndKeepsTheSchema()
    {
        using var db = GrafeoDB.Memory();
        var error = Assert.ThrowsAny<GrafeoException>(() => db.SetSchema("missing"));
        Assert.Contains("Schema 'missing' does not exist", error.Message);
        Assert.Null(db.CurrentSchema());
    }
}
