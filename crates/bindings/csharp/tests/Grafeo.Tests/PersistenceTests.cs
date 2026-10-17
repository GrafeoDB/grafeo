using Xunit;

namespace Grafeo.Tests;

/// <summary>Persistent storage tests: open, close, reopen, verify data survives.</summary>
public sealed class PersistenceTests
{
    private static string TempDbPath(string name)
    {
        var dir = Path.Combine(Path.GetTempPath(), $"grafeo-csharp-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        var fileName = Path.GetFileName(name);
        if (string.IsNullOrEmpty(fileName))
            fileName = "db.grafeo";
        return Path.Combine(dir, fileName);
    }

    private static void Cleanup(string dbPath)
    {
        var dir = Path.GetDirectoryName(dbPath)!;
        try { Directory.Delete(dir, recursive: true); }
        catch (IOException)
        {
            // Best-effort cleanup in tests: ignore IO failures when deleting the temp directory.
        }
        catch (UnauthorizedAccessException)
        {
            // Best-effort cleanup in tests: ignore permission issues when deleting the temp directory.
        }
    }

    [Fact]
    public void CreateAndReopen()
    {
        var dbPath = TempDbPath("reopen.grafeo");
        try
        {
            // Create and populate
            using (var db = GrafeoDB.Open(dbPath))
            {
                db.Execute("INSERT (:Person {name: 'Alix', age: 30})");
                db.Execute("INSERT (:Person {name: 'Gus', age: 25})");
                db.Execute(
                    "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) " +
                    "INSERT (a)-[:KNOWS]->(b)");

                Assert.Equal(2, db.NodeCount);
                Assert.Equal(1, db.EdgeCount);
            }

            // Reopen and verify
            using (var db = GrafeoDB.Open(dbPath))
            {
                Assert.Equal(2, db.NodeCount);
                Assert.Equal(1, db.EdgeCount);

                var result = db.Execute("MATCH (p:Person) RETURN p.name ORDER BY p.name");
                var names = result.Rows
                    .Select(r => r["p.name"]?.ToString())
                    .Order()
                    .ToList();
                Assert.Equal(["Alix", "Gus"], names);
            }
        }
        finally
        {
            Cleanup(dbPath);
        }
    }

    [Fact]
    public void SaveInMemoryToFile()
    {
        var dbPath = TempDbPath("saved.grafeo");
        try
        {
            using (var db = GrafeoDB.Memory())
            {
                db.Execute("INSERT (:City {name: 'Amsterdam'})");
                db.Execute("INSERT (:City {name: 'Berlin'})");
                db.Save(dbPath);
            }

            using (var db = GrafeoDB.Open(dbPath))
            {
                Assert.Equal(2, db.NodeCount);

                var result = db.Execute("MATCH (c:City) RETURN c.name ORDER BY c.name");
                var names = result.Rows
                    .Select(r => r["c.name"]?.ToString())
                    .Order()
                    .ToList();
                Assert.Equal(["Amsterdam", "Berlin"], names);
            }
        }
        finally
        {
            Cleanup(dbPath);
        }
    }

    [Fact]
    public void MultipleReopenCycles()
    {
        var dbPath = TempDbPath("cycles.grafeo");
        try
        {
            // Cycle 1
            using (var db = GrafeoDB.Open(dbPath))
                db.Execute("INSERT (:Person {name: 'Alix'})");

            // Cycle 2
            using (var db = GrafeoDB.Open(dbPath))
            {
                Assert.Equal(1, db.NodeCount);
                db.Execute("INSERT (:Person {name: 'Gus'})");
            }

            // Cycle 3
            using (var db = GrafeoDB.Open(dbPath))
            {
                Assert.Equal(2, db.NodeCount);
                db.Execute("INSERT (:Person {name: 'Vincent'})");
            }

            // Final check
            using (var db = GrafeoDB.Open(dbPath))
            {
                Assert.Equal(3, db.NodeCount);
                var result = db.Execute("MATCH (p:Person) RETURN p.name");
                var names = result.Rows
                    .Select(r => r["p.name"]?.ToString())
                    .Order()
                    .ToList();
                Assert.Equal(["Alix", "Gus", "Vincent"], names);
            }
        }
        finally
        {
            Cleanup(dbPath);
        }
    }

    [Fact]
    public void EdgePropertiesPersist()
    {
        var dbPath = TempDbPath("edgeprops.grafeo");
        try
        {
            using (var db = GrafeoDB.Open(dbPath))
            {
                db.Execute("INSERT (:Person {name: 'Alix'})");
                db.Execute("INSERT (:Person {name: 'Gus'})");
                db.Execute(
                    "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) " +
                    "INSERT (a)-[:KNOWS {since: 2020}]->(b)");
            }

            using (var db = GrafeoDB.Open(dbPath))
            {
                var result = db.Execute("MATCH ()-[e:KNOWS]->() RETURN e.since");
                Assert.Single(result.Rows);
                Assert.Equal(2020L, result.Rows[0]["e.since"]);
            }
        }
        finally
        {
            Cleanup(dbPath);
        }
    }

    [Fact]
    public void GrafeoFileIsSingleFile()
    {
        var dbPath = TempDbPath("single.grafeo");
        try
        {
            using (var db = GrafeoDB.Open(dbPath))
                db.Execute("INSERT (:Node {x: 1})");

            Assert.True(File.Exists(dbPath), ".grafeo path should be a file, not a directory");
            Assert.False(Directory.Exists(dbPath), ".grafeo path should not be a directory");
        }
        finally
        {
            Cleanup(dbPath);
        }
    }
}
