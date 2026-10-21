using Xunit;

namespace Grafeo.Tests;

/// <summary>Time travel: reads of the database as it was at an earlier epoch or time.</summary>
public sealed class TimeTravelTests : IDisposable
{
    private readonly GrafeoDB _db = GrafeoDB.Memory(HistoryRetention.Forever);

    public TimeTravelTests() => _db.Execute("INSERT (:Person {name: 'Alix'})");

    public void Dispose() => _db.Dispose();

    private static List<object?> People(QueryResult result) =>
        result.Rows.Select(row => row["name"]).ToList();

    private const string PeopleQuery = "MATCH (p:Person) RETURN p.name AS name ORDER BY name";

    [Fact]
    public void AsOfAnEpochReadsTheStateThen()
    {
        var then = _db.CurrentEpoch;
        _db.Execute("INSERT (:Person {name: 'Gus'})");
        using var view = _db.AsOf(TimePoint.Epoch(then));
        Assert.Equal(then, view.Epoch);
        Assert.Equal(["Alix"], People(view.Execute(PeopleQuery)));
        Assert.Equal(["Alix", "Gus"], People(_db.Execute(PeopleQuery)));
    }

    [Fact]
    public void TheViewRefusesAWrite()
    {
        using var view = _db.AsOf(TimePoint.Epoch(_db.CurrentEpoch));
        Assert.ThrowsAny<GrafeoException>(() => view.Execute("INSERT (:Person {name: 'Mia'})"));
        Assert.Equal(["Alix"], People(_db.Execute(PeopleQuery)));
    }

    [Fact]
    public void TheViewReadsNodesEdgesAndPropertiesAsTheyWere()
    {
        var ids = _db.Execute(
            "MATCH (a:Person {name: 'Alix'}) INSERT (a)-[k:KNOWS {since: 2020}]->(:Person {name: 'Gus'}) RETURN id(a) AS a, id(k) AS k");
        var (alix, knows) = ((long)ids.Rows[0]["a"]!, (long)ids.Rows[0]["k"]!);
        var then = TimePoint.Epoch(_db.CurrentEpoch);
        _db.Execute("MATCH (p:Person {name: 'Alix'}) SET p.name = 'Alix B.'");
        _db.Execute("MATCH ()-[k:KNOWS]->() DELETE k");

        using var view = _db.AsOf(then);
        Assert.Equal("Alix", view.GetNode(alix)!.Properties["name"]);
        Assert.Equal("Alix", view.GetNodeProperty(alix, "name"));
        Assert.Null(view.GetNodeProperty(alix, "city"));
        Assert.Null(view.GetNode(alix + 1000));
        Assert.Equal(2020L, view.GetEdge(knows)!.Properties["since"]);
        using var now = _db.AsOf(TimePoint.Ago(TimeSpan.Zero));
        Assert.Null(now.GetEdge(knows));
        Assert.Equal("Alix B.", now.GetNodeProperty(alix, "name"));
    }

    [Fact]
    public void ExecuteAsOfTakesParameters()
    {
        var then = TimePoint.Epoch(_db.CurrentEpoch);
        _db.Execute("INSERT (:Person {name: 'Gus'})");
        var parameters = new Dictionary<string, object?> { ["not"] = "Vincent" };
        var result = _db.ExecuteAsOf("MATCH (p:Person) WHERE p.name <> $not RETURN count(p) AS n", then, parameters);
        Assert.Equal(1L, result.Rows[0]["n"]);
        using var view = _db.AsOf(then);
        Assert.Equal(["Alix"], People(view.ExecuteWithParams(
            "MATCH (p:Person) WHERE p.name <> $not RETURN p.name AS name", parameters)));
    }

    [Fact]
    public void ATimeReadsTheLastCommitAtOrBeforeIt()
    {
        Thread.Sleep(10);
        var beforeGus = DateTimeOffset.UtcNow;
        // Commit times have the resolution of a millisecond: Gus commits after it.
        Thread.Sleep(10);
        _db.Execute("INSERT (:Person {name: 'Gus'})");
        using (var view = _db.AsOf(TimePoint.At(beforeGus)))
            Assert.Equal(["Alix"], People(view.Execute(PeopleQuery)));
        // Another offset names the same instant.
        Assert.Equal(
            _db.EpochAsOf(TimePoint.At(beforeGus)),
            _db.EpochAsOf(TimePoint.At(beforeGus.ToOffset(TimeSpan.FromHours(2)))));
        Assert.Equal(_db.CurrentEpoch, _db.EpochAsOf(TimePoint.At(DateTimeOffset.UtcNow.AddSeconds(1))));
    }

    [Fact]
    public void AgoIsADurationBeforeNow()
    {
        Assert.Equal(_db.CurrentEpoch, _db.EpochAsOf(TimePoint.Ago(TimeSpan.Zero)));
    }

    [Fact]
    public void ALaterEpochReadsNow()
    {
        Assert.Equal(_db.CurrentEpoch, _db.EpochAsOf(TimePoint.Epoch(_db.CurrentEpoch + 1000)));
    }

    [Fact]
    public void NegativeAndOutOfRangePointsThrow()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => TimePoint.Ago(TimeSpan.FromSeconds(-1)));
        Assert.Throws<ArgumentOutOfRangeException>(() => TimePoint.Epoch(ulong.MaxValue));
        Assert.Throws<ArgumentOutOfRangeException>(() => HistoryRetention.For(TimeSpan.FromHours(-1)));
    }

    [Fact]
    public void ATimeWithoutHistoryThrows()
    {
        using var db = GrafeoDB.Memory();
        var error = Assert.ThrowsAny<GrafeoException>(
            () => db.AsOf(TimePoint.At(new DateTimeOffset(2000, 1, 1, 0, 0, 0, TimeSpan.Zero))));
        Assert.Contains("needs history", error.Message);
    }

    [Fact]
    public void ATimeBeforeTheHistoryIsNotRetained()
    {
        using var db = GrafeoDB.Memory(HistoryRetention.For(TimeSpan.FromHours(1)));
        db.Execute("INSERT (:Person {name: 'Alix'})");
        var error = Assert.ThrowsAny<GrafeoException>(
            () => db.EpochAsOf(TimePoint.At(new DateTimeOffset(2000, 1, 1, 0, 0, 0, TimeSpan.Zero))));
        Assert.Contains("not retained", error.Message);
    }

    [Fact]
    public void ADisposedViewThrows()
    {
        var view = _db.AsOf(TimePoint.Epoch(_db.CurrentEpoch));
        view.Dispose();
        view.Dispose();
        Assert.Throws<ObjectDisposedException>(() => view.Execute("MATCH (n) RETURN n"));
    }

    [Fact]
    public void TheFileStoresTheRetention()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"grafeo-csharp-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        var path = Path.Combine(dir, "history.grafeo");
        try
        {
            ulong then;
            using (var db = GrafeoDB.Open(path, HistoryRetention.Forever))
            {
                db.Execute("INSERT (:Person {name: 'Alix'})");
                then = db.CurrentEpoch;
                db.Execute("INSERT (:Person {name: 'Gus'})");
            }
            using (var db = GrafeoDB.Open(path))
            using (var view = db.AsOf(TimePoint.Epoch(then)))
                Assert.Equal(["Alix"], People(view.Execute(PeopleQuery)));
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); }
            catch (IOException)
            {
                // Best-effort cleanup in tests.
            }
        }
    }
}
