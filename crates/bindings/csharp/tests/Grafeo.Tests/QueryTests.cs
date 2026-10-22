using Xunit;

namespace Grafeo.Tests;

/// <summary>GQL query execution and parameterized query tests.</summary>
public sealed class QueryTests : IDisposable
{
    private readonly GrafeoDB _db = GrafeoDB.Memory();

    public void Dispose() => _db.Dispose();

    [Fact]
    public void ExecutesSimpleReturn()
    {
        var result = _db.Execute("RETURN 1 AS x");
        Assert.Single(result.Rows);
        Assert.Equal(1L, result.Rows[0]["x"]);
    }

    [Fact]
    public void CreatesAndRetrievesNode()
    {
        _db.Execute("INSERT (:Person {name: 'Alix', age: 30})");

        var result = _db.Execute("MATCH (p:Person) WHERE p.name = 'Alix' RETURN p.name, p.age");

        Assert.Single(result.Rows);
        Assert.Equal("Alix", result.Rows[0]["p.name"]);
        Assert.Equal(30L, result.Rows[0]["p.age"]);
    }

    [Fact]
    public void HandlesParameterizedQueries()
    {
        _db.Execute("INSERT (:City {name: 'Amsterdam'})");

        var result = _db.ExecuteWithParams(
            "MATCH (c:City) WHERE c.name = $name RETURN c.name",
            new Dictionary<string, object?> { ["name"] = "Amsterdam" });

        Assert.Single(result.Rows);
        Assert.Equal("Amsterdam", result.Rows[0]["c.name"]);
    }

    [Fact]
    public void HandlesEmptyResult()
    {
        var result = _db.Execute("MATCH (n:NonExistent) RETURN n");
        Assert.Empty(result.Rows);
    }

    [Fact]
    public void ReturnsMultipleRows()
    {
        _db.Execute("INSERT (:Person {name: 'Mia'})");
        _db.Execute("INSERT (:Person {name: 'Butch'})");
        _db.Execute("INSERT (:Person {name: 'Django'})");

        var result = _db.Execute("MATCH (p:Person) RETURN p.name ORDER BY p.name");
        Assert.Equal(3, result.Rows.Count);
    }

    [Fact]
    public void ReturnsQueryMetrics()
    {
        _db.Execute("INSERT (:Person {name: 'Shosanna'})");
        var result = _db.Execute("MATCH (p:Person) RETURN p.name");

        Assert.True(result.ExecutionTimeMs >= 0);
    }

    [Fact]
    public void ThrowsOnInvalidQuery()
    {
        Assert.Throws<QueryException>(() => _db.Execute("THIS IS NOT VALID GQL"));
    }

    [Fact]
    public async Task ExecuteAsyncWorks()
    {
        _db.Execute("INSERT (:Person {name: 'Hans', age: 55})");

        var result = await _db.ExecuteAsync("MATCH (p:Person) RETURN p.name, p.age");

        Assert.Single(result.Rows);
        Assert.Equal("Hans", result.Rows[0]["p.name"]);
    }

    [Fact]
    public async Task ExecuteAsyncSupportsCancellation()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => _db.ExecuteAsync("RETURN 1", cts.Token));
    }

    [Fact]
    public void HandlesMultiplePropertyTypes()
    {
        _db.Execute("""
            INSERT (:Person {
                name: 'Beatrix',
                age: 35,
                active: true,
                score: 9.5
            })
        """);

        var result = _db.Execute(
            "MATCH (p:Person) WHERE p.name = 'Beatrix' RETURN p.name, p.age, p.active, p.score");

        Assert.Single(result.Rows);
        var row = result.Rows[0];
        Assert.Equal("Beatrix", row["p.name"]);
        Assert.Equal(35L, row["p.age"]);
        Assert.Equal(true, row["p.active"]);
        Assert.Equal(9.5, row["p.score"]);
    }
}
