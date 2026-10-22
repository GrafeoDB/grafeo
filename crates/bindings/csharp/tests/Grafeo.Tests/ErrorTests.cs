using Xunit;

namespace Grafeo.Tests;

/// <summary>
/// A database error carries its GQLSTATUS (#639): 22012 for a division by
/// zero, 22003 for a value out of range, 42001 for a syntax error.
/// </summary>
public sealed class ErrorTests : IDisposable
{
    private readonly GrafeoDB _db = GrafeoDB.Memory();

    public void Dispose() => _db.Dispose();

    [Theory]
    [InlineData("RETURN 1 / 0 AS x", "22012", "division by zero: 1 / 0")]
    [InlineData("RETURN 9223372036854775807 + 1 AS x", "22003", "numeric value out of range")]
    [InlineData("MATCH (n RETURN n", "42001", "GRAFEO-Q001")]
    public void ErrorsCarryTheirGqlStatus(string query, string status, string message)
    {
        var error = Assert.Throws<QueryException>(() => _db.Execute(query));
        Assert.Equal(status, error.GqlStatus);
        Assert.Contains(message, error.Message);
    }

    [Fact]
    public async Task AsyncErrorsCarryTheirGqlStatusToo()
    {
        var error = await Assert.ThrowsAsync<QueryException>(
            () => _db.ExecuteAsync("RETURN 1 % 0 AS x"));
        Assert.Equal("22012", error.GqlStatus);
    }
}
