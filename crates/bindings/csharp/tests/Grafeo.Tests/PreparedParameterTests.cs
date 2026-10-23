using Xunit;

namespace Grafeo.Tests;

/// <summary>
/// A statement called again with other values answers for those values: the
/// engine optimizes it once with its parameters in and fills a copy of that
/// plan with each call's values, here through the C# binding.
/// </summary>
public sealed class PreparedParameterTests : IDisposable
{
    private readonly GrafeoDB _db = GrafeoDB.Memory();

    public PreparedParameterTests()
    {
        foreach (var (name, age) in new[] { ("Alix", 30L), ("Gus", 25L), ("Vincent", 41L), ("Mia", 25L) })
        {
            _db.ExecuteWithParams(
                "INSERT (:Person {name: $name, age: $age})",
                new Dictionary<string, object?> { ["name"] = name, ["age"] = age });
        }
    }

    public void Dispose() => _db.Dispose();

    private List<string> Column(QueryResult result, string name) =>
        result.Rows.Select(row => Convert.ToString(row[name], System.Globalization.CultureInfo.InvariantCulture)!).ToList();

    [Fact]
    public void ALookupTakesEachCallsValue()
    {
        const string query = "MATCH (p:Person) WHERE p.name = $name RETURN p.age AS age";
        foreach (var (name, ages) in new[]
        {
            ("Gus", new[] { "25" }),
            ("Alix", new[] { "30" }),
            ("Nobody", Array.Empty<string>()),
            ("Gus", new[] { "25" }),
        })
        {
            var result = _db.ExecuteWithParams(query, new Dictionary<string, object?> { ["name"] = name });
            Assert.Equal(ages, Column(result, "age"));
        }
    }

    [Fact]
    public void SkipAndLimitTakeEachCallsValues()
    {
        const string query = "MATCH (p:Person) RETURN p.name AS name ORDER BY p.name SKIP $skip LIMIT $limit";
        var names = new[] { "Alix", "Gus", "Mia", "Vincent" };
        foreach (var (skip, limit) in new[] { (0, 1), (1, 2), (3, 5), (0, 4) })
        {
            var result = _db.ExecuteWithParams(
                query,
                new Dictionary<string, object?> { ["skip"] = (long)skip, ["limit"] = (long)limit });
            Assert.Equal(names.Skip(skip).Take(limit), Column(result, "name"));
        }
    }

    [Fact]
    public void AValueOfAnotherTypeAnswersAsItsLiteral()
    {
        const string query = "MATCH (p:Person) WHERE p.age = $age RETURN p.name AS name";
        foreach (var (value, literal) in new (object, string)[] { (25L, "25"), ("25", "'25'"), (25L, "25"), (true, "true") })
        {
            var got = Column(_db.ExecuteWithParams(query, new Dictionary<string, object?> { ["age"] = value }), "name");
            var expected = Column(_db.Execute(query.Replace("$age", literal)), "name");
            got.Sort(StringComparer.Ordinal);
            expected.Sort(StringComparer.Ordinal);
            Assert.Equal(expected, got);
        }
    }
}
