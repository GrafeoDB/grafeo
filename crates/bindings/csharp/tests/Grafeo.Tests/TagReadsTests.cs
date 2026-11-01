using Xunit;

namespace Grafeo.Tests;

/// <summary>
/// Every read decodes the tags a value travels in ($bytes, $decimal, $float
/// and the temporal ones): query rows, streamed rows, nodes, edges and the
/// reads of an AsOfView. A row is a map of columns, never a tagged value, and
/// a tag is the map's only key: a map with a tag key and a second key is a
/// map.
/// </summary>
public sealed class TagReadsTests : IDisposable
{
    private readonly GrafeoDB _db = GrafeoDB.Memory(HistoryRetention.Forever);
    private static readonly byte[] Data = { 0, 1, 2, 255 };

    public void Dispose() => _db.Dispose();

    private static void AssertDecoded(string where, IReadOnlyDictionary<string, object?> values)
    {
        Assert.True(values["data"] is byte[] data && data.SequenceEqual(Data), $"{where}: data {values["data"]}");
        Assert.True(
            values["amount"] is GrafeoDecimal amount && amount.Equals(new GrafeoDecimal("12.5")),
            $"{where}: amount {values["amount"]}");
        Assert.True(values["ratio"] is double ratio && double.IsNaN(ratio), $"{where}: ratio {values["ratio"]}");
    }

    private (long Node, long Edge) StoreTagged()
    {
        var row = _db.ExecuteWithParams(
            "INSERT (a:Blob {name: 'Mia', data: $data, amount: $amount, ratio: $ratio})"
            + "-[e:HOLDS {data: $data, amount: $amount, ratio: $ratio}]->(:Blob {name: 'Butch'}) "
            + "RETURN id(a) AS node, id(e) AS edge",
            new Dictionary<string, object?>
            {
                ["data"] = Data,
                ["amount"] = new GrafeoDecimal("12.50"),
                ["ratio"] = double.NaN,
            }).Rows[0];
        return ((long)row["node"]!, (long)row["edge"]!);
    }

    private const string TaggedQuery =
        "MATCH (n:Blob {name: 'Mia'}) RETURN n.data AS data, n.amount AS amount, n.ratio AS ratio";

    [Fact]
    public void QueryRowsNodesAndEdgesDecodeTags()
    {
        var (node, edge) = StoreTagged();
        AssertDecoded("Execute", _db.Execute(TaggedQuery).Rows[0]);
        AssertDecoded("GetNode", _db.GetNode(node)!.Properties);
        AssertDecoded("GetEdge", _db.GetEdge(edge)!.Properties);
    }

    [Fact]
    public void StreamedRowsDecodeTags()
    {
        StoreTagged();
        using var stream = _db.ExecuteStream(TaggedQuery);
        var rows = stream.ToList();
        AssertDecoded("ExecuteStream", Assert.Single(rows));
    }

    [Fact]
    public void AsOfReadsDecodeTags()
    {
        var (node, edge) = StoreTagged();
        var then = _db.CurrentEpoch;
        _db.Execute("MATCH (n:Blob {name: 'Mia'})-[e:HOLDS]->() SET n.data = NULL, e.data = NULL");

        using var view = _db.AsOf(TimePoint.Epoch(then));
        AssertDecoded("AsOfView.GetNode", view.GetNode(node)!.Properties);
        AssertDecoded("AsOfView.GetEdge", view.GetEdge(edge)!.Properties);
        AssertDecoded("AsOfView.Execute", view.Execute(TaggedQuery).Rows[0]);
        var properties = new[] { "data", "amount", "ratio" }
            .ToDictionary(key => key, key => view.GetNodeProperty(node, key));
        AssertDecoded("AsOfView.GetNodeProperty", properties);
    }

    public static TheoryData<string, string, object> RowsNamedLikeTags() => new()
    {
        { "$bytes", "'AQID'", "AQID" },
        { "$decimal", "'1.50'", "1.50" },
        { "$date", "'2024-01-02'", "2024-01-02" },
        { "$float", "'NaN'", "NaN" },
        { "$timestamp_us", "5", 5L },
        { "$duration", "'P1D'", "P1D" },
    };

    /// <summary>A one-column row named like a tag is a row, through Execute and a stream.</summary>
    [Theory]
    [MemberData(nameof(RowsNamedLikeTags))]
    public void ARowNamedLikeATagStaysARow(string column, string literal, object value)
    {
        var query = $"RETURN {literal} AS `{column}`";
        var row = Assert.Single(_db.Execute(query).Rows);
        Assert.Equal(new Dictionary<string, object?> { [column] = value }, row);

        using var stream = _db.ExecuteStream(query);
        var streamed = Assert.Single(stream.ToList());
        Assert.Single(streamed);
        // The value and its type: a `$timestamp_us` row stays a long
        Assert.Equal(value, streamed[column]);
    }

    public static TheoryData<string> TagKeys() => new()
    {
        "$date", "$time", "$duration", "$zoned_datetime", "$timestamp_us", "$bytes", "$decimal", "$float",
    };

    /// <summary>A map with a tag key and a second key is a map, not the tagged value.</summary>
    [Theory]
    [MemberData(nameof(TagKeys))]
    public void AMapWithATagKeyAndASecondKeyStaysAMap(string key)
    {
        object tagValue = key == "$timestamp_us" ? 5L : "2024-01-02";
        var map = new Dictionary<string, object?> { [key] = tagValue, ["name"] = "Gus" };
        var row = _db.ExecuteWithParams("RETURN $m AS m", new Dictionary<string, object?> { ["m"] = map }).Rows[0];
        var back = Assert.IsAssignableFrom<IReadOnlyDictionary<string, object?>>(row["m"]);
        Assert.Equal(2, back.Count);
        Assert.Equal(tagValue, back[key]);
        Assert.Equal("Gus", back["name"]);
    }

    /// <summary>A temporal tag key holding a value of another type is a map, never an exception.</summary>
    [Fact]
    public void ATimestampTagThatIsNoNumberStaysAMap()
    {
        var map = new Dictionary<string, object?> { ["$timestamp_us"] = "soon" };
        var row = _db.ExecuteWithParams("RETURN $m AS m", new Dictionary<string, object?> { ["m"] = map }).Rows[0];
        var back = Assert.IsAssignableFrom<IReadOnlyDictionary<string, object?>>(row["m"]);
        Assert.Equal("soon", back["$timestamp_us"]);
    }

    /// <summary>A timestamp a DateTime cannot hold stays a map, and the rest of the result reads.</summary>
    [Theory]
    [InlineData(9_000_000_000_000_000_000L)]
    [InlineData(-9_000_000_000_000_000_000L)]
    public void ATimestampOutsideTheDateTimeRangeStaysAMap(long microseconds)
    {
        var map = new Dictionary<string, object?> { ["$timestamp_us"] = microseconds };
        var row = _db.ExecuteWithParams(
            "RETURN $m AS m, 'Gus' AS name",
            new Dictionary<string, object?> { ["m"] = map }).Rows[0];
        var back = Assert.IsAssignableFrom<IReadOnlyDictionary<string, object?>>(row["m"]);
        Assert.Equal(microseconds, back["$timestamp_us"]);
        Assert.Equal("Gus", row["name"]);
    }

    /// <summary>A timestamp a DateTime can hold, at both ends of its range, is a DateTime, to the microsecond.</summary>
    [Fact]
    public void TimestampsAtTheEndsOfTheDateTimeRangeAreDateTimes()
    {
        foreach (var at in new[] { DateTime.MinValue, DateTime.MaxValue.AddTicks(-9) })
        {
            var microseconds = (at.Ticks - DateTime.UnixEpoch.Ticks) / 10;
            var map = new Dictionary<string, object?> { ["$timestamp_us"] = microseconds };
            var row = _db.ExecuteWithParams("RETURN $m AS m", new Dictionary<string, object?> { ["m"] = map }).Rows[0];
            Assert.Equal(DateTime.SpecifyKind(at, DateTimeKind.Utc), row["m"]);
        }
    }
}
