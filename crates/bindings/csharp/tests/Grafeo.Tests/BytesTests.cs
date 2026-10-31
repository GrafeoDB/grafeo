using Xunit;

namespace Grafeo.Tests;

/// <summary>
/// A byte[] travels in the {"$bytes": "..."} tag (0.6.1): it arrives as BYTES
/// (a base64 string before, which arrived as STRING), and BYTES come back as a
/// byte[] with every byte equal: a few bytes, none, all 256 values. A string
/// that looks like base64 stays a string, and a $bytes object that is not
/// base64, or has a second key, stays a map.
/// </summary>
public sealed class BytesTests
{
    private static byte[] AllBytes() => Enumerable.Range(0, 256).Select(i => (byte)i).ToArray();

    public static TheoryData<byte[]> Values() => new() { new byte[] { 0, 1, 2, 255 }, Array.Empty<byte>(), AllBytes() };

    [Theory]
    [MemberData(nameof(Values))]
    public void BytesRoundTrip(byte[] value)
    {
        using var db = GrafeoDB.Memory();
        var row = db.ExecuteWithParams(
            "RETURN $b AS b, $b IS TYPED BYTES AS bytes, $b IS TYPED STRING AS str",
            new Dictionary<string, object?> { ["b"] = value }).Rows[0];
        Assert.Equal((true, false), ((bool)row["bytes"]!, (bool)row["str"]!));
        Assert.Equal(value, Assert.IsType<byte[]>(row["b"]));
    }

    [Fact]
    public void StoredBytesComeBackAsBytes()
    {
        using var db = GrafeoDB.Memory();
        var all = AllBytes();
        db.ExecuteWithParams(
            "INSERT (:Blob {name: 'Mia', data: $b, parts: $l})",
            new Dictionary<string, object?>
            {
                ["b"] = all,
                ["l"] = new List<object?> { new byte[] { 7 }, Array.Empty<byte>() },
            });
        var row = db.Execute("MATCH (n:Blob) RETURN n.data AS data, n.parts AS parts").Rows[0];
        Assert.Equal(all, Assert.IsType<byte[]>(row["data"]));
        var parts = Assert.IsAssignableFrom<IList<object?>>(row["parts"]);
        Assert.Equal(new byte[] { 7 }, Assert.IsType<byte[]>(parts[0]));
        Assert.Empty(Assert.IsType<byte[]>(parts[1]));

        var inner = db.ExecuteWithParams(
            "RETURN $m.k IS TYPED BYTES AS inner",
            new Dictionary<string, object?> { ["m"] = new Dictionary<string, object?> { ["k"] = new byte[] { 1, 2 } } }).Rows[0];
        Assert.True((bool)inner["inner"]!);
    }

    [Fact]
    public void AStringThatLooksLikeBase64StaysAString()
    {
        using var db = GrafeoDB.Memory();
        var row = db.ExecuteWithParams(
            "RETURN $s AS s, $s IS TYPED STRING AS str",
            new Dictionary<string, object?> { ["s"] = "AAEC/w==" }).Rows[0];
        Assert.Equal("AAEC/w==", row["s"]);
        Assert.True((bool)row["str"]!);
    }

    public static TheoryData<string, bool> Lookalikes() => new()
    {
        { "not base64!", false },
        { "AAEC/w", false },
        { "AAEC_w==", false },
        { "AAEC/x==", false },
        { "AAEC /w==", false },
        { "AAEC\n/w==", false },
        { "AAEC/w==", true },
    };

    [Theory]
    [MemberData(nameof(Lookalikes))]
    public void ABytesMapThatIsNoTagStaysAMap(string text, bool secondKey)
    {
        using var db = GrafeoDB.Memory();
        var map = new Dictionary<string, object?> { ["$bytes"] = text };
        if (secondKey)
            map["name"] = "Gus";
        var row = db.ExecuteWithParams(
            "RETURN $m AS m, $m IS TYPED BYTES AS bytes",
            new Dictionary<string, object?> { ["m"] = map }).Rows[0];
        Assert.False((bool)row["bytes"]!);
        var back = Assert.IsAssignableFrom<IReadOnlyDictionary<string, object?>>(row["m"]);
        Assert.Equal(text, back["$bytes"]);
        Assert.Equal(map.Count, back.Count);
    }
}
