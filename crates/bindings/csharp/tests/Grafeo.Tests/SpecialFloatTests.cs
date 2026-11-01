using Xunit;

namespace Grafeo.Tests;

/// <summary>
/// Floats keep their type through the JSON the binding sends (0.6.1): a whole
/// double (System.Text.Json writes 3.0 as "3") arrives as FLOAT64, not INT64;
/// NaN and the infinities, which JSON has no number for, travel as
/// {"$float": "NaN"} and come back as NaN and the infinities; and the string
/// "NaN" stays a string.
/// </summary>
public sealed class SpecialFloatTests
{
    private static IReadOnlyDictionary<string, object?> Typed(GrafeoDB db, object? value) =>
        db.ExecuteWithParams(
            "RETURN $x AS x, $x IS TYPED FLOAT64 AS float, $x IS TYPED STRING AS str",
            new Dictionary<string, object?> { ["x"] = value }).Rows[0];

    [Theory]
    [InlineData(3.0)]
    [InlineData(0.0)]
    [InlineData(-20.0)]
    [InlineData(1e6)]
    [InlineData(9223372036854775808.0)]
    public void AWholeDoubleIsFloat64(double value)
    {
        using var db = GrafeoDB.Memory();
        var row = Typed(db, value);
        Assert.True((bool)row["float"]!, $"{value} arrived as {row["x"]} ({row["x"]?.GetType().Name})");
        Assert.Equal(value, Convert.ToDouble(row["x"]));
    }

    [Fact]
    public void AWholeSingleIsFloat64()
    {
        using var db = GrafeoDB.Memory();
        Assert.True((bool)Typed(db, 3f)["float"]!);
    }

    [Fact]
    public void WholeDoublesInsideAListAndAMapAreFloat64()
    {
        using var db = GrafeoDB.Memory();
        var row = db.ExecuteWithParams(
            "RETURN $xs[0] IS TYPED FLOAT64 AS first, $m.w IS TYPED FLOAT64 AS w",
            new Dictionary<string, object?>
            {
                ["xs"] = new List<object?> { 3.0, 1.5 },
                ["m"] = new Dictionary<string, object?> { ["w"] = 20.0 },
            }).Rows[0];
        Assert.True((bool)row["first"]!);
        Assert.True((bool)row["w"]!);
    }

    public static TheoryData<double> Special() => new() { double.NaN, double.PositiveInfinity, double.NegativeInfinity };

    [Theory]
    [MemberData(nameof(Special))]
    public void NanAndTheInfinitiesRoundTrip(double value)
    {
        using var db = GrafeoDB.Memory();
        var row = Typed(db, value);
        Assert.True((bool)row["float"]!);
        var back = Assert.IsType<double>(row["x"]);
        Assert.True(back.Equals(value), $"{value} came back as {back}");
    }

    [Fact]
    public void ANanPropertyComesBackAsNan()
    {
        using var db = GrafeoDB.Memory();
        db.ExecuteWithParams(
            "INSERT (:Reading {v: $v})",
            new Dictionary<string, object?> { ["v"] = double.NaN });
        var back = Assert.IsType<double>(db.Execute("MATCH (r:Reading) RETURN r.v AS v").Rows[0]["v"]);
        Assert.True(double.IsNaN(back));
    }

    [Theory]
    [InlineData("NaN")]
    [InlineData("Infinity")]
    [InlineData("-Infinity")]
    public void TheStringNanStaysAString(string text)
    {
        using var db = GrafeoDB.Memory();
        var row = Typed(db, text);
        Assert.Equal(text, row["x"]);
        Assert.Equal((false, true), ((bool)row["float"]!, (bool)row["str"]!));
    }
}
