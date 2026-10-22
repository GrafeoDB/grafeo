using Xunit;

namespace Grafeo.Tests;

/// <summary>Exact decimals cross the C# binding as <see cref="GrafeoDecimal"/> (#610).</summary>
public sealed class DecimalTests : IDisposable
{
    private readonly GrafeoDB _db = GrafeoDB.Memory();

    public void Dispose() => _db.Dispose();

    [Fact]
    public void ADecimalParameterComesBackWithItsScale()
    {
        var result = _db.ExecuteWithParams(
            "RETURN $a + $b AS total",
            new Dictionary<string, object?>
            {
                ["a"] = new GrafeoDecimal("0.10"),
                ["b"] = new GrafeoDecimal("0.2"),
            });

        var total = Assert.IsType<GrafeoDecimal>(result.Rows[0]["total"]);
        Assert.Equal("0.30", total.ToString());
        Assert.Equal(0.30m, total.ToDecimal());
    }

    [Fact]
    public void ANativeDecimalParameterIsExact()
    {
        var result = _db.ExecuteWithParams(
            "RETURN $price * 3 AS total",
            new Dictionary<string, object?> { ["price"] = 19.99m });

        var total = Assert.IsType<GrafeoDecimal>(result.Rows[0]["total"]);
        Assert.Equal("59.97", total.ToString());
    }

    [Fact]
    public void DecimalsAreEqualByValueWhateverTheirScale()
    {
        Assert.Equal(new GrafeoDecimal("1.50"), new GrafeoDecimal("1.5"));
        Assert.Equal(new GrafeoDecimal("1.50").GetHashCode(), new GrafeoDecimal("1.5").GetHashCode());
        Assert.Equal(new GrafeoDecimal("-0.00"), new GrafeoDecimal("0"));
        Assert.NotEqual(new GrafeoDecimal("1.5"), new GrafeoDecimal("1.05"));
    }

    [Fact]
    public void ADecimalPastTwentyEightDigitsStaysExact()
    {
        var wide = new GrafeoDecimal("12345678901234567890.123456789012345678");
        Assert.False(wide.TryToDecimal(out _));
        var result = _db.ExecuteWithParams(
            "RETURN $x AS v",
            new Dictionary<string, object?> { ["x"] = wide });
        Assert.Equal(wide.ToString(), result.Rows[0]["v"]?.ToString());
    }

    [Fact]
    public void TextThatIsNotADecimalIsRefused()
    {
        Assert.Throws<ArgumentException>(() => new GrafeoDecimal("1.2.3"));
        Assert.Throws<ArgumentException>(() => new GrafeoDecimal("1e5"));
    }
}
