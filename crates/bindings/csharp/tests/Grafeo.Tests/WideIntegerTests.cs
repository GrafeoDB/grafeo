using System.Numerics;
using Xunit;

namespace Grafeo.Tests;

/// <summary>
/// An integer parameter or property past 64 bits is an exact decimal, up to
/// 38 digits: one that fits 64 bits stays INT64, one past it but within 38
/// digits arrives as a DECIMAL with scale 0 (a ulong, Int128, UInt128 or
/// BigInteger used to arrive as a string), and one past 38 digits is an error
/// naming the parameter and the value.
/// </summary>
public sealed class WideIntegerTests
{
    private static IReadOnlyDictionary<string, object?> Typed(GrafeoDB db, object value) =>
        db.ExecuteWithParams(
            "RETURN $x AS x, $x IS TYPED INT64 AS int, $x IS TYPED DECIMAL AS decimal",
            new Dictionary<string, object?> { ["x"] = value }).Rows[0];

    public static TheoryData<object, string> Wide() => new()
    {
        { 9223372036854775808UL, "9223372036854775808" },
        { ulong.MaxValue, "18446744073709551615" },
        { BigInteger.Pow(2, 64), "18446744073709551616" },
        { -(BigInteger.Pow(2, 63) + 1), "-9223372036854775809" },
        { Int128.MaxValue / 2, (Int128.MaxValue / 2).ToString() },
        { BigInteger.Pow(10, 38) - 1, new string('9', 38) },
    };

    [Theory]
    [MemberData(nameof(Wide))]
    public void AnIntegerPast64BitsIsAnExactDecimal(object value, string digits)
    {
        using var db = GrafeoDB.Memory();
        var row = Typed(db, value);
        Assert.Equal(new GrafeoDecimal(digits), row["x"]);
        Assert.Equal((false, true), ((bool)row["int"]!, (bool)row["decimal"]!));
    }

    [Fact]
    public void EveryIntegerTypeThatFits64BitsIsInt64()
    {
        using var db = GrafeoDB.Memory();
        foreach (var value in new object[] { (byte)5, (sbyte)-5, (short)7, (ushort)7, 5u, 5UL, (Int128)5, (BigInteger)5, long.MaxValue })
        {
            var row = Typed(db, value);
            Assert.True((bool)row["int"]!, $"{value.GetType().Name} {value}");
        }
    }

    [Fact]
    public void AnIntegerPast38DigitsIsAnErrorNamingTheParameter()
    {
        using var db = GrafeoDB.Memory();
        var tooWide = BigInteger.Pow(10, 38);
        var error = Assert.ThrowsAny<GrafeoException>(() => Typed(db, tooWide));
        Assert.Contains("parameter 'x'", error.Message);
        Assert.Contains($"{tooWide}", error.Message);
        Assert.Contains("38 digits", error.Message);
    }

    [Fact]
    public void ArithmeticOverInt64AndDecimalPropertiesStaysExact()
    {
        using var db = GrafeoDB.Memory();
        db.CreateNode(["N"], new Dictionary<string, object?> { ["v"] = 5L });
        db.CreateNode(["N"], new Dictionary<string, object?> { ["v"] = BigInteger.Pow(2, 64) });
        var total = db.Execute("MATCH (n:N) RETURN sum(n.v) AS total").Rows[0]["total"];
        Assert.Equal(new GrafeoDecimal("18446744073709551621"), total);
        var error = Assert.ThrowsAny<GrafeoException>(() =>
            db.CreateNode(["N"], new Dictionary<string, object?> { ["v"] = BigInteger.Pow(10, 38) }));
        Assert.Contains("property 'v'", error.Message);
    }
}
