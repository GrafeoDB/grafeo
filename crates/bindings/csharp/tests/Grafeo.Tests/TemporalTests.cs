using Xunit;

namespace Grafeo.Tests;

/// <summary>
/// Temporal values cross the C# binding as their GQL types, both ways: a
/// value read back goes in again as the type it came as.
/// </summary>
public sealed class TemporalTests : IDisposable
{
    private readonly GrafeoDB _db = GrafeoDB.Memory();

    public void Dispose() => _db.Dispose();

    /// <summary>Runs <c>RETURN $v IS TYPED type, $v</c>: whether it arrived as the type, and what came back.</summary>
    private (bool Typed, object? Back) TypedBack(string gqlType, object? value)
    {
        var row = _db.ExecuteWithParams(
            $"RETURN $v IS TYPED {gqlType} AS typed, $v AS back",
            new Dictionary<string, object?> { ["v"] = value }).Rows[0];
        return ((bool)row["typed"]!, row["back"]);
    }

    public static TheoryData<string, object> Values() => new()
    {
        { "LOCAL DATETIME", new DateTime(2024, 3, 5, 10, 30, 0, DateTimeKind.Utc).AddTicks(1_234_560) },
        { "TIMESTAMP", new DateTime(1969, 7, 20, 20, 17, 40, DateTimeKind.Utc).AddTicks(6_543_210) },
        { "ZONED DATETIME", new DateTimeOffset(2024, 3, 5, 10, 30, 0, 250, TimeSpan.FromHours(1)) },
        { "ZONED DATETIME", new DateTimeOffset(2024, 3, 5, 10, 30, 0, TimeSpan.Zero) },
        { "ZONED DATETIME", new DateTimeOffset(2026, 10, 18, 14, 30, 0, TimeSpan.FromHours(2)) },
        { "ZONED DATETIME", new DateTimeOffset(1969, 7, 20, 20, 17, 40, TimeSpan.FromMinutes(-330)).AddTicks(10) },
        { "DATE", new DateOnly(2024, 2, 29) },
        { "TIME", new TimeOnly(14, 30, 15).Add(TimeSpan.FromTicks(1_234_567)) },
        { "TIME", new GrafeoTime(12, 30, 0, 0, TimeSpan.FromHours(2)) },
        { "TIME", new GrafeoTime(23, 59, 59, 999_999_999) },
        { "TIME", new GrafeoTime(0, 0, 0, 0, TimeSpan.Zero) },
        { "DURATION", new GrafeoDuration(14, 3, 14_706_500_000_000) },
        { "DURATION", new GrafeoDuration(-1, -2, -3) },
        { "DURATION", new GrafeoDuration(0, 1, 0) },
        { "DURATION", new GrafeoDuration(0, 0, 0) },
    };

    /// <summary>Each value goes in as its type, comes back as it went in, and goes in again as its type.</summary>
    [Theory]
    [MemberData(nameof(Values))]
    public void AValueRoundTripsAsItsType(string gqlType, object value)
    {
        var (typed, back) = TypedBack(gqlType, value);
        Assert.True(typed, $"{value} did not arrive as {gqlType}");
        Assert.Equal(value.GetType(), back?.GetType());
        Assert.Equal(value, back);
        if (value is DateTime at)
            Assert.Equal(at.Kind, ((DateTime)back!).Kind);
        if (value is DateTimeOffset zoned)
            Assert.Equal(zoned.Offset, ((DateTimeOffset)back!).Offset);
        Assert.True(TypedBack(gqlType, back).Typed, $"{back}, read back, did not arrive as {gqlType} again");
    }

    /// <summary>Microseconds are kept, also before 1970, where a finer part is dropped downwards.</summary>
    [Fact]
    public void ATimestampKeepsItsMicroseconds()
    {
        var at = new DateTime(1969, 12, 31, 23, 59, 59, DateTimeKind.Utc).AddTicks(9_999_999);
        var row = _db.ExecuteWithParams(
            "RETURN $t AS t, $t = local_datetime('1969-12-31T23:59:59.999999') AS exact",
            new Dictionary<string, object?> { ["t"] = at }).Rows[0];
        Assert.Equal(true, row["exact"]);
        Assert.Equal(at.AddTicks(-9), row["t"]);
        Assert.Equal(DateTimeKind.Utc, ((DateTime)row["t"]!).Kind);
    }

    /// <summary>A local DateTime is the same instant as a TIMESTAMP, and comes back in UTC.</summary>
    [Fact]
    public void ALocalDateTimeComesBackInUtc()
    {
        var local = new DateTime(2024, 3, 5, 10, 30, 0, DateTimeKind.Local);
        var (typed, back) = TypedBack("TIMESTAMP", local);
        Assert.True(typed);
        Assert.Equal(local.ToUniversalTime(), back);
        Assert.Equal(DateTimeKind.Utc, ((DateTime)back!).Kind);
    }

    /// <summary>A TimeSpan is a DURATION of its days and the rest; one without months converts back.</summary>
    [Fact]
    public void ATimeSpanIsADuration()
    {
        var span = new TimeSpan(1, 2, 3, 4).Add(TimeSpan.FromTicks(5));
        var (typed, back) = TypedBack("DURATION", span);
        Assert.True(typed);
        var duration = Assert.IsType<GrafeoDuration>(back);
        Assert.Equal(new GrafeoDuration(0, 1, (span.Ticks - TimeSpan.TicksPerDay) * 100), duration);
        Assert.Equal("P1DT2H3M4.0000005S", duration.Text);
        Assert.Equal(span, duration.ToTimeSpan());
        Assert.Equal(-span, GrafeoDuration.FromTimeSpan(-span).ToTimeSpan());
        Assert.False(new GrafeoDuration(1, 0, 0).TryToTimeSpan(out _));
        Assert.False(new GrafeoDuration(0, 0, 1).TryToTimeSpan(out _));
        Assert.Throws<InvalidOperationException>(() => new GrafeoDuration(1, 0, 0).ToTimeSpan());
    }

    /// <summary>Values a query makes come back typed, alone and inside lists and maps, and go in again as their types.</summary>
    [Fact]
    public void QueryValuesComeBackTypedInListsAndMaps()
    {
        var row = _db.Execute(
            "RETURN DATE '2024-03-05' AS d, TIME '12:30:00+02:00' AS zt, TIME '08:15:30.25' AS lt, " +
            "TIME '08:15:30.123456789' AS fine, DURATION 'P1Y2M3DT4H5M6.5S' AS dur, " +
            "ZONED DATETIME '2024-03-05T10:30:00.5+05:30' AS zdt, " +
            "local_datetime('2024-03-05T10:30:00.000001') AS ldt, " +
            "[DATE '2024-03-05', DATE '2024-03-06'] AS dates, " +
            "{at: TIME '12:30:00Z', took: DURATION 'PT1M'} AS m").Rows[0];
        Assert.Equal(new DateOnly(2024, 3, 5), row["d"]);
        Assert.Equal(new GrafeoTime(12, 30, 0, 0, TimeSpan.FromHours(2)), row["zt"]);
        Assert.Equal(new TimeOnly(8, 15, 30, 250), row["lt"]);
        // Finer than TimeOnly's 100 ns: a GrafeoTime, exactly
        Assert.Equal(new GrafeoTime(8, 15, 30, 123_456_789), row["fine"]);
        Assert.Equal(new GrafeoDuration(14, 3, 14_706_500_000_000), row["dur"]);
        var zdt = Assert.IsType<DateTimeOffset>(row["zdt"]);
        Assert.Equal(TimeSpan.FromMinutes(330), zdt.Offset);
        Assert.Equal(new DateTime(2024, 3, 5, 5, 0, 0, 500, DateTimeKind.Utc), zdt.UtcDateTime);
        Assert.Equal(new DateTime(2024, 3, 5, 10, 30, 0, DateTimeKind.Utc).AddTicks(10), row["ldt"]);
        var dates = Assert.IsType<List<object?>>(row["dates"]);
        Assert.Equal(new object?[] { new DateOnly(2024, 3, 5), new DateOnly(2024, 3, 6) }, dates);
        var m = Assert.IsType<Dictionary<string, object?>>(row["m"]);
        Assert.Equal(new GrafeoTime(12, 30, 0, 0, TimeSpan.Zero), m["at"]);
        Assert.Equal("12:30:00Z", m["at"]!.ToString());
        Assert.Equal(new GrafeoDuration(0, 0, 60_000_000_000), m["took"]);

        var again = _db.ExecuteWithParams(
            "RETURN $d IS TYPED DATE AND $zt IS TYPED TIME AND $lt IS TYPED TIME AND $fine IS TYPED TIME " +
            "AND $dur IS TYPED DURATION AND $zdt IS TYPED ZONED DATETIME AND $ldt IS TYPED LOCAL DATETIME " +
            "AND $dates IS TYPED LIST<DATE> AND $m.at IS TYPED TIME AND $m.took IS TYPED DURATION AS typed, " +
            "$zt = TIME '12:30:00+02:00' AND $lt = TIME '08:15:30.25' AND $fine = TIME '08:15:30.123456789' " +
            "AND $zdt = ZONED DATETIME '2024-03-05T10:30:00.5+05:30' AS equal",
            row.ToDictionary(entry => entry.Key, entry => entry.Value)).Rows[0];
        Assert.Equal(true, again["typed"]);
        Assert.Equal(true, again["equal"]);
    }

    /// <summary>Stored temporal properties come back typed from GetNode and a stream.</summary>
    [Fact]
    public void StoredPropertiesComeBackTyped()
    {
        var born = new DateOnly(1994, 5, 12);
        var wakes = new GrafeoTime(7, 30, 0, 0, TimeSpan.FromHours(1));
        var seen = new DateTime(2024, 3, 5, 10, 30, 0, DateTimeKind.Utc).AddTicks(10);
        var id = (long)_db.ExecuteWithParams(
            "INSERT (p:Person {name: 'Shosanna', born: $born, wakes: $wakes, seen: $seen}) RETURN id(p) AS id",
            new Dictionary<string, object?>
            {
                ["born"] = born,
                ["wakes"] = wakes,
                ["seen"] = new List<object?> { seen },
            }).Rows[0]["id"]!;

        var properties = _db.GetNode(id)!.Properties;
        Assert.Equal(born, properties["born"]);
        Assert.Equal(wakes, properties["wakes"]);
        Assert.Equal(new List<object?> { seen }, properties["seen"]);

        using var stream = _db.ExecuteStream("MATCH (p:Person) RETURN p.born AS born, p.wakes AS wakes");
        var streamed = Assert.Single(stream.ToList());
        Assert.Equal(born, streamed["born"]);
        Assert.Equal(wakes, streamed["wakes"]);
    }

    /// <summary>The texts agree with the engine's: each reads to its value and writes back as the same text.</summary>
    [Theory]
    [InlineData("14:30:00")]
    [InlineData("14:30:00.5")]
    [InlineData("00:00:00.000000001")]
    [InlineData("12:30:00Z")]
    [InlineData("12:30:00+02:00")]
    [InlineData("12:30:00-05:30")]
    public void TimeTextsRoundTrip(string text) => Assert.Equal(text, new GrafeoTime(text).Text);

    [Theory]
    [InlineData("24:00:00")]
    [InlineData("12:60:00")]
    [InlineData("12:30:00.")]
    [InlineData("12")]
    [InlineData("12:30:00+02:60")]
    [InlineData("12:30:00+24:00")]
    [InlineData("12:30:00+99:59")]
    [InlineData("12:30:00+-1:00")]
    [InlineData("noon")]
    public void TextThatIsNotATimeIsRefused(string text)
    {
        Assert.False(GrafeoTime.TryParse(text, out _));
        Assert.Throws<ArgumentException>(() => new GrafeoTime(text));
    }

    [Theory]
    [InlineData("P1Y2M3DT4H5M6.5S")]
    [InlineData("PT0S")]
    [InlineData("-P1M2DT0.000000003S")]
    [InlineData("P1D")]
    [InlineData("PT24H")]
    [InlineData("P1M-2D")]
    [InlineData("P1MT-1.5S")]
    [InlineData("P1MT-0.5S")]
    [InlineData("P1MT-1S")]
    [InlineData("P2DT-0.005S")]
    [InlineData("P1MT-1H-30M-1.25S")]
    [InlineData("-P1DT0.5S")]
    [InlineData("P-1MT1.5S")]
    public void DurationTextsRoundTrip(string text) => Assert.Equal(text, new GrafeoDuration(text).Text);

    /// <summary>A fraction of a second takes the sign of its seconds, as the engine reads it.</summary>
    [Theory]
    [InlineData("PT-1.5S", 0, -1_500_000_000)]
    [InlineData("PT-0.5S", 0, -500_000_000)]
    [InlineData("-PT-1.5S", 0, 1_500_000_000)]
    [InlineData("P1MT-1.5S", 1, -1_500_000_000)]
    // The text 0.6.0 wrote for a negative fraction in a duration of mixed signs
    [InlineData("P1MT-1.-5S", 1, -1_500_000_000)]
    [InlineData("P1MT0.-5S", 1, -500_000_000)]
    [InlineData("P1MT0.-05S", 1, -5_000_000)]
    [InlineData("P1MT0.-0000001S", 1, -10)]
    [InlineData("P1MT0.-123456789S", 1, -123_456_789)]
    public void NegativeSecondsCarryTheirSignIntoTheFraction(string text, long months, long nanoseconds) =>
        Assert.Equal(new GrafeoDuration(months, 0, nanoseconds), new GrafeoDuration(text));

    /// <summary>The engine writes a negative fraction of a duration of mixed signs after the sign of its seconds.</summary>
    [Fact]
    public void MixedSignsWriteTheSignOnceOnTheSeconds()
    {
        Assert.Equal("P1MT-1.5S", new GrafeoDuration(1, 0, -1_500_000_000).Text);
        Assert.Equal("P1MT-0.5S", new GrafeoDuration(1, 0, -500_000_000).Text);
        Assert.Equal("P2DT-0.000000001S", new GrafeoDuration(0, 2, -1).Text);
    }

    [Theory]
    [InlineData("P")]
    [InlineData("1D")]
    [InlineData("P1X")]
    [InlineData("PT1D")]
    [InlineData("P+1D")]
    // Components past 64 bits: refused, not wrapped around to another duration
    [InlineData("P768614336404564651Y")]
    [InlineData("P1317624576693539402W")]
    [InlineData("PT2562048H")]
    [InlineData("PT153722867M281S")]
    [InlineData("PT9223372037S")]
    [InlineData("P9223372036854775807M1Y")]
    [InlineData("-P-9223372036854775808D")]
    [InlineData("PT-9223372037S")]
    // 0.6.0 never wrote a negative fraction after positive seconds
    [InlineData("P1MT1.-5S")]
    public void TextThatIsNotADurationIsRefused(string text)
    {
        Assert.False(GrafeoDuration.TryParse(text, out _));
        Assert.Throws<ArgumentException>(() => new GrafeoDuration(text));
    }

    /// <summary>Components at the edge of 64 bits still read.</summary>
    [Fact]
    public void DurationsAtTheEdgeOf64BitsRead()
    {
        Assert.Equal(new GrafeoDuration(long.MaxValue, 0, 0), new GrafeoDuration("P9223372036854775807M"));
        Assert.Equal(new GrafeoDuration(0, long.MinValue, 0), new GrafeoDuration("P-9223372036854775808D"));
        Assert.Equal(new GrafeoDuration(0, 0, 2_562_047 * 3_600_000_000_000), new GrafeoDuration("PT2562047H"));
        Assert.Equal(new GrafeoDuration(0, 0, long.MaxValue), new GrafeoDuration("PT9223372036.854775807S"));
        Assert.Equal(new GrafeoDuration(0, 0, long.MinValue), new GrafeoDuration("PT-9223372036.854775808S"));
    }

    /// <summary>A part out of its range is named in the exception.</summary>
    [Theory]
    [InlineData(24, 0, 0, 0, "hour")]
    [InlineData(12, 60, 0, 0, "minute")]
    [InlineData(12, 0, -1, 0, "second")]
    [InlineData(12, 0, 0, 1_000_000_000, "nanosecond")]
    public void AnOutOfRangePartIsNamed(int hour, int minute, int second, int nanosecond, string name)
    {
        var error = Assert.Throws<ArgumentOutOfRangeException>(() => new GrafeoTime(hour, minute, second, nanosecond));
        Assert.Equal(name, error.ParamName);
    }

    [Fact]
    public void TimePartsAreChecked()
    {
        var time = new GrafeoTime("14:30");
        Assert.Equal("14:30:00", time.Text);
        Assert.Null(time.Offset);
        Assert.Equal(new GrafeoDuration(0, 14, 0), new GrafeoDuration("P2W"));
        Assert.Throws<ArgumentOutOfRangeException>(() => new GrafeoTime(24, 0, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new GrafeoTime(12, 0, 0, 0, TimeSpan.FromSeconds(90)));
    }
}
