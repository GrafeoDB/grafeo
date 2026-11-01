// A GQL DURATION: months, days and nanoseconds, kept apart.

using System.Globalization;
using System.Text;

namespace Grafeo;

/// <summary>
/// A GQL <c>DURATION</c>: months, days and nanoseconds, kept apart because a
/// month and a day have no fixed length (<c>P1M</c> is not <c>P30D</c>, and
/// <c>P1D</c> is not <c>PT24H</c>). A <see cref="TimeSpan"/> cannot hold
/// months, so every duration comes back as this type; a
/// <see cref="TimeSpan"/> parameter is a duration of its days and the rest.
/// Passed as a parameter it is the same <c>DURATION</c> again. Two durations
/// are equal when their three parts are.
/// </summary>
public readonly struct GrafeoDuration : IEquatable<GrafeoDuration>
{
    private const long NanosPerSecond = 1_000_000_000;
    private const long NanosPerMinute = 60 * NanosPerSecond;
    private const long NanosPerHour = 60 * NanosPerMinute;

    /// <summary>A duration of <paramref name="months"/>, <paramref name="days"/> and <paramref name="nanoseconds"/>.</summary>
    public GrafeoDuration(long months, long days, long nanoseconds)
    {
        Months = months;
        Days = days;
        Nanoseconds = nanoseconds;
    }

    /// <summary>
    /// A duration from its ISO 8601 text as GQL writes it:
    /// <c>P[nY][nM][nW][nD][T[nH][nM][n[.fraction]S]]</c>, with a leading minus
    /// for a negative one.
    /// </summary>
    /// <exception cref="ArgumentException">The text is not a duration.</exception>
    public GrafeoDuration(string text)
    {
        this = TryParse(text, out var duration)
            ? duration
            : throw new ArgumentException($"'{text}' is not a duration", nameof(text));
    }

    /// <summary>The duration of <paramref name="span"/>: its days, and the rest as nanoseconds.</summary>
    public static GrafeoDuration FromTimeSpan(TimeSpan span) =>
        new(0, span.Days, (span.Ticks - span.Days * TimeSpan.TicksPerDay) * 100);

    /// <summary>Reads a duration as <see cref="GrafeoDuration(string)"/> does; false when the text is not one.</summary>
    public static bool TryParse(string? text, out GrafeoDuration duration)
    {
        duration = default;
        if (string.IsNullOrEmpty(text))
            return false;
        var negative = text.StartsWith('-');
        var body = negative ? text[1..] : text;
        if (!body.StartsWith('P'))
            return false;
        body = body[1..];
        long months = 0, days = 0, nanos = 0;
        var inTime = false;
        var start = 0;
        var content = false;
        for (var i = 0; i < body.Length; i++)
        {
            var c = body[i];
            if (c == 'T' && !inTime && start == i)
            {
                inTime = true;
                start = i + 1;
                continue;
            }
            if (!(inTime ? "HMS" : "YMWD").Contains(c))
                continue;
            var number = body[start..i];
            if (c == 'S')
            {
                if (!SecondsNanos(number, out var secondsNanos))
                    return false;
                nanos += secondsNanos;
            }
            else
            {
                if (!Integer(number, out var n))
                    return false;
                switch (c)
                {
                    case 'Y': months += 12 * n; break;
                    case 'M' when !inTime: months += n; break;
                    case 'W': days += 7 * n; break;
                    case 'D': days += n; break;
                    case 'H': nanos += n * NanosPerHour; break;
                    default: nanos += n * NanosPerMinute; break;
                }
            }
            start = i + 1;
            content = true;
        }
        if (!content || start != body.Length)
            return false;
        duration = negative ? new(-months, -days, -nanos) : new(months, days, nanos);
        return true;
    }

    /// <summary>The months (a year is 12).</summary>
    public long Months { get; }

    /// <summary>The days.</summary>
    public long Days { get; }

    /// <summary>The part shorter than a day, in nanoseconds.</summary>
    public long Nanoseconds { get; }

    /// <summary>
    /// The duration as a <see cref="TimeSpan"/> (a day as 24 hours), when it has no
    /// months and its nanoseconds are whole 100 ns ticks.
    /// </summary>
    public bool TryToTimeSpan(out TimeSpan span)
    {
        span = default;
        if (Months != 0 || Nanoseconds % 100 != 0)
            return false;
        try
        {
            span = new TimeSpan(checked(Days * TimeSpan.TicksPerDay + Nanoseconds / 100));
            return true;
        }
        catch (OverflowException)
        {
            return false;
        }
    }

    /// <summary>The duration as a <see cref="TimeSpan"/> (a day as 24 hours).</summary>
    /// <exception cref="InvalidOperationException">The duration has months, or a part finer than 100 ns.</exception>
    public TimeSpan ToTimeSpan() =>
        TryToTimeSpan(out var span)
            ? span
            : throw new InvalidOperationException($"{Text} has months or a part finer than 100 ns: no TimeSpan holds it");

    /// <summary>The text as GQL writes it: <c>P1Y2M3DT4H5M6.5S</c>, <c>PT0S</c>.</summary>
    public string Text
    {
        get
        {
            long months = Months, days = Days, nanos = Nanoseconds;
            var negative = (months < 0 || days < 0 || nanos < 0) && months <= 0 && days <= 0 && nanos <= 0;
            if (negative)
            {
                months = -months;
                days = -days;
                nanos = -nanos;
            }
            var text = new StringBuilder(negative ? "-P" : "P");
            var invariant = CultureInfo.InvariantCulture;
            if (months / 12 != 0)
                text.Append(invariant, $"{months / 12}Y");
            if (months % 12 != 0)
                text.Append(invariant, $"{months % 12}M");
            if (days != 0)
                text.Append(invariant, $"{days}D");
            if (nanos != 0)
            {
                text.Append('T');
                var hours = nanos / NanosPerHour;
                var minutes = nanos % NanosPerHour / NanosPerMinute;
                var seconds = nanos % NanosPerMinute / NanosPerSecond;
                var fraction = nanos % NanosPerSecond;
                if (hours != 0)
                    text.Append(invariant, $"{hours}H");
                if (minutes != 0)
                    text.Append(invariant, $"{minutes}M");
                if (fraction != 0)
                    text.Append(invariant, $"{seconds}.{Fraction(fraction)}S");
                else if (seconds != 0)
                    text.Append(invariant, $"{seconds}S");
            }
            if (!negative && months == 0 && days == 0 && nanos == 0)
                text.Append("T0S");
            return text.ToString();
        }
    }

    /// <inheritdoc/>
    public override string ToString() => Text;

    /// <inheritdoc/>
    public bool Equals(GrafeoDuration other) =>
        Months == other.Months && Days == other.Days && Nanoseconds == other.Nanoseconds;

    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is GrafeoDuration other && Equals(other);

    /// <inheritdoc/>
    public override int GetHashCode() => HashCode.Combine(Months, Days, Nanoseconds);

    /// <summary>Whether two durations have the same three parts.</summary>
    public static bool operator ==(GrafeoDuration left, GrafeoDuration right) => left.Equals(right);

    /// <summary>Whether two durations differ in a part.</summary>
    public static bool operator !=(GrafeoDuration left, GrafeoDuration right) => !left.Equals(right);

    /// <summary>
    /// The digits of a fraction of a second as the engine writes them: nine
    /// digits (a sign first for a negative part of a duration of mixed signs),
    /// without trailing zeros.
    /// </summary>
    private static string Fraction(long nanos) =>
        (nanos < 0 ? "-" + (-nanos).ToString("00000000", CultureInfo.InvariantCulture)
            : nanos.ToString("000000000", CultureInfo.InvariantCulture)).TrimEnd('0');

    /// <summary>An integer with an optional minus, as the engine reads one.</summary>
    private static bool Integer(string text, out long value) =>
        long.TryParse(text, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out value)
        && !text.StartsWith('+');

    /// <summary>
    /// The seconds of a duration, <c>n</c> or <c>n.fraction</c>, as the engine
    /// reads them: the fraction's first nine characters as a number of their
    /// scale, so that the text the engine writes for any duration reads back as it.
    /// </summary>
    private static bool SecondsNanos(string text, out long nanos)
    {
        nanos = 0;
        var point = text.IndexOf('.');
        if (!Integer(point < 0 ? text : text[..point], out var whole))
            return false;
        nanos = whole * NanosPerSecond;
        if (point < 0)
            return true;
        var fraction = text[(point + 1)..];
        if (fraction.Length > 9)
            fraction = fraction[..9];
        if (!Integer(fraction, out var digits))
            return false;
        nanos += digits * (long)Math.Pow(10, 9 - fraction.Length);
        return true;
    }
}
