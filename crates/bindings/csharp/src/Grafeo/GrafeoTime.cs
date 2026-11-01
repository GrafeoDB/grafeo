// A GQL TIME: a time of day to the nanosecond, with or without a UTC offset.

using System.Globalization;

namespace Grafeo;

/// <summary>
/// A GQL <c>TIME</c>: a time of day to the nanosecond, with a UTC offset
/// (<c>12:30:00+02:00</c>) or without one (a local time, <c>12:30:00</c>).
/// A time with an offset, or one finer than <see cref="TimeOnly"/>'s 100 ns,
/// comes back as this type; any other time comes back as a
/// <see cref="TimeOnly"/>. Passed as a parameter it is the same
/// <c>TIME</c> again. Two times are equal when they have the same clock
/// reading and offset.
/// </summary>
public readonly struct GrafeoTime : IEquatable<GrafeoTime>
{
    private const long NanosPerSecond = 1_000_000_000;
    private const long NanosPerMinute = 60 * NanosPerSecond;
    private const long NanosPerHour = 60 * NanosPerMinute;

    private readonly long _nanos;
    private readonly int _offsetSeconds;
    private readonly bool _hasOffset;

    /// <summary>
    /// A time of day from its parts, with an <paramref name="offset"/> of whole
    /// minutes less than a day, or none for a local time.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">A part is out of its range.</exception>
    public GrafeoTime(int hour, int minute, int second, int nanosecond = 0, TimeSpan? offset = null)
    {
        if (hour is < 0 or > 23 || minute is < 0 or > 59 || second is < 0 or > 59
            || nanosecond is < 0 or >= (int)NanosPerSecond)
            throw new ArgumentOutOfRangeException(nameof(hour),
                $"{hour:00}:{minute:00}:{second:00}.{nanosecond:000000000} is not a time of day");
        if (offset is { } span && (span.Ticks % TimeSpan.TicksPerMinute != 0 || span.Duration() >= TimeSpan.FromDays(1)))
            throw new ArgumentOutOfRangeException(nameof(offset),
                $"an offset of {span} is not whole minutes within a day");
        _nanos = hour * NanosPerHour + minute * NanosPerMinute + second * NanosPerSecond + nanosecond;
        _offsetSeconds = offset is { } o ? (int)(o.Ticks / TimeSpan.TicksPerSecond) : 0;
        _hasOffset = offset.HasValue;
    }

    /// <summary>
    /// A time from its text as GQL writes it: <c>HH:MM[:SS[.fraction]]</c>, then
    /// <c>Z</c>, <c>+HH:MM</c> or <c>-HH:MM</c> for an offset.
    /// </summary>
    /// <exception cref="ArgumentException">The text is not a time.</exception>
    public GrafeoTime(string text)
    {
        this = TryParse(text, out var time)
            ? time
            : throw new ArgumentException($"'{text}' is not a time", nameof(text));
    }

    /// <summary>The clock reading of <paramref name="time"/>, with an offset or none.</summary>
    public static GrafeoTime FromTimeOnly(TimeOnly time, TimeSpan? offset = null) =>
        new(time.Hour, time.Minute, time.Second,
            (int)(time.Ticks % TimeSpan.TicksPerSecond * 100), offset);

    /// <summary>Reads a time as <see cref="GrafeoTime(string)"/> does; false when the text is not one.</summary>
    public static bool TryParse(string? text, out GrafeoTime time)
    {
        time = default;
        if (string.IsNullOrEmpty(text))
            return false;
        var body = text;
        TimeSpan? offset = null;
        if (text.EndsWith('Z'))
        {
            body = text[..^1];
            offset = TimeSpan.Zero;
        }
        else if (text.Length >= 6 && text[^6] is '+' or '-' && text[^3] == ':')
        {
            if (!Digits(text[^5..^3], out var hours) || !Digits(text[^2..], out var minutes) || minutes > 59)
                return false;
            var span = new TimeSpan(hours, minutes, 0);
            offset = text[^6] == '-' ? -span : span;
            body = text[..^6];
        }
        var point = body.IndexOf('.');
        var clock = point < 0 ? body : body[..point];
        var fields = clock.Split(':');
        if (fields.Length is < 2 or > 3)
            return false;
        var hms = new int[3];
        for (var i = 0; i < fields.Length; i++)
        {
            if (!Digits(fields[i], out hms[i]))
                return false;
        }
        var nanos = 0;
        if (point >= 0)
        {
            var fraction = body[(point + 1)..];
            if (fraction.Length == 0 || !fraction.All(char.IsAsciiDigit))
                return false;
            nanos = int.Parse((fraction + "000000000")[..9], CultureInfo.InvariantCulture);
        }
        if (hms[0] > 23 || hms[1] > 59 || hms[2] > 59
            || offset is { } o && o.Duration() >= TimeSpan.FromDays(1))
            return false;
        time = new GrafeoTime(hms[0], hms[1], hms[2], nanos, offset);
        return true;
    }

    /// <summary>The hour, 0 to 23.</summary>
    public int Hour => (int)(_nanos / NanosPerHour);

    /// <summary>The minute, 0 to 59.</summary>
    public int Minute => (int)(_nanos % NanosPerHour / NanosPerMinute);

    /// <summary>The second, 0 to 59.</summary>
    public int Second => (int)(_nanos % NanosPerMinute / NanosPerSecond);

    /// <summary>The fraction of the second in nanoseconds.</summary>
    public int Nanosecond => (int)(_nanos % NanosPerSecond);

    /// <summary>The UTC offset, or null for a local time.</summary>
    public TimeSpan? Offset => _hasOffset ? TimeSpan.FromSeconds(_offsetSeconds) : null;

    /// <summary>The clock reading as a <see cref="TimeOnly"/>, to 100 ns, without the offset.</summary>
    public TimeOnly ToTimeOnly() => new(_nanos / 100);

    /// <summary>The text as GQL writes it: <c>14:30:00</c>, <c>14:30:00.5+02:00</c>, <c>14:30:00Z</c>.</summary>
    public string Text
    {
        get
        {
            var text = $"{Hour:00}:{Minute:00}:{Second:00}";
            if (Nanosecond != 0)
                text += "." + Nanosecond.ToString("000000000", CultureInfo.InvariantCulture).TrimEnd('0');
            if (!_hasOffset)
                return text;
            if (_offsetSeconds == 0)
                return text + "Z";
            var abs = Math.Abs(_offsetSeconds);
            return $"{text}{(_offsetSeconds < 0 ? '-' : '+')}{abs / 3600:00}:{abs % 3600 / 60:00}";
        }
    }

    /// <inheritdoc/>
    public override string ToString() => Text;

    /// <inheritdoc/>
    public bool Equals(GrafeoTime other) =>
        _nanos == other._nanos && _hasOffset == other._hasOffset && _offsetSeconds == other._offsetSeconds;

    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is GrafeoTime other && Equals(other);

    /// <inheritdoc/>
    public override int GetHashCode() => HashCode.Combine(_nanos, _hasOffset, _offsetSeconds);

    /// <summary>Whether two times have the same clock reading and offset.</summary>
    public static bool operator ==(GrafeoTime left, GrafeoTime right) => left.Equals(right);

    /// <summary>Whether two times differ in clock reading or offset.</summary>
    public static bool operator !=(GrafeoTime left, GrafeoTime right) => !left.Equals(right);

    /// <summary>A text of ASCII digits only, as a number.</summary>
    private static bool Digits(string text, out int value)
    {
        value = 0;
        return text.Length > 0 && text.All(char.IsAsciiDigit)
            && int.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out value);
    }
}
