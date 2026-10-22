// An exact decimal as Grafeo stores it: up to 38 digits, with its scale.

using System.Globalization;

namespace Grafeo;

/// <summary>
/// An exact decimal from Grafeo: up to 38 digits, kept as its text with the
/// scale it was written with (<c>1.50</c>). Equal values are equal whatever
/// their scale (<c>1.50</c> equals <c>1.5</c>). <see cref="decimal"/> holds
/// 28 digits, so <see cref="TryToDecimal"/> may fail where this type does not.
/// </summary>
public readonly struct GrafeoDecimal : IEquatable<GrafeoDecimal>
{
    private readonly string _text;

    /// <summary>A decimal from its text: <c>[+-]digits[.digits]</c>.</summary>
    /// <exception cref="ArgumentException">The text is not a decimal number.</exception>
    public GrafeoDecimal(string text)
    {
        if (!IsDecimalText(text))
            throw new ArgumentException($"'{text}' is not a decimal number", nameof(text));
        _text = text;
    }

    /// <summary>A decimal with the value and scale of <paramref name="value"/>.</summary>
    public static GrafeoDecimal FromDecimal(decimal value) =>
        new(value.ToString(CultureInfo.InvariantCulture));

    /// <summary>The value as a <see cref="decimal"/>, when it fits 28 digits.</summary>
    public bool TryToDecimal(out decimal value) =>
        decimal.TryParse(Text, NumberStyles.AllowLeadingSign | NumberStyles.AllowDecimalPoint,
            CultureInfo.InvariantCulture, out value)
        && Normalized(value.ToString(CultureInfo.InvariantCulture)) == Normalized(Text);

    /// <summary>The value as a <see cref="decimal"/>.</summary>
    /// <exception cref="OverflowException">The value needs more than 28 digits.</exception>
    public decimal ToDecimal() =>
        TryToDecimal(out var value)
            ? value
            : throw new OverflowException($"{Text} does not fit a System.Decimal");

    /// <summary>The nearest double.</summary>
    public double ToDouble() => double.Parse(Text, CultureInfo.InvariantCulture);

    /// <summary>The text with its scale, as Grafeo writes it.</summary>
    public string Text => _text ?? "0";

    /// <inheritdoc/>
    public override string ToString() => Text;

    /// <inheritdoc/>
    public bool Equals(GrafeoDecimal other) => Normalized(Text) == Normalized(other.Text);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is GrafeoDecimal other && Equals(other);

    /// <inheritdoc/>
    public override int GetHashCode() => Normalized(Text).GetHashCode(StringComparison.Ordinal);

    /// <summary>Whether two decimals have the same value.</summary>
    public static bool operator ==(GrafeoDecimal left, GrafeoDecimal right) => left.Equals(right);

    /// <summary>Whether two decimals have different values.</summary>
    public static bool operator !=(GrafeoDecimal left, GrafeoDecimal right) => !left.Equals(right);

    private static bool IsDecimalText(string? text)
    {
        if (string.IsNullOrEmpty(text))
            return false;
        var body = text[0] is '+' or '-' ? text[1..] : text;
        var point = body.IndexOf('.');
        var integer = point < 0 ? body : body[..point];
        var fraction = point < 0 ? "" : body[(point + 1)..];
        return (integer.Length > 0 || fraction.Length > 0)
            && integer.All(char.IsAsciiDigit)
            && fraction.All(char.IsAsciiDigit);
    }

    /// <summary>The value without leading or trailing zeros: one text per value.</summary>
    private static string Normalized(string text)
    {
        var negative = text.StartsWith('-');
        var body = text[0] is '+' or '-' ? text[1..] : text;
        var point = body.IndexOf('.');
        var integer = (point < 0 ? body : body[..point]).TrimStart('0');
        var fraction = (point < 0 ? "" : body[(point + 1)..]).TrimEnd('0');
        if (integer.Length == 0 && fraction.Length == 0)
            return "0";
        return $"{(negative ? "-" : "")}{integer}.{fraction}";
    }
}
