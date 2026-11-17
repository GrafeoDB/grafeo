// Bidirectional conversion between C# types and Grafeo's JSON wire format.
// Follows grafeo-bindings-common temporal markers ($timestamp_us, $date, etc.).

using System.Text.Json;
using System.Text.Json.Nodes;

namespace Grafeo;

/// <summary>
/// Converts C# values to JSON for the grafeo-c wire format and parses
/// JSON results back into C# objects.
/// </summary>
internal static class ValueConverter
{
    /// <summary>The Unix microseconds of DateTime.MinValue and DateTime.MaxValue.</summary>
    private const long MinUnixMicroseconds = -62_135_596_800_000_000;
    private const long MaxUnixMicroseconds = 253_402_300_799_999_999;

    /// <summary>Encode a parameter dictionary as a JSON string for grafeo-c.</summary>
    internal static string EncodeParams(Dictionary<string, object?> parameters)
    {
        var obj = new JsonObject();
        foreach (var (key, value) in parameters)
        {
            obj[key] = ToJsonNode(value);
        }
        return obj.ToJsonString();
    }

    /// <summary>Encode a single value as a JSON string.</summary>
    internal static string EncodeValue(object? value)
    {
        var node = ToJsonNode(value);
        return node?.ToJsonString() ?? "null";
    }

    /// <summary>Convert a C# object to a JsonNode for serialization.</summary>
    internal static JsonNode? ToJsonNode(object? value) =>
        value switch
        {
            null => null,
            bool b => JsonValue.Create(b),
            int i => JsonValue.Create((long)i),
            long l => JsonValue.Create(l),
            // Every integer goes as a JSON number with its exact digits: one
            // past 64 bits arrives as an exact decimal up to 38 digits, and
            // one past that is an error naming the parameter.
            sbyte i => JsonValue.Create((long)i),
            byte i => JsonValue.Create((long)i),
            short i => JsonValue.Create((long)i),
            ushort i => JsonValue.Create((long)i),
            uint i => JsonValue.Create((long)i),
            ulong u => JsonValue.Create(u),
            Int128 i => IntegerNumber(i.ToString(System.Globalization.CultureInfo.InvariantCulture)),
            UInt128 u => IntegerNumber(u.ToString(System.Globalization.CultureInfo.InvariantCulture)),
            System.Numerics.BigInteger b => IntegerNumber(b.ToString(System.Globalization.CultureInfo.InvariantCulture)),
            float f => FloatNode(f),
            double d => FloatNode(d),
            string s => JsonValue.Create(s),
            GrafeoDecimal g => new JsonObject { ["$decimal"] = g.Text },
            RdfTerm term => term.ToJson(),
            decimal m => new JsonObject
            {
                ["$decimal"] = m.ToString(System.Globalization.CultureInfo.InvariantCulture)
            },
            // A TIMESTAMP, to the microsecond (a finer part dropped)
            DateTime dt => new JsonObject { ["$timestamp_us"] = UnixMicroseconds(dt.ToUniversalTime()) },
            // A ZONED DATETIME: the offset is kept
            DateTimeOffset dto => new JsonObject { ["$zoned_datetime"] = ZonedText(dto) },
            DateOnly date => new JsonObject
            {
                ["$date"] = date.ToString("yyyy'-'MM'-'dd", System.Globalization.CultureInfo.InvariantCulture)
            },
            TimeOnly time => new JsonObject { ["$time"] = GrafeoTime.FromTimeOnly(time).Text },
            GrafeoTime time => new JsonObject { ["$time"] = time.Text },
            TimeSpan duration => new JsonObject { ["$duration"] = GrafeoDuration.FromTimeSpan(duration).Text },
            GrafeoDuration duration => new JsonObject { ["$duration"] = duration.Text },
            byte[] bytes => new JsonObject { ["$bytes"] = Convert.ToBase64String(bytes) },
            float[] vector => VectorToJsonArray(vector),
            ReadOnlyMemory<float> mem => VectorToJsonArray(mem.Span),
            IList<object?> list => ListToJsonArray(list),
            IDictionary<string, object?> dict => DictToJsonObject(dict),
            _ => JsonValue.Create(value.ToString()),
        };

    /// <summary>The microseconds of a UTC <paramref name="utc"/> since the Unix epoch, rounded down.</summary>
    private static long UnixMicroseconds(DateTime utc)
    {
        var ticks = utc.Ticks - DateTime.UnixEpoch.Ticks;
        return ticks / 10 - (ticks % 10 < 0 ? 1 : 0);
    }

    /// <summary>A ZONED DATETIME's text: the clock reading to the microsecond, and the offset.</summary>
    private static string ZonedText(DateTimeOffset dto) =>
        dto.ToString("yyyy'-'MM'-'dd'T'HH':'mm':'ss.FFFFFF", System.Globalization.CultureInfo.InvariantCulture)
        + dto.ToString("zzz", System.Globalization.CultureInfo.InvariantCulture);

    /// <summary>
    /// A double as JSON: a number that reads back as a float (a whole one keeps
    /// a fraction, as System.Text.Json writes 3.0 as 3, which reads as an
    /// integer), and NaN or an infinity, which JSON cannot write, as the
    /// {"$float": "NaN"} tag.
    /// </summary>
    private static JsonNode FloatNode(double d)
    {
        if (double.IsNaN(d))
            return new JsonObject { ["$float"] = "NaN" };
        if (double.IsInfinity(d))
            return new JsonObject { ["$float"] = d > 0 ? "Infinity" : "-Infinity" };
        var text = d.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
        if (text.IndexOfAny(['.', 'E', 'e']) < 0)
            text += ".0";
        return JsonNode.Parse(text)!;
    }

    /// <summary>A JSON number of an integer's decimal digits, written as they are.</summary>
    private static JsonNode IntegerNumber(string digits) => JsonNode.Parse(digits)!;

    private static JsonArray VectorToJsonArray(ReadOnlySpan<float> vector)
    {
        var arr = new JsonArray();
        foreach (var f in vector)
            arr.Add(FloatNode(f));
        return arr;
    }

    private static JsonArray VectorToJsonArray(float[] vector) =>
        VectorToJsonArray(vector.AsSpan());

    private static JsonArray ListToJsonArray(IList<object?> list)
    {
        var arr = new JsonArray();
        foreach (var item in list)
            arr.Add(ToJsonNode(item));
        return arr;
    }

    private static JsonObject DictToJsonObject(IDictionary<string, object?> dict)
    {
        var obj = new JsonObject();
        foreach (var (key, val) in dict)
            obj[key] = ToJsonNode(val);
        return obj;
    }

    // =========================================================================
    // JSON result parsing
    // =========================================================================

    /// <summary>Parse a JSON array string (from grafeo_result_json) into rows.</summary>
    internal static IReadOnlyList<IReadOnlyDictionary<string, object?>> ParseRows(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var rows = new List<IReadOnlyDictionary<string, object?>>();
        foreach (var element in doc.RootElement.EnumerateArray())
        {
            var row = new Dictionary<string, object?>();
            foreach (var prop in element.EnumerateObject())
            {
                row[prop.Name] = FromJsonElement(prop.Value);
            }
            rows.Add(row);
        }
        return rows;
    }

    /// <summary>Parse a JSON object string into a dictionary.</summary>
    internal static IReadOnlyDictionary<string, object?> ParseObject(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var dict = new Dictionary<string, object?>();
        foreach (var prop in doc.RootElement.EnumerateObject())
        {
            dict[prop.Name] = FromJsonElement(prop.Value);
        }
        return dict;
    }

    /// <summary>Parse a JSON array string into a list of strings.</summary>
    internal static IReadOnlyList<string> ParseStringArray(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var list = new List<string>();
        foreach (var element in doc.RootElement.EnumerateArray())
        {
            if (element.ValueKind == JsonValueKind.String)
                list.Add(element.GetString()!);
        }
        return list;
    }

    /// <summary>Convert a JsonElement to a C# object, handling temporal markers.</summary>
    internal static object? FromJsonElement(JsonElement element) =>
        element.ValueKind switch
        {
            JsonValueKind.Null or JsonValueKind.Undefined => null,
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            JsonValueKind.String => element.GetString(),
            JsonValueKind.Number => ParseNumber(element),
            JsonValueKind.Array => ParseJsonArray(element),
            JsonValueKind.Object => ParseJsonObject(element),
            _ => element.GetRawText(),
        };

    private static object ParseNumber(JsonElement element)
    {
        // Prefer long if the number is an integer
        if (element.TryGetInt64(out var l))
            return l;
        return element.GetDouble();
    }

    private static object ParseJsonArray(JsonElement element)
    {
        var list = new List<object?>();
        foreach (var item in element.EnumerateArray())
            list.Add(FromJsonElement(item));

        // Check if all elements are doubles (vector)
        if (list.Count > 0 && list.All(x => x is double or long))
        {
            // Could be a vector, but return as list for generality
        }
        return list;
    }

    private static object? ParseJsonObject(JsonElement element)
    {
        // An RDF term: the "$term" tag as the object's only key
        if (RdfTerm.FromJson(element) is { } term)
            return term;

        // Bytes: the "$bytes" tag, standard base64, as the object's only key
        if (element.TryGetProperty("$bytes", out var bytesElement)
            && element.EnumerateObject().Count() == 1
            && bytesElement.ValueKind == JsonValueKind.String)
        {
            // Canonical standard base64 only, as the engine reads it: a text it
            // kept as a map (whitespace, unused bits set) stays a map
            var text = bytesElement.GetString()!;
            var buffer = new byte[text.Length];
            if (Convert.TryFromBase64String(text, buffer, out var written)
                && Convert.ToBase64String(buffer, 0, written) == text)
                return buffer[..written];
        }

        // A float JSON cannot write: the "$float" tag as the object's only key
        if (element.TryGetProperty("$float", out var floatElement)
            && element.EnumerateObject().Count() == 1
            && floatElement.ValueKind == JsonValueKind.String)
        {
            switch (floatElement.GetString())
            {
                case "NaN": return double.NaN;
                case "Infinity": return double.PositiveInfinity;
                case "-Infinity": return double.NegativeInfinity;
            }
        }

        // An exact decimal: the "$decimal" tag as the object's only key
        if (element.TryGetProperty("$decimal", out var decimalElement)
            && element.EnumerateObject().Count() == 1
            && decimalElement.ValueKind == JsonValueKind.String)
        {
            return new GrafeoDecimal(decimalElement.GetString()!);
        }

        // Temporal tags from grafeo-bindings-common, each as the object's only
        // key and with a value of its type: an object with another key, or a
        // value of another type, stays a map, and so does a value the type it
        // comes back as cannot hold (a timestamp before year 1 or after 9999),
        // where it would fail the whole result. As a parameter, such a map is
        // the tagged value again.
        if (element.EnumerateObject().Count() == 1 && TemporalOf(element) is { } temporal)
            return temporal;

        // Regular object
        var dict = new Dictionary<string, object?>();
        foreach (var prop in element.EnumerateObject())
        {
            dict[prop.Name] = FromJsonElement(prop.Value);
        }
        return dict;
    }

    /// <summary>
    /// The value of a temporal tag: a TIMESTAMP as a UTC <see cref="DateTime"/>,
    /// a ZONED DATETIME as a <see cref="DateTimeOffset"/>, a DATE as a
    /// <see cref="DateOnly"/>, a TIME as a <see cref="TimeOnly"/> (local, in
    /// whole 100 ns) or else a <see cref="GrafeoTime"/>, a DURATION as a
    /// <see cref="GrafeoDuration"/>; null for another tag or a value these
    /// cannot hold.
    /// </summary>
    private static object? TemporalOf(JsonElement element)
    {
        if (element.TryGetProperty("$timestamp_us", out var tsElement))
        {
            return tsElement.ValueKind == JsonValueKind.Number
                && tsElement.TryGetInt64(out var microseconds)
                && microseconds is >= MinUnixMicroseconds and <= MaxUnixMicroseconds
                ? DateTime.UnixEpoch.AddTicks(microseconds * 10)
                : null;
        }
        var property = element.EnumerateObject().First();
        if (property.Value.ValueKind != JsonValueKind.String)
            return null;
        var text = property.Value.GetString()!;
        var invariant = System.Globalization.CultureInfo.InvariantCulture;
        var styles = System.Globalization.DateTimeStyles.None;
        switch (property.Name)
        {
            case "$zoned_datetime":
                return DateTimeOffset.TryParseExact(text, ZonedFormats, invariant, styles, out var zoned)
                    ? zoned
                    : null;
            case "$date":
                return DateOnly.TryParseExact(text, "yyyy'-'MM'-'dd", invariant, styles, out var date)
                    ? date
                    : null;
            case "$time":
                if (!GrafeoTime.TryParse(text, out var time))
                    return null;
                return time.Offset is null && time.Nanosecond % 100 == 0 ? time.ToTimeOnly() : time;
            case "$duration":
                return GrafeoDuration.TryParse(text, out var duration) ? duration : null;
            default:
                return null;
        }
    }

    /// <summary>The texts of a ZONED DATETIME: a fraction of up to six digits or none, then Z or an offset.</summary>
    private static readonly string[] ZonedFormats =
    [
        "yyyy'-'MM'-'dd'T'HH':'mm':'ss.FFFFFFK",
        "yyyy'-'MM'-'dd'T'HH':'mm':'ssK",
    ];

    // =========================================================================
    // Entity extraction
    // =========================================================================

    /// <summary>
    /// The nodes and edges of a result, read from <c>grafeo_result_nodes_json</c> and
    /// <c>grafeo_result_edges_json</c>: the engine says by type which values are nodes and
    /// edges, so a map with the keys of a node (<c>{_id: 1, _labels: []}</c>) is a map.
    /// </summary>
    internal static (IReadOnlyList<Node> Nodes, IReadOnlyList<Edge> Edges) ParseEntities(
        string nodesJson, string edgesJson)
    {
        var nodes = new List<Node>();
        using (var doc = JsonDocument.Parse(nodesJson))
        {
            foreach (var node in doc.RootElement.EnumerateArray())
            {
                var labels = node.GetProperty("labels").EnumerateArray()
                    .Select(label => label.GetString() ?? "")
                    .ToList();
                nodes.Add(new Node(node.GetProperty("id").GetInt64(), labels, Properties(node)));
            }
        }
        var edges = new List<Edge>();
        using (var doc = JsonDocument.Parse(edgesJson))
        {
            foreach (var edge in doc.RootElement.EnumerateArray())
            {
                edges.Add(new Edge(
                    edge.GetProperty("id").GetInt64(),
                    edge.GetProperty("type").GetString() ?? "",
                    edge.GetProperty("source_id").GetInt64(),
                    edge.GetProperty("target_id").GetInt64(),
                    Properties(edge)));
            }
        }
        return (nodes, edges);
    }

    /// <summary>The <c>properties</c> of a node or edge object of the C layer.</summary>
    private static IReadOnlyDictionary<string, object?> Properties(JsonElement entity)
    {
        var props = new Dictionary<string, object?>();
        foreach (var prop in entity.GetProperty("properties").EnumerateObject())
            props[prop.Name] = FromJsonElement(prop.Value);
        return props;
    }
}
