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
            float f => JsonValue.Create((double)f),
            double d => JsonValue.Create(d),
            string s => JsonValue.Create(s),
            DateTime dt => new JsonObject
            {
                ["$timestamp_us"] = new DateTimeOffset(dt.ToUniversalTime())
                    .ToUnixTimeMilliseconds() * 1000
            },
            DateTimeOffset dto => new JsonObject
            {
                ["$timestamp_us"] = dto.ToUnixTimeMilliseconds() * 1000
            },
            DateOnly date => new JsonObject
            {
                ["$date"] = date.ToString("yyyy-MM-dd")
            },
            TimeOnly time => new JsonObject
            {
                ["$time"] = time.ToString("HH:mm:ss")
            },
            TimeSpan duration => new JsonObject
            {
                ["$duration"] = System.Xml.XmlConvert.ToString(duration)
            },
            byte[] bytes => JsonValue.Create(Convert.ToBase64String(bytes)),
            float[] vector => VectorToJsonArray(vector),
            ReadOnlyMemory<float> mem => VectorToJsonArray(mem.Span),
            IList<object?> list => ListToJsonArray(list),
            IDictionary<string, object?> dict => DictToJsonObject(dict),
            _ => JsonValue.Create(value.ToString()),
        };

    private static JsonArray VectorToJsonArray(ReadOnlySpan<float> vector)
    {
        var arr = new JsonArray();
        foreach (var f in vector)
            arr.Add(JsonValue.Create((double)f));
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

    /// <summary>Extract column names from the first row (or return empty).</summary>
    internal static IReadOnlyList<string> ExtractColumns(
        IReadOnlyList<IReadOnlyDictionary<string, object?>> rows)
    {
        if (rows.Count == 0)
            return Array.Empty<string>();
        return rows[0].Keys.ToList();
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
        // Check for temporal markers from grafeo-bindings-common
        if (element.TryGetProperty("$timestamp_us", out var tsElement))
        {
            var microseconds = tsElement.GetInt64();
            var milliseconds = microseconds / 1000;
            return DateTimeOffset.FromUnixTimeMilliseconds(milliseconds).UtcDateTime;
        }
        if (element.TryGetProperty("$date", out var dateElement))
        {
            return dateElement.GetString() ?? "";
        }
        if (element.TryGetProperty("$time", out var timeElement))
        {
            return timeElement.GetString() ?? "";
        }
        if (element.TryGetProperty("$duration", out var durElement))
        {
            return durElement.GetString() ?? "";
        }
        if (element.TryGetProperty("$zoned_datetime", out var zdtElement))
        {
            return zdtElement.GetString() ?? "";
        }

        // Regular object
        var dict = new Dictionary<string, object?>();
        foreach (var prop in element.EnumerateObject())
        {
            dict[prop.Name] = FromJsonElement(prop.Value);
        }
        return dict;
    }

    // =========================================================================
    // Entity extraction
    // =========================================================================

    /// <summary>
    /// Extract Node and Edge entities from query result rows.
    /// Mirrors grafeo-bindings-common entity::extract_entities.
    /// </summary>
    internal static (IReadOnlyList<Node> Nodes, IReadOnlyList<Edge> Edges) ExtractEntities(
        IReadOnlyList<IReadOnlyDictionary<string, object?>> rows)
    {
        var nodeIds = new HashSet<long>();
        var edgeIds = new HashSet<long>();
        var nodes = new List<Node>();
        var edges = new List<Edge>();

        foreach (var row in rows)
        {
            foreach (var value in row.Values)
            {
                if (value is not Dictionary<string, object?> map) continue;

                if (map.TryGetValue("_id", out var idObj) && idObj is long id)
                {
                    if (map.ContainsKey("_labels"))
                    {
                        // Node
                        if (!nodeIds.Add(id)) continue;
                        var labels = map.TryGetValue("_labels", out var labelsObj) && labelsObj is List<object?> labelList
                            ? labelList.OfType<string>().ToList()
                            : new List<string>();
                        var props = ExtractProperties(map);
                        nodes.Add(new Node(id, labels, props));
                    }
                    else if (map.ContainsKey("_type"))
                    {
                        // Edge
                        if (!edgeIds.Add(id)) continue;
                        var edgeType = map.TryGetValue("_type", out var typeObj) && typeObj is string t ? t : "";
                        var sourceId = map.TryGetValue("_source", out var srcObj) && srcObj is long src ? src : 0;
                        var targetId = map.TryGetValue("_target", out var tgtObj) && tgtObj is long tgt ? tgt : 0;
                        var props = ExtractProperties(map);
                        edges.Add(new Edge(id, edgeType, sourceId, targetId, props));
                    }
                }
            }
        }

        return (nodes, edges);
    }

    /// <summary>Extract user properties, stripping internal _-prefixed keys.</summary>
    private static IReadOnlyDictionary<string, object?> ExtractProperties(
        Dictionary<string, object?> map)
    {
        var props = new Dictionary<string, object?>();
        foreach (var (key, value) in map)
        {
            if (!key.StartsWith('_'))
                props[key] = value;
        }
        return props;
    }
}
