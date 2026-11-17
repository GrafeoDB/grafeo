// The openCypher TCK's value notation, for `values:`: parse a cell and compare it to a result
// value, type-exact (a long never matches a double). Mirrors the Rust runner's src/tck.rs; every
// runner checks itself against tests/spec/runners/values_cases.json (TckValueCaseTests below).

using System.Globalization;
using System.Text;
using System.Text.Json;
using Grafeo;
using Xunit;

namespace SpecRunner;

/// <summary>The kind of a TCK cell.</summary>
public enum TckKind { Null, Bool, Int, Float, String, List, Map, Node, Relationship, Path }

/// <summary>One cell of the TCK's value notation.</summary>
public sealed class TckCell
{
    public TckKind Kind { get; init; }
    public bool Bool { get; init; }
    public long Int { get; init; }
    public double Float { get; init; }
    /// <summary>A string, or a relationship's type.</summary>
    public string Text { get; init; } = "";
    public List<TckCell> Items { get; init; } = [];
    /// <summary>A map's entries, or an entity's properties.</summary>
    public Dictionary<string, TckCell> Entries { get; init; } = [];
    public HashSet<string> Labels { get; init; } = [];
    public List<TckCell> Nodes { get; init; } = [];
    public List<TckCell> Relationships { get; init; } = [];
    /// <summary>Whether each path relationship starts at the node on its left.</summary>
    public List<bool> Forward { get; init; } = [];
}

/// <summary>Parses and compares TCK cells.</summary>
public static class TckValues
{
    private sealed class Parser(string text)
    {
        private int _at;

        private FormatException Fail(string what) => new($"{what} at {_at} in '{text}'");

        private void Skip()
        {
            while (_at < text.Length && char.IsWhiteSpace(text[_at])) _at++;
        }

        private bool Peek(string token)
        {
            Skip();
            return string.CompareOrdinal(text, _at, token, 0, token.Length) == 0;
        }

        private void Take(string token)
        {
            if (!Peek(token)) throw Fail($"expected '{token}'");
            _at += token.Length;
        }

        private bool Word(string word)
        {
            if (string.CompareOrdinal(text, _at, word, 0, word.Length) != 0) return false;
            var after = _at + word.Length;
            if (after < text.Length && char.IsLetterOrDigit(text, after)) return false;
            _at = after;
            return true;
        }

        private string Name()
        {
            Skip();
            if (Peek("`"))
            {
                var end = text.IndexOf('`', _at + 1);
                if (end < 0) throw Fail("unterminated name");
                var quoted = text[(_at + 1)..end];
                _at = end + 1;
                return quoted;
            }
            var start = _at;
            // By code point: a letter outside the BMP is a surrogate pair
            while (_at < text.Length && (char.IsLetterOrDigit(text, _at) || text[_at] == '_'))
                _at += char.IsSurrogatePair(text, _at) ? 2 : 1;
            if (start == _at) throw Fail("expected a name");
            return text[start.._at];
        }

        public TckCell Value()
        {
            Skip();
            if (Word("null")) return new TckCell { Kind = TckKind.Null };
            if (Word("true")) return new TckCell { Kind = TckKind.Bool, Bool = true };
            if (Word("false")) return new TckCell { Kind = TckKind.Bool };
            foreach (var (word, value) in new[] { ("NaN", double.NaN), ("-Inf", double.NegativeInfinity), ("Inf", double.PositiveInfinity) })
            {
                if (string.CompareOrdinal(text, _at, word, 0, word.Length) == 0)
                {
                    _at += word.Length;
                    return new TckCell { Kind = TckKind.Float, Float = value };
                }
            }
            if (_at >= text.Length) throw Fail("expected a value");
            switch (text[_at])
            {
                case '\'': return new TckCell { Kind = TckKind.String, Text = Str() };
                case '{': return new TckCell { Kind = TckKind.Map, Entries = Mapping() };
                case '(': return Node();
                case '<': return Path();
                case '[' when Peek("[:"): return Relationship();
                case '[':
                    Take("[");
                    return new TckCell { Kind = TckKind.List, Items = Items("]", Value) };
            }
            return Number();
        }

        private string Str()
        {
            Take("'");
            var sb = new StringBuilder();
            while (_at < text.Length)
            {
                var c = text[_at];
                if (c == '\'') { _at++; return sb.ToString(); }
                if (c == '\\' && _at + 1 < text.Length)
                {
                    var code = text[_at + 1];
                    if (code == 'u' && _at + 6 <= text.Length)
                    {
                        sb.Append((char)int.Parse(text.AsSpan(_at + 2, 4), NumberStyles.HexNumber, CultureInfo.InvariantCulture));
                        _at += 6;
                        continue;
                    }
                    sb.Append(code switch
                    {
                        '\'' or '"' or '\\' => code.ToString(),
                        'n' => "\n", 't' => "\t", 'r' => "\r", 'b' => "\b", 'f' => "\f",
                        _ => "\\" + code,
                    });
                    _at += 2;
                    continue;
                }
                sb.Append(c);
                _at++;
            }
            throw Fail("unterminated string");
        }

        private TckCell Number()
        {
            var start = _at;
            while (_at < text.Length && "+-0123456789.eE".Contains(text[_at])) _at++;
            var number = text[start.._at];
            if (number.Length == 0) throw Fail("expected a value");
            if (number.IndexOfAny(['.', 'e', 'E']) >= 0)
            {
                if (!double.TryParse(number, NumberStyles.Float, CultureInfo.InvariantCulture, out var f)) throw Fail("bad float");
                return new TckCell { Kind = TckKind.Float, Float = f };
            }
            if (!long.TryParse(number, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out var i)) throw Fail("bad integer");
            return new TckCell { Kind = TckKind.Int, Int = i };
        }

        private List<T> Items<T>(string close, Func<T> item)
        {
            var out_ = new List<T>();
            if (Peek(close)) { Take(close); return out_; }
            while (true)
            {
                out_.Add(item());
                if (Peek(",")) { Take(","); continue; }
                Take(close);
                return out_;
            }
        }

        private Dictionary<string, TckCell> Mapping()
        {
            Take("{");
            var entries = Items("}", () =>
            {
                var key = Name();
                Take(":");
                return (key, Value());
            });
            return entries.ToDictionary(e => e.key, e => e.Item2);
        }

        private Dictionary<string, TckCell> Properties() => Peek("{") ? Mapping() : [];

        private TckCell Node()
        {
            Take("(");
            var labels = new HashSet<string>();
            while (Peek(":")) { Take(":"); labels.Add(Name()); }
            var props = Properties();
            Take(")");
            return new TckCell { Kind = TckKind.Node, Labels = labels, Entries = props };
        }

        private TckCell Relationship()
        {
            Take("[");
            Take(":");
            var type = Name();
            var props = Properties();
            Take("]");
            return new TckCell { Kind = TckKind.Relationship, Text = type, Entries = props };
        }

        private TckCell Path()
        {
            Take("<");
            var cell = new TckCell { Kind = TckKind.Path };
            cell.Nodes.Add(Node());
            while (!Peek(">"))
            {
                var forward = !Peek("<-");
                Take(forward ? "-" : "<-");
                cell.Relationships.Add(Relationship());
                Take(forward ? "->" : "-");
                cell.Forward.Add(forward);
                cell.Nodes.Add(Node());
            }
            Take(">");
            return cell;
        }

        public void End()
        {
            Skip();
            if (_at != text.Length) throw Fail("unexpected text");
        }
    }

    /// <summary>Parses one cell of the TCK's value notation.</summary>
    public static TckCell Parse(string text)
    {
        var parser = new Parser(text);
        var cell = parser.Value();
        parser.End();
        return cell;
    }

    /// <summary>A node or relationship as a result gives it.</summary>
    public sealed record Entity(bool IsNode, long Id, HashSet<string> Labels, string Type, long Source, long Target, Dictionary<string, object?> Properties);

    /// <summary>
    /// The one place that recognises entities, by the shape the binding gives them: a map with
    /// a long <c>_id</c> and a <c>_labels</c> list is a node; one with <c>_id</c>, a string
    /// <c>_type</c>, <c>_source</c> and <c>_target</c> a relationship. A map the query builds
    /// with those keys comes back as an entity too: results have no typed node value.
    /// </summary>
    public static Entity? EntityOf(object? value)
    {
        if (value is not IReadOnlyDictionary<string, object?> map || map.GetValueOrDefault("_id") is not long id) return null;
        Dictionary<string, object?> Props(params string[] reserved) =>
            map.Where(kv => !reserved.Contains(kv.Key)).ToDictionary(kv => kv.Key, kv => kv.Value);
        if (map.GetValueOrDefault("_labels") is IEnumerable<object?> labels)
            return new Entity(true, id, labels.OfType<string>().ToHashSet(), "", 0, 0, Props("_id", "_labels"));
        if (map.GetValueOrDefault("_type") is string type && map.GetValueOrDefault("_source") is long source
            && map.GetValueOrDefault("_target") is long target)
            return new Entity(false, id, [], type, source, target, Props("_id", "_type", "_source", "_target"));
        return null;
    }

    /// <summary>The nodes and relationships of a path, which the binding gives as <c>{"$path": {...}}</c>.</summary>
    public static (List<object?> Nodes, List<object?> Edges)? PathOf(object? value)
    {
        if (value is not IReadOnlyDictionary<string, object?> map || map.Count != 1
            || map.GetValueOrDefault("$path") is not IReadOnlyDictionary<string, object?> inner) return null;
        if (inner.GetValueOrDefault("nodes") is not IEnumerable<object?> nodes
            || inner.GetValueOrDefault("edges") is not IEnumerable<object?> edges) return null;
        return (nodes.ToList(), edges.ToList());
    }

    private static bool PropertiesMatch(Dictionary<string, TckCell> expected, IReadOnlyDictionary<string, object?> actual, bool unordered) =>
        expected.Count == actual.Count
        && expected.All(e => actual.TryGetValue(e.Key, out var v) && Matches(e.Value, v, unordered));

    private static bool BagMatches(List<TckCell> expected, List<object?> actual, bool unordered)
    {
        if (expected.Count != actual.Count) return false;
        var left = new List<object?>(actual);
        foreach (var cell in expected)
        {
            var hit = left.FindIndex(v => Matches(cell, v, unordered));
            if (hit < 0) return false;
            left.RemoveAt(hit);
        }
        return true;
    }

    /// <summary>Whether <paramref name="value"/> is the value <paramref name="cell"/> writes, type-exact.</summary>
    public static bool Matches(TckCell cell, object? value, bool unordered)
    {
        switch (cell.Kind)
        {
            case TckKind.Null: return value is null;
            case TckKind.Bool: return value is bool b && b == cell.Bool;
            case TckKind.Int: return value is long l && l == cell.Int;
            case TckKind.Float: return value is double d && (d.Equals(cell.Float) || (double.IsNaN(d) && double.IsNaN(cell.Float)));
            case TckKind.String: return value is string s && s == cell.Text;
            case TckKind.List:
                if (value is not IEnumerable<object?> list || value is string || value is IReadOnlyDictionary<string, object?>) return false;
                var items = list.ToList();
                if (unordered) return BagMatches(cell.Items, items, unordered);
                return items.Count == cell.Items.Count && cell.Items.Select((c, i) => Matches(c, items[i], unordered)).All(x => x);
            case TckKind.Map:
                return value is IReadOnlyDictionary<string, object?> map && EntityOf(value) is null && PathOf(value) is null
                    && PropertiesMatch(cell.Entries, map, unordered);
            case TckKind.Node:
                return EntityOf(value) is { IsNode: true } node && node.Labels.SetEquals(cell.Labels)
                    && PropertiesMatch(cell.Entries, node.Properties, unordered);
            case TckKind.Relationship:
                return EntityOf(value) is { IsNode: false } rel && rel.Type == cell.Text
                    && PropertiesMatch(cell.Entries, rel.Properties, unordered);
            case TckKind.Path:
                if (PathOf(value) is not var (nodes, edges) || nodes.Count != cell.Nodes.Count || edges.Count != cell.Relationships.Count) return false;
                for (var i = 0; i < edges.Count; i++)
                {
                    var (start, end) = cell.Forward[i] ? (nodes[i], nodes[i + 1]) : (nodes[i + 1], nodes[i]);
                    if (EntityOf(edges[i]) is not { IsNode: false } edge || EntityOf(start) is not { } startNode || EntityOf(end) is not { } endNode
                        || edge.Source != startNode.Id || edge.Target != endNode.Id || !Matches(cell.Relationships[i], edges[i], unordered)) return false;
                }
                return cell.Nodes.Select((c, i) => Matches(c, nodes[i], unordered)).All(x => x);
        }
        return false;
    }

    /// <summary>The rows of a <c>values:</c> table, each a list of cell texts (<c>\|</c> is a pipe in a cell).</summary>
    public static List<List<string>> TableRows(string table)
    {
        var rows = new List<List<string>>();
        foreach (var raw in table.Split('\n'))
        {
            var line = raw.Trim();
            if (line.Length == 0) continue;
            var inner = line.StartsWith('|') && line.EndsWith('|') && line.Length >= 2 ? line[1..^1] : line;
            var cells = new List<string>();
            var cell = new StringBuilder();
            for (var i = 0; i < inner.Length; i++)
            {
                if (inner[i] == '\\' && i + 1 < inner.Length && inner[i + 1] == '|') { cell.Append('|'); i++; }
                else if (inner[i] == '|') { cells.Add(cell.ToString().Trim()); cell.Clear(); }
                else cell.Append(inner[i]);
            }
            cells.Add(cell.ToString().Trim());
            rows.Add(cells);
        }
        return rows;
    }

    private static bool IsTemporal(object? value) =>
        value is DateOnly or TimeOnly or DateTime or DateTimeOffset or GrafeoTime or GrafeoDuration or TimeSpan;

    /// <summary>
    /// <paramref name="value"/> with each temporal in it written as its Cypher toString(), the
    /// engine's own text: how a <c>values_dialect: tck</c> file compares temporals.
    /// </summary>
    public static object? TemporalsAsText(GrafeoDB db, object? value)
    {
        if (IsTemporal(value))
        {
            var result = db.ExecuteLanguage("cypher", "RETURN toString($v) AS s", new Dictionary<string, object?> { ["v"] = value });
            return result.Rows[0]["s"];
        }
        if (value is IReadOnlyDictionary<string, object?> map)
            return map.ToDictionary(kv => kv.Key, kv => TemporalsAsText(db, kv.Value));
        if (value is IEnumerable<object?> list)
            return list.Select(v => TemporalsAsText(db, v)).ToList();
        return value;
    }

    /// <summary>
    /// Asserts the result holds the rows of <paramref name="table"/>, a <c>values:</c> table whose
    /// first row names the columns: in order when <paramref name="ordered"/>, otherwise as a bag.
    /// </summary>
    public static void AssertValues(GrafeoDB db, QueryResult result, string table, bool ordered, bool unordered, bool tckDialect)
    {
        var rows = TableRows(table);
        var header = rows[0].Where(c => c.Length > 0).ToList();
        var columns = result.Columns.ToList();
        if (!columns.SequenceEqual(header))
            throw new Xunit.Sdk.XunitException($"Columns differ: [{string.Join(", ", columns)}], expected [{string.Join(", ", header)}]");
        var expected = rows.Skip(1).Select(row => row.Select(Parse).ToList()).ToList();
        var actual = result.Rows
            .Select(row => columns.Select(c => tckDialect ? TemporalsAsText(db, row[c]) : row[c]).ToList())
            .ToList();
        bool RowMatches(List<TckCell> cells, List<object?> values) =>
            cells.Count == values.Count && cells.Select((c, i) => Matches(c, values[i], unordered)).All(x => x);
        var holds = expected.Count == actual.Count;
        if (holds && ordered)
        {
            holds = expected.Select((cells, i) => RowMatches(cells, actual[i])).All(x => x);
        }
        else if (holds)
        {
            var left = new List<List<object?>>(actual);
            foreach (var cells in expected)
            {
                var hit = left.FindIndex(values => RowMatches(cells, values));
                if (hit < 0) { holds = false; break; }
                left.RemoveAt(hit);
            }
        }
        if (!holds)
        {
            var shown = JsonSerializer.Serialize(actual.Select(r => r.Select(v => v?.ToString())));
            throw new Xunit.Sdk.XunitException($"Rows differ{(ordered ? " (in order)" : "")}:\nactual:   {shown}\nexpected: {table}");
        }
    }

    /// <summary>What a graph holds, as the openCypher TCK observes side effects.</summary>
    public static Dictionary<string, HashSet<string>> ObserveGraph(GrafeoDB db)
    {
        var seen = new Dictionary<string, HashSet<string>>
        {
            ["nodes"] = [], ["relationships"] = [], ["labels"] = [], ["properties"] = [],
        };
        foreach (var query in new[] { "MATCH (n) RETURN n", "MATCH ()-[r]->() RETURN r" })
        {
            foreach (var row in db.Execute(query).Rows)
            {
                foreach (var value in row.Values)
                {
                    if (EntityOf(value) is not { } e) continue;
                    var kind = e.IsNode ? "nodes" : "relationships";
                    var id = e.Id.ToString(CultureInfo.InvariantCulture);
                    seen[kind].Add(id);
                    seen["labels"].UnionWith(e.Labels);
                    foreach (var (key, prop) in e.Properties)
                        seen["properties"].Add($"{kind}|{id}|{key}|{prop?.GetType().Name}|{Convert.ToString(prop, CultureInfo.InvariantCulture)}");
                }
            }
        }
        return seen;
    }

    /// <summary>Asserts the side effects from <paramref name="before"/> to <paramref name="after"/>.</summary>
    public static void AssertSideEffects(Dictionary<string, HashSet<string>> before, Dictionary<string, HashSet<string>> after, Dictionary<string, int> expected)
    {
        var differ = new List<string>();
        foreach (var key in new[] { "nodes", "relationships", "labels", "properties" })
        {
            var added = after[key].Count(x => !before[key].Contains(x));
            var removed = before[key].Count(x => !after[key].Contains(x));
            foreach (var (name, count) in new[] { ($"+{key}", added), ($"-{key}", removed) })
            {
                var want = expected.GetValueOrDefault(name);
                if (count != want) differ.Add($"{name}: {count}, expected {want}");
            }
        }
        if (differ.Count > 0) throw new Xunit.Sdk.XunitException($"Side effects differ: {string.Join("; ", differ)}");
    }
}

/// <summary>The shared value cases (tests/spec/runners/values_cases.json).</summary>
public class TckValueCaseTests
{
    private static readonly JsonElement CaseFile = JsonDocument.Parse(File.ReadAllText(
        Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "values_cases.json"))).RootElement;

    /// <summary>A neutral value of the case file as the C# binding returns it.</summary>
    private static object? Native(GrafeoDB db, JsonElement actual)
    {
        switch (actual.ValueKind)
        {
            case JsonValueKind.Null: return null;
            case JsonValueKind.True: return true;
            case JsonValueKind.False: return false;
            case JsonValueKind.Array: return actual.EnumerateArray().Select(a => Native(db, a)).ToList();
        }
        var (kind, inner) = actual.EnumerateObject().Select(p => (p.Name, p.Value)).Single();
        Dictionary<string, object?> Props(JsonElement p) => p.EnumerateObject().ToDictionary(x => x.Name, x => Native(db, x.Value));
        switch (kind)
        {
            case "int": return long.Parse(inner.GetString()!, CultureInfo.InvariantCulture);
            case "float":
                return inner.GetString() switch
                {
                    "NaN" => double.NaN,
                    "Inf" => double.PositiveInfinity,
                    "-Inf" => double.NegativeInfinity,
                    var t => double.Parse(t!, CultureInfo.InvariantCulture),
                };
            case "string": return inner.GetString();
            case "map": return Props(inner);
            case "node":
            {
                var map = Props(inner.GetProperty("properties"));
                map["_id"] = inner.GetProperty("id").GetInt64();
                map["_labels"] = inner.GetProperty("labels").EnumerateArray().Select(l => (object?)l.GetString()).ToList();
                return map;
            }
            case "relationship":
            {
                var map = Props(inner.GetProperty("properties"));
                map["_id"] = inner.GetProperty("id").GetInt64();
                map["_type"] = inner.GetProperty("type").GetString();
                map["_source"] = inner.GetProperty("source").GetInt64();
                map["_target"] = inner.GetProperty("target").GetInt64();
                return map;
            }
            case "path":
            {
                var nodes = inner.GetProperty("nodes").EnumerateArray()
                    .Select(n => Native(db, JsonDocument.Parse($"{{\"node\": {n.GetRawText()}}}").RootElement)).ToList();
                var edges = inner.GetProperty("relationships").EnumerateArray()
                    .Select(r => Native(db, JsonDocument.Parse($"{{\"relationship\": {r.GetRawText()}}}").RootElement)).ToList();
                return new Dictionary<string, object?> { ["$path"] = new Dictionary<string, object?> { ["nodes"] = nodes, ["edges"] = edges } };
            }
            case "cypher": return db.ExecuteCypher($"RETURN {inner.GetString()} AS v").Rows[0]["v"];
        }
        throw new InvalidOperationException($"unknown neutral kind {kind}");
    }

    private static bool HasGap(JsonElement element) =>
        element.TryGetProperty("runner_gaps", out var gaps) && gaps.TryGetProperty("csharp", out _);

    [Fact]
    public void TheSharedValueCasesAgree()
    {
        using var db = GrafeoDB.Memory();
        var wrong = new List<string>();
        foreach (var c in CaseFile.GetProperty("cases").EnumerateArray())
        {
            var cell = TckValues.Parse(c.GetProperty("cell").GetString()!);
            var value = Native(db, c.GetProperty("actual"));
            if (c.TryGetProperty("dialect", out var dialect) && dialect.GetString() == "tck")
                value = TckValues.TemporalsAsText(db, value);
            var unordered = c.TryGetProperty("unordered_lists", out var u) && u.GetBoolean();
            // A known limitation, or a gap of this binding, gives the opposite verdict until fixed
            var gap = c.TryGetProperty("limitation", out _) || HasGap(c);
            if (TckValues.Matches(cell, value, unordered) != (c.GetProperty("match").GetBoolean() != gap))
                wrong.Add($"{c.GetProperty("cell").GetString()} vs {c.GetProperty("actual").GetRawText()}");
        }
        foreach (var trip in CaseFile.GetProperty("round_trips").EnumerateArray())
        {
            var value = Native(db, JsonDocument.Parse($"{{\"cypher\": {trip.GetProperty("cypher").GetRawText()}}}").RootElement);
            var holds = Equals(TckValues.TemporalsAsText(db, value), trip.GetProperty("text").GetString());
            if (holds == HasGap(trip)) wrong.Add($"round trip {trip.GetProperty("cypher").GetString()}");
        }
        Assert.Empty(wrong);
    }

    [Fact]
    public void MalformedCellsAreRefused()
    {
        foreach (var text in new[] { "'open", "[1, 2", "1 2", "(:A", "{k 1}", "", "<(:A)-[:T]-(:B)>" })
            Assert.Throws<FormatException>(() => TckValues.Parse(text));
    }
}
