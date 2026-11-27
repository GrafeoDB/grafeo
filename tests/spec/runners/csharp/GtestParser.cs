// Line-based parser for .gtest spec files (no YAML dependency).
// Ported from the Node.js parser at tests/spec/runners/node/parser.mjs.

namespace SpecRunner;

/// <summary>
/// Parses .gtest files into <see cref="GtestFile"/> structures using a
/// line-based approach that mirrors the Node.js and Rust parsers.
/// </summary>
public static class GtestParser
{
    // Simple mutable context that tracks our position in the file.
    private sealed class ParseContext
    {
        public readonly string[] Lines;
        public int Idx;

        public ParseContext(string[] lines)
        {
            Lines = lines;
            Idx = 0;
        }

        public bool AtEnd => Idx >= Lines.Length;
    }

    /// <summary>Parse a .gtest file at the given path.</summary>
    public static GtestFile ParseFile(string path)
    {
        var content = File.ReadAllText(path);
        var file = Parse(content);
        foreach (var tc in file.Tests)
            CheckMarks(tc, path);
        if (file.Meta.Model == "rdf")
        {
            var lpg = file.Tests.FirstOrDefault(tc =>
                (tc.Language ?? file.Meta.Language) == "graphql" || tc.Variants.ContainsKey("graphql"));
            if (lpg is not null)
                throw new FormatException($"test '{lpg.Name}': {GraphqlOnRdf}");
        }
        return file;
    }

    /// <summary>
    /// Why a <c>graphql</c> case in a <c>model: rdf</c> file is refused: <c>graphql</c> reads the
    /// property graph. Every runner says the same.
    /// </summary>
    public const string GraphqlOnRdf =
        "`language: graphql` reads the property graph; on `model: rdf` write `language: graphql-rdf`";

    /// <summary>
    /// The rules of the marks, as every runner checks them: an <c>xfail:</c> names its spec and
    /// its issue, a case is never both <c>skip:</c> and <c>xfail:</c>, and <c>probe:</c> is for
    /// the runner probes only, in a file under a <c>spec-probes</c> directory.
    /// </summary>
    private static void CheckMarks(TestCase tc, string path)
    {
        if (tc.Xfail is not null)
        {
            if (string.IsNullOrWhiteSpace(tc.Xfail.Spec))
                throw new FormatException($"test '{tc.Name}': an xfail names its `spec:`");
            if (string.IsNullOrWhiteSpace(tc.Xfail.Issue))
                throw new FormatException($"test '{tc.Name}': an xfail names its `issue:`");
        }
        if (tc.Skip is not null && tc.Xfail is not null)
            throw new FormatException(
                $"test '{tc.Name}': a case is either skipped or expected to fail, not both");
        if (tc.Probe is not null)
        {
            if (!path.Split('/', '\\').Contains("spec-probes"))
                throw new FormatException(
                    $"test '{tc.Name}': `probe:` is for the runner probes only, under a `spec-probes` directory, not in {path}");
            if (tc.Probe != "runner_error")
                throw new FormatException($"test '{tc.Name}': unknown probe '{tc.Probe}' (known: runner_error)");
        }
    }

    /// <summary>The <c>spec:</c> and <c>issue:</c> under an <c>xfail:</c> at the current line.</summary>
    private static Xfail ParseXfail(ParseContext ctx)
    {
        static int Indent(string line) => line.Length - line.TrimStart().Length;
        var own = Indent(ctx.Lines[ctx.Idx]);
        ctx.Idx++;
        var xfail = new Xfail();
        while (!ctx.AtEnd)
        {
            var line = ctx.Lines[ctx.Idx];
            var trimmed = line.Trim();
            if (trimmed.Length == 0 || trimmed.StartsWith('#')) { ctx.Idx++; continue; }
            if (Indent(line) <= own) break;
            var kv = ParseKV(trimmed);
            switch (kv?.Key)
            {
                case "spec": xfail.Spec = Unquote(kv.Value.Value); break;
                case "issue": xfail.Issue = Unquote(kv.Value.Value); break;
                case "passes_in": xfail.PassesIn = KnownRunners(ParseYamlList(kv.Value.Value)); break;
                default: throw new FormatException($"unknown xfail key at line {ctx.Idx + 1}: '{trimmed}'");
            }
            ctx.Idx++;
        }
        return xfail;
    }

    /// <summary>Parse .gtest content.</summary>
    public static GtestFile Parse(string content)
    {
        var lines = content.Split(["\r\n", "\n"], StringSplitOptions.None);
        var ctx = new ParseContext(lines);

        SkipBlankAndComments(ctx);
        var meta = ParseMeta(ctx);
        SkipBlankAndComments(ctx);
        var tests = ParseTests(ctx);

        return new GtestFile { Meta = meta, Tests = tests };
    }

    // =========================================================================
    // Meta block
    // =========================================================================

    private static Meta ParseMeta(ParseContext ctx)
    {
        var meta = new Meta();
        ExpectLine(ctx, "meta:");

        while (!ctx.AtEnd)
        {
            SkipBlankAndComments(ctx);
            if (ctx.AtEnd) break;

            var line = ctx.Lines[ctx.Idx];
            if (!line.StartsWith(' ') && !line.StartsWith('\t')) break;

            var kv = ParseKV(line.Trim());
            if (kv == null) { ctx.Idx++; continue; }

            var (key, value) = kv.Value;
            switch (key)
            {
                case "language": meta.Language = value; break;
                case "model": meta.Model = value; break;
                case "section": meta.Section = Unquote(value); break;
                case "title": meta.Title = value; break;
                case "dataset": meta.Dataset = value; break;
                case "requires": meta.Requires = ParseYamlList(value); break;
                case "tags": meta.Tags = ParseYamlList(value); break;
                case "values_dialect":
                    meta.ValuesDialect = Unquote(value);
                    if (meta.ValuesDialect is not ("strict" or "tck"))
                        throw new FormatException($"unknown values_dialect '{meta.ValuesDialect}' (known: strict, tck)");
                    break;
            }
            ctx.Idx++;
        }

        return meta;
    }

    // =========================================================================
    // Tests list
    // =========================================================================

    private static List<TestCase> ParseTests(ParseContext ctx)
    {
        SkipBlankAndComments(ctx);
        ExpectLine(ctx, "tests:");

        var tests = new List<TestCase>();
        while (!ctx.AtEnd)
        {
            SkipBlankAndComments(ctx);
            if (ctx.AtEnd) break;

            var trimmed = ctx.Lines[ctx.Idx].Trim();
            if (trimmed.StartsWith("- name:"))
                tests.Add(ParseSingleTest(ctx));
            else
                break;
        }
        return tests;
    }

    private static TestCase ParseSingleTest(ParseContext ctx)
    {
        var tc = new TestCase();

        // First line: "- name: xxx"
        var first = ctx.Lines[ctx.Idx].Trim();
        var kv = ParseKV(first[2..]); // strip "- "
        if (kv != null) tc.Name = Unquote(kv.Value.Value);
        ctx.Idx++;

        while (!ctx.AtEnd)
        {
            var line = ctx.Lines[ctx.Idx];
            var trimmed = line.Trim();

            if (trimmed.StartsWith('#')) { ctx.Idx++; continue; }
            if (trimmed.StartsWith("- name:")) break;
            if (string.IsNullOrEmpty(trimmed)) { ctx.Idx++; continue; }

            var kv2 = ParseKV(trimmed);
            if (kv2 == null) { ctx.Idx++; continue; }

            var (key, value) = kv2.Value;
            switch (key)
            {
                case "query":
                    if (value == "|")
                    {
                        tc.Query = ParseBlockScalar(ctx);
                    }
                    else
                    {
                        tc.Query = Unquote(value);
                        ctx.Idx++;
                    }
                    break;
                case "skip":
                    tc.Skip = Unquote(value); ctx.Idx++; break;
                case "xfail":
                    tc.Xfail = ParseXfail(ctx); break;
                case "probe":
                    tc.Probe = Unquote(value); ctx.Idx++; break;
                case "language":
                    tc.Language = Unquote(value); ctx.Idx++; break;
                case "dataset":
                    tc.Dataset = Unquote(value); ctx.Idx++; break;
                case "setup":
                    ctx.Idx++; tc.Setup = ParseStringList(ctx); break;
                case "setup_gql":
                    ctx.Idx++; tc.SetupGql = ParseStringList(ctx); break;
                case "runner_gap":
                    tc.RunnerGap = ParseRunnerGap(ctx); break;
                case "config":
                    tc.Config = ParseConfig(value); ctx.Idx++; break;
                case "statements":
                    ctx.Idx++; tc.Statements = ParseStringList(ctx); break;
                case "tags":
                    tc.Tags = ParseYamlList(value); ctx.Idx++; break;
                case "requires":
                    tc.Requires = ParseYamlList(value); ctx.Idx++; break;
                case "params":
                    ctx.Idx++; tc.Params = ParseParams(ctx); break;
                case "expect":
                    ctx.Idx++; tc.Expect = ParseExpectBlock(ctx); break;
                case "variants":
                    ctx.Idx++; tc.Variants = ParseMap(ctx, 6); break;
                default:
                    ctx.Idx++; break;
            }
        }

        return tc;
    }

    // =========================================================================
    // Expect block
    // =========================================================================

    private static Expect ParseExpectBlock(ParseContext ctx)
    {
        var expect = new Expect();
        // The indent of the expectation's own keys: a line indented less is the
        // test's next key (a skip: or note: after expect:).
        int? childIndent = null;

        while (!ctx.AtEnd)
        {
            var line = ctx.Lines[ctx.Idx];
            var trimmed = line.Trim();

            if (trimmed.StartsWith('#')) { ctx.Idx++; continue; }
            if (trimmed.StartsWith("- name:")) break;
            if (!line.StartsWith(' ') && !line.StartsWith('\t') && !string.IsNullOrEmpty(trimmed)) break;
            if (string.IsNullOrEmpty(trimmed)) { ctx.Idx++; continue; }
            var indent = line.Length - line.TrimStart().Length;
            childIndent ??= indent;
            if (indent < childIndent) break;

            var kv = ParseKV(trimmed);
            if (kv == null) break;

            var (key, value) = kv.Value;
            switch (key)
            {
                case "ordered":
                    expect.Ordered = value == "true"; ctx.Idx++; break;
                case "count":
                    if (int.TryParse(value, out var countVal)) expect.Count = countVal;
                    ctx.Idx++; break;
                case "empty":
                    expect.Empty = value == "true"; ctx.Idx++; break;
                case "error":
                    expect.Error = Unquote(value); ctx.Idx++; break;
                case "gql_status":
                    expect.GqlStatus = Unquote(value); ctx.Idx++; break;
                case "hash":
                    expect.Hash = Unquote(value); ctx.Idx++; break;
                case "precision":
                    if (int.TryParse(value, out var precVal)) expect.Precision = precVal;
                    ctx.Idx++; break;
                case "columns":
                    expect.Columns = ParseYamlList(value); ctx.Idx++; break;
                case "warnings":
                    expect.Warnings = ParseYamlList(value); ctx.Idx++; break;
                case "rows":
                    ctx.Idx++; expect.Rows = ParseRows(ctx); break;
                case "terms":
                    ctx.Idx++; expect.Terms = ParseRows(ctx); break;
                case "values":
                    if (value != "|") throw new FormatException("`values:` is a block scalar (`|`)");
                    expect.Values = ParseBlockScalar(ctx).Trim(); break;
                case "unordered_lists":
                    expect.UnorderedLists = value == "true"; ctx.Idx++; break;
                case "side_effects":
                    expect.SideEffects = ParseSideEffects(ctx, value); break;
                default:
                    expect.UnknownKey ??= key;
                    ctx.Idx++; break;
            }
        }

        var others = expect.Rows.Count > 0 || expect.Terms.Count > 0 || expect.Count != null || expect.Empty;
        if (expect.Values != null && (others || expect.Error != null || expect.GqlStatus != null))
            throw new FormatException("`values:` excludes `rows`, `terms`, `count`, `empty`, `error` and `gql_status`");
        return expect;
    }

    /// <summary>The runners a <c>runner_gap:</c> or <c>passes_in:</c> may name.</summary>
    private static readonly HashSet<string> Runners = ["rust", "python", "node", "wasm", "go", "csharp", "dart"];

    /// <summary>The keys a <c>config:</c> may set: Config settings every binding exposes.</summary>
    internal static readonly string[] ConfigKeys = ["path_search_budget"];

    /// <summary>A <c>config:</c> value as the file writes it: an inline map of
    /// <see cref="ConfigKeys"/> to positive integers (<c>{path_search_budget: 1048576}</c>).
    /// An unknown or repeated key is refused, never ignored.</summary>
    internal static Dictionary<string, long> ParseConfig(string text)
    {
        var inner = text.Trim();
        if (!inner.StartsWith('{') || !inner.EndsWith('}') || inner[1..^1].Trim().Length == 0)
            throw new FormatException($"config must be an inline map of settings, got '{text}'");
        var settings = new Dictionary<string, long>();
        foreach (var entry in inner[1..^1].Split(','))
        {
            var colon = entry.IndexOf(':');
            var key = colon < 0 ? "" : entry[..colon].Trim();
            var value = colon < 0 ? "" : entry[(colon + 1)..].Trim();
            if (key.Length == 0)
                throw new FormatException($"config entry '{entry.Trim()}' is not 'key: value'");
            if (settings.ContainsKey(key))
                throw new FormatException($"config sets '{key}' twice");
            if (!ConfigKeys.Contains(key))
                throw new FormatException($"unknown config key '{key}' (known: {string.Join(", ", ConfigKeys)})");
            if (value.Length == 0 || !value.All(char.IsAsciiDigit)
                || !long.TryParse(value, out var number) || number == 0)
                throw new FormatException($"config '{key}' needs a positive integer, got '{value}'");
            settings[key] = number;
        }
        return settings;
    }

    /// <summary>The keys of <c>side_effects:</c>, as the openCypher TCK names them.</summary>
    private static readonly HashSet<string> SideEffectKeys =
        ["+nodes", "-nodes", "+relationships", "-relationships", "+labels", "-labels", "+properties", "-properties"];

    private static List<string> KnownRunners(List<string> runners)
    {
        foreach (var runner in runners)
            if (!Runners.Contains(runner)) throw new FormatException($"unknown runner '{runner}'");
        return runners;
    }

    /// <summary>The <c>key: value</c> lines indented under the key at the current line.</summary>
    private static List<(string Key, string Value)> NestedPairs(ParseContext ctx)
    {
        static int Indent(string line) => line.Length - line.TrimStart().Length;
        var own = Indent(ctx.Lines[ctx.Idx]);
        ctx.Idx++;
        var pairs = new List<(string Key, string Value)>();
        while (!ctx.AtEnd)
        {
            var line = ctx.Lines[ctx.Idx];
            var trimmed = line.Trim();
            if (trimmed.Length == 0 || trimmed.StartsWith('#')) { ctx.Idx++; continue; }
            if (Indent(line) <= own) break;
            var kv = ParseKV(trimmed) ?? throw new FormatException($"not a key at line {ctx.Idx + 1}: '{trimmed}'");
            pairs.Add(kv);
            ctx.Idx++;
        }
        return pairs;
    }

    /// <summary>A <c>runner_gap:</c> at the current line: its runners, issue and reason, all required.</summary>
    private static RunnerGap ParseRunnerGap(ParseContext ctx)
    {
        var gap = new RunnerGap();
        foreach (var (key, value) in NestedPairs(ctx))
        {
            switch (key)
            {
                case "runners": gap.Runners = KnownRunners(ParseYamlList(value)); break;
                case "issue": gap.Issue = Unquote(value); break;
                case "reason": gap.Reason = Unquote(value); break;
                default: throw new FormatException($"unknown runner_gap key '{key}'");
            }
        }
        if (gap.Runners.Count == 0 || gap.Issue.Trim().Length == 0 || gap.Reason.Trim().Length == 0)
            throw new FormatException("a runner_gap names its `runners:`, `issue:` and `reason:`");
        return gap;
    }

    /// <summary>The counts of a <c>side_effects:</c> key whose value is <paramref name="value"/>:
    /// <c>{}</c>, or one key per line below it.</summary>
    private static Dictionary<string, int> ParseSideEffects(ParseContext ctx, string value)
    {
        if (value == "{}") { ctx.Idx++; return new Dictionary<string, int>(); }
        if (value.Length > 0) throw new FormatException($"`side_effects:` is `{{}}` or one key per line, not '{value}'");
        var counts = new Dictionary<string, int>();
        foreach (var (key, count) in NestedPairs(ctx))
        {
            if (!SideEffectKeys.Contains(key) || !int.TryParse(count, out var n))
                throw new FormatException($"bad side effect '{key}: {count}'");
            counts[key] = n;
        }
        return counts;
    }

    private static List<List<string>> ParseRows(ParseContext ctx)
    {
        var rows = new List<List<string>>();
        while (!ctx.AtEnd)
        {
            var trimmed = ctx.Lines[ctx.Idx].Trim();
            if (trimmed.StartsWith('#')) { ctx.Idx++; continue; }
            if (string.IsNullOrEmpty(trimmed)) { ctx.Idx++; continue; }

            if (trimmed.StartsWith("- ["))
            {
                rows.Add(ParseInlineList(trimmed[2..]));
                ctx.Idx++;
            }
            else
            {
                break;
            }
        }
        return rows;
    }

    // =========================================================================
    // YAML primitives
    // =========================================================================

    /// <summary>
    /// Split a string on the first unquoted colon. Returns null if no valid
    /// key-value pair is found. Respects single and double quotes so that
    /// colons inside quoted strings or query bodies are not treated as
    /// separators.
    /// </summary>
    internal static (string Key, string Value)? ParseKV(string s)
    {
        var inSingle = false;
        var inDouble = false;

        for (var i = 0; i < s.Length; i++)
        {
            var c = s[i];
            if (c == '\'' && !inDouble) inSingle = !inSingle;
            else if (c == '"' && !inSingle) inDouble = !inDouble;
            else if (c == ':' && !inSingle && !inDouble)
            {
                var key = s[..i].Trim();
                var value = s[(i + 1)..].Trim();
                if (!string.IsNullOrEmpty(key))
                    return (key, value);
            }
        }

        return null;
    }

    /// <summary>Whether <paramref name="s"/> is written in single or double quotes.</summary>
    internal static bool IsQuoted(string s)
    {
        s = s.Trim();
        return s.Length >= 2 &&
            ((s[0] == '"' && s[^1] == '"') || (s[0] == '\'' && s[^1] == '\''));
    }

    /// <summary>
    /// Strip surrounding quotes and unescape YAML-level escapes only.
    /// Does NOT process \n or \t: those are GQL string escapes handled
    /// by the engine's parser.
    /// </summary>
    internal static string Unquote(string s)
    {
        s = s.Trim();
        if (IsQuoted(s))
        {
            return s[1..^1]
                .Replace("\\\\", "\x00")
                .Replace("\\\"", "\"")
                .Replace("\\'", "'")
                .Replace("\x00", "\\");
        }
        return s;
    }

    /// <summary>
    /// Parse a YAML-style inline list: <c>[val1, val2]</c>.
    /// Handles nested brackets and braces for complex row values.
    /// </summary>
    internal static List<string> ParseInlineList(string s)
    {
        s = s.Trim();
        if (!s.StartsWith('[') || !s.EndsWith(']'))
            return [Unquote(s)];

        var inner = s[1..^1];
        var items = new List<string>();
        var current = new System.Text.StringBuilder();
        var depth = 0;
        var inSingle = false;
        var inDouble = false;

        foreach (var c in inner)
        {
            if (c == '\'' && !inDouble && depth == 0)
            {
                inSingle = !inSingle;
                current.Append(c);
            }
            else if (c == '"' && !inSingle && depth == 0)
            {
                inDouble = !inDouble;
                current.Append(c);
            }
            else if ((c == '[' || c == '{') && !inSingle && !inDouble)
            {
                depth++;
                current.Append(c);
            }
            else if ((c == ']' || c == '}') && !inSingle && !inDouble)
            {
                depth--;
                current.Append(c);
            }
            else if (c == ',' && depth == 0 && !inSingle && !inDouble)
            {
                items.Add(Unquote(current.ToString().Trim()));
                current.Clear();
            }
            else
            {
                current.Append(c);
            }
        }

        var remainder = current.ToString().Trim();
        if (!string.IsNullOrEmpty(remainder))
            items.Add(Unquote(remainder));

        return items;
    }

    /// <summary>
    /// Parse a simple YAML inline list used for tags, requires, columns:
    /// <c>[a, b, c]</c> or a bare scalar value.
    /// </summary>
    private static List<string> ParseYamlList(string s)
    {
        s = s.Trim();
        if (s == "[]" || string.IsNullOrEmpty(s)) return [];

        if (s.StartsWith('[') && s.EndsWith(']'))
        {
            return s[1..^1]
                .Split(',')
                .Select(v => Unquote(v.Trim()))
                .Where(v => !string.IsNullOrEmpty(v))
                .ToList();
        }

        return [Unquote(s)];
    }

    /// <summary>
    /// Parse a YAML list of <c>- item</c> entries. Supports block scalar
    /// values (<c>|</c>) for multi-line items.
    /// </summary>
    private static List<string> ParseStringList(ParseContext ctx)
    {
        var items = new List<string>();
        while (!ctx.AtEnd)
        {
            var trimmed = ctx.Lines[ctx.Idx].Trim();
            if (trimmed.StartsWith('#')) { ctx.Idx++; continue; }
            if (string.IsNullOrEmpty(trimmed)) { ctx.Idx++; continue; }

            if (trimmed.StartsWith("- "))
            {
                var value = trimmed[2..];
                if (value == "|")
                {
                    items.Add(ParseBlockScalar(ctx));
                }
                else
                {
                    items.Add(Unquote(value));
                    ctx.Idx++;
                }
            }
            else
            {
                break;
            }
        }
        return items;
    }

    /// <summary>
    /// Parse a YAML-style key-value map at the given minimum indentation.
    /// Supports block scalar values (<c>|</c>).
    /// </summary>
    private static Dictionary<string, string> ParseMap(ParseContext ctx, int minIndent)
    {
        var map = new Dictionary<string, string>();
        while (!ctx.AtEnd)
        {
            var line = ctx.Lines[ctx.Idx];
            var trimmed = line.Trim();

            if (trimmed.StartsWith('#') || string.IsNullOrEmpty(trimmed))
            { ctx.Idx++; continue; }

            if (trimmed.StartsWith("- name:")) break;

            var indent = line.Length - line.TrimStart().Length;
            if (indent < minIndent) break;

            var kv = ParseKV(trimmed);
            if (kv != null)
            {
                if (kv.Value.Value == "|")
                {
                    map[kv.Value.Key] = ParseBlockScalar(ctx);
                }
                else
                {
                    map[kv.Value.Key] = Unquote(kv.Value.Value);
                    ctx.Idx++;
                }
            }
            else
            {
                break;
            }
        }
        return map;
    }

    /// <summary>
    /// Parse the params entries (indent 6 or deeper). Each value keeps its
    /// quotes, so <c>SpecTests.ParamValue</c> can tell a quoted string from a
    /// bare value.
    /// </summary>
    private static Dictionary<string, string> ParseParams(ParseContext ctx)
    {
        var parameters = new Dictionary<string, string>();
        while (!ctx.AtEnd)
        {
            var line = ctx.Lines[ctx.Idx];
            var trimmed = line.Trim();

            if (trimmed.StartsWith('#') || string.IsNullOrEmpty(trimmed))
            { ctx.Idx++; continue; }

            var indent = line.Length - line.TrimStart().Length;
            if (indent < 6) break;

            var kv = ParseKV(trimmed);
            if (kv == null) break;
            parameters[kv.Value.Key] = kv.Value.Value;
            ctx.Idx++;
        }
        return parameters;
    }

    /// <summary>
    /// Parse a YAML block scalar (indicated by <c>|</c>). Collects indented
    /// continuation lines and joins them with newlines, trimming trailing
    /// whitespace.
    /// </summary>
    private static string ParseBlockScalar(ParseContext ctx)
    {
        ctx.Idx++; // skip the "|" line
        if (ctx.AtEnd) return "";

        var blockIndent = ctx.Lines[ctx.Idx].Length - ctx.Lines[ctx.Idx].TrimStart().Length;
        var parts = new List<string>();

        while (!ctx.AtEnd)
        {
            var line = ctx.Lines[ctx.Idx];
            var trimmed = line.Trim();

            if (string.IsNullOrEmpty(trimmed))
            {
                parts.Add("");
                ctx.Idx++;
                continue;
            }

            var indent = line.Length - line.TrimStart().Length;
            if (indent < blockIndent) break;

            parts.Add(line[blockIndent..]);
            ctx.Idx++;
        }

        return string.Join("\n", parts).TrimEnd();
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    private static void SkipBlankAndComments(ParseContext ctx)
    {
        while (!ctx.AtEnd)
        {
            var trimmed = ctx.Lines[ctx.Idx].Trim();
            if (string.IsNullOrEmpty(trimmed) || trimmed.StartsWith('#'))
                ctx.Idx++;
            else
                break;
        }
    }

    private static void ExpectLine(ParseContext ctx, string expected)
    {
        SkipBlankAndComments(ctx);
        if (ctx.AtEnd || ctx.Lines[ctx.Idx].Trim() != expected)
        {
            var got = ctx.AtEnd ? "<EOF>" : ctx.Lines[ctx.Idx].Trim();
            throw new InvalidOperationException(
                $"Expected '{expected}' at line {ctx.Idx + 1}, got '{got}'");
        }
        ctx.Idx++;
    }
}
