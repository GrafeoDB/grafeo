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
        return Parse(content);
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
                case "language":
                    tc.Language = Unquote(value); ctx.Idx++; break;
                case "dataset":
                    tc.Dataset = Unquote(value); ctx.Idx++; break;
                case "setup":
                    ctx.Idx++; tc.Setup = ParseStringList(ctx); break;
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

        while (!ctx.AtEnd)
        {
            var line = ctx.Lines[ctx.Idx];
            var trimmed = line.Trim();

            if (trimmed.StartsWith('#')) { ctx.Idx++; continue; }
            if (trimmed.StartsWith("- name:")) break;
            if (!line.StartsWith(' ') && !line.StartsWith('\t') && !string.IsNullOrEmpty(trimmed)) break;
            if (string.IsNullOrEmpty(trimmed)) { ctx.Idx++; continue; }

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
                case "hash":
                    expect.Hash = Unquote(value); ctx.Idx++; break;
                case "precision":
                    if (int.TryParse(value, out var precVal)) expect.Precision = precVal;
                    ctx.Idx++; break;
                case "columns":
                    expect.Columns = ParseYamlList(value); ctx.Idx++; break;
                case "rows":
                    ctx.Idx++; expect.Rows = ParseRows(ctx); break;
                default:
                    ctx.Idx++; break;
            }
        }

        return expect;
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
