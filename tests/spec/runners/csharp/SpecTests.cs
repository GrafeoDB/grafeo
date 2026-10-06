// xUnit spec runner for .gtest files through the C# Grafeo bindings.
//
// Discovers all .gtest files under tests/spec/, parses them, and creates
// parameterized xUnit tests that execute queries and assert expected results.

using System;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

using Grafeo;

using Xunit;

namespace SpecRunner;

/// <summary>
/// xUnit test class that discovers and runs .gtest spec files.
/// Each test case is a parameterized <c>[Theory]</c> test backed by
/// <c>[MemberData]</c> that enumerates all discovered .gtest files.
/// </summary>
public class SpecTests : IDisposable
{
    // =========================================================================
    // Constants
    // =========================================================================

    /// <summary>
    /// Whether the Grafeo native library is available. When false, all tests skip.
    /// </summary>
    private static readonly bool GrafeoAvailable = CheckGrafeoAvailable();

    /// <summary>Repository root, resolved from this file's location.</summary>
    private static readonly string RepoRoot = FindRepoRoot();

    /// <summary>Path to the spec directory containing .gtest files.</summary>
    private static readonly string SpecDir = Path.Combine(RepoRoot, "tests", "spec");

    /// <summary>Path to the datasets directory.</summary>
    private static readonly string DatasetsDir = Path.Combine(SpecDir, "datasets");

    /// <summary>Cached set of compiled feature flags from db.Info().</summary>
    private static readonly HashSet<string> CompiledFeatures = LoadCompiledFeatures();

    /// <summary>
    /// Capabilities provided by this runner itself (not compiled into the native
    /// library). For example, C# natively supports 64-bit integers, so
    /// "int64-safe" is always available.
    /// </summary>
    private static readonly HashSet<string> RunnerCapabilities = new() { "int64-safe" };

    private GrafeoDB? _db;

    public void Dispose()
    {
        _db?.Dispose();
        _db = null;
        GC.SuppressFinalize(this);
    }

    // =========================================================================
    // Test entry point
    // =========================================================================

    [SkippableTheory]
    [MemberData(nameof(DiscoverTestCases))]
    public void RunGtestCase(
        string displayName,
        string filePath,
        string testName,
        string? variantLang,
        string? variantQuery,
        string? skipReason)
    {
        if (!GrafeoAvailable)
        {
            Skip.If(true, "Grafeo native library not available");
            return;
        }

        if (skipReason is not null)
        {
            Skip.If(true, skipReason);
            return;
        }

        var gtestFile = GtestParser.ParseFile(filePath);
        var tc = gtestFile.Tests.FirstOrDefault(t => t.Name == testName);
        Assert.NotNull(tc);

        var meta = gtestFile.Meta;
        var language = variantLang ?? tc.Language ?? meta.Language;

        // Check language availability
        if (language != "gql" && language != "")
        {
            if (!HasFeature(language))
            {
                Skip.If(true, $"Feature '{language}' not available in this build");
                return;
            }
        }

        // Check requires
        foreach (var req in meta.Requires)
        {
            if (!HasFeature(req))
            {
                Skip.If(true, $"Required capability '{req}' not available");
                return;
            }
        }
        foreach (var req in tc.Requires)
        {
            if (!HasFeature(req))
            {
                Skip.If(true, $"Required capability '{req}' not available");
                return;
            }
        }

        // Create fresh database
        _db = GrafeoDB.Memory();

        try
        {
            // Load dataset (per-test override takes priority)
            var effectiveDataset = tc.Dataset ?? meta.Dataset;
            if (!string.IsNullOrEmpty(effectiveDataset) && effectiveDataset != "empty")
                LoadDataset(_db, effectiveDataset);

            // Run setup queries in the file's declared language
            var setupLanguage = string.IsNullOrEmpty(meta.Language) ? "gql" : meta.Language;
            foreach (var setupQuery in tc.Setup)
                ExecuteQuery(_db, setupLanguage, setupQuery);

            // Determine query/statements
            var query = variantQuery ?? tc.Query;
            var queries = tc.Statements.Count > 0
                ? tc.Statements
                : query is not null ? new List<string> { query } : [];

            Assert.True(queries.Count > 0, $"No query or statements in test '{tc.Name}'");

            // Coerce params (only applied to the last query)
            var parameters = CoerceParams(tc.Params);

            var expect = tc.Expect;

            // Error tests
            if (expect.Error is not null)
            {
                RunErrorTest(_db, language, queries, expect.Error, parameters);
                return;
            }

            // Execute all-but-last (with params so $param works in INSERT/SET)
            for (var i = 0; i < queries.Count - 1; i++)
                ExecuteQuery(_db, language, queries[i], parameters);

            // Last query: capture result (with params if present)
            var result = ExecuteQuery(_db, language, queries[^1], parameters);

            // Column assertion
            if (expect.Columns.Count > 0)
                AssertColumns(result, expect.Columns);

            // Value assertions
            if (expect.Empty)
            {
                AssertEmpty(result);
            }
            else if (expect.Count is not null)
            {
                AssertCount(result, expect.Count.Value);
            }
            else if (expect.Hash is not null)
            {
                AssertHash(result, expect.Hash);
            }
            else if (expect.Rows.Count > 0)
            {
                if (expect.Precision is not null)
                    AssertRowsWithPrecision(result, expect.Rows, expect.Precision.Value);
                else if (expect.Ordered)
                    AssertRowsOrdered(result, expect.Rows);
                else
                    AssertRowsSorted(result, expect.Rows);
            }
            // If none of the above, the test just checks the query does not error
        }
        finally
        {
            _db.Dispose();
            _db = null;
        }
    }

    // =========================================================================
    // Test case discovery
    // =========================================================================

    /// <summary>
    /// Discovers all .gtest files and yields test case data for xUnit MemberData.
    /// Each item is: (displayName, filePath, testName, variantLang, variantQuery, skipReason).
    /// </summary>
    public static IEnumerable<object?[]> DiscoverTestCases()
    {
        if (!Directory.Exists(SpecDir))
            yield break;

        foreach (var filePath in FindGtestFiles(SpecDir))
        {
            // Skip files inside the runners directory
            if (filePath.Replace('\\', '/').Contains("/runners/"))
                continue;

            GtestFile? gtestFile = null;
            string? parseError = null;
            try
            {
                gtestFile = GtestParser.ParseFile(filePath);
            }
            catch (Exception ex)
            {
                parseError = ex.Message;
            }

            if (parseError != null)
            {
                var relPath = Path.GetRelativePath(SpecDir, filePath).Replace('\\', '/');
                yield return [
                    $"{relPath}::PARSE_ERROR",
                    filePath,
                    "PARSE_ERROR",
                    null,
                    null,
                    $"Failed to parse: {parseError}",
                ];
                continue;
            }

            var relativePath = Path.GetRelativePath(SpecDir, filePath).Replace('\\', '/');

            foreach (var tc in gtestFile.Tests)
            {
                if (tc.Variants.Count > 0)
                {
                    // Rosetta: one test per variant language
                    foreach (var (lang, query) in tc.Variants)
                    {
                        var display = $"{relativePath}::{tc.Name}[{lang}]";
                        yield return [display, filePath, tc.Name, lang, query, tc.Skip];
                    }
                }
                else
                {
                    var display = $"{relativePath}::{tc.Name}";
                    yield return [display, filePath, tc.Name, null, null, tc.Skip];
                }
            }
        }
    }

    // =========================================================================
    // Query execution
    // =========================================================================

    /// <summary>Execute a query in the specified language, optionally with parameters.</summary>
    private static QueryResult ExecuteQuery(
        GrafeoDB db, string language, string query,
        Dictionary<string, object?>? parameters = null)
    {
        // When parameters are provided, use the universal ExecuteLanguage path
        // which supports all languages with params in a single call.
        if (parameters is not null)
        {
            var lang = language switch
            {
                "" => "gql",
                "sql-pgq" or "sql_pgq" => "sql",
                _ => language,
            };
            return db.ExecuteLanguage(lang, query, parameters);
        }

        return language switch
        {
            "gql" or "" => db.Execute(query),
            "cypher" => db.ExecuteCypher(query),
            "sparql" => db.ExecuteSparql(query),
            "gremlin" => db.ExecuteGremlin(query),
            "graphql" => db.ExecuteGraphql(query),
            "sql-pgq" or "sql_pgq" => db.ExecuteSql(query),
            "graphql-rdf" => ExecuteViaTransaction(db, "graphql-rdf", query),
            _ => throw new InvalidOperationException($"Unsupported language: {language}"),
        };
    }

    /// <summary>
    /// Execute a query via a transaction for languages that only have
    /// ExecuteLanguage support (e.g. graphql-rdf).
    /// </summary>
    private static QueryResult ExecuteViaTransaction(GrafeoDB db, string language, string query)
    {
        using var tx = db.BeginTransaction();
        var result = tx.ExecuteLanguage(language, query);
        tx.Commit();
        return result;
    }

    /// <summary>
    /// Coerce raw string parameter values to typed C# objects.
    /// Mirrors the Rust build.rs coercion order: a list or map written as JSON,
    /// int, float, bool, string.
    /// Returns null when the params dict is empty (so callers can skip it).
    /// </summary>
    private static Dictionary<string, object?>? CoerceParams(Dictionary<string, string> rawParams)
    {
        if (rawParams.Count == 0)
            return null;

        var coerced = new Dictionary<string, object?>(rawParams.Count);
        foreach (var (key, value) in rawParams)
        {
            var trimmed = value.Trim();
            if (trimmed.StartsWith('[') || trimmed.StartsWith('{'))
            {
                using var doc = JsonDocument.Parse(trimmed);
                coerced[key] = FromJson(doc.RootElement);
            }
            else if (long.TryParse(value, System.Globalization.NumberStyles.Integer,
                    System.Globalization.CultureInfo.InvariantCulture, out var l))
            {
                coerced[key] = l;
            }
            else if (double.TryParse(value, System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out var d))
            {
                coerced[key] = d;
            }
            else if (value == "true")
            {
                coerced[key] = true;
            }
            else if (value == "false")
            {
                coerced[key] = false;
            }
            else
            {
                coerced[key] = value;
            }
        }

        return coerced;
    }

    /// <summary>
    /// A JSON value as the objects the binding converts: a list, a map with
    /// string keys, a string, a long or double, a bool, or null.
    /// </summary>
    private static object? FromJson(JsonElement element) =>
        element.ValueKind switch
        {
            JsonValueKind.Array => element.EnumerateArray().Select(FromJson).ToList(),
            JsonValueKind.Object => element.EnumerateObject()
                .ToDictionary(property => property.Name, property => FromJson(property.Value)),
            JsonValueKind.String => element.GetString(),
            JsonValueKind.Number => element.TryGetInt64(out var l) ? l : element.GetDouble(),
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            _ => null,
        };

    /// <summary>Check if a feature (language or capability) is available.</summary>
    private static bool HasFeature(string requirement)
    {
        if (requirement is "gql" or "") return true;
        var key = requirement.Replace('_', '-');
        // Compound language key: "graphql-rdf" requires both "graphql" and "rdf"
        if (key == "graphql-rdf")
            return CompiledFeatures.Contains("graphql") && CompiledFeatures.Contains("rdf");
        return CompiledFeatures.Contains(key) || RunnerCapabilities.Contains(key);
    }

    // =========================================================================
    // Dataset loading
    // =========================================================================

    /// <summary>Load a .setup dataset file, executing each line as GQL.</summary>
    private static void LoadDataset(GrafeoDB db, string datasetName)
    {
        var setupPath = Path.Combine(DatasetsDir, $"{datasetName}.setup");
        Assert.True(File.Exists(setupPath), $"Dataset file not found: {setupPath}");

        var content = File.ReadAllText(setupPath);
        foreach (var line in content.Split('\n'))
        {
            var trimmed = line.Trim();
            if (string.IsNullOrEmpty(trimmed) || trimmed.StartsWith('#'))
                continue;
            db.Execute(trimmed);
        }
    }

    // =========================================================================
    // Error test
    // =========================================================================

    private static void RunErrorTest(
        GrafeoDB db, string language, List<string> queries, string expectedSubstring,
        Dictionary<string, object?>? parameters = null)
    {
        // Execute all-but-last normally (with params so $param works)
        for (var i = 0; i < queries.Count - 1; i++)
            ExecuteQuery(db, language, queries[i], parameters);

        // Last query should fail (with params if present)
        var ex = Assert.ThrowsAny<Exception>(() =>
            ExecuteQuery(db, language, queries[^1], parameters));
        Assert.Contains(expectedSubstring, ex.Message);
    }

    // =========================================================================
    // Value serialization
    // =========================================================================

    /// <summary>
    /// Convert a C# value (from QueryResult rows) to its canonical string
    /// representation for comparison with .gtest expected values.
    /// Matches the Rust value_to_string in grafeo-spec-tests/src/lib.rs.
    /// </summary>
    private static string ValueToCanonical(object? value)
    {
        if (value is null)
            return "null";
        if (value is bool b)
            return b ? "true" : "false";
        if (value is long l)
            return l.ToString();
        if (value is int i)
            return i.ToString();
        if (value is double d)
        {
            if (double.IsNaN(d)) return "NaN";
            if (double.IsPositiveInfinity(d)) return "Infinity";
            if (double.IsNegativeInfinity(d)) return "-Infinity";
            // Rust's Display for f64 drops ".0" for whole numbers
            if (d == Math.Floor(d) && Math.Abs(d) < (1L << 53))
                return ((long)d).ToString();
            return d.ToString(System.Globalization.CultureInfo.InvariantCulture);
        }
        if (value is float f)
        {
            if (float.IsNaN(f)) return "NaN";
            if (float.IsPositiveInfinity(f)) return "Infinity";
            if (float.IsNegativeInfinity(f)) return "-Infinity";
            double fd = f;
            if (fd == Math.Floor(fd) && Math.Abs(fd) < (1L << 53))
                return ((long)fd).ToString();
            return fd.ToString(System.Globalization.CultureInfo.InvariantCulture);
        }
        if (value is JsonElement je)
            return JsonElementToCanonical(je);
        if (value is List<object?> list)
        {
            var inner = string.Join(", ", list.Select(ValueToCanonical));
            return $"[{inner}]";
        }
        if (value is Dictionary<string, object?> dict)
        {
            // Temporal type-tagged objects from C FFI: {"$date": "2024-06-15"}
            if (dict.Count == 1)
            {
                var kvp = dict.First();
                switch (kvp.Key)
                {
                    case "$date":
                    case "$time":
                    case "$datetime":
                    case "$zoned_datetime":
                    case "$duration":
                        return kvp.Value?.ToString() ?? "null";
                    case "$timestamp_us":
                        return Convert.ToInt64(kvp.Value ?? 0).ToString();
                }
            }
            if (IsDurationDict(dict))
                return DurationToIso(dict);
            var entries = dict
                .Select(kv => $"{kv.Key}: {ValueToCanonical(kv.Value)}")
                .OrderBy(e => e)
                .ToList();
            return "{" + string.Join(", ", entries) + "}";
        }
        if (value is IReadOnlyDictionary<string, object?> roDict)
        {
            // Same type-tag check for readonly dictionaries
            if (roDict.Count == 1)
            {
                var kvp = roDict.First();
                switch (kvp.Key)
                {
                    case "$date":
                    case "$time":
                    case "$datetime":
                    case "$zoned_datetime":
                    case "$duration":
                        return kvp.Value?.ToString() ?? "null";
                    case "$timestamp_us":
                        return Convert.ToInt64(kvp.Value ?? 0).ToString();
                }
            }
            if (roDict.Count == 3 && roDict.ContainsKey("months") && roDict.ContainsKey("days") && roDict.ContainsKey("nanos"))
                return DurationToIso(roDict.ToDictionary(kv => kv.Key, kv => kv.Value));
            var entries = roDict
                .Select(kv => $"{kv.Key}: {ValueToCanonical(kv.Value)}")
                .OrderBy(e => e)
                .ToList();
            return "{" + string.Join(", ", entries) + "}";
        }
        if (value is byte[] bytes)
            return $"bytes[{bytes.Length}]";
        // DateTime/DateTimeOffset: force ISO 8601 format (system locale may differ)
        if (value is DateTime dt)
            return dt.ToString("yyyy-MM-dd");
        if (value is DateTimeOffset dto)
            return dto.ToString("yyyy-MM-ddTHH:mm:ssK");
        return value.ToString() ?? "null";
    }

    private static bool IsDurationDict(Dictionary<string, object?> dict)
        => dict.Count == 3 && dict.ContainsKey("months") && dict.ContainsKey("days") && dict.ContainsKey("nanos");

    private static string DurationToIso(Dictionary<string, object?> dict)
    {
        long totalMonths = Convert.ToInt64(dict["months"] ?? 0);
        long days = Convert.ToInt64(dict["days"] ?? 0);
        long nanos = Convert.ToInt64(dict["nanos"] ?? 0);
        long years = totalMonths / 12;
        long months = totalMonths % 12;
        long hours = nanos / 3_600_000_000_000;
        long rem = nanos % 3_600_000_000_000;
        long minutes = rem / 60_000_000_000;
        rem %= 60_000_000_000;
        long seconds = rem / 1_000_000_000;
        long subNanos = rem % 1_000_000_000;

        var sb = new System.Text.StringBuilder("P");
        if (years != 0) sb.Append($"{years}Y");
        if (months != 0) sb.Append($"{months}M");
        if (days != 0) sb.Append($"{days}D");
        var tp = new System.Text.StringBuilder();
        if (hours != 0) tp.Append($"{hours}H");
        if (minutes != 0) tp.Append($"{minutes}M");
        if (seconds != 0 || subNanos != 0)
        {
            if (subNanos != 0)
                tp.Append($"{seconds}.{subNanos:D9}".TrimEnd('0') + "S");
            else
                tp.Append($"{seconds}S");
        }
        if (tp.Length > 0) { sb.Append('T'); sb.Append(tp); }
        var result = sb.ToString();
        return result == "P" ? "P0D" : result;
    }

    /// <summary>Handle JsonElement values that may appear in result rows.</summary>
    private static string JsonElementToCanonical(JsonElement element)
    {
        return element.ValueKind switch
        {
            JsonValueKind.Null or JsonValueKind.Undefined => "null",
            JsonValueKind.True => "true",
            JsonValueKind.False => "false",
            JsonValueKind.String => element.GetString() ?? "null",
            JsonValueKind.Number => FormatJsonNumber(element),
            JsonValueKind.Array => FormatJsonArray(element),
            JsonValueKind.Object => FormatJsonObject(element),
            _ => element.GetRawText(),
        };
    }

    private static string FormatJsonNumber(JsonElement element)
    {
        if (element.TryGetInt64(out var l))
            return l.ToString();
        var d = element.GetDouble();
        if (double.IsNaN(d)) return "NaN";
        if (double.IsPositiveInfinity(d)) return "Infinity";
        if (double.IsNegativeInfinity(d)) return "-Infinity";
        if (d == Math.Floor(d) && Math.Abs(d) < (1L << 53))
            return ((long)d).ToString();
        return d.ToString(System.Globalization.CultureInfo.InvariantCulture);
    }

    private static string FormatJsonArray(JsonElement element)
    {
        var items = element.EnumerateArray()
            .Select(JsonElementToCanonical)
            .ToList();
        return "[" + string.Join(", ", items) + "]";
    }

    private static string FormatJsonObject(JsonElement element)
    {
        // Temporal type-tagged objects from C FFI: {"$date": "2024-06-15"}
        var props = element.EnumerateObject().ToList();
        if (props.Count == 1)
        {
            var prop = props[0];
            switch (prop.Name)
            {
                case "$date":
                case "$time":
                case "$datetime":
                case "$zoned_datetime":
                case "$duration":
                    return prop.Value.GetString() ?? "null";
                case "$timestamp_us":
                    return prop.Value.GetInt64().ToString();
            }
        }

        var entries = element.EnumerateObject()
            .Select(p => $"{p.Name}: {JsonElementToCanonical(p.Value)}")
            .OrderBy(e => e)
            .ToList();
        return "{" + string.Join(", ", entries) + "}";
    }

    // =========================================================================
    // Result to rows
    // =========================================================================

    /// <summary>
    /// Convert a QueryResult into rows of canonical strings using the result's
    /// column order, or an explicit column list if provided.
    /// </summary>
    private static List<List<string>> ResultToRows(
        QueryResult result, IReadOnlyList<string>? columns = null)
    {
        var cols = columns ?? result.Columns;
        var rows = new List<List<string>>();

        foreach (var rowDict in result.Rows)
        {
            var row = new List<string>();
            foreach (var col in cols)
            {
                rowDict.TryGetValue(col, out var val);
                row.Add(ValueToCanonical(val));
            }
            rows.Add(row);
        }

        return rows;
    }

    // =========================================================================
    // Assertions
    // =========================================================================

    private static void AssertRowsSorted(QueryResult result, List<List<string>> expected)
    {
        var actual = ResultToRows(result);
        var actualSorted = actual.OrderBy(r => string.Join("|", r)).ToList();
        var expectedSorted = expected.OrderBy(r => string.Join("|", r)).ToList();

        Assert.Equal(expectedSorted.Count, actualSorted.Count);

        for (var i = 0; i < actualSorted.Count; i++)
        {
            var actRow = actualSorted[i];
            var expRow = expectedSorted[i];
            Assert.Equal(expRow.Count, actRow.Count);

            for (var j = 0; j < actRow.Count; j++)
            {
                Assert.Equal(expRow[j], actRow[j]);
            }
        }
    }

    private static void AssertRowsOrdered(QueryResult result, List<List<string>> expected)
    {
        var actual = ResultToRows(result);

        Assert.Equal(expected.Count, actual.Count);

        for (var i = 0; i < actual.Count; i++)
        {
            var actRow = actual[i];
            var expRow = expected[i];
            Assert.Equal(expRow.Count, actRow.Count);

            for (var j = 0; j < actRow.Count; j++)
            {
                Assert.Equal(expRow[j], actRow[j]);
            }
        }
    }

    private static void AssertRowsWithPrecision(
        QueryResult result, List<List<string>> expected, int precision)
    {
        var actual = ResultToRows(result);
        var tolerance = Math.Pow(10, -precision);

        Assert.Equal(expected.Count, actual.Count);

        for (var i = 0; i < actual.Count; i++)
        {
            var actRow = actual[i];
            var expRow = expected[i];
            Assert.Equal(expRow.Count, actRow.Count);

            for (var j = 0; j < actRow.Count; j++)
            {
                if (double.TryParse(actRow[j], System.Globalization.NumberStyles.Float,
                        System.Globalization.CultureInfo.InvariantCulture, out var af) &&
                    double.TryParse(expRow[j], System.Globalization.NumberStyles.Float,
                        System.Globalization.CultureInfo.InvariantCulture, out var ef))
                {
                    Assert.True(Math.Abs(af - ef) < tolerance,
                        $"Float mismatch at row {i}, col {j}: got {af}, expected {ef} (tolerance {tolerance})");
                }
                else
                {
                    Assert.Equal(expRow[j], actRow[j]);
                }
            }
        }
    }

    private static void AssertCount(QueryResult result, int expectedCount)
    {
        Assert.Equal(expectedCount, result.Rows.Count);
    }

    private static void AssertEmpty(QueryResult result)
    {
        Assert.Empty(result.Rows);
    }

    private static void AssertColumns(QueryResult result, List<string> expectedColumns)
    {
        var actual = result.Columns.ToList();
        Assert.Equal(expectedColumns, actual);
    }

    private static void AssertHash(QueryResult result, string expectedHash)
    {
        var rows = ResultToRows(result);
        rows.Sort((a, b) => string.Compare(
            string.Join("|", a), string.Join("|", b), StringComparison.Ordinal));

        using var md5 = MD5.Create();
        foreach (var row in rows)
        {
            var line = string.Join("|", row) + "\n";
            var bytes = Encoding.UTF8.GetBytes(line);
            md5.TransformBlock(bytes, 0, bytes.Length, null, 0);
        }
        md5.TransformFinalBlock([], 0, 0);

        var actualHash = Convert.ToHexString(md5.Hash!).ToLowerInvariant();
        Assert.Equal(expectedHash, actualHash);
    }

    // =========================================================================
    // File discovery
    // =========================================================================

    private static IEnumerable<string> FindGtestFiles(string dir)
    {
        if (!Directory.Exists(dir))
            return [];

        return Directory.EnumerateFiles(dir, "*.gtest", SearchOption.AllDirectories)
            .OrderBy(p => p);
    }

    // =========================================================================
    // Setup helpers
    // =========================================================================

    private static bool CheckGrafeoAvailable()
    {
        try
        {
            using var db = GrafeoDB.Memory();
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static HashSet<string> LoadCompiledFeatures()
    {
        try
        {
            using var db = GrafeoDB.Memory();
            var info = db.Info();
            if (info.TryGetValue("features", out var featuresObj)
                && featuresObj is IEnumerable<object?> features)
            {
                return features
                    .Where(f => f is not null)
                    .Select(f => f!.ToString()!)
                    .ToHashSet();
            }
        }
        catch (DllNotFoundException)
        {
            // Native library not available
        }
        catch (BadImageFormatException)
        {
            // Native library not available
        }
        catch (TypeInitializationException ex)
        {
            if (ex.InnerException is DllNotFoundException or BadImageFormatException)
            {
                // Native library not available
            }
            else
            {
                throw;
            }
        }
        return [];
    }

    private static string FindRepoRoot()
    {
        // This file is at tests/spec/runners/csharp/SpecTests.cs
        // Repo root is 4 directories up.
        var dir = AppContext.BaseDirectory;

        // Walk up from the build output directory to find the repo root.
        // Look for Cargo.toml as the marker.
        var candidate = new DirectoryInfo(dir);
        while (candidate is not null)
        {
            if (File.Exists(Path.Combine(candidate.FullName, "Cargo.toml")))
                return candidate.FullName;
            candidate = candidate.Parent;
        }

        // Fallback: relative from source file location
        return Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..", "..", ".."));
    }
}
