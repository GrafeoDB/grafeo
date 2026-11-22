using Xunit;

namespace Grafeo.Tests;

/// <summary>
/// The memory settings through C# (#751): a <see cref="GrafeoOptions"/> object with the memory
/// limit, the query memory limit and the spill limit (zero means nothing spills, unlimited is its
/// own setter), opened with <c>GrafeoDB.Open(path or null, options)</c>. Null options are the
/// defaults, a zero memory limit fails at the open, the settings read back through
/// <c>MemoryUsage()</c>, and a file opened over its limit opens, reads and warns.
/// </summary>
public sealed class MemorySettingsTests
{
    private const long MiB = 1024 * 1024;

    private static string TempDbPath(string name)
    {
        var dir = Path.Join(Path.GetTempPath(), $"grafeo-csharp-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        return Path.Join(dir, name);
    }

    private static void Cleanup(string dbPath)
    {
        var dir = Path.GetDirectoryName(dbPath)!;
        try { Directory.Delete(dir, recursive: true); }
        catch (IOException)
        {
            // Best-effort cleanup in tests: ignore IO failures when deleting the temp directory.
        }
        catch (UnauthorizedAccessException)
        {
            // Best-effort cleanup in tests: ignore permission issues when deleting the temp directory.
        }
    }

    private static long? Bytes(object? value) => value is null ? null : Convert.ToInt64(value);

    private static IReadOnlyDictionary<string, object?> Spill(IReadOnlyDictionary<string, object?> usage) =>
        (IReadOnlyDictionary<string, object?>)usage["spill"]!;

    [Fact]
    public void EachSettingReadsBackThroughTheReport()
    {
        using var options = new GrafeoOptions();
        options.SetMemoryLimit(64 * MiB);
        options.SetQueryMemoryLimit(8 * MiB);
        options.SetSpillLimit(0);
        using var db = GrafeoDB.Open(null, options);
        var usage = db.MemoryUsage();
        Assert.Equal(64 * MiB, Bytes(usage["limit_bytes"]));
        Assert.Equal(8 * MiB, Bytes(usage["query_limit_bytes"]));
        Assert.Equal(0L, Bytes(Spill(usage)["limit_bytes"]));
    }

    /// <summary>
    /// A database file has a spill directory, so unlimited spill reads back as no limit.
    /// </summary>
    [Fact]
    public void UnlimitedSpillReadsBackAsNoLimit()
    {
        var path = TempDbPath("unlimited.grafeo");
        try
        {
            using var options = new GrafeoOptions();
            options.SetSpillLimit(16 * MiB);
            options.SetUnlimitedSpill(true);
            using var db = GrafeoDB.Open(path, options);
            Assert.Null(Bytes(Spill(db.MemoryUsage())["limit_bytes"]));
        }
        finally
        {
            Cleanup(path);
        }
    }

    /// <summary>
    /// An in-memory database has no spill directory: nothing spills, whatever the settings say.
    /// </summary>
    [Fact]
    public void AnInMemoryOpenSpillsNothingWhateverTheSettings()
    {
        using var options = new GrafeoOptions();
        options.SetUnlimitedSpill(true);
        using var db = GrafeoDB.Open(null, options);
        Assert.Equal(0L, Bytes(Spill(db.MemoryUsage())["limit_bytes"]));
    }

    [Fact]
    public void NullOptionsAreTheDefaults()
    {
        using var defaults = GrafeoDB.Memory();
        using var db = GrafeoDB.Open(null, null);
        Assert.Equal(Bytes(defaults.MemoryUsage()["limit_bytes"]), Bytes(db.MemoryUsage()["limit_bytes"]));
        Assert.Equal(
            Bytes(defaults.MemoryUsage()["query_limit_bytes"]),
            Bytes(db.MemoryUsage()["query_limit_bytes"]));
    }

    [Fact]
    public void AZeroMemoryLimitFailsAtTheOpen()
    {
        using var options = new GrafeoOptions();
        options.SetMemoryLimit(0);
        Assert.Throws<GrafeoException>(() => GrafeoDB.Open(null, options));
    }

    [Fact]
    public void OneOptionsObjectOpensTwoDatabasesAlike()
    {
        using var options = new GrafeoOptions();
        options.SetMemoryLimit(48 * MiB);
        using var first = GrafeoDB.Open(null, options);
        using var second = GrafeoDB.Open(null, options);
        Assert.Equal(48 * MiB, Bytes(first.MemoryUsage()["limit_bytes"]));
        Assert.Equal(48 * MiB, Bytes(second.MemoryUsage()["limit_bytes"]));
    }

    /// <summary>
    /// A file whose data passes the limit it is opened with opens, reads, and says so in its
    /// open warnings; opened with room, it says nothing of the kind.
    /// </summary>
    [Fact]
    public void AnOpenOverTheLimitWarnsAndReads()
    {
        var path = TempDbPath("over.grafeo");
        try
        {
            using (var db = GrafeoDB.Open(path))
            {
                // Distinct 1 KiB strings: data that stays past the limit once reloaded.
                db.Execute("UNWIND range(1, 20000) AS i INSERT (:Fill {i: i, text: CAST(i AS STRING) || '" + new string('x', 1000) + "'})");
            }

            using (var options = new GrafeoOptions())
            {
                options.SetMemoryLimit(MiB);
                using var db = GrafeoDB.Open(path, options);
                Assert.Contains(db.OpenWarnings(), w => (string?)w["code"] == "MEMORY_LIMIT_PASSED");
                Assert.Equal(20000L, db.NodeCount);
            }

            using (var db = GrafeoDB.Open(path))
            {
                Assert.DoesNotContain(db.OpenWarnings(), w => (string?)w["code"] == "MEMORY_LIMIT_PASSED");
            }
        }
        finally
        {
            Cleanup(path);
        }
    }

    /// <summary>
    /// The plan cache's limit (#674) reads back as <c>caches.plan_cache_limit_bytes</c>, capped at
    /// the memory limit; zero turns the cache off.
    /// </summary>
    [Theory]
    [InlineData(64 * MiB, 8 * MiB, 8 * MiB)]
    [InlineData(32 * MiB, 1024 * MiB, 32 * MiB)]
    [InlineData(64 * MiB, 0L, 0L)]
    public void ThePlanCacheLimitReadsBack(long memory, long limit, long expected)
    {
        using var options = new GrafeoOptions();
        options.SetMemoryLimit(memory);
        options.SetPlanCacheLimit(limit);
        using var db = GrafeoDB.Open(null, options);
        var caches = (IReadOnlyDictionary<string, object?>)db.MemoryUsage()["caches"]!;
        Assert.Equal(expected, Bytes(caches["plan_cache_limit_bytes"]));
    }

    /// <summary>A negative plan cache limit is refused.</summary>
    [Fact]
    public void ANegativePlanCacheLimitIsRefused()
    {
        using var options = new GrafeoOptions();
        Assert.Throws<ArgumentOutOfRangeException>(() => options.SetPlanCacheLimit(-1));
    }

    /// <summary>
    /// The change log's memory limit (#679) reads back as <c>cdc.limit_bytes</c>: a sixteenth of
    /// the memory limit when unset, capped at the memory limit; zero fails at the open.
    /// </summary>
    [Theory]
    [InlineData(64 * MiB, 512 * 1024L, 512 * 1024L)]
    [InlineData(64 * MiB, -1L, 4 * MiB)]
    [InlineData(32 * MiB, 1024 * MiB, 32 * MiB)]
    public void TheCdcMemoryLimitReadsBack(long memory, long limit, long expected)
    {
        using var options = new GrafeoOptions();
        options.SetMemoryLimit(memory);
        if (limit >= 0)
        {
            options.SetCdcMemoryLimit(limit);
        }
        using var db = GrafeoDB.Open(null, options);
        var cdc = (IReadOnlyDictionary<string, object?>)db.MemoryUsage()["cdc"]!;
        Assert.Equal(expected, Bytes(cdc["limit_bytes"]));
    }

    [Fact]
    public void AZeroCdcMemoryLimitFailsAtTheOpen()
    {
        using var options = new GrafeoOptions();
        options.SetCdcMemoryLimit(0);
        Assert.ThrowsAny<GrafeoException>(() => GrafeoDB.Open(null, options));
    }
}
