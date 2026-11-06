using Xunit;

namespace Grafeo.Tests;

/// <summary>
/// The backup cursor can be released: <c>ReleaseBackupCursor()</c> returns whether one was released,
/// and the next incremental backup fails with GRAFEO-S005 until a full backup starts a new chain.
/// </summary>
public sealed class BackupRetentionTests
{
    private static void WriteBatch(GrafeoDB db, int start) =>
        db.Execute($"UNWIND range({start}, {start + 99}) AS i INSERT (:Person {{i: i, note: '{new string('x', 300)}'}})");

    [Fact]
    public void TheCursorIsReleasedOnceAndAsksForAFullBackup()
    {
        var dir = Path.Join(Path.GetTempPath(), $"grafeo-csharp-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            var backups = Path.Join(dir, "backups");
            using var db = GrafeoDB.Open(Path.Join(dir, "lisbon.grafeo"));
            WriteBatch(db, 0);
            db.BackupFull(backups);
            WriteBatch(db, 100);
            Assert.True(db.ReleaseBackupCursor());
            Assert.False(db.ReleaseBackupCursor());
            var error = Assert.ThrowsAny<GrafeoException>(() => db.BackupIncremental(backups));
            Assert.Contains("GRAFEO-S005", error.Message);
            Assert.Contains("cursor", error.Message);
            db.BackupFull(backups);
            WriteBatch(db, 200);
            db.BackupIncremental(backups);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
