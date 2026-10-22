using Xunit;

namespace Grafeo.Tests;

/// <summary>
/// Full and incremental backups and a restore to an epoch: a full backup is an image of the database
/// at an epoch, written while commits go on; the incremental backups after it hold the WAL after it.
/// </summary>
public sealed class BackupTests
{
    private static List<object?> People(GrafeoDB db) =>
        db.Execute("MATCH (p:Person) RETURN p.name AS name ORDER BY name").Rows.Select(row => row["name"]).ToList();

    [Fact]
    public void FullAndIncrementalBackupsRestoreToEachEpoch()
    {
        var dir = Path.Join(Path.GetTempPath(), $"grafeo-csharp-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            var backups = Path.Join(dir, "backups");
            ulong fullEpoch, latest;
            using (var db = GrafeoDB.Open(Path.Join(dir, "paris.grafeo")))
            {
                db.Execute("INSERT (:Person {name: 'Alix'})");
                db.BackupFull(backups);
                fullEpoch = db.CurrentEpoch;
                db.Execute("INSERT (:Person {name: 'Gus'})");
                db.BackupIncremental(backups);
                latest = db.CurrentEpoch;
            }

            var atFull = Path.Join(dir, "full.grafeo");
            GrafeoDB.RestoreToEpoch(backups, fullEpoch, atFull);
            using (var restored = GrafeoDB.Open(atFull))
                Assert.Equal(["Alix"], People(restored));

            var atLatest = Path.Join(dir, "latest.grafeo");
            GrafeoDB.RestoreToEpoch(backups, latest, atLatest);
            using (var restored = GrafeoDB.Open(atLatest))
                Assert.Equal(["Alix", "Gus"], People(restored));
        }
        finally
        {
            Cleanup(dir);
        }
    }

    private const int Writers = 3;

    /// <summary>The (writer, seq) of every node with <paramref name="label"/>, sorted.</summary>
    private static List<(long Writer, long Seq)> Committed(GrafeoDB db, string label) =>
        db.Execute($"MATCH (n:{label}) RETURN n.writer AS w, n.seq AS s ORDER BY w, s")
            .Rows.Select(row => ((long)row["w"]!, (long)row["s"]!))
            .ToList();

    /// <summary>
    /// Writer threads commit a Person and a City per transaction, and roll one back each time,
    /// while a full backup is taken: its image holds every transaction whole, each writer's
    /// commits from the first on, at least the ones before the backup started, and nothing rolled
    /// back. The incremental backup after them restores every commit, also after a reopen.
    /// </summary>
    [Fact]
    public void FullBackupDuringConcurrentWritersHoldsWholeTransactions()
    {
        var dir = Path.Join(Path.GetTempPath(), $"grafeo-csharp-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            var backups = Path.Join(dir, "backups");
            var counts = new int[Writers];
            using var stop = new ManualResetEventSlim();
            using var started = new CountdownEvent(Writers);
            ulong latest;
            int[] before;
            using (var db = GrafeoDB.Open(Path.Join(dir, "lyon.grafeo")))
            {
                var threads = Enumerable.Range(0, Writers).Select(writer => new Thread(() =>
                {
                    for (var seq = 0; seq < 20 || !stop.IsSet; seq++)
                    {
                        using (var tx = db.BeginTransaction())
                        {
                            tx.Execute($"INSERT (:Person {{writer: {writer}, seq: {seq}}})");
                            tx.Execute($"INSERT (:City {{writer: {writer}, seq: {seq}}})");
                            tx.Commit();
                        }
                        using (var tx = db.BeginTransaction())
                        {
                            tx.Execute($"INSERT (:Person {{writer: {writer}, seq: -1}})");
                            tx.Rollback();
                        }
                        Volatile.Write(ref counts[writer], seq + 1);
                        if (seq == 4)
                            started.Signal();
                    }
                })).ToList();
                threads.ForEach(thread => thread.Start());
                Assert.True(started.Wait(TimeSpan.FromMinutes(1)));
                before = Enumerable.Range(0, Writers).Select(writer => Volatile.Read(ref counts[writer])).ToArray();
                db.BackupFull(backups);
                stop.Set();
                threads.ForEach(thread => thread.Join());

                var image = Path.Join(dir, "image.grafeo");
                File.Copy(Path.Join(backups, "backup_full_0000.grafeo"), image);
                using (var copy = GrafeoDB.Open(image))
                {
                    var people = Committed(copy, "Person");
                    Assert.Equal(people, Committed(copy, "City"));
                    for (var writer = 0; writer < Writers; writer++)
                    {
                        var seqs = people.Where(p => p.Writer == writer).Select(p => p.Seq).ToList();
                        Assert.Equal(Enumerable.Range(0, seqs.Count).Select(seq => (long)seq), seqs);
                        Assert.InRange(seqs.Count, before[writer], counts[writer]);
                    }
                }

                db.BackupIncremental(backups);
                latest = db.CurrentEpoch;
            }

            var expected = Enumerable.Range(0, Writers)
                .SelectMany(writer => Enumerable.Range(0, counts[writer]).Select(seq => ((long)writer, (long)seq)))
                .ToList();
            var restored = Path.Join(dir, "restored.grafeo");
            GrafeoDB.RestoreToEpoch(backups, latest, restored);
            for (var reopen = 0; reopen < 2; reopen++)
            {
                using var db = GrafeoDB.Open(restored);
                Assert.Equal(expected, Committed(db, "Person"));
                Assert.Equal(expected, Committed(db, "City"));
            }
        }
        finally
        {
            Cleanup(dir);
        }
    }

    private static void Cleanup(string dir)
    {
        try { Directory.Delete(dir, recursive: true); }
        catch (IOException)
        {
            // Best-effort cleanup in tests.
        }
    }
}
