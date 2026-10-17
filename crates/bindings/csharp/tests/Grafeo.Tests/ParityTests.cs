using Xunit;

namespace Grafeo.Tests;

/// <summary>Batch writes, write counters, change history, upserts and graph handles.</summary>
public sealed class ParityTests : IDisposable
{
    private readonly GrafeoDB _db = GrafeoDB.Memory();

    public void Dispose() => _db.Dispose();

    private static Dictionary<string, object?> Row(params (string Key, object? Value)[] fields) =>
        fields.ToDictionary(f => f.Key, f => f.Value);

    [Fact]
    public void BatchCreatesNodesAndEdges()
    {
        var ids = _db.BatchCreateNodesWithProps(["Person", "Employee"], [Row(("name", "Alix")), Row(("name", "Gus"))]);
        Assert.Equal(2, ids.Count);
        Assert.Equal("Gus", _db.GetNode(ids[1])!.Properties["name"]);
        Assert.Equal(new[] { "Employee", "Person" }, _db.GetNode(ids[0])!.Labels.Order());

        var edgeIds = _db.BatchCreateEdges([
            new BatchEdge(ids[0], ids[1], "KNOWS", Row(("since", 2020))),
            new BatchEdge(ids[1], ids[0], "KNOWS"),
        ]);
        Assert.Equal(2, edgeIds.Count);
        var edge = _db.GetEdge(edgeIds[0])!;
        Assert.Equal((ids[0], ids[1], 2020L), (edge.SourceId, edge.TargetId, edge.Properties["since"]));

        Assert.ThrowsAny<GrafeoException>(() => _db.BatchCreateEdges([new BatchEdge(ids[0], 88888, "KNOWS")]));
        Assert.Equal(2, _db.EdgeCount);
    }

    [Fact]
    public void EmptyBatchCreatesNothing()
    {
        Assert.Empty(_db.BatchCreateNodesWithProps(["Person"], []));
        Assert.Equal(0, _db.NodeCount);
    }

    [Fact]
    public void ResultsCountWrites()
    {
        var write = _db.Execute("INSERT (:Person {name: 'Vincent'})-[:KNOWS]->(:Person {name: 'Mia'})");
        Assert.Equal(new WriteCounters(2, 0, 1, 0, 2, 2, 0), write.Counters);

        var read = _db.Execute("MATCH (p:Person) RETURN p.name");
        Assert.Equal(WriteCounters.None, read.Counters);

        using var tx = _db.BeginTransaction();
        var deleted = tx.Execute("MATCH (p:Person {name: 'Vincent'}) DETACH DELETE p");
        Assert.Equal((1L, 1L), (deleted.Counters.NodesDeleted, deleted.Counters.EdgesDeleted));
    }

    [Fact]
    public void BatchVectorSearchAnswersEachQuery()
    {
        _db.Execute("INSERT (:Doc {topic: 'a', e: vector([1.0, 0.0])}), (:Doc {topic: 'b', e: vector([0.0, 1.0])})");
        _db.Execute("CREATE VECTOR INDEX doc_e ON :Doc(e) DIMENSION 2");

        var results = _db.BatchVectorSearch("Doc", "e", [[1f, 0f], [0f, 1f]], 1);
        Assert.Equal(2, results.Count);
        Assert.Equal("a", _db.GetNode(results[0].Single().NodeId)!.Properties["topic"]);
        Assert.Equal("b", _db.GetNode(results[1].Single().NodeId)!.Properties["topic"]);

        var filtered = _db.BatchVectorSearch("Doc", "e", [[1f, 0f]], 2, filters: Row(("topic", "b")));
        Assert.Equal("b", _db.GetNode(filtered[0].Single().NodeId)!.Properties["topic"]);

        Assert.Throws<ArgumentException>(() => _db.BatchVectorSearch("Doc", "e", [[1f, 0f], [1f]], 1));
    }

    [Fact]
    public void ChangeHistoryRecordsDetachDelete()
    {
        _db.CdcEnabled = true;
        _db.Execute("INSERT (:Person {name: 'Butch'})-[:KNOWS]->(:Person {name: 'Hans'})");
        _db.Execute("MATCH (p:Person {name: 'Butch'}) DETACH DELETE p");

        var node = _db.NodeHistory(0);
        Assert.Equal(new[] { "create", "delete" }, node.Select(e => e.Kind));
        Assert.Equal("node", node[0].EntityType);
        Assert.Equal("Butch", node[0].After!["name"]);
        Assert.Null(node[0].Before);
        Assert.Null(node[0].Graph);

        var edgeDelete = _db.EdgeHistory(0)[^1];
        Assert.Equal(("delete", "KNOWS", 0L, 0L, 1L), (
            edgeDelete.Kind, edgeDelete.EdgeType, edgeDelete.DeletedWith, edgeDelete.SourceId, edgeDelete.TargetId));

        var since = _db.NodeHistorySince(0, node[1].Epoch);
        Assert.Equal("delete", Assert.Single(since).Kind);

        var all = _db.ChangesBetween(0, ulong.MaxValue);
        Assert.Equal(5, all.Count);
        Assert.True(all.Zip(all.Skip(1)).All(p => p.First.Epoch <= p.Second.Epoch));
    }

    [Fact]
    public void UpsertsCreateUpdateAndSkip()
    {
        var created = _db.UpsertNodes(["File"], [Row(("path", "a.rs"), ("size", 3)), Row(("path", "b.rs")), Row(("size", 1))],
            new UpsertNodesOptions { Key = "path" });
        Assert.Equal((2, 0, 1), (created.Created, created.Updated, created.Skipped));
        Assert.Equal(new[] { 2 }, created.SkippedRows);

        var updated = _db.UpsertNodes(["File"], [Row(("path", "a.rs"), ("size", 19))], new UpsertNodesOptions { Key = "path" });
        Assert.Equal(1, updated.Updated);
        Assert.Equal(2, _db.NodeCount);

        var edges = _db.UpsertEdges("IMPORTS", [Row(("id", "a-b"), ("from", "a.rs"), ("to", "b.rs")), Row(("id", "a-x"), ("from", "a.rs"), ("to", "x.rs"))],
            new UpsertEdgesOptions { EndpointKey = "path", EndpointLabels = ["File"], SourceField = "from", TargetField = "to" });
        Assert.Equal((1, 1), (edges.Created, edges.Skipped));
        Assert.Equal(1, _db.EdgeCount);
    }

    [Fact]
    public void GraphHandleWorksInItsGraph()
    {
        _db.Execute("CREATE GRAPH trips");
        Assert.ThrowsAny<GrafeoException>(() => _db.Graph("nowhere"));

        using var trips = _db.Graph("trips");
        var prague = trips.CreateNode(["City"], Row(("name", "Prague")));
        var berlin = trips.CreateNode(["City"], Row(("name", "Berlin")));
        var route = trips.CreateEdge(prague, berlin, "ROUTE", Row(("km", 350)));
        trips.SetNodeProperty(prague, "country", "CZ");
        Assert.Equal("CZ", trips.GetNode(prague).Properties["country"]);
        Assert.Equal(350L, trips.GetEdge(route).Properties["km"]);
        Assert.True(trips.AddNodeLabel(prague, "Capital"));
        Assert.False(trips.AddNodeLabel(prague, "Capital"));

        var result = trips.ExecuteWithParams(
            "MATCH (c:City {name: $name})-[:ROUTE]->(d) RETURN d.name AS d", Row(("name", "Prague")));
        Assert.Equal("Berlin", Assert.Single(result.Rows)["d"]);
        Assert.Equal(0, _db.NodeCount);

        var upsert = trips.UpsertNodes(["City"], [Row(("name", "Prague"), ("country", "CZ"))], new UpsertNodesOptions { Key = "name" });
        Assert.Equal(1, upsert.Updated);
        Assert.Equal(2, trips.BatchCreateNodesWithProps(["City"], [Row(("name", "Paris")), Row(("name", "Amsterdam"))]).Count);

        Assert.ThrowsAny<GrafeoException>(() => trips.DeleteNode(prague));
        Assert.True(trips.DetachDeleteNode(prague));
        Assert.False(trips.DetachDeleteNode(prague));

        _db.Execute("DROP GRAPH trips");
        Assert.ThrowsAny<GrafeoException>(() => trips.Execute("RETURN 1"));
        Assert.ThrowsAny<GrafeoException>(() => trips.CreateNode(["City"]));
    }
}
