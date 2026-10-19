using Xunit;

namespace Grafeo.Tests;

/// <summary>GetNodeEdges: a node's edges and neighbors without a query.</summary>
public sealed class NodeEdgesTests : IDisposable
{
    private readonly GrafeoDB _db = GrafeoDB.Memory();

    public void Dispose() => _db.Dispose();

    private static NodeEdge[] Sorted(IEnumerable<NodeEdge> edges) => edges.OrderBy(e => e.EdgeId).ToArray();

    [Fact]
    public void ListsEdgesByDirectionAndType()
    {
        var alix = _db.CreateNode(["Person"]);
        var gus = _db.CreateNode(["Person"]);
        var vincent = _db.CreateNode(["Person"]);
        var knowsGus = _db.CreateEdge(alix, gus, "KNOWS");
        var knowsVincent = _db.CreateEdge(alix, vincent, "KNOWS");
        var worksWith = _db.CreateEdge(gus, alix, "WORKS_WITH");
        var note = _db.CreateEdge(alix, alix, "NOTE");

        Assert.Equal(
            new[] { new NodeEdge(gus, knowsGus), new NodeEdge(vincent, knowsVincent), new NodeEdge(alix, note) },
            Sorted(_db.GetNodeEdges(alix)));
        Assert.Equal(
            new[] { new NodeEdge(gus, worksWith), new NodeEdge(alix, note) },
            Sorted(_db.GetNodeEdges(alix, Direction.In)));
        Assert.Equal(
            new[]
            {
                new NodeEdge(gus, knowsGus), new NodeEdge(vincent, knowsVincent),
                new NodeEdge(gus, worksWith), new NodeEdge(alix, note),
            },
            Sorted(_db.GetNodeEdges(alix, Direction.Both)));
        Assert.Equal(
            new[] { new NodeEdge(gus, knowsGus), new NodeEdge(vincent, knowsVincent) },
            Sorted(_db.GetNodeEdges(alix, Direction.Both, ["KNOWS"])));

        _db.DeleteEdge(knowsGus);
        Assert.Empty(_db.GetNodeEdges(gus, Direction.In));
        Assert.Empty(_db.GetNodeEdges(88888, Direction.Both));
        Assert.ThrowsAny<GrafeoException>(() => _db.GetNodeEdges(alix, (Direction)7));
    }

    [Fact]
    public void GraphHandleListsTheEdgesOfItsGraph()
    {
        _db.Execute("CREATE GRAPH trips");
        using var trips = _db.Graph("trips");
        var prague = trips.CreateNode(["City"]);
        var berlin = trips.CreateNode(["City"]);
        var route = trips.CreateEdge(prague, berlin, "ROUTE");
        Assert.Equal(
            new NodeEdge(prague, route),
            Assert.Single(trips.GetNodeEdges(berlin, Direction.In, ["ROUTE"])));

        _db.Execute("DROP GRAPH trips");
        Assert.ThrowsAny<GrafeoException>(() => trips.GetNodeEdges(berlin));
    }
}
