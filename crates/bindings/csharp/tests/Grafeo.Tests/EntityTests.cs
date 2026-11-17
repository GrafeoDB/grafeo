using Xunit;

namespace Grafeo.Tests;

/// <summary>
/// The nodes and edges of a query result: a property named like a structural key (<c>_id</c>,
/// <c>_labels</c>, <c>_type</c>, <c>_source</c>, <c>_target</c>) never replaces the identity, and a
/// property whose name starts with <c>_</c> is kept (#254).
/// </summary>
public sealed class EntityTests : IDisposable
{
    private readonly GrafeoDB _db = GrafeoDB.Memory();

    public void Dispose() => _db.Dispose();

    [Fact]
    public void PropertiesNamedLikeStructuralKeysKeepTheIdentity()
    {
        _db.Execute(
            "INSERT (:Person {name: 'Alix', _id: 'x', _created: 2020})"
            + "-[:KNOWS {_note: 'Paris', _source: 'manual'}]->(:Person {name: 'Gus', _id: 0})");
        var ids = _db.Execute("MATCH (p:Person) RETURN p.name AS name, id(p) AS id")
            .Rows.ToDictionary(row => (string)row["name"]!, row => (long)row["id"]!);

        var nodes = _db.Execute("MATCH (p:Person) RETURN p")
            .Nodes.ToDictionary(node => (string)node.Properties["name"]!);
        Assert.Equal(["Alix", "Gus"], nodes.Keys.Order());
        Assert.Equal(ids["Alix"], nodes["Alix"].Id);
        Assert.Equal(ids["Gus"], nodes["Gus"].Id);
        Assert.Equal(2020L, nodes["Alix"].Properties["_created"]);
        Assert.False(nodes["Alix"].Properties.ContainsKey("_id"));

        var edge = Assert.Single(_db.Execute("MATCH ()-[e:KNOWS]->() RETURN e").Edges);
        Assert.Equal((ids["Alix"], ids["Gus"]), (edge.SourceId, edge.TargetId));
        Assert.Equal("Paris", edge.Properties["_note"]);
    }

    /// <summary>
    /// A value is a node or an edge where the engine typed it so: a map built with the keys of
    /// a node is no node, an edge with a property named <c>_labels</c> is an edge, and nested
    /// nodes count.
    /// </summary>
    [Fact]
    public void NodesAndEdgesAreKnownByTypeNotByKeys()
    {
        _db.Execute(
            "INSERT (:Person {name: 'Alix'})-[:KNOWS {_labels: ['L']}]->(:Person {name: 'Gus'})");

        var forged = _db.Execute(
            "MATCH (a:Person {name: 'Alix'}) "
            + "RETURN a, {_id: 99, _labels: ['Person']} AS fake, "
            + "{_id: 98, _type: 'KNOWS', _source: 1, _target: 2} AS fakeEdge");
        Assert.Equal("Alix", Assert.Single(forged.Nodes).Properties["name"]);
        Assert.Empty(forged.Edges);

        var edges = _db.Execute("MATCH ()-[r:KNOWS]->() RETURN r");
        Assert.Empty(edges.Nodes);
        var edge = Assert.Single(edges.Edges);
        Assert.Equal("KNOWS", edge.Type);
        Assert.Equal(new List<object?> { "L" }, edge.Properties["_labels"]);

        var nested = _db.Execute(
            "MATCH (a:Person {name: 'Alix'})-[r]->(b) RETURN [a] AS l, {k: b} AS m");
        Assert.Equal(["Alix", "Gus"], nested.Nodes.Select(node => (string)node.Properties["name"]!).Order());
    }
}
