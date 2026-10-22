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
}
