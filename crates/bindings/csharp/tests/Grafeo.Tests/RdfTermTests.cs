using Xunit;

namespace Grafeo.Tests;

/// <summary>SPARQL results hold exact RDF terms, as <see cref="RdfTerm"/> (#464).</summary>
public sealed class RdfTermTests : IDisposable
{
    private const string Xsd = "http://www.w3.org/2001/XMLSchema#";
    private readonly GrafeoDB _db = GrafeoDB.Memory();

    public void Dispose() => _db.Dispose();

    private void InsertFruit() =>
        _db.ExecuteSparql(
            $$"""
            INSERT DATA {
                <http://example.org/apple> <http://example.org/label> "Pomme"@fr ;
                    <http://example.org/price> "1.50"^^<{{Xsd}}decimal> ;
                    <http://example.org/note> "fresh" ;
                    <http://example.org/grower> _:farm .
            }
            """);

    [Fact]
    public void SparqlResultsHoldEveryKindOfTerm()
    {
        InsertFruit();
        var result = _db.ExecuteSparql("SELECT ?p ?o WHERE { <http://example.org/apple> ?p ?o }");

        Assert.Equal(ResultKind.Rows, result.Kind);
        var objects = result.Rows.ToDictionary(
            row => Assert.IsType<RdfTerm>(row["p"]).Value,
            row => Assert.IsType<RdfTerm>(row["o"]));
        Assert.Equal(
            new RdfTerm(RdfTermType.Literal, "Pomme", Language: "fr"),
            objects["http://example.org/label"]);
        Assert.Equal(RdfTerm.RdfLangString, objects["http://example.org/label"].LiteralDatatype);
        Assert.Equal(
            new RdfTerm(RdfTermType.Literal, "1.50", Datatype: Xsd + "decimal"),
            objects["http://example.org/price"]);
        Assert.Equal(new RdfTerm(RdfTermType.Literal, "fresh"), objects["http://example.org/note"]);
        Assert.Equal(RdfTerm.XsdString, objects["http://example.org/note"].LiteralDatatype);
        Assert.Equal(RdfTermType.BlankNode, objects["http://example.org/grower"].Type);
    }

    [Fact]
    public void AskIsOneBooleanAndConstructAGraph()
    {
        InsertFruit();
        var ask = _db.ExecuteSparql("ASK { ?s <http://example.org/label> \"Pomme\"@fr }");
        Assert.Equal(ResultKind.Boolean, ask.Kind);
        Assert.True(ask.Boolean);
        Assert.False(_db.ExecuteSparql("ASK { ?s <http://example.org/label> \"Pomme\"@en }").Boolean);

        var graph = _db.ExecuteSparql("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }");
        Assert.Equal(ResultKind.Graph, graph.Kind);
        Assert.Equal(4, graph.Rows.Count);
        Assert.Null(graph.Boolean);
    }

    [Fact]
    public void ATermRoundTripsAsAParameter()
    {
        var terms = new[]
        {
            new RdfTerm(RdfTermType.Iri, "http://example.org/Kraków"),
            new RdfTerm(RdfTermType.BlankNode, "b0"),
            new RdfTerm(RdfTermType.Literal, "Pomme", Language: "fr"),
            new RdfTerm(RdfTermType.Literal, "2026-01-15", Datatype: Xsd + "date"),
            new RdfTerm(RdfTermType.Literal, "Apple"),
        };
        foreach (var term in terms)
        {
            var result = _db.ExecuteWithParams(
                "RETURN $term AS term",
                new Dictionary<string, object?> { ["term"] = term });
            Assert.Equal(term, result.Rows[0]["term"]);
        }
    }

    [Fact]
    public void AMapShapedLikeASparqlJsonTermStaysAMap()
    {
        var maps = new[]
        {
            new Dictionary<string, object?> { ["type"] = "uri", ["value"] = "http://example.org/Paris" },
            new Dictionary<string, object?> { ["type"] = "literal", ["value"] = "Pomme", ["xml:lang"] = "fr" },
            new Dictionary<string, object?> { ["type"] = "bnode", ["value"] = "b0" },
        };
        foreach (var map in maps)
        {
            var result = _db.ExecuteWithParams(
                "RETURN $m AS m, $m.type AS kind",
                new Dictionary<string, object?> { ["m"] = map });
            var back = Assert.IsAssignableFrom<IReadOnlyDictionary<string, object?>>(result.Rows[0]["m"]);
            Assert.Equal(map.Count, back.Count);
            foreach (var (key, value) in map)
                Assert.Equal(value, back[key]);
            Assert.Equal(map["type"], result.Rows[0]["kind"]);
        }
    }

    [Fact]
    public void ALanguageTagIsKeptInLowercase()
    {
        var upper = new RdfTerm(RdfTermType.Literal, "Pomme", Language: "FR");
        Assert.Equal("fr", upper.Language);
        Assert.Equal(new RdfTerm(RdfTermType.Literal, "Pomme", Language: "fr"), upper);
        Assert.Equal(
            new RdfTerm(RdfTermType.Literal, "Pomme", Language: "fr").GetHashCode(),
            upper.GetHashCode());
        Assert.Equal("en-gb", new RdfTerm(RdfTermType.Literal, "Colour", Language: "en-GB").Language);
        Assert.Equal("nl", (upper with { Language = "NL" }).Language);
        Assert.NotEqual(new RdfTerm(RdfTermType.Literal, "Pomme", Language: "nl"), upper);

        var result = _db.ExecuteWithParams(
            "RETURN $term AS term",
            new Dictionary<string, object?> { ["term"] = upper });
        var back = Assert.IsType<RdfTerm>(result.Rows[0]["term"]);
        Assert.Equal("fr", back.Language);
        Assert.Equal(upper, back);
    }

    [Fact]
    public void ALiteralOfDatatypeXsdStringIsTheSimpleLiteral()
    {
        var typed = new RdfTerm(RdfTermType.Literal, "Apple", Datatype: RdfTerm.XsdString);
        Assert.Null(typed.Datatype);
        Assert.Equal(new RdfTerm(RdfTermType.Literal, "Apple"), typed);
        Assert.Equal(RdfTerm.XsdString, typed.LiteralDatatype);
        Assert.Null(new RdfTerm(RdfTermType.Literal, "Pomme", Language: "fr", Datatype: RdfTerm.RdfLangString).Datatype);
        Assert.Null(new RdfTerm(RdfTermType.Iri, "http://example.org/Berlin", Language: "de").Language);
    }
}
