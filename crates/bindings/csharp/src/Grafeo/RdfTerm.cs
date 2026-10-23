using System.Text.Json;
using System.Text.Json.Nodes;

namespace Grafeo;

/// <summary>The kind of an RDF term.</summary>
public enum RdfTermType
{
    /// <summary>An IRI.</summary>
    Iri,
    /// <summary>A literal: its value with a language tag or a datatype.</summary>
    Literal,
    /// <summary>A blank node: its value is the label, without <c>_:</c>.</summary>
    BlankNode,
}

/// <summary>
/// An RDF term as a SPARQL result holds it: an IRI, a blank node, or a literal
/// with its language tag or datatype. As a query parameter it is that exact term.
/// </summary>
/// <remarks>
/// Language tags compare without case, so a term keeps its tag in lowercase
/// (<c>"Pomme"@FR</c> is <c>"Pomme"@fr</c>), and a literal of datatype
/// <c>xsd:string</c> is the simple literal: two terms are equal when they are
/// the same RDF term.
/// </remarks>
/// <param name="Type">The kind of term.</param>
/// <param name="Value">The IRI, the blank node's label or the literal's lexical form.</param>
/// <param name="Language">The tag of a language-tagged literal, <c>null</c> otherwise.</param>
/// <param name="Datatype">The datatype IRI of a typed literal other than <c>xsd:string</c>, <c>null</c> otherwise.</param>
public sealed record RdfTerm(RdfTermType Type, string Value, string? Language = null, string? Datatype = null)
{
    /// <summary>The datatype of a simple literal.</summary>
    public const string XsdString = "http://www.w3.org/2001/XMLSchema#string";

    /// <summary>The datatype of a language-tagged literal.</summary>
    public const string RdfLangString = "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString";

    /// <summary>The key of the tag an RDF term travels in as JSON.</summary>
    internal const string TermTag = "$term";

    private readonly string? _language = NormalizedLanguage(Type, Language);
    private readonly string? _datatype = NormalizedDatatype(Type, Language, Datatype);

    /// <summary>The tag of a language-tagged literal, in lowercase; <c>null</c> otherwise.</summary>
    public string? Language
    {
        get => _language;
        init => _language = NormalizedLanguage(Type, value);
    }

    /// <summary>The datatype IRI of a typed literal other than <c>xsd:string</c>, <c>null</c> otherwise.</summary>
    public string? Datatype
    {
        get => _datatype;
        init => _datatype = NormalizedDatatype(Type, _language, value);
    }

    private static string? NormalizedLanguage(RdfTermType type, string? language) =>
        type == RdfTermType.Literal && !string.IsNullOrEmpty(language)
            ? language.ToLowerInvariant()
            : null;

    private static string? NormalizedDatatype(RdfTermType type, string? language, string? datatype) =>
        type != RdfTermType.Literal || !string.IsNullOrEmpty(language) || datatype == XsdString
            ? null
            : datatype;

    /// <summary>
    /// A literal's datatype as RDF reads it: <c>rdf:langString</c> for a tagged
    /// literal, <c>xsd:string</c> for a simple one; <c>null</c> for an IRI or a blank node.
    /// </summary>
    public string? LiteralDatatype => Type switch
    {
        RdfTermType.Literal when Language is not null => RdfLangString,
        RdfTermType.Literal => Datatype ?? XsdString,
        _ => null,
    };

    /// <summary>Whether <paramref name="other"/> is the same RDF term.</summary>
    public bool Equals(RdfTerm? other) =>
        other is not null
        && Type == other.Type
        && Value == other.Value
        && Language == other.Language
        && LiteralDatatype == other.LiteralDatatype;

    /// <inheritdoc/>
    public override int GetHashCode() => HashCode.Combine(Type, Value, Language, LiteralDatatype);

    /// <summary>The term's text: the IRI, the blank node's label or the literal's lexical form.</summary>
    public override string ToString() => Value;

    /// <summary>
    /// The term in the JSON form the C library reads and writes: the <c>$term</c>
    /// tag as the object's only key, holding <c>type</c> (<c>uri</c>, <c>bnode</c>
    /// or <c>literal</c>) and <c>value</c>, and for a literal <c>xml:lang</c> or
    /// <c>datatype</c>.
    /// </summary>
    internal JsonObject ToJson()
    {
        var json = new JsonObject
        {
            ["type"] = Type switch
            {
                RdfTermType.Iri => "uri",
                RdfTermType.BlankNode => "bnode",
                _ => "literal",
            },
            ["value"] = Value,
        };
        if (Language is not null)
            json["xml:lang"] = Language;
        else if (Datatype is not null)
            json["datatype"] = Datatype;
        return new JsonObject { [TermTag] = json };
    }

    /// <summary>
    /// The term a JSON object holds in the form <see cref="ToJson"/> writes;
    /// <c>null</c> for any other object, also one with <c>type</c> and
    /// <c>value</c> keys but no <c>$term</c> tag.
    /// </summary>
    internal static RdfTerm? FromJson(JsonElement tagged)
    {
        if (!tagged.TryGetProperty(TermTag, out var element)
            || element.ValueKind != JsonValueKind.Object
            || tagged.EnumerateObject().Count() != 1)
            return null;
        string? Text(string name) =>
            element.TryGetProperty(name, out var item) && item.ValueKind == JsonValueKind.String
                ? item.GetString()
                : null;
        var type = Text("type");
        var value = Text("value");
        if (type is null || value is null)
            return null;
        var keys = element.EnumerateObject().Count();
        var language = Text("xml:lang");
        var datatype = Text("datatype");
        return type switch
        {
            "uri" when keys == 2 => new RdfTerm(RdfTermType.Iri, value),
            "bnode" when keys == 2 => new RdfTerm(RdfTermType.BlankNode, value),
            "literal" when language is not null && datatype is null && keys == 3 =>
                new RdfTerm(RdfTermType.Literal, value, Language: language),
            "literal" when datatype is not null && language is null && keys == 3 =>
                new RdfTerm(RdfTermType.Literal, value, Datatype: datatype),
            "literal" when keys == 2 => new RdfTerm(RdfTermType.Literal, value),
            _ => null,
        };
    }
}

/// <summary>What a <see cref="QueryResult"/> holds.</summary>
public enum ResultKind
{
    /// <summary>Rows: SPARQL SELECT and updates, and every other language.</summary>
    Rows = 0,
    /// <summary>One boolean: SPARQL ASK, one row with a <c>boolean</c> column.</summary>
    Boolean = 1,
    /// <summary>An RDF graph: SPARQL CONSTRUCT and DESCRIBE, rows of <c>subject</c>, <c>predicate</c> and <c>object</c> terms.</summary>
    Graph = 2,
}
