// Data types for the .gtest YAML spec format.
// Mirrors the structures in the Python and Node.js runners.

namespace SpecRunner;

/// <summary>File-level metadata block from a .gtest file.</summary>
public sealed class Meta
{
    public string Language { get; set; } = "gql";
    public string Model { get; set; } = "";
    public string Section { get; set; } = "";
    public string Title { get; set; } = "";
    public string Dataset { get; set; } = "empty";
    public List<string> Requires { get; set; } = [];
    public List<string> Tags { get; set; } = [];
}

/// <summary>Expected result block for a single test case.</summary>
public sealed class Expect
{
    public List<List<string>> Rows { get; set; } = [];
    public bool Ordered { get; set; }
    public int? Count { get; set; }
    public bool Empty { get; set; }
    public string? Error { get; set; }
    public string? GqlStatus { get; set; }
    public string? Hash { get; set; }
    public int? Precision { get; set; }
    public List<string> Columns { get; set; } = [];

    /// <summary>
    /// The codes of the warnings the statement raises, in the result's order (by code): none
    /// without the key, so every test that runs to a result also checks that it raises no other.
    /// </summary>
    public List<string> Warnings { get; set; } = [];

    /// <summary>Rows of RDF terms in N-Triples form, each cell compared as the exact term.</summary>
    public List<List<string>> Terms { get; set; } = [];

    /// <summary>An expect key no runner reads: the test fails on it, as it would otherwise check nothing.</summary>
    public string? UnknownKey { get; set; }
}

/// <summary>A single test case within a .gtest file.</summary>
public sealed class TestCase
{
    public string Name { get; set; } = "";
    public string? Query { get; set; }
    public List<string> Statements { get; set; } = [];
    public List<string> Setup { get; set; } = [];
    /// <summary>
    /// Each parameter value as the file writes it, quotes included: whether it
    /// was quoted decides its type (see <c>SpecTests.ParamValue</c>).
    /// </summary>
    public Dictionary<string, string> Params { get; set; } = new();
    public List<string> Tags { get; set; } = [];
    public List<string> Requires { get; set; } = [];
    public string? Skip { get; set; }
    /// <summary>The spec and the issue of a case expected to fail (<c>xfail:</c>), null otherwise.</summary>
    public Xfail? Xfail { get; set; }
    /// <summary>A runner probe (<c>probe:</c>), only in a file under a spec-probes directory.</summary>
    public string? Probe { get; set; }
    public string? Language { get; set; }
    public string? Dataset { get; set; }
    public Expect Expect { get; set; } = new();
    public Dictionary<string, string> Variants { get; set; } = new();
}

/// <summary>
/// The spec a case's expected result is held against and the issue that fixes it
/// (<c>#N</c> or <c>fix in X.Y.Z</c>).
/// </summary>
public sealed class Xfail
{
    public string Spec { get; set; } = "";
    public string Issue { get; set; } = "";
}

/// <summary>Top-level structure of a parsed .gtest file.</summary>
public sealed class GtestFile
{
    public Meta Meta { get; set; } = new();
    public List<TestCase> Tests { get; set; } = [];
}
