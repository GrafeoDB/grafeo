namespace Grafeo;

/// <summary>
/// Interface for the Grafeo database, enabling testability via mocking.
/// </summary>
public interface IGrafeoDB : IDisposable, IAsyncDisposable
{
    // Query execution
    QueryResult Execute(string query);
    Task<QueryResult> ExecuteAsync(string query, CancellationToken ct = default);
    QueryResult ExecuteWithParams(string query, Dictionary<string, object?> parameters);
    QueryResult ExecuteLanguage(string language, string query, Dictionary<string, object?>? parameters = null);

    // Transactions
    ITransaction BeginTransaction();
    ITransaction BeginTransaction(IsolationLevel isolationLevel);

    // Node CRUD
    long CreateNode(IEnumerable<string> labels, Dictionary<string, object?>? properties = null);
    Node? GetNode(long id);
    bool DeleteNode(long id);
    void SetNodeProperty(long id, string key, object? value);
    bool RemoveNodeProperty(long id, string key);
    bool AddNodeLabel(long id, string label);
    bool RemoveNodeLabel(long id, string label);

    // Edge CRUD
    long CreateEdge(long sourceId, long targetId, string edgeType, Dictionary<string, object?>? properties = null);
    Edge? GetEdge(long id);
    bool DeleteEdge(long id);
    void SetEdgeProperty(long id, string key, object? value);
    bool RemoveEdgeProperty(long id, string key);

    // Admin
    long NodeCount { get; }
    long EdgeCount { get; }
    IReadOnlyDictionary<string, object?> Info();
    void Save(string path);
    void ClearPlanCache();

    // Schema
    void SetSchema(string name);
    void ResetSchema();
    string? CurrentSchema();
}
