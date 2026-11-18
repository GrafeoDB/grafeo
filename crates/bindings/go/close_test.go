package grafeo

import (
	"errors"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

// isClosedError tells whether err is the GRAFEO-T007 error of a closed database.
func isClosedError(err error) bool {
	var grafeoErr *Error
	return errors.As(err, &grafeoErr) && strings.Contains(grafeoErr.Message, "GRAFEO-T007")
}

func openPersistent(t *testing.T) (*Database, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "amsterdam.grafeo")
	db, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})"); err != nil {
		t.Fatal(err)
	}
	return db, path
}

// TestEveryMethodOfAClosedDatabaseFailsWithT007 calls every method of a closed
// Database. Each one that returns an error returns GRAFEO-T007; the four
// without an error return answer their neutral value. The table names every
// exported method.
func TestEveryMethodOfAClosedDatabaseFailsWithT007(t *testing.T) {
	db, _ := openPersistent(t)
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("a second Close: %v", err)
	}

	query := "MATCH (p:Person) RETURN p.name"
	dir := t.TempDir()
	vector := []float32{0.1, 0.2, 0.3, 0.4}
	calls := map[string]func() error{
		"Info":                  func() error { _, err := db.Info(); return err },
		"Save":                  func() error { return db.Save(filepath.Join(dir, "copy.grafeo")) },
		"SetCheckpointTriggers": func() error { return db.SetCheckpointTriggers(0, 0) },
		"CheckpointStatus":      func() error { _, err := db.CheckpointStatus(); return err },
		"WalCheckpoint":         func() error { return db.WalCheckpoint() },
		"CreateEdge":            func() error { _, err := db.CreateEdge(0, 1, "KNOWS", nil); return err },
		"GetEdge":               func() error { _, err := db.GetEdge(0); return err },
		"DeleteEdge":            func() error { _, err := db.DeleteEdge(0); return err },
		"SetEdgeProperty":       func() error { return db.SetEdgeProperty(0, "since", 2020) },
		"RemoveEdgeProperty":    func() error { _, err := db.RemoveEdgeProperty(0, "since"); return err },
		"Execute":               func() error { _, err := db.Execute(query); return err },
		"ExecuteParams":         func() error { _, err := db.ExecuteParams(query, nil); return err },
		"ExecuteWithParams":     func() error { _, err := db.ExecuteWithParams(query, "{}"); return err },
		"ExecuteCypher":         func() error { _, err := db.ExecuteCypher(query); return err },
		"ExecuteGremlin":        func() error { _, err := db.ExecuteGremlin("g.V()"); return err },
		"ExecuteGraphQL":        func() error { _, err := db.ExecuteGraphQL("{ Person { name } }"); return err },
		"ExecuteSPARQL":         func() error { _, err := db.ExecuteSPARQL("SELECT ?s WHERE { ?s ?p ?o }"); return err },
		"ExecuteSQL":            func() error { _, err := db.ExecuteSQL(query); return err },
		"ExecuteCypherWithParams": func() error {
			_, err := db.ExecuteCypherWithParams(query, "{}")
			return err
		},
		"ExecuteGremlinWithParams": func() error {
			_, err := db.ExecuteGremlinWithParams("g.V()", "{}")
			return err
		},
		"ExecuteGraphQLWithParams": func() error {
			_, err := db.ExecuteGraphQLWithParams("{ Person { name } }", "{}")
			return err
		},
		"ExecuteSPARQLWithParams": func() error {
			_, err := db.ExecuteSPARQLWithParams("SELECT ?s WHERE { ?s ?p ?o }", "{}")
			return err
		},
		"ExecuteSQLWithParams": func() error { _, err := db.ExecuteSQLWithParams(query, "{}"); return err },
		"ExecuteLanguage":      func() error { _, err := db.ExecuteLanguage("gql", query, ""); return err },
		"DropVectorIndex":      func() error { _, err := db.DropVectorIndex("Item", "embedding"); return err },
		"RebuildVectorIndex":   func() error { return db.RebuildVectorIndex("Item", "embedding") },
		"MmrSearch": func() error {
			_, err := db.MmrSearch("Item", "embedding", vector, 1, 4, 0.5, 16)
			return err
		},
		"NodeCount":           func() error { _, err := db.NodeCount(); return err },
		"EdgeCount":           func() error { _, err := db.EdgeCount(); return err },
		"Graph":               func() error { _, err := db.Graph("berlin"); return err },
		"CreatePropertyIndex": func() error { return db.CreatePropertyIndex("name") },
		"DropPropertyIndex":   func() error { _, err := db.DropPropertyIndex("name"); return err },
		"FindNodesByProperty": func() error { _, err := db.FindNodesByProperty("name", "Alix"); return err },
		"MemoryUsage":         func() error { _, err := db.MemoryUsage(); return err },
		"OpenWarnings":        func() error { _, err := db.OpenWarnings(); return err },
		"CreateNode":          func() error { _, err := db.CreateNode([]string{"Person"}, nil); return err },
		"GetNode":             func() error { _, err := db.GetNode(0); return err },
		"DeleteNode":          func() error { _, err := db.DeleteNode(0); return err },
		"DetachDeleteNode":    func() error { _, err := db.DetachDeleteNode(0); return err },
		"SetNodeProperty":     func() error { return db.SetNodeProperty(0, "city", "Amsterdam") },
		"RemoveNodeProperty":  func() error { _, err := db.RemoveNodeProperty(0, "name"); return err },
		"AddNodeLabel":        func() error { _, err := db.AddNodeLabel(0, "Director"); return err },
		"RemoveNodeLabel":     func() error { _, err := db.RemoveNodeLabel(0, "Person"); return err },
		"GetNodeLabels":       func() error { _, err := db.GetNodeLabels(0); return err },
		"GetNodeEdges":        func() error { _, err := db.GetNodeEdges(0, Direction(0)); return err },
		"BatchCreateNodesWithProps": func() error {
			_, err := db.BatchCreateNodesWithProps([]string{"Person"}, []map[string]any{{"name": "Mia"}})
			return err
		},
		"BatchCreateEdges": func() error { _, err := db.BatchCreateEdges(nil); return err },
		"BatchVectorSearch": func() error {
			_, err := db.BatchVectorSearch("Item", "embedding", [][]float32{vector}, 1, 0, nil)
			return err
		},
		"NodeHistory":      func() error { _, err := db.NodeHistory(0); return err },
		"EdgeHistory":      func() error { _, err := db.EdgeHistory(0); return err },
		"NodeHistorySince": func() error { _, err := db.NodeHistorySince(0, 0); return err },
		"ChangesBetween":   func() error { _, err := db.ChangesBetween(0, 1); return err },
		"UpsertNodes": func() error {
			_, err := db.UpsertNodes([]string{"Person"}, []map[string]any{{"id": 1}}, nil)
			return err
		},
		"UpsertEdges": func() error {
			_, err := db.UpsertEdges("KNOWS", []map[string]any{{"id": 1, "src": 1, "dst": 2}}, nil)
			return err
		},
		"ExecuteStream":        func() error { _, err := db.ExecuteStream(query); return err },
		"EpochAsOf":            func() error { _, err := db.EpochAsOf(AtEpoch(0)); return err },
		"ExecuteAsOf":          func() error { _, err := db.ExecuteAsOf(query, AtEpoch(0), nil); return err },
		"AsOf":                 func() error { _, err := db.AsOf(AtEpoch(0)); return err },
		"BeginTransaction":     func() error { _, err := db.BeginTransaction(); return err },
		"BeginTransactionWith": func() error { _, err := db.BeginTransactionWith(IsolationLevel(0)); return err },
		"CreateVectorIndex":    func() error { return db.CreateVectorIndex("Item", "embedding") },
		"VectorSearch":         func() error { _, err := db.VectorSearch("Item", "embedding", vector, 1); return err },
		"BatchCreateNodes": func() error {
			_, err := db.BatchCreateNodes("Item", "embedding", [][]float32{vector})
			return err
		},
	}
	// The methods without an error return: their neutral value after Close.
	neutral := map[string]func() bool{
		"HasPropertyIndex": func() bool { return !db.HasPropertyIndex("name") },
		"SetCDCEnabled":    func() bool { db.SetCDCEnabled(true); return true },
		"IsCDCEnabled":     func() bool { return !db.IsCDCEnabled() },
		"CurrentEpoch":     func() bool { return db.CurrentEpoch() == 0 },
	}

	var unlisted []string
	dbType := reflect.TypeOf(db)
	for i := 0; i < dbType.NumMethod(); i++ {
		name := dbType.Method(i).Name
		_, inCalls := calls[name]
		_, inNeutral := neutral[name]
		if name != "Close" && !inCalls && !inNeutral {
			unlisted = append(unlisted, name)
		}
	}
	if len(unlisted) > 0 {
		t.Errorf("methods without a row in this table: %v", unlisted)
	}

	var missed []string
	for name, call := range calls {
		if err := call(); !isClosedError(err) {
			missed = append(missed, name+": "+errorText(err))
		}
	}
	for name, answeredNeutral := range neutral {
		if !answeredNeutral() {
			missed = append(missed, name+": not its neutral value")
		}
	}
	sort.Strings(missed)
	if len(missed) > 0 {
		t.Errorf("calls that did not fail with GRAFEO-T007:\n%s", strings.Join(missed, "\n"))
	}
}

func errorText(err error) string {
	if err == nil {
		return "answered"
	}
	return err.Error()
}

// TestWhatAClosedDatabaseHandedOutFailsWithT007 holds a graph, a transaction,
// a stream and a view across Close: each call on them fails with GRAFEO-T007,
// and the transaction leaves nothing in the file.
func TestWhatAClosedDatabaseHandedOutFailsWithT007(t *testing.T) {
	db, path := openPersistent(t)
	if _, err := db.Execute("CREATE GRAPH berlin"); err != nil {
		t.Fatal(err)
	}
	graph, err := db.Graph("berlin")
	if err != nil {
		t.Fatal(err)
	}
	tx, err := db.BeginTransaction()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := tx.Execute("INSERT (:Person {name: 'Mia'})"); err != nil {
		t.Fatal(err)
	}
	query := "MATCH (p:Person) RETURN p.name"
	stream, err := db.ExecuteStream(query)
	if err != nil {
		t.Fatal(err)
	}
	view, err := db.AsOf(AtEpoch(db.CurrentEpoch()))
	if err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	rows := map[string]func() error{
		"graph.Execute": func() error { _, err := graph.Execute(query); return err },
		"tx.Execute":    func() error { _, err := tx.Execute(query); return err },
		"tx.Commit":     func() error { return tx.Commit() },
		"stream.Next":   func() error { _, err := stream.Next(); return err },
		"view.Execute":  func() error { _, err := view.Execute(query); return err },
		"view.GetNode":  func() error { _, err := view.GetNode(0); return err },
	}
	var missed []string
	for name, call := range rows {
		if err := call(); !isClosedError(err) {
			missed = append(missed, name+": "+errorText(err))
		}
	}
	sort.Strings(missed)
	if len(missed) > 0 {
		t.Errorf("calls that did not fail with GRAFEO-T007:\n%s", strings.Join(missed, "\n"))
	}
	stream.Close()
	view.Close()

	reopened, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	count, err := reopened.NodeCount()
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Errorf("the transaction open across Close left %d nodes, want 2", count)
	}
}
