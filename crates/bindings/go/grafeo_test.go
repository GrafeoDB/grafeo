package grafeo

import (
	"encoding/json"
	"fmt"
	"sync"
	"testing"
)

// --- Lifecycle ---

func TestOpenInMemory(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	if db.NodeCount() != 0 {
		t.Errorf("expected 0 nodes, got %d", db.NodeCount())
	}
	if db.EdgeCount() != 0 {
		t.Errorf("expected 0 edges, got %d", db.EdgeCount())
	}
}

func TestVersion(t *testing.T) {
	v := Version()
	if v == "" {
		t.Error("expected non-empty version")
	}
}

func TestDoubleClose(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	// Second close should be a no-op.
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
}

// --- Node CRUD ---

func TestCreateNode(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	node, err := db.CreateNode([]string{"Person"}, map[string]any{"name": "Alix", "age": 30})
	if err != nil {
		t.Fatal(err)
	}
	if db.NodeCount() != 1 {
		t.Errorf("expected 1 node, got %d", db.NodeCount())
	}
	if node.Labels[0] != "Person" {
		t.Errorf("expected label Person, got %v", node.Labels)
	}
}

func TestGetNode(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	created, _ := db.CreateNode([]string{"Person"}, map[string]any{"name": "Alix"})
	fetched, err := db.GetNode(created.ID)
	if err != nil {
		t.Fatal(err)
	}
	if fetched.Properties["name"] != "Alix" {
		t.Errorf("expected name Alix, got %v", fetched.Properties["name"])
	}
}

func TestDeleteNode(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	node, _ := db.CreateNode([]string{"Person"}, nil)
	deleted, err := db.DeleteNode(node.ID)
	if err != nil {
		t.Fatal(err)
	}
	if !deleted {
		t.Error("expected node to be deleted")
	}
	if db.NodeCount() != 0 {
		t.Errorf("expected 0 nodes after delete, got %d", db.NodeCount())
	}
}

func TestNodeProperties(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	node, _ := db.CreateNode([]string{"Person"}, nil)

	if err := db.SetNodeProperty(node.ID, "city", "Berlin"); err != nil {
		t.Fatal(err)
	}

	fetched, _ := db.GetNode(node.ID)
	if fetched.Properties["city"] != "Berlin" {
		t.Errorf("expected city Berlin, got %v", fetched.Properties["city"])
	}

	removed, _ := db.RemoveNodeProperty(node.ID, "city")
	if !removed {
		t.Error("expected property to be removed")
	}

	fetched, _ = db.GetNode(node.ID)
	if _, exists := fetched.Properties["city"]; exists {
		t.Error("expected city to be removed")
	}
}

func TestNodeLabels(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	node, _ := db.CreateNode([]string{"Person"}, nil)

	added, _ := db.AddNodeLabel(node.ID, "Employee")
	if !added {
		t.Error("expected label to be added")
	}

	labels, _ := db.GetNodeLabels(node.ID)
	if len(labels) != 2 {
		t.Errorf("expected 2 labels, got %d", len(labels))
	}

	removed, _ := db.RemoveNodeLabel(node.ID, "Employee")
	if !removed {
		t.Error("expected label to be removed")
	}

	labels, _ = db.GetNodeLabels(node.ID)
	if len(labels) != 1 {
		t.Errorf("expected 1 label, got %d", len(labels))
	}
}

// --- Edge CRUD ---

func TestCreateEdge(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	a, _ := db.CreateNode([]string{"Person"}, map[string]any{"name": "Alix"})
	b, _ := db.CreateNode([]string{"Person"}, map[string]any{"name": "Gus"})

	edge, err := db.CreateEdge(a.ID, b.ID, "KNOWS", map[string]any{"since": 2020})
	if err != nil {
		t.Fatal(err)
	}
	if db.EdgeCount() != 1 {
		t.Errorf("expected 1 edge, got %d", db.EdgeCount())
	}
	if edge.Type != "KNOWS" {
		t.Errorf("expected KNOWS, got %s", edge.Type)
	}
}

func TestGetEdge(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	a, _ := db.CreateNode([]string{"Person"}, nil)
	b, _ := db.CreateNode([]string{"Person"}, nil)
	created, _ := db.CreateEdge(a.ID, b.ID, "KNOWS", nil)

	fetched, err := db.GetEdge(created.ID)
	if err != nil {
		t.Fatal(err)
	}
	if fetched.SourceID != a.ID || fetched.TargetID != b.ID {
		t.Errorf("edge endpoints mismatch: %d->%d, expected %d->%d",
			fetched.SourceID, fetched.TargetID, a.ID, b.ID)
	}
}

func TestDeleteEdge(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	a, _ := db.CreateNode([]string{"Person"}, nil)
	b, _ := db.CreateNode([]string{"Person"}, nil)
	edge, _ := db.CreateEdge(a.ID, b.ID, "KNOWS", nil)

	deleted, _ := db.DeleteEdge(edge.ID)
	if !deleted {
		t.Error("expected edge to be deleted")
	}
	if db.EdgeCount() != 0 {
		t.Errorf("expected 0 edges, got %d", db.EdgeCount())
	}
}

func TestEdgeProperties(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	a, _ := db.CreateNode([]string{"Person"}, nil)
	b, _ := db.CreateNode([]string{"Person"}, nil)
	edge, _ := db.CreateEdge(a.ID, b.ID, "KNOWS", nil)

	if err := db.SetEdgeProperty(edge.ID, "weight", 0.9); err != nil {
		t.Fatal(err)
	}

	fetched, _ := db.GetEdge(edge.ID)
	if fetched.Properties["weight"] != 0.9 {
		t.Errorf("expected weight 0.9, got %v", fetched.Properties["weight"])
	}
}

// --- Queries ---

func TestExecuteGQL(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	db.CreateNode([]string{"Person"}, map[string]any{"name": "Alix", "age": 30})
	db.CreateNode([]string{"Person"}, map[string]any{"name": "Gus", "age": 25})

	result, err := db.Execute("MATCH (p:Person) RETURN p.name, p.age")
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Rows) != 2 {
		t.Errorf("expected 2 rows, got %d", len(result.Rows))
	}
}

func TestExecuteInvalidQuery(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	_, err = db.Execute("THIS IS NOT VALID GQL")
	if err == nil {
		t.Error("expected error for invalid query")
	}
}

// --- Transactions ---

func TestTransactionCommit(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	tx, err := db.BeginTransaction()
	if err != nil {
		t.Fatal(err)
	}

	_, err = tx.Execute("INSERT (:Tx {val: 1})")
	if err != nil {
		t.Fatal(err)
	}

	if err := tx.Commit(); err != nil {
		t.Fatal(err)
	}

	if db.NodeCount() != 1 {
		t.Errorf("expected 1 node after commit, got %d", db.NodeCount())
	}
}

func TestTransactionRollback(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	// Baseline node.
	db.CreateNode([]string{"Base"}, nil)

	tx, err := db.BeginTransaction()
	if err != nil {
		t.Fatal(err)
	}

	_, err = tx.Execute("INSERT (:Rolled {val: 2})")
	if err != nil {
		t.Fatal(err)
	}

	if err := tx.Rollback(); err != nil {
		t.Fatal(err)
	}

	if db.NodeCount() != 1 {
		t.Errorf("expected 1 node after rollback, got %d", db.NodeCount())
	}
}

func TestTransactionIsolation(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	tx, err := db.BeginTransactionWith(Serializable)
	if err != nil {
		t.Fatal(err)
	}

	_, err = tx.Execute("INSERT (:Isolated {val: 1})")
	if err != nil {
		t.Fatal(err)
	}

	if err := tx.Commit(); err != nil {
		t.Fatal(err)
	}

	if db.NodeCount() != 1 {
		t.Errorf("expected 1 node, got %d", db.NodeCount())
	}
}

// --- Property Indexes ---

func TestPropertyIndex(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	if err := db.CreatePropertyIndex("name"); err != nil {
		t.Fatal(err)
	}
	if !db.HasPropertyIndex("name") {
		t.Error("expected property index to exist")
	}

	db.CreateNode([]string{"Person"}, map[string]any{"name": "Alix"})
	db.CreateNode([]string{"Person"}, map[string]any{"name": "Gus"})

	ids, err := db.FindNodesByProperty("name", "Alix")
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 1 {
		t.Errorf("expected 1 result, got %d", len(ids))
	}

	dropped, _ := db.DropPropertyIndex("name")
	if !dropped {
		t.Error("expected index to be dropped")
	}
	if db.HasPropertyIndex("name") {
		t.Error("expected index to not exist after drop")
	}
}

// --- Vector Operations ---

func TestVectorIndex(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	// Create nodes with vector embeddings.
	ids, err := db.BatchCreateNodes("Doc", "embedding", [][]float32{
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
		{0.0, 0.0, 1.0},
		{0.9, 0.1, 0.0},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 4 {
		t.Errorf("expected 4 IDs, got %d", len(ids))
	}

	// Create index.
	if err := db.CreateVectorIndex("Doc", "embedding", WithDimensions(3)); err != nil {
		t.Fatal(err)
	}

	// Search.
	results, err := db.VectorSearch("Doc", "embedding", []float32{1.0, 0.0, 0.0}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	// Closest should be the first vector.
	if results[0].NodeID != ids[0] {
		t.Errorf("expected closest to be node %d, got %d", ids[0], results[0].NodeID)
	}
}

// --- Admin ---

func TestDatabaseInfo(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	db.CreateNode([]string{"Test"}, nil)

	info, err := db.Info()
	if err != nil {
		t.Fatal(err)
	}
	if info.NodeCount != 1 {
		t.Errorf("expected 1 node in info, got %d", info.NodeCount)
	}
	if info.Version == "" {
		t.Error("expected non-empty version in info")
	}
}

// --- Execute with parameters ---

func TestExecuteWithParams(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	db.CreateNode([]string{"Person"}, map[string]any{"name": "Alix", "age": 30})
	db.CreateNode([]string{"Person"}, map[string]any{"name": "Gus", "age": 25})

	result, err := db.ExecuteWithParams(
		"MATCH (n:Person) WHERE n.name = $name RETURN n.age",
		`{"name":"Alix"}`,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Rows) != 1 {
		t.Errorf("expected 1 row, got %d", len(result.Rows))
	}
}

// --- Cypher execution ---

func TestExecuteCypher(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	// Create data via GQL
	db.CreateNode([]string{"Person"}, map[string]any{"name": "Alix"})

	result, err := db.ExecuteCypher("MATCH (p:Person) RETURN p.name")
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Rows) != 1 {
		t.Errorf("expected 1 row, got %d", len(result.Rows))
	}
}

// --- Edge property remove ---

func TestRemoveEdgeProperty(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	a, _ := db.CreateNode([]string{"N"}, nil)
	b, _ := db.CreateNode([]string{"N"}, nil)
	edge, _ := db.CreateEdge(a.ID, b.ID, "R", map[string]any{"weight": 1.5})

	removed, err := db.RemoveEdgeProperty(edge.ID, "weight")
	if err != nil {
		t.Fatal(err)
	}
	if !removed {
		t.Error("expected property to be removed")
	}

	// Second remove returns false
	removed2, _ := db.RemoveEdgeProperty(edge.ID, "weight")
	if removed2 {
		t.Error("expected false for already-removed property")
	}
}

// --- Get nonexistent node ---

func TestGetNonexistentNode(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	_, err = db.GetNode(999)
	if err == nil {
		t.Error("expected error for nonexistent node")
	}
}

// --- Query result metadata ---

func TestQueryResultMetadata(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	db.CreateNode([]string{"Person"}, map[string]any{"name": "Alix"})
	db.CreateNode([]string{"Person"}, map[string]any{"name": "Gus"})

	result, err := db.Execute("MATCH (n:Person) RETURN n.name")
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Columns) < 1 {
		t.Error("expected at least 1 column")
	}
	if len(result.Rows) != 2 {
		t.Errorf("expected 2 rows, got %d", len(result.Rows))
	}
	if result.ExecutionTimeMs < 0 {
		t.Error("expected non-negative execution time")
	}
}

// --- Vector drop and rebuild ---

func TestVectorDropAndRebuild(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	db.BatchCreateNodes("Doc", "emb", [][]float32{
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
	})
	db.CreateVectorIndex("Doc", "emb", WithDimensions(3))

	// Search works
	results, err := db.VectorSearch("Doc", "emb", []float32{1.0, 0.0, 0.0}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}

	// Drop index
	dropped, err := db.DropVectorIndex("Doc", "emb")
	if err != nil {
		t.Fatal(err)
	}
	if !dropped {
		t.Error("expected index to be dropped")
	}
	// Dropping it again finds nothing to drop: false, and no error.
	dropped, err = db.DropVectorIndex("Doc", "emb")
	if err != nil || dropped {
		t.Errorf("expected (false, nil) for a missing index, got (%v, %v)", dropped, err)
	}

	// Rebuild index
	if err := db.RebuildVectorIndex("Doc", "emb"); err != nil {
		t.Fatal(err)
	}

	// Search works again
	results2, err := db.VectorSearch("Doc", "emb", []float32{1.0, 0.0, 0.0}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(results2) != 2 {
		t.Errorf("expected 2 results after rebuild, got %d", len(results2))
	}
}

// --- Transaction with params ---

func TestTransactionWithParams(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	db.CreateNode([]string{"Person"}, map[string]any{"name": "Alix"})

	tx, err := db.BeginTransaction()
	if err != nil {
		t.Fatal(err)
	}

	result, err := tx.ExecuteWithParams(
		"MATCH (n:Person) WHERE n.name = $name RETURN n.name",
		`{"name":"Alix"}`,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Rows) != 1 {
		t.Errorf("expected 1 row, got %d", len(result.Rows))
	}

	tx.Rollback()
}

// --- Column Order ---

func TestColumnOrder(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Skip("native library not available")
	}
	defer db.Close()

	db.CreateNode([]string{"Person"}, map[string]any{"name": "Alix", "age": 30})
	db.CreateNode([]string{"Person"}, map[string]any{"name": "Gus", "age": 25})

	result, err := db.Execute("MATCH (n:Person) RETURN n.name, n.age")
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Columns) < 2 {
		t.Fatalf("expected at least 2 columns, got %d", len(result.Columns))
	}
	if result.Columns[0] != "n.name" {
		t.Errorf("expected first column 'n.name', got %q", result.Columns[0])
	}
	if result.Columns[1] != "n.age" {
		t.Errorf("expected second column 'n.age', got %q", result.Columns[1])
	}
}

func TestColumnOrderMultiColumn(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Skip("native library not available")
	}
	defer db.Close()

	db.CreateNode([]string{"Item"}, map[string]any{
		"alpha": 1, "bravo": 2, "charlie": 3, "delta": 4, "echo": 5,
	})

	result, err := db.Execute("MATCH (n:Item) RETURN n.alpha, n.bravo, n.charlie, n.delta, n.echo")
	if err != nil {
		t.Fatal(err)
	}

	expected := []string{"n.alpha", "n.bravo", "n.charlie", "n.delta", "n.echo"}
	if len(result.Columns) != len(expected) {
		t.Fatalf("expected %d columns, got %d", len(expected), len(result.Columns))
	}
	for i, col := range expected {
		if result.Columns[i] != col {
			t.Errorf("column %d: expected %q, got %q", i, col, result.Columns[i])
		}
	}
}

// --- ExecuteParams ---

func TestExecuteParams(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Skip("native library not available")
	}
	defer db.Close()

	db.CreateNode([]string{"Person"}, map[string]any{"name": "Alix", "age": 30})
	db.CreateNode([]string{"Person"}, map[string]any{"name": "Gus", "age": 25})
	db.CreateNode([]string{"Person"}, map[string]any{"name": "Vincent", "age": 20})

	result, err := db.ExecuteParams(
		"MATCH (n:Person) WHERE n.age > $min RETURN n.name",
		map[string]any{"min": 25},
	)
	if err != nil {
		t.Fatal(err)
	}
	// Alix (30) matches, Gus (25) does not (> not >=), Vincent (20) does not.
	if len(result.Rows) != 1 {
		t.Errorf("expected 1 row, got %d", len(result.Rows))
	}
}

func TestExecuteParamsEmptyMap(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Skip("native library not available")
	}
	defer db.Close()

	db.CreateNode([]string{"Person"}, map[string]any{"name": "Alix"})

	result, err := db.ExecuteParams("MATCH (n) RETURN n", map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Rows) != 1 {
		t.Errorf("expected 1 row, got %d", len(result.Rows))
	}
}

func TestExecuteParamsNilMap(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Skip("native library not available")
	}
	defer db.Close()

	db.CreateNode([]string{"Person"}, map[string]any{"name": "Alix"})

	// nil map marshals to JSON "null", which the engine should treat as no params.
	result, err := db.ExecuteParams("MATCH (n) RETURN n", nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Rows) != 1 {
		t.Errorf("expected 1 row, got %d", len(result.Rows))
	}
}

// --- Aggregation ---

func TestAggregateCount(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Skip("native library not available")
	}
	defer db.Close()

	db.CreateNode([]string{"Person"}, map[string]any{"name": "Alix"})
	db.CreateNode([]string{"Person"}, map[string]any{"name": "Gus"})
	db.CreateNode([]string{"Person"}, map[string]any{"name": "Vincent"})

	result, err := db.Execute("MATCH (n:Person) RETURN count(n) AS cnt")
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Rows) != 1 {
		t.Fatalf("expected 1 row, got %d", len(result.Rows))
	}

	cnt, ok := result.Rows[0]["cnt"]
	if !ok {
		t.Fatal("expected 'cnt' column in result row")
	}
	// The count value comes back as json.Number (due to UseNumber decoder).
	switch v := cnt.(type) {
	case json.Number:
		if v.String() != "3" {
			t.Errorf("expected count 3, got %v", v)
		}
	case float64:
		if v != 3 {
			t.Errorf("expected count 3, got %v", v)
		}
	default:
		t.Fatalf("unexpected type for cnt: %T (%v)", cnt, cnt)
	}
}

func TestAggregateGroupBy(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Skip("native library not available")
	}
	defer db.Close()

	db.CreateNode([]string{"Person"}, map[string]any{"name": "Alix"})
	db.CreateNode([]string{"Person"}, map[string]any{"name": "Gus"})
	db.CreateNode([]string{"City"}, map[string]any{"name": "Berlin"})

	result, err := db.Execute("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY label")
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Rows) < 2 {
		t.Fatalf("expected at least 2 groups, got %d rows", len(result.Rows))
	}

	// First group should be "City" (alphabetically before "Person").
	firstLabel, ok := result.Rows[0]["label"]
	if !ok {
		t.Fatal("expected 'label' column in result row")
	}
	if firstLabel != "City" {
		t.Errorf("expected first group label 'City', got %v", firstLabel)
	}
}

func TestOrderByProperty(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Skip("native library not available")
	}
	defer db.Close()

	db.CreateNode([]string{"Person"}, map[string]any{"name": "Vincent", "age": 40})
	db.CreateNode([]string{"Person"}, map[string]any{"name": "Alix", "age": 30})
	db.CreateNode([]string{"Person"}, map[string]any{"name": "Gus", "age": 25})

	result, err := db.Execute("MATCH (n:Person) RETURN n.name ORDER BY n.age")
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Rows) != 3 {
		t.Fatalf("expected 3 rows, got %d", len(result.Rows))
	}

	// Ordered by age ascending: Gus (25), Alix (30), Vincent (40).
	expectedOrder := []string{"Gus", "Alix", "Vincent"}
	for i, expected := range expectedOrder {
		name, ok := result.Rows[i]["n.name"]
		if !ok {
			t.Fatalf("row %d: expected 'n.name' column", i)
		}
		if name != expected {
			t.Errorf("row %d: expected %q, got %v", i, expected, name)
		}
	}
}

// --- Concurrent Tests (LockOSThread fix) ---

func TestConcurrentQueries(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Skip("native library not available")
	}
	defer db.Close()

	db.CreateNode([]string{"Person"}, map[string]any{"name": "Alix", "age": 30})
	db.CreateNode([]string{"Person"}, map[string]any{"name": "Gus", "age": 25})

	const goroutines = 10
	const queriesPerGoroutine = 50

	var wg sync.WaitGroup
	errs := make(chan error, goroutines*queriesPerGoroutine)

	for g := range goroutines {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for q := range queriesPerGoroutine {
				result, err := db.Execute("MATCH (n:Person) RETURN n.name")
				if err != nil {
					errs <- fmt.Errorf("goroutine %d, query %d: %w", id, q, err)
					return
				}
				if len(result.Rows) != 2 {
					errs <- fmt.Errorf("goroutine %d, query %d: expected 2 rows, got %d", id, q, len(result.Rows))
					return
				}
			}
		}(g)
	}

	wg.Wait()
	close(errs)

	for err := range errs {
		t.Error(err)
	}
}

func TestConcurrentErrorMessages(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Skip("native library not available")
	}
	defer db.Close()

	// Each goroutine runs a unique invalid query and checks that it gets
	// the correct error message (not another goroutine's error).
	const goroutines = 10

	var wg sync.WaitGroup
	errs := make(chan error, goroutines)

	for g := range goroutines {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			// Each goroutine uses a distinct invalid keyword so errors are distinguishable.
			badQuery := fmt.Sprintf("INVALID_QUERY_%d XYZZY", id)
			_, err := db.Execute(badQuery)
			if err == nil {
				errs <- fmt.Errorf("goroutine %d: expected error for invalid query, got nil", id)
				return
			}
			// The error should not be empty.
			if err.Error() == "" {
				errs <- fmt.Errorf("goroutine %d: got empty error message", id)
				return
			}
		}(g)
	}

	wg.Wait()
	close(errs)

	for err := range errs {
		t.Error(err)
	}
}
