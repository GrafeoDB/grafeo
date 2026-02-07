package grafeo

import (
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

	node, err := db.CreateNode([]string{"Person"}, map[string]any{"name": "Alice", "age": 30})
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

	created, _ := db.CreateNode([]string{"Person"}, map[string]any{"name": "Alice"})
	fetched, err := db.GetNode(created.ID)
	if err != nil {
		t.Fatal(err)
	}
	if fetched.Properties["name"] != "Alice" {
		t.Errorf("expected name Alice, got %v", fetched.Properties["name"])
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

	a, _ := db.CreateNode([]string{"Person"}, map[string]any{"name": "Alice"})
	b, _ := db.CreateNode([]string{"Person"}, map[string]any{"name": "Bob"})

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

	db.CreateNode([]string{"Person"}, map[string]any{"name": "Alice", "age": 30})
	db.CreateNode([]string{"Person"}, map[string]any{"name": "Bob", "age": 25})

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

	_, err = tx.Execute("CREATE (:Tx {val: 1})")
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

	_, err = tx.Execute("CREATE (:Rolled {val: 2})")
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

	_, err = tx.Execute("CREATE (:Isolated {val: 1})")
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

	db.CreateNode([]string{"Person"}, map[string]any{"name": "Alice"})
	db.CreateNode([]string{"Person"}, map[string]any{"name": "Bob"})

	ids, err := db.FindNodesByProperty("name", "Alice")
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
	if len(results) < 1 {
		t.Fatal("expected at least 1 result")
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
