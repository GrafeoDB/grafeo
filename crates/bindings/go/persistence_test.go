package grafeo

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// tempDbPath creates a temp directory and returns (dir, dbPath).
func tempDbPath(t *testing.T, name string) (string, string) {
	t.Helper()
	dir := t.TempDir()
	return dir, filepath.Join(dir, name)
}

func TestCreateAndReopen(t *testing.T) {
	_, dbPath := tempDbPath(t, "reopen.grafeo")

	// Create and populate
	db, err := Open(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Execute("INSERT (:Person {name: 'Alix', age: 30})"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Execute("INSERT (:Person {name: 'Gus', age: 25})"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Execute(
		"MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) INSERT (a)-[:KNOWS]->(b)",
	); err != nil {
		t.Fatal(err)
	}
	if db.NodeCount() != 2 {
		t.Errorf("expected 2 nodes, got %d", db.NodeCount())
	}
	if db.EdgeCount() != 1 {
		t.Errorf("expected 1 edge, got %d", db.EdgeCount())
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	// Reopen and verify
	db2, err := Open(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db2.Close()

	if db2.NodeCount() != 2 {
		t.Errorf("expected 2 nodes after reopen, got %d", db2.NodeCount())
	}
	if db2.EdgeCount() != 1 {
		t.Errorf("expected 1 edge after reopen, got %d", db2.EdgeCount())
	}

	result, err := db2.Execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Rows) != 2 {
		t.Fatalf("expected 2 rows, got %d", len(result.Rows))
	}
}

func TestSaveInMemoryToFile(t *testing.T) {
	_, dbPath := tempDbPath(t, "saved.grafeo")

	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Execute("INSERT (:City {name: 'Amsterdam'})"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Execute("INSERT (:City {name: 'Berlin'})"); err != nil {
		t.Fatal(err)
	}
	if err := db.Save(dbPath); err != nil {
		t.Fatal(err)
	}
	db.Close()

	db2, err := Open(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db2.Close()

	if db2.NodeCount() != 2 {
		t.Errorf("expected 2 nodes, got %d", db2.NodeCount())
	}

	result, err := db2.Execute("MATCH (c:City) RETURN c.name ORDER BY c.name")
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Rows) != 2 {
		t.Fatalf("expected 2 rows, got %d", len(result.Rows))
	}
}

func TestMultipleReopenCycles(t *testing.T) {
	_, dbPath := tempDbPath(t, "cycles.grafeo")

	// Cycle 1
	db, err := Open(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	db.Execute("INSERT (:Person {name: 'Alix'})")
	db.Close()

	// Cycle 2
	db, err = Open(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	if db.NodeCount() != 1 {
		t.Errorf("cycle 2: expected 1 node, got %d", db.NodeCount())
	}
	db.Execute("INSERT (:Person {name: 'Gus'})")
	db.Close()

	// Cycle 3
	db, err = Open(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	if db.NodeCount() != 2 {
		t.Errorf("cycle 3: expected 2 nodes, got %d", db.NodeCount())
	}
	db.Execute("INSERT (:Person {name: 'Vincent'})")
	db.Close()

	// Final check
	db, err = Open(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	if db.NodeCount() != 3 {
		t.Errorf("final: expected 3 nodes, got %d", db.NodeCount())
	}
	result, err := db.Execute("MATCH (p:Person) RETURN p.name")
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Rows) != 3 {
		t.Fatalf("expected 3 rows, got %d", len(result.Rows))
	}
}

func TestCheckpointAndContinuedWrites(t *testing.T) {
	_, dbPath := tempDbPath(t, "checkpoint.grafeo")

	db, err := Open(dbPath)
	if err != nil {
		t.Fatal(err)
	}

	db.Execute("INSERT (:Person {name: 'Alix'})")
	if err := db.WalCheckpoint(); err != nil {
		t.Fatal(err)
	}
	db.Execute("INSERT (:Person {name: 'Gus'})")
	db.Close()

	db2, err := Open(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db2.Close()

	if db2.NodeCount() != 2 {
		t.Errorf("expected 2 nodes after checkpoint+close, got %d", db2.NodeCount())
	}
}

func TestEdgePropertiesPersist(t *testing.T) {
	_, dbPath := tempDbPath(t, "edgeprops.grafeo")

	db, err := Open(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	db.Execute("INSERT (:Person {name: 'Alix'})")
	db.Execute("INSERT (:Person {name: 'Gus'})")
	db.Execute(
		"MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) " +
			"INSERT (a)-[:KNOWS {since: 2020}]->(b)")
	db.Close()

	db2, err := Open(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db2.Close()

	result, err := db2.Execute("MATCH ()-[e:KNOWS]->() RETURN e.since")
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Rows) != 1 {
		t.Fatalf("expected 1 row, got %d", len(result.Rows))
	}
	since, ok := result.Rows[0]["e.since"]
	if !ok {
		t.Fatal("expected 'e.since' column")
	}
	// JSON numbers come back as json.Number (due to UseNumber decoder).
	switch v := since.(type) {
	case json.Number:
		if v.String() != "2020" {
			t.Errorf("expected since=2020, got %v", v)
		}
	case float64:
		if v != 2020 {
			t.Errorf("expected since=2020, got %v", v)
		}
	default:
		t.Errorf("expected numeric since, got %T: %v", since, since)
	}
}

func TestGrafeoFileIsSingleFile(t *testing.T) {
	_, dbPath := tempDbPath(t, "single.grafeo")

	db, err := Open(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	db.Execute("INSERT (:Node {x: 1})")
	db.Close()

	info, err := os.Stat(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	if info.IsDir() {
		t.Error(".grafeo path should be a file, not a directory")
	}
	if !info.Mode().IsRegular() {
		t.Error(".grafeo path should be a regular file")
	}
}

func TestOpenSingleFileExplicit(t *testing.T) {
	_, dbPath := tempDbPath(t, "explicit.grafeo")

	db, err := OpenSingleFile(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	db.Execute("INSERT (:Node {x: 1})")
	db.Close()

	info, err := os.Stat(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	if info.IsDir() {
		t.Error("OpenSingleFile should produce a file, not a directory")
	}
}
