package grafeo

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
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
	if mustCount(t, db.NodeCount) != 2 {
		t.Errorf("expected 2 nodes, got %d", mustCount(t, db.NodeCount))
	}
	if mustCount(t, db.EdgeCount) != 1 {
		t.Errorf("expected 1 edge, got %d", mustCount(t, db.EdgeCount))
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

	if mustCount(t, db2.NodeCount) != 2 {
		t.Errorf("expected 2 nodes after reopen, got %d", mustCount(t, db2.NodeCount))
	}
	if mustCount(t, db2.EdgeCount) != 1 {
		t.Errorf("expected 1 edge after reopen, got %d", mustCount(t, db2.EdgeCount))
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

	if mustCount(t, db2.NodeCount) != 2 {
		t.Errorf("expected 2 nodes, got %d", mustCount(t, db2.NodeCount))
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
	if mustCount(t, db.NodeCount) != 1 {
		t.Errorf("cycle 2: expected 1 node, got %d", mustCount(t, db.NodeCount))
	}
	db.Execute("INSERT (:Person {name: 'Gus'})")
	db.Close()

	// Cycle 3
	db, err = Open(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	if mustCount(t, db.NodeCount) != 2 {
		t.Errorf("cycle 3: expected 2 nodes, got %d", mustCount(t, db.NodeCount))
	}
	db.Execute("INSERT (:Person {name: 'Vincent'})")
	db.Close()

	// Final check
	db, err = Open(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	if mustCount(t, db.NodeCount) != 3 {
		t.Errorf("final: expected 3 nodes, got %d", mustCount(t, db.NodeCount))
	}
	result, err := db.Execute("MATCH (p:Person) RETURN p.name")
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Rows) != 3 {
		t.Fatalf("expected 3 rows, got %d", len(result.Rows))
	}
}

func TestDetachDeleteNode(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	for _, query := range []string{
		"INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})",
		"MATCH (a:Person {name: 'Alix'}) INSERT (a)-[:KNOWS]->(:Person {name: 'Vincent'})",
	} {
		if _, err := db.Execute(query); err != nil {
			t.Fatal(err)
		}
	}
	result, err := db.Execute("MATCH (a:Person {name: 'Alix'}) RETURN id(a) AS id")
	if err != nil {
		t.Fatal(err)
	}
	number, ok := result.Rows[0]["id"].(json.Number)
	if !ok {
		t.Fatalf("expected a number id, got %T", result.Rows[0]["id"])
	}
	id, err := number.Int64()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.DeleteNode(uint64(id)); err == nil {
		t.Error("expected DeleteNode to refuse a node with edges")
	}
	deleted, err := db.DetachDeleteNode(uint64(id))
	if err != nil || !deleted {
		t.Fatalf("expected the detach delete to succeed, got %v, %v", deleted, err)
	}
	if mustCount(t, db.NodeCount) != 2 || mustCount(t, db.EdgeCount) != 0 {
		t.Errorf("expected 2 nodes and no edges, got %d and %d", mustCount(t, db.NodeCount), mustCount(t, db.EdgeCount))
	}
}

func TestCheckpointTriggersReadBack(t *testing.T) {
	_, dbPath := tempDbPath(t, "triggers.grafeo")

	db, err := Open(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	if err := db.SetCheckpointTriggers(300*time.Millisecond, 64<<20); err != nil {
		t.Fatal(err)
	}
	status, err := db.CheckpointStatus()
	if err != nil {
		t.Fatal(err)
	}
	if status.IntervalMs == nil || *status.IntervalMs != 300 {
		t.Errorf("expected an interval of 300 ms, got %v", status.IntervalMs)
	}
	if status.WalBytes == nil || *status.WalBytes != 64<<20 {
		t.Errorf("expected a WAL size of 64 MiB, got %v", status.WalBytes)
	}
	if status.Failures != 0 || status.Stopped {
		t.Errorf("expected no failures, got %+v", status)
	}

	if err := db.SetCheckpointTriggers(0, 0); err != nil {
		t.Fatal(err)
	}
	status, err = db.CheckpointStatus()
	if err != nil {
		t.Fatal(err)
	}
	if status.IntervalMs != nil || status.WalBytes != nil {
		t.Errorf("expected the triggers off, got %+v", status)
	}
}

func TestCheckpointTriggersRefusedInMemory(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if err := db.SetCheckpointTriggers(0, 1024); err == nil {
		t.Error("expected an in-memory database to refuse checkpoint triggers")
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

	if mustCount(t, db2.NodeCount) != 2 {
		t.Errorf("expected 2 nodes after checkpoint+close, got %d", mustCount(t, db2.NodeCount))
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
