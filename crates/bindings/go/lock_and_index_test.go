package grafeo

import (
	"os"
	"os/exec"
	"strings"
	"testing"
)

// The path TestOpenInAChildProcess opens; it does nothing when unset.
const childOpenPathVar = "GRAFEO_GO_LOCK_CHILD_PATH"

// A database file has one writer (#405): a second open while the first is
// open fails with a "locked" error, and after the close it opens again
// with every write of the first.
func TestSecondOpenIsRefusedWhileTheFirstIsOpen(t *testing.T) {
	for _, name := range []string{"people", "people.grafeo"} {
		_, path := tempDbPath(t, name)
		first, err := Open(path)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := first.Execute("INSERT (:Person {name: 'A'})"); err != nil {
			t.Fatal(err)
		}
		second, err := Open(path)
		if err == nil {
			second.Close()
			t.Fatalf("%s: a second open of an open file succeeded", name)
		}
		if !strings.Contains(err.Error(), "locked") {
			t.Fatalf("%s: expected a locked error, got %v", name, err)
		}
		if _, err := first.Execute("INSERT (:Person {name: 'B'})"); err != nil {
			t.Fatal(err)
		}
		if err := first.Close(); err != nil {
			t.Fatal(err)
		}

		reopened, err := Open(path)
		if err != nil {
			t.Fatal(err)
		}
		result, err := reopened.Execute("MATCH (p:Person) RETURN p.name AS n ORDER BY n")
		if err != nil {
			t.Fatal(err)
		}
		got := []any{}
		for _, row := range result.Rows {
			got = append(got, row["n"])
		}
		if len(got) != 2 || got[0] != "A" || got[1] != "B" {
			t.Fatalf("%s: expected [A B], got %v", name, got)
		}
		reopened.Close()
	}
}

// Another process cannot open the file while this one has it open (#405,
// the reporter's case): its open fails with a "locked" error and writes
// nothing, and every write of this process is there after the close.
func TestAnotherProcessCannotOpenAnOpenFile(t *testing.T) {
	for _, name := range []string{"people", "people.grafeo"} {
		_, path := tempDbPath(t, name)
		first, err := Open(path)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := first.Execute("INSERT (:Person {name: 'A'})"); err != nil {
			t.Fatal(err)
		}
		child := exec.Command(os.Args[0], "-test.run=^TestOpenInAChildProcess$", "-test.v")
		child.Env = append(os.Environ(), childOpenPathVar+"="+path)
		output, err := child.CombinedOutput()
		if err != nil {
			t.Fatalf("%s: the child failed: %v\n%s", name, err, output)
		}
		if !strings.Contains(string(output), "the open is locked") {
			t.Fatalf("%s: the child did not report a locked open:\n%s", name, output)
		}
		if _, err := first.Execute("INSERT (:Person {name: 'B'})"); err != nil {
			t.Fatal(err)
		}
		if err := first.Close(); err != nil {
			t.Fatal(err)
		}

		reopened, err := Open(path)
		if err != nil {
			t.Fatal(err)
		}
		result, err := reopened.Execute("MATCH (p:Person) RETURN p.name AS n ORDER BY n")
		if err != nil {
			t.Fatal(err)
		}
		got := []any{}
		for _, row := range result.Rows {
			got = append(got, row["n"])
		}
		if len(got) != 2 || got[0] != "A" || got[1] != "B" {
			t.Fatalf("%s: expected [A B], got %v", name, got)
		}
		reopened.Close()
	}
}

// The child of TestAnotherProcessCannotOpenAnOpenFile: opens the path it is
// given and expects the locked error. Does nothing when run directly.
func TestOpenInAChildProcess(t *testing.T) {
	path := os.Getenv(childOpenPathVar)
	if path == "" {
		return
	}
	db, err := Open(path)
	if err == nil {
		db.Close()
		t.Fatal("opened a database that another process holds")
	}
	if !strings.Contains(err.Error(), "locked") {
		t.Fatalf("expected a locked error, got %v", err)
	}
	t.Log("the open is locked")
}

// A property index survives a close and a reopen (#638), and an equality
// on its property looks it up.
func TestPropertyIndexSurvivesCloseAndReopen(t *testing.T) {
	_, path := tempDbPath(t, "indexed.grafeo")
	db, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, statement := range []string{
		"INSERT (:Person {name: 'Alix'}), (:Person {name: 'Gus'})",
		"CREATE INDEX person_name FOR (p:Person) ON (p.name)",
		"INSERT (:Person {name: 'Vincent'})",
	} {
		if _, err := db.Execute(statement); err != nil {
			t.Fatal(err)
		}
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	db, err = Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if !db.HasPropertyIndex("name") {
		t.Fatal("the index on name is gone after the reopen")
	}
	query := "MATCH (p:Person {name: 'Vincent'}) RETURN p.name AS n"
	plan, err := db.Execute("EXPLAIN " + query)
	if err != nil {
		t.Fatal(err)
	}
	text, _ := plan.Rows[0][plan.Columns[0]].(string)
	if !strings.Contains(text, "[index: name]") {
		t.Fatalf("the lookup does not use the index: %s", text)
	}
	result, err := db.Execute(query)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Rows) != 1 || result.Rows[0]["n"] != "Vincent" {
		t.Fatalf("expected [Vincent], got %v", result.Rows)
	}
}
