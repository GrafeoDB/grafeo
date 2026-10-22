package grafeo

import (
	"strings"
	"testing"
)

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
