package grafeo

import (
	"strings"
	"testing"
)

// The path search budget through Go (#492): Options.PathSearchBudget. Zero fails at the open;
// the budget binds, with its error code and the setting named in the message.

const limitsStar = "INSERT (c:Star {n: 0}), (c)-[:E]->(:Leaf {n: 1}), (c)-[:E]->(:Leaf {n: 2}), " +
	"(c)-[:E]->(:Leaf {n: 3}), (c)-[:E]->(:Leaf {n: 4}), (c)-[:E]->(:Leaf {n: 5})"

const limitsPaths = "MATCH (s:Star)-[:E*1..2]->(t) RETURN t.n AS n"

func openedWith(t *testing.T, options *Options) *Database {
	t.Helper()
	db, err := OpenWithOptions("", options)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Execute(limitsStar); err != nil {
		t.Fatal(err)
	}
	return db
}

func assertRefused(t *testing.T, db *Database, query, code, setting string) {
	t.Helper()
	_, err := db.Execute(query)
	if err == nil {
		t.Fatalf("%s: expected a refusal", query)
	}
	if !strings.Contains(err.Error(), code) || !strings.Contains(err.Error(), setting) {
		t.Errorf("%s: want %s naming %s, got %v", query, code, setting, err)
	}
}

func TestZeroPathSearchBudgetIsRefused(t *testing.T) {
	if db, err := OpenWithOptions("", &Options{PathSearchBudget: ptr(0)}); err == nil {
		db.Close()
		t.Error("a zero path search budget opened")
	}
}

func TestPathSearchBudgetBinds(t *testing.T) {
	db := openedWith(t, &Options{PathSearchBudget: ptr(1)})
	assertRefused(t, db, limitsPaths, "GRAFEO-S001", "path_search_budget")
	db.Close()
}
