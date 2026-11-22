package grafeo

import (
	"fmt"
	"strings"
	"testing"
)

// The query resource limits through Go (#492): Options.PathSearchBudget, PathExpansionLimit,
// MaxQueryLength and MaxIdentifierLength. Zero fails at the open; each setting binds, with its
// error code and the setting named in the message. The snapshot
// retention limit, Options.SnapshotRetentionLimit: below 1 MiB fails at the open; past it an open
// transaction fails with GRAFEO-T012.

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

func TestZeroQueryLimitsAreRefused(t *testing.T) {
	for _, options := range []*Options{
		{PathSearchBudget: ptr(0)},
		{PathExpansionLimit: ptr(0)},
		{MaxQueryLength: ptr(0)},
		{MaxIdentifierLength: ptr(0)},
	} {
		if db, err := OpenWithOptions("", options); err == nil {
			db.Close()
			t.Errorf("%+v: a zero setting opened", options)
		}
	}
}

func TestEachQueryLimitBinds(t *testing.T) {
	db := openedWith(t, &Options{PathExpansionLimit: ptr(4)})
	assertRefused(t, db, limitsPaths, "GRAFEO-Q009", "path_expansion_limit")
	db.Close()

	db = openedWith(t, &Options{PathExpansionLimit: ptr(5)})
	if _, err := db.Execute(limitsPaths); err != nil {
		t.Errorf("five expansions at five: %v", err)
	}
	db.Close()

	db = openedWith(t, &Options{PathSearchBudget: ptr(1)})
	assertRefused(t, db, limitsPaths, "GRAFEO-S001", "path_search_budget")
	db.Close()

	db = openedWith(t, &Options{MaxQueryLength: ptr(200)})
	if _, err := db.Execute(limitsPaths); err != nil {
		t.Errorf("under the length: %v", err)
	}
	assertRefused(t, db, limitsPaths+strings.Repeat(" ", 200), "GRAFEO-Q002", "max_query_length")
	db.Close()

	db = openedWith(t, &Options{MaxIdentifierLength: ptr(8)})
	if _, err := db.Execute("MATCH (abcdefgh:Star) RETURN abcdefgh.n"); err != nil {
		t.Errorf("a name of the limit: %v", err)
	}
	assertRefused(t, db, "MATCH (abcdefghi:Star) RETURN 1", "GRAFEO-Q002", "max_identifier_length")
	db.Close()
}

func TestSnapshotRetentionLimitBelow1MiBIsRefused(t *testing.T) {
	db, err := OpenWithOptions("", &Options{SnapshotRetentionLimit: ptr(512 * 1024)})
	if err == nil {
		db.Close()
		t.Fatal("a 512 KiB snapshot retention limit opened")
	}
	if !strings.Contains(err.Error(), "at least 1 MiB") {
		t.Errorf("want the minimum named, got %v", err)
	}
}

// Past the limit (20 commits replace 64 KiB each, over 1 MiB) the open transaction that read
// Alix fails with GRAFEO-T012, and rolls back.
func TestSnapshotRetentionLimitBinds(t *testing.T) {
	db, err := OpenWithOptions("", &Options{SnapshotRetentionLimit: ptr(1 << 20)})
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Execute("INSERT (:Person {name: 'Alix', bio: 'Amsterdam'})"); err != nil {
		t.Fatal(err)
	}
	tx, err := db.BeginTransaction()
	if err != nil {
		t.Fatal(err)
	}
	const readBio = "MATCH (p:Person {name: 'Alix'}) RETURN p.bio AS bio"
	if _, err := tx.Execute(readBio); err != nil {
		t.Fatal(err)
	}
	text := strings.Repeat("Vincent ", 8*1024)
	for round := 0; round < 20; round++ {
		query := fmt.Sprintf("MATCH (p:Person {name: 'Alix'}) SET p.bio = '%d%s'", round, text)
		if _, err := db.Execute(query); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := tx.Execute(readBio); err == nil || !strings.Contains(err.Error(), "GRAFEO-T012") {
		t.Errorf("want GRAFEO-T012, got %v", err)
	}
	if err := tx.Rollback(); err != nil {
		t.Fatal(err)
	}
}

// The history retention limit reaches the open: zero fails it; a limit opens.
func TestHistoryRetentionLimitReachesTheOpen(t *testing.T) {
	if db, err := OpenWithOptions("", &Options{HistoryRetentionLimit: ptr(0)}); err == nil {
		db.Close()
		t.Error("a zero history retention limit opened")
	} else if !strings.Contains(err.Error(), "history_retention_limit") {
		t.Errorf("want the setting named, got %v", err)
	}
	db, err := OpenWithOptions("", &Options{HistoryRetentionLimit: ptr(1 << 20)})
	if err != nil {
		t.Fatal(err)
	}
	db.Close()
}
