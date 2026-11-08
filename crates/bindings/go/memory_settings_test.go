package grafeo

import (
	"path/filepath"
	"strings"
	"testing"
)

// The memory settings through Go (#751): an Options struct with the memory limit, the query
// memory limit and the spill limit (nil is the default; a zero spill limit means nothing
// spills) and UnlimitedSpill, opened with OpenWithOptions(path or "", options or nil). nil
// options are the defaults, a zero memory limit fails at the open, the settings read back
// through MemoryUsage, and a file opened over its limit opens, reads and warns.

const mib uint64 = 1024 * 1024

func ptr(v uint64) *uint64 { return &v }

func usageOf(t *testing.T, db *Database) *MemoryUsage {
	t.Helper()
	usage, err := db.MemoryUsage()
	if err != nil {
		t.Fatal(err)
	}
	return usage
}

func TestMemoryOptionsReadBackThroughTheReport(t *testing.T) {
	db, err := OpenWithOptions("", &Options{
		MemoryLimit:      ptr(64 * mib),
		QueryMemoryLimit: ptr(8 * mib),
		SpillLimit:       ptr(0),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	usage := usageOf(t, db)
	if usage.LimitBytes != 64*mib {
		t.Errorf("limit: got %d, want %d", usage.LimitBytes, 64*mib)
	}
	if usage.QueryLimitBytes != 8*mib {
		t.Errorf("query limit: got %d, want %d", usage.QueryLimitBytes, 8*mib)
	}
	if usage.Spill.LimitBytes == nil || *usage.Spill.LimitBytes != 0 {
		t.Errorf("spill limit: got %v, want 0", usage.Spill.LimitBytes)
	}
}

// A database file has a spill directory, so unlimited spill reads back as no limit.
func TestUnlimitedSpillReadsBackAsNoLimit(t *testing.T) {
	path := filepath.Join(t.TempDir(), "unlimited.grafeo")
	db, err := OpenWithOptions(path, &Options{SpillLimit: ptr(16 * mib), UnlimitedSpill: true})
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if limit := usageOf(t, db).Spill.LimitBytes; limit != nil {
		t.Errorf("unlimited spill: got %d, want no limit", *limit)
	}
}

// An in-memory database has no spill directory: nothing spills, whatever the settings say.
func TestAnInMemoryOpenSpillsNothingWhateverTheSettings(t *testing.T) {
	db, err := OpenWithOptions("", &Options{UnlimitedSpill: true})
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if limit := usageOf(t, db).Spill.LimitBytes; limit == nil || *limit != 0 {
		t.Errorf("in memory: got %v, want 0", limit)
	}
}

func TestNilOptionsAreTheDefaults(t *testing.T) {
	defaults, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer defaults.Close()
	db, err := OpenWithOptions("", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	want, got := usageOf(t, defaults), usageOf(t, db)
	if got.LimitBytes != want.LimitBytes || got.QueryLimitBytes != want.QueryLimitBytes {
		t.Errorf("nil options: got %d/%d, want %d/%d",
			got.LimitBytes, got.QueryLimitBytes, want.LimitBytes, want.QueryLimitBytes)
	}
}

func TestAZeroMemoryLimitFailsAtTheOpen(t *testing.T) {
	db, err := OpenWithOptions("", &Options{MemoryLimit: ptr(0)})
	if err == nil {
		db.Close()
		t.Fatal("a zero memory limit opened a database")
	}
}

func hasOpenWarning(t *testing.T, db *Database, code string) bool {
	t.Helper()
	warnings, err := db.OpenWarnings()
	if err != nil {
		t.Fatal(err)
	}
	for _, warning := range warnings {
		if warning.Code == code {
			return true
		}
	}
	return false
}

// A file whose data passes the limit it is opened with opens, reads, and says so in its open
// warnings; opened with room, it says nothing of the kind.
func TestAnOpenOverTheLimitWarnsAndReads(t *testing.T) {
	path := filepath.Join(t.TempDir(), "over.grafeo")
	db, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	// Distinct 1 KiB strings: data that stays past the limit once reloaded.
	if _, err := db.Execute("UNWIND range(1, 20000) AS i INSERT (:Fill {i: i, text: CAST(i AS STRING) || '" + strings.Repeat("x", 1000) + "'})"); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	tight, err := OpenWithOptions(path, &Options{MemoryLimit: ptr(mib)})
	if err != nil {
		t.Fatal(err)
	}
	if !hasOpenWarning(t, tight, "MEMORY_LIMIT_PASSED") {
		t.Error("an open over the limit gave no MEMORY_LIMIT_PASSED warning")
	}
	if count := mustCount(t, tight.NodeCount); count != 20000 {
		t.Errorf("reads over the limit: got %d nodes, want 20000", count)
	}
	if err := tight.Close(); err != nil {
		t.Fatal(err)
	}

	roomy, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer roomy.Close()
	if hasOpenWarning(t, roomy, "MEMORY_LIMIT_PASSED") {
		t.Error("an open with room warned")
	}
}
