package grafeo

/*
#include "grafeo.h"
#include <stdlib.h>
*/
import "C"
import (
	"encoding/json"
	"runtime"
	"unsafe"
)

// Options are the memory, spill and query limit settings a database opens with (see
// OpenWithOptions). A nil field is the default: 75% of the memory the process may use for
// MemoryLimit, MemoryLimit for QueryMemoryLimit, a quarter of MemoryLimit (at most 256 MiB) for
// PathSearchBudget, 100 million for PathExpansionLimit, 16 MiB for MaxQueryLength, 16 KiB for
// MaxIdentifierLength, a quarter of MemoryLimit (at least 1 MiB) for SnapshotRetentionLimit,
// half the spill volume's free space for SpillLimit. A write past the memory limit fails with
// GRAFEO-S001, and so does a path search over its budget. One Options value may open several
// databases.
type Options struct {
	// MemoryLimit is how many bytes the database may hold; zero fails at the open.
	MemoryLimit *uint64
	// QueryMemoryLimit is how many bytes one statement's sorts, joins and aggregates may hold,
	// never more than MemoryLimit; zero fails at the open.
	QueryMemoryLimit *uint64
	// PathSearchBudget is how many bytes one path search may hold: the paths a variable-length or
	// shortest-path search keeps at once, or what an RDF property path search visits (the nodes an
	// ANY SHORTEST or shortestPath search visits are not counted yet). Never more than
	// MemoryLimit; zero fails at the open.
	PathSearchBudget *uint64
	// PathExpansionLimit is how many path expansions one statement's variable-length and
	// shortest-path searches may make together: each edge they read to extend a path, and each
	// node and edge they copy out of a path into a row. 100 million by default; a statement over
	// it fails with GRAFEO-Q009. Zero fails at the open.
	PathExpansionLimit *uint64
	// MaxQueryLength is how many bytes a statement's text may have; the values passed as
	// parameters do not count. 16 MiB by default; a longer statement fails with GRAFEO-Q002.
	// Zero fails at the open.
	MaxQueryLength *uint64
	// MaxIdentifierLength is how many bytes one name in a statement may have, as written between
	// its delimiters (string literals and IRIs are not names). 16 KiB by default; a statement
	// with a longer name fails with GRAFEO-Q002. Zero fails at the open.
	MaxIdentifierLength *uint64
	// SnapshotRetentionLimit is how many bytes of replaced data an open transaction's snapshot may
	// keep: what the commits since it began replaced that it can still read. Past it the
	// transaction's next call fails with the retryable GRAFEO-T012 and rolls it back. Below 1 MiB
	// fails at the open.
	SnapshotRetentionLimit *uint64
	// HistoryRetentionLimit is how many bytes of replaced data the history window may keep: once the
	// commits since its start replaced more, the start moves forward, whichever of time or bytes
	// comes first, and a read before it is refused. No limit by default; zero fails at the open. Not
	// stored with the database: pass it at every open.
	HistoryRetentionLimit *uint64
	// SpillLimit is how many bytes may spill to disk; zero means nothing spills.
	SpillLimit *uint64
	// UnlimitedSpill lifts the spill limit; it wins over SpillLimit. An in-memory database
	// spills nothing whatever the settings.
	UnlimitedSpill bool
}

// apply sets the options on the native options object. Must be called on the OS thread that
// reads the error.
func (o *Options) apply(native *C.GrafeoOptions) error {
	if o.MemoryLimit != nil {
		if err := statusToError(C.grafeo_options_set_memory_limit(native, C.uint64_t(*o.MemoryLimit))); err != nil {
			return err
		}
	}
	if o.QueryMemoryLimit != nil {
		if err := statusToError(C.grafeo_options_set_query_memory_limit(native, C.uint64_t(*o.QueryMemoryLimit))); err != nil {
			return err
		}
	}
	if o.PathSearchBudget != nil {
		if err := statusToError(C.grafeo_options_set_path_search_budget(native, C.uint64_t(*o.PathSearchBudget))); err != nil {
			return err
		}
	}
	if o.PathExpansionLimit != nil {
		if err := statusToError(C.grafeo_options_set_path_expansion_limit(native, C.uint64_t(*o.PathExpansionLimit))); err != nil {
			return err
		}
	}
	if o.MaxQueryLength != nil {
		if err := statusToError(C.grafeo_options_set_max_query_length(native, C.uint64_t(*o.MaxQueryLength))); err != nil {
			return err
		}
	}
	if o.MaxIdentifierLength != nil {
		if err := statusToError(C.grafeo_options_set_max_identifier_length(native, C.uint64_t(*o.MaxIdentifierLength))); err != nil {
			return err
		}
	}
	if o.SnapshotRetentionLimit != nil {
		if err := statusToError(C.grafeo_options_set_snapshot_retention_limit(native, C.uint64_t(*o.SnapshotRetentionLimit))); err != nil {
			return err
		}
	}
	if o.HistoryRetentionLimit != nil {
		if err := statusToError(C.grafeo_options_set_history_retention_limit(native, C.uint64_t(*o.HistoryRetentionLimit))); err != nil {
			return err
		}
	}
	if o.UnlimitedSpill {
		return statusToError(C.grafeo_options_set_unlimited_spill(native))
	}
	if o.SpillLimit != nil {
		return statusToError(C.grafeo_options_set_spill_limit(native, C.uint64_t(*o.SpillLimit)))
	}
	return nil
}

// OpenWithOptions opens a database with options: in memory when path is "", else the file at
// path (created when it does not exist). nil options are the defaults. A setting the database
// refuses (a zero memory limit) fails here.
func OpenWithOptions(path string, options *Options) (*Database, error) {
	var cPath *C.char
	if path != "" {
		cPath = C.CString(path)
		defer C.free(unsafe.Pointer(cPath))
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	native := C.grafeo_options_new()
	if native == nil {
		return nil, lastError()
	}
	defer C.grafeo_options_free(native)
	if options != nil {
		if err := options.apply(native); err != nil {
			return nil, err
		}
	}
	h := C.grafeo_open_with_options(cPath, native)
	if h == nil {
		return nil, lastError()
	}
	db := &Database{handle: h}
	runtime.SetFinalizer(db, (*Database).free)
	return db, nil
}

// SpillUsage is what may spill to disk and what has, in bytes.
type SpillUsage struct {
	TotalBytes uint64 `json:"total_bytes"`
	// LimitBytes is nil for no limit.
	LimitBytes        *uint64 `json:"limit_bytes"`
	SortBytes         uint64  `json:"sort_bytes"`
	AggregateBytes    uint64  `json:"aggregate_bytes"`
	VectorColumnBytes uint64  `json:"vector_column_bytes"`
}

// MemoryUsage is the bytes a database holds, as its accounts count them, and its limits. The
// parts' keys follow the storage engine and may change in any release.
type MemoryUsage struct {
	TotalBytes uint64 `json:"total_bytes"`
	// LimitBytes is the memory limit.
	LimitBytes uint64 `json:"limit_bytes"`
	// QueryLimitBytes is what one statement's operators may hold.
	QueryLimitBytes uint64                     `json:"query_limit_bytes"`
	Store           map[string]any             `json:"store"`
	Indexes         map[string]any             `json:"indexes"`
	Mvcc            map[string]any             `json:"mvcc"`
	Caches          map[string]any             `json:"caches"`
	StringPool      map[string]any             `json:"string_pool"`
	BufferManager   map[string]any             `json:"buffer_manager"`
	Rdf             map[string]json.RawMessage `json:"rdf,omitempty"`
	Cdc             map[string]json.RawMessage `json:"cdc,omitempty"`
	Spill           SpillUsage                 `json:"spill"`
}

// MemoryUsage returns the database's memory report.
func (db *Database) MemoryUsage() (*MemoryUsage, error) {
	defer runtime.KeepAlive(db)
	if err := db.checkOpen(); err != nil {
		var zero0 *MemoryUsage
		return zero0, err
	}
	runtime.LockOSThread()
	cJSON := C.grafeo_memory_usage_json(db.handle)
	if cJSON == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	defer C.grafeo_free_string(cJSON)

	var usage MemoryUsage
	if err := json.Unmarshal([]byte(C.GoString(cJSON)), &usage); err != nil {
		return nil, err
	}
	return &usage, nil
}

// OpenWarning is what an open had to say: MEMORY_LIMIT_PASSED when the data passes the memory
// limit (reads work, writes that need memory fail with GRAFEO-S001), MEMORY_DETECTION_FALLBACK,
// GRAPH_TYPE_NOT_BOUND.
type OpenWarning struct {
	Code    string  `json:"code"`
	Message string  `json:"message"`
	Context *string `json:"context"`
}

// OpenWarnings returns what the open had to say; empty for a clean open.
func (db *Database) OpenWarnings() ([]OpenWarning, error) {
	defer runtime.KeepAlive(db)
	if err := db.checkOpen(); err != nil {
		var zero0 []OpenWarning
		return zero0, err
	}
	runtime.LockOSThread()
	cJSON := C.grafeo_open_warnings(db.handle)
	if cJSON == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	defer C.grafeo_free_string(cJSON)

	var warnings []OpenWarning
	if err := json.Unmarshal([]byte(C.GoString(cJSON)), &warnings); err != nil {
		return nil, err
	}
	return warnings, nil
}
