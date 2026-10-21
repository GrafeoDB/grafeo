package grafeo

/*
#include "grafeo.h"
#include <stdlib.h>
*/
import "C"
import (
	"encoding/json"
	"math"
	"runtime"
	"time"
	"unsafe"
)

// TimePoint is a point in the database's history: an epoch (AtEpoch), a time (AtTime) or
// a duration before now (Ago). Times have the resolution of a millisecond.
type TimePoint struct {
	kind  C.int32_t
	value C.int64_t
}

// AtEpoch is the committed state of epoch; a later epoch reads now.
func AtEpoch(epoch uint64) TimePoint {
	// Every epoch above the largest the C API takes is later than now.
	return TimePoint{kind: C.GRAFEO_POINT_EPOCH, value: C.int64_t(min(epoch, math.MaxInt64))}
}

// AtTime is the last commit at or before t.
func AtTime(t time.Time) TimePoint {
	return TimePoint{kind: C.GRAFEO_POINT_TIME, value: C.int64_t(t.UnixMilli())}
}

// Ago is the last commit at or before d before now; a negative d fails when used.
func Ago(d time.Duration) TimePoint {
	return TimePoint{kind: C.GRAFEO_POINT_AGO, value: C.int64_t(d.Milliseconds())}
}

// HistoryRetention is how much history a database keeps for time-travel reads: none
// (HistoryOff, the default), all of it (HistoryForever) or what was committed within a
// duration before now (HistoryFor). A database file stores the setting; one given at
// open replaces it.
type HistoryRetention struct {
	kind       C.int32_t
	durationMs C.uint64_t
}

var (
	// HistoryOff keeps no history: only the current epoch can be read.
	HistoryOff = HistoryRetention{kind: C.GRAFEO_HISTORY_OFF}
	// HistoryForever keeps all history.
	HistoryForever = HistoryRetention{kind: C.GRAFEO_HISTORY_FOREVER}
)

// HistoryFor keeps the history committed within d before now; a negative d keeps none.
func HistoryFor(d time.Duration) HistoryRetention {
	return HistoryRetention{kind: C.GRAFEO_HISTORY_DURATION, durationMs: C.uint64_t(max(d.Milliseconds(), 0))}
}

// OpenInMemoryWithHistory creates a new in-memory database that keeps history for
// time-travel reads.
func OpenInMemoryWithHistory(history HistoryRetention) (*Database, error) {
	return openWithHistory(nil, history)
}

// OpenWithHistory opens or creates a persistent database at path that keeps history for
// time-travel reads. The file stores the setting.
func OpenWithHistory(path string, history HistoryRetention) (*Database, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	return openWithHistory(cPath, history)
}

func openWithHistory(cPath *C.char, history HistoryRetention) (*Database, error) {
	runtime.LockOSThread()
	h := C.grafeo_open_with_history(cPath, history.kind, history.durationMs)
	if h == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	db := &Database{handle: h}
	runtime.SetFinalizer(db, (*Database).free)
	return db, nil
}

// CurrentEpoch returns the epoch of the last commit.
func (db *Database) CurrentEpoch() uint64 {
	return uint64(C.grafeo_current_epoch(db.handle))
}

// EpochAsOf returns the epoch point names: the last commit at or before it. It fails for a
// point before the history the database keeps.
func (db *Database) EpochAsOf(point TimePoint) (uint64, error) {
	var epoch C.uint64_t
	err := lockAndCheckStatus(func() C.GrafeoStatus {
		return C.grafeo_epoch_as_of(db.handle, point.kind, point.value, &epoch)
	})
	return uint64(epoch), err
}

// ExecuteAsOf runs one GQL query with parameters (nil for none) as the database was at
// point; a query that writes fails.
func (db *Database) ExecuteAsOf(query string, point TimePoint, params map[string]any) (*QueryResult, error) {
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	cParams, err := optionalJSON(params)
	if err != nil {
		return nil, err
	}
	defer freeC(cParams)
	return viewResult(func() *C.GrafeoResult {
		return C.grafeo_execute_as_of(db.handle, cQuery, cParams, point.kind, point.value)
	})
}

// AsOfView is a read-only view of the database as it was at an earlier epoch, from
// Database.AsOf. Its queries see the committed state of Epoch(); a query that writes fails.
// It keeps the history of its epoch until Close.
type AsOfView struct {
	handle *C.GrafeoAsOf
	epoch  uint64
}

// AsOf returns a read-only view of the database as it was at point. It fails for a point
// before the history the database keeps.
func (db *Database) AsOf(point TimePoint) (*AsOfView, error) {
	runtime.LockOSThread()
	handle := C.grafeo_as_of(db.handle, point.kind, point.value)
	if handle == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	v := &AsOfView{handle: handle, epoch: uint64(C.grafeo_as_of_epoch(handle))}
	runtime.SetFinalizer(v, (*AsOfView).Close)
	return v, nil
}

// Epoch returns the epoch the view reads at.
func (v *AsOfView) Epoch() uint64 {
	return v.epoch
}

// Close releases the view and the history it keeps; calling it again does nothing.
func (v *AsOfView) Close() {
	if v.handle != nil {
		C.grafeo_free_as_of(v.handle)
		v.handle = nil
	}
	runtime.SetFinalizer(v, nil)
}

// Execute runs a GQL query as the database was at the view's epoch.
func (v *AsOfView) Execute(query string) (*QueryResult, error) {
	return v.ExecuteParams(query, nil)
}

// ExecuteParams runs a GQL query with parameters (nil for none) as the database was at the
// view's epoch.
func (v *AsOfView) ExecuteParams(query string, params map[string]any) (*QueryResult, error) {
	if v.handle == nil {
		return nil, errViewClosed
	}
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	cParams, err := optionalJSON(params)
	if err != nil {
		return nil, err
	}
	defer freeC(cParams)
	return viewResult(func() *C.GrafeoResult { return C.grafeo_as_of_execute(v.handle, cQuery, cParams) })
}

// GetNode returns the node as it was at the view's epoch; an error when it did not exist
// then, as Database.GetNode.
func (v *AsOfView) GetNode(id uint64) (*Node, error) {
	if v.handle == nil {
		return nil, errViewClosed
	}
	var cNode *C.GrafeoNode
	if err := lockAndCheckStatus(func() C.GrafeoStatus {
		return C.grafeo_as_of_get_node(v.handle, C.uint64_t(id), &cNode)
	}); err != nil {
		return nil, err
	}
	return nodeFromC(cNode), nil
}

// GetEdge returns the edge as it was at the view's epoch; an error when it did not exist
// then, as Database.GetEdge.
func (v *AsOfView) GetEdge(id uint64) (*Edge, error) {
	if v.handle == nil {
		return nil, errViewClosed
	}
	var cEdge *C.GrafeoEdge
	if err := lockAndCheckStatus(func() C.GrafeoStatus {
		return C.grafeo_as_of_get_edge(v.handle, C.uint64_t(id), &cEdge)
	}); err != nil {
		return nil, err
	}
	return edgeFromC(cEdge), nil
}

// GetNodeProperty returns a node's value of key as it was at the view's epoch: nil when the
// node or the property did not exist then.
func (v *AsOfView) GetNodeProperty(id uint64, key string) (any, error) {
	if v.handle == nil {
		return nil, errViewClosed
	}
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	ptr := C.grafeo_as_of_get_node_property(v.handle, C.uint64_t(id), cKey)
	if ptr == nil {
		return nil, nil
	}
	defer C.grafeo_free_string(ptr)
	var value any
	if err := json.Unmarshal([]byte(C.GoString(ptr)), &value); err != nil {
		return nil, err
	}
	return value, nil
}

var errViewClosed = &Error{Message: "the AsOfView is closed: open another with Database.AsOf"}

// optionalJSON is params as a C JSON string, or nil for nil params.
func optionalJSON(params map[string]any) (*C.char, error) {
	if params == nil {
		return nil, nil
	}
	return cJSON(params)
}

func viewResult(call func() *C.GrafeoResult) (*QueryResult, error) {
	runtime.LockOSThread()
	r := call()
	if r == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	defer C.grafeo_free_result(r)
	return parseResult(r)
}
